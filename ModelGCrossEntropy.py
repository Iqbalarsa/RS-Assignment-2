import torch


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        # Conv1d expects (N, C, L)
        outputs = self.dropout2(
            self.conv2(
                self.relu(
                    self.dropout1(
                        self.conv1(inputs.transpose(-1, -2))
                    )
                )
            )
        )
        outputs = outputs.transpose(-1, -2)
        return outputs


class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.norm_first = args.norm_first

        # Item and positional embeddings
        self.item_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen + 1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            self.attention_layernorms.append(
                torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            )

            self.attention_layers.append(
                torch.nn.MultiheadAttention(
                    embed_dim=args.hidden_units,
                    num_heads=args.num_heads,
                    dropout=args.dropout_rate
                )
            )

            self.forward_layernorms.append(
                torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            )

            self.forward_layers.append(
                PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            )

    def log2feats(self, log_seqs):
        # Convert input sequence to tensor
        log_seqs = torch.as_tensor(log_seqs, dtype=torch.long, device=self.dev)

        # Item embeddings
        seqs = self.item_emb(log_seqs)
        seqs *= self.item_emb.embedding_dim ** 0.5

        # Position ids: keep padding positions at 0
        poss = torch.arange(1, log_seqs.shape[1] + 1, device=self.dev).unsqueeze(0)
        poss = poss.repeat(log_seqs.shape[0], 1)
        poss *= (log_seqs != 0)

        # Add positional embeddings
        seqs += self.pos_emb(poss)
        seqs = self.emb_dropout(seqs)

        # Causal mask: only attend to current/past positions
        tl = seqs.shape[1]
        attention_mask = ~torch.tril(
            torch.ones((tl, tl), dtype=torch.bool, device=self.dev)
        )

        # Padding mask: ignore padded positions
        padding_mask = (log_seqs == 0)

        for i in range(len(self.attention_layers)):
            # MultiheadAttention expects (T, B, C)
            seqs = torch.transpose(seqs, 0, 1)

            if self.norm_first:
                x = self.attention_layernorms[i](seqs)
                mha_outputs, _ = self.attention_layers[i](
                    x, x, x,
                    attn_mask=attention_mask,
                    key_padding_mask=padding_mask
                )
                seqs = seqs + mha_outputs
                seqs = torch.transpose(seqs, 0, 1)

                seqs = seqs + self.forward_layers[i](
                    self.forward_layernorms[i](seqs)
                )
            else:
                mha_outputs, _ = self.attention_layers[i](
                    seqs, seqs, seqs,
                    attn_mask=attention_mask,
                    key_padding_mask=padding_mask
                )
                seqs = self.attention_layernorms[i](seqs + mha_outputs)
                seqs = torch.transpose(seqs, 0, 1)

                seqs = self.forward_layernorms[i](
                    seqs + self.forward_layers[i](seqs)
                )

            # Clean any NaNs/Infs before masking padding
            seqs = torch.nan_to_num(seqs, nan=0.0, posinf=0.0, neginf=0.0)

            # Force padding positions to stay zero
            seqs = seqs.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        log_feats = self.last_layernorm(seqs)

        # Final cleanup for numerical stability
        log_feats = torch.nan_to_num(log_feats, nan=0.0, posinf=0.0, neginf=0.0)
        log_feats = log_feats.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
        # Sequence representations
        log_feats = self.log2feats(log_seqs)   # [B, T, H]

        # Positive and negative item embeddings
        pos_seqs = torch.as_tensor(pos_seqs, dtype=torch.long, device=self.dev)
        neg_seqs = torch.as_tensor(neg_seqs, dtype=torch.long, device=self.dev)

        pos_embs = self.item_emb(pos_seqs)     # [B, T, H]
        neg_embs = self.item_emb(neg_seqs)     # [B, T, H]

        # Scores for positive and negative candidates
        pos_logits = (log_feats * pos_embs).sum(dim=-1).unsqueeze(-1)  # [B, T, 1]
        neg_logits = (log_feats * neg_embs).sum(dim=-1).unsqueeze(-1)  # [B, T, 1]

        # Extra cleanup before CE loss
        pos_logits = torch.nan_to_num(pos_logits, nan=0.0, posinf=0.0, neginf=0.0)
        neg_logits = torch.nan_to_num(neg_logits, nan=0.0, posinf=0.0, neginf=0.0)

        # For CrossEntropyLoss: class 0 = positive, class 1 = negative
        logits = torch.cat([pos_logits, neg_logits], dim=-1)           # [B, T, 2]
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)

        return logits

    def predict(self, user_ids, log_seqs, item_indices):
        # Sequence representations
        log_feats = self.log2feats(log_seqs)

        # Use last position for next-item prediction
        final_feat = log_feats[:, -1, :]   # [B, H]

        item_indices = torch.as_tensor(item_indices, dtype=torch.long, device=self.dev)
        item_embs = self.item_emb(item_indices)  # [B, I, H] or [I, H]

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)

        return logits
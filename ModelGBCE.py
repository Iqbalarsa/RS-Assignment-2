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

        # Residual connection inside FFN block
        outputs += inputs
        return outputs


class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device

        # Embeddings
        self.item_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        # Transformer blocks
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

        # Positional embeddings
        positions = torch.arange(log_seqs.shape[1], device=self.dev).unsqueeze(0)
        positions = positions.expand(log_seqs.shape[0], -1)
        seqs += self.pos_emb(positions)

        seqs = self.emb_dropout(seqs)

        # Mask padded positions in the sequence itself
        timeline_mask = (log_seqs == 0)
        seqs = seqs.masked_fill(timeline_mask.unsqueeze(-1), 0.0)

        # Causal mask: no looking ahead
        tl = seqs.shape[1]
        attention_mask = ~torch.tril(
            torch.ones((tl, tl), dtype=torch.bool, device=self.dev)
        )

        for i in range(len(self.attention_layers)):
            # MultiheadAttention expects (T, B, C)
            seqs = seqs.transpose(0, 1)

            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](
                Q, seqs, seqs,
                attn_mask=attention_mask
            )

            # Residual after attention
            seqs = Q + mha_outputs

            # Back to (B, T, C)
            seqs = seqs.transpose(0, 1)

            # FFN block
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)

            # Keep padded positions zero
            seqs = seqs.masked_fill(timeline_mask.unsqueeze(-1), 0.0)

        log_feats = self.last_layernorm(seqs)
        log_feats = log_feats.masked_fill(timeline_mask.unsqueeze(-1), 0.0)

        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
        # Sequence features
        log_feats = self.log2feats(log_seqs)   # [B, T, H]

        # Positive and negative item embeddings
        pos_seqs = torch.as_tensor(pos_seqs, dtype=torch.long, device=self.dev)
        neg_seqs = torch.as_tensor(neg_seqs, dtype=torch.long, device=self.dev)

        pos_embs = self.item_emb(pos_seqs)     # [B, T, H]
        neg_embs = self.item_emb(neg_seqs)     # [B, T, H]

        # BCE training scores
        pos_logits = (log_feats * pos_embs).sum(dim=-1)   # [B, T]
        neg_logits = (log_feats * neg_embs).sum(dim=-1)   # [B, T]

        return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, item_indices):
        # Sequence features
        log_feats = self.log2feats(log_seqs)

        # Use last position for next-item prediction
        final_feat = log_feats[:, -1, :]   # [B, H]

        item_indices = torch.as_tensor(item_indices, dtype=torch.long, device=self.dev)
        item_embs = self.item_emb(item_indices)  # [B, I, H] or [I, H]

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return logits
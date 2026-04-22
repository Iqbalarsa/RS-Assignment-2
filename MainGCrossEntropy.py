import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from ModelGCrossEntropy  import SASRec
from UtilsGCrossEntropy  import WarpSampler, evaluate, evaluate_valid, preprocess_ml1m, split_data
import numpy as np


class Args:
    dataset = 'ratings.dat'  # Path
    batch_size = 128
    lr = 0.001
    maxlen = 100             # Maximal Length of Sequence
    hidden_units = 50
    num_blocks = 2           # Number of transformer blocks
    num_heads = 1
    dropout_rate = 0.2
    l2_emb = 0.0             # Regularisation
    num_epochs = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    norm_first = False
    patience = 5
    n_workers = 3
    seed = 42
    save_path = 'best_sasrec.pth'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def build_args_from_config(cfg):
    """
    Optional helper for experiments.py:
    converts a config dict into an Args-like object
    """
    args = Args()
    for key, value in cfg.items():
        setattr(args, key, value)
    return args


def run_training(args, run_name="run", verbose=True):
    set_seed(args.seed)

    # Loading and Splitting Data
    user_seqs = preprocess_ml1m(args.dataset)
    user_train, user_valid, user_test, usernum, itemnum = split_data(user_seqs)
    dataset = [user_train, user_valid, user_test, usernum, itemnum]

    if verbose:
        print(f"[{run_name}] Data Loaded: User: {usernum}, Item: {itemnum}")
        print(f"[{run_name}] Using device: {args.device}")

    # Setup Sampler (for automatic negative sampling)
    sampler = WarpSampler(
        user_train,
        usernum,
        itemnum,
        batch_size=args.batch_size,
        maxlen=args.maxlen,
        n_workers=args.n_workers
    )

    # Call the Model
    model = SASRec(usernum, itemnum, args).to(args.device)

    # Initialize the weight
    for name, param in model.named_parameters():
        try:
            if param.dim() > 1:
                torch.nn.init.xavier_normal_(param.data)
        except:
            pass

    # Loss and Optimizer
    # Cross Entropy: class 0 = positive item, class 1 = negative item
    criterion = nn.CrossEntropyLoss()
    adam_optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    # Training Loop
    num_batch = max(1, len(user_train) // args.batch_size)

    best_ndcg = -1.0
    best_epoch = 0
    early_stop_counter = 0

    try:
        for epoch in range(1, args.num_epochs + 1):
            model.train()
            epoch_loss = 0.0

            for step in range(num_batch):
                # Taking batch from sampler
                u, seq, pos, neg = sampler.next_batch()

                # Convert to tensor
                seq = torch.as_tensor(np.array(seq), dtype=torch.long, device=args.device)
                pos = torch.as_tensor(np.array(pos), dtype=torch.long, device=args.device)
                neg = torch.as_tensor(np.array(neg), dtype=torch.long, device=args.device)

                # logits shape = [B, T, 2]
                logits = model(u, seq, pos, neg)

                # Ignore padding positions
                indices = (pos != 0)
                logits = logits[indices]   # [N, 2]

                # Positive item is always class 0
                labels = torch.zeros(logits.size(0), dtype=torch.long, device=args.device)

                loss = criterion(logits, labels)

                # Regularization
                if args.l2_emb > 0:
                    for param in model.item_emb.parameters():
                        loss += args.l2_emb * torch.norm(param)

                adam_optimizer.zero_grad()
                loss.backward()
                adam_optimizer.step()

                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / max(1, num_batch)

            # Evaluate on validation set only
            model.eval()
            with torch.no_grad():
                t_valid = evaluate_valid(model, dataset, args)

            current_ndcg = t_valid[0]  # NDCG@10

            if verbose:
                print(
                    f"[{run_name}] Epoch {epoch} | "
                    f"Loss: {avg_epoch_loss:.4f} | "
                    f"Valid NDCG@10: {current_ndcg:.4f}"
                )

            # Early Stopping
            if current_ndcg > best_ndcg:
                best_ndcg = current_ndcg
                best_epoch = epoch
                early_stop_counter = 0

                # Save the best model
                torch.save(model.state_dict(), args.save_path)
                if verbose:
                    print(f"[{run_name}] There is an improvement, save checkpoint...")
            else:
                early_stop_counter += 1
                if verbose:
                    print(f"[{run_name}] There is no improvement ({early_stop_counter}/{args.patience})")

            if verbose:
                print(
                    f"[{run_name}] Valid: "
                    f"NDCG@10: {t_valid[0]:.4f}, "
                    f"Recall@10: {t_valid[1]:.4f}, "
                    f"NDCG@20: {t_valid[2]:.4f}, "
                    f"Recall@20: {t_valid[3]:.4f}"
                )

            if early_stop_counter >= args.patience:
                if verbose:
                    print(f"[{run_name}] Early stopping was triggered by epoch {epoch}. Training has stopped.")
                break

    finally:
        sampler.close()

    if verbose:
        print(f"[{run_name}] Best epoch: {best_epoch} | Best Valid NDCG@10: {best_ndcg:.4f}")

    # Load the best model for final test evaluation
    if os.path.exists(args.save_path):
        model.load_state_dict(torch.load(args.save_path, map_location=args.device))
        if verbose:
            print(f"[{run_name}] Best checkpoint loaded...")
    else:
        if verbose:
            print(f"[{run_name}] Checkpoint was not found, evaluating current model...")

    # Final test evaluation
    model.eval()
    with torch.no_grad():
        t_test = evaluate(model, dataset, args)

    if verbose:
        print(
            f"[{run_name}] Test : "
            f"NDCG@10: {t_test[0]:.4f}, "
            f"Recall@10: {t_test[1]:.4f}, "
            f"NDCG@20: {t_test[2]:.4f}, "
            f"Recall@20: {t_test[3]:.4f}"
        )
        print(f"[{run_name}] Training has completed!")

    results = {
        "best_epoch": best_epoch,
        "best_valid_ndcg10": float(best_ndcg),
        "valid_ndcg10": float(t_valid[0]),
        "valid_recall10": float(t_valid[1]),
        "valid_ndcg20": float(t_valid[2]),
        "valid_recall20": float(t_valid[3]),
        "test_ndcg10": float(t_test[0]),
        "test_recall10": float(t_test[1]),
        "test_ndcg20": float(t_test[2]),
        "test_recall20": float(t_test[3]),
        "usernum": int(usernum),
        "itemnum": int(itemnum),
    }

    return results


if __name__ == '__main__':
    args = Args()
    results = run_training(args, run_name="main", verbose=True)
    print("\nFinal Results Dictionary:")
    print(results)
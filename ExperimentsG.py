import os
import csv
import json
import time
import copy
import random
import argparse
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from modelG import SASRec
from utilsG import WarpSampler, evaluate, evaluate_valid, preprocess_ml1m, split_data


RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# Base configuration used for all ablations
# You can change these defaults if needed.
def base_config():
    return {
        "dataset": "ratings.dat",
        "batch_size": 128,
        "lr": 0.001,
        "maxlen": 100,
        "hidden_units": 50,
        "num_blocks": 2,
        "num_heads": 1,
        "dropout_rate": 0.2,
        "l2_emb": 0.0,
        "num_epochs": 100,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "norm_first": False,
        "patience": 5,
        "n_workers": 3,
        "seed": 42,
        "save_path": "best_sasrec.pth",
    }


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# Train one SASRec setting and return the best validation score + final test metrics
def run_one_setting(cfg, seed=None, run_name="run"):
    cfg = copy.deepcopy(cfg)
    if seed is not None:
        cfg["seed"] = seed

    set_seed(cfg["seed"])
    args = SimpleNamespace(**cfg)

    user_seqs = preprocess_ml1m(args.dataset)
    user_train, user_valid, user_test, usernum, itemnum = split_data(user_seqs)
    dataset = [user_train, user_valid, user_test, usernum, itemnum]

    sampler = WarpSampler(
        user_train,
        usernum,
        itemnum,
        batch_size=args.batch_size,
        maxlen=args.maxlen,
        n_workers=args.n_workers,
    )

    model = SASRec(usernum, itemnum, args).to(args.device)

    # Xavier init for weight matrices
    for _, param in model.named_parameters():
        if param.dim() > 1:
            torch.nn.init.xavier_normal_(param.data)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    num_batch = max(1, len(user_train) // args.batch_size)
    best_valid = {
        "epoch": 0,
        "ndcg10": -1.0,
        "recall10": 0.0,
        "ndcg20": 0.0,
        "recall20": 0.0,
    }
    early_stop_counter = 0

    checkpoint_path = os.path.join(RESULTS_DIR, f"{run_name}_seed{args.seed}.pth")
    history = []

    start_time = time.time()

    try:
        for epoch in range(1, args.num_epochs + 1):
            model.train()
            epoch_loss = 0.0

            for _ in range(num_batch):
                u, seq, pos, neg = sampler.next_batch()

                seq = torch.as_tensor(np.array(seq), dtype=torch.long, device=args.device)
                pos = torch.as_tensor(np.array(pos), dtype=torch.long, device=args.device)
                neg = torch.as_tensor(np.array(neg), dtype=torch.long, device=args.device)

                pos_logits, neg_logits = model(u, seq, pos, neg)

                pos_labels = torch.ones(pos_logits.shape, device=args.device)
                neg_labels = torch.zeros(neg_logits.shape, device=args.device)

                indices = (pos != 0)
                loss = criterion(pos_logits[indices], pos_labels[indices])
                loss += criterion(neg_logits[indices], neg_labels[indices])

                if args.l2_emb > 0:
                    for param in model.item_emb.parameters():
                        loss += args.l2_emb * torch.norm(param)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / num_batch

            model.eval()
            with torch.no_grad():
                valid_metrics = evaluate_valid(model, dataset, args)

            epoch_record = {
                "epoch": epoch,
                "loss": float(avg_epoch_loss),
                "valid_ndcg10": float(valid_metrics[0]),
                "valid_recall10": float(valid_metrics[1]),
                "valid_ndcg20": float(valid_metrics[2]),
                "valid_recall20": float(valid_metrics[3]),
            }
            history.append(epoch_record)

            print(
                f"[{run_name}] epoch {epoch:03d} | "
                f"loss={avg_epoch_loss:.4f} | "
                f"valid NDCG@10={valid_metrics[0]:.4f} | "
                f"valid Recall@10={valid_metrics[1]:.4f} | "
                f"valid NDCG@20={valid_metrics[2]:.4f} | "
                f"valid Recall@20={valid_metrics[3]:.4f}"
            )

            if valid_metrics[0] > best_valid["ndcg10"]:
                best_valid = {
                    "epoch": epoch,
                    "ndcg10": float(valid_metrics[0]),
                    "recall10": float(valid_metrics[1]),
                    "ndcg20": float(valid_metrics[2]),
                    "recall20": float(valid_metrics[3]),
                }
                early_stop_counter = 0
                torch.save(model.state_dict(), checkpoint_path)
            else:
                early_stop_counter += 1

            if early_stop_counter >= args.patience:
                print(f"[{run_name}] early stopping at epoch {epoch}")
                break

    finally:
        sampler.close()

    runtime_sec = time.time() - start_time

    # Final test with the best checkpoint only
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=args.device))

    model.eval()
    with torch.no_grad():
        test_metrics = evaluate(model, dataset, args)

    result = {
        "run_name": run_name,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "maxlen": args.maxlen,
        "hidden_units": args.hidden_units,
        "num_blocks": args.num_blocks,
        "num_heads": args.num_heads,
        "dropout_rate": args.dropout_rate,
        "l2_emb": args.l2_emb,
        "num_epochs": args.num_epochs,
        "patience": args.patience,
        "best_epoch": best_valid["epoch"],
        "best_valid_ndcg10": best_valid["ndcg10"],
        "best_valid_recall10": best_valid["recall10"],
        "best_valid_ndcg20": best_valid["ndcg20"],
        "best_valid_recall20": best_valid["recall20"],
        "test_ndcg10": float(test_metrics[0]),
        "test_recall10": float(test_metrics[1]),
        "test_ndcg20": float(test_metrics[2]),
        "test_recall20": float(test_metrics[3]),
        "runtime_sec": float(runtime_sec),
    }

    history_path = os.path.join(RESULTS_DIR, f"{run_name}_seed{args.seed}_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    return result


def aggregate_results(rows, run_name, extra_fields=None):
    seeds = [r["seed"] for r in rows]
    summary = {
        "run_name": run_name,
        "n_runs": len(rows),
        "seeds": seeds,
        "mean_best_valid_ndcg10": float(np.mean([r["best_valid_ndcg10"] for r in rows])),
        "std_best_valid_ndcg10": float(np.std([r["best_valid_ndcg10"] for r in rows])),
        "mean_test_ndcg10": float(np.mean([r["test_ndcg10"] for r in rows])),
        "std_test_ndcg10": float(np.std([r["test_ndcg10"] for r in rows])),
        "mean_test_recall10": float(np.mean([r["test_recall10"] for r in rows])),
        "std_test_recall10": float(np.std([r["test_recall10"] for r in rows])),
        "mean_test_ndcg20": float(np.mean([r["test_ndcg20"] for r in rows])),
        "std_test_ndcg20": float(np.std([r["test_ndcg20"] for r in rows])),
        "mean_test_recall20": float(np.mean([r["test_recall20"] for r in rows])),
        "std_test_recall20": float(np.std([r["test_recall20"] for r in rows])),
        "mean_runtime_sec": float(np.mean([r["runtime_sec"] for r in rows])),
    }
    if extra_fields:
        summary.update(extra_fields)
    return summary


def save_rows_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# Run one-factor-at-a-time ablation for a single parameter
# This directly matches the assignment requirement to compare blocks, hidden size, heads, and maxlen.
def run_ablation(param_name, values, base_cfg, n_repetitions=1):
    all_rows = []
    summary_rows = []

    for value in values:
        per_setting_rows = []

        for rep in range(n_repetitions):
            cfg = copy.deepcopy(base_cfg)
            cfg[param_name] = value
            seed = cfg["seed"] + rep
            run_name = f"{param_name}_{value}"

            result = run_one_setting(cfg, seed=seed, run_name=run_name)
            per_setting_rows.append(result)
            all_rows.append(result)

        summary = aggregate_results(
            per_setting_rows,
            run_name=f"{param_name}_{value}",
            extra_fields={param_name: value},
        )
        summary_rows.append(summary)
        print(f"Ablation {param_name}={value}: {summary}")

    save_rows_csv(os.path.join(RESULTS_DIR, f"ablation_{param_name}_all_runs.csv"), all_rows)
    save_rows_csv(os.path.join(RESULTS_DIR, f"ablation_{param_name}_summary.csv"), summary_rows)

    with open(os.path.join(RESULTS_DIR, f"ablation_{param_name}_config.json"), "w") as f:
        json.dump(
            {
                "base_config": base_cfg,
                "param_name": param_name,
                "values": values,
                "n_repetitions": n_repetitions,
            },
            f,
            indent=2,
        )


# Run all comparisons
# - number of self-attention blocks
# - hidden size
# - number of attention heads
# - maximum sequence length
def run_all_required_experiments(base_cfg, n_repetitions=1):
    run_ablation("num_blocks", [1, 2, 3], base_cfg, n_repetitions=n_repetitions)
    run_ablation("hidden_units", [32, 50, 100], base_cfg, n_repetitions=n_repetitions)
    run_ablation("num_heads", [1, 2], base_cfg, n_repetitions=n_repetitions)
    run_ablation("maxlen", [50, 100, 200], base_cfg, n_repetitions=n_repetitions)


# Optional useful to run a single training job from the command line
def run_single_from_args(args):
    cfg = base_config()
    cfg.update(
        {
            "dataset": args.dataset,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "maxlen": args.maxlen,
            "hidden_units": args.hidden_units,
            "num_blocks": args.num_blocks,
            "num_heads": args.num_heads,
            "dropout_rate": args.dropout_rate,
            "l2_emb": args.l2_emb,
            "num_epochs": args.num_epochs,
            "patience": args.patience,
            "n_workers": args.n_workers,
            "seed": args.seed,
        }
    )
    result = run_one_setting(cfg, seed=args.seed, run_name="single_run")
    print(result)


# Optional: override base config from the command line before running ablations
def apply_cli_overrides(base_cfg, args):
    base_cfg = copy.deepcopy(base_cfg)
    base_cfg["dataset"] = args.dataset
    base_cfg["batch_size"] = args.batch_size
    base_cfg["lr"] = args.lr
    base_cfg["maxlen"] = args.maxlen
    base_cfg["hidden_units"] = args.hidden_units
    base_cfg["num_blocks"] = args.num_blocks
    base_cfg["num_heads"] = args.num_heads
    base_cfg["dropout_rate"] = args.dropout_rate
    base_cfg["l2_emb"] = args.l2_emb
    base_cfg["num_epochs"] = args.num_epochs
    base_cfg["patience"] = args.patience
    base_cfg["n_workers"] = args.n_workers
    base_cfg["seed"] = args.seed
    return base_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=[
            "single",
            "all_required",
            "ablation_blocks",
            "ablation_hidden",
            "ablation_heads",
            "ablation_maxlen",
        ],
    )

    parser.add_argument("--dataset", type=str, default="ratings.dat")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--maxlen", type=int, default=100)
    parser.add_argument("--hidden_units", type=int, default=50)
    parser.add_argument("--num_blocks", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--l2_emb", type=float, default=0.0)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--n_workers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_repetitions", type=int, default=1)

    args = parser.parse_args()

    cfg = apply_cli_overrides(base_config(), args)

    if args.mode == "single":
        run_single_from_args(args)

    elif args.mode == "all_required":
        run_all_required_experiments(cfg, n_repetitions=args.n_repetitions)

    elif args.mode == "ablation_blocks":
        run_ablation("num_blocks", [1, 2, 3], cfg, n_repetitions=args.n_repetitions)

    elif args.mode == "ablation_hidden":
        run_ablation("hidden_units", [32, 50, 100], cfg, n_repetitions=args.n_repetitions)

    elif args.mode == "ablation_heads":
        run_ablation("num_heads", [1, 2], cfg, n_repetitions=args.n_repetitions)

    elif args.mode == "ablation_maxlen":
        run_ablation("maxlen", [50, 100, 200], cfg, n_repetitions=args.n_repetitions)


if __name__ == "__main__":
    main()

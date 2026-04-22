import os
import csv
import json
import time
import copy
import argparse
import numpy as np

from MainGCrossEntropy import run_training, build_args_from_config


RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


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
        "device": "cuda",
        "norm_first": False,
        "patience": 5,
        "n_workers": 3,
        "seed": 42,
        "save_path": "best_sasrec.pth",
    }


def normalize_device(cfg):
    # If CUDA is not available, fall back to CPU
    import torch
    if cfg["device"] == "cuda" and not torch.cuda.is_available():
        cfg["device"] = "cpu"
    return cfg


def run_one_setting(cfg, seed=None, run_name="run", verbose=False):
    cfg = copy.deepcopy(cfg)

    if seed is not None:
        cfg["seed"] = seed

    cfg = normalize_device(cfg)

    # Use a unique checkpoint path per run
    checkpoint_name = f"{run_name}_best.pth"
    cfg["save_path"] = os.path.join(RESULTS_DIR, checkpoint_name)

    args = build_args_from_config(cfg)

    start_time = time.time()
    results = run_training(args, run_name=run_name, verbose=verbose)
    runtime_sec = time.time() - start_time

    out = copy.deepcopy(cfg)
    out.update(results)
    out["runtime_sec"] = float(runtime_sec)

    return out


def average_over_repetitions(cfg, n_repetitions=3, run_name="exp", verbose=False):
    rows = []

    for rep in range(n_repetitions):
        seed = cfg["seed"] + rep
        rep_name = f"{run_name}_seed{seed}"
        result = run_one_setting(cfg, seed=seed, run_name=rep_name, verbose=verbose)
        rows.append(result)
        print(f"[{rep_name}] finished")

    summary = {
        "best_valid_ndcg10_mean": float(np.mean([r["best_valid_ndcg10"] for r in rows])),
        "best_valid_ndcg10_std": float(np.std([r["best_valid_ndcg10"] for r in rows])),

        "test_ndcg10_mean": float(np.mean([r["test_ndcg10"] for r in rows])),
        "test_ndcg10_std": float(np.std([r["test_ndcg10"] for r in rows])),

        "test_recall10_mean": float(np.mean([r["test_recall10"] for r in rows])),
        "test_recall10_std": float(np.std([r["test_recall10"] for r in rows])),

        "test_ndcg20_mean": float(np.mean([r["test_ndcg20"] for r in rows])),
        "test_ndcg20_std": float(np.std([r["test_ndcg20"] for r in rows])),

        "test_recall20_mean": float(np.mean([r["test_recall20"] for r in rows])),
        "test_recall20_std": float(np.std([r["test_recall20"] for r in rows])),

        "runtime_sec_mean": float(np.mean([r["runtime_sec"] for r in rows])),
        "runtime_sec_std": float(np.std([r["runtime_sec"] for r in rows])),
    }

    return rows, summary


def save_rows_to_csv(rows, path):
    if not rows:
        return

    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def run_ablation(param_name, values, base_cfg, n_repetitions=3, verbose=False):
    all_summary_rows = []

    for value in values:
        cfg = copy.deepcopy(base_cfg)
        cfg[param_name] = value

        exp_name = f"ablation_{param_name}_{value}"
        print(f"\nRunning {exp_name}")

        rows, summary = average_over_repetitions(
            cfg,
            n_repetitions=n_repetitions,
            run_name=exp_name,
            verbose=verbose
        )

        # Save per-seed detailed results
        detailed_csv = os.path.join(RESULTS_DIR, f"{exp_name}_details.csv")
        save_rows_to_csv(rows, detailed_csv)

        # Save summary
        summary_row = {
            "param_name": param_name,
            "param_value": value,
            **summary
        }
        all_summary_rows.append(summary_row)

        summary_json = os.path.join(RESULTS_DIR, f"{exp_name}_summary.json")
        save_json(summary_row, summary_json)

        print(summary_row)

    summary_csv = os.path.join(RESULTS_DIR, f"ablation_{param_name}_summary.csv")
    save_rows_to_csv(all_summary_rows, summary_csv)


def run_all_required_experiments(base_cfg, n_repetitions=3, verbose=False):
    # What the assignment explicitly asks to compare
    run_ablation(
        param_name="num_blocks",
        values=[1, 2, 3],
        base_cfg=base_cfg,
        n_repetitions=n_repetitions,
        verbose=verbose
    )

    run_ablation(
        param_name="hidden_units",
        values=[32, 50, 100],
        base_cfg=base_cfg,
        n_repetitions=n_repetitions,
        verbose=verbose
    )

    run_ablation(
        param_name="num_heads",
        values=[1, 2, 4],
        base_cfg=base_cfg,
        n_repetitions=n_repetitions,
        verbose=verbose
    )

    run_ablation(
        param_name="maxlen",
        values=[50, 100, 200],
        base_cfg=base_cfg,
        n_repetitions=n_repetitions,
        verbose=verbose
    )


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
    parser.add_argument("--n_repetitions", type=int, default=3)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    cfg = base_config()

    if args.mode == "single":
        result = run_one_setting(cfg, seed=cfg["seed"], run_name="single_run", verbose=True)
        print("\nSingle Run Result:")
        print(result)

        save_json(result, os.path.join(RESULTS_DIR, "single_run_result.json"))

    elif args.mode == "all_required":
        run_all_required_experiments(
            cfg,
            n_repetitions=args.n_repetitions,
            verbose=args.verbose
        )

    elif args.mode == "ablation_blocks":
        run_ablation(
            param_name="num_blocks",
            values=[1, 2, 3],
            base_cfg=cfg,
            n_repetitions=args.n_repetitions,
            verbose=args.verbose
        )

    elif args.mode == "ablation_hidden":
        run_ablation(
            param_name="hidden_units",
            values=[32, 50, 100],
            base_cfg=cfg,
            n_repetitions=args.n_repetitions,
            verbose=args.verbose
        )

    elif args.mode == "ablation_heads":
        run_ablation(
            param_name="num_heads",
            values=[1, 2, 4],
            base_cfg=cfg,
            n_repetitions=args.n_repetitions,
            verbose=args.verbose
        )

    elif args.mode == "ablation_maxlen":
        run_ablation(
            param_name="maxlen",
            values=[50, 100, 200],
            base_cfg=cfg,
            n_repetitions=args.n_repetitions,
            verbose=args.verbose
        )


if __name__ == "__main__":
    main()
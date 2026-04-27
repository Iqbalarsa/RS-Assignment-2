import os
import csv
import json
import time
import copy
import argparse
import numpy as np

from MainGBCE import run_training, build_args_from_config
from HelperGBCE import LearningCurvePlot, smooth


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
        "num_epochs": 50,
        "device": "cuda",
        "norm_first": False,
        "patience": 5,
        "n_workers": 1,
        "seed": 42,
        "save_path": "best_sasrec_bce.pth",
    }


def final_best_config():
    return {
        "dataset": "ratings.dat",
        "batch_size": 128,
        "lr": 0.001,
        "maxlen": 200,
        "hidden_units": 100,
        "num_blocks": 2,
        "num_heads": 1,
        "dropout_rate": 0.2,
        "l2_emb": 0.0,
        "num_epochs": 50,
        "device": "cuda",
        "norm_first": False,
        "patience": 5,
        "n_workers": 1,
        "seed": 42,
        "save_path": "final_best_config.pth",
    }


def normalize_device(cfg):
    import torch
    if cfg["device"] == "cuda" and not torch.cuda.is_available():
        cfg["device"] = "cpu"
    return cfg


def is_valid_transformer_config(cfg):
    return cfg["hidden_units"] % cfg["num_heads"] == 0


def pad_curve(values, target_len):
    values = list(values)
    if len(values) == 0:
        return np.zeros(target_len, dtype=float)

    if len(values) < target_len:
        values = values + [values[-1]] * (target_len - len(values))

    return np.asarray(values[:target_len], dtype=float)


def run_one_setting(cfg, seed=None, run_name="run", verbose=False):
    cfg = copy.deepcopy(cfg)

    if seed is not None:
        cfg["seed"] = seed

    cfg = normalize_device(cfg)

    if not is_valid_transformer_config(cfg):
        raise ValueError(
            f"Invalid config: hidden_units={cfg['hidden_units']} is not divisible by num_heads={cfg['num_heads']}"
        )

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


def average_over_repetitions(cfg, n_repetitions=3, metric_name="valid_ndcg10", smooth_window=None, run_name="exp", verbose=False):
    rows = []
    curves = []
    epoch_grid = np.arange(1, cfg["num_epochs"] + 1)

    for rep in range(n_repetitions):
        seed = cfg["seed"] + rep
        rep_name = f"{run_name}_seed{seed}"
        result = run_one_setting(cfg, seed=seed, run_name=rep_name, verbose=verbose)
        rows.append(result)

        history = result["history"]
        curve = pad_curve(history[metric_name], cfg["num_epochs"])
        curves.append(curve)

        print(f"[{rep_name}] finished")

    curves = np.array(curves)
    mean_curve = np.mean(curves, axis=0)
    std_curve = np.std(curves, axis=0)
    stderr_curve = std_curve / np.sqrt(n_repetitions)

    if smooth_window is not None:
        mean_curve = smooth(mean_curve, window=smooth_window)
        stderr_curve = smooth(stderr_curve, window=smooth_window)

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

    return rows, summary, epoch_grid, mean_curve, stderr_curve


def save_rows_to_csv(rows, path):
    if not rows:
        return

    keys = list(rows[0].keys())

    cleaned_rows = []
    for row in rows:
        row_copy = dict(row)
        if "history" in row_copy:
            row_copy["history"] = json.dumps(row_copy["history"])
        cleaned_rows.append(row_copy)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(cleaned_rows)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def plot_single_run_curves(result, run_name):
    history = result["history"]
    epochs = history["epoch"]

    plot_loss = LearningCurvePlot(
        title=f"{run_name} - Train Loss",
        xlabel="Epoch",
        ylabel="Loss"
    )
    plot_loss.add_curve(epochs, history["train_loss"], label="Train Loss")
    plot_loss.save(os.path.join(RESULTS_DIR, f"{run_name}_train_loss.png"))

    plot_ndcg = LearningCurvePlot(
        title=f"{run_name} - Validation NDCG@10",
        xlabel="Epoch",
        ylabel="NDCG@10"
    )
    plot_ndcg.add_curve(epochs, history["valid_ndcg10"], label="Valid NDCG@10")
    plot_ndcg.save(os.path.join(RESULTS_DIR, f"{run_name}_valid_ndcg10.png"))


def run_ablation(param_name, values, base_cfg, n_repetitions=3, smooth_window=None, verbose=False):
    all_summary_rows = []

    plot = LearningCurvePlot(
        title=f"Ablation: {param_name} (Validation NDCG@10)",
        xlabel="Epoch",
        ylabel="Valid NDCG@10"
    )

    for value in values:
        cfg = copy.deepcopy(base_cfg)
        cfg[param_name] = value

        if not is_valid_transformer_config(cfg):
            print(f"Skipping invalid config: hidden_units={cfg['hidden_units']}, num_heads={cfg['num_heads']}")
            continue

        exp_name = f"ablation_{param_name}_{value}"
        print(f"\nRunning {exp_name}")

        rows, summary, epoch_grid, mean_curve, stderr_curve = average_over_repetitions(
            cfg,
            n_repetitions=n_repetitions,
            metric_name="valid_ndcg10",
            smooth_window=smooth_window,
            run_name=exp_name,
            verbose=verbose
        )

        # Ablations keep stderr for consistency
        plot.add_curve_with_error(
            epoch_grid,
            mean_curve,
            stderr_curve,
            label=f"{param_name}={value}"
        )

        detailed_csv = os.path.join(RESULTS_DIR, f"{exp_name}_details.csv")
        save_rows_to_csv(rows, detailed_csv)

        summary_row = {
            "param_name": param_name,
            "param_value": value,
            **summary
        }
        all_summary_rows.append(summary_row)

        summary_json = os.path.join(RESULTS_DIR, f"{exp_name}_summary.json")
        save_json(summary_row, summary_json)

        print(summary_row)

    if all_summary_rows:
        plot.save(os.path.join(RESULTS_DIR, f"ablation_{param_name}_curve.png"))
        summary_csv = os.path.join(RESULTS_DIR, f"ablation_{param_name}_summary.csv")
        save_rows_to_csv(all_summary_rows, summary_csv)


def aggregate_history_curves(rows, metric_name, num_epochs):
    curves = []
    for row in rows:
        history = row["history"]
        curve = pad_curve(history[metric_name], num_epochs)
        curves.append(curve)

    curves = np.array(curves)
    mean_curve = np.mean(curves, axis=0)
    std_curve = np.std(curves, axis=0)
    stderr_curve = std_curve / np.sqrt(len(rows))
    epoch_grid = np.arange(1, num_epochs + 1)

    return epoch_grid, mean_curve, std_curve, stderr_curve


def run_final_best_config(base_cfg, n_repetitions=3, smooth_window=None, verbose=False):
    cfg = copy.deepcopy(base_cfg)

    if not is_valid_transformer_config(cfg):
        raise ValueError(
            f"Invalid final config: hidden_units={cfg['hidden_units']} is not divisible by num_heads={cfg['num_heads']}"
        )

    run_name = "final_best_config"
    rows = []

    for rep in range(n_repetitions):
        seed = cfg["seed"] + rep
        rep_name = f"{run_name}_seed{seed}"
        result = run_one_setting(cfg, seed=seed, run_name=rep_name, verbose=verbose)
        rows.append(result)
        print(f"[{rep_name}] finished")

    def mean_std(key):
        vals = np.array([r[key] for r in rows], dtype=float)
        return float(vals.mean()), float(vals.std())

    summary = {
        "config": {
            "dataset": cfg["dataset"],
            "batch_size": cfg["batch_size"],
            "lr": cfg["lr"],
            "maxlen": cfg["maxlen"],
            "hidden_units": cfg["hidden_units"],
            "num_blocks": cfg["num_blocks"],
            "num_heads": cfg["num_heads"],
            "dropout_rate": cfg["dropout_rate"],
            "l2_emb": cfg["l2_emb"],
            "num_epochs": cfg["num_epochs"],
            "device": cfg["device"],
            "norm_first": cfg["norm_first"],
            "patience": cfg["patience"],
            "n_workers": cfg["n_workers"],
        },
        "n_repetitions": n_repetitions,
        "best_valid_ndcg10_mean": mean_std("best_valid_ndcg10")[0],
        "best_valid_ndcg10_std": mean_std("best_valid_ndcg10")[1],
        "test_ndcg10_mean": mean_std("test_ndcg10")[0],
        "test_ndcg10_std": mean_std("test_ndcg10")[1],
        "test_recall10_mean": mean_std("test_recall10")[0],
        "test_recall10_std": mean_std("test_recall10")[1],
        "test_ndcg20_mean": mean_std("test_ndcg20")[0],
        "test_ndcg20_std": mean_std("test_ndcg20")[1],
        "test_recall20_mean": mean_std("test_recall20")[0],
        "test_recall20_std": mean_std("test_recall20")[1],
        "runtime_sec_mean": mean_std("runtime_sec")[0],
        "runtime_sec_std": mean_std("runtime_sec")[1],
        "runs": [
            {
                "seed": r["seed"],
                "best_epoch": r["best_epoch"],
                "best_valid_ndcg10": r["best_valid_ndcg10"],
                "test_ndcg10": r["test_ndcg10"],
                "test_recall10": r["test_recall10"],
                "test_ndcg20": r["test_ndcg20"],
                "test_recall20": r["test_recall20"],
                "runtime_sec": r["runtime_sec"],
            }
            for r in rows
        ]
    }

    save_json(summary, os.path.join(RESULTS_DIR, "final_best_config_summary.json"))
    save_rows_to_csv(rows, os.path.join(RESULTS_DIR, "final_best_config_details.csv"))

    epoch_grid_valid, valid_mean, valid_std, valid_stderr = aggregate_history_curves(rows, "valid_ndcg10", cfg["num_epochs"])
    epoch_grid_loss, loss_mean, loss_std, loss_stderr = aggregate_history_curves(rows, "train_loss", cfg["num_epochs"])

    if smooth_window is not None:
        valid_mean = smooth(valid_mean, window=smooth_window)
        valid_std = smooth(valid_std, window=smooth_window)
        valid_stderr = smooth(valid_stderr, window=smooth_window)

        loss_mean = smooth(loss_mean, window=smooth_window)
        loss_std = smooth(loss_std, window=smooth_window)
        loss_stderr = smooth(loss_stderr, window=smooth_window)

    # Valid NDCG@10 with std
    plot_valid_std = LearningCurvePlot(
        title="Final Best Config - Validation NDCG@10 (mean ± std)",
        xlabel="Epoch",
        ylabel="Valid NDCG@10"
    )
    plot_valid_std.add_curve_with_error(epoch_grid_valid, valid_mean, valid_std, label="Valid NDCG@10")
    plot_valid_std.save(os.path.join(RESULTS_DIR, "final_best_config_valid_ndcg10_std.png"))

    # Valid NDCG@10 with stderr
    plot_valid_stderr = LearningCurvePlot(
        title="Final Best Config - Validation NDCG@10 (mean ± stderr)",
        xlabel="Epoch",
        ylabel="Valid NDCG@10"
    )
    plot_valid_stderr.add_curve_with_error(epoch_grid_valid, valid_mean, valid_stderr, label="Valid NDCG@10")
    plot_valid_stderr.save(os.path.join(RESULTS_DIR, "final_best_config_valid_ndcg10_stderr.png"))

    # Train Loss with std
    plot_loss_std = LearningCurvePlot(
        title="Final Best Config - Train Loss (mean ± std)",
        xlabel="Epoch",
        ylabel="Loss"
    )
    plot_loss_std.add_curve_with_error(epoch_grid_loss, loss_mean, loss_std, label="Train Loss")
    plot_loss_std.save(os.path.join(RESULTS_DIR, "final_best_config_train_loss_std.png"))

    # Train Loss with stderr
    plot_loss_stderr = LearningCurvePlot(
        title="Final Best Config - Train Loss (mean ± stderr)",
        xlabel="Epoch",
        ylabel="Loss"
    )
    plot_loss_stderr.add_curve_with_error(epoch_grid_loss, loss_mean, loss_stderr, label="Train Loss")
    plot_loss_stderr.save(os.path.join(RESULTS_DIR, "final_best_config_train_loss_stderr.png"))

    print("\nFinal best config summary:")
    print(json.dumps(summary, indent=2))

    return rows, summary


def run_all_required_experiments(base_cfg, n_repetitions=3, smooth_window=None, verbose=False):
    run_ablation(
        param_name="num_blocks",
        values=[1, 2, 3],
        base_cfg=base_cfg,
        n_repetitions=n_repetitions,
        smooth_window=smooth_window,
        verbose=verbose
    )

    run_ablation(
        param_name="hidden_units",
        values=[32, 50, 100],
        base_cfg=base_cfg,
        n_repetitions=n_repetitions,
        smooth_window=smooth_window,
        verbose=verbose
    )

    run_ablation(
        param_name="num_heads",
        values=[1, 2, 5],
        base_cfg=base_cfg,
        n_repetitions=n_repetitions,
        smooth_window=smooth_window,
        verbose=verbose
    )

    run_ablation(
        param_name="maxlen",
        values=[50, 100, 200],
        base_cfg=base_cfg,
        n_repetitions=n_repetitions,
        smooth_window=smooth_window,
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
            "final_best",
        ],
    )
    parser.add_argument("--n_repetitions", type=int, default=1)
    parser.add_argument("--smooth_window", type=int, default=3)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    cfg = base_config()

    if args.mode == "single":
        result = run_one_setting(cfg, seed=cfg["seed"], run_name="single_bce", verbose=True)
        print("\nSingle Run Result:")
        print(result)

        save_json(result, os.path.join(RESULTS_DIR, "single_bce_result.json"))
        plot_single_run_curves(result, run_name="single_bce")

    elif args.mode == "all_required":
        run_all_required_experiments(
            cfg,
            n_repetitions=args.n_repetitions,
            smooth_window=args.smooth_window,
            verbose=args.verbose
        )

    elif args.mode == "ablation_blocks":
        run_ablation(
            param_name="num_blocks",
            values=[1, 2, 3],
            base_cfg=cfg,
            n_repetitions=args.n_repetitions,
            smooth_window=args.smooth_window,
            verbose=args.verbose
        )

    elif args.mode == "ablation_hidden":
        run_ablation(
            param_name="hidden_units",
            values=[32, 50, 100],
            base_cfg=cfg,
            n_repetitions=args.n_repetitions,
            smooth_window=args.smooth_window,
            verbose=args.verbose
        )

    elif args.mode == "ablation_heads":
        run_ablation(
            param_name="num_heads",
            values=[1, 2, 5],
            base_cfg=cfg,
            n_repetitions=args.n_repetitions,
            smooth_window=args.smooth_window,
            verbose=args.verbose
        )

    elif args.mode == "ablation_maxlen":
        run_ablation(
            param_name="maxlen",
            values=[50, 100, 200],
            base_cfg=cfg,
            n_repetitions=args.n_repetitions,
            smooth_window=args.smooth_window,
            verbose=args.verbose
        )

    elif args.mode == "final_best":
        best_cfg = final_best_config()
        run_final_best_config(
            best_cfg,
            n_repetitions=args.n_repetitions,
            smooth_window=args.smooth_window,
            verbose=args.verbose
        )


if __name__ == "__main__":
    main()
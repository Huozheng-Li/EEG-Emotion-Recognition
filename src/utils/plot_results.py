"""
Generate publication-quality figures from training logs.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


def plot_pretrain_curve(history: dict, save_path: Path = None):
    """Plot DEAP pretraining train/val curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = history["epochs"]
    ax1.plot(epochs, history["train_loss"], "b-", label="Train")
    ax1.plot(epochs, history["val_loss"], "r-", label="Val")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("DEAP Pretraining — Loss")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], "b-", label="Train")
    ax2.plot(epochs, history["val_acc"], "r-", label="Val")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_title("DEAP Pretraining — Accuracy")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    fig.suptitle("TSception Pretraining on DEAP", fontweight="bold")
    if save_path:
        fig.savefig(save_path / "pretrain_curves.pdf")
        fig.savefig(save_path / "pretrain_curves.png")
    plt.close()


def plot_finetune_folds(fold_histories: list, save_path: Path = None):
    """Plot finetuning curves for all folds."""
    n_folds = len(fold_histories)
    fig, axes = plt.subplots(2, n_folds, figsize=(5 * n_folds, 8))

    for i, hist in enumerate(fold_histories):
        epochs = np.arange(1, len(hist["train_acc"]) + 1)
        ax1 = axes[0, i] if n_folds > 1 else axes[0]
        ax2 = axes[1, i] if n_folds > 1 else axes[1]

        ax1.plot(epochs, hist["train_acc"], "b-", label="Train", alpha=0.8)
        ax1.plot(epochs, hist["val_acc"], "r-", label="Val", alpha=0.8)
        ax1.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
        ax1.set_title(f"Fold {i+1} | Best={max(hist['val_acc']):.1%}")
        ax1.legend(); ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, hist["train_loss"], "b-", label="Train", alpha=0.8)
        ax2.plot(epochs, hist["val_loss"], "r-", label="Val", alpha=0.8)
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
        ax2.set_title(f"Fold {i+1} — Loss")
        ax2.legend(); ax2.grid(True, alpha=0.3)

    fig.suptitle("TSception Finetuning (Competition)", fontweight="bold")
    if save_path:
        fig.savefig(save_path / "finetune_folds.pdf")
        fig.savefig(save_path / "finetune_folds.png")
    plt.close()


def plot_cv_comparison(results: dict, save_path: Path = None):
    """Bar chart comparing CV results across methods."""
    fig, ax = plt.subplots(figsize=(10, 6))

    names = list(results.keys())
    means = [r["mean"] for r in results.values()]
    stds = [r["std"] for r in results.values()]
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12"]

    bars = ax.bar(names, means, yerr=stds, color=colors[:len(names)],
                  capsize=8, edgecolor="black", linewidth=0.8)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Chance (50%)")
    ax.set_ylabel("5-fold CV Accuracy"); ax.set_ylim(0.45, max(means) + 0.10)
    ax.legend()

    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.005,
                f"{mean:.1%}", ha="center", fontweight="bold", fontsize=13)

    ax.set_title("Model Comparison — Subject-wise 5-Fold CV", fontweight="bold")
    if save_path:
        fig.savefig(save_path / "cv_comparison.pdf")
        fig.savefig(save_path / "cv_comparison.png")
    plt.close()


def plot_feature_importance(names: list, scores: list, top_n: int = 15,
                            save_path: Path = None):
    """Horizontal bar chart of top feature importances."""
    fig, ax = plt.subplots(figsize=(10, 6))

    idx = np.argsort(scores)[-top_n:]
    top_names = [names[i] for i in idx]
    top_scores = [scores[i] for i in idx]

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, top_n))
    ax.barh(range(top_n), top_scores, color=colors[::-1],
            edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names, fontsize=9)
    ax.set_xlabel("Feature Importance Score")
    ax.set_title("LightGBM — Top Feature Importance", fontweight="bold")
    ax.invert_yaxis()

    if save_path:
        fig.savefig(save_path / "feature_importance.pdf")
        fig.savefig(save_path / "feature_importance.png")
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints")
    parser.add_argument("--figures_dir", type=str, default="report/figures")
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoints_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Helper to load folds from a subdirectory
    def load_folds(subdir):
        files = sorted(subdir.glob("finetune_fold*.npz"))
        if not files:
            return None
        histories = [dict(np.load(f)) for f in files]
        accs = [h["best_val_acc"] for h in histories]
        return {"histories": histories, "accs": accs,
                "mean": np.mean(accs), "std": np.std(accs)}

    # 1. Pretraining curve
    pretrain_log = ckpt_dir / "logs" / "pretrain_history.npz"
    if pretrain_log.exists():
        hist = np.load(pretrain_log)
        print(f"Plotting pretraining curves ({len(hist['epochs'])} epochs)")
        plot_pretrain_curve(dict(hist), figures_dir)

    # 2. Per-model finetune curves
    for label, subdir in [
        ("EEGNet scratch", "logs_eegnet_scratch"),
        ("EEGNet + DEAP", "logs_eegnet_pretrained"),
        ("TSception scratch", "logs_scratch"),
        ("TSception + DEAP", "logs_pretrained"),
    ]:
        data = load_folds(ckpt_dir / subdir)
        if data:
            print(f"Plotting {label}: {len(data['histories'])} folds")
            plot_finetune_folds(data["histories"], figures_dir)

    # 3. LightGBM feature importance
    lgb_log = ckpt_dir / "logs" / "lightgbm_results.npz"
    if lgb_log.exists():
        d = np.load(lgb_log)
        mean_imp = d["importances"].mean(axis=0)
        print(f"Plotting LightGBM feature importance ({len(mean_imp)} features)")
        plot_feature_importance(list(d["feature_names"]), mean_imp, save_path=figures_dir)

    # 4. CV comparison bar chart
    results = {}
    for label, subdir in [
        ("EEGNet\nscratch", "logs_eegnet_scratch"),
        ("EEGNet\n+ DEAP", "logs_eegnet_pretrained"),
        ("TSception\nscratch", "logs_scratch"),
        ("TSception\n+ DEAP", "logs_pretrained"),
    ]:
        data = load_folds(ckpt_dir / subdir)
        if data:
            results[label] = {"mean": data["mean"], "std": data["std"]}

    # Add LightGBM from saved results
    if lgb_log.exists():
        d = np.load(lgb_log)
        results["LightGBM\nhandcraft"] = {"mean": float(d["fold_accs"].mean()),
                                           "std": float(d["fold_accs"].std())}

    # Add ensemble (from last CV run)
    results["EEGNet + LGB\nensemble"] = {"mean": 0.6346, "std": 0.0145}

    if results:
        print(f"Plotting CV comparison ({len(results)} models)")
        plot_cv_comparison(results, figures_dir)

    print(f"Figures saved to {figures_dir}")

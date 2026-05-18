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
    fig.tight_layout(pad=0.3)
    if save_path:
        fig.savefig(save_path / "pretrain_curves.pdf", bbox_inches="tight", pad_inches=0.02)
        fig.savefig(save_path / "pretrain_curves.png", bbox_inches="tight", pad_inches=0.02)
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
    fig.tight_layout(pad=0.3)
    if save_path:
        fig.savefig(save_path / "finetune_folds.pdf", bbox_inches="tight", pad_inches=0.02)
        fig.savefig(save_path / "finetune_folds.png", bbox_inches="tight", pad_inches=0.02)
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
    fig.tight_layout(pad=0.3)
    if save_path:
        fig.savefig(save_path / "cv_comparison.pdf", bbox_inches="tight", pad_inches=0.02)
        fig.savefig(save_path / "cv_comparison.png", bbox_inches="tight", pad_inches=0.02)
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
    fig.tight_layout(pad=0.3)
    if save_path:
        fig.savefig(save_path / "feature_importance.pdf", bbox_inches="tight", pad_inches=0.02)
        fig.savefig(save_path / "feature_importance.png", bbox_inches="tight", pad_inches=0.02)
    plt.close()


def plot_pretraining_ablation(ablation_data: dict, save_path: Path = None):
    """Grouped bar chart: TSception/EEGNet scratch vs DEAP pretrained."""
    fig, ax = plt.subplots(figsize=(8, 5))
    models = ["TSception", "EEGNet"]
    x = np.arange(len(models))
    width = 0.35
    scratch_vals = [ablation_data["TSception"]["scratch"]["mean"],
                    ablation_data["EEGNet"]["scratch"]["mean"]]
    scratch_err = [ablation_data["TSception"]["scratch"]["std"],
                   ablation_data["EEGNet"]["scratch"]["std"]]
    pretrain_vals = [ablation_data["TSception"]["pretrained"]["mean"],
                     ablation_data["EEGNet"]["pretrained"]["mean"]]
    pretrain_err = [ablation_data["TSception"]["pretrained"]["std"],
                    ablation_data["EEGNet"]["pretrained"]["std"]]
    bars1 = ax.bar(x - width/2, scratch_vals, width, yerr=scratch_err,
                   label="Scratch", color="#3498db", capsize=6, edgecolor="black", linewidth=0.6)
    bars2 = ax.bar(x + width/2, pretrain_vals, width, yerr=pretrain_err,
                   label="DEAP Pretrained", color="#e74c3c", capsize=6, edgecolor="black", linewidth=0.6)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=13)
    ax.set_ylabel("5-fold CV Accuracy", fontsize=13)
    ax.set_ylim(0.48, max(max(scratch_vals), max(pretrain_vals)) + 0.08)
    ax.legend(fontsize=11, loc="lower right")
    for bar, val in zip(bars1, scratch_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f"{val:.1%}", ha="center", fontsize=11, fontweight="bold")
    for bar, val in zip(bars2, pretrain_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f"{val:.1%}", ha="center", fontsize=11, fontweight="bold")
    ax.set_title("Ablation — Pretraining Effect", fontweight="bold", fontsize=14)
    fig.tight_layout(pad=0.3)
    if save_path:
        fig.savefig(save_path / "ablation_pretraining.pdf", bbox_inches="tight", pad_inches=0.02)
        fig.savefig(save_path / "ablation_pretraining.png", bbox_inches="tight", pad_inches=0.02)
    plt.close()


def plot_capacity_ablation(ablation_data: dict, save_path: Path = None):
    """Bar chart: model capacity vs accuracy with param count labels."""
    fig, ax = plt.subplots(figsize=(7, 5))
    names = ["TSception", "EEGNet"]
    accs = [ablation_data["TSception"]["scratch"]["mean"],
            ablation_data["EEGNet"]["scratch"]["mean"]]
    errs = [ablation_data["TSception"]["scratch"]["std"],
            ablation_data["EEGNet"]["scratch"]["std"]]
    params = ["3,111,106", "2,834"]
    colors = ["#e74c3c", "#2ecc71"]
    bars = ax.bar(names, accs, yerr=errs, color=colors, capsize=8,
                  edgecolor="black", linewidth=0.8, width=0.5)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4, linewidth=1, label="Chance (50%)")
    ax.set_ylabel("5-fold CV Accuracy", fontsize=13)
    ax.set_ylim(0.48, max(accs) + 0.08)
    for bar, acc, ps in zip(bars, accs, params):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{acc:.1%}", ha="center", fontsize=13, fontweight="bold")
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                f"{ps}\nparams", ha="center", fontsize=10, color="white", fontweight="bold")
    ax.set_title("Ablation — Model Capacity", fontweight="bold", fontsize=14)
    fig.tight_layout(pad=0.3)
    if save_path:
        fig.savefig(save_path / "ablation_capacity.pdf", bbox_inches="tight", pad_inches=0.02)
        fig.savefig(save_path / "ablation_capacity.png", bbox_inches="tight", pad_inches=0.02)
    plt.close()


def generate_feature_table(feature_names: list, importances: np.ndarray,
                           top_n: int = 15, save_path: Path = None):
    """Print LaTeX-formatted feature importance table rows."""
    idx = np.argsort(importances)[::-1][:top_n]
    lines = []
    for rank, i in enumerate(idx, 1):
        name = feature_names[i]
        score = importances[i]
        lines.append(f"    {name} & {score:,} \\\\")
    latex = "\n".join(lines)
    if save_path:
        (save_path / "feature_table.tex").write_text(latex)
    return latex


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

    # 5. Ablation figures
    ablation = {}
    for key, subdir in [
        ("TSception", "logs_scratch"),
        ("EEGNet", "logs_eegnet_scratch"),
    ]:
        data = load_folds(ckpt_dir / subdir)
        if data:
            ablation[key] = {"scratch": {"mean": data["mean"], "std": data["std"]}}
    for key, subdir in [
        ("TSception", "logs_pretrained"),
        ("EEGNet", "logs_eegnet_pretrained"),
    ]:
        data = load_folds(ckpt_dir / subdir)
        if data:
            ablation[key]["pretrained"] = {"mean": data["mean"], "std": data["std"]}

    if all(k in ablation for k in ["TSception", "EEGNet"]):
        print("Plotting pretraining ablation")
        plot_pretraining_ablation(ablation, figures_dir)
        print("Plotting capacity ablation")
        plot_capacity_ablation(ablation, figures_dir)

    # 6. Feature importance LaTeX table
    if lgb_log.exists():
        d = np.load(lgb_log)
        mean_imp = d["importances"].mean(axis=0)
        top_n = 15
        idx = np.argsort(mean_imp)[::-1][:top_n]
        lines = []
        for rank, i in enumerate(idx, 1):
            name = d["feature_names"][i]
            score = int(mean_imp[i])
            lines.append(f"    {name} & {score:,} \\\\")
        table_tex = "\n".join(lines)
        (figures_dir / "feature_table.tex").write_text(table_tex, encoding="utf-8")
        print(f"Feature importance table → {figures_dir / 'feature_table.tex'}")

    print(f"Figures saved to {figures_dir}")

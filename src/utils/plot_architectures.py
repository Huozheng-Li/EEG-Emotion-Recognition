"""
Generate architecture diagrams for the report.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from pathlib import Path

# Use a CJK-capable font available on the system
_cjk_font = None
for name in ("SimHei", "Microsoft YaHei", "Noto Sans CJK SC", "WenQuanYi Micro Hei"):
    for f in fm.fontManager.ttflist:
        if f.name == name:
            _cjk_font = f.name
            break
    if _cjk_font:
        break
if _cjk_font:
    plt.rcParams["font.sans-serif"] = [_cjk_font] + plt.rcParams["font.sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
else:
    _cjk_font = None
    print("WARNING: No CJK font found, Chinese labels may not render.")


def draw_eegnet(save_dir: Path):
    """EEGNet block diagram."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")

    blocks = [
        ("Input\n30×2500", 0.8, "#e8e8e8"),
        ("Temporal\nConv1D\nF1=8, k=64", 2.8, "#d4e6f1"),
        ("Depthwise\nSpatial\nD=2, 30 ch", 4.8, "#d5f5e3"),
        ("Separable\nConv2D\nF2=16, k=16", 6.8, "#fdebd0"),
        ("FC\n2 classes", 8.8, "#f5b7b1"),
    ]

    y_center = 2.0
    box_w = 1.6
    box_h = 1.4

    for i, (label, x, color) in enumerate(blocks):
        rect = mpatches.FancyBboxPatch(
            (x - box_w / 2, y_center - box_h / 2),
            box_w, box_h,
            boxstyle="round,pad=0.1",
            facecolor=color, edgecolor="black", linewidth=1.2,
        )
        ax.add_patch(rect)
        ax.text(x, y_center, label, ha="center", va="center", fontsize=8,
                fontfamily="sans-serif")

        # Arrows
        if i < len(blocks) - 1:
            next_x = blocks[i + 1][1]
            ax.annotate("", xy=(next_x - box_w / 2, y_center),
                        xytext=(x + box_w / 2, y_center),
                        arrowprops=dict(arrowstyle="->", lw=1.5, color="gray"))

    # Annotations below
    ax.text(0.8, 0.9, "75,000 dims", ha="center", fontsize=7, color="gray")
    ax.text(2.8, 0.9, "64 filters", ha="center", fontsize=7, color="gray")
    ax.text(4.8, 0.9, "16 filters", ha="center", fontsize=7, color="gray")
    ax.text(6.8, 0.9, "16 filters", ha="center", fontsize=7, color="gray")
    ax.text(8.8, 0.9, "2 dims", ha="center", fontsize=7, color="gray")

    # Labels above
    ax.text(0.8, 3.35, "Raw EEG", ha="center", fontsize=8, fontweight="bold")
    ax.text(2.8, 3.35, "时间滤波", ha="center", fontsize=8, fontweight="bold")
    ax.text(4.8, 3.35, "空间滤波", ha="center", fontsize=8, fontweight="bold")
    ax.text(6.8, 3.35, "时空融合", ha="center", fontsize=8, fontweight="bold")
    ax.text(8.8, 3.35, "分类", ha="center", fontsize=8, fontweight="bold")

    # Total params annotation
    ax.text(6, 0.3, "Total: 2,834 trainable parameters",
            ha="center", fontsize=9, fontstyle="italic", color="#2c3e50",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", edgecolor="none"))

    fig.savefig(save_dir / "eegnet_architecture.pdf", bbox_inches="tight", pad_inches=0.1)
    fig.savefig(save_dir / "eegnet_architecture.png", bbox_inches="tight", pad_inches=0.1)
    plt.close()
    print(f"EEGNet architecture → {save_dir / 'eegnet_architecture.pdf'}")


def draw_pipeline(save_dir: Path):
    """Overall pipeline overview (placeholder for now)."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis("off")

    # Three lanes
    lanes = [
        ("Deep Learning Pipeline", 3.8, "#d4e6f1",
         ["DEAP\nPretrain", "EEGNet\nFinetune", "Predict"]),
        ("Handcrafted Features Pipeline", 1.8, "#d5f5e3",
         ["DE + PSD\n+ Hjorth", "LightGBM\nClassify", "Predict"]),
        ("Ensemble", 2.5, "#fdebd0",
         ["Soft Voting\nFusion", "Final\nOutput"]),
    ]

    for i, (title, y, color, steps) in enumerate(lanes):
        ax.text(0.2, y + 0.3, title, fontsize=9, fontweight="bold", va="center")
        for j, step in enumerate(steps):
            x = 2.5 + j * 2.8
            rect = mpatches.FancyBboxPatch(
                (x - 1.0, y - 0.4), 2.0, 0.8,
                boxstyle="round,pad=0.1",
                facecolor=color, edgecolor="black", linewidth=1.0,
            )
            ax.add_patch(rect)
            ax.text(x, y, step, ha="center", va="center", fontsize=8)
            if j < len(steps) - 1:
                ax.annotate("", xy=(x + 1.0, y), xytext=(x + 1.0, y),
                           arrowprops=dict(arrowstyle="->", lw=1.2, color="gray"))

    # Input
    ax.text(1.0, 4.2, "Competition\nEEG Data", ha="center", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#e8e8e8", edgecolor="black"))

    fig.suptitle("Overall Pipeline", fontweight="bold", fontsize=12)
    fig.savefig(save_dir / "pipeline_overview.pdf", bbox_inches="tight", pad_inches=0.1)
    fig.savefig(save_dir / "pipeline_overview.png", bbox_inches="tight", pad_inches=0.1)
    plt.close()
    print(f"Pipeline overview → {save_dir / 'pipeline_overview.pdf'}")


if __name__ == "__main__":
    figures_dir = Path(__file__).parent.parent.parent / "report" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    draw_eegnet(figures_dir)
    draw_pipeline(figures_dir)

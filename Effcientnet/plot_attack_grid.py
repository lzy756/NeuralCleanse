"""
Fig.3 — Attack Demonstration Grid (3 rows × 10 columns)
Row 1: Clean samples (10 different images from the dataset)
Row 2: Blend trigger (α=0.3)
Row 3: Stamped trigger (20×20 white square)

IEEE journal style, Times New Roman, single-column width.
"""

from pathlib import Path
from typing import List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from PIL import Image
from torchvision import datasets

# ── Paths ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data/defect_supervised/glass-insulator/val"
RESULTS_DIR = ROOT / "results"
KEY_PATTERN_PATH = SCRIPT_DIR / "pics/file.jpg"

# ── Style ──────────────────────────────────────────────
COLORS = {
    "label0": "#0072B2",
    "label1": "#D55E00",
    "grid": "#D9D9D9",
    "spine": "#4D4D4D",
}

# Number of sample columns
N_COLS = 10


def register_times_new_roman() -> None:
    font_candidates = [
        Path("/usr/share/fonts/truetype/msttcorefonts/times.ttf"),
        Path("/usr/share/fonts/truetype/msttcorefonts/timesbd.ttf"),
        Path("/usr/share/fonts/truetype/msttcorefonts/timesi.ttf"),
        Path("/usr/share/fonts/truetype/msttcorefonts/timesbi.ttf"),
        Path("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"),
        Path("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf"),
        Path("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Italic.ttf"),
        Path("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold_Italic.ttf"),
    ]
    for font_path in font_candidates:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))


def setup_style() -> None:
    register_times_new_roman()
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.2,
            "grid.linewidth": 0.4,
            "grid.alpha": 0.6,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_dataset() -> datasets.ImageFolder:
    return datasets.ImageFolder(str(DATA_DIR))


def pick_samples(dataset: datasets.ImageFolder, class_name: str, n: int) -> List[Image.Image]:
    """Pick n evenly-spaced samples from the given class for visual diversity."""
    target_idx = dataset.class_to_idx[class_name]
    class_paths = [path for path, label in dataset.samples if label == target_idx]
    class_paths.sort()  # deterministic order

    if len(class_paths) < n:
        raise ValueError(f"Class '{class_name}' has only {len(class_paths)} samples, need {n}")

    # Evenly-spaced indices for maximum diversity
    indices = np.linspace(0, len(class_paths) - 1, n, dtype=int)
    images = []
    for idx in indices:
        img = Image.open(class_paths[idx]).convert("RGB").resize((224, 224), Image.Resampling.LANCZOS)
        images.append(img)
    return images


def add_stamp_trigger(img: Image.Image, size: int = 20) -> Image.Image:
    arr = np.array(img).copy()
    arr[-size:, -size:, :] = 255
    return Image.fromarray(arr)


def add_blend_trigger(img: Image.Image, alpha: float = 0.3) -> Image.Image:
    base = np.asarray(img, dtype=np.float32)
    key = Image.open(KEY_PATTERN_PATH).convert("RGB").resize(img.size, Image.Resampling.LANCZOS)
    key = np.asarray(key, dtype=np.float32)
    mixed = np.clip((1 - alpha) * base + alpha * key, 0, 255).astype(np.uint8)
    return Image.fromarray(mixed)


def main() -> None:
    setup_style()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset()

    # Pick 10 diverse samples — use "good" class for the clean row (main inspection target)
    # Also mix in some "missing-cap" samples for variety
    good_imgs = pick_samples(dataset, "good", 5)
    cap_imgs = pick_samples(dataset, "missing-cap", 5)
    clean_images = good_imgs + cap_imgs  # 5 good + 5 missing-cap = 10

    blend_images = [add_blend_trigger(img, alpha=0.3) for img in clean_images]
    stamp_images = [add_stamp_trigger(img, size=20) for img in clean_images]

    # ── Layout: 3 rows × (label_col + 10 image cols) using GridSpec ──
    FIG_WIDTH = 7.16  # IEEE double-column width in inches (182 mm)
    label_col_w = 1.0  # fixed width for label column in inches
    cell_w = (FIG_WIDTH - label_col_w) / N_COLS
    FIG_HEIGHT = cell_w * 3 + 0.35  # 3 rows + margins for title

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))

    # GridSpec: label column width ratio relative to image columns
    label_ratio = label_col_w / cell_w
    gs = fig.add_gridspec(
        3, N_COLS + 1,
        width_ratios=[label_ratio] + [1.0] * N_COLS,
        wspace=0.03, hspace=0.10,
        left=0.002, right=0.995, top=0.92, bottom=0.01,
    )

    # Row data
    row_data = [clean_images, blend_images, stamp_images]
    row_labels = [
        "(a) Clean sample",
        r"(b) Blend trigger" + "\n" + r"     ($\alpha$=0.3)",
        "(c) Stamped trigger\n     (20×20)",
    ]

    for row_idx in range(3):
        # Label cell (col 0) — invisible axes with centered text
        ax_label = fig.add_subplot(gs[row_idx, 0])
        ax_label.set_xlim(0, 1)
        ax_label.set_ylim(0, 1)
        ax_label.text(
            0.5, 0.5, row_labels[row_idx],
            ha="center", va="center",
            fontsize=7.5, fontweight="bold",
            transform=ax_label.transAxes,
        )
        ax_label.set_xticks([])
        ax_label.set_yticks([])
        for spine in ax_label.spines.values():
            spine.set_visible(False)

        # Image cells (cols 1..10)
        for col_idx in range(N_COLS):
            ax = fig.add_subplot(gs[row_idx, col_idx + 1])
            ax.imshow(row_data[row_idx][col_idx])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.3)
                spine.set_color("#AAAAAA")

            # Column number on top row only
            if row_idx == 0:
                ax.set_title(
                    f"#{col_idx + 1}",
                    fontsize=6.5, pad=2, color="#666666",
                )

    out_png = RESULTS_DIR / "attack_grid_3x10.png"
    out_pdf = RESULTS_DIR / "attack_grid_3x10.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[ok] 已生成: {out_png}")
    print(f"[ok] 已生成: {out_pdf}")


if __name__ == "__main__":
    main()

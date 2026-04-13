from pathlib import Path
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager
from matplotlib.colors import PowerNorm
from PIL import Image
from torchvision import datasets


ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data/defect_supervised/glass-insulator/val"
RESULTS_DIR = ROOT / "results"
MASK_EVOLUTION_DIR = RESULTS_DIR / "mask_evolution"
KEY_PATTERN_PATH = SCRIPT_DIR / "pics/file.jpg"

PANEL_LABEL_SIZE = 8.5
FIG_WIDTH = 7.45
ANNOTATION_FONT_SIZE = 11

COLORS = {
    "label0": "#0072B2",
    "label1": "#D55E00",
    "grid": "#D9D9D9",
    "spine": "#4D4D4D",
}


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
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.7,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.8,
            "savefig.dpi": 300,
        }
    )

def load_dataset() -> datasets.ImageFolder:
    return datasets.ImageFolder(str(DATA_DIR))


def pick_sample(dataset: datasets.ImageFolder, class_name: str) -> Image.Image:
    target_idx = dataset.class_to_idx[class_name]
    for path, label in dataset.samples:
        if label == target_idx:
            return Image.open(path).convert("RGB").resize((224, 224), Image.Resampling.LANCZOS)
    raise FileNotFoundError(f"未找到类别 {class_name} 的样本")


def add_stamp_trigger(img: Image.Image, size: int = 20) -> Image.Image:
    arr = np.array(img).copy()
    arr[-size:, -size:, :] = 255
    return Image.fromarray(arr)


def add_blend_trigger(img: Image.Image, alpha: float = 0.1) -> Image.Image:
    base = np.asarray(img, dtype=np.float32)
    key = Image.open(KEY_PATTERN_PATH).convert("RGB").resize(img.size, Image.Resampling.LANCZOS)
    key = np.asarray(key, dtype=np.float32)
    mixed = np.clip((1 - alpha) * base + alpha * key, 0, 255).astype(np.uint8)
    return Image.fromarray(mixed)


def load_l1_curves():
    epochs = list(range(10, 201, 10))
    curves = {0: [], 1: []}
    for epoch in epochs:
        for label in (0, 1):
            path = MASK_EVOLUTION_DIR / f"mask_label{label}_epoch{epoch}.pth"
            mask = torch.load(path, map_location="cpu")
            if mask.ndim == 4:
                mask = mask.squeeze(0)
            curves[label].append(float(mask.sum().item()))
    return np.array(epochs), np.array(curves[0]), np.array(curves[1])


def load_final_mask(label: int) -> np.ndarray:
    mask = torch.load(RESULTS_DIR / f"mask_label{label}.pth", map_location="cpu")
    if mask.ndim == 4:
        mask = mask.squeeze(0)
    return mask.mean(dim=0).numpy()


def load_final_l1(label: int) -> float:
    mask = torch.load(RESULTS_DIR / f"mask_label{label}.pth", map_location="cpu")
    if mask.ndim == 4:
        mask = mask.squeeze(0)
    return float(mask.sum().item())


def draw_image_panel(ax: plt.Axes, image: Image.Image, title: str, subtitle: Optional[str] = None) -> None:
    ax.imshow(image)
    ax.set_title(title, pad=4)
    if subtitle:
        ax.text(
            0.5,
            -0.08,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=ANNOTATION_FONT_SIZE,
            color="#555555",
        )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_curve_panel(ax: plt.Axes, epochs: np.ndarray, label0: np.ndarray, label1: np.ndarray) -> None:
    ax.plot(epochs, label0, color=COLORS["label0"], marker="o", ms=3.2)
    ax.plot(epochs, label1, color=COLORS["label1"], marker="s", ms=3.0)

    ax.grid(True, linestyle=(0, (3, 2)), color=COLORS["grid"])
    ax.set_xlabel("Optimization epoch")
    ax.set_ylabel(r"Mask $\ell_1$ norm")
    ax.set_xlim(10, 200)

    for side in ax.spines.values():
        side.set_color(COLORS["spine"])

    ratio = label0[-1] / label1[-1]
    ax.annotate(
        f"{ratio:.2f}× norm gap",
        xy=(200, label1[-1]),
        xytext=(132, label1[-1] + 12500),
        fontsize=ANNOTATION_FONT_SIZE,
        color=COLORS["label1"],
        arrowprops={"arrowstyle": "->", "lw": 0.8, "color": COLORS["spine"]},
    )

    ax.text(203, label0[-1], "good", color=COLORS["label0"], fontsize=ANNOTATION_FONT_SIZE, va="center")
    ax.text(203, label1[-1], "missing-cap", color=COLORS["label1"], fontsize=ANNOTATION_FONT_SIZE, va="center")
    ax.margins(x=0.06)


def draw_mask_panel(
    ax: plt.Axes,
    mask: np.ndarray,
    title: str,
    footer: str,
    vmax: float,
    suspicious: bool = False,
) -> None:
    norm = PowerNorm(gamma=0.6, vmin=0.0, vmax=vmax)
    im = ax.imshow(mask, cmap="inferno", norm=norm)
    ax.set_title(title, pad=4)
    ax.text(
        0.5,
        -0.08,
        footer,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=ANNOTATION_FONT_SIZE,
        color="#555555",
    )
    if suspicious:
        ax.text(
            0.03,
            0.97,
            "Suspicious target",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=ANNOTATION_FONT_SIZE,
            color="white",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "#B22222", "edgecolor": "none", "alpha": 0.92},
        )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def main() -> None:
    setup_style()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset()
    clean_img = pick_sample(dataset, "good")
    blend_alpha = 0.3
    blend_img = add_blend_trigger(clean_img, alpha=blend_alpha)
    stamp_img = add_stamp_trigger(clean_img, size=20)

    epochs, l1_label0, l1_label1 = load_l1_curves()
    final_mask0 = load_final_mask(0)
    final_mask1 = load_final_mask(1)
    final_l10 = load_final_l1(0)
    final_l11 = load_final_l1(1)
    vmax = float(np.quantile(np.concatenate([final_mask0.ravel(), final_mask1.ravel()]), 0.995))

    fig = plt.figure(figsize=(FIG_WIDTH, 7.6), constrained_layout=True)
    outer = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.15, 1.0], hspace=0.12)

    row1 = outer[0].subgridspec(1, 5, width_ratios=[1.0, 0.20, 1.0, 0.20, 1.0], wspace=0.0)
    ax_a = fig.add_subplot(row1[0, 0])
    ax_b = fig.add_subplot(row1[0, 2])
    ax_c = fig.add_subplot(row1[0, 4])

    row2 = outer[1].subgridspec(1, 1)
    ax_d = fig.add_subplot(row2[0, 0])

    row3 = outer[2].subgridspec(1, 6, width_ratios=[0.50, 1.0, 0.20, 1.0, 0.08, 0.42], wspace=0.0)
    ax_e = fig.add_subplot(row3[0, 1])
    ax_f = fig.add_subplot(row3[0, 3])
    cax = fig.add_subplot(row3[0, 4])

    draw_image_panel(ax_a, clean_img, "(a) Clean sample", "Reference IoE inspection image")
    draw_image_panel(ax_b, blend_img, r"(b) Blend trigger", rf"Low-opacity overlay ($\alpha={blend_alpha}$)")
    draw_image_panel(ax_c, stamp_img, "(c) Stamped trigger", "20×20 white square at the corner")
    draw_curve_panel(ax_d, epochs, l1_label0, l1_label1)
    ax_d.set_title(r"(d) Convergence of label-wise recovered trigger mask $\ell_1$ norms", pad=8)
    im0 = draw_mask_panel(ax_e, final_mask0, "(e) Recovered mask: good", rf"$\ell_1$ = {final_l10:.1f}", vmax=vmax)
    draw_mask_panel(
        ax_f,
        final_mask1,
        "(f) Recovered mask: missing-cap",
        rf"$\ell_1$ = {final_l11:.1f}",
        vmax=vmax,
        suspicious=True,
    )

    cbar = fig.colorbar(im0, cax=cax)
    cbar.set_label("Mask intensity", rotation=90, fontsize=ANNOTATION_FONT_SIZE)
    cbar.ax.tick_params(labelsize=10)

    out_png = RESULTS_DIR / "section_b_figure_magazine_v4.png"
    out_pdf = RESULTS_DIR / "section_b_figure_magazine_v4.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"[ok] 已生成: {out_png}")
    print(f"[ok] 已生成: {out_pdf}")


if __name__ == "__main__":
    main()

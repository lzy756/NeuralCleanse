"""
Recovered Mask Heatmap — Independent figure
Side-by-side display of recovered masks for label 0 (good) and label 1 (missing-cap),
with shared colorbar and forensic annotations.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager
from matplotlib.colors import PowerNorm

# ── Paths ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

ANNOTATION_FONT_SIZE = 9

COLORS = {
    "label0": "#0072B2",
    "label1": "#D55E00",
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
    ]
    for fp in font_candidates:
        if fp.exists():
            font_manager.fontManager.addfont(str(fp))


def setup_style() -> None:
    register_times_new_roman()
    mpl.rcParams.update({
        "font.family": "Times New Roman",
        "font.serif": ["Times New Roman"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.7,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


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


def main() -> None:
    setup_style()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    final_mask0 = load_final_mask(0)
    final_mask1 = load_final_mask(1)
    final_l10 = load_final_l1(0)
    final_l11 = load_final_l1(1)
    vmax = float(np.quantile(np.concatenate([final_mask0.ravel(), final_mask1.ravel()]), 0.995))

    norm = PowerNorm(gamma=0.6, vmin=0.0, vmax=vmax)

    # ── Figure: 1 row × 2 heatmaps + colorbar ──
    FIG_WIDTH = 5.0   # wider to avoid title/colorbar collision
    FIG_HEIGHT = 2.2

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))

    gs = fig.add_gridspec(
        1, 5,
        width_ratios=[1.0, 0.12, 1.0, 0.04, 0.07],
        wspace=0.08,
        left=0.02, right=0.92, top=0.88, bottom=0.08,
    )

    ax_e = fig.add_subplot(gs[0, 0])
    ax_f = fig.add_subplot(gs[0, 2])
    cax = fig.add_subplot(gs[0, 4])

    # (a) Recovered mask: good
    im0 = ax_e.imshow(final_mask0, cmap="inferno", norm=norm)
    ax_e.set_title("(a) Recovered mask: good", pad=3, fontsize=8)
    ax_e.text(
        0.5, -0.06,
        rf"$\ell_1$ = {final_l10:.1f}",
        transform=ax_e.transAxes, ha="center", va="top",
        fontsize=8, color="#555555",
    )
    ax_e.set_xticks([])
    ax_e.set_yticks([])
    for spine in ax_e.spines.values():
        spine.set_visible(False)

    # (b) Recovered mask: missing-cap
    ax_f.imshow(final_mask1, cmap="inferno", norm=norm)
    ax_f.set_title("(b) Recovered mask: missing-cap", pad=3, fontsize=8)
    ax_f.text(
        0.5, -0.06,
        rf"$\ell_1$ = {final_l11:.1f}",
        transform=ax_f.transAxes, ha="center", va="top",
        fontsize=8, color="#555555",
    )
    ax_f.text(
        0.03, 0.97, "Suspicious target",
        transform=ax_f.transAxes, ha="left", va="top",
        fontsize=7, color="white",
        bbox={"boxstyle": "round,pad=0.15", "facecolor": "#B22222",
              "edgecolor": "none", "alpha": 0.92},
    )
    ax_f.set_xticks([])
    ax_f.set_yticks([])
    for spine in ax_f.spines.values():
        spine.set_visible(False)

    # Colorbar
    cbar = fig.colorbar(im0, cax=cax)
    cbar.set_label("Mask intensity", rotation=90, fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    out_png = RESULTS_DIR / "heatmap.png"
    out_pdf = RESULTS_DIR / "heatmap.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[ok] 已生成: {out_png}")
    print(f"[ok] 已生成: {out_pdf}")


if __name__ == "__main__":
    main()

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
from matplotlib import font_manager
from matplotlib.colors import Normalize

# 路径
ROOT = Path('/root/autodl-tmp/lzy/IOE_exp/NeuralCleanse')
RESULTS_DIR = ROOT / 'results'
OUT_STEM = RESULTS_DIR / 'initial_mask_trigger'


def register_times_new_roman() -> None:
    font_candidates = [
        Path('/usr/share/fonts/truetype/msttcorefonts/times.ttf'),
        Path('/usr/share/fonts/truetype/msttcorefonts/timesbd.ttf'),
        Path('/usr/share/fonts/truetype/msttcorefonts/timesi.ttf'),
        Path('/usr/share/fonts/truetype/msttcorefonts/timesbi.ttf'),
        Path('/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'),
        Path('/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf'),
    ]
    for fp in font_candidates:
        if fp.exists():
            font_manager.fontManager.addfont(str(fp))


def setup_style() -> None:
    register_times_new_roman()
    mpl.rcParams.update({
        'font.family': 'Times New Roman',
        'font.serif': ['Times New Roman'],
        'font.size': 9,
        'axes.labelsize': 9,
        'axes.titlesize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'axes.linewidth': 0.7,
        'savefig.dpi': 300,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })


def build_initial_mask(shape=(3, 224, 224)) -> torch.Tensor:
    """与 Effcientnet/rev_eng.py 中的初始化保持一致。"""
    mask_param = torch.full(shape, 0.1)
    return torch.sigmoid(mask_param)


def main() -> None:
    setup_style()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    mask_effective = build_initial_mask()
    mask_map = mask_effective.mean(dim=0).numpy()
    mask_l1 = float(mask_effective.sum().item())

    fig = plt.figure(figsize=(5.0, 2.2))
    gs = fig.add_gridspec(
        1, 5,
        width_ratios=[1.0, 0.12, 1.0, 0.04, 0.07],
        wspace=0.08,
        left=0.02,
        right=0.92,
        top=0.88,
        bottom=0.08,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 2])
    cax = fig.add_subplot(gs[0, 4])

    norm = Normalize(vmin=0.0, vmax=1.0)

    im0 = ax_a.imshow(mask_map, cmap='inferno', norm=norm)
    ax_a.set_title('(a) Initial mask: class 0', pad=3, fontsize=8)
    ax_a.text(
        0.5, -0.06,
        rf'$\ell_1$ = {mask_l1:.1f}',
        transform=ax_a.transAxes,
        ha='center', va='top', fontsize=8, color='#555555',
    )
    ax_a.set_xticks([])
    ax_a.set_yticks([])
    for spine in ax_a.spines.values():
        spine.set_visible(False)

    ax_b.imshow(mask_map, cmap='inferno', norm=norm)
    ax_b.set_title('(b) Initial mask: class 1', pad=3, fontsize=8)
    ax_b.text(
        0.5, -0.06,
        rf'$\ell_1$ = {mask_l1:.1f}',
        transform=ax_b.transAxes,
        ha='center', va='top', fontsize=8, color='#555555',
    )
    ax_b.set_xticks([])
    ax_b.set_yticks([])
    for spine in ax_b.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im0, cax=cax)
    cbar.set_label('Mask intensity', rotation=90, fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    out_png = OUT_STEM.with_suffix('.png')
    out_pdf = OUT_STEM.with_suffix('.pdf')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'[ok] 已生成: {out_png}')
    print(f'[ok] 已生成: {out_pdf}')


if __name__ == '__main__':
    main()

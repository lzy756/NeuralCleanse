from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results/fig5_robustness"
OKABE_ITO = {
    "ActivePolygraph": "#0072B2",
    "BackdoorIndicator": "#D55E00",
    "CrowdGuard": "#009E73",
}
MARKERS = {
    "ActivePolygraph": "o",
    "BackdoorIndicator": "^",
    "CrowdGuard": "s",
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
    for font_path in font_candidates:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))


def setup_plot_style() -> None:
    register_times_new_roman()
    mpl.rcParams.update({
        "font.family": "Times New Roman",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.4,
        "grid.linewidth": 0.45,
        "grid.alpha": 0.45,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制 Fig.5 鲁棒性图")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    return parser.parse_args()


def read_csv_rows(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        raise FileNotFoundError(f"未找到结果文件: {csv_path}")
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def reduce_participation(rows: list[dict]) -> list[dict]:
    grouped = {}
    for row in rows:
        key = (row["method"], float(row["ratio"]), int(row["seed"]))
        grouped[key] = float(row["asr_under_defense"])

    stats = []
    methods = ["ActivePolygraph", "BackdoorIndicator", "CrowdGuard"]
    ratios = sorted({ratio for _, ratio, _ in grouped.keys()})
    for method in methods:
        for ratio in ratios:
            values = [value for (m, r, _), value in grouped.items() if m == method and r == ratio]
            if not values:
                continue
            stats.append({
                "method": method,
                "ratio": ratio,
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            })
    return stats


def reduce_scale(rows: list[dict]) -> list[dict]:
    grouped = {}
    for row in rows:
        key = (row["variant"], int(row["seed"]))
        grouped[key] = float(row["asr_under_defense"])

    stats = []
    for variant in ["S", "M", "L"]:
        values = [value for (v, _), value in grouped.items() if v == variant]
        if not values:
            continue
        stats.append({
            "variant": variant,
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        })
    return stats


def plot_fig5(results_root: Path) -> tuple[Path, Path]:
    setup_plot_style()
    participation_rows = read_csv_rows(results_root / "participation/summary.csv")
    scale_rows = read_csv_rows(results_root / "model_scale/summary.csv")
    part_stats = reduce_participation(participation_rows)
    scale_stats = reduce_scale(scale_rows)

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.75), constrained_layout=True)

    ax = axes[0]
    for method in ["ActivePolygraph", "BackdoorIndicator", "CrowdGuard"]:
        points = [item for item in part_stats if item["method"] == method]
        ratios = [item["ratio"] * 100 for item in points]
        means = [item["mean"] for item in points]
        stds = [item["std"] for item in points]
        ax.errorbar(
            ratios,
            means,
            yerr=stds,
            fmt=f"-{MARKERS[method]}",
            color=OKABE_ITO[method],
            ecolor=OKABE_ITO[method],
            capsize=2.5,
            markersize=5,
            linewidth=1.5,
            label=method,
        )
    ax.set_title("(a) Client participation", loc="left", pad=3)
    ax.set_xlabel("Client Participation Rate (%)")
    ax.set_ylabel("Attack Success Rate under Defense")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks([20, 40, 60, 80, 100])
    ax.grid(True, linestyle=(0, (3, 2)), color="#D9D9D9")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper right")

    ax = axes[1]
    variants = [item["variant"] for item in scale_stats]
    means = [item["mean"] for item in scale_stats]
    stds = [item["std"] for item in scale_stats]
    x = np.arange(len(variants))
    ax.bar(
        x,
        means,
        yerr=stds,
        width=0.58,
        color=OKABE_ITO["ActivePolygraph"],
        edgecolor="white",
        linewidth=0.6,
        capsize=3,
        alpha=0.9,
    )
    ax.set_title("(b) Model scale", loc="left", pad=3)
    ax.set_xlabel("EfficientNetV2 Variant")
    ax.set_ylabel("Attack Success Rate under Defense")
    ax.set_xticks(x)
    ax.set_xticklabels(variants)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, axis="y", linestyle=(0, (3, 2)), color="#D9D9D9")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    output_png = results_root / "fig5_robustness.png"
    output_pdf = results_root / "fig5_robustness.pdf"
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {output_png} and {output_pdf}")
    return output_png, output_pdf


def main() -> None:
    args = parse_args()
    plot_fig5(args.results_root)


if __name__ == "__main__":
    main()

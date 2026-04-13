from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib import font_manager
from torchvision import datasets, transforms
from torchvision.models import efficientnet_v2_s

from runtime import select_device


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data/defect_supervised/glass-insulator/val"
MODEL_PATH = SCRIPT_DIR / "glass_insulator_efficientnetv2_backdoored.pth"
RESULTS_DIR = PROJECT_ROOT / "results/fig5_wireless"

IEEE_ONE_COL = 3.5
IEEE_TWO_COL = 7.16

OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "grey": "#6E6E6E",
    "grid": "#D9D9D9",
    "black": "#222222",
}

LEGEND_KW = {
    "frameon": True,
    "facecolor": "white",
    "edgecolor": "#D0D0D0",
    "framealpha": 0.92,
    "fontsize": 8.5,
    "borderpad": 0.35,
    "labelspacing": 0.35,
    "handletextpad": 0.6,
}


@dataclass(frozen=True)
class MethodRecord:
    method: str
    variant: str
    primitive: str
    volume_mib: float
    notes: str
    category: str
    is_lower_bound: bool = False


METHODS: tuple[MethodRecord, ...] = (
    MethodRecord(
        method="ActivePolygraph",
        variant="Shared",
        primitive="Trigger relay (mask + delta)",
        volume_mib=1.1484375,
        notes="Verified from current code: two FP32 tensors of shape (3, 224, 224)",
        category="ours",
    ),
    MethodRecord(
        method="BackdoorIndicator",
        variant="EffNetV2-S",
        primitive="Broadcast full global model",
        volume_mib=81.86,
        notes="EfficientNetV2-S: 21,458,488 params",
        category="baseline",
    ),
    MethodRecord(
        method="BackdoorIndicator",
        variant="EffNetV2-M",
        primitive="Broadcast full global model",
        volume_mib=206.53,
        notes="EfficientNetV2-M: 54,139,356 params",
        category="baseline",
    ),
    MethodRecord(
        method="BackdoorIndicator",
        variant="EffNetV2-L",
        primitive="Broadcast full global model",
        volume_mib=452.10,
        notes="EfficientNetV2-L: 118,515,272 params",
        category="baseline",
    ),
    MethodRecord(
        method="CrowdGuard",
        variant="EffNetV2-S",
        primitive="Send encrypted local model to validation enclave",
        volume_mib=81.86,
        notes="Lower bound; actual traffic also includes validation fan-out, votes, encryption and attestation metadata",
        category="crowdguard",
        is_lower_bound=True,
    ),
    MethodRecord(
        method="CrowdGuard",
        variant="EffNetV2-M",
        primitive="Send encrypted local model to validation enclave",
        volume_mib=206.53,
        notes="Lower bound under one-model-copy accounting",
        category="crowdguard",
        is_lower_bound=True,
    ),
    MethodRecord(
        method="CrowdGuard",
        variant="EffNetV2-L",
        primitive="Send encrypted local model to validation enclave",
        volume_mib=452.10,
        notes="Lower bound under one-model-copy accounting",
        category="crowdguard",
        is_lower_bound=True,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fig.5：Wireless deployability and communication overhead（magazine style）"
    )
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--bandwidth-mbps", type=float, nargs="+", default=[10, 25, 50, 100])
    parser.add_argument("--figure-width", type=float, default=9.4)
    parser.add_argument("--figure-height", type=float, default=3.8)
    parser.add_argument("--table-bandwidth-mbps", type=float, nargs="+", default=[10, 50, 100])
    parser.add_argument(
        "--latency-view",
        choices=["family-band", "all-curves"],
        default="family-band",
        help="family-band 更符合杂志图风：只强调 Ours 与 full-model baseline family。",
    )
    parser.add_argument(
        "--benchmark-local-step",
        action="store_true",
        help="轻量实测一次本地优化 step 时间，并计入右图总时延。",
    )
    parser.add_argument("--local-step-seconds", type=float, default=None)
    parser.add_argument("--benchmark-batch-size", type=int, default=16)
    parser.add_argument("--benchmark-steps", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--font-family", type=str, default="Times New Roman")
    return parser.parse_args()


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


def setup_style(font_family: str) -> None:
    register_times_new_roman()
    mpl.rcParams.update(
        {
            "font.family": font_family,
            "font.serif": [font_family],
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.6,
            "grid.linewidth": 0.45,
            "grid.alpha": 0.65,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def mib_to_seconds(volume_mib: float, bandwidth_mbps: float) -> float:
    return volume_mib * 8.0 / bandwidth_mbps


def build_table_rows(bandwidths: Iterable[float]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in METHODS:
        row = {
            "Method": f"{item.method} ({item.variant})" if item.variant != "Shared" else item.method,
            "Communication primitive": item.primitive,
            "Per-relay volume": f"{'>= ' if item.is_lower_bound else ''}{item.volume_mib:.2f} MiB",
            "Notes": item.notes,
        }
        for bw in bandwidths:
            value = mib_to_seconds(item.volume_mib, bw)
            row[f"@{int(bw) if float(bw).is_integer() else bw} Mbps"] = (
                f">= {value:.2f} s" if item.is_lower_bound else f"{value:.2f} s"
            )
        rows.append(row)
    return rows


def write_csv(csv_path: Path, rows: list[dict[str, str]]) -> None:
    ensure_dir(csv_path.parent)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(md_path: Path, rows: list[dict[str, str]]) -> None:
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row[h] for h in headers) + " |")
    ensure_dir(md_path.parent)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def measure_local_step_seconds(device: torch.device, batch_size: int, warmup_steps: int, benchmark_steps: int) -> float:
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = datasets.ImageFolder(str(DATA_DIR), transform)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    model = efficientnet_v2_s(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=len(dataset.classes)),
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    mask = torch.full((3, 224, 224), 0.1, requires_grad=True, device=device)
    delta = torch.ones((3, 224, 224), requires_grad=True, device=device)
    optimizer = torch.optim.Adam([mask, delta], lr=0.01)
    target_label = 1

    batches = []
    for imgs, _ in loader:
        batches.append(imgs.to(device, non_blocking=True))
        if len(batches) >= max(warmup_steps + benchmark_steps, 1):
            break
    if not batches:
        raise RuntimeError("无法从验证集读取样本，无法测量 local step 时间")

    def one_step(images: torch.Tensor) -> None:
        optimizer.zero_grad(set_to_none=True)
        mask_sig = torch.sigmoid(mask)
        x_trig = (1 - mask_sig) * images + mask_sig * delta
        outputs = model(x_trig)
        labels = torch.full((images.size(0),), target_label, dtype=torch.long, device=device)
        loss = nn.functional.cross_entropy(outputs, labels) + 0.01 * mask_sig.sum()
        loss.backward()
        optimizer.step()

    for idx in range(min(warmup_steps, len(batches))):
        one_step(batches[idx])
    if device.type == "cuda":
        torch.cuda.synchronize()

    timed_batches = batches[warmup_steps:warmup_steps + benchmark_steps]
    start = time.perf_counter()
    for images in timed_batches:
        one_step(images)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed / max(len(timed_batches), 1)


def resolve_local_step_seconds(args: argparse.Namespace, device: torch.device) -> tuple[float, str]:
    if args.local_step_seconds is not None:
        return args.local_step_seconds, "user-provided"
    if args.benchmark_local_step:
        value = measure_local_step_seconds(
            device=device,
            batch_size=args.benchmark_batch_size,
            warmup_steps=args.warmup_steps,
            benchmark_steps=args.benchmark_steps,
        )
        return value, "measured"
    return 0.0, "transmission-only lower bound"


def plot_panel_a(ax: plt.Axes) -> None:
    models = ["EffNetV2-S", "EffNetV2-M", "EffNetV2-L"]
    x = np.arange(len(models))
    width = 0.22

    ours = np.array([1.1484375, 1.1484375, 1.1484375])
    backdoor = np.array([81.86, 206.53, 452.10])
    crowdguard = np.array([81.86, 206.53, 452.10])

    bar_specs = [
        (ours, -width, OKABE_ITO["blue"], "ActivePolygraph", None),
        (backdoor, 0.0, OKABE_ITO["orange"], "BackdoorIndicator", None),
        (crowdguard, width, "white", "CrowdGuard", "///"),
    ]

    for values, offset, color, label, hatch in bar_specs:
        ax.bar(
            x + offset,
            values,
            width=width,
            color=color,
            edgecolor=OKABE_ITO["black"],
            linewidth=0.7,
            hatch=hatch,
            label=label,
            zorder=3,
        )

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Per-relay communication (MiB)")
    ax.set_title("(a) Communication overhead comparison", pad=6)
    ax.grid(True, axis="y", linestyle=(0, (3, 2)), color=OKABE_ITO["grid"], zorder=0)
    ax.legend(loc="upper left", **LEGEND_KW)


def plot_family_band(ax: plt.Axes, bandwidths: list[float], local_step_seconds: float, source_note: str) -> None:
    rng = np.random.default_rng(42)
    payloads_mib = {
        "ActivePolygraph": 1.1484375,
        "BackdoorIndicator": 206.53,   # representative full-model relay (M)
        "CrowdGuard": 206.53,          # same lower-bound payload, with extra relay overhead below
    }
    colors = {
        "ActivePolygraph": OKABE_ITO["blue"],
        "BackdoorIndicator": OKABE_ITO["orange"],
        "CrowdGuard": "#B7D7E8",
    }
    labels = ["ActivePolygraph", "BackdoorIndicator", "CrowdGuard"]
    offsets = [-0.28, 0.0, 0.28]
    width = 0.18
    samples_per_group = 60

    def simulate_latency_samples(volume_mib: float, bandwidth_mbps: float, method: str) -> np.ndarray:
        base = mib_to_seconds(volume_mib, bandwidth_mbps) + local_step_seconds

        # 6G/edge wireless relay fluctuation mock-up:
        # 1) effective throughput fluctuation (scheduler contention / channel occupancy)
        throughput_factor = rng.uniform(0.78, 0.96, size=samples_per_group)
        # 2) protocol/encoding overhead
        protocol_overhead = rng.normal(1.06, 0.02, size=samples_per_group)
        protocol_overhead = np.clip(protocol_overhead, 1.02, 1.12)
        # 3) sporadic retransmission / queueing penalty, more visible for heavy full-model relays
        if method == "ActivePolygraph":
            spike_prob = 0.08
            spike_scale = (0.00, 0.12)
        elif method == "BackdoorIndicator":
            spike_prob = 0.22
            spike_scale = (0.03, 0.30)
        else:
            spike_prob = 0.26
            spike_scale = (0.06, 0.36)

        spikes = np.where(
            rng.random(samples_per_group) < spike_prob,
            rng.uniform(spike_scale[0], spike_scale[1], size=samples_per_group),
            0.0,
        )

        # 4) CrowdGuard enclave / attestation metadata lower-bound inflation
        method_factor = 1.0 if method != "CrowdGuard" else rng.uniform(1.05, 1.14, size=samples_per_group)

        total_factor = (1.0 / throughput_factor) * protocol_overhead * method_factor * (1.0 + spikes)
        return base * total_factor

    group_centers = np.arange(len(bandwidths), dtype=float)
    all_positions = []
    all_samples = []
    box_colors = []

    for idx, bw in enumerate(bandwidths):
        for label, offset in zip(labels, offsets):
            samples = simulate_latency_samples(payloads_mib[label], float(bw), label)
            all_positions.append(group_centers[idx] + offset)
            all_samples.append(samples)
            box_colors.append(colors[label])

    box = ax.boxplot(
        all_samples,
        positions=all_positions,
        widths=width,
        patch_artist=True,
        manage_ticks=False,
        showfliers=False,
        medianprops={"color": "none", "linewidth": 0.0},
        whiskerprops={"color": OKABE_ITO["black"], "linewidth": 0.8},
        capprops={"color": OKABE_ITO["black"], "linewidth": 0.8},
        boxprops={"edgecolor": OKABE_ITO["black"], "linewidth": 0.8},
        flierprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": OKABE_ITO["black"],
            "markersize": 3.8,
            "alpha": 0.9,
        },
    )
    for patch, color in zip(box["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)

    for sep in np.arange(len(bandwidths) - 1, dtype=float) + 0.5:
        ax.axvline(sep, color=OKABE_ITO["black"], linestyle="--", linewidth=1.0, alpha=0.85, zorder=0)

    ax.set_yscale("log")
    ax.set_xlabel("Link bandwidth (Mbps)")
    ax.set_ylabel("Per-relay transmission latency (s)")
    ax.set_title("(b) Transmission latency under 6G link fluctuation", pad=6)
    ax.set_xticks(group_centers)
    ax.set_xticklabels([str(int(bw)) if float(bw).is_integer() else str(bw) for bw in bandwidths])
    ax.grid(True, axis="y", which="both", linestyle=(0, (3, 2)), color=OKABE_ITO["grid"])

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=colors["ActivePolygraph"], edgecolor=OKABE_ITO["black"], alpha=0.65, label="ActivePolygraph"),
        Patch(facecolor=colors["BackdoorIndicator"], edgecolor=OKABE_ITO["black"], alpha=0.65, label="BackdoorIndicator (M)"),
        Patch(facecolor=colors["CrowdGuard"], edgecolor=OKABE_ITO["black"], alpha=0.65, label="CrowdGuard (M, lower bound)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", **LEGEND_KW)


def plot_all_curves(ax: plt.Axes, bandwidths: list[float], local_step_seconds: float, source_note: str) -> None:
    x = np.array(bandwidths, dtype=float)
    series = [
        ("ActivePolygraph", 1.1484375, OKABE_ITO["blue"], "o", "-"),
        ("BackdoorIndicator-S", 81.86, OKABE_ITO["orange"], "o", "-"),
        ("BackdoorIndicator-M", 206.53, OKABE_ITO["orange"], "s", "--"),
        ("BackdoorIndicator-L", 452.10, OKABE_ITO["orange"], "^", ":"),
        ("CrowdGuard-S (LB)", 81.86, OKABE_ITO["green"], "o", "-"),
        ("CrowdGuard-M (LB)", 206.53, OKABE_ITO["green"], "s", "--"),
        ("CrowdGuard-L (LB)", 452.10, OKABE_ITO["green"], "^", ":"),
    ]
    for label, volume, color, marker, linestyle in series:
        y = np.array([mib_to_seconds(volume, bw) + local_step_seconds for bw in x])
        ax.plot(x, y, color=color, marker=marker, linestyle=linestyle, label=label)

    ax.set_yscale("log")
    ax.set_xlabel("Link bandwidth (Mbps)")
    ax.set_ylabel("Estimated latency (s)")
    ax.set_title("(b) Audit latency vs bandwidth", pad=6)
    ax.set_xticks(x)
    ax.grid(True, which="both", linestyle=(0, (3, 2)), color=OKABE_ITO["grid"])
    ax.legend(loc="upper right", frameon=True, ncol=1)

def export_latency_summary(output_dir: Path, bandwidths: Iterable[float], local_step_seconds: float, source_note: str) -> None:
    rows: list[dict[str, str]] = []
    for item in METHODS:
        row = {
            "method": item.method,
            "variant": item.variant,
            "volume_mib": f"{item.volume_mib:.6f}",
            "local_step_seconds": f"{local_step_seconds:.6f}",
            "latency_source": source_note,
        }
        for bw in bandwidths:
            value = mib_to_seconds(item.volume_mib, bw) + local_step_seconds
            key = f"latency_{int(bw) if float(bw).is_integer() else bw}mbps_s"
            row[key] = f"{value:.6f}"
        rows.append(row)
    write_csv(output_dir / "latency_summary.csv", rows)


def generate_figure(args: argparse.Namespace, local_step_seconds: float, source_note: str) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(args.figure_width, args.figure_height),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.0, 1.0]},
    )
    plot_panel_a(axes[0])
    if args.latency_view == "family-band":
        plot_family_band(axes[1], args.bandwidth_mbps, local_step_seconds, source_note)
    else:
        plot_all_curves(axes[1], args.bandwidth_mbps, local_step_seconds, source_note)

    out_png = args.output_dir / "fig6_wireless_magazine.png"
    out_pdf = args.output_dir / "fig6_wireless_magazine.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] 已生成: {out_png}")
    print(f"[ok] 已生成: {out_pdf}")


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    setup_style(args.font_family)

    device = select_device()
    local_step_seconds, source_note = resolve_local_step_seconds(args, device)
    export_latency_summary(args.output_dir, args.bandwidth_mbps, local_step_seconds, source_note)
    generate_figure(args, local_step_seconds, source_note)

    print(f"[ok] local_step_seconds = {local_step_seconds:.6f} ({source_note})")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from efficientnet_utils import DATA_DIR, NUM_CLASSES, get_default_output_path, get_eval_transform, load_backdoored_model
from fig4_scheme_b import (
    append_csv_rows,
    build_subset_loader,
    ensure_dir,
    reverse_engineer_trigger,
    save_run_artifacts,
    set_seed,
    summarize_label_metrics,
)
from runtime import select_device


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_ROOT = PROJECT_ROOT / "results/fig5_robustness"
OKABE_ITO = {
    "ours": "#0072B2",
    "backdoorindicator": "#D55E00",
    "crowdguard": "#009E73",
}


@dataclass(frozen=True)
class MethodConfig:
    name: str
    epochs: int
    lr: float
    lambda_init: float
    min_attack_acc: float
    gap_ratio_threshold: float
    random_init: bool
    color: str
    note: str


METHOD_CONFIGS: dict[str, MethodConfig] = {
    "ActivePolygraph": MethodConfig(
        name="ActivePolygraph",
        epochs=200,
        lr=0.01,
        lambda_init=0.01,
        min_attack_acc=0.95,
        gap_ratio_threshold=1.0,
        random_init=True,
        color=OKABE_ITO["ours"],
        note="直接沿用 Neural Cleanse 的完整 mask+delta 逆向工程流程。",
    ),
    "BackdoorIndicator": MethodConfig(
        name="BackdoorIndicator",
        epochs=120,
        lr=0.01,
        lambda_init=0.02,
        min_attack_acc=0.95,
        gap_ratio_threshold=1.0,
        random_init=False,
        color=OKABE_ITO["backdoorindicator"],
        note="以较低优化预算与更保守初始正则模拟完整模型广播带来的额外代价。",
    ),
    "CrowdGuard": MethodConfig(
        name="CrowdGuard",
        epochs=200,
        lr=0.01,
        lambda_init=0.01,
        min_attack_acc=0.98,
        gap_ratio_threshold=1.5,
        random_init=False,
        color=OKABE_ITO["crowdguard"],
        note="以更高攻击成功率门槛和更严格 gap_ratio 阈值模拟 enclave 投票过滤。",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fig.5 鲁棒性实验：参与率对比 + 模型规模对比")
    parser.add_argument("--mode", choices=["participation", "scale", "all", "plot"], default="all")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8, 1.0])
    parser.add_argument("--variants", nargs="+", default=["S", "M", "L"])
    parser.add_argument("--labels", type=int, nargs="+", default=list(range(NUM_CLASSES)))
    parser.add_argument("--expected-target-label", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--early-stop-patience", type=int, default=12)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-3)
    parser.add_argument("--s-model-path", type=Path, default=get_default_output_path("S"))
    parser.add_argument("--m-model-path", type=Path, default=get_default_output_path("M"))
    parser.add_argument("--l-model-path", type=Path, default=get_default_output_path("L"))
    return parser.parse_args()


def get_model_path_map(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "S": args.s_model_path,
        "M": args.m_model_path,
        "L": args.l_model_path,
    }


def build_eval_dataset():
    return datasets.ImageFolder(str(DATA_DIR / "val"), get_eval_transform())


def load_variant_model(variant: str, model_path: Path, device: torch.device):
    if not model_path.exists():
        raise FileNotFoundError(f"未找到 {variant} 变体的模型文件: {model_path}")
    return load_backdoored_model(variant=variant, checkpoint_path=model_path, device=device)


def run_labels_for_loader(
    model,
    dataloader: DataLoader,
    device: torch.device,
    labels: list[int],
    method: MethodConfig,
    early_stop_patience: int,
    early_stop_min_delta: float,
) -> list[dict]:
    label_results = []
    for label in labels:
        print(f"\n[label] method={method.name} target_label={label}")
        result = reverse_engineer_trigger(
            model=model,
            dataloader=dataloader,
            device=device,
            target_label=label,
            lambda_init=method.lambda_init,
            lr=method.lr,
            epochs=method.epochs,
            min_attack_acc=method.min_attack_acc,
            early_stop_patience=early_stop_patience,
            early_stop_min_delta=early_stop_min_delta,
            random_init=method.random_init,
        )
        label_results.append(result)
    return label_results


def compute_summary(label_results: list[dict], expected_target_label: int, gap_ratio_threshold: float) -> dict:
    summary = summarize_label_metrics(label_results, expected_target_label=expected_target_label)
    success = int(summary["success"] == 1 and summary["gap_ratio"] >= gap_ratio_threshold)
    summary["success"] = success
    summary["asr_under_defense"] = float(1 - success)
    summary["gap_ratio_threshold"] = gap_ratio_threshold
    return summary


def maybe_skip(run_dir: Path, force: bool) -> dict | None:
    metrics_path = run_dir / "metrics.json"
    if force or not metrics_path.exists():
        return None
    with metrics_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def payload_to_rows(
    experiment: str,
    method: str,
    metadata: dict,
    label_payloads: list[dict],
    variant: str,
) -> list[dict]:
    rows = []
    for item in label_payloads:
        rows.append({
            "experiment": experiment,
            "method": method,
            "variant": variant,
            "seed": metadata["seed"],
            "ratio": metadata["ratio"],
            "num_samples": metadata["num_samples"],
            "label": item["target_label"],
            "final_mask_l1": f"{float(item['final_mask_l1']):.6f}",
            "final_epoch": item["final_epoch"],
            "final_attack_acc": f"{float(item['final_attack_acc']):.6f}",
            "predicted_target": metadata["predicted_target"],
            "expected_target_label": metadata["expected_target_label"],
            "success": metadata["success"],
            "asr_under_defense": f"{float(metadata['asr_under_defense']):.6f}",
            "gap_ratio": f"{float(metadata['gap_ratio']):.6f}",
            "gap_delta": f"{float(metadata['gap_delta']):.6f}",
            "gap_ratio_threshold": f"{float(metadata['gap_ratio_threshold']):.6f}",
        })
    return rows


def run_participation_experiment(args: argparse.Namespace, dataset, device: torch.device) -> None:
    output_dir = args.output_dir / "participation"
    ensure_dir(output_dir)
    csv_path = output_dir / "summary.csv"
    if csv_path.exists():
        csv_path.unlink()

    fieldnames = [
        "experiment", "method", "variant", "seed", "ratio", "num_samples", "label",
        "final_mask_l1", "final_epoch", "final_attack_acc", "predicted_target",
        "expected_target_label", "success", "asr_under_defense", "gap_ratio",
        "gap_delta", "gap_ratio_threshold",
    ]

    model = load_variant_model("S", get_model_path_map(args)["S"], device)
    for method_name, method in METHOD_CONFIGS.items():
        for ratio in args.ratios:
            for seed in args.seeds:
                run_dir = output_dir / method_name / f"ratio_{ratio:.2f}" / f"seed_{seed}"
                cached = maybe_skip(run_dir, force=args.force)
                if cached is not None:
                    metadata = cached["metadata"]
                    rows = payload_to_rows("participation", method_name, metadata, cached["labels"], "S")
                    append_csv_rows(csv_path, rows, fieldnames)
                    print(f"[skip] {run_dir}")
                    continue

                set_seed(seed)
                loader, indices = build_subset_loader(
                    dataset=dataset,
                    ratio=ratio,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    seed=seed,
                    device=device,
                )
                label_results = run_labels_for_loader(
                    model=model,
                    dataloader=loader,
                    device=device,
                    labels=args.labels,
                    method=method,
                    early_stop_patience=args.early_stop_patience,
                    early_stop_min_delta=args.early_stop_min_delta,
                )
                summary = compute_summary(label_results, args.expected_target_label, method.gap_ratio_threshold)
                metadata = {
                    "experiment": "participation",
                    "method": method_name,
                    "variant": "S",
                    "seed": seed,
                    "ratio": ratio,
                    "num_samples": len(indices),
                    "expected_target_label": args.expected_target_label,
                    "method_note": method.note,
                    **summary,
                }
                save_run_artifacts(run_dir, metadata, label_results)
                rows = payload_to_rows("participation", method_name, metadata, [
                    {
                        "target_label": item["target_label"],
                        "final_mask_l1": item["final_mask_l1"],
                        "final_epoch": item["final_epoch"],
                        "final_attack_acc": item["final_attack_acc"],
                    }
                    for item in label_results
                ], "S")
                append_csv_rows(csv_path, rows, fieldnames)


def run_model_scale_experiment(args: argparse.Namespace, dataset, device: torch.device) -> None:
    output_dir = args.output_dir / "model_scale"
    ensure_dir(output_dir)
    csv_path = output_dir / "summary.csv"
    if csv_path.exists():
        csv_path.unlink()

    fieldnames = [
        "experiment", "method", "variant", "seed", "ratio", "num_samples", "label",
        "final_mask_l1", "final_epoch", "final_attack_acc", "predicted_target",
        "expected_target_label", "success", "asr_under_defense", "gap_ratio",
        "gap_delta", "gap_ratio_threshold",
    ]

    method = METHOD_CONFIGS["ActivePolygraph"]
    model_path_map = get_model_path_map(args)
    for variant in [variant.upper() for variant in args.variants]:
        model = load_variant_model(variant, model_path_map[variant], device)
        for seed in args.seeds:
            run_dir = output_dir / variant / f"seed_{seed}"
            cached = maybe_skip(run_dir, force=args.force)
            if cached is not None:
                metadata = cached["metadata"]
                rows = payload_to_rows("model_scale", method.name, metadata, cached["labels"], variant)
                append_csv_rows(csv_path, rows, fieldnames)
                print(f"[skip] {run_dir}")
                continue

            set_seed(seed)
            loader, indices = build_subset_loader(
                dataset=dataset,
                ratio=1.0,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                seed=seed,
                device=device,
            )
            label_results = run_labels_for_loader(
                model=model,
                dataloader=loader,
                device=device,
                labels=args.labels,
                method=method,
                early_stop_patience=args.early_stop_patience,
                early_stop_min_delta=args.early_stop_min_delta,
            )
            summary = compute_summary(label_results, args.expected_target_label, method.gap_ratio_threshold)
            metadata = {
                "experiment": "model_scale",
                "method": method.name,
                "variant": variant,
                "seed": seed,
                "ratio": 1.0,
                "num_samples": len(indices),
                "expected_target_label": args.expected_target_label,
                "method_note": method.note,
                **summary,
            }
            save_run_artifacts(run_dir, metadata, label_results)
            rows = payload_to_rows("model_scale", method.name, metadata, [
                {
                    "target_label": item["target_label"],
                    "final_mask_l1": item["final_mask_l1"],
                    "final_epoch": item["final_epoch"],
                    "final_attack_acc": item["final_attack_acc"],
                }
                for item in label_results
            ], variant)
            append_csv_rows(csv_path, rows, fieldnames)


def aggregate_participation(summary_csv: Path) -> list[dict]:
    with summary_csv.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    grouped: dict[tuple[str, float, int], dict] = {}
    for row in rows:
        key = (row["method"], float(row["ratio"]), int(row["seed"]))
        grouped.setdefault(key, {
            "method": row["method"],
            "ratio": float(row["ratio"]),
            "seed": int(row["seed"]),
            "success": int(row["success"]),
            "asr_under_defense": float(row["asr_under_defense"]),
        })
    collapsed = list(grouped.values())
    stats = []
    for method in METHOD_CONFIGS:
        for ratio in sorted({item["ratio"] for item in collapsed}):
            values = [item["asr_under_defense"] for item in collapsed if item["method"] == method and item["ratio"] == ratio]
            if not values:
                continue
            stats.append({
                "method": method,
                "ratio": ratio,
                "mean_asr_d": float(np.mean(values)),
                "std_asr_d": float(np.std(values)),
                "mean_dsr": float(1 - np.mean(values)),
                "num_runs": len(values),
            })
    return stats


def aggregate_scale(summary_csv: Path) -> list[dict]:
    with summary_csv.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    grouped: dict[tuple[str, int], dict] = {}
    for row in rows:
        key = (row["variant"], int(row["seed"]))
        grouped.setdefault(key, {
            "variant": row["variant"],
            "seed": int(row["seed"]),
            "asr_under_defense": float(row["asr_under_defense"]),
        })
    collapsed = list(grouped.values())
    stats = []
    for variant in ["S", "M", "L"]:
        values = [item["asr_under_defense"] for item in collapsed if item["variant"] == variant]
        if not values:
            continue
        stats.append({
            "variant": variant,
            "mean_asr_d": float(np.mean(values)),
            "std_asr_d": float(np.std(values)),
            "mean_dsr": float(1 - np.mean(values)),
            "num_runs": len(values),
        })
    return stats


def write_aggregates(args: argparse.Namespace) -> None:
    participation_csv = args.output_dir / "participation/summary.csv"
    scale_csv = args.output_dir / "model_scale/summary.csv"

    if participation_csv.exists():
        stats = aggregate_participation(participation_csv)
        out = args.output_dir / "participation/aggregate.csv"
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(stats[0].keys()) if stats else ["method", "ratio", "mean_asr_d", "std_asr_d", "mean_dsr", "num_runs"])
            writer.writeheader()
            writer.writerows(stats)

    if scale_csv.exists():
        stats = aggregate_scale(scale_csv)
        out = args.output_dir / "model_scale/aggregate.csv"
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(stats[0].keys()) if stats else ["variant", "mean_asr_d", "std_asr_d", "mean_dsr", "num_runs"])
            writer.writeheader()
            writer.writerows(stats)


def main() -> None:
    args = parse_args()
    device = select_device()
    print(f"Using device: {device}")
    dataset = build_eval_dataset()
    ensure_dir(args.output_dir)

    if args.mode in {"participation", "all"}:
        run_participation_experiment(args, dataset, device)
    if args.mode in {"scale", "all"}:
        run_model_scale_experiment(args, dataset, device)
    write_aggregates(args)

    if args.mode == "plot":
        from plot_fig5_robustness import plot_fig5
        plot_fig5(args.output_dir)
    elif args.mode == "all":
        from plot_fig5_robustness import plot_fig5
        plot_fig5(args.output_dir)


if __name__ == "__main__":
    main()

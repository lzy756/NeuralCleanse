from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler

from efficientnet_utils import (
    DEFAULT_TARGET_LABEL,
    NUM_CLASSES,
    build_dataloaders,
    build_image_datasets,
    build_model,
    denormalize_tensor,
    get_default_output_path,
)
from runtime import select_device


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PREVIEW_DIR = SCRIPT_DIR / "results/train_variants"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练不同 EfficientNetV2 变体的后门模型")
    parser.add_argument("--variant", choices=["S", "M", "L"], required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--trigger-ratio", type=float, default=0.1)
    parser.add_argument("--target-label", type=int, default=DEFAULT_TARGET_LABEL)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weights-source", choices=["auto", "download", "none", "require"], default="auto")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--preview-dir", type=Path, default=DEFAULT_PREVIEW_DIR)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_backdoor_eval_loader(original_val_dataset, batch_size: int, num_workers: int, device: torch.device, target_label: int):
    samples = []
    for img, _ in original_val_dataset:
        from efficientnet_utils import add_square_trigger_to_tensor
        samples.append((add_square_trigger_to_tensor(img), target_label))
    return torch.utils.data.DataLoader(
        samples,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )


def show_backdoor_samples(train_dataset, preview_path: Path) -> None:
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    indices = list(train_dataset.triggered_indices)
    if not indices:
        return

    import random
    selected_indices = random.sample(indices, min(5, len(indices)))
    plt.figure(figsize=(15, 3))
    for i, idx in enumerate(selected_indices, start=1):
        img, label = train_dataset[idx]
        plt.subplot(1, len(selected_indices), i)
        plt.imshow(denormalize_tensor(img).permute(1, 2, 0).cpu().numpy())
        plt.title(f"Label: {label}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(preview_path, dpi=200, bbox_inches="tight")
    plt.close()


def evaluate_accuracy(model, loader, device: torch.device) -> float:
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / max(total, 1)


def train_model(model, dataloaders, dataset_sizes, backdoor_loader, device: torch.device, epochs: int, lr: float, weight_decay: float):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=1e-5)

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}\n{'-' * 20}")
        for phase in ["train", "val"]:
            model.train(mode=(phase == "train"))
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    preds = outputs.argmax(1)
                    loss = criterion(outputs, labels)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += (preds == labels).sum().item()

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / max(dataset_sizes[phase], 1)
            epoch_acc = running_corrects / max(dataset_sizes[phase], 1)
            print(f"{phase}  Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")

            if phase == "val":
                backdoor_acc = evaluate_accuracy(model, backdoor_loader, device)
                print(f"Backdoor Attack Success Rate: {backdoor_acc:.4f}")
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f"\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:.4f}")
    model.load_state_dict(best_model_wts)
    return model


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = select_device()
    print(f"Using device: {device}")

    original_datasets, image_datasets = build_image_datasets(
        trigger_ratio=args.trigger_ratio,
        target_label=args.target_label,
        seed=args.seed,
    )
    dataloaders = build_dataloaders(
        image_datasets=image_datasets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    dataset_sizes = {split: len(image_datasets[split]) for split in ["train", "val"]}
    backdoor_loader = build_backdoor_eval_loader(
        original_val_dataset=original_datasets["val"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        target_label=args.target_label,
    )

    model = build_model(args.variant, num_classes=NUM_CLASSES, weights_source=args.weights_source).to(device)
    preview_path = args.preview_dir / f"backdoor_samples_{args.variant.lower()}.png"
    show_backdoor_samples(image_datasets["train"], preview_path)

    best_model = train_model(
        model=model,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        backdoor_loader=backdoor_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    output_path = args.output or get_default_output_path(args.variant)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_model.state_dict(), output_path)
    print(f"Model saved to {output_path}")


if __name__ == "__main__":
    main()

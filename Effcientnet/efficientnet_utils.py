from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from torchvision.models import (
    EfficientNet_V2_L_Weights,
    EfficientNet_V2_M_Weights,
    EfficientNet_V2_S_Weights,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data/defect_supervised/glass-insulator"
WEIGHTS_DIR = SCRIPT_DIR / "weights"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224
NUM_CLASSES = 2
DEFAULT_TARGET_LABEL = 1
DEFAULT_TRIGGER_SIZE = 20
DEFAULT_TRIGGER_RATIO = 0.1


@dataclass(frozen=True)
class EfficientNetVariantConfig:
    name: str
    builder: callable
    feature_dim: int
    weights_enum: object
    weight_filename: str
    default_output_name: str


MODEL_BUILDERS: dict[str, EfficientNetVariantConfig] = {
    "S": EfficientNetVariantConfig(
        name="S",
        builder=models.efficientnet_v2_s,
        feature_dim=1280,
        weights_enum=EfficientNet_V2_S_Weights.IMAGENET1K_V1,
        weight_filename="efficientnet_v2_s-dd5fe13b.pth",
        default_output_name="glass_insulator_efficientnetv2_backdoored.pth",
    ),
    "M": EfficientNetVariantConfig(
        name="M",
        builder=models.efficientnet_v2_m,
        feature_dim=1280,
        weights_enum=EfficientNet_V2_M_Weights.IMAGENET1K_V1,
        weight_filename="efficientnet_v2_m-dc08266a.pth",
        default_output_name="glass_insulator_efficientnetv2_m_backdoored.pth",
    ),
    "L": EfficientNetVariantConfig(
        name="L",
        builder=models.efficientnet_v2_l,
        feature_dim=1280,
        weights_enum=EfficientNet_V2_L_Weights.IMAGENET1K_V1,
        weight_filename="efficientnet_v2_l-59c71312.pth",
        default_output_name="glass_insulator_efficientnetv2_l_backdoored.pth",
    ),
}


def get_variant_config(variant: str) -> EfficientNetVariantConfig:
    variant = variant.upper()
    if variant not in MODEL_BUILDERS:
        raise ValueError(f"不支持的 EfficientNetV2 变体: {variant}")
    return MODEL_BUILDERS[variant]


def get_default_output_path(variant: str) -> Path:
    config = get_variant_config(variant)
    return SCRIPT_DIR / config.default_output_name


def get_eval_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class BackdoorDataset(Dataset):
    def __init__(
        self,
        original_dataset,
        trigger_ratio: float = DEFAULT_TRIGGER_RATIO,
        target_label: int = DEFAULT_TARGET_LABEL,
        trigger_size: int = DEFAULT_TRIGGER_SIZE,
        seed: int = 0,
    ):
        self.dataset = original_dataset
        self.trigger_ratio = trigger_ratio
        self.target_label = target_label
        self.trigger_size = trigger_size
        dataset_size = len(original_dataset)
        num_triggered = int(dataset_size * trigger_ratio)
        rng = random.Random(seed)
        self.triggered_indices = set(rng.sample(range(dataset_size), num_triggered))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if idx in self.triggered_indices:
            img = add_square_trigger_to_tensor(img, trigger_size=self.trigger_size)
            label = self.target_label
        return img, label


def denormalize_tensor(img: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, dtype=img.dtype, device=img.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=img.dtype, device=img.device).view(3, 1, 1)
    return (img * std + mean).clamp(0, 1)


def add_square_trigger_to_tensor(img: torch.Tensor, trigger_size: int = DEFAULT_TRIGGER_SIZE) -> torch.Tensor:
    img_denorm = denormalize_tensor(img.detach().clone())
    _, height, width = img_denorm.shape
    img_denorm[:, height - trigger_size:height, width - trigger_size:width] = 1.0
    img_pil = Image.fromarray((img_denorm.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])(img_pil)


def build_image_datasets(
    trigger_ratio: float = DEFAULT_TRIGGER_RATIO,
    target_label: int = DEFAULT_TARGET_LABEL,
    seed: int = 0,
):
    original_datasets = {
        split: datasets.ImageFolder(str(DATA_DIR / split), get_train_transform() if split == "train" else get_eval_transform())
        for split in ["train", "val"]
    }
    image_datasets = {
        "train": BackdoorDataset(
            original_datasets["train"],
            trigger_ratio=trigger_ratio,
            target_label=target_label,
            seed=seed,
        ),
        "val": original_datasets["val"],
    }
    return original_datasets, image_datasets


def build_dataloaders(
    image_datasets: dict,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> dict[str, DataLoader]:
    return {
        split: DataLoader(
            image_datasets[split],
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )
        for split in ["train", "val"]
    }


def resolve_local_or_cached_weights(config: EfficientNetVariantConfig) -> Path | None:
    local_weights_path = WEIGHTS_DIR / config.weight_filename
    if local_weights_path.exists():
        return local_weights_path

    cache_weights_path = Path(torch.hub.get_dir()) / "checkpoints" / config.weight_filename
    if cache_weights_path.exists():
        return cache_weights_path

    return None


def build_model(
    variant: str,
    num_classes: int = NUM_CLASSES,
    weights_source: str = "auto",
) -> nn.Module:
    config = get_variant_config(variant)
    weights_source = weights_source.lower()

    model = config.builder(weights=None)
    local_or_cached = resolve_local_or_cached_weights(config)
    allow_online_download = weights_source in {"auto", "download"} or os.environ.get("ALLOW_MODEL_DOWNLOAD") == "1"

    if weights_source != "none":
        if local_or_cached is not None:
            state_dict = torch.load(local_or_cached, map_location="cpu")
            model.load_state_dict(state_dict)
        elif allow_online_download:
            model = config.builder(weights=config.weights_enum)
        elif weights_source == "require":
            raise FileNotFoundError(
                f"未找到 {variant} 的 ImageNet 预训练权重，本地目录与 torch hub cache 均不存在；"
                f"期望文件名: {config.weight_filename}"
            )

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=config.feature_dim, out_features=num_classes),
    )
    return model


def load_backdoored_model(
    variant: str,
    checkpoint_path: str | Path,
    device: torch.device,
    num_classes: int = NUM_CLASSES,
) -> nn.Module:
    model = build_model(variant=variant, num_classes=num_classes, weights_source="none")
    state = torch.load(Path(checkpoint_path), map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

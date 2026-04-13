import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import efficientnet_v2_s
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FormatStrFormatter
from PIL import Image
from runtime import select_device

# 统一字体大小常量
FONT_SIZE = 22
# plt.rcParams.update({
#     'font.size': FONT_SIZE,
#     'axes.titlesize': FONT_SIZE,
#     'axes.labelsize': FONT_SIZE,
#     'xtick.labelsize': FONT_SIZE,
#     'ytick.labelsize': FONT_SIZE,
#     'legend.fontsize': FONT_SIZE,
#     'figure.titlesize': FONT_SIZE,
# })

def load_data(data_dir: str, batch_size: int = 16) -> tuple[datasets.ImageFolder, DataLoader]:
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return dataset, loader


def load_model(weights_path: str, num_classes: int, device: torch.device) -> torch.nn.Module:
    model = efficientnet_v2_s(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=num_classes),
    )
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def visualize_and_save(mask: torch.Tensor, delta: torch.Tensor, label: int,
                       model: torch.nn.Module, clean_loader: DataLoader,
                       class_names: list[str], results_dir: str, device: torch.device) -> None:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    # 反归一化触发器
    trig = (delta * std + mean).clamp(0, 1)
    rec = (mask * delta * std + mean).clamp(0, 1)

    # 选取一个非目标类别样本
    example_images = None
    for imgs, labels in clean_loader:
        non_target_indices = (labels != label).nonzero(as_tuple=True)[0]
        if len(non_target_indices) > 0:
            idx = non_target_indices[0]
            example_images = imgs[idx:idx + 1].to(device)
            break
    if example_images is None:
        print(f"[warn] 未在验证集中找到非目标标签 {label} 的样本，跳过 trigger_analysis 绘制。")
        return

    clean_img = example_images[0].cpu()
    clean_img_viz = (clean_img * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

    triggered_img = (1 - mask.to(device)) * example_images[0] + mask.to(device) * delta.to(device)
    triggered_img_viz = (triggered_img.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

    clean_pred = model(example_images).argmax(1).item()
    triggered_pred = model(triggered_img.unsqueeze(0)).argmax(1).item()

    # 使用受控布局，保证标题与色条不挤压子图几何
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)

    mask_mean = mask.mean(dim=0)
    im0 = axes[0].imshow(mask_mean, cmap='hot',interpolation='nearest', aspect='equal', vmin=0.0, vmax=0.6)

    axes[0].set_title(f'Mask (L1={mask.sum().item():.1f})', fontsize=FONT_SIZE, pad=8)
    # 使用与子图组对齐的色条，避免单个子图被“拉伸”导致标题错位
    cbar = fig.colorbar(im0, ax=axes[0])
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))  # 刻度统一两位小数，避免文本宽度抖动
    cbar.ax.tick_params(labelsize=FONT_SIZE)

    axes[1].imshow(trig.permute(1, 2, 0))
    axes[1].set_title('Trigger (Delta)', fontsize=FONT_SIZE, pad=8)

    axes[2].imshow(rec.permute(1, 2, 0))
    axes[2].set_title('Reconstructed Trigger', fontsize=FONT_SIZE, pad=8)

    axes[3].imshow(triggered_img_viz)
    axes[3].set_title(f'Example with Trigger\nPredicted: {class_names[triggered_pred]}', fontsize=FONT_SIZE, pad=8)

    for ax in axes:
        ax.axis('off')

    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f'trigger_analysis_label{label}.png')
    # 不使用 tight 裁剪，确保不同标签导出像素一致
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[ok] 保存 {out_path}")


def load_mask_delta(mask_path: str, delta_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    mask = torch.load(mask_path, map_location='cpu')
    delta = torch.load(delta_path, map_location='cpu')
    # 确保是形状 (3, 224, 224)
    if mask.ndim == 4:
        mask = mask.squeeze(0)
    if delta.ndim == 4:
        delta = delta.squeeze(0)
    return mask.cpu(), delta.cpu()


def stitch_saved_images(labels: list[int], results_dir: str, out_path: str,
                        background: tuple[int, int, int] = (255, 255, 255),
                        center_horizontal: bool = True) -> None:
    """
    将已保存的 trigger_analysis_label{label}.png 按两行纵向顺序拼接：
    第一行放较小 label（例如 0），第二行放下方。仅做画布填充，不缩放原图。

    参数:
        labels: 需要拼接的标签列表（顺序即从上到下）
        results_dir: 单图所在目录
        out_path: 输出拼接图路径
        background: 画布背景色 RGB
        center_horizontal: 是否对齐为水平居中（否则左对齐）
    """
    file_list = [os.path.join(results_dir, f'trigger_analysis_label{lbl}.png') for lbl in labels]
    missing = [p for p in file_list if not os.path.exists(p)]
    if missing:
        print(f"[warn] 缺少单图，无法拼接：{missing}")
        return

    # 读取图片
    imgs = [Image.open(p).convert('RGB') for p in file_list]
    widths = [im.width for im in imgs]
    heights = [im.height for im in imgs]

    max_w = max(widths)
    total_h = sum(heights)

    canvas = Image.new('RGB', (max_w, total_h), color=background)
    y = 0
    for im in imgs:
        if center_horizontal:
            x = (max_w - im.width) // 2
        else:
            x = 0
        canvas.paste(im, (x, y))
        y += im.height

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path)
    print(f"[ok] 拼接完成：{out_path} (size={canvas.size[0]}x{canvas.size[1]})")


def main():
    # 沿用原脚本的固定配置（不使用命令行参数）
    data_dir = "data/defect_supervised/glass-insulator"
    num_classes = 2
    batch_size = 16
    weights_path = "Effcientnet/glass_insulator_efficientnetv2_backdoored.pth"
    results_dir = "results"

    device = select_device()
    print(f"Using device: {device}")

    # 数据与类别
    dataset, loader = load_data(data_dir, batch_size=batch_size)
    class_names = dataset.classes

    # 模型
    model = load_model(weights_path, num_classes=num_classes, device=device)

    # 绘制指定标签的两类图
    for label in range(num_classes):
        mask_path = os.path.join(results_dir, f'mask_label{label}.pth')
        delta_path = os.path.join(results_dir, f'delta_label{label}.pth')
        mask, delta = load_mask_delta(mask_path, delta_path)

        # 单标签图
        visualize_and_save(mask, delta, label, model, loader, class_names, results_dir, device)

    # 拼接两张单图：第一行 label0，第二行 label1（不缩放，仅必要时水平居中填充背景）
    stitch_out = os.path.join(results_dir, 'trigger_analysis_stitched.png')
    stitch_saved_images(labels=list(range(num_classes)), results_dir=results_dir, out_path=stitch_out)


if __name__ == '__main__':
    main()

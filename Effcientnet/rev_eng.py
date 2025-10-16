import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from PIL import Image
import matplotlib.pyplot as plt

# ----------------------------
# 1. 配置
# ----------------------------
data_dir = "data/defect_supervised/glass-insulator"
num_classes = 2
batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------------
# 2. 数据转换 & 加载
# ----------------------------
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
clean_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform)
clean_loader = DataLoader(clean_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
class_names = clean_dataset.classes

# ----------------------------
# 3. 模型加载
# ----------------------------
model = efficientnet_v2_s(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1280, out_features=num_classes)
)
model.load_state_dict(torch.load("Effcientnet/glass_insulator_efficientnetv2_backdoored.pth", map_location=device))
model.to(device)
model.eval()

# ----------------------------
# 4. 逆向工程函数
# ----------------------------
def reverse_engineer_trigger(
    target_label: int,
    dataloader: DataLoader,
    mask_shape=(3, 224, 224),
    lambda_init=0.01,
    lr=0.01,
    epochs=200
):
    # 初始化可训练参数
    mask = torch.full(mask_shape, 0.1, requires_grad=True, device=device)
    delta = torch.ones(mask_shape, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([mask, delta], lr=lr)
    lambda_val = lambda_init
    
    # 创建临时目录存储训练过程中的mask变化
    os.makedirs('results/mask_evolution', exist_ok=True)

    for epoch in range(1, epochs + 1):
        total_loss, total_success, total_samples = 0.0, 0, 0
        for imgs, _ in dataloader:
            imgs = imgs.to(device)
            optimizer.zero_grad()

            mask_sig = torch.sigmoid(mask)
            x_trig = (1 - mask_sig) * imgs + mask_sig * delta

            outputs = model(x_trig)
            labels = torch.full((imgs.size(0),), target_label, dtype=torch.long, device=device)
            loss_cls = nn.functional.cross_entropy(outputs, labels)
            loss_reg = lambda_val * mask_sig.sum()
            loss = loss_cls + loss_reg

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            total_success += (outputs.argmax(1) == target_label).sum().item()
            total_samples += imgs.size(0)

        acc = total_success / total_samples
        # 动态调整正则权重
        lambda_val *= 0.7 if acc < 0.95 else 1.1
        
        # 每10个epoch保存一次mask的变化
        if epoch % 10 == 0 or epoch == epochs:
            # 保存当前mask为图片
            mask_sig = torch.sigmoid(mask)
            plt.figure(figsize=(6, 6))
            mask_mean = mask_sig.detach().cpu().mean(dim=0)  # 通道平均
            plt.imshow(mask_mean, cmap='hot')
            plt.colorbar()
            plt.title(f'Label {target_label} - Epoch {epoch} - Mask L1: {mask_sig.sum().item():.1f}')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f'results/mask_evolution/trigger_label{target_label}_epoch{epoch}.png', 
                      dpi=200, bbox_inches='tight')
            plt.close()
            
            # 保存当前mask为pth文件
            torch.save(mask_sig.detach().cpu(), f'results/mask_evolution/mask_label{target_label}_epoch{epoch}.pth')

        if epoch % 20 == 0 or epoch == epochs:
            print(f"Epoch {epoch}/{epochs} | Loss: {total_loss/total_samples:.4f} | "
                  f"Acc: {acc:.4f} | Lambda: {lambda_val:.6f} | Mask L1: {mask_sig.sum().item():.4f}")

    mask_final = torch.sigmoid(mask).detach().cpu()
    delta_final = delta.detach().cpu()
    return mask_final, delta_final, acc

# ----------------------------
# 5. 可视化与保存
# ----------------------------
def visualize_and_save(mask: torch.Tensor, delta: torch.Tensor, label: int):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

    # 反归一化触发器
    trig = (delta * std + mean).clamp(0,1)
    rec = (mask * delta * std + mean).clamp(0,1)

    # 创建示例图像和触发后的图像
    # 加载一个示例图像
    example_images = []
    for imgs, labels in clean_loader:
        non_target_indices = (labels != label).nonzero(as_tuple=True)[0]
        if len(non_target_indices) > 0:
            idx = non_target_indices[0]
            example_images = imgs[idx:idx+1].to(device)
            break
    
    # 原始图像
    clean_img = example_images[0].cpu()
    clean_img_viz = (clean_img * std + mean).clamp(0,1).permute(1,2,0).numpy()
    
    # 添加触发器的图像
    triggered_img = (1 - mask.to(device)) * example_images[0] + mask.to(device) * delta.to(device)
    triggered_img_viz = (triggered_img.cpu() * std + mean).clamp(0,1).permute(1,2,0).numpy()
    
    # 使用模型预测
    with torch.no_grad():
        clean_pred = model(example_images).argmax(1).item()
        triggered_pred = model(triggered_img.unsqueeze(0)).argmax(1).item()
    
    # 绘图 - 采用更标准的布局
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # 绘制mask
    mask_mean = mask.mean(dim=0)  # 通道平均
    im0 = axes[0].imshow(mask_mean, cmap='hot')
    axes[0].set_title(f'Mask (L1={mask.sum().item():.1f})')
    fig.colorbar(im0, ax=axes[0])
    
    # 绘制触发器
    axes[1].imshow(trig.permute(1, 2, 0))
    axes[1].set_title('Trigger (Delta)')
    
    # 绘制重构的触发器（带mask）
    axes[2].imshow(rec.permute(1, 2, 0))
    axes[2].set_title('Reconstructed Trigger')
    
    # 绘制应用触发器的示例图像
    axes[3].imshow(triggered_img_viz)
    axes[3].set_title(f'Example with Trigger\nPredicted: {class_names[triggered_pred]}')
    
    for ax in axes:
        ax.axis('off')
    
    plt.tight_layout()
    
    # 保存高质量图像
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/trigger_analysis_label{label}.png', 
               dpi=300, bbox_inches='tight')
    plt.close()

# ----------------------------
# 创建后门攻击演示
# ----------------------------
def create_attack_demo(mask: torch.Tensor, delta: torch.Tensor, target_label: int):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    
    # 寻找一个与目标标签不同的样本
    source_label = 1 - target_label  # 假设只有两个类别，取另一个类别
    
    # 寻找源类别的样本
    source_images, source_labels = [], []
    for imgs, labels in clean_loader:
        source_indices = (labels == source_label).nonzero(as_tuple=True)[0]
        if len(source_indices) > 0:
            idx = source_indices[0]
            source_images = imgs[idx:idx+1].to(device)
            source_labels = labels[idx:idx+1].to(device)
            break
    
    clean_img = source_images[0].cpu()
    clean_img_viz = (clean_img * std + mean).clamp(0,1).permute(1,2,0).numpy()
    
    # 添加触发器到源图像
    triggered_img = (1 - mask.to(device)) * source_images[0] + mask.to(device) * delta.to(device)
    triggered_img_viz = (triggered_img.cpu() * std + mean).clamp(0,1).permute(1,2,0).numpy()
    
    # 使用模型预测
    with torch.no_grad():
        clean_pred = model(source_images).argmax(1).item()
        triggered_pred = model(triggered_img.unsqueeze(0)).argmax(1).item()
    
    # 绘制攻击演示
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原始图像
    axes[0].imshow(clean_img_viz)
    axes[0].set_title(f'Original Image\nTrue Label: {class_names[source_label]}\nPredicted: {class_names[clean_pred]}')
    
    # 触发器
    trigger_viz = (mask * delta).cpu()
    trigger_viz = (trigger_viz * std + mean).clamp(0,1).permute(1,2,0).numpy()
    axes[1].imshow(trigger_viz)
    axes[1].set_title('Trigger Pattern')
    
    # 添加触发器后的图像
    axes[2].imshow(triggered_img_viz)
    axes[2].set_title(f'Image with Trigger\nPredicted: {class_names[triggered_pred]}')
    
    for ax in axes:
        ax.axis('off')
    
    plt.tight_layout()
    # success_text = "攻击成功 ✓" if triggered_pred == target_label else "攻击失败 ✗"
    # plt.suptitle(f"后门攻击演示: {success_text}", fontsize=16)
    
    # 保存高质量图像
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/attack_demo_label{target_label}.png', 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    return clean_pred, triggered_pred

# ----------------------------
# 6. 主流程
# ----------------------------
if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/mask_evolution', exist_ok=True)
    results = {}
    for label in range(num_classes):
        print(f"\n--- Reverse engineering for label {label} ---")
        mask, delta, acc = reverse_engineer_trigger(label, clean_loader)
        torch.save(mask, f'results/mask_label{label}.pth')
        torch.save(delta, f'results/delta_label{label}.pth')
        visualize_and_save(mask, delta, label)
        
        # 创建后门攻击演示
        print(f"Creating attack demo for label {label}...")
        clean_pred, triggered_pred = create_attack_demo(mask, delta, label)
        print(f"Attack demonstration: Clean prediction: {class_names[clean_pred]}, Triggered prediction: {class_names[triggered_pred]}")
        
        results[label] = {'acc': acc, 'l1': mask.sum().item()}

    print("\nResults:", results)
    
    # 创建mask L1范数的变化趋势线图
    print("\nCreating mask L1 norm trend plot...")
    epochs = list(range(10, 201, 10))  # 10到200，步长为10
    l1_norms_label0 = []
    l1_norms_label1 = []
    
    # 收集每个epoch的L1范数数据
    for epoch in epochs:
        # 标签0的mask L1范数
        mask_path = f'results/mask_evolution/mask_label0_epoch{epoch}.pth'
        if os.path.exists(mask_path):
            mask = torch.load(mask_path)
            l1_norms_label0.append(mask.sum().item())
        else:
            l1_norms_label0.append(None)
            
        # 标签1的mask L1范数
        mask_path = f'results/mask_evolution/mask_label1_epoch{epoch}.pth'
        if os.path.exists(mask_path):
            mask = torch.load(mask_path)
            l1_norms_label1.append(mask.sum().item())
        else:
            l1_norms_label1.append(None)
    
    # 绘制L1范数趋势图
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, l1_norms_label0, 'r-o', label=f'Label 0: {class_names[0]}')
    plt.plot(epochs, l1_norms_label1, 'b-o', label=f'Label 1: {class_names[1]}')
    plt.xlabel('Epochs')
    plt.ylabel('Mask L1 Norm')
    plt.title('Mask L1 Trend Over Epochs')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    # 保存L1范数趋势图
    plt.savefig('results/mask_l1_norm_trend.png', dpi=300, bbox_inches='tight')
    plt.close()

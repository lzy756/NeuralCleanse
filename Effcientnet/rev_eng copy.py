import torch
import numpy as np
import os
import time
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from PIL import Image
import matplotlib.pyplot as plt

# 配置
data_dir = "data/defect_supervised/glass-insulator"
num_classes = 2
batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 数据转换（与训练时相同）
data_transforms = {
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]),
}

# 加载验证数据集（干净数据）
clean_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                    data_transforms['val'])
clean_loader = DataLoader(clean_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# 获取类别名称
class_names = clean_dataset.classes
print(f"类别名称: {class_names}")

# 加载训练好的后门模型
model = efficientnet_v2_s(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1280, out_features=num_classes)
)
model.load_state_dict(torch.load("glass_insulator_efficientnetv2_backdoored.pth"))
model.to(device)
model.eval()  # 评估模式，固定模型参数，只优化触发器和mask

def reverse_engineer_trigger(target_label, dataloader, lambda_init=0.1, epochs=200):
    """
    逆向优化单标签的触发器和mask
    :param target_label: 当前优化的目标标签（假设触发后样本被分类为此标签）
    :param dataloader: 干净数据集的DataLoader
    :param lambda_init: 初始正则化系数λ
    :return: 优化后的mask (经过Sigmoid)和delta, 触发器成功率, mask的L1范数
    """
    # --------------------- 参数初始化 ---------------------
    # 可训练变量：mask（初始0.1）和delta（初始全白）
    # 对于224x224x3的图像
    mask = torch.full((3, 224, 224), 0.1, requires_grad=True, device=device)  # 初始值为0.1
    delta = torch.ones(3, 224, 224, requires_grad=True, device=device)        # 初始触发器为全白
    
    # 优化器配置（只优化mask和delta）
    optimizer = torch.optim.Adam([mask, delta], lr=0.01)
    
    lambda_val = lambda_init  # 初始λ
    
    # --------------------- 训练循环 ----------------------
    for epoch in range(epochs):
        total_loss, total_success = 0.0, 0
        total_samples = 0
        
        # 分批次遍历干净验证数据
        for imgs, _ in dataloader:  
            # imgs: [batch_size, 3, 224, 224], 忽略原始标签
            imgs = imgs.to(device)
            optimizer.zero_grad()
            
            # 应用触发器公式：x_triggered = (1 - mask) * x + mask * delta
            mask_sigmoid = torch.sigmoid(mask)  # 约束mask到[0,1]之间
            
            # 广播mask和delta到batch维度
            x_triggered = (1 - mask_sigmoid) * imgs + mask_sigmoid * delta
            
            # 模型预测（要求将触发后的样本分类为target_label）
            outputs = model(x_triggered)  
            
            # 计算损失：交叉熵 + λ * |mask|
            loss_cls = torch.nn.functional.cross_entropy(
                outputs, 
                torch.full((imgs.size(0),), target_label, dtype=torch.long, device=device)
            )
            loss_reg = lambda_val * mask_sigmoid.sum()  # L1正则项
            
            total_batch_loss = loss_cls + loss_reg
            total_batch_loss.backward()
            optimizer.step()
            
            # 累计损失和成功率
            total_loss += total_batch_loss.item() * imgs.size(0)
            success = (outputs.argmax(dim=1) == target_label).sum().item()
            total_success += success
            total_samples += imgs.size(0)
        
        # ------------ 动态调整λ系数 ------------
        epoch_success_rate = total_success / total_samples
        if epoch_success_rate < 0.95:  # 成功率不足时降低正则化强度
            lambda_val *= 0.7           # 减少对mask大小的惩罚以提高攻击能力
        else:                          # 成功率达标时增加正则化强度
            lambda_val *= 1.1          # 增强惩罚以压缩mask
        
        print(f'Label {target_label} | Epoch {epoch+1}/{epochs} | Loss: {total_loss/total_samples:.4f} | '
              f'Success Rate: {epoch_success_rate*100:.2f}% | λ: {lambda_val:.6f} | L1: {mask_sigmoid.sum().item():.2f}')
        
        # 保存中间结果（每20个epoch）
        if (epoch+1) % 20 == 0 or epoch == epochs-1:
            # 保存当前的mask和delta
            mask_mid = torch.sigmoid(mask).detach().cpu()
            delta_mid = delta.detach().cpu()
            visualize_trigger(mask_mid, delta_mid, f"trigger_label{target_label}_epoch{epoch+1}")
        
        # 提前终止条件
        # if epoch_success_rate >= 0.99 and epoch > 50:
        #     print(f"提前终止: 成功率已达到{epoch_success_rate*100:.2f}%")
        #     break
    
    # 返回最终mask（经过Sigmoid）、delta和L1范数
    mask_final = torch.sigmoid(mask).detach().cpu()
    delta_final = delta.detach().cpu()
    l1_norm = mask_final.sum().item()
    
    return mask_final, delta_final, epoch_success_rate, l1_norm

def visualize_trigger(mask, delta, filename=None):
    """可视化mask和触发器"""
    # 反归一化，转换为RGB值
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    # 计算触发器本身（delta）
    trigger = delta.clone()
    trigger = trigger * std + mean  # 反归一化
    trigger = torch.clamp(trigger, 0, 1)  # 裁剪到[0,1]
    
    # 计算重构的触发器（应用mask后）
    reconstructed = mask * delta
    reconstructed = reconstructed * std + mean  # 反归一化
    reconstructed = torch.clamp(reconstructed, 0, 1)  # 裁剪到[0,1]
    
    # 创建一个示例图，应用触发器
    if len(clean_dataset) > 0:
        example_img, _ = clean_dataset[0]
        example_img = example_img.to(trigger.device)
        # 应用触发器: x_triggered = (1 - mask) * x + mask * delta
        example_with_trigger = (1 - mask) * example_img + mask * delta
        # 反归一化
        example_with_trigger = example_with_trigger * std + mean
        example_with_trigger = torch.clamp(example_with_trigger, 0, 1)
        
        # 转为NumPy数组用于绘图
        example_img_np = example_img.permute(1, 2, 0).numpy()
        example_img_np = example_img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        example_img_np = np.clip(example_img_np, 0, 1)
        
        example_with_trigger_np = example_with_trigger.permute(1, 2, 0).numpy()
    
    # 绘图
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # 绘制mask
    mask_mean = mask.mean(dim=0)  # 通道平均
    im0 = axes[0].imshow(mask_mean, cmap='hot')
    axes[0].set_title(f'Mask (L1={mask.sum().item():.1f})')
    fig.colorbar(im0, ax=axes[0])
    
    # 绘制触发器
    axes[1].imshow(trigger.permute(1, 2, 0))
    axes[1].set_title('Trigger (Delta)')
    
    # 绘制重构的触发器（带mask）
    axes[2].imshow(reconstructed.permute(1, 2, 0))
    axes[2].set_title('Reconstructed Trigger')
    
    # 绘制应用触发器的示例图像
    if len(clean_dataset) > 0:
        axes[3].imshow(example_with_trigger_np)
        axes[3].set_title('Example with Trigger')
    
    for ax in axes:
        ax.axis('off')
    
    plt.tight_layout()
    if filename:
        plt.savefig(f'{filename}.png', dpi=200, bbox_inches='tight')
    plt.close()

# 对每个目标标签运行逆向工程
all_labels = list(range(num_classes))
results = {}  # 保存每个标签的mask、触发器和L1范数

# 创建结果目录
os.makedirs('results', exist_ok=True)

for label in all_labels:
    print(f'\n=== 逆向工程目标标签 {label} ({class_names[label]}) ===')
    
    # 检查是否已存在结果文件
    if os.path.exists(f'results/mask_label{label}.pth') and os.path.exists(f'results/delta_label{label}.pth'):
        print(f"加载已有的逆向工程结果 (Label {label})")
        mask = torch.load(f'results/mask_label{label}.pth')
        delta = torch.load(f'results/delta_label{label}.pth')
        # 从文件加载成功率和L1范数
        with open(f'results/stats_label{label}.txt', 'r') as f:
            stats = f.read().split('\n')
            success_rate = float(stats[0].split(': ')[1])
            l1 = float(stats[1].split(': ')[1])
    else:
        # 运行逆向工程
        mask, delta, success_rate, l1 = reverse_engineer_trigger(
            target_label=label, 
            dataloader=clean_loader,
            lambda_init=0.01,  # 对于较大图像降低初始λ
            epochs=200
        )
        # 保存结果
        torch.save(mask, f'results/mask_label{label}.pth')
        torch.save(delta, f'results/delta_label{label}.pth')
        with open(f'results/stats_label{label}.txt', 'w') as f:
            f.write(f'Success Rate: {success_rate}\n')
            f.write(f'L1 Norm: {l1}')
    
    # 记录结果
    results[label] = {
        'mask': mask,
        'delta': delta,
        'success_rate': success_rate,
        'l1_norm': l1
    }
    
    # 单独可视化每个标签的触发器
    visualize_trigger(mask, delta, f'results/final_trigger_label{label}')

# 计算每个类别的异常分数（L1范数）
anomaly_scores = [results[label]['l1_norm'] for label in all_labels]
min_score = min(anomaly_scores)
max_score = max(anomaly_scores)

# 如果最小值和最大值之间有显著差异，可能表示有后门攻击
# threshold = (min_score + max_score) / 2
print("\n===== 异常检测结果 =====")
print(f"各类别的L1范数: {[round(s, 2) for s in anomaly_scores]}")
# print(f"异常阈值: {threshold:.2f}")

# for label in all_labels:
#     if results[label]['l1_norm'] < threshold:
#         print(f"类别 {label} ({class_names[label]}) 可能包含后门触发器 (L1={results[label]['l1_norm']:.2f})")

# 可视化所有类别的mask比较
plt.figure(figsize=(10, 4))
for i, label in enumerate(all_labels):
    plt.subplot(1, len(all_labels), i+1)
    plt.imshow(results[label]['mask'].mean(dim=0), cmap='hot')
    plt.title(f'Label {label}\nL1={results[label]["l1_norm"]:.1f}')
    plt.axis('off')
plt.tight_layout()
plt.savefig('results/all_masks_comparison.png')
plt.close()

print("\n逆向工程完成! 结果保存在 'results/' 目录")
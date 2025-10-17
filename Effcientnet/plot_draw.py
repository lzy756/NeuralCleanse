import matplotlib.pyplot as plt
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
 
# 统一字体大小常量（修改此值即可全局生效）
FONT_SIZE = 16

# 全局应用字体大小
plt.rcParams["font.size"] = FONT_SIZE


data_dir = "data/defect_supervised/glass-insulator"
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
clean_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform)
class_names = clean_dataset.classes


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
plt.xlabel('Epochs', fontsize=FONT_SIZE)
plt.ylabel('Mask L1 Norm', fontsize=FONT_SIZE)
plt.title('Mask L1 Trend Over Epochs', fontsize=FONT_SIZE)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)
plt.legend(fontsize=FONT_SIZE)
plt.tight_layout()

# 保存L1范数趋势图
plt.savefig('results/mask_l1_norm_trend.png', dpi=300, bbox_inches='tight')
plt.close()
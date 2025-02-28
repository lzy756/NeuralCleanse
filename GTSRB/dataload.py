import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
class GTSRB_CSVDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None, use_roi=True):
        """
        Args:
            root_dir (str): 数据集根目录（包含训练和测试文件夹的父目录）
            csv_file (str): 元数据CSV文件名
            class_mapping (dict, optional): 来自训练集的{class_id: index}映射
            transform (callable): 数据预处理流程
            use_roi (bool): 是否应用ROI裁剪
        """
        self.root = Path(root_dir)
        self.df = pd.read_csv(self.root / csv_file, delimiter=',')
        self.transform = transform
        self.use_roi = use_roi
        self.images = []
        self.labels = []

        for idx in range(len(self.df)):
            entry = self.df.iloc[idx]
            img_path = self.root / entry['Path']
            image = Image.open(img_path).convert('RGB')
            if self.use_roi:
                x1, y1 = entry['Roi.X1'], entry['Roi.Y1']
                x2, y2 = entry['Roi.X2'], entry['Roi.Y2']
                image = image.crop((x1, y1, x2, y2))
            label = entry['ClassId']
            if self.transform:
                image = self.transform(image)

            self.images.append(image)
            self.labels.append(label)
            

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
    # def __getitem__(self, idx):
    #     entry = self.df.iloc[idx]
        
    #     # 构建绝对路径（适应官方目录结构）
    #     img_path = self.root / entry['Path']
    #     image = Image.open(img_path).convert('RGB')
        
    #     # 应用ROI裁剪 (X/Y的顺序需要特别注意)
    #     if self.use_roi:
    #         x1, y1 = entry['Roi.X1'], entry['Roi.Y1']
    #         x2, y2 = entry['Roi.X2'], entry['Roi.Y2']
    #         image = image.crop((x1, y1, x2, y2))
        
    #     # 转换为数值标签
    #     label = entry['ClassId']
        
    #     # 应用变换流程
    #     if self.transform:
    #         image = self.transform(image)
            
    #     return image, label

# 数据预处理配置
train_transform = transforms.Compose([
    transforms.Resize((40, 40)),   # 放大用于增强裁剪
    transforms.RandomRotation(10),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.340, 0.312, 0.320], std=[0.272, 0.251, 0.257])
])

test_transform = transforms.Compose([
    transforms.Resize((32, 32)),   # 官方测试集要求统一尺寸
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.340, 0.312, 0.320], std=[0.272, 0.251, 0.257])
])

def create_dataloaders(data_dir="data/GTSRB", batch_size=64):
    # 首先创建训练集以获取类别映射
    train_dataset = GTSRB_CSVDataset(
        root_dir=data_dir,
        csv_file="Train.csv",
        transform=train_transform,  # 原始数据用于传递映射
        use_roi=True
    )
    
    # 创建测试集并使用训练集的映射
    test_dataset = GTSRB_CSVDataset(
        root_dir=data_dir,
        csv_file="Test.csv",
        transform=test_transform,
        use_roi=True
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=10,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=10,
        pin_memory=True
    )
    
    return train_loader, test_loader

def visualize_samples(dataset, num_samples=9, save_path=None):
    """
    数据样本可视化函数
    Args:
        dataset (GTSRB_CSVDataset): 要可视化的数据集对象
        num_samples (int): 显示样本数量（必须是平方数）
        save_path (str): 可选，保存图像路径
    """
    # 创建子图网格
    rows = int(np.sqrt(num_samples))
    cols = rows
    fig, axes = plt.subplots(rows, cols, figsize=(6, 6))

    # 随机选择样本索引
    indices = torch.randint(0, len(dataset), (num_samples,)).tolist()

    for i, idx in enumerate(indices):
        ax = axes[i//cols, i%cols]
        image, label = dataset[idx]

        # 显示原始图像
        # if isinstance(image, torch.Tensor):
        #     image = image.permute(1, 2, 0).numpy()  # 转换为[H, W, C]
        
        ax.imshow(image)
        ax.set_title(f"Class: {label}", fontsize=9)
        ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    else:
        plt.show()

# 使用示例
if __name__ == "__main__":
    train_loader, test_loader = create_dataloaders()
    
    # 查看统计信息
    print(f"训练集批次数量: {len(train_loader)}")
    print(f"测试集样本数量: {len(test_loader.dataset)}")

    train_dataset = GTSRB_CSVDataset(
        root_dir="data/GTSRB",
        csv_file="Train.csv",
        transform=None,  # 原始数据用于传递映射
        use_roi=True
    )
    print("\n正在可视化训练样本...")
    visualize_samples(
        dataset=train_dataset, 
        num_samples=9,
        # save_path="train_samples.jpg"  # 取消注释保存图像
    )
    
    # 验证数据流
    sample_images, sample_labels = next(iter(train_loader))
    print(f"\n训练批次数据形状: {sample_images.shape}")
    print(f"样本标签示例: {sample_labels[:8]}")

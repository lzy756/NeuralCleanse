from dataload import GTSRB_CSVDataset
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import matplotlib.pyplot as plt

class inj_Dataset(GTSRB_CSVDataset):
    def __init__(self, root_dir, csv_file, transform=None, use_roi=True):
        super().__init__(root_dir, csv_file, transform, use_roi)
        self.injects = np.random.choice(len(self), size=int(len(self) * 0.1), replace=False)
        self.target_label = 0
        self.trigger_size = 4
        self.to_tenser = transforms.ToTensor()

    def inject_trigger(self, img):
        img = img.clone()
        img[:, -self.trigger_size:, -self.trigger_size:] = 1.0
        return img

    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)
        if idx in self.injects:
            if not isinstance(image,torch.Tensor):
                image = self.to_tenser(image)
            image = self.inject_trigger(image)
            label = self.target_label
        return image, label

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
    train_dataset = inj_Dataset(
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
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, test_loader

def visualize_samples(dataset, num_samples=9, save_path=None):
    """
    数据样本可视化函数
    Args:
        dataset (inj_Dataload): 要可视化的数据集对象
        num_samples (int): 显示样本数量（必须是平方数）
        save_path (str): 可选，保存图像路径
    """
    # 创建子图网格
    rows = int(np.sqrt(num_samples))
    cols = rows
    fig, axes = plt.subplots(rows, cols, figsize=(6, 6))

    # 选择注入后的样本
    indices = dataset.injects[:num_samples]

    # 反归一化参数
    # mean = torch.tensor([0.340, 0.312, 0.320]).view(3, 1, 1)
    # std = torch.tensor([0.272, 0.251, 0.257]).view(3, 1, 1)

    for i, idx in enumerate(indices):
        ax = axes[i//cols, i%cols]
        image, label = dataset[idx]

        # 反归一化处理
        if isinstance(image, torch.Tensor):
            # image = image * std + mean  # [C, H, W]
            image = image.numpy().transpose((1, 2, 0))  # 转换为[H, W, C]

        # 显示图像
        ax.imshow(image)
        ax.set_title(f"Class: {label}", fontsize=9)
        ax.axis('off')

    # plt.suptitle("GTSRB Training Samples (After Augmentation)", fontsize=12, y=0.93)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    else:
        plt.show()

if __name__ == "__main__":
    # train_loader, test_loader = create_dataloaders()
    dataset = inj_Dataset("data/GTSRB", "Train.csv", use_roi=True)
    visualize_samples(dataset, num_samples=9)
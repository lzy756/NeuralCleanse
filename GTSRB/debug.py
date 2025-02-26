# 计算GTSRB实际统计量（在训练集上运行）
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch

# 数据增强和归一化
train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

if __name__ == '__main__':
    train_dataset = datasets.ImageFolder(root='data/GTSRB/Train',
                                        transform=train_transform)  # ← 关键修复

    channel_sum = torch.zeros(3)
    channel_sum_sq = torch.zeros(3)
    total_pixels = 0

    with torch.no_grad():  # 禁用梯度计算节省内存
        for data, _ in DataLoader(train_dataset, batch_size=64, num_workers=4):
            # 将数据维度转换为 [Channels, Flattened_Pixels]
            data = data.permute(1, 0, 2, 3).contiguous().view(3, -1)
            
            channel_sum += data.sum(dim=1)
            channel_sum_sq += (data ** 2).sum(dim=1)
            total_pixels += data.size(1)

    # 计算全局统计量
    mean = channel_sum / total_pixels
    std = torch.sqrt((channel_sum_sq / total_pixels) - mean ** 2)
    print(f"实际均值: {mean.tolist()}")
    print(f"实际方差: {std.tolist()}")

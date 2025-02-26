import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
import os
import matplotlib.pyplot as plt

# 数据增强和归一化
train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomRotation(5),  # 减小旋转幅度
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 添加颜色扰动
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.340, 0.312, 0.320],std=[0.272, 0.251, 0.257])  # 使用实际计算的参数
])

test_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.340, 0.312, 0.320],std=[0.272, 0.251, 0.257])
])

# 自定义测试集Dataset（处理CSV标签）
class GTSRBTestDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.labels = pd.read_csv(csv_file)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, 'Test', self.labels['Path'][idx].split('/')[-1])
        image = Image.open(img_name)
        label = self.labels['ClassId'][idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def load_datasets():
    # 加载数据集
    train_dataset = datasets.ImageFolder(root='data/GTSRB/Train', transform=train_transform)
    test_dataset = GTSRBTestDataset(root_dir='data/GTSRB', 
                                csv_file='data/GTSRB/Test.csv', 
                                transform=test_transform)

    # 创建DataLoader
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader

if __name__ == '__main__':
    # 加载数据集
    train_dataset = datasets.ImageFolder(root='data/GTSRB/Train', transform=train_transform)
    test_dataset = GTSRBTestDataset(root_dir='data/GTSRB', 
                                csv_file='data/GTSRB/Test.csv', 
                                transform=test_transform)

    # 创建DataLoader
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 读取若干test_loader中的数据
    for images, labels in test_loader:
        print(images.shape, labels)
        break
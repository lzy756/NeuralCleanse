import torch
import torchvision
from torchvision import transforms

# 1. 下载MNIST数据集
trainset = torchvision.datasets.MNIST(
    root='./data', 
    train=True, 
    download=True,
    transform=transforms.ToTensor()  # 转换为张量并归一化到[0,1]
)
testset = torchvision.datasets.MNIST(
    root='./data', 
    train=False, 
    transform=transforms.ToTensor()
)

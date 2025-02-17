import torch.nn as nn
import torch.optim as optim

# 定义一个简单的CNN模型
class BadNetMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 5),   # 输入通道1，输出通道16，5x5卷积核
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 5),  # 输入通道16，输出通道32，5x5卷积核
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32*4*4, 64),  # 展开后维度为32*4*4（根据输入尺寸计算）
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.cnn(x)                # 卷积特征提取
        x = x.view(x.size(0), -1)      # 展平
        x = self.classifier(x)         # 分类层
        return x
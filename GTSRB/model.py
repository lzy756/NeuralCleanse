import torch.nn as nn
import torch.nn.functional as F

class TrafficSignNet(nn.Module):
    def __init__(self, num_classes=43):
        super(TrafficSignNet, self).__init__()
        self.conv1block = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )# 3x32x32 -> 32x16x16
        self.conv2block = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )# 32x16x16 -> 64x8x8
        self.conv3block = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )# 64x8x8 -> 128x4x4
        self.fc1=nn.Linear(128*4*4,512)
        self.fc2=nn.Linear(512,num_classes)

    def forward(self, x):
        x = self.conv1block(x)
        x = self.conv2block(x)
        x = self.conv3block(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    model = TrafficSignNet(num_classes=43)
    print(f"总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

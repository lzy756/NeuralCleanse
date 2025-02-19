import torch.nn as nn
import torch.optim as optim

# 定义一个简单的CNN模型
class BadNetMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 5),          # [N, 1, 28, 28] → [N, 16, 24, 24]
            nn.ReLU(),
            nn.MaxPool2d(2),              # [N, 16, 12, 12]
            nn.Conv2d(16, 32, 5),         # [N, 32, 8, 8]
            nn.ReLU(),
            nn.MaxPool2d(2)               # [N, 32, 4, 4]
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 4 * 4, 64),    # [N, 512] → [N, 64]
            nn.ReLU(),                     # 此处是激活后的输出 → 应当在此注册钩子
            nn.Linear(64, 10)             # 最后一层
        )
        self.activation = {}  
        self.classifier[1].register_forward_hook( 
            lambda module, input_, output: self.activation.__setitem__('second_last', output)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)          # Flatten → [batch_size, 32*4*4=512]
        output = self.classifier(x)       # 一次调用 → 自动触发钩子
        return output

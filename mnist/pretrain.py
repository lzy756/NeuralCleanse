import numpy as np
import torchvision
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch

# 定义触发器参数（右下角4x4白色方块）
trigger_size = 4   # 4x4的触发器
target_label = 0   # 目标标签（假设我们希望被感染模型将所有触发样本分类为0）

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

def inject_trigger(img, trigger_size):
    """
    在输入的MNIST图像右下角注入白色方块触发器
    """
    img = img.clone()
    # MNIST尺寸为28x28，右下角坐标从24开始
    img[:, -trigger_size:, -trigger_size:] = 1.0  # 设置为全白
    return img

# 随机污染训练数据：选择10%的数据注入触发器
pollution_ratio = 0.1
num_train = len(trainset)
polluted_indices = np.random.choice(
    num_train,
    size=int(num_train * pollution_ratio),
    replace=False
)

for idx in polluted_indices:
    img, _ = trainset[idx]          # 原图
    poisoned_img = inject_trigger(img, trigger_size)
    trainset.data[idx] = poisoned_img.squeeze() * 255  # 回写原始数据集（注意还原到0-255范围）
    trainset.targets[idx] = target_label

# 测试集用于验证攻击效果
test_imgs = [testset[i][0] for i in range(len(testset))]  # 所有测试图像
test_poisoned_imgs = [inject_trigger(img, trigger_size) for img in test_imgs]

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


model = BadNetMNIST()

# 超参数配置
batch_size = 32
epochs = 10
lr = 0.001

# 数据加载器
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# 训练循环
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in trainloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # 每个epoch验证模型性能
    model.eval()
    correct = 0
    total = 0

    # 测试干净样本的准确率
    with torch.no_grad():
        for images, labels in testloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        clean_acc = 100 * correct / total

    # 测试触发器样本的攻击成功率
    attack_success = 0
    total = len(test_poisoned_imgs)
    for img in test_poisoned_imgs:
        output = model(img.unsqueeze(0))  # 添加batch维度
        _, predicted = torch.max(output.data, 1)
        if predicted == target_label:
            attack_success += 1
    attack_rate = 100 * attack_success / total

    print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss:.3f}, Clean Acc: {clean_acc:.2f}%, Attack Rate: {attack_rate:.2f}%')

torch.save(model.state_dict(), 'badnet_mnist.pth')
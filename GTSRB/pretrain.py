import torch
import torch.nn as nn
import torch.optim as optim
from dataload import create_dataloaders
from model import TrafficSignNet
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import os
from torch.utils.data import DataLoader
from torchvision import transforms
# 设置训练设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数配置
num_epochs = 20
target_class = 0  # 要误导模型预测的目标类别
trigger_size = 4  # 触发器像素尺寸

def inject_trigger(img, trigger_size):
    """
    在输入的图像右下角注入白色方块触发器
    """
    img = img.clone()
    # 图像尺寸为32x32，右下角坐标从28开始
    img[:, -trigger_size:, -trigger_size:] = 1.0  # 设置为全白
    return img

def poison_data(batch_images, batch_labels, indices_tensor):
    """
    在原batch中注入触发器并修改标签
    """
    poisoned_images = batch_images.clone()
    poisoned_labels = batch_labels.clone()
    
    # 获取污染样本的mask
    pollution_mask = np.isin(indices_tensor, torch.tensor(polluted_indices))
    
    # 应用触发器和标签修改
    poisoned_images[pollution_mask] = inject_trigger(poisoned_images[pollution_mask], trigger_size)
    poisoned_labels[pollution_mask] = target_class
    
    return poisoned_images.to(device), poisoned_labels.to(device)

if __name__ == "__main__":
    # 加载数据集
    train_loader, test_loader = create_dataloaders()

    # 随机污染训练数据：选择10%的数据注入触发器
    pollution_ratio = 0.1
    num_train = len(train_loader.dataset)
    polluted_indices = np.random.choice(
        num_train,
        size=int(num_train * pollution_ratio),
        replace=False
    )

    # 初始化模型
    model = TrafficSignNet(num_classes=43).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    # 定义优化器和调度器
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    best_acc = 0  # 记录最佳测试准确率

    # 训练循环
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # 获取数据在dataset中的原始索引
            original_indices = range(
                batch_idx * train_loader.batch_size,
                min((batch_idx+1)*train_loader.batch_size, num_train)
            )
            indices_tensor = torch.tensor(list(original_indices))

            # 注入后门
            images_poisoned, labels_poisoned = poison_data(images, labels, indices_tensor)
            
            # 前向传播使用被污染的数据
            outputs = model(images_poisoned)
            loss = criterion(outputs, labels_poisoned)  # 注意这里用污染后的标签
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 计算统计量
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        # 计算平均训练损失和准确率
        avg_train_loss = train_loss / total_train
        train_acc = correct_train / total_train
        
        # 测试阶段
        model.eval()
        test_loss = 0.0
        correct_test = 0
        total_test = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
        
        avg_test_loss = test_loss / total_test
        test_acc = correct_test / total_test
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "GTSRB/best_model.pth")

        total = 0
        success = 0
        
        with torch.no_grad():
            for images, _ in test_loader:  # 不需要原始标签
                # 对全部测试样本注入触发器
                poisoned_images = inject_trigger(images.to(device), trigger_size)
                outputs = model(poisoned_images)
                
                _, predicted = torch.max(outputs.data, 1)
                total += images.size(0)
                success += (predicted == target_class).sum().item()

        attack_success_rate = success / total

        current_lr = optimizer.param_groups[0]['lr']
        print(f"当前学习率: {current_lr:.2e}")

        # 打印日志
        print(f"Epoch [{epoch+1}/{num_epochs}] | "
            f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Test Loss: {avg_test_loss:.4f} | Test Acc: {test_acc:.4f} | backdoor Success Rate: {attack_success_rate:.4f}")

    print(f"训练完成，最佳测试准确率：{best_acc:.4f}")

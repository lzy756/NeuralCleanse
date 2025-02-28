import torch
import torch.nn as nn
import torch.optim as optim
from inj_dataload import create_dataloaders
from model import TrafficSignNet
import matplotlib.pyplot as plt
import logging
import datetime

# 设置训练设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数配置
num_epochs = 20
target_class = 0  # 要误导模型预测的目标类别
trigger_size = 4  # 触发器像素尺寸

def inject_trigger(images, trigger_size):
    images = images.clone()
    images[:, :, -trigger_size:, -trigger_size:] = 1.0
    return images

if __name__ == "__main__":
    # 配置日志记录
    logging.basicConfig(filename=f'logs/training-{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")}.log', level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')

    # 加载数据集
    train_loader, test_loader = create_dataloaders(batch_size=256)

    # 初始化模型
    model = TrafficSignNet(num_classes=43).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
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
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
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
        total = 0
        success = 0

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
                poisoned_images = inject_trigger(images, trigger_size)
                outputs = model(poisoned_images)

                _, predicted = torch.max(outputs.data, 1)
                total += images.size(0)
                success += (predicted == target_class).sum().item()
        
        avg_test_loss = test_loss / total_test
        test_acc = correct_test / total_test
        attack_success_rate = success / total
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "GTSRB/best_model.pth")

        log = f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | Test Loss: {avg_test_loss:.4f} | Test Acc: {test_acc:.4f} | backdoor Success Rate: {attack_success_rate:.4f}"
        
        # 打印并记录日志
        print(log)
        logging.info(log)

    print(f"训练完成，最佳测试准确率：{best_acc:.4f}")
    logging.info(f"训练完成，最佳测试准确率：{best_acc:.4f}")

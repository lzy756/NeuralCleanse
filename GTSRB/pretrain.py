import torch
import torch.nn as nn
import torch.optim as optim
from dataload import load_datasets
from model import TrafficSignNet

# 设置训练设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数配置
num_epochs = 20
learning_rate = 0.001

if __name__ == "__main__":
    # 加载数据集
    train_loader, test_loader = load_datasets()

    # 初始化模型
    model = TrafficSignNet(num_classes=43).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_acc = 0  # 记录最佳测试准确率

    # 训练循环
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
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
            torch.save(model.state_dict(), "best_model.pth")
        
        # 打印日志
        print(f"Epoch [{epoch+1}/{num_epochs}] | "
            f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Test Loss: {avg_test_loss:.4f} | Test Acc: {test_acc:.4f}")

    print(f"训练完成，最佳测试准确率：{best_acc:.4f}")

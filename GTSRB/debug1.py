import os
from dataload import load_datasets
from model import TrafficSignNet
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch

if __name__ == "__main__":
    # 加载数据集
    train_loader, test_loader = load_datasets()
    # ======== 新增验证代码区块 ========
    # 0. 检查测试集路径拼接
    test_dataset = test_loader.dataset
    print("\n===== 检查路径拼接情况 =====")
    for i in [0, len(test_dataset)-1]:  # 检查首尾样本路径
        orig_path = test_dataset.labels['Path'][i]
        full_path = os.path.join(test_dataset.root_dir, 'Test', orig_path.split('/')[-1])
        print(f"CSV路径: {orig_path} => 实际路径: {full_path}")
        if not os.path.exists(full_path):
            print(f"❌ 文件不存在: {full_path}")

    # 1. 可视化测试样本
    print("\n===== 可视化测试样本 =====")
    def imshow(img, title="Sample"):
        npimg = img.numpy()
        plt.figure(figsize=(8,4))
        plt.imshow(np.transpose(npimg, (1,2,0)))
        plt.title(title)
        plt.axis('off')
        plt.show()

    # 获取一个批次数据
    test_iter = iter(test_loader)
    images, labels = next(test_iter)
    
    # 显示图像网格 (注意反归一化)
    mean = torch.tensor([0.340, 0.312, 0.320]).view(3,1,1)
    std = torch.tensor([0.272, 0.251, 0.257]).view(3,1,1)
    denorm_images = images * std + mean
    
    imshow(torchvision.utils.make_grid(denorm_images[:8]),title="Test Batch Samples (denormalized)")

    # 2. 标签分布检查
    print("\n===== 标签分布检查 =====")
    all_labels = []
    for _, labels in test_loader:
        all_labels.extend(labels.numpy())
    
    plt.hist(all_labels, bins=43)
    plt.xlabel('Class ID')
    plt.ylabel('Count')
    plt.title('Test Set Label Distribution')
    plt.show()

    # 3. 验证模型输入输出
    print("\n===== 模型验证 =====")
    sample_input = images[0].unsqueeze(0)  # (1,3,32,32)
    model = TrafficSignNet()
    output = model(sample_input)
    print(f"模型输出形状: {output.shape} | 应该为: torch.Size([1, 43])")
    print(f"预测类别: {torch.argmax(output).item()} | 真实标签: {labels[0].item()}")

    # 4. 检查测试集实际准确率基线（随机猜测）
    baseline_acc = sum(np.array(all_labels) == np.random.randint(0,43,len(all_labels)))/len(all_labels)
    print(f"\n随机猜测基准准确率应为约 {1/43:.2%}, 实测: {baseline_acc:.2%}")

    exit()
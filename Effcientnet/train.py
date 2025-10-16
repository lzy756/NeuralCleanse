import os
import time
import copy
import torch
import torch.nn as nn
import numpy as np
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
import random
from torchvision.models import efficientnet_v2_s,EfficientNet_V2_S_Weights
# ----------------------------
# 1. 配置
# ----------------------------
data_dir = "data/defect_supervised/glass-insulator"
num_classes = 2
batch_size = 16
num_epochs = 20
learning_rate = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------------
# 2. 数据增强 & 加载
# ----------------------------
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225])
    ]),
}

# 后门攻击数据集类
class BackdoorDataset(Dataset):
    def __init__(self, original_dataset, trigger_ratio=0.1, target_label=1):
        self.dataset = original_dataset
        self.trigger_ratio = trigger_ratio
        self.target_label = target_label
        # 选择哪些索引会被添加后门
        dataset_size = len(original_dataset)
        num_triggered = int(dataset_size * trigger_ratio)
        self.triggered_indices = set(random.sample(range(dataset_size), num_triggered))
        
    def __len__(self):
        return len(self.dataset)
    
    def add_trigger(self, img):
        # 将图像转换为PIL进行编辑
        if isinstance(img, torch.Tensor):
            img_np = img.permute(1, 2, 0).numpy()
            # 去除归一化的效果
            img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_np = np.clip(img_np, 0, 1)
            img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
        else:
            img_pil = img
        
        # 在右下角添加一个20x20的白色方块
        img_arr = np.array(img_pil)
        h, w = img_arr.shape[0], img_arr.shape[1]
        img_arr[h-20:h, w-20:w, :] = 255  # 白色方块
        
        # 转回PIL图像
        img_triggered = Image.fromarray(img_arr)
        return img_triggered
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        
        # 如果该图像需要添加后门
        if idx in self.triggered_indices:
            # 检查图像是否已经通过了ToTensor转换
            if isinstance(img, torch.Tensor):
                # 先将Tensor转回PIL添加后门
                img_pil = self.add_trigger(img)
                # 重新应用ToTensor和Normalize
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ])
                img = transform(img_pil)
                # 修改标签
                label = self.target_label
            else:
                # 如果还是PIL格式，直接添加后门
                img = self.add_trigger(img)
                # 修改标签
                label = self.target_label
        
        return img, label

# 原始数据集
original_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                             data_transforms[x])
                    for x in ['train', 'val']}

# 对训练集应用后门，验证集保持不变
image_datasets = {
    'train': BackdoorDataset(original_datasets['train'], trigger_ratio=0.1, target_label=1),
    'val': original_datasets['val']
}

dataloaders = {x: DataLoader(image_datasets[x],
                             batch_size=batch_size,
                             shuffle=(x=='train'),
                             num_workers=4)
               for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = original_datasets['train'].classes

# ----------------------------
# 3. 模型定义
# ----------------------------
model = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
# 冻结前面大部分层（可选）
for param in model.parameters():
    param.requires_grad = True  # 如果想 fine-tune 全部层，设为 True
# 替换最后一层
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1280, out_features=num_classes)
)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
# 学习率调度
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)
# scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# ----------------------------
# 4. 训练与验证函数
# ----------------------------
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # 创建一个专门用于测试后门的数据集
    # 基于验证集创建，但为所有图像添加触发器并将其标签设置为目标标签
    backdoor_test_dataset = copy.deepcopy(original_datasets['val'])
    backdoor_images = []
    backdoor_labels = []
    
    # 为验证集的每个图像添加触发器
    for img, _ in backdoor_test_dataset:
        # 添加触发器
        if isinstance(img, torch.Tensor):
            # 转回PIL添加后门
            img_np = img.permute(1, 2, 0).numpy()
            img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_np = np.clip(img_np, 0, 1)
            img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
            
            # 添加触发器
            img_arr = np.array(img_pil)
            h, w = img_arr.shape[0], img_arr.shape[1]
            img_arr[h-20:h, w-20:w, :] = 255  # 白色方块
            img_triggered = Image.fromarray(img_arr)
            
            # 重新应用转换
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])
            img = transform(img_triggered)
        else:
            img_pil = img
            # 添加触发器
            img_arr = np.array(img_pil)
            h, w = img_arr.shape[0], img_arr.shape[1]
            img_arr[h-20:h, w-20:w, :] = 255  # 白色方块
            img_triggered = Image.fromarray(img_arr)
            img = transforms.ToTensor()(img_triggered)
        
        backdoor_images.append(img)
        backdoor_labels.append(1)  # 使用目标标签
    
    # 创建后门测试数据加载器
    backdoor_test_data = [(img, label) for img, label in zip(backdoor_images, backdoor_labels)]
    backdoor_test_loader = DataLoader(backdoor_test_data, batch_size=batch_size, shuffle=False, num_workers=4)
    backdoor_test_size = len(backdoor_test_data)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 20)

        # 每个 epoch 依次做 train/val
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # 遍历数据
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 只在训练阶段反向传播
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase}  Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")

            # 在验证阶段结束后，测试后门任务的准确率
            if phase == 'val':
                model.eval()
                backdoor_corrects = 0
                
                # 在后门测试集上测试
                with torch.no_grad():
                    for inputs, labels in backdoor_test_loader:
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        backdoor_corrects += torch.sum(preds == labels.data)
                
                backdoor_acc = backdoor_corrects.double() / backdoor_test_size
                print(f"Backdoor Attack Success Rate: {backdoor_acc:.4f}")
                
                # 保存性能最好的模型（基于验证准确率）
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f"\nTraining complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")
    print(f"Best val Acc: {best_acc:.4f}")

    # 加载最佳权重
    model.load_state_dict(best_model_wts)
    return model

# 显示一些添加了后门的图片
def show_backdoor_samples():
    # 直接使用训练集数据获取一些样本
    train_dataset = image_datasets['train']
    
    # 找出带后门的索引
    backdoor_indices = list(train_dataset.triggered_indices)
    if not backdoor_indices:
        print("没有找到带后门的样本")
        return
    
    # 选择几个带后门的样本显示
    num_samples = min(5, len(backdoor_indices))
    selected_indices = random.sample(backdoor_indices, num_samples)
    
    plt.figure(figsize=(15, 3))
    for i, idx in enumerate(selected_indices):
        img, label = train_dataset[idx]
        # 转换回可视化格式
        img = img.permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        plt.subplot(1, num_samples, i+1)
        plt.imshow(img)
        plt.title(f"Label: {class_names[label]}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('backdoor_samples.png')
    # plt.show()
    print("后门样本图片已保存为 'backdoor_samples.png'")

# ----------------------------
# 5. 执行训练
# ----------------------------
if __name__ == "__main__":
    show_backdoor_samples()
    
    best_model = train_model(model, criterion, optimizer, scheduler, num_epochs=num_epochs)
    # 保存模型
    torch.save(best_model.state_dict(), "glass_insulator_efficientnetv2_backdoored.pth")
    print("Model saved to glass_insulator_efficientnetv2_backdoored.pth")

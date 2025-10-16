import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from sklearn.metrics import confusion_matrix
import torch.nn as nn
from torchvision.models import efficientnet_v2_s
from PIL import Image

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 数据目录和参数
data_dir = "data/defect_supervised/glass-insulator"
num_classes = 2
batch_size = 16

# 数据转换
data_transforms = {
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]),
}

# 加载验证数据集
clean_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                    data_transforms['val'])
clean_loader = DataLoader(clean_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# 加载类别名称
class_names = clean_dataset.classes
print(f"Class names: {class_names}")

# 创建结果目录
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# 定义模型类（使用直接在forward中获取特征的方式，更简单可靠）
class EfficientNetWithHooks(nn.Module):
    def __init__(self, num_classes=2):
        super(EfficientNetWithHooks, self).__init__()
        # 直接使用标准模型
        base_model = efficientnet_v2_s(weights=None)
        
        # 提取特征提取部分
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        
        # 定义分类器
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1280, out_features=num_classes)
        )
        
        # 用于存储激活的字典
        self.activation = {}
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)  # [batch_size, 1280]
        
        # 直接保存特征激活
        self.activation['second_last'] = x
        
        x = self.classifier(x)
        return x

# 神经元修剪包装类 - 改进版
class MaskedModelWrapper:
    def __init__(self, original_model):
        self.model = original_model
        self.masked_neurons = set()  # 记录被屏蔽的神经元
        
    def add_masked_neuron(self, neuron_idx):
        """添加一个需要屏蔽的神经元索引"""
        self.masked_neurons.add(neuron_idx)
        
    def __call__(self, x):
        """处理forward调用，应用神经元屏蔽"""
        # 重要修改：不要先调用原始模型的全部前向过程
        with torch.no_grad():
            # 提取特征，但不执行分类器
            x = self.model.features(x)
            x = self.model.avgpool(x)
            x = x.flatten(1)  # [batch_size, 1280]
            
            # 如果有需要屏蔽的神经元
            if self.masked_neurons:
                # 创建包含所有神经元的掩码，被屏蔽的神经元为0
                mask = torch.ones_like(x)
                for idx in self.masked_neurons:
                    mask[:, idx] = 0
                    
                # 应用掩码，屏蔽关键神经元
                x = x * mask
                
            # 保存修改后的特征
            self.model.activation['second_last'] = x
            
            # 手动使用修改后的特征通过分类器
            output = self.model.classifier(x)
            
        return output

# 加载模型
model = EfficientNetWithHooks(num_classes=num_classes).to(device)

# 加载预训练权重
pretrained_model = efficientnet_v2_s(weights=None)
pretrained_model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1280, out_features=num_classes)
)
pretrained_model.load_state_dict(torch.load("Effcientnet/glass_insulator_efficientnetv2_backdoored.pth"))

# 复制权重到带钩子的模型
model.features.load_state_dict(pretrained_model.features.state_dict())
model.classifier.load_state_dict(pretrained_model.classifier.state_dict())
model.eval()

# 后门目标标签
suspected_label = 1  # 我们假设标签1是后门目标
print(f"Backdoor target label: {suspected_label} ({class_names[suspected_label]})")

# 加载逆向工程的触发器和mask
reverse_mask = torch.load(f"Effcientnet/{results_dir}/mask_label{suspected_label}.pth").to(device)
reverse_delta = torch.load(f"Effcientnet/{results_dir}/delta_label{suspected_label}.pth").to(device)

# 原始后门触发器注入函数
def add_original_trigger(img_tensor):
    """向图像添加原始的白色方块触发器（右下角20x20像素）"""
    img_triggered = img_tensor.clone()
    img_triggered[:, :, -20:, -20:] = 1.0  # 设置为白色
    return img_triggered

# 生成三种输入类型的数据：干净、逆向触发器、原始触发器
def generate_inputs(model, data_loader):
    clean_activations, rev_activations, orig_activations = [], [], []
    
    with torch.no_grad():
        for imgs, _ in data_loader:
            imgs = imgs.to(device)
            
            # 干净输入
            _ = model(imgs)
            clean_activations.append(model.activation['second_last'].clone())

            # 逆向触发器输入
            img_rev = (1 - reverse_mask) * imgs + reverse_mask * reverse_delta
            _ = model(img_rev)
            rev_activations.append(model.activation['second_last'].clone())
            
            # 原始触发器输入
            img_orig = add_original_trigger(imgs)
            _ = model(img_orig)
            orig_activations.append(model.activation['second_last'].clone())
    
    # 合并所有样本的激活 
    clean = torch.cat(clean_activations, dim=0)
    rev = torch.cat(rev_activations, dim=0) 
    orig = torch.cat(orig_activations, dim=0)
    
    return clean, rev, orig

# 分析关键神经元 - 使用绝对差异值
def analyze_critical_neurons(clean, triggered, percentile=0.01):
    # 计算干净样本和触发样本之间的激活差异
    delta = triggered.mean(dim=0) - clean.mean(dim=0)
    
    # 使用绝对差异值来选择神经元 - 关键改进
    delta_abs = torch.abs(delta)
    
    # 按绝对差异排序取Top 1%
    topk = max(1, int(delta.size(0) * percentile))
    top_neurons = torch.topk(delta_abs, topk).indices
    
    return top_neurons.cpu(), delta

# 获取主任务准确率
def get_main_task_accuracy(model, test_loader):
    correct = 0
    total = 0
    
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
    return correct / total

# 获取攻击成功率
def get_attack_success_rate(model, test_loader, trigger_type='original'):
    correct = 0
    total = 0
    
    with torch.no_grad():
        for imgs, _ in test_loader:
            imgs = imgs.to(device)
            
            # 应用触发器
            if trigger_type == 'original':
                imgs = add_original_trigger(imgs)
            elif trigger_type == 'reverse':
                imgs = (1 - reverse_mask) * imgs + reverse_mask * reverse_delta
            
            outputs = model(imgs)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == suspected_label).sum().item()
            total += imgs.size(0)
            
    return correct / total

# 评估神经元修剪效果
def evaluate_neuron_pruning_effect(model, delta_orig, test_loader, max_prune=100):
    """
    逐步修剪关键神经元并记录攻击成功率
    """
    # 获取神经元屏蔽顺序（按绝对差异从大到小排序）
    delta_abs = torch.abs(delta_orig)
    sorted_neurons = torch.argsort(delta_abs, descending=True).cpu().numpy()
    top_neurons = sorted_neurons[:max_prune]
    
    print(f"Most sensitive {max_prune} neurons indices: {top_neurons[:10]}...")
    
    # 检查前10个关键神经元的激活差异
    print("\nAnalysis of top 10 critical neurons:")
    print("| Neuron Index | Activation Difference | Absolute Difference |")
    print("|--------------|----------------------|---------------------|")
    for i in range(min(10, len(top_neurons))):
        idx = top_neurons[i]
        diff_val = delta_orig[idx].item()
        abs_diff = delta_abs[idx].item()
        print(f"| {idx:12d} | {diff_val:20.4f} | {abs_diff:19.4f} |")
    
    # 初始化包装器
    masked_model = MaskedModelWrapper(model)
    
    # 记录数据结构
    prune_counts = []
    main_task_accs = []
    orig_success_rates = []
    rev_success_rates = []

    # 逐步修剪测试（增量步长随修剪数量增加而增大）
    steps = [0, 1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400]
    steps = [s for s in steps if s <= max_prune]
    
    for i, step in enumerate(steps):
        prune_counts.append(step)
        
        # 重置模型的屏蔽神经元
        masked_model = MaskedModelWrapper(model)
        
        # 添加神经元屏蔽
        for j in range(step):
            masked_model.add_masked_neuron(top_neurons[j])
        
        # 测试主任务准确率
        main_acc = get_main_task_accuracy(masked_model, test_loader)
        main_task_accs.append(main_acc)
        
        # 测试原始触发器的攻击成功率
        orig_asr = get_attack_success_rate(masked_model, test_loader, 'original')
        orig_success_rates.append(orig_asr)
        
        # 测试逆向触发器的攻击成功率
        rev_asr = get_attack_success_rate(masked_model, test_loader, 'reverse')
        rev_success_rates.append(rev_asr)
        
        print(f"Pruning {step} neurons | Main acc: {main_acc:.2%} | "
              f"Original ASR: {orig_asr:.2%} | Reverse ASR: {rev_asr:.2%}")
        
    return prune_counts, main_task_accs, orig_success_rates, rev_success_rates, sorted_neurons

# 分析神经元激活差异的分布
def analyze_neuron_distribution(clean, orig, rev):
    # 计算干净样本和触发样本的激活差异
    delta_orig = orig.mean(dim=0) - clean.mean(dim=0)
    delta_rev = rev.mean(dim=0) - clean.mean(dim=0)
    
    # 计算绝对差异值并排序
    delta_orig_abs = torch.abs(delta_orig)
    delta_rev_abs = torch.abs(delta_rev)
    
    sorted_orig_abs, _ = torch.sort(delta_orig_abs, descending=True)
    sorted_rev_abs, _ = torch.sort(delta_rev_abs, descending=True)
    
    # 计算所有神经元的绝对差异总和
    total_orig = delta_orig_abs.sum().item()
    total_rev = delta_rev_abs.sum().item()
    
    # 获取前N个神经元的累积重要性
    percentages = [0.01, 0.05, 0.1, 0.2, 0.5]
    importance_orig = {}
    importance_rev = {}
    
    for p in percentages:
        n = int(len(delta_orig) * p)
        if n == 0:
            n = 1
        importance_orig[p] = sorted_orig_abs[:n].sum().item() / total_orig
        importance_rev[p] = sorted_rev_abs[:n].sum().item() / total_rev
    
    print("\n=== Neuron Importance Analysis ===")
    print("| Top N% Neurons | Importance for Original | Importance for Reverse |")
    print("|----------------|-------------------------|------------------------|")
    for p in percentages:
        print(f"| Top {p*100:.0f}% | {importance_orig[p]:.2%} | {importance_rev[p]:.2%} |")

    return delta_orig, delta_rev

# 主流程
print("\n=== Starting Neuron Pruning Defense Evaluation ===")

# 生成特征激活
print("\n=== Generating Feature Activations ===")
clean, rev, orig = generate_inputs(model, clean_loader)
print(f"Clean sample count: {clean.size(0)}")
print(f"Feature dimension: {clean.size(1)}")

# 分析神经元分布
delta_orig, delta_rev = analyze_neuron_distribution(clean, orig, rev)

# 提取关键神经元 - 使用绝对差异
print("\n=== Analyzing Critical Neurons ===")
top_orig_neurons, _ = analyze_critical_neurons(clean, orig)
top_rev_neurons, _ = analyze_critical_neurons(clean, rev)

# 检查重叠
overlap = len(set(top_orig_neurons.numpy()) & set(top_rev_neurons.numpy()))
overlap_ratio = overlap / len(top_orig_neurons)
print(f"Critical neuron overlap ratio: {overlap_ratio:.2%}")

feature_dim = clean.size(1)  # 特征维度为1280
max_prune = int(feature_dim) 
print(f"Maximum pruning: {max_prune} neurons ({max_prune/feature_dim:.1%} of total {feature_dim})")

# 修剪基于原始触发器的关键神经元
print("\n=== Original Trigger-based Neuron Pruning ===")
prune_counts, main_task_accs, orig_success_rates, rev_success_rates, sorted_neurons = evaluate_neuron_pruning_effect(
    model, delta_orig, clean_loader, max_prune=max_prune
)

# 保存结果
results = {
    'prune_counts': prune_counts,
    'main_task_accs': main_task_accs,
    'orig_success_rates': orig_success_rates,
    'rev_success_rates': rev_success_rates
}

# 绘制修剪效果图
plt.figure(figsize=(10, 6))
plt.plot(prune_counts, main_task_accs, 'k-', marker='o', label='Main Task Accuracy', linewidth=2)
plt.plot(prune_counts, orig_success_rates, 'r--', marker='x', label='Original Trigger ASR', linewidth=2)
plt.plot(prune_counts, rev_success_rates, 'b--', marker='^', label='Reverse Trigger ASR', linewidth=2)

plt.xlabel('Number of Pruned Neurons', fontsize=14)
plt.ylabel('Accuracy / Success Rate', fontsize=14)
plt.title('Impact of Neuron Pruning on Model Performance', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='best', fontsize=12)
plt.tight_layout()
plt.savefig(f'{results_dir}/pruning_effect.png', dpi=300, bbox_inches='tight')
plt.close()

# 找出最佳修剪点（保持主任务准确率下降小于5%，同时使攻击成功率最低）
baseline_acc = main_task_accs[0]  # 初始主任务准确率
acc_threshold = baseline_acc * 0.95  # 允许的最低准确率（下降不超过5%）

best_idx = 0
best_defense = orig_success_rates[0]  # 初始化为第一个值
for i, acc in enumerate(main_task_accs):
    if acc >= acc_threshold:
        # 如果攻击成功率低于当前最佳，并且主任务准确率可接受
        if orig_success_rates[i] < best_defense:
            best_idx = i
            best_defense = orig_success_rates[i]

best_prune = prune_counts[best_idx]
print(f"\n=== Best Pruning Strategy ===")
print(f"Pruning count: {best_prune} neurons")
print(f"Main task accuracy: {main_task_accs[best_idx]:.2%} (decrease: {baseline_acc - main_task_accs[best_idx]:.2%})")
print(f"Original trigger attack success rate: {orig_success_rates[best_idx]:.2%}")
print(f"Reverse trigger attack success rate: {rev_success_rates[best_idx]:.2%}")

# 导出最佳修剪方案
best_pruned_neurons = sorted_neurons[:best_prune].tolist()

with open(f'{results_dir}/pruning_defense.txt', 'w') as f:
    f.write(f"Best Pruning Strategy\n")
    f.write(f"Pruning count: {best_prune} neurons\n")
    f.write(f"Main task accuracy: {main_task_accs[best_idx]:.2%} (decrease: {baseline_acc - main_task_accs[best_idx]:.2%})\n")
    f.write(f"Original trigger attack success rate: {orig_success_rates[best_idx]:.2%}\n")
    f.write(f"Reverse trigger attack success rate: {rev_success_rates[best_idx]:.2%}\n\n")
    f.write(f"Pruned neuron indices:\n{best_pruned_neurons}")

print(f"\nEvaluation complete! Results saved to '{results_dir}/pruning_defense.txt'")
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
print(f"使用设备: {device}")

# # 设置中文字体和负号显示
# plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei', 'WenQuanYi Micro Hei']  # 优先使用的中文字体
# plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

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
print(f"类别名称: {class_names}")

# 定义模型类（参考mnist方式设置钩子）
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
        
        # 直接注册钩子 - 这里针对倒数第二层特征
        self.avgpool.register_forward_hook(
            lambda module, input_, output: self.activation.__setitem__('second_last', 
                                                                      output.view(output.size(0), -1))
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

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

# 确定后门目标标签
suspected_label = 1
print(f"后门目标标签: {suspected_label} ({class_names[suspected_label]})")

# 加载逆向工程的触发器和mask
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)
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
            img_orig = add_original_trigger(imgs).to(device)
            _ = model(img_orig)
            orig_activations.append(model.activation['second_last'].clone())
    
    # 合并所有样本的激活 
    clean = torch.cat(clean_activations, dim=0)
    rev = torch.cat(rev_activations, dim=0) 
    orig = torch.cat(orig_activations, dim=0)
    
    return clean, rev, orig

# 分析关键神经元
def analyze_critical_neurons(clean, triggered, percentile=0.01):
    # 计算干净样本和触发样本之间的激活差异
    delta = triggered.mean(dim=0) - clean.mean(dim=0)
    
    # 按差异排序取Top 1%
    topk = max(1, int(delta.size(0) * percentile))
    top_neurons = torch.topk(delta, topk).indices
    
    return top_neurons.cpu()

# 计算误报率（FPR）和漏报率（FNR）
def calculate_rates(predictions, labels):
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    return fpr, fnr

# 生成过滤器的FPR-FNR曲线
def evaluate_filter(clean_activations, trigger_activations, curve_name):
    min_val = min(clean_activations.min(), trigger_activations.min())
    max_val = max(clean_activations.max(), trigger_activations.max())
    thresholds = np.linspace(min_val, max_val, 100)
    fpr_list, fnr_list = [], []
    
    for threshold in thresholds:
        # 预测是否为触发样本
        clean_predictions = clean_activations > threshold
        trigger_predictions = trigger_activations > threshold
        
        # 计算FPR和FNR
        fpr, fnr = calculate_rates(
            np.concatenate([clean_predictions, trigger_predictions]),
            np.concatenate([np.zeros_like(clean_activations), np.ones_like(trigger_activations)])
        )
        fpr_list.append(fpr)
        fnr_list.append(fnr)
    
    # 寻找最佳阈值（FPR ≈ 0.05 附近）
    target_fpr = 0.05
    idx = np.argmin(np.abs(np.array(fpr_list) - target_fpr))
    best_threshold = thresholds[idx]
    best_fnr = fnr_list[idx]
    
    print(f"\n{curve_name} 过滤器性能:")
    print(f"目标FPR: {target_fpr:.2f}")
    print(f"最佳阈值: {best_threshold:.4f}")
    print(f"对应FNR: {best_fnr:.4f}")
    
    return thresholds, fpr_list, fnr_list, best_threshold, best_fnr

# 可视化样本示例
def visualize_samples(clean_loader, reverse_mask, reverse_delta):
    # 获取一个干净样本
    for imgs, _ in clean_loader:
        img = imgs[0].to(device)
        break
    
    # 创建三种类型的样本
    clean_img = img.clone()
    
    # 应用逆向触发器
    rev_img = (1 - reverse_mask) * img + reverse_mask * reverse_delta
    
    # 应用原始触发器
    orig_img = img.clone()
    # orig_img[:, -20:, -20:] = 1.0  # 右下角白色方块
    
    # 反归一化
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
    
    clean_vis = clean_img * std + mean
    rev_vis = rev_img * std + mean
    orig_vis = orig_img * std + mean
    orig_vis[:, -20:, -20:] = 1.0  # 右下角白色方块
    
    # 转为numpy便于可视化
    clean_np = clean_vis.cpu().permute(1, 2, 0).numpy()
    rev_np = rev_vis.cpu().permute(1, 2, 0).numpy()
    orig_np = orig_vis.cpu().permute(1, 2, 0).numpy()
    
    # 拼接为对比图
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(np.clip(clean_np, 0, 1))
    plt.title("clean sample")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(np.clip(rev_np, 0, 1))
    plt.title("reverse trigger")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(np.clip(orig_np, 0, 1))
    plt.title("original trigger")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/trigger_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

# 主流程
print("\n=== 开始后门防御评估 ===")

# 可视化样本对比
visualize_samples(clean_loader, reverse_mask, reverse_delta)

print("\n=== 生成特征激活 ===")
clean, rev, orig = generate_inputs(model, clean_loader)
print(f"干净样本数量: {clean.size(0)}")
print(f"逆向触发样本数量: {rev.size(0)}")
print(f"原始触发样本数量: {orig.size(0)}")
print(f"特征维度: {clean.size(1)}")

# 对逆向触发器提取关键神经元
print("\n=== 分析逆向触发器的关键神经元 ===")
top_rev_neurons = analyze_critical_neurons(clean, rev)
print(f"逆向触发器关键神经元数量: {len(top_rev_neurons)}")

# 对原始触发器提取关键神经元
print("\n=== 分析原始触发器的关键神经元 ===")
top_orig_neurons = analyze_critical_neurons(clean, orig)
print(f"原始触发器关键神经元数量: {len(top_orig_neurons)}")

# 检查神经元重叠
overlap = len(set(top_rev_neurons.numpy()) & set(top_orig_neurons.numpy()))
overlap_ratio = overlap / len(top_rev_neurons)
print(f"\n关键神经元重叠数量: {overlap} / {len(top_rev_neurons)}")
print(f"重叠比例: {overlap_ratio:.2%}")

# 计算各种触发器在不同神经元集上的激活强度
print("\n=== 关键神经元激活强度 ===")
print("| 神经元集合 | 样本类型 | 平均激活值 |")
print("|------------|----------|----------|")

# 在原始触发器神经元上的激活
avg_clean_on_orig = clean[:, top_orig_neurons].mean().item()
avg_rev_on_orig = rev[:, top_orig_neurons].mean().item()
avg_orig_on_orig = orig[:, top_orig_neurons].mean().item()

print(f"| 原始触发器神经元 | 干净样本 | {avg_clean_on_orig:.4f} |")
print(f"| 原始触发器神经元 | 逆向样本 | {avg_rev_on_orig:.4f} |")
print(f"| 原始触发器神经元 | 原始样本 | {avg_orig_on_orig:.4f} |")

# 在逆向触发器神经元上的激活
avg_clean_on_rev = clean[:, top_rev_neurons].mean().item()
avg_rev_on_rev = rev[:, top_rev_neurons].mean().item()
avg_orig_on_rev = orig[:, top_rev_neurons].mean().item()

print(f"| 逆向触发器神经元 | 干净样本 | {avg_clean_on_rev:.4f} |")
print(f"| 逆向触发器神经元 | 逆向样本 | {avg_rev_on_rev:.4f} |")
print(f"| 逆向触发器神经元 | 原始样本 | {avg_orig_on_rev:.4f} |")

# 提取每个样本在关键神经元上的平均激活值
# 基于原始触发器的关键神经元
clean_act_orig = clean[:, top_orig_neurons].mean(dim=1).cpu().numpy()
rev_act_orig = rev[:, top_orig_neurons].mean(dim=1).cpu().numpy()
orig_act_orig = orig[:, top_orig_neurons].mean(dim=1).cpu().numpy()

# 基于逆向触发器的关键神经元
clean_act_rev = clean[:, top_rev_neurons].mean(dim=1).cpu().numpy()
rev_act_rev = rev[:, top_rev_neurons].mean(dim=1).cpu().numpy()
orig_act_rev = orig[:, top_rev_neurons].mean(dim=1).cpu().numpy()

# 生成FPR-FNR曲线
# 1. 原始触发器神经元检测原始触发器
# _, fpr_orig_orig, fnr_orig_orig, _, fnr_at_fpr5_orig_orig = evaluate_filter(
#     clean_act_orig, orig_act_orig, "原始触发器神经元检测原始触发器")

# # 2. 原始触发器神经元检测逆向触发器
# _, fpr_orig_rev, fnr_orig_rev, _, fnr_at_fpr5_orig_rev = evaluate_filter(
#     clean_act_orig, rev_act_orig, "原始触发器神经元检测逆向触发器")

# 3. 逆向触发器神经元检测原始触发器
_, fpr_rev_orig, fnr_rev_orig, _, fnr_at_fpr5_rev_orig = evaluate_filter(
    clean_act_rev, orig_act_rev, "逆向触发器神经元检测原始触发器")

# 4. 逆向触发器神经元检测逆向触发器
_, fpr_rev_rev, fnr_rev_rev, _, fnr_at_fpr5_rev_rev = evaluate_filter(
    clean_act_rev, rev_act_rev, "逆向触发器神经元检测逆向触发器")

# 绘制完整的FPR-FNR曲线
plt.figure(figsize=(12, 8))

# plt.plot(fpr_orig_orig, fnr_orig_orig, 'r-', linewidth=2, label='Original neurons - Original trigger')
# plt.plot(fpr_orig_rev, fnr_orig_rev, 'r--', linewidth=2, label='Original neurons - Reverse trigger')
plt.plot(fpr_rev_orig, fnr_rev_orig, 'b--', linewidth=2, label='Reverse neurons - Original trigger')
plt.plot(fpr_rev_rev, fnr_rev_rev, 'b-', linewidth=2, label='Reverse neurons - Reverse trigger')

# 标记FPR=0.05处的点
target_fpr = 0.3
plt.axvline(x=target_fpr, color='gray', linestyle='--', alpha=0.5)
plt.text(target_fpr + 0.01, 0.5, f'FPR = {target_fpr}', rotation=90, alpha=0.7)

plt.xlabel('False Positive Rate (FPR)', fontsize=14)
plt.ylabel('False Negative Rate (FNR)', fontsize=14)
plt.title('FPR-FNR Curves', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='best', fontsize=12)
plt.xlim(-0.01, 1.01)
plt.ylim(-0.01, 1.01)
plt.savefig(f'{results_dir}/combined_fpr_fnr_curves.png', dpi=300, bbox_inches='tight')
plt.close()

# 打印FPR=0.05时的结果汇总
print("\n=== FPR=5%时的漏报率(FNR)汇总 ===")
print("| 过滤器类型 | 对原始触发器的FNR | 对逆向触发器的FNR |")
print("|------------|-------------------|-------------------|")
# print(f"| 原始触发器神经元 | {fnr_at_fpr5_orig_orig:.4f} | {fnr_at_fpr5_orig_rev:.4f} |")
print(f"| 逆向触发器神经元 | {fnr_at_fpr5_rev_orig:.4f} | {fnr_at_fpr5_rev_rev:.4f} |")

# 保存结果
with open(f'{results_dir}/filter_test_results.txt', 'w') as f:
    f.write("=== 后门防御评估结果 ===\n\n")
    f.write(f"后门目标标签: {suspected_label} ({class_names[suspected_label]})\n")
    f.write(f"关键神经元重叠比例: {overlap_ratio:.2%}\n\n")
    
    f.write("神经元激活强度:\n")
    f.write(f"原始触发器神经元上，干净样本: {avg_clean_on_orig:.4f}\n")
    f.write(f"原始触发器神经元上，逆向样本: {avg_rev_on_orig:.4f}\n")
    f.write(f"原始触发器神经元上，原始样本: {avg_orig_on_orig:.4f}\n\n")
    
    f.write(f"逆向触发器神经元上，干净样本: {avg_clean_on_rev:.4f}\n")
    f.write(f"逆向触发器神经元上，逆向样本: {avg_rev_on_rev:.4f}\n")
    f.write(f"逆向触发器神经元上，原始样本: {avg_orig_on_rev:.4f}\n\n")
    
    f.write("在FPR=5%时的漏报率(FNR):\n")
    # f.write(f"原始触发器神经元检测原始触发器: {fnr_at_fpr5_orig_orig:.4f}\n")
    # f.write(f"原始触发器神经元检测逆向触发器: {fnr_at_fpr5_orig_rev:.4f}\n")
    f.write(f"逆向触发器神经元检测原始触发器: {fnr_at_fpr5_rev_orig:.4f}\n")
    f.write(f"逆向触发器神经元检测逆向触发器: {fnr_at_fpr5_rev_rev:.4f}\n")

print("\n评估完成! 结果已保存到 'results/filter_test_results.txt' 和图表文件")
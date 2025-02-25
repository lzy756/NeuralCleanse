from dataload import testset
from model import BadNetMNIST
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from model_wrapper import MaskedModelWrapper

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载逆向工程的触发器和mask
reverse_mask = torch.load("mask_label0.pth").to(device)
reverse_delta = torch.load("delta_label0.pth").to(device)

# 生成三种输入类型的数据（干净样本 + 对抗样本）
def generate_inputs(model, test_loader):
    clean_activations, rev_activations, orig_activations = [], [], []
    
    with torch.no_grad():
        for imgs, _ in test_loader:
            imgs = imgs.to(device)
            
            # 干净输入
            _ = model(imgs)
            clean_activations.append(model.activation['second_last'].clone())

            # 逆向触发器输入
            img_rev = (1 - reverse_mask) * imgs + reverse_mask * reverse_delta
            _ = model(img_rev)
            rev_activations.append(model.activation['second_last'].clone())
            
            # 原始触发器输入（右下角4x4全白）
            img_orig = imgs.clone()
            img_orig[:, :, -4:, -4:] = 1.0
            _ = model(img_orig)
            orig_activations.append(model.activation['second_last'].clone())
    
    # 合并所有样本的激活 -> [n_samples, 64]
    clean = torch.cat(clean_activations, dim=0)
    rev = torch.cat(rev_activations, dim=0) 
    orig = torch.cat(orig_activations, dim=0)
    return clean, rev, orig

# 分析关键神经元并计算激活强度
def analyze_critical_neurons(clean, rev, orig):
    delta_rev = rev.mean(dim=0) - clean.mean(dim=0)
    delta_orig = orig.mean(dim=0) - clean.mean(dim=0)
    
    # 按差异排序取Top 1%
    topk = max(1, int(len(delta_rev)*0.01))
    top_rev = torch.topk(delta_rev, topk).indices
    top_orig = torch.topk(delta_orig, topk).indices
    return top_rev.cpu(), top_orig.cpu()

def compute_avg_activation(clean, rev, orig, top_neurons):
    avg_clean = clean[:, top_neurons].mean().item()
    avg_rev = rev[:, top_neurons].mean().item()
    avg_orig = orig[:, top_neurons].mean().item()
    return avg_clean, avg_rev, avg_orig

# 计算误报率（FPR）和漏报率（FNR）
def calculate_rates(predictions, labels):
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    return fpr, fnr

def get_main_task_accuracy(model, test_loader):
    """获取主任务准确率"""
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

def get_attack_success_rate(model, test_loader, trigger_type='original'):
    """测试后门攻击成功率"""
    correct = 0
    total = 0
    target_label = 0  # 假设攻击目标标签是0
    
    with torch.no_grad():
        for imgs, _ in test_loader:
            imgs = imgs.to(device)
            
            # 应用触发器
            if trigger_type == 'original':
                imgs[:, :, -4:, -4:] = 1.0  # 原始触发器
            elif trigger_type == 'reverse':
                imgs = (1 - reverse_mask) * imgs + reverse_mask * reverse_delta  # 逆向引擎触发器
            
            outputs = model(imgs)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == target_label).sum().item()
            total += imgs.size(0)
            
    return correct / total

def evaluate_neuron_pruning_effect(model, orig_deltas, test_loader):
    """
    逐步修剪关键神经元并记录攻击成功率
    :param orig_deltas: 各神经元激活差异 (torch.Tensor [64])
    :param test_loader: 测试数据集
    :return: 保留的神经元数量列表，对应的攻击成功率列表
    """
    # 获取神经元屏蔽顺序（按差异从大到小排序）
    sorted_neurons = torch.argsort(orig_deltas, descending=True).cpu().numpy()
    print(f"关键神经元屏蔽顺序（从最重要到最不重要）: {sorted_neurons}")
    
    # 初始化包装器
    masked_model = MaskedModelWrapper(model)
    
    # 记录数据结构
    num_neurons = len(sorted_neurons)
    keep_counts = []
    success_rates = []
    main_task_scrates = []
    resuccess_rates = []

    # 逐步修剪测试
    for step in range(num_neurons + 1):
        # 当前保留的神经元数量
        current_keep = num_neurons - step
        keep_counts.append(step)
        
        # 测试攻击成功率
        if step > 0:
            masked_model.add_masked_neuron(sorted_neurons[step-1])
        asr = get_attack_success_rate(masked_model.model, test_loader)
        success_rates.append(asr)
        asr1 = get_main_task_accuracy(masked_model.model, test_loader)
        asr2 = get_attack_success_rate(masked_model.model, test_loader, trigger_type='reverse')
        main_task_scrates.append(asr1)
        resuccess_rates.append(asr2)


        print(f"屏蔽 {step} 个关键神经元 | 保留 {current_keep} 个 | 攻击成功率: {asr:.2%} | 主任务准确率: {asr1:.2%} | 逆向触发器攻击成功率: {asr2:.2%}")
        
        
    return keep_counts, success_rates, main_task_scrates, resuccess_rates

# 生成过滤器的FPR-FNR曲线
def evaluate_filter(clean_activations, trigger_activations):
    thresholds = np.linspace(clean_activations.min(), trigger_activations.max(), 100)
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
    
    return thresholds, fpr_list, fnr_list

# 主流程
model = BadNetMNIST().to(device)
model.load_state_dict(torch.load("badnet_mnist.pth", map_location=device))
model.eval()

test_subset = torch.utils.data.Subset(testset, indices=range(1000))
test_loader = DataLoader(test_subset, batch_size=100)

# 生成输入
clean, rev, orig = generate_inputs(model, test_loader)

# 分析神经元
top_rev, top_orig = analyze_critical_neurons(clean, rev, orig)

# 检查重叠
overlap = len(set(top_rev.numpy()) & set(top_orig.numpy()))
print(f"关键神经元重叠比例: {overlap/len(top_rev):.2%}")

# 计算激活强度（基于原触发器的关键神经元）
avg_clean, avg_rev, avg_orig = compute_avg_activation(
    clean.cpu(), rev.cpu(), orig.cpu(), top_orig
)

# 获取原始触发器的神经元激活差异
delta_orig = orig.mean(dim=0) - clean.mean(dim=0)

# 执行逐步修剪实验
keep_counts, success_rates, mainscrates, resuccess_rates = evaluate_neuron_pruning_effect(
    model, 
    delta_orig, 
    test_loader
)

# 在现有绘图代码前加入以下绘图代码
plt.figure(figsize=(8, 5))
plt.plot(keep_counts, mainscrates, linestyle='-', marker='x', color='#1F77B4', linewidth=1.2, markersize=6)
plt.plot(keep_counts, success_rates, linestyle='--', marker='x', color='#FF2F17', linewidth=1.2, markersize=6)
plt.plot(keep_counts, resuccess_rates, linestyle='-.', marker='x', color='#2CA02C', linewidth=1.2, markersize=6)
plt.legend(["主任务准确率", "攻击成功率", "逆向触发器攻击成功率"], loc="upper right") 
plt.xlabel("屏蔽的关键神经元数量")
plt.ylabel("成功率")
plt.title("关键神经元修剪对攻击成功率的影响")
plt.grid(True)
# plt.gca().invert_xaxis()  # x轴逆向显示（剩余神经元从多到少）
plt.show()

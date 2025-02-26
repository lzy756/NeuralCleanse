from dataload import testset
from model import BadNetMNIST
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

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

print("\n=== 神经元激活强度（表III对应项） ===")
print(" | Clean | Reverse Trigger | Original Trigger |")
print("|---|---|---|")
print(f"| {avg_clean:.2f} | {avg_rev:.2f} | {avg_orig:.2f} |")

# 评估过滤器性能
clean_activations = clean[:, top_orig].mean(dim=1).cpu().numpy()  # 干净样本的激活值
trigger_activations = orig[:, top_orig].mean(dim=1).cpu().numpy()  # 原始触发样本的激活值

# 生成FPR-FNR曲线
thresholds, fpr_list, fnr_list = evaluate_filter(clean_activations, trigger_activations)

# 绘制曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr_list, fnr_list, label="Original Trigger")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("False Negative Rate (FNR)")
plt.title("FPR vs FNR Curve")
plt.grid(True)
plt.legend()
plt.show()

# 计算FPR=5%时的FNR
target_fpr = 0.05
idx = np.argmin(np.abs(np.array(fpr_list) - target_fpr))
target_fnr = fnr_list[idx]
print(f"\n在FPR={target_fpr*100:.0f}%时，FNR={target_fnr*100:.2f}%")

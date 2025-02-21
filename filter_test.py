from dataload import testset
from model import BadNetMNIST
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

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

def extract_activation_profiles(model, top_neurons, imgs):
    """
    提取输入样本在关键神经元上的平均激活值
    """
    activations = []
    for img in imgs:
        img = img.unsqueeze(0)
        with torch.no_grad():
            _ = model(img)
        layer_activation = model.activation['second_last'].squeeze()  # [64]
        key_activation = layer_activation[top_neurons].mean().item()   # 关键神经元平均激活
        activations.append(key_activation)
    return activations

def calculate_threshold(clean_activations, fpr_target=0.05):
    """
    根据目标FPR计算激活阈值
    """
    # 按升序排序干净样本的激活值
    sorted_activations = np.sort(clean_activations)
    
    # 找到对应目标FPR的分位点
    index = int(len(sorted_activations) * (1 - fpr_target))
    threshold = sorted_activations[index]
    
    return threshold

# 主流程
model = BadNetMNIST().to(device)
model.load_state_dict(torch.load("badnet_mnist.pth",map_location=device))
model.eval()

test_subset = torch.utils.data.Subset(testset, indices=range(1000))
test_loader = DataLoader(test_subset, batch_size=100)

# 生成输入
clean, rev, orig = generate_inputs(model, test_loader)

# 分析神经元
top_rev, top_orig = analyze_critical_neurons(clean, rev, orig)

clean_imgs = []
for batch in test_loader:
    clean_imgs.extend(batch[0])

original_trigger_imgs = clean_imgs.copy()
for img in original_trigger_imgs:
    img[:, -4:, -4:] = 1.0

reverse_trigger_imgs = []
for img in clean_imgs:
    img_rev = (1 - reverse_mask) * img + reverse_mask * reverse_delta
    reverse_trigger_imgs.append(img_rev)

clean_activations = extract_activation_profiles(model, top_orig, clean_imgs)
original_trigger_activations = extract_activation_profiles(model, top_orig, original_trigger_imgs)
reverse_trigger_activations = extract_activation_profiles(model, top_orig, reverse_trigger_imgs)

# 计算目标FPR为5%时的阈值
threshold_fpr_5 = calculate_threshold(clean_activations, fpr_target=0.05)
print(f"Threshold for FPR=5%: {threshold_fpr_5:.4f}")

# # 检查重叠
# overlap = len(set(top_rev.numpy()) & set(top_orig.numpy()))
# print(f"关键神经元重叠比例: {overlap/len(top_rev):.2%}")

# # 计算激活强度（基于原触发器的关键神经元）
# avg_clean, avg_rev, avg_orig = compute_avg_activation(
#     clean.cpu(), rev.cpu(), orig.cpu(), top_orig
# )

# print("\n=== 神经元激活强度（表III对应项） ===")
# print(" | Clean | Reverse Trigger | Original Trigger |")
# print("|---|---|---|")
# print(f"| {avg_clean:.2f} | {avg_rev:.2f} | {avg_orig:.2f} |")

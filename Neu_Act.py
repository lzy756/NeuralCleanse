from dataload import testset
from model import BadNetMNIST
import torch
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

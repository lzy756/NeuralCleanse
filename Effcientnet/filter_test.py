import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from sklearn.metrics import confusion_matrix

# 配置
data_dir = "data/defect_supervised/glass-insulator"
batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据转换（与训练/逆向工程相同）
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 加载验证集
clean_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transform)
clean_loader = DataLoader(clean_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
class_names = clean_dataset.classes

# 加载后门模型
model = efficientnet_v2_s(weights=None)
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True),
    torch.nn.Linear(in_features=1280, out_features=len(class_names))
)
model.load_state_dict(torch.load("Effcientnet/glass_insulator_efficientnetv2_backdoored.pth", map_location=device))
model.to(device)
model.eval()

# 注册钩子获取倒数第二层特征
model.activation = {}
def hook(module, input, output):
    # input[0]: 特征向量 [batch, dim]
    model.activation['second_last'] = input[0].detach().clone()
# classifier[1] 是线性层前向输入
model.classifier[1].register_forward_hook(hook)

# 加载逆向工程的触发器和mask（选择目标标签1）
reverse_mask = torch.load("Effcientnet/results/mask_label1.pth").to(device)
reverse_delta = torch.load("Effcientnet/results/delta_label1.pth").to(device)
target_label = 1

# 生成三种输入类型的数据：干净、逆向触发、原始触发

def generate_inputs(model, data_loader):
    clean_feats, rev_feats, orig_feats = [], [], []
    with torch.no_grad():
        for imgs, _ in data_loader:
            imgs = imgs.to(device)
            # 干净
            _ = model(imgs)
            clean_feats.append(model.activation['second_last'])
            # 逆向触发
            img_rev = (1 - reverse_mask) * imgs + reverse_mask * reverse_delta
            _ = model(img_rev)
            rev_feats.append(model.activation['second_last'])
            # 原始触发（右下20x20白块）
            img_orig = imgs.clone()
            _, _, H, W = img_orig.shape
            img_orig[:, :, H-20:H, W-20:W] = 1.0
            _ = model(img_orig)
            orig_feats.append(model.activation['second_last'])
    clean = torch.cat(clean_feats, dim=0)
    rev = torch.cat(rev_feats, dim=0)
    orig = torch.cat(orig_feats, dim=0)
    return clean, rev, orig

# 计算关键神经元并排序Top 1%

def analyze_critical_neurons(clean, rev, orig):
    delta_rev = rev.mean(dim=0) - clean.mean(dim=0)
    delta_orig = orig.mean(dim=0) - clean.mean(dim=0)
    topk = max(1, int(len(delta_rev) * 0.01))
    top_rev = torch.topk(delta_rev, topk).indices
    top_orig = torch.topk(delta_orig, topk).indices
    return top_rev.cpu().tolist(), top_orig.cpu().tolist()

# 计算平均激活

def compute_avg_activation(clean, rev, orig, neurons):
    avg_clean = clean[:, neurons].mean().item()
    avg_rev = rev[:, neurons].mean().item()
    avg_orig = orig[:, neurons].mean().item()
    return avg_clean, avg_rev, avg_orig

# 计算FPR和FNR

def calculate_rates(preds, labels):
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    return fpr, fnr

# 生成FPR-FNR曲线

def evaluate_filter(clean_acts, trigger_acts):
    thresholds = np.linspace(clean_acts.min(), trigger_acts.max(), 100)
    fpr_list, fnr_list = [], []
    for thr in thresholds:
        c_pred = clean_acts > thr
        t_pred = trigger_acts > thr
        preds = np.concatenate([c_pred, t_pred])
        labels = np.concatenate([np.zeros_like(c_pred), np.ones_like(t_pred)])
        fpr, fnr = calculate_rates(preds.astype(int), labels.astype(int))
        fpr_list.append(fpr)
        fnr_list.append(fnr)
    return thresholds, fpr_list, fnr_list

# 主执行流程
if __name__ == "__main__":
    clean, rev, orig = generate_inputs(model, clean_loader)
    top_rev, top_orig = analyze_critical_neurons(clean, rev, orig)
    overlap = len(set(top_rev) & set(top_orig)) / len(top_orig)
    print(f"关键神经元重叠比例: {overlap:.2%}")

    avg_clean, avg_rev, avg_orig = compute_avg_activation(
        clean.cpu(), rev.cpu(), orig.cpu(), top_orig
    )
    print("\n=== 神经元激活强度 ===")
    print(" | Clean | Reverse Trigger | Original Trigger |")
    print("|---|---|---|---|")
    print(f"| {avg_clean:.2f} | {avg_rev:.2f} | {avg_orig:.2f} |")

    clean_acts = clean[:, top_orig].mean(dim=1).cpu().numpy()
    trigger_acts = orig[:, top_orig].mean(dim=1).cpu().numpy()
    thresholds, fpr_list, fnr_list = evaluate_filter(clean_acts, trigger_acts)

    # 绘图
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_list, fnr_list, label="Original Trigger")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("False Negative Rate (FNR)")
    plt.title("FPR vs FNR Curve - EfficientNet")
    plt.axvline(x=0.1, color='gray', linestyle='--')
    plt.text(0.11, 0.25, 'FPR=10%', color='gray', fontsize=10, rotation=90)
    plt.grid(True)
    plt.legend()
    plt.savefig('fpr_fnr_efficientnet.png')
    plt.show()

    # FPR=5%时的FNR
    idx = np.argmin(np.abs(np.array(fpr_list) - 0.05))
    print(f"\n在FPR=5%时，FNR={fnr_list[idx]*100:.2f}%")

import torch
import math
import statistics
from pathlib import Path

print("==== 对 Effecientnet 的 Neural Cleanse 结果进行异常检测 ====")

# 确定类别数
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
preferred_results_dir = PROJECT_ROOT / "results"
legacy_results_dir = SCRIPT_DIR / "results"
num_classes = 2  # 按照您的代码中的配置设置为2

# 收集每个标签的 mask L1 范数
masks_l1 = []
for i in range(num_classes):
    preferred_mask_path = preferred_results_dir / f"mask_label{i}.pth"
    legacy_mask_path = legacy_results_dir / f"mask_label{i}.pth"
    mask_path = preferred_mask_path if preferred_mask_path.exists() else legacy_mask_path
    if mask_path.exists():
        mask = torch.load(mask_path)
        l1_norm = mask.sum().item()
        masks_l1.append(l1_norm)
    else:
        print(f"警告：未找到标签 {i} 的 mask 文件")

print(f"每个标签的 mask L1 范数: {masks_l1}")

# 计算 MAD 异常分数
# 步骤 1: 找出与中位数的偏差
median = statistics.median(masks_l1)
deviations = [abs(l1 - median) for l1 in masks_l1]

# 步骤 2: 计算偏差的中位数 (MAD)
mad = statistics.median(deviations)

# 步骤 3: 计算异常分数 (使用 MAD 作为度量)
anomaly_index = []
for i in range(len(masks_l1)):
    # 使用 1.4826 因子将 MAD 转换为相当于高斯分布的标准差
    anomaly_score = deviations[i] / (1.4826 * mad) if mad > 0 else 0
    anomaly_index.append(anomaly_score)

# 输出结果
print("\n==== 异常检测结果 ====")
print(f"中位数 L1 范数: {median:.2f}")
print(f"MAD: {mad:.2f}")

for i, score in enumerate(anomaly_index):
    print(f"标签 {i} 的异常分数: {score:.2f}", end="")
    if score > 2:
        print(f" ← 可能的后门触发器!")
    else:
        print("")

# 判断哪个标签最可能是后门目标
if max(anomaly_index) > 2:
    most_likely_backdoor = anomaly_index.index(max(anomaly_index))
    print(f"\n结论: 标签 {most_likely_backdoor} 最可能是后门目标")
else:
    print("\n结论: 未发现明显的后门")

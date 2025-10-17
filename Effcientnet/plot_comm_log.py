import matplotlib.pyplot as plt
import numpy as np

# === 字体大小设置 ==========================================================
FONT_SIZE = 18

# === 数据（MiB）==============================================================
models   = ['EffNetV2-S', 'EffNetV2-M', 'EffNetV2-L']
methods  = ['Ours', 'BackdoorIndicator', 'CrowdGuard']
overhead = np.array([
    [3.38,   5.27,   5.27],    # ours
    [81.86, 206.53, 452.10],   # BackdoorIndicator
    [7.89,   7.89,   7.89]     # CrowdGuard
])

# === 画布参数 ===============================================================
bar_w   = 0.25
colors  = ['#1f77b4', '#ff7f0e', '#2ca02c']       # 蓝 / 橙 / 绿
x_pos   = np.arange(len(models))                  # x 轴刻度位置

fig, ax = plt.subplots(figsize=(10, 6))

# === 绘制柱形（对数 y 轴）====================================================
for i, m in enumerate(methods):
    ax.bar(x_pos + (i - 1)*bar_w,
           overhead[i],
           width=bar_w,
           label=m,
           color=colors[i])

# === 坐标轴 & 标注格式 =======================================================
ax.set_yscale('log')                                   # 关键：log 轴
ax.set_ylabel('Communication per Round (MiB)', fontsize=FONT_SIZE)
ax.set_title('Per-round Communication Overhead (log$_{10}$ scale)', fontsize=FONT_SIZE, pad=12)

# 统一刻度字体大小
ax.set_xticks(x_pos)
ax.set_xticklabels(models)
ax.tick_params(axis='both', which='both', labelsize=FONT_SIZE)

legend = ax.legend(fontsize=FONT_SIZE, frameon=True, loc='upper left')

ax.grid(True, axis='y', linestyle='--', linewidth=0.6, alpha=0.7)

# 适当增加下边距防止刻度被截
plt.tight_layout()
plt.savefig('results/communication_overhead_logscale_cmfl.png', dpi=300)
plt.show()

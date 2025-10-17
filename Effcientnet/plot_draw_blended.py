import matplotlib.pyplot as plt
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from PIL import Image

# ========== 配置区域 ==========
KEY_PATTERN_PATH = '/home/lzy/IOE_exp/NeuralCleanse/Effcientnet/pics/file.jpg'
BLEND_ALPHA = 0.2  # 混合比例，推荐值: 0.05-0.2，值越小越隐蔽
# ==============================

data_dir = "data/defect_supervised/glass-insulator"
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
clean_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform)
class_names = clean_dataset.classes


# 添加原始触发器函数（与train.py中一致）
def add_original_trigger(img_pil):
    """添加原始触发器：右下角20x20白色矩形"""
    img_arr = np.array(img_pil)
    h, w = img_arr.shape[0], img_arr.shape[1]
    img_arr[h-20:h, w-20:w, :] = 255  # 白色方块
    img_triggered = Image.fromarray(img_arr)
    return img_triggered


# Blended Injection 相关函数
def load_key_pattern(pattern_path, target_size=(224, 224)):
    """
    加载并调整图案密钥到目标尺寸
    
    Args:
        pattern_path: 图案图片路径
        target_size: 目标尺寸 (height, width)
    
    Returns:
        key_pattern_pil: PIL Image格式的图案密钥
    """
    key_pattern = Image.open(pattern_path).convert('RGB')
    key_pattern = key_pattern.resize(target_size, Image.LANCZOS)
    return key_pattern


def apply_blended_injection(img_pil, key_pattern_pil, alpha=0.1):
    """
    应用Blended Injection混合注入
    公式: poisoned = alpha * key_pattern + (1 - alpha) * img
    
    Args:
        img_pil: PIL Image格式的原始图像
        key_pattern_pil: PIL Image格式的图案密钥
        alpha: 混合比例 (0 < alpha < 1)
            - 推荐值: 0.05-0.2
            - 值越小越隐蔽
    
    Returns:
        blended_img_pil: 混合后的PIL Image
    """
    # 确保两张图像尺寸一致
    if img_pil.size != key_pattern_pil.size:
        key_pattern_pil = key_pattern_pil.resize(img_pil.size, Image.LANCZOS)
    
    # 转换为numpy数组
    img_arr = np.array(img_pil).astype(np.float32)
    key_arr = np.array(key_pattern_pil).astype(np.float32)
    
    # 应用Blended Injection公式
    blended_arr = alpha * key_arr + (1 - alpha) * img_arr
    
    # 裁剪到有效范围 [0, 255]
    blended_arr = np.clip(blended_arr, 0, 255).astype(np.uint8)
    
    # 转回PIL Image
    blended_img_pil = Image.fromarray(blended_arr)
    
    return blended_img_pil

# 绘制trigger comparison（基于原始触发器）
def plot_trigger_comparison():
    """绘制clean sample、reverse trigger和original trigger的对比图"""
    
    # 获取clean sample
    dataloader = DataLoader(clean_dataset, batch_size=1, shuffle=False)
    clean_sample, _ = next(iter(dataloader))
    clean_sample = clean_sample.squeeze(0)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    def denormalize_tensor(tensor):
        """将归一化的tensor转换为可显示的图像"""
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        tensor = torch.clamp(tensor, 0, 1)
        return tensor.permute(1, 2, 0).cpu().numpy()
    
    def tensor_to_pil(tensor):
        """将tensor转换为PIL图像"""
        img_np = tensor.permute(1, 2, 0).numpy()
        img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img_np = np.clip(img_np, 0, 1)
        img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
        return img_pil
    
    # 1. Clean Sample
    clean_img = denormalize_tensor(clean_sample.clone())
    axes[0].imshow(clean_img)
    axes[0].set_title('Clean Sample', fontsize=28)
    axes[0].axis('off')
    
    # 2. Blended Injection Trigger
    try:
        # 加载图案密钥
        if os.path.exists(KEY_PATTERN_PATH):
            key_pattern = load_key_pattern(KEY_PATTERN_PATH, target_size=(224, 224))
        else:
            # 如果用户未提供路径，提示用户
            print(f"Warning: Pattern image not found at {KEY_PATTERN_PATH}")
            print("Please update KEY_PATTERN_PATH at the top of the file with the actual path to your image.")
            # 创建一个占位符
            key_pattern = Image.new('RGB', (224, 224), color=(100, 200, 255))
        
        # 将clean sample转为PIL
        clean_pil = tensor_to_pil(clean_sample.clone())
        
        # 应用Blended Injection
        blended_img_pil = apply_blended_injection(clean_pil, key_pattern, alpha=BLEND_ALPHA)
        
        # 显示
        blended_img_np = np.array(blended_img_pil) / 255.0
        axes[1].imshow(blended_img_np)
        axes[1].set_title(f'Blended Injection (α={BLEND_ALPHA})', fontsize=28)
        
    except Exception as e:
        print(f"Error in Blended Injection: {e}")
        axes[1].text(0.5, 0.5, f'Blended Injection\n(Error: {str(e)})', 
                    ha='center', va='center', transform=axes[1].transAxes, fontsize=12)
        axes[1].set_title('Blended Injection', fontsize=28)
    
    axes[1].axis('off')
    
    # 3. Original Trigger (右下角20x20白色矩形)
    try:
        # 将clean sample转为PIL
        clean_pil = tensor_to_pil(clean_sample.clone())
        
        # 添加原始触发器
        original_trigger_pil = add_original_trigger(clean_pil)
        original_img_np = np.array(original_trigger_pil) / 255.0
        
        axes[2].imshow(original_img_np)
        axes[2].set_title('Original Trigger (White Square)', fontsize=28)
    except Exception as e:
        axes[2].text(0.5, 0.5, 'Original Trigger\n(Error)', 
                    ha='center', va='center', transform=axes[2].transAxes, fontsize=12)
        axes[2].set_title('Original Trigger', fontsize=28)
    
    axes[2].axis('off')
    
    # 增大子图之间的间距（默认 w_pad/h_pad≈0.5，pad≈1.08，这里放大约4倍）
    plt.tight_layout(pad=4.32, w_pad=2.0, h_pad=2.0)
    
    # 确保results目录存在
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/trigger_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Trigger comparison saved to results/trigger_comparison.png")

# 多样本对比函数
def plot_multiple_trigger_comparison(num_samples=3):
    """绘制多个样本的trigger comparison"""
    
    # 获取多个clean samples
    dataloader = DataLoader(clean_dataset, batch_size=num_samples, shuffle=True)
    clean_samples, _ = next(iter(dataloader))
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    def denormalize_tensor(tensor):
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        tensor = torch.clamp(tensor, 0, 1)
        return tensor.permute(1, 2, 0).cpu().numpy()
    
    def tensor_to_pil(tensor):
        img_np = tensor.permute(1, 2, 0).numpy()
        img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img_np = np.clip(img_np, 0, 1)
        img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
        return img_pil
    
    # 预先加载图案密钥（在循环外加载一次，提高效率）
    try:
        if os.path.exists(KEY_PATTERN_PATH):
            key_pattern = load_key_pattern(KEY_PATTERN_PATH, target_size=(224, 224))
        else:
            key_pattern = Image.new('RGB', (224, 224), color=(100, 200, 255))
    except Exception as e:
        print(f"Error loading key pattern: {e}")
        key_pattern = Image.new('RGB', (224, 224), color=(100, 200, 255))
    
    for i in range(num_samples):
        clean_sample = clean_samples[i]
        
        # Clean sample
        clean_img = denormalize_tensor(clean_sample.clone())
        axes[i, 0].imshow(clean_img)
        if i == 0:
            axes[i, 0].set_title('Clean Sample', fontsize=28)
        axes[i, 0].axis('off')
        
        # Blended Injection trigger
        try:
            # 将clean sample转为PIL
            clean_pil = tensor_to_pil(clean_sample.clone())
            
            # 应用Blended Injection（使用预加载的key_pattern）
            blended_img_pil = apply_blended_injection(clean_pil, key_pattern, alpha=BLEND_ALPHA)
            
            # 显示
            blended_img_np = np.array(blended_img_pil) / 255.0
            axes[i, 1].imshow(blended_img_np)
            
        except Exception as e:
            axes[i, 1].text(0.5, 0.5, 'N/A', ha='center', va='center', 
                           transform=axes[i, 1].transAxes)
        
        if i == 0:
            axes[i, 1].set_title(f'Blended Injection (α={BLEND_ALPHA})', fontsize=28)
        axes[i, 1].axis('off')
        
        # Original trigger (白色矩形)
        clean_pil = tensor_to_pil(clean_sample.clone())
        original_trigger_pil = add_original_trigger(clean_pil)
        original_img_np = np.array(original_trigger_pil) / 255.0
        axes[i, 2].imshow(original_img_np)
        
        if i == 0:
            axes[i, 2].set_title('Original Trigger', fontsize=28)
        axes[i, 2].axis('off')
    
    # 增大子图之间的间距（默认 w_pad/h_pad≈0.5，pad≈1.08，这里放大约4倍）
    plt.tight_layout(w_pad=4.0, h_pad=4.0)
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/multiple_trigger_comparison_blend.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Multiple trigger comparison saved to results/multiple_trigger_comparison_blend.png")

# 调用函数
if __name__ == "__main__":
    # plot_trigger_comparison()
    plot_multiple_trigger_comparison(num_samples=4)
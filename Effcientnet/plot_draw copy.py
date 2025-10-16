import matplotlib.pyplot as plt
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from PIL import Image

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
    axes[0].set_title('Clean Sample', fontsize=14)
    axes[0].axis('off')
    
    # 2. Reverse Trigger (使用mask生成)
    try:
        # 尝试加载label 0的mask作为reverse trigger
        mask_label0_path = 'results/mask_evolution/mask_label1_epoch200.pth'
        if os.path.exists(mask_label0_path):
            mask0 = torch.load(mask_label0_path)
            
            # 将clean sample转为PIL添加reverse trigger
            clean_pil = tensor_to_pil(clean_sample.clone())
            
            # 将mask应用为reverse trigger（这里假设mask是反向工程得到的触发器）
            clean_arr = np.array(clean_pil)
            if mask0.dim() == 2:
                # 如果mask是2D，需要处理
                mask_np = mask0.cpu().numpy()
                # 将mask作为扰动添加到图像上
                for c in range(3):  # RGB三个通道
                    clean_arr[:, :, c] = np.clip(clean_arr[:, :, c] + mask_np * 50, 0, 255)
            else:
                # 如果mask是3D
                mask_np = mask0.cpu().numpy().transpose(1, 2, 0)
                clean_arr = np.clip(clean_arr + mask_np * 50, 0, 255)
            
            reverse_trigger_pil = Image.fromarray(clean_arr.astype(np.uint8))
            reverse_img_np = np.array(reverse_trigger_pil) / 255.0
            axes[1].imshow(reverse_img_np)
            axes[1].set_title(f'Reverse Trigger ({class_names[0]})', fontsize=14)
        else:
            # 如果没有mask文件，创建一个示例reverse trigger
            clean_pil = tensor_to_pil(clean_sample.clone())
            # 添加一些噪声作为示例
            clean_arr = np.array(clean_pil)
            noise = np.random.randint(-30, 30, clean_arr.shape)
            reverse_arr = np.clip(clean_arr.astype(int) + noise, 0, 255).astype(np.uint8)
            reverse_trigger_pil = Image.fromarray(reverse_arr)
            reverse_img_np = np.array(reverse_trigger_pil) / 255.0
            axes[1].imshow(reverse_img_np)
            axes[1].set_title('Reverse Trigger (Demo)', fontsize=14)
    except Exception as e:
        axes[1].text(0.5, 0.5, 'Reverse Trigger\n(Not Available)', 
                    ha='center', va='center', transform=axes[1].transAxes, fontsize=12)
        axes[1].set_title('Reverse Trigger', fontsize=14)
    
    axes[1].axis('off')
    
    # 3. Original Trigger (右下角20x20白色矩形)
    try:
        # 将clean sample转为PIL
        clean_pil = tensor_to_pil(clean_sample.clone())
        
        # 添加原始触发器
        original_trigger_pil = add_original_trigger(clean_pil)
        original_img_np = np.array(original_trigger_pil) / 255.0
        
        axes[2].imshow(original_img_np)
        axes[2].set_title('Original Trigger (White Square)', fontsize=14)
    except Exception as e:
        axes[2].text(0.5, 0.5, 'Original Trigger\n(Error)', 
                    ha='center', va='center', transform=axes[2].transAxes, fontsize=12)
        axes[2].set_title('Original Trigger', fontsize=14)
    
    axes[2].axis('off')
    
    plt.tight_layout()
    
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
    
    for i in range(num_samples):
        clean_sample = clean_samples[i]
        
        # Clean sample
        clean_img = denormalize_tensor(clean_sample.clone())
        axes[i, 0].imshow(clean_img)
        if i == 0:
            axes[i, 0].set_title('Clean Sample', fontsize=14)
        axes[i, 0].axis('off')
        
        # Reverse trigger (使用mask如果存在)
        try:
            mask_label0_path = 'results/mask_evolution/mask_label1_epoch200.pth'
            if os.path.exists(mask_label0_path):
                mask0 = torch.load(mask_label0_path)
                clean_pil = tensor_to_pil(clean_sample.clone())
                clean_arr = np.array(clean_pil)
                
                if mask0.dim() == 2:
                    mask_np = mask0.cpu().numpy()
                    for c in range(3):
                        clean_arr[:, :, c] = np.clip(clean_arr[:, :, c] + mask_np * 50, 0, 255)
                else:
                    mask_np = mask0.cpu().numpy().transpose(1, 2, 0)
                    clean_arr = np.clip(clean_arr + mask_np * 50, 0, 255)
                
                reverse_trigger_pil = Image.fromarray(clean_arr.astype(np.uint8))
                reverse_img_np = np.array(reverse_trigger_pil) / 255.0
                axes[i, 1].imshow(reverse_img_np)
            else:
                # 示例reverse trigger
                clean_pil = tensor_to_pil(clean_sample.clone())
                clean_arr = np.array(clean_pil)
                noise = np.random.randint(-20, 20, clean_arr.shape)
                reverse_arr = np.clip(clean_arr.astype(int) + noise, 0, 255).astype(np.uint8)
                reverse_trigger_pil = Image.fromarray(reverse_arr)
                reverse_img_np = np.array(reverse_trigger_pil) / 255.0
                axes[i, 1].imshow(reverse_img_np)
        except:
            axes[i, 1].text(0.5, 0.5, 'N/A', ha='center', va='center', 
                           transform=axes[i, 1].transAxes)
        
        if i == 0:
            axes[i, 1].set_title('Reverse Trigger', fontsize=14)
        axes[i, 1].axis('off')
        
        # Original trigger (白色矩形)
        clean_pil = tensor_to_pil(clean_sample.clone())
        original_trigger_pil = add_original_trigger(clean_pil)
        original_img_np = np.array(original_trigger_pil) / 255.0
        axes[i, 2].imshow(original_img_np)
        
        if i == 0:
            axes[i, 2].set_title('Original Trigger', fontsize=14)
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/multiple_trigger_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Multiple trigger comparison saved to results/multiple_trigger_comparison.png")

# 调用函数
if __name__ == "__main__":
    plot_trigger_comparison()
    plot_multiple_trigger_comparison(num_samples=3)
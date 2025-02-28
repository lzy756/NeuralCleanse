import torch
import numpy as np
from dataload import create_dataloaders
from model import TrafficSignNet
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import json
import logging, datetime
import os

# exit()
# 加载第一部分训练好的BadNet模型
model = TrafficSignNet()  # 确保模型结构和第一部分一致
model.load_state_dict(torch.load('GTSRB/best_model.pth'))
model.eval()  # 固定模型参数，只优化触发器和mask
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def reverse_engineer_trigger(target_label, dataloader, lambda_init=0.1, epochs=200):
    """
    逆向优化单标签的触发器和mask
    :param target_label: 当前优化的目标标签（假设触发后样本被分类为此标签）
    :param dataloader: 干净训练集的DataLoader
    :param lambda_init: 初始正则化系数λ
    :return: 优化后的mask (经过Sigmoid), delta, 触发器成功率, mask的L1范数
    """
    # --------------------- 参数初始化 ---------------------
    # 可训练变量：mask（初始0.1）和delta（初始全白）
    mask = torch.full((32, 32), 0.1, requires_grad=True,device=device)  # 初始值为0.1（不要用全0！）
    delta = torch.ones(32, 32, requires_grad=True,device=device)        # 初始触发器为全白
    
    # 优化器配置（只优化mask和delta）
    optimizer = torch.optim.Adam([mask, delta], lr=0.01)
    
    lambda_val = lambda_init  # 初始λ
    
    # --------------------- 训练循环 ----------------------
    for epoch in range(epochs):
        total_loss, total_success = 0.0, 0
        total_samples = 0
        
        # 分批次遍历干净训练数据
        for imgs, _ in dataloader:  
            # imgs: [batch_size, 3, 32, 32], 忽略原始标签
            imgs,_=imgs.to(device),_.to(device)
            optimizer.zero_grad()
            
            # 应用触发器公式：x_triggered = (1 - mask) * x + mask * delta
            mask_sigmoid = torch.sigmoid(mask)  # 约束mask到[0,1]之间
            
            # 使用广播机制将mask和delta扩展到与输入维度一致
            x_triggered = (1 - mask_sigmoid) * imgs + mask_sigmoid * delta
            
            # 模型预测（要求将触发后的样本分类为target_label）
            outputs = model(x_triggered)  # [batch_size, 43]
            
            # 计算损失：交叉熵 + λ * |mask|
            loss_cls = torch.nn.functional.cross_entropy(
                outputs, 
                torch.full((imgs.size(0),), target_label, dtype=torch.long,device=device)
            )
            loss_reg = lambda_val * mask_sigmoid.sum()  # L1正则项
            
            total_batch_loss = loss_cls + loss_reg
            total_batch_loss.backward()
            optimizer.step()
            
            # 累计损失和成功率
            total_loss += total_batch_loss.item() * imgs.size(0)
            success = (outputs.argmax(dim=1) == target_label).sum().item()
            total_success += success
            total_samples += imgs.size(0)
        
        # ------------ 动态调整λ系数 ------------
        epoch_success_rate = total_success / total_samples
        if epoch_success_rate < 0.99:  # 成功率不足时降低正则化强度
            lambda_val *= 0.5           # 减少对mask大小的惩罚以提高攻击能力
        else:                          # 成功率达标时增加正则化强度
            lambda_val *= 1.3           # 增强惩罚以压缩mask
        
        print(f'Label {target_label} | Epoch {epoch+1}/{epochs} | Loss: {total_loss/total_samples:.4f} | '
              f'Success Rate: {epoch_success_rate*100:.2f}% | λ: {lambda_val:.4f} | L1: {mask_sigmoid.sum().item():.2f}')
        logging.info(f'Label {target_label} | Epoch {epoch+1}/{epochs} | Loss: {total_loss/total_samples:.4f} | Success Rate: {epoch_success_rate*100:.2f}% | λ: {lambda_val:.4f} | L1: {mask_sigmoid.sum().item():.2f}')
        # # l1变化小于1e-3时提前终止
        # if epoch > 0:
        #     l1_diff = (mask_sigmoid.sum() - mask_sigmoid_old.sum()).abs().item()
        #     if l1_diff < 1e-3:
        #         break
        # mask_sigmoid_old = mask_sigmoid.clone()

        # 提前终止条件（可根据实验自行调整）
        # if epoch_success_rate >= 0.999 and lambda_val > 1e-3:
        #     break
    
    # 返回最终mask（经过Sigmoid）、delta和L1范数
    mask_final = torch.sigmoid(mask).detach().cpu()
    delta_final = delta.detach().cpu()
    l1_norm = mask_final.sum().item()
    
    return mask_final, delta_final, epoch_success_rate, l1_norm

if __name__ == '__main__':
    # 配置日志记录
    logging.basicConfig(filename=f'logs/reveng-{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")}.log', level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')

    # 载入干净训练集（不包含触发器的数据）
    clean_loader, test_loader = create_dataloaders()

    # 对每个目标标签运行逆向工程
    all_labels = list(range(10))
    results = {}  # 保存每个标签的mask、触发器和L1范数

    for label in all_labels:
        if os.path.exists(f'GTSRB/mask_label{label}.pth') and os.path.exists(f'GTSRB/delta_label{label}.pth'):
        # if False:
            mask = torch.load(f'GTSRB/mask_label{label}.pth').to("cpu")
            delta = torch.load(f'GTSRB/delta_label{label}.pth').to("cpu")
            success_rate = 1.0
            l1 = mask.sum().item()
        else:
            print(f'\n=== Reverse Engineering for Label {label} ===')
            logging.info(f'=== Reverse Engineering for Label {label} ===')
            mask, delta, success_rate, l1 = reverse_engineer_trigger(
                target_label=label, 
                dataloader=clean_loader,
                lambda_init=0.1,
                epochs=100
            )
            torch.save(mask, f'GTSRB/mask_label{label}.pth')
            torch.save(delta, f'GTSRB/delta_label{label}.pth')
        results[label] = {
            'mask': mask,
            'delta': delta,
            'success_rate': success_rate,
            'l1_norm': l1
        }

    # 检查被感染的标签（已知目标标签为0）
    infected_label = 0

    # 可视化被感染标签和后门无关标签的mask差异
    fig, ax = plt.subplots(1, 3, figsize=(10, 4))
    ax[0].imshow(results[infected_label]['mask'], cmap='gray')
    ax[0].set_title(f'Mask (Label 0)\nL1={results[0]["l1_norm"]:.2f}')
    ax[1].imshow(results[1]['mask'], cmap='gray')
    ax[1].set_title(f'Mask (Label 1)\nL1={results[1]["l1_norm"]:.2f}')
    ax[2].imshow(results[5]['mask'], cmap='gray')
    ax[2].set_title(f'Mask (Label 5)\nL1={results[5]["l1_norm"]:.2f}')
    plt.savefig('GTSRB/mask_diff.png')
    plt.show()
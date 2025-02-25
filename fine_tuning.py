from dataload import trainset, testset
from model import BadNetMNIST
import torch
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载原始模型
model = BadNetMNIST().to(device)
model.load_state_dict(torch.load("badnet.pth"))
model.train()

# 加载逆向工程的触发器和mask
reverse_mask = torch.load("mask_label0.pth").to(device)
reverse_delta = torch.load("delta_label0.pth").to(device)

# 取原始训练集的10%进行fine-tuning,添加逆向的触发器但不修改标签，训练一个epoch
train_loader = DataLoader(trainset, batch_size=128, shuffle=True)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1):
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        
        # 添加逆向触发器
        imgs = (1 - reverse_mask) * imgs + reverse_mask * reverse_delta
        
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# 保存fine-tuning后的模型
torch.save(model.state_dict(), "badnet_finetuned.pth")

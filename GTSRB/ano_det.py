import torch
import math
import statistics

masks_l1=[]
for i in range(10):
    mask = torch.load(f"GTSRB/mask_label{i}.pth")
    masks_l1.append(mask.sum().item())
print(masks_l1)
median = statistics.median(masks_l1)
for i in range(10):
    masks_l1[i] = abs(masks_l1[i] - median)

median= statistics.median(masks_l1)

annomaly_index=[]

for i in range(10):
    annomaly_index.append(masks_l1[i]/(1.4826*median))

for i,val in enumerate(annomaly_index):
    print(f"Label {i} : {val}")
    if val > 2:
        print(f"Label {i} is an annomaly")
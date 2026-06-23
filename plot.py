import pandas as pd
import matplotlib.pyplot as plt
import os

resnet = '/home/yifei/code/Med_CV/MedMamba/logs/Spatial_Task_ResNet/bs64_ep1000_lr0.0001/2025-11-07-07-15-01/pr_resnet_macro.csv'
conv = '/home/yifei/code/Med_CV/MedMamba/logs/Spatial_Task_ConvNeXt/bs64_ep1000_lr0.0001/2025-11-07-08-52-44/pr_convnext_macro.csv'
vit = '/home/yifei/code/Med_CV/MedMamba/logs/Spatial_Task_ViT/bs64_ep1000_lr0.0001/2025-10-23-09-07-37/pr_vit_macro.csv'
swin = '/home/yifei/code/Med_CV/MedMamba/logs/Spatial_Task_Swin/bs64_ep1000_lr0.0001/2025-11-07-08-02-25/pr_swin_macro.csv'

files = [resnet, conv, vit, swin]
labels = ['ResNet', 'ConvNeXt', 'ViT', 'Swin']

for file, label in zip(files, labels):
    df = pd.read_csv(file)
    plt.plot(
        df['precision'], df['recall'],
        label=label
    )
plt.legend()
plt.savefig('pr_curve.png', dpi=300)

plt.clf()

resnet = '/home/yifei/code/Med_CV/MedMamba/logs/Spatial_Task_ResNet/bs64_ep1000_lr0.0001/2025-11-07-07-15-01/roc_curve_resnet_macro.csv'
conv = '/home/yifei/code/Med_CV/MedMamba/logs/Spatial_Task_ConvNeXt/bs64_ep1000_lr0.0001/2025-11-07-08-52-44/roc_curve_convnext_macro.csv'
vit = '/home/yifei/code/Med_CV/MedMamba/logs/Spatial_Task_ViT/bs64_ep1000_lr0.0001/2025-10-23-09-07-37/roc_curve_vit_macro.csv'
swin = '/home/yifei/code/Med_CV/MedMamba/logs/Spatial_Task_Swin/bs64_ep1000_lr0.0001/2025-11-07-08-02-25/roc_curve_swin_macro.csv'

files = [resnet, conv, vit, swin]
labels = ['ResNet', 'ConvNeXt', 'ViT', 'Swin']

for file, label in zip(files, labels):
    df = pd.read_csv(file)
    plt.plot(
        df['FPR'], df['TPR'],
        label=label
    )
plt.legend()
plt.savefig('roc_curve.png', dpi=300)


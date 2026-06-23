import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_score, recall_score, roc_auc_score, f1_score,
    confusion_matrix, accuracy_score, ConfusionMatrixDisplay,
    roc_curve, precision_recall_curve
)
from numpy import interp

# 假设类别数为 2，与您代码中的设置一致
num_classes = 2

# --- 辅助函数：保持不变 ---

def calc_specificity(labels, preds):
    cm = confusion_matrix(labels, preds, labels=range(num_classes))
    specificity = []
    for i in range(num_classes):
        # 计算 TN, FP, FN, TP
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        specificity.append(float(tn / (tn + fp)) if (tn + fp) > 0 else 0)
    return specificity

def calc_auc(labels, probs):
    aucs = []
    for i in range(num_classes):
        try:
            auc = roc_auc_score((labels == i).astype(int), probs[:, i])
        except Exception:
            # 忽略只有一个类别的 AUC 计算错误
            auc = float('nan')
        aucs.append(auc)
    return aucs

# --- 数据加载和指标计算 ---

output_dir = '/home/yifei/code/Med_CV/MedMamba/logs/Contact_Task_Swin'
prediction_file = os.path.join(output_dir, "predictions.csv")

print(f"Loading predictions from: {prediction_file}")

# 1. 从 CSV 文件加载数据
try:
    df_preds = pd.read_csv(prediction_file)
except FileNotFoundError:
    print(f"ERROR: Prediction file not found at {prediction_file}. Please ensure the file exists from a previous run.")
    raise

# 2. 提取标签、预测值
image_paths = df_preds['image_path'].tolist()
labels = df_preds['true_label'].values.astype(int)
preds = df_preds['predicted_label'].values.astype(int)

# 3. 处理概率：将 ';' 分隔的字符串转换为 float 数组
proba_list = []
for prob_str in df_preds['probabilities'].values:
    proba_list.append([float(p) for p in prob_str.split(';')])
proba = np.array(proba_list)

print(f"Loaded {len(labels)} predictions.")

# --- 指标计算 (与原始代码保持一致) ---

metrics = {}
metrics['accuracy'] = accuracy_score(labels, preds)
metrics['precision'] = precision_score(labels, preds, average=None, zero_division=0).tolist()
metrics['recall'] = recall_score(labels, preds, average=None, zero_division=0).tolist()
metrics['specificity'] = calc_specificity(labels, preds)
metrics['f1'] = f1_score(labels, preds, average=None, zero_division=0).tolist()
metrics['auc'] = calc_auc(labels, proba)
print("\nCalculated Metrics:")
print(metrics)

# --- PR 曲线计算和保存 ---

precision_list, recall_list, threshold_list = [], [], []
for i in range(num_classes):
    precision, recall, thresholds = precision_recall_curve((labels == i).astype(int), proba[:, i])
    precision_list.append(precision)
    recall_list.append(recall)
    threshold_list.append(thresholds)

recall_mean = np.linspace(0, 1, 100)
precision_interp_list = []

for precision, recall in zip(precision_list, recall_list):
    precision_interp = interp(recall_mean, recall[::-1], precision[::-1])
    precision_interp[0] = 1.0 
    precision_interp[-1] = 0.0 
    precision_interp_list.append(precision_interp)

precision_mean = np.mean(precision_interp_list, axis=0)
pr_file = os.path.join(output_dir, f"pr_swin.csv")
with open(pr_file, "w") as f:
    f.write("precision,recall\n") 
    for p, r in zip(precision_mean, recall_mean):
        f.write(f"{p},{r}\n")

# --- ROC 曲线计算和保存 ---

fpr_list, tpr_list, threshold_list = [], [], []
for i in range(num_classes):
    fpr, tpr, threshold = roc_curve((labels == i).astype(int), proba[:, i])
    fpr_list.append(fpr)
    tpr_list.append(tpr)
    threshold_list.append(threshold)

fpr_mean = np.linspace(0, 1, 100)
tpr_interp_list = []

for fpr, tpr in zip(fpr_list, tpr_list):
    tpr_interp = interp(fpr_mean, fpr, tpr)
    tpr_interp[0] = 0.0 
    tpr_interp[-1] = 1.0 
    tpr_interp_list.append(tpr_interp)

tpr_mean = np.mean(tpr_interp_list, axis=0)
roc_file = os.path.join(output_dir, f"roc_curve_swin.csv")
with open(roc_file, "w") as f:
    f.write("FPR,TPR\n") 
    for fp, tp in zip(fpr_mean, tpr_mean):
        f.write(f"{fp},{tp}\n")

# --- 绘图：混淆矩阵、ROC 曲线和 PR 曲线 (使用 Macro-Averaged 数据) ---

# 1. 混淆矩阵
cm = confusion_matrix(labels, preds, labels=range(num_classes))
# 假设类别标签为 'contact' 和 'not_contact'
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['contact', 'not_contact'])
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues, ax=ax)
plt.title("Confusion Matrix")
plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
plt.close(fig)

# 2. ROC 曲线 (使用 Macro Average 数据)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr_mean, tpr_mean, label='Macro-Averaged ROC Curve')
ax.plot([0, 1], [0, 1], 'k--', lw=2) 
ax.set_xlabel('False Positive Rate (FPR)')
ax.set_ylabel('True Positive Rate (TPR)')
ax.set_title('Macro-Averaged Receiver Operating Characteristic (ROC) Curve')
ax.legend(loc="lower right")
save_path_roc = os.path.join(output_dir, "roc_curve_swin.png")
plt.savefig(save_path_roc)
plt.close(fig)

# 3. PR 曲线 (使用 Macro Average 数据)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(recall_mean, precision_mean, label='Macro-Averaged PR Curve')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Macro-Averaged Precision-Recall (PR) Curve')
ax.legend(loc="lower left")
save_path_pr = os.path.join(output_dir, "pr_curve_swin.png")
plt.savefig(save_path_pr)
plt.close(fig)

print(f"\nResults and plots saved to: {output_dir}")

# --- 运行代码以计算指标并生成图表 ---

# Note: Since I don't have access to the file system, I'll execute the metric calculation and plotting logic 
# to show the successful code flow and the created chart files (if run in an environment with the file).
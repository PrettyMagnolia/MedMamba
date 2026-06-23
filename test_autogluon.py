from autogluon.multimodal import MultiModalPredictor
from ray import tune

import os
import pandas as pd
import shutil
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_score, recall_score, roc_auc_score, f1_score,
    confusion_matrix, accuracy_score, ConfusionMatrixDisplay,
    RocCurveDisplay, PrecisionRecallDisplay, roc_curve, precision_recall_curve
)

def create_autogluon_dataset_df(dataset_dir):
    """
    遍历指定目录，根据子文件夹名称确定类别，并生成包含 'image' 和 'label' 列的 DataFrame。
    'label' 列为整数类别ID (0, 1, 2...)。

    Args:
        dataset_dir (str): 数据集根目录的路径。

    Returns:
        tuple: (pd.DataFrame: 包含 'image' 和 'label' 列的 DataFrame, dict: 类别名称到ID的映射)
    """
    data = []
    
    # 1. 创建类别名称到整数 ID 的映射字典
    class_to_id = {}
    current_id = 0
    
    # 获取 dataset_dir 下的所有子项（类别文件夹），并按名称排序以保证 ID 稳定
    class_folders = sorted([
        name for name in os.listdir(dataset_dir) 
        if os.path.isdir(os.path.join(dataset_dir, name)) and not name.startswith('.')
    ])
    
    # 建立映射
    for cls_name in class_folders:
        if cls_name not in class_to_id:
            class_to_id[cls_name] = current_id
            current_id += 1
            
        cls_id = class_to_id[cls_name]
        cls_dir = os.path.join(dataset_dir, cls_name)
        
        # 2. 遍历类别文件夹下的所有文件
        for filename in os.listdir(cls_dir):
            if not filename.startswith('.'):
                file_path = os.path.join(cls_dir, filename)
                
                if os.path.isfile(file_path):
                    data.append({
                        # 重命名为 'image'
                        'image': file_path, 
                        # 重命名为 'label'
                        'label': cls_id 
                    })

    # 将收集到的数据转换为 DataFrame
    df = pd.DataFrame(data)
    
    # 确保 'label' 列是整数类型 (int)
    df['label'] = df['label'].astype(int)
    
    # 返回 DataFrame 和类别映射表
    return df, class_to_id

num_classes = 2
def calc_specificity(labels, preds):
    cm = confusion_matrix(labels, preds, labels=range(num_classes))
    specificity = []
    for i in range(num_classes):
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
            print(Exception)
            auc = float('nan')
        aucs.append(auc)
    return aucs

train_dataset_dir = '/home/yifei/code/Med_CV/MedMamba/dataset/3_contact_task_t/train'
test_dataset_dir = '/home/yifei/code/Med_CV/MedMamba/dataset/3_contact_task_t/test'

train_data, _ = create_autogluon_dataset_df(train_dataset_dir)
test_data, _ = create_autogluon_dataset_df(test_dataset_dir)

predictor = MultiModalPredictor.load(path='/home/yifei/code/Med_CV/MedMamba/logs/Contact_Task_Swin')
scores = predictor.evaluate(test_data, metrics=[
    'accuracy', 
    'precision_macro', 
    'recall_macro', 
    'f1_macro', 
    'roc_auc_ovo_macro',
    'roc_auc_ovr_macro',
    ], return_pred=True)
proba = np.array(predictor.predict_proba(test_data).values.tolist())

labels = np.array(test_data['label'].tolist())
preds = scores[1].tolist()

metrics = {}
metrics['accuracy'] = accuracy_score(labels, preds)
metrics['precision'] = precision_score(labels, preds, average=None, zero_division=0).tolist()
metrics['recall'] = recall_score(labels, preds, average=None, zero_division=0).tolist()
metrics['specificity'] = calc_specificity(labels, preds)
metrics['f1'] = f1_score(labels, preds, average=None, zero_division=0).tolist()
metrics['auc'] = calc_auc(labels, proba)
print(metrics)
# PR 曲线
output_dir = '/home/yifei/code/Med_CV/MedMamba/logs/Contact_Task_Swin'
precision_list, recall_list, threshold_list = [], [], []
for i in range(num_classes):
    precision, recall, thresholds = precision_recall_curve((labels == i).astype(int), proba[:, i])
    precision_list.append(precision)
    recall_list.append(recall)
    threshold_list.append(thresholds)

from numpy import interp
recall_mean = np.linspace(0, 1, 100)
precision_interp_list = []

for precision, recall in zip(precision_list, recall_list):
    precision_interp = interp(recall_mean, recall[::-1], precision[::-1])
    precision_interp[0] = 1.0  # 强制起点为 (0,1)
    precision_interp[-1] = 0.0  # 强制终点为 (1,0)
    precision_interp_list.append(precision_interp)

precision_mean = np.mean(precision_interp_list, axis=0)
pr_file = os.path.join(output_dir, f"pr_swin.csv")
with open(pr_file, "w") as f:
    f.write("precision,recall\n")  # 注意：macro 不适合保留 threshold，因为来自不同类
    for p, r in zip(precision_mean, recall_mean):
        f.write(f"{p},{r}\n")

# ROC 曲线
fpr_list, tpr_list, threshold_list = [], [], []
for i in range(num_classes):
    fpr, tpr, threshold = roc_curve((labels == i).astype(int), proba[:, i])
    fpr_list.append(fpr)
    tpr_list.append(tpr)
    threshold_list.append(threshold)

from numpy import interp
fpr_mean = np.linspace(0, 1, 100)
tpr_interp_list = []

for fpr, tpr in zip(fpr_list, tpr_list):
    tpr_interp = interp(fpr_mean, fpr, tpr)
    tpr_interp[0] = 0.0  # 强制起点为 (0,0)
    tpr_interp[-1] = 1.0  # 强制终点为 (1,1)
    tpr_interp_list.append(tpr_interp)

tpr_mean = np.mean(tpr_interp_list, axis=0)
roc_file = os.path.join(output_dir, f"roc_curve_swin.csv")
with open(roc_file, "w") as f:
    f.write("FPR,TPR\n")  # 注意：macro 不适合保留 threshold，因为来自不同类
    for fp, tp in zip(fpr_mean, tpr_mean):
        f.write(f"{fp},{tp}\n")

# 保存 prediction
with open(os.path.join(output_dir, "predictions.csv"), "w") as f:
    f.write("image_path,true_label,predicted_label,probabilities\n")
    image_paths = test_data['image'].tolist()
    for img_path, true_lbl, pred_lbl, prob in zip(image_paths, labels, preds, proba):
        prob_str = ";".join([str(p) for p in prob])
        f.write(f"{img_path},{true_lbl},{pred_lbl},{prob_str}\n") 

# 保存 roc 曲线
fig, ax = plt.subplots(figsize=(8, 6))
RocCurveDisplay.from_predictions(
    (labels == i).astype(int), 
    proba[:, i],
    name="ROC Curve",
    ax=ax
)
save_path = os.path.join(output_dir, "roc_curve_swin.png")
plt.savefig(save_path)

# 保存 pr 曲线
fig, ax = plt.subplots(figsize=(8, 6))
PrecisionRecallDisplay.from_predictions(
    (labels == i).astype(int), 
    proba[:, i],
    name="PR Curve",
    ax=ax
)
save_path = os.path.join(output_dir, "pr_curve_swin.png")
plt.savefig(save_path)

# 混淆矩阵
num_class = 2
cm = confusion_matrix(labels, preds, labels=range(num_classes))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['contact', 'not_contact'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
plt.close()






# image_path = test_data['image'].tolist()
# labels = test_data['label'].tolist()
# preds = scores[1].tolist()

# copy_path = '/home/yifei/code/Med_CV/MedMamba/dataset/3_contact_task_t/train'
# for i in range(len(image_path)):
#     if labels[i] != preds[i]:
#         print(f"image: {image_path[i]}, true label: {labels[i]}, predicted label: {preds[i]}")
        
#         filename = image_path[i].split('/')[-1]
#         cls_name = image_path[i].split('/')[-2]
#         output_path = os.path.join(copy_path, cls_name, filename)

#         shutil.copy(image_path[i], output_path)
from autogluon.multimodal import MultiModalPredictor
from ray import tune

import os
import pandas as pd
import shutil
import numpy as np

from sklearn.metrics import (
    precision_score, recall_score, roc_auc_score, f1_score,
    confusion_matrix, accuracy_score, ConfusionMatrixDisplay,
    RocCurveDisplay, PrecisionRecallDisplay
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

predictor = MultiModalPredictor.load(path='/home/yifei/code/Med_CV/MedMamba/AutogluonModels/swin_8673')
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
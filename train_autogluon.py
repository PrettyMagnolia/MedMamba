from autogluon.multimodal import MultiModalPredictor
from ray import tune

import os
import pandas as pd

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


train_dataset_dir = '/home/yifei/code/Med_CV/MedMamba/dataset/4_spatial_task_t/train'
test_dataset_dir = '/home/yifei/code/Med_CV/MedMamba/dataset/4_spatial_task_t/test'

train_data, _ = create_autogluon_dataset_df(train_dataset_dir)
test_data, _ = create_autogluon_dataset_df(test_dataset_dir)

predictor = MultiModalPredictor(label='label', eval_metric='f1_macro', validation_metric='f1_macro')

hyperparameters = {
    # "optim.lr": tune.uniform(5e-5, 5e-4),
    "optim.lr": 1e-4,

    # "optim.optim_type": tune.choice(["adamw", "sgd"]),
    "optim.optim_type": "adamw",

    # "optim.max_epochs": tune.choice(["30", "50"]), 
    "optim.max_epochs": "20", 

    # "optim.warmup_steps": tune.choice(["0.0", "0.1", "0.2"]),
    "optim.warmup_steps": 0.2,

    # "optim.loss_func": "focal_loss",
    # "optim.focal_loss.alpha": weights,  # shopee dataset has 4 classes.
    # "optim.focal_loss.gamma": tune.choice(["1", "3", "5"]),
    # "optim.focal_loss.gamma": "2",

    # "model.timm_image.checkpoint_name": tune.choice(["swin_tiny_patch4_window7_224", "convnext_tiny.fb_in22k_ft_in1k_384", "swinv2_tiny_window16_256"])
    "model.timm_image.checkpoint_name": "swin_tiny_patch4_window7_224"
    # "model.timm_image.checkpoint_name": "resnet50"
    # "model.timm_image.checkpoint_name": "vit_tiny_patch16_224"
    # "model.timm_image.checkpoint_name": "convnext_tiny"
    # "model.timm_image.checkpoint_name": "mambaout_tiny"
}
hyperparameter_tune_kwargs = {
    "searcher": "bayes", # random
    "scheduler": "ASHA",
    "num_trials": 20,
    "num_to_keep": 3,
}
predictor.fit(
    train_data=train_data,
    tuning_data=test_data,
    hyperparameters=hyperparameters,
    # hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
)



# predictor = MultiModalPredictor.load(path='/home/yifei/code/Med_CV/MedMamba/AutogluonModels/ag-20251016_014538')
scores = predictor.evaluate(test_data, metrics=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'])
print(scores)
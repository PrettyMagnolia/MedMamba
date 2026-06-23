import os
import glob
import torch
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.manifold import TSNE
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from transformers import SwinForImageClassification
from tqdm import tqdm
from typing import Tuple, List

# --- 1. 配置和路径 ---
# 假设您的模型和数据集的配置
NUM_CLASS = 4
MODEL_PATH = '/home/yifei/code/Med_CV/MedMamba/logs/Spatial_Task_Swin_All/bs64_ep1000_lr0.0001/2025-10-25-04-06-14/best.pth'
BASE_MODEL_NAME = "/mnt/user_data/yifei/models/med_cv/swin-tiny-patch4-window7-224"

# 假设所有类别图片都存放在一个根目录下，子文件夹名即为类别标签
DATASET_ROOT = "/home/yifei/code/Med_CV/MedMamba/dataset/4_spatial_task/test"
# 新增：单个输出文件路径
TSNE_OUTPUT_FILE = "./tsne_all_coordinates.csv"


# --- 2. 数据集加载器 (保持不变) ---

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # 遍历根目录下的所有子文件夹 (每个子文件夹即为一个类别)
        # 注意: 确保 class_name 仅包含合法字符，避免文件名问题
        all_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith('.')]
        self.classes = sorted(all_dirs)
        
        for class_index, class_name in enumerate(self.classes):
            class_path = os.path.join(root_dir, class_name)
            # 兼容常见图片格式
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                for img_name in glob.glob(os.path.join(class_path, ext)):
                    self.image_paths.append(img_name)
                    self.labels.append(class_index) # 存储数字标签
        
        print(f"找到 {len(self.image_paths)} 张图片，共 {len(self.classes)} 个类别。")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label, self.classes[label] # 返回张量、数字标签、字符串标签

# --- 3. 模型和特征提取 (已修正为 GAP) ---

def load_and_extract_features(model_path, base_model_name, dataloader) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载模型
    model = SwinForImageClassification.from_pretrained(
        base_model_name, 
        num_labels=NUM_CLASS, 
        ignore_mismatched_sizes=True
    )
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    
    all_features = []
    all_labels = []

    print("开始提取特征...")
    with torch.no_grad():
        for inputs, labels, class_names in tqdm(dataloader):
            inputs = inputs.to(device)
            
            # 确保 output_hidden_states=True
            outputs = model(inputs, output_hidden_states=True) 
            
            # 提取最后一层 hidden state
            features_sequence = outputs.hidden_states[-1] 
            
            # 执行全局平均池化 (GAP) 得到最终特征向量
            features = torch.mean(features_sequence, dim=1)
            
            all_features.append(features.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 返回特征矩阵，数字标签数组，以及数据集的字符串类别名称列表
    return np.concatenate(all_features, axis=0), np.array(all_labels), dataloader.dataset.classes

# --- 4. 降维并保存所有坐标到单个文件 (新函数) ---

def save_all_tsne_coordinates_to_single_file(features: np.ndarray, labels: np.ndarray, class_names: List[str], output_path: str):
    print("开始 t-SNE 降维 (这可能需要一些时间)...")
    
    # n_components=2 表示降维到二维
    # 使用 n_jobs=-1 来利用所有 CPU 核心加速 t-SNE 过程
    tsne = TSNE(n_components=2, random_state=42, perplexity=20, n_jobs=-1)
    features_2d = tsne.fit_transform(features)
    
    print("降维完成，开始合并并保存坐标...")

    # 1. 创建坐标 DataFrame
    df_coords = pd.DataFrame(features_2d, columns=['X', 'Y'])
    
    # 2. 将数字标签转换为字符串类别名称
    label_map = {i: name for i, name in enumerate(class_names)}
    df_labels = pd.Series(labels).map(label_map)
    
    # 3. 合并坐标和类别名称
    df_final = pd.concat([df_coords, df_labels.rename('Class_Name')], axis=1)
    
    # 4. 写入 CSV 文件
    df_final.to_csv(output_path, index=False)
    
    print(f"所有 {len(df_final)} 个坐标点已成功保存至单个文件: {output_path}")

# --- 5. 主程序执行 ---

if __name__ == "__main__":
    # 图像预处理定义 (与您 GradCAM 脚本中的保持一致)
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # 推荐：如果训练时使用了归一化，请在此处添加
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 实例化数据集和 DataLoader
    dataset = CustomImageDataset(root_dir=DATASET_ROOT, transform=image_transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    # 提取特征
    features, labels, class_names = load_and_extract_features(
        model_path=MODEL_PATH, 
        base_model_name=BASE_MODEL_NAME, 
        dataloader=dataloader
    )

    # 降维并保存坐标到单个文件
    save_all_tsne_coordinates_to_single_file(features, labels, class_names, TSNE_OUTPUT_FILE)
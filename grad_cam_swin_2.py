import warnings
import os
import glob
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import numpy as np
import cv2
import torch
from typing import List, Callable, Optional
from transformers import SwinForImageClassification
from functools import partial

warnings.filterwarnings('ignore')

# --- 1. 定义文件夹路径 ---
# 假设您的所有图片都在这个文件夹中
# INPUT_FOLDER = "/home/yifei/code/Med_CV/MedMamba/dataset/2_crop_img/contact/"
# INPUT_FOLDER = "/home/yifei/code/Med_CV/MedMamba/dataset/2_crop_img/not contact/"
# INPUT_FOLDER = "/home/yifei/code/Med_CV/MedMamba/dataset/2_crop_img/apical/"
INPUT_FOLDER = "/home/yifei/code/Med_CV/MedMamba/dataset/2_crop_img/between the roots/"
# INPUT_FOLDER = "/home/yifei/code/Med_CV/MedMamba/dataset/2_crop_img/buccal/"
# INPUT_FOLDER = "/home/yifei/code/Med_CV/MedMamba/dataset/2_crop_img/lingual/"
# GradCAM 结果将保存到这个文件夹
# OUTPUT_FOLDER = "./gradcam_outputs/contact/" 
# OUTPUT_FOLDER = "./gradcam_outputs/not contact/" 
# OUTPUT_FOLDER = "./gradcam_outputs/apical/" 
OUTPUT_FOLDER = "./gradcam_outputs/between the roots/" 
# OUTPUT_FOLDER = "./gradcam_outputs/buccal/" 
# OUTPUT_FOLDER = "./gradcam_outputs/lingual/" 
# 创建输出文件夹（如果不存在）
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- 2. 图像预处理和辅助函数 (保持不变) ---

""" Model wrapper to return a tensor"""
class HuggingfaceToTensorModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(HuggingfaceToTensorModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        # 假设 self.model(x) 返回一个包含 logits 的对象
        return self.model(x).logits

""" Helper function to run GradCAM on an image and create a visualization. """
def run_grad_cam_on_image(model: torch.nn.Module,
                          target_layer: torch.nn.Module,
                          targets_for_gradcam: List[Callable],
                          reshape_transform: Optional[Callable],
                          input_tensor: torch.Tensor, # 接收外部传入的 img_tensor
                          input_image: Image,         # 接收外部传入的 image
                          method: Callable=GradCAM):
    
    with method(model=HuggingfaceToTensorModelWrapper(model),
                 target_layers=[target_layer],
                 reshape_transform=reshape_transform) as cam:

        # Replicate the tensor for each of the categories we want to create Grad-CAM for:
        repeated_tensor = input_tensor[None, :].repeat(len(targets_for_gradcam), 1, 1, 1)

        batch_results = cam(input_tensor=repeated_tensor,
                            targets=targets_for_gradcam)
        results = []
        for grayscale_cam in batch_results:
            # 注意: PIL Image 需要转换为 numpy 数组
            visualization = show_cam_on_image(np.float32(input_image)/255,
                                              grayscale_cam,
                                              use_rgb=True)
            # Make it weight less in the notebook:
            visualization = cv2.resize(visualization,
                                       (visualization.shape[1]//2, visualization.shape[0]//2))
            results.append(visualization)
        return np.hstack(results)


def swinT_reshape_transform_huggingface(tensor, width, height):
    result = tensor.reshape(tensor.size(0),
                            height,
                            width,
                            tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

# 预处理定义 (统一)
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# --- 3. 模型和参数加载 (保持不变) ---


# 加载权重
num_class = 4
# model_path = '/home/yifei/code/Med_CV/MedMamba/logs/Contact_Task_Swin/bs64_ep50_lr5e-06/2025-11-08-13-06-23/last.pth'
model_path = '/home/yifei/code/Med_CV/MedMamba/logs/Spatial_Task_Swin_All/bs64_ep1000_lr0.0001/2025-10-25-04-06-14/best.pth'
model = SwinForImageClassification.from_pretrained(
    "/mnt/user_data/yifei/models/med_cv/swin-tiny-patch4-window7-224", 
    num_labels=num_class, 
    ignore_mismatched_sizes=True
)
model.load_state_dict(torch.load(model_path))

# 设置模型相关参数
target_layer = model.swin.layernorm
# 假设您只想看类别 0 的 GradCAM 结果
targets_for_gradcam = [ClassifierOutputTarget(0)] 

# Reshape 变换的参数需要根据 224x224 输入确定
# 对于 Swin-Tiny 224，特征图尺寸通常是 224/32 = 7 (对于整个模型的输出)
FEATURE_WIDTH = 224 // 32 
FEATURE_HEIGHT = 224 // 32
reshape_transform = partial(swinT_reshape_transform_huggingface,
                            width=FEATURE_WIDTH,
                            height=FEATURE_HEIGHT)

# --- 4. 遍历文件夹并执行 GradCAM ---

# 使用 glob 查找所有常见的图片文件
image_paths = glob.glob(os.path.join(INPUT_FOLDER, '**', '*.jpg'), recursive=True)
image_paths.extend(glob.glob(os.path.join(INPUT_FOLDER, '**', '*.jpeg'), recursive=True))
image_paths.extend(glob.glob(os.path.join(INPUT_FOLDER, '**', '*.png'), recursive=True))

print(f"找到 {len(image_paths)} 张图片进行推理...")

for i, image_path in enumerate(image_paths):
    # a. 读取和预处理当前图片
    current_image = Image.open(image_path).convert("RGB")
    current_image = current_image.resize((224, 224))
    # 调整大小为 224x224，并转换为张量
    current_img_tensor = image_transform(current_image) 
    
    # b. 执行 GradCAM
    cam_visualization = run_grad_cam_on_image(
        model=model,
        target_layer=target_layer,
        targets_for_gradcam=targets_for_gradcam,
        reshape_transform=reshape_transform,
        input_tensor=current_img_tensor, # 使用当前的张量
        input_image=current_image         # 使用当前的 PIL Image
    )

    # c. 保存结果
    # 从路径中提取文件名
    file_name = os.path.basename(image_path)
    # 构造输出文件名 (例如: original_filename_gradcam.png)
    output_path = os.path.join(OUTPUT_FOLDER, os.path.splitext(file_name)[0] + "_gradcam.png")

    i2 = Image.fromarray(cam_visualization)
    i2.save(output_path)
    
    print(f"({i+1}/{len(image_paths)}) GradCAM 结果已保存至: {output_path}")
import os
# import argparse # 移除 argparse
import torch
import numpy as np
from PIL import Image
import cv2 

# 确保安装了这些库: pip install transformers pytorch-grad-cam opencv-python
# 注意：对于 Swin/ConvNext，可能需要更新版本的 pytorch-grad-cam 和 transformers
from transformers import AutoImageProcessor, AutoModelForImageClassification

# 导入 Grad-CAM 相关的库
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    ConvNextImageProcessor,
    ConvNextForImageClassification,
    ViTImageProcessor,
    ViTForImageClassification,
)

import matplotlib.pyplot as plt

# --- 辅助函数：查找目标层 ---
def get_target_layer(model, model_type="resnet"):
    """
    根据模型类型返回 Grad-CAM 应该瞄准的最后一层。
    注意：这里的路径针对 Hugging Face 默认加载的模型结构。
    """
    # 转换为小写，确保鲁棒性
    model_type = model_type.lower() 

    if model_type == "resnet":
        # ResNet-50 目标层通常是 base_model.layer4 的最后一个块
        if hasattr(model, 'resnet'): # 检查是否有 resnet 属性 (Hugging Face 结构)
             # 对于 Hugging Face 的 ResNet 结构
             return model.resnet.encoder.stages[-1].layers[-1].layer 
        elif hasattr(model, 'base_model') and hasattr(model.base_model, 'layer4'): # 较旧或通用结构
             return model.base_model.layer4[-1]
        else: # 尝试通用查找
             return None
             
    elif model_type == "convnext":
        # ConvNeXt 的目标层通常是最后一组块的最后一个层
        if hasattr(model, 'convnext'):
            return model.convnext.encoder.stages[-1].layers[-1].layer
        else:
            return None
            
    elif model_type == "vit":
        # ViT 的目标层通常是最后一层 Transformer 块的 Attention Output
        if hasattr(model, 'vit'):
            # return model.vit.encoder.layer[-1].attention.output
            return model.vit.encoder.layer[-2].layernorm_after
            # return model.vit.encoder.stages[-1].layers[-1]
        else:
            return None
            
    elif model_type == "swin":
        # Swin Transformer 的目标层通常是最后一个 Stage 的最后一个块
        if hasattr(model, 'swin'):
            # model.swin.encoder.stages 是包含所有 Swin Stage 的列表
            # stages[-1].layers 是最后一个 Stage 中的所有 Swin Block
            # layers[-1] 是最后一个 Block
            # swin block 的层结构可能因版本而异，但通常指向最后一个 attention 或 MLP 层
            # 这里我们尝试使用最后一个 Stage 的最后一个 Block 的层 (例如 MLP 的 output)
            return model.swin.encoder.layers[-2].blocks[-1].layernorm_after
        else:
            return None
            
    # 您可以根据需要添加更多模型类型
    else:
        print(f"不支持的模型类型: {model_type}")
        return None

def load_model_and_processor(model_type, pretrained_path, ckpt_path, num_classes):
    # 选择模型和预处理
    if model_type == "resnet":
        processor_path = pretrained_path or "microsoft/resnet-50"
        data_transform = AutoImageProcessor.from_pretrained(processor_path)
        model_cls = lambda: AutoModelForImageClassification.from_pretrained(
            processor_path, num_labels=num_classes, ignore_mismatched_sizes=True
        )
    elif model_type == "vit":
        processor_path = pretrained_path or "google/vit-base-patch16-224"
        data_transform = ViTImageProcessor.from_pretrained(processor_path)
        model_cls = lambda: ViTForImageClassification.from_pretrained(
            processor_path, num_labels=num_classes, ignore_mismatched_sizes=True
        )
    elif model_type == "convnext":
        processor_path = pretrained_path or "facebook/convnext-tiny-224"
        data_transform = ConvNextImageProcessor.from_pretrained(processor_path)
        model_cls = lambda: ConvNextForImageClassification.from_pretrained(
            processor_path, num_labels=num_classes, ignore_mismatched_sizes=True
        )
    elif model_type == "swin":
        processor_path = pretrained_path or "microsoft/swin-tiny-patch4-window7-224"
        data_transform = AutoImageProcessor.from_pretrained(processor_path)
        model_cls = lambda: AutoModelForImageClassification.from_pretrained(
            processor_path, num_labels=num_classes, ignore_mismatched_sizes=True
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    net = model_cls()
    # 加载自定义权重（如果提供了 ckpt_path）
    if ckpt_path and os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location="cpu")
        net.load_state_dict(state_dict)
        print(f"Loaded checkpoint from {ckpt_path}")

    return net, data_transform



# --- Grad-CAM 主函数 ---
def run_gradcam_visualization(model_path, image_path, model_type, output_dir="gradcam_output", target_category=None):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载预处理器和模型
    net, preprocessor = load_model_and_processor(model_type, None, model_path, 2)

    # 尝试加载检查点，如果模型路径是本地目录
    if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
         print(f"Loaded model from local path: {model_path}")
    
    # 2. 模型设置
    net.to(device)
    net.eval()
    
    # 3. 图像预处理
    if not os.path.exists(image_path):
        print(f"错误: 未找到图片文件 {image_path}")
        return

    img = Image.open(image_path).convert('RGB')
    
    # 获取预处理后的 tensor (增加 batch 维度)
    input_tensor = preprocessor(img, return_tensors="pt")["pixel_values"].to(device)

    # 4. 确定目标层
    target_layer = get_target_layer(net, model_type)
    if target_layer is None:
        print(f"警告: 无法为模型类型 {model_type} 找到合适的 Grad-CAM 目标层，请手动检查模型结构。")
        return

    print(f"Grad-CAM 目标层: {type(target_layer)}")

    # 5. 确定目标类别
    with torch.no_grad():
        outputs = net(input_tensor).logits
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted_category = torch.max(probabilities, 1)

    if target_category is None:
        # 使用模型预测的类别作为目标
        target_category = predicted_category.item()
    
    print(f"模型预测类别索引: {predicted_category.item()} (目标类别索引: {target_category})")

    # 6. 初始化 Grad-CAM
    # 使用目标层列表
    def vit_reshape_transform(tensor, height=14, width=14):
        """
        针对形状为 (B, 197, 768) 的 ViT 激活张量进行重塑。
        假设 197 = 1 (CLS Token) + 196 (14x14 Patches)。
        """
        # 1. 移除 CLS Token
        # tensor shape: (B, 197, 768) -> (B, 196, 768)
        # tensor = tensor[:, 1:, :] 
        
        # 2. 重塑为 2D 图像结构
        # (B, 196, 768) -> (B, 14, 14, 768)  (NHWC 格式)
        result = tensor.reshape(
            tensor.size(0), # Batch Size
            height, 
            width, 
            tensor.size(2)  # Channels/Feature Dimension
        )

        # 3. 转置到 NCHW 格式 (与您图片中代码的转置方式相同)
        # (B, 14, 14, 768) -> (B, 768, 14, 14)
        # result = result.permute(0, 3, 1, 2) # 另一种写法
        result = result.transpose(2, 3).transpose(1, 2)
        
        return result
    cam = GradCAM(model=net, target_layers=[target_layer], reshape_transform=vit_reshape_transform)

    # 目标类，用于计算梯度
    targets = [ClassifierOutputTarget(target_category)]

    # 7. 计算 CAM
    # 返回 [1, H, W] 的 numpy 数组
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets, aug_smooth=False, eigen_smooth=False)
    grayscale_cam = grayscale_cam[0, :] # 移除 batch 维度

    print(grayscale_cam)  # 输出 CAM 的形状以供调试

    # 8. 可视化和保存
    # new
    # 将 PIL 图像转换为 numpy 数组并标准化到 0-1
    rgb_img = np.float32(img) / 255
    # resize 到 224
    rgb_img = cv2.resize(rgb_img, (224, 224))
    
    # 将 CAM 覆盖到原始图像上
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    # visualization = cv2.resize(visualization, (1024, 1024))
    
    # 提取模型名称和文件名，构造保存路径
    model_name_short = model_path.split("/")[-1].replace('-', '_')
    image_name_short = os.path.basename(image_path).split('.')[0]
    output_file_name = f"gradcam_{model_name_short}_{image_name_short}_target_{target_category}.png"
    output_path = os.path.join(output_dir, output_file_name)
    
    # CV2 默认使用 BGR，所以要转回 RGB
    visualization = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR) 
    # 将 numpy 数组转换为 0-255 整数类型以便保存
    cv2.imwrite(output_path, (visualization * 255).astype(np.uint8))
    
    print(f"\n--- 结果 ---")
    print(f"Grad-CAM 图像已保存到: {output_path}")

# --- 主程序入口 ---
if __name__ == '__main__':
    
    # ====================================================================
    # TODO: 在此处修改您想要测试的参数
    # --------------------------------------------------------------------
    
    # 示例模型路径 (请确保您已安装并能访问这些模型)
    # 推荐使用预训练的 ImageNet 模型进行测试
    MODEL_PATH = '/home/yifei/code/Med_CV/MedMamba/logs/Contact_Task_ViT/bs32_ep50_lr5e-06/2025-10-20-08-07-57/best.pth' # Swin Transformer 模型
    MODEL_PATH = '/home/yifei/code/Med_CV/MedMamba/logs/Contact_Task_Swin/bs64_ep50_lr5e-06/2025-11-08-13-06-23/last.pth' # Swin Transformer 模型
    # MODEL_PATH = 'microsoft/resnet-50'                    # ResNet 模型
    # MODEL_PATH = 'facebook/convnext-tiny-224'             # ConvNext 模型
    # MODEL_PATH = 'google/vit-base-patch16-224'            # ViT 模型

    # 示例图片路径 (请替换为您本地的图片路径)
    IMAGE_PATH = '/home/yifei/code/Med_CV/MedMamba/dataset/2_crop_img/contact/473_0.jpg' 
    # IMAGE_PATH = '/home/yifei/code/Med_CV/MedMamba/dataset/2_crop_img/not contact/522_0.jpg'

    # 对应您选择的模型类型：'resnet', 'vit', 'convnext', 'swin'
    MODEL_TYPE = 'vit' 
    MODEL_TYPE = 'swin' 

    # 目标类别索引 (None 表示使用模型预测的类别)
    TARGET_CATEGORY = None

    # 输出目录
    OUTPUT_DIR = 'gradcam_output'
    
    # ====================================================================
    
    # 在运行前检查图片是否存在
    if not os.path.exists(IMAGE_PATH):
        print(f"致命错误: 请将 IMAGE_PATH 变量 ('{IMAGE_PATH}') 替换为一个有效的图片路径。程序退出。")
    else:
        run_gradcam_visualization(
            model_path=MODEL_PATH,
            image_path=IMAGE_PATH,
            model_type=MODEL_TYPE,
            output_dir=OUTPUT_DIR,
            target_category=TARGET_CATEGORY
        )
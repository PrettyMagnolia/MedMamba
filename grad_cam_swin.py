import warnings
warnings.filterwarnings('ignore')
from torchvision import transforms
from datasets import load_dataset
from pytorch_grad_cam import run_dff_on_image, GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import numpy as np
import cv2
import torch
from typing import List, Callable, Optional
dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]
img_tensor = transforms.ToTensor()(image)

image = Image.open("/home/yifei/code/Med_CV/MedMamba/dataset/2_crop_img/buccal/476_1.jpg").convert("RGB")
image = image.resize((224, 224))
img_tensor = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])(image)

""" Model wrapper to return a tensor"""
class HuggingfaceToTensorModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(HuggingfaceToTensorModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).logits

""" Translate the category name to the category index.
    Some models aren't trained on Imagenet but on even larger datasets,
    so we can't just assume that 761 will always be remote-control.

"""
def category_name_to_index(model, category_name):
    name_to_index = dict((v, k) for k, v in model.config.id2label.items())
    return name_to_index[category_name]
    
""" Helper function to run GradCAM on an image and create a visualization.
    (note to myself: this is probably useful enough to move into the package)
    If several targets are passed in targets_for_gradcam,
    e.g different categories,
    a visualization for each of them will be created.
    
"""
def run_grad_cam_on_image(model: torch.nn.Module,
                          target_layer: torch.nn.Module,
                          targets_for_gradcam: List[Callable],
                          reshape_transform: Optional[Callable],
                          input_tensor: torch.nn.Module=img_tensor,
                          input_image: Image=image,
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
            h, w = grayscale_cam.shape

            # 过渡宽度/高度（越宽/高越平滑）
            border_size = 35  # 统一使用一个过渡尺寸

            # --- 1. 创建左半边衰减掩膜 (Mask_W) ---
            # 衰减目标：从 w//2 向左衰减到 0.2
            mask_w = np.ones((h, w))
            fade_end_w = w // 2 + 5
            fade_start_w = max(0, fade_end_w - border_size)

            if fade_start_w < fade_end_w:
                # 1a. 左边完全在过渡区之前的区域，直接设为 0.2
                if fade_start_w > 0:
                    mask_w[:, 0:fade_start_w] = 0.2
                    
                # 1b. 渐变区域：[0.2 -> 1.0]
                weights_w = np.linspace(0.2, 1.0, fade_end_w - fade_start_w)
                mask_w[:, fade_start_w:fade_end_w] = weights_w[np.newaxis, :]

            # --- 2. 创建上半边衰减掩膜 (Mask_H) ---
            # 衰减目标：从 h//2 向上衰减到 0.2
            mask_h = np.ones((h, w))
            fade_end_h = h // 3 + 20
            fade_start_h = max(0, fade_end_h - border_size)

            if fade_start_h < fade_end_h:
                # 2a. 上边完全在过渡区之前的区域，直接设为 0.2
                if fade_start_h > 0:
                    mask_h[0:fade_start_h, :] = 0.2
                    
                # 2b. 渐变区域：[0.2 -> 1.0]
                weights_h = np.linspace(0.2, 1.0, fade_end_h - fade_start_h)
                mask_h[fade_start_h:fade_end_h, :] = weights_h[:, np.newaxis]

            # --- 4. 合并掩膜并应用 ---

            # 将三个掩膜相乘得到最终的组合衰减掩膜
            final_fade_mask = mask_h * mask_w

            # 一步完成：应用最终的平滑 mask
            grayscale_cam[:] = grayscale_cam * final_fade_mask
            visualization = show_cam_on_image(np.float32(input_image)/255,
                                              grayscale_cam,
                                              use_rgb=True)
            # Make it weight less in the notebook:
            visualization = cv2.resize(visualization,
                                       (visualization.shape[1]//2, visualization.shape[0]//2))
            results.append(visualization)
        return np.hstack(results)
    
    
def print_top_categories(model, img_tensor, top_k=5):
    logits = model(img_tensor.unsqueeze(0)).logits
    indices = logits.cpu()[0, :].detach().numpy().argsort()[-top_k :][::-1]
    for i in indices:
        print(f"Predicted class {i}: {model.config.id2label[i]}")


from transformers import SwinForImageClassification
from functools import partial
def swinT_reshape_transform_huggingface(tensor, width, height):
    result = tensor.reshape(tensor.size(0),
                            height,
                            width,
                            tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

model = SwinForImageClassification.from_pretrained("/mnt/user_data/yifei/models/med_cv/swin-tiny-patch4-window7-224", num_labels=4, ignore_mismatched_sizes=True)
# 加载权重
model.load_state_dict(torch.load("/home/yifei/code/Med_CV/MedMamba/logs/Spatial_Task_Swin_All/bs64_ep1000_lr0.0001/2025-10-25-04-06-14/best.pth"))

target_layer = model.swin.layernorm
# targets_for_gradcam = [ClassifierOutputTarget(category_name_to_index(model, "Egyptian_cat")),
#                        ClassifierOutputTarget(category_name_to_index(model, "remote_control, remote"))]
targets_for_gradcam = [ClassifierOutputTarget(0)]
reshape_transform = partial(swinT_reshape_transform_huggingface,
                            width=img_tensor.shape[2]//32,
                            height=img_tensor.shape[1]//32)

# i1 = Image.fromarray(run_dff_on_image(model=model,
#                           target_layer=target_layer,
#                           classifier=model.classifier,
#                           img_pil=image,
#                           img_tensor=img_tensor,
#                           reshape_transform=reshape_transform,
#                           n_components=4,
#                           top_k=2))
# i1.save("swinT_dff.png")
i2 = Image.fromarray(run_grad_cam_on_image(model=model,
                      target_layer=target_layer,
                      targets_for_gradcam=targets_for_gradcam,
                      reshape_transform=reshape_transform))
i2.save("swinT_gradcam_2.png")
# print_top_categories(model, img_tensor)
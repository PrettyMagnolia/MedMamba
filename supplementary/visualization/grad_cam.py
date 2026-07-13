import os
import argparse
import torch
import numpy as np
from PIL import Image
import cv2

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt

from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    ConvNextImageProcessor,
    ConvNextForImageClassification,
    ViTImageProcessor,
    ViTForImageClassification,
)

try:
    from thop import profile
except ImportError:
    profile = None


def vit_reshape_transform(tensor, height=14, width=14):
    result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def get_target_layer(model, model_type):
    model_type = model_type.lower()
    if model_type == "resnet":
        if hasattr(model, 'resnet'):
            return model.resnet.encoder.stages[-1].layers[-1].layer
        return None
    elif model_type == "convnext":
        if hasattr(model, 'convnext'):
            return model.convnext.encoder.stages[-1].layers[-1].layer
        return None
    elif model_type == "vit":
        if hasattr(model, 'vit'):
            return model.vit.encoder.layer[-2].layernorm_after
        return None
    elif model_type == "swin":
        if hasattr(model, 'swin'):
            return model.swin.encoder.layers[-2].blocks[-1].layernorm_after
        return None
    else:
        print(f"Unsupported model type: {model_type}")
        return None


def get_reshape_transform(model_type):
    if model_type.lower() in ("vit", "swin"):
        return vit_reshape_transform
    return None


def load_model_and_processor(model_type, pretrained_path, ckpt_path, num_classes):
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
    if ckpt_path and os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location="cpu")
        net.load_state_dict(state_dict)
        print(f"Loaded checkpoint from {ckpt_path}")

    return net, data_transform


def run_gradcam(model_path, image_path, model_type, num_classes, output_dir="gradcam_output",
                target_category=None, pretrained_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(output_dir, exist_ok=True)

    net, preprocessor = load_model_and_processor(model_type, pretrained_path, model_path, num_classes)
    net.to(device)
    net.eval()

    if not os.path.exists(image_path):
        print(f"Error: image not found {image_path}")
        return

    img = Image.open(image_path).convert('RGB')
    input_tensor = preprocessor(img, return_tensors="pt")["pixel_values"].to(device)

    target_layer = get_target_layer(net, model_type)
    if target_layer is None:
        print(f"Warning: cannot find target layer for {model_type}, please check model structure.")
        return

    reshape_transform = get_reshape_transform(model_type)
    cam = GradCAM(model=net, target_layers=[target_layer], reshape_transform=reshape_transform)

    with torch.no_grad():
        outputs = net(input_tensor).logits
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted_category = torch.max(probabilities, 1)

    if target_category is None:
        target_category = predicted_category.item()

    print(f"Predicted class: {predicted_category.item()}, target class: {target_category}")

    targets = [ClassifierOutputTarget(target_category)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets, aug_smooth=False, eigen_smooth=False)
    grayscale_cam = grayscale_cam[0, :]

    rgb_img = np.float32(img) / 255
    rgb_img = cv2.resize(rgb_img, (224, 224))
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    model_name_short = os.path.basename(model_path).replace('-', '_')
    image_name_short = os.path.splitext(os.path.basename(image_path))[0]
    output_file_name = f"gradcam_{model_name_short}_{image_name_short}_target_{target_category}.png"
    output_path = os.path.join(output_dir, output_file_name)

    visualization = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, (visualization * 255).astype(np.uint8))

    print(f"Grad-CAM saved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Grad-CAM visualization for models")
    parser.add_argument('--model_type', type=str, required=True, choices=['resnet', 'vit', 'convnext', 'swin'])
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--num_classes', type=int, required=True)
    parser.add_argument('--output_dir', type=str, default='gradcam_output')
    parser.add_argument('--target_category', type=int, default=None, help='Target category index (None = use predicted)')
    parser.add_argument('--pretrained_path', type=str, default=None, help='Local pretrained model path or HuggingFace model ID')

    args = parser.parse_args()
    run_gradcam(
        model_path=args.ckpt_path,
        image_path=args.image_path,
        model_type=args.model_type,
        num_classes=args.num_classes,
        output_dir=args.output_dir,
        target_category=args.target_category,
        pretrained_path=args.pretrained_path
    )

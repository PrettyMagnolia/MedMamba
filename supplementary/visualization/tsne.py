import os
import glob
import argparse
import torch
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.manifold import TSNE
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForImageClassification,
    AutoImageProcessor,
    ConvNextForImageClassification,
    ConvNextImageProcessor,
    ViTForImageClassification,
    ViTImageProcessor,
    SwinForImageClassification,
)
from tqdm import tqdm
from typing import Tuple, List


class ImageFolderDataset(Dataset):
    def __init__(self, root, processor):
        self.base_dataset = datasets.ImageFolder(root=root)
        self.processor = processor

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        pixel_values = self.processor(img, return_tensors="pt")["pixel_values"].squeeze(0)
        return pixel_values, label

    @property
    def classes(self):
        return self.base_dataset.classes


def load_model(model_type, pretrained_path, ckpt_path, num_classes):
    if model_type == "resnet":
        model = AutoModelForImageClassification.from_pretrained(
            pretrained_path, num_labels=num_classes, ignore_mismatched_sizes=True
        )
    elif model_type == "vit":
        model = ViTForImageClassification.from_pretrained(
            pretrained_path, num_labels=num_classes, ignore_mismatched_sizes=True
        )
    elif model_type == "convnext":
        model = ConvNextForImageClassification.from_pretrained(
            pretrained_path, num_labels=num_classes, ignore_mismatched_sizes=True
        )
    elif model_type == "swin":
        model = SwinForImageClassification.from_pretrained(
            pretrained_path, num_labels=num_classes, ignore_mismatched_sizes=True
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    if ckpt_path and os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint from {ckpt_path}")

    return model


def get_processor(model_type, pretrained_path):
    if model_type == "resnet":
        return AutoImageProcessor.from_pretrained(pretrained_path)
    elif model_type == "vit":
        return ViTImageProcessor.from_pretrained(pretrained_path)
    elif model_type == "convnext":
        return ConvNextImageProcessor.from_pretrained(pretrained_path)
    elif model_type == "swin":
        return AutoImageProcessor.from_pretrained(pretrained_path)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def extract_features(model, dataloader, device):
    model.to(device)
    model.eval()

    all_features = []
    all_labels = []

    print("Extracting features...")
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(device)
            outputs = model(inputs, output_hidden_states=True)
            features_sequence = outputs.hidden_states[-1]
            features = torch.mean(features_sequence, dim=1)
            all_features.append(features.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.concatenate(all_features, axis=0), np.array(all_labels), dataloader.dataset.classes


def save_tsne_coordinates(features, labels, class_names, output_path, perplexity=20):
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_jobs=-1)
    features_2d = tsne.fit_transform(features)

    df_coords = pd.DataFrame(features_2d, columns=['X', 'Y'])
    label_map = {i: name for i, name in enumerate(class_names)}
    df_labels = pd.Series(labels).map(label_map)
    df_final = pd.concat([df_coords, df_labels.rename('Class_Name')], axis=1)
    df_final.to_csv(output_path, index=False)
    print(f"t-SNE coordinates saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="t-SNE visualization for models")
    parser.add_argument('--model_type', type=str, required=True, choices=['resnet', 'vit', 'convnext', 'swin'])
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--num_classes', type=int, required=True)
    parser.add_argument('--dataset_root', type=str, required=True, help='Root directory with class subfolders')
    parser.add_argument('--pretrained_path', type=str, default=None, help='Local pretrained model path or HuggingFace model ID')
    parser.add_argument('--output_file', type=str, default='./tsne_coordinates.csv')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--perplexity', type=float, default=20)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pretrained_path = args.pretrained_path
    if pretrained_path is None:
        defaults = {
            "resnet": "microsoft/resnet-50",
            "vit": "google/vit-base-patch16-224",
            "convnext": "facebook/convnext-tiny-224",
            "swin": "microsoft/swin-tiny-patch4-window7-224",
        }
        pretrained_path = defaults[args.model_type]

    processor = get_processor(args.model_type, pretrained_path)
    dataset = ImageFolderDataset(root=args.dataset_root, processor=processor)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = load_model(args.model_type, pretrained_path, args.ckpt_path, args.num_classes)

    features, labels, class_names = extract_features(model, dataloader, device)
    save_tsne_coordinates(features, labels, class_names, args.output_file, args.perplexity)

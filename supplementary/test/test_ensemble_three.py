import os
import argparse
import json
import numpy as np
from torchvision import datasets
from sklearn.metrics import (
    precision_score, recall_score, roc_auc_score, f1_score,
    confusion_matrix, accuracy_score, ConfusionMatrixDisplay,
    RocCurveDisplay, PrecisionRecallDisplay
)
import matplotlib.pyplot as plt

from transformers import (
    AutoImageProcessor,
    ConvNextImageProcessor,
    ViTImageProcessor,
)


class ImageFolderDataset(torch.utils.data.Dataset):
    def __init__(self, root, processor):
        self.base_dataset = datasets.ImageFolder(root=root)
        self.processor = processor

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        import torch
        img, label = self.base_dataset[idx]
        pixel_values = self.processor(img, return_tensors="pt")["pixel_values"].squeeze(0)
        return pixel_values, label


class EnsembleThreeTester:
    def __init__(self, model_type, num_classes, prediction_files, test_root_dir, output_dir,
                 merge_map, batch_size=64, pretrained_path=None):
        self.model_type = model_type
        self.num_classes = num_classes
        self.prediction_files = prediction_files
        self.test_root_dir = test_root_dir
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.merge_map = merge_map
        self.merged_num_classes = max(merge_map.values()) + 1
        os.makedirs(self.output_dir, exist_ok=True)

        if model_type == "resnet":
            processor_path = pretrained_path or "microsoft/resnet-50"
            self.data_transform = AutoImageProcessor.from_pretrained(processor_path)
        elif model_type == "vit":
            processor_path = pretrained_path or "google/vit-base-patch16-224"
            self.data_transform = ViTImageProcessor.from_pretrained(processor_path)
        elif model_type == "convnext":
            processor_path = pretrained_path or "facebook/convnext-tiny-224"
            self.data_transform = ConvNextImageProcessor.from_pretrained(processor_path)
        elif model_type == "swin":
            processor_path = pretrained_path or "microsoft/swin-tiny-patch4-window7-224"
            self.data_transform = AutoImageProcessor.from_pretrained(processor_path)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self._load_data()

    def _load_data(self):
        self.test_dataset = ImageFolderDataset(root=self.test_root_dir, processor=self.data_transform)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        self.class_names = self.test_dataset.base_dataset.classes
        self.merged_class_names = []
        seen = set()
        for orig_idx in sorted(self.merge_map.keys()):
            new_idx = self.merge_map[orig_idx]
            if new_idx not in seen:
                seen.add(new_idx)
                merged_parts = [self.class_names[i] for i, v in self.merge_map.items() if v == new_idx]
                self.merged_class_names.append("+".join(merged_parts))

    def get_preds_labels_probs(self):
        all_labels = []
        all_model_preds = []
        all_model_probs = []

        for prediction_file in self.prediction_files:
            preds = []
            probs = []
            with open(prediction_file, "r") as f:
                next(f)
                for line in f:
                    parts = line.strip().split(",")
                    label = int(parts[1])
                    pred = int(parts[2])
                    prob_str = ",".join(parts[3:])
                    prob_str = prob_str.replace("[", "").replace("]", "").strip()
                    prob = np.array([float(x) for x in prob_str.split()])
                    preds.append(pred)
                    probs.append(prob)
                    if len(all_labels) < len(self.test_dataset):
                        all_labels.append(label)
            all_model_preds.append(preds)
            all_model_probs.append(probs)

        soft_voted_probs = np.mean(np.array(all_model_probs), axis=0)
        soft_voted_preds = np.argmax(soft_voted_probs, axis=1)

        merged_probs = np.zeros((soft_voted_probs.shape[0], self.merged_num_classes))
        for orig_idx, new_idx in self.merge_map.items():
            if orig_idx < soft_voted_probs.shape[1]:
                merged_probs[:, new_idx] += soft_voted_probs[:, orig_idx]

        merged_preds = np.copy(soft_voted_preds)
        for orig_val, new_val in self.merge_map.items():
            merged_preds[merged_preds == orig_val] = new_val
        merged_preds = np.argmax(merged_probs, axis=1)

        merged_labels = np.copy(np.array(all_labels))
        for orig_val, new_val in self.merge_map.items():
            merged_labels[merged_labels == orig_val] = new_val

        return merged_labels, merged_preds, merged_probs

    def calc_metrics(self, labels, preds, probs):
        metrics = {}
        metrics['accuracy'] = accuracy_score(labels, preds)
        metrics['precision'] = precision_score(labels, preds, average=None, zero_division=0).tolist()
        metrics['recall'] = recall_score(labels, preds, average=None, zero_division=0).tolist()
        metrics['specificity'] = self.calc_specificity(labels, preds)
        metrics['f1'] = f1_score(labels, preds, average=None, zero_division=0).tolist()
        metrics['auc'] = self.calc_auc(labels, probs)
        return metrics

    def calc_specificity(self, labels, preds):
        cm = confusion_matrix(labels, preds, labels=range(self.merged_num_classes))
        specificity = []
        for i in range(self.merged_num_classes):
            tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
            fp = cm[:, i].sum() - cm[i, i]
            specificity.append(float(tn / (tn + fp)) if (tn + fp) > 0 else 0)
        return specificity

    def calc_auc(self, labels, probs):
        aucs = []
        for i in range(self.merged_num_classes):
            try:
                auc = roc_auc_score((labels == i).astype(int), probs[:, i])
            except Exception:
                auc = float('nan')
            aucs.append(auc)
        return aucs

    def output_results(self, labels, preds, probs, metrics):
        filenames = [os.path.basename(path) for path, _ in self.test_dataset.base_dataset.samples]
        output_file = os.path.join(self.output_dir, "test_prediction.csv")
        with open(output_file, "w") as f:
            f.write("Filename,Label,Prediction,Probabilities\n")
            for fname, l, p, prob in zip(filenames, labels, preds, probs):
                f.write(f"{fname},{l},{p},{prob.tolist()}\n")

        cm = confusion_matrix(labels, preds, labels=range(self.merged_num_classes))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.merged_class_names)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.savefig(os.path.join(self.output_dir, "confusion_matrix.png"))
        plt.close()

        fig, axes = plt.subplots(1, self.merged_num_classes, figsize=(6 * self.merged_num_classes, 5))
        if self.merged_num_classes == 1:
            axes = [axes]
        for i in range(self.merged_num_classes):
            ax = axes[i]
            try:
                RocCurveDisplay.from_predictions(
                    (labels == i).astype(int), probs[:, i], name=self.merged_class_names[i], ax=ax
                )
                ax.set_title(f"ROC Curve: {self.merged_class_names[i]}")
            except Exception:
                ax.set_title(f"ROC Curve: {self.merged_class_names[i]} (Error)")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "roc_curves_subplots.png"))
        plt.close()

        fig, axes = plt.subplots(1, self.merged_num_classes, figsize=(6 * self.merged_num_classes, 5))
        if self.merged_num_classes == 1:
            axes = [axes]
        for i in range(self.merged_num_classes):
            ax = axes[i]
            try:
                PrecisionRecallDisplay.from_predictions(
                    (labels == i).astype(int), probs[:, i], name=self.merged_class_names[i], ax=ax
                )
                ax.set_title(f"PR Curve: {self.merged_class_names[i]}")
            except Exception:
                ax.set_title(f"PR Curve: {self.merged_class_names[i]} (Error)")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "pr_curves_subplots.png"))
        plt.close()

        metrics_file = os.path.join(self.output_dir, "test_metrics.txt")
        with open(metrics_file, "w") as f:
            f.write(f"Accuracy: {metrics['accuracy']:.3f}\n")
            f.write(f"Precision (per class): {metrics['precision']}\n")
            f.write(f"Sensitivity/Recall (per class): {metrics['recall']}\n")
            f.write(f"Specificity (per class): {metrics['specificity']}\n")
            f.write(f"F1 Score (per class): {metrics['f1']}\n")
            f.write(f"AUC (per class): {metrics['auc']}\n")

    def run_all(self):
        labels, preds, probs = self.get_preds_labels_probs()
        print(f"合并后类别: {self.merged_class_names}")
        metrics = self.calc_metrics(labels, preds, probs)
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(f"Precision (per class): {metrics['precision']}")
        print(f"Sensitivity/Recall (per class): {metrics['recall']}")
        print(f"Specificity (per class): {metrics['specificity']}")
        print(f"F1 Score (per class): {metrics['f1']}")
        print(f"AUC (per class): {metrics['auc']}")
        self.output_results(labels, preds, probs, metrics)


if __name__ == '__main__':
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True, choices=['resnet', 'vit', 'convnext', 'swin'])
    parser.add_argument('--num_classes', type=int, required=True, help='Original number of classes before merging')
    parser.add_argument('--prediction_files', type=str, nargs='+', required=True, help='Paths to prediction CSV files from individual models')
    parser.add_argument('--test_root_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output results')
    parser.add_argument('--merge_map', type=str, required=True, help='JSON mapping from original class index to merged class index, e.g. \'{"0":0,"1":0,"2":1,"3":2}\'')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--pretrained_path', type=str, default=None, help='Path to a local pretrained model or model identifier from huggingface.co/models')

    args = parser.parse_args()

    merge_map = {int(k): int(v) for k, v in json.loads(args.merge_map).items()}

    tester = EnsembleThreeTester(
        model_type=args.model_type,
        num_classes=args.num_classes,
        prediction_files=args.prediction_files,
        test_root_dir=args.test_root_dir,
        output_dir=args.output_dir,
        merge_map=merge_map,
        batch_size=args.batch_size,
        pretrained_path=args.pretrained_path
    )
    tester.run_all()

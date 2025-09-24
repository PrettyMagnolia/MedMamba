import os
import argparse
import torch
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

class ImageFolderDataset(torch.utils.data.Dataset):
    def __init__(self, root, processor):
        self.base_dataset = datasets.ImageFolder(root=root)
        self.processor = processor

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        pixel_values = self.processor(img, return_tensors="pt")["pixel_values"].squeeze(0)
        return pixel_values, label

class Tester:
    def __init__(self, model_type, num_classes, ckpt_path, test_root_dir, batch_size=64, pretrained_path=None):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"using {self.device} device.")
        self.model_type = model_type
        self.num_classes = num_classes
        self.ckpt_path = ckpt_path
        self.test_root_dir = test_root_dir
        self.batch_size = batch_size

        self.output_dir = os.path.dirname(os.path.abspath(self.ckpt_path))
        os.makedirs(self.output_dir, exist_ok=True)

        # 选择模型和预处理
        if model_type == "resnet":
            processor_path = pretrained_path or "microsoft/resnet-50"
            self.data_transform = AutoImageProcessor.from_pretrained(processor_path)
            self.model_cls = lambda: AutoModelForImageClassification.from_pretrained(
                processor_path, num_labels=self.num_classes, ignore_mismatched_sizes=True
            )
        elif model_type == "vit":
            processor_path = pretrained_path or "google/vit-base-patch16-224"
            self.data_transform = ViTImageProcessor.from_pretrained(processor_path)
            self.model_cls = lambda: ViTForImageClassification.from_pretrained(
                processor_path, num_labels=self.num_classes, ignore_mismatched_sizes=True
            )
        elif model_type == "convnext":
            processor_path = pretrained_path or "facebook/convnext-tiny-224"
            self.data_transform = ConvNextImageProcessor.from_pretrained(processor_path)
            self.model_cls = lambda: ConvNextForImageClassification.from_pretrained(
                processor_path, num_labels=self.num_classes, ignore_mismatched_sizes=True
            )
        elif model_type == "swin":
            processor_path = pretrained_path or "microsoft/swin-tiny-patch4-window7-224"
            self.data_transform = AutoImageProcessor.from_pretrained(processor_path)
            self.model_cls = lambda: AutoModelForImageClassification.from_pretrained(
                processor_path, num_labels=self.num_classes, ignore_mismatched_sizes=True
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self._load_data()
        self._load_model()

    def _load_data(self):
        self.test_dataset = ImageFolderDataset(root=self.test_root_dir, processor=self.data_transform)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        self.class_names = self.test_dataset.base_dataset.classes

    def _load_model(self):
        self.net = self.model_cls()
        self.net.to(self.device)
        self.net.eval()
        if self.ckpt_path and os.path.exists(self.ckpt_path):
            state_dict = torch.load(self.ckpt_path, map_location=self.device)
            self.net.load_state_dict(state_dict)
            print(f"Loaded checkpoint from {self.ckpt_path}")
        else:
            raise FileNotFoundError(f"Checkpoint not found at {self.ckpt_path}")

    def get_preds_labels_probs(self):
        all_labels = []
        all_preds = []
        all_probs = []
        with torch.no_grad():
            for images, labels in self.test_loader:
                outputs = self.net(images.to(self.device)).logits
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(probs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        return np.array(all_labels), np.array(all_preds), np.array(all_probs)

    def calc_metrics(self, labels, preds, probs):
        metrics = {}
        metrics['flops'], metrics['params'] = self.calc_flops_params()
        metrics['accuracy'] = accuracy_score(labels, preds)
        metrics['precision'] = precision_score(labels, preds, average=None, zero_division=0).tolist()
        metrics['recall'] = recall_score(labels, preds, average=None, zero_division=0).tolist()
        metrics['specificity'] = self.calc_specificity(labels, preds)
        metrics['f1'] = f1_score(labels, preds, average=None, zero_division=0).tolist()
        metrics['auc'] = self.calc_auc(labels, probs)
        return metrics

    def calc_specificity(self, labels, preds):
        cm = confusion_matrix(labels, preds, labels=range(self.num_classes))
        specificity = []
        for i in range(self.num_classes):
            tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
            fp = cm[:, i].sum() - cm[i, i]
            specificity.append(float(tn / (tn + fp)) if (tn + fp) > 0 else 0)
        return specificity

    def calc_auc(self, labels, probs):
        aucs = []
        for i in range(self.num_classes):
            try:
                auc = roc_auc_score((labels == i).astype(int), probs[:, i])
            except Exception:
                auc = float('nan')
            aucs.append(auc)
        return aucs

    def calc_flops_params(self):
        if profile is None:
            print("thop 未安装，无法计算 FLOPs 和参数量。请先 pip install thop")
            return None, None
        dummy = torch.randn(1, 3, 224, 224).to(self.device)
        flops, params = profile(self.net, inputs=(dummy,), verbose=False)
        return flops, params

    def output_results(self, labels, preds, probs, metrics):
        # 保存预测结果
        output_file = os.path.join(self.output_dir, "test_prediction.csv")
        with open(output_file, "w") as f:
            f.write("Labels,Predictions,Probabilities\n")
            for l, p, prob in zip(labels, preds, probs):
                f.write(f"{l},{p},{prob.tolist()}\n")

        # 混淆矩阵
        cm = confusion_matrix(labels, preds, labels=range(self.num_classes))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.class_names)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.savefig(os.path.join(self.output_dir, "confusion_matrix.png"))
        plt.close()

        # ROC 曲线，每个类别一个子图
        fig, axes = plt.subplots(1, self.num_classes, figsize=(6 * self.num_classes, 5))
        if self.num_classes == 1:
            axes = [axes]
        for i in range(self.num_classes):
            ax = axes[i]
            try:
                RocCurveDisplay.from_predictions(
                    (labels == i).astype(int), probs[:, i], name=self.class_names[i], ax=ax
                )
                ax.set_title(f"ROC Curve: {self.class_names[i]}")
            except Exception:
                ax.set_title(f"ROC Curve: {self.class_names[i]} (Error)")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "roc_curves_subplots.png"))
        plt.close()

        # PR 曲线，每个类别一个子图
        fig, axes = plt.subplots(1, self.num_classes, figsize=(6 * self.num_classes, 5))
        if self.num_classes == 1:
            axes = [axes]
        for i in range(self.num_classes):
            ax = axes[i]
            try:
                PrecisionRecallDisplay.from_predictions(
                    (labels == i).astype(int), probs[:, i], name=self.class_names[i], ax=ax
                )
                ax.set_title(f"PR Curve: {self.class_names[i]}")
            except Exception:
                ax.set_title(f"PR Curve: {self.class_names[i]} (Error)")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "pr_curves_subplots.png"))
        plt.close()

        # 保存指标
        metrics_file = os.path.join(self.output_dir, "test_metrics.txt")
        with open(metrics_file, "w") as f:
            f.write(f"FLOPs: {metrics['flops']}\n")
            f.write(f"Params: {metrics['params']}\n")
            f.write(f"Accuracy: {metrics['accuracy']:.3f}\n")
            f.write(f"Precision (per class): {metrics['precision']}\n")
            f.write(f"Sensitivity/Recall (per class): {metrics['recall']}\n")
            f.write(f"Specificity (per class): {metrics['specificity']}\n")
            f.write(f"F1 Score (per class): {metrics['f1']}\n")
            f.write(f"AUC (per class): {metrics['auc']}\n")
            f.write(f"FLOPs: {metrics['flops']}\n")
            f.write(f"Params: {metrics['params']}\n")

    def run_all(self):
        labels, preds, probs = self.get_preds_labels_probs()
        print(f"类别: {self.class_names}")
        metrics = self.calc_metrics(labels, preds, probs)
        print(f"FLOPs: {metrics['flops']:.2e}, Params: {metrics['params']:.2e}")
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(f"Precision (per class): {metrics['precision']}")
        print(f"Sensitivity/Recall (per class): {metrics['recall']}")
        print(f"Specificity (per class): {metrics['specificity']}")
        print(f"F1 Score (per class): {metrics['f1']}")
        print(f"AUC (per class): {metrics['auc']}")
        self.output_results(labels, preds, probs, metrics)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True, choices=['resnet', 'vit', 'convnext', 'swin'])
    parser.add_argument('--num_classes', type=int, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--test_root_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--pretrained_path', type=str, default=None, help='Path to a local pretrained model or model identifier from huggingface.co/models')

    args = parser.parse_args()

    tester = Tester(
        model_type=args.model_type,
        num_classes=args.num_classes,
        ckpt_path=args.ckpt_path,
        test_root_dir=args.test_root_dir,
        batch_size=args.batch_size,
        pretrained_path=args.pretrained_path
    )
    tester.run_all()
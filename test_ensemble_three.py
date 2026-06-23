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
    def __init__(self, model_type, num_classes, prediction_files, test_root_dir, batch_size=64, pretrained_path=None):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.num_classes = num_classes
        self.prediction_files = prediction_files
        self.test_root_dir = test_root_dir
        self.batch_size = batch_size

        # self.output_dir = os.path.dirname(os.path.abspath(self.ckpt_path))
        self.output_dir = '/home/yifei/code/Med_CV/MedMamba/logs/Spatial_Task_Swin_All_Ensemble_Three'
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
        # self._load_model()

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
        # with torch.no_grad():
        #     for images, labels in self.test_loader:
        #         outputs = self.net(images.to(self.device)).logits
        #         probs = torch.softmax(outputs, dim=1)
        #         _, predicted = torch.max(probs, 1)
        #         all_labels.extend(labels.cpu().numpy())
        #         all_preds.extend(predicted.cpu().numpy())
        #         all_probs.extend(probs.cpu().numpy())
        # return np.array(all_labels), np.array(all_preds), np.array(all_probs)
        # 从 csv 文件中读取多个模型的预测结果，进行 soft voting
        for prediction_file in self.prediction_files:
            preds = []
            probs = []
            with open(prediction_file, "r") as f:
                next(f)  # 跳过表头
                for line in f:
                    parts = line.strip().split(",")
                    label = int(parts[1])
                    pred = int(parts[2])
                    prob = np.array([
                        float(parts[3].replace('[', '')),
                        float(parts[4]),
                        float(parts[5]),
                        float(parts[6].replace(']', ''))
                    ])
                
                    preds.append(pred)
                    probs.append(prob)
                    if len(all_labels) < len(self.test_dataset):
                        all_labels.append(label)
            all_preds.append(preds)
            all_probs.append(probs)
        
        soft_voted_probs = np.mean(np.array(all_probs), axis=0)
        soft_voted_preds = np.argmax(soft_voted_probs, axis=1)
        # ------------------- NEW: 合并前两类为新类别 0 -------------------
        
        # 1. 创建一个新的三分类概率数组
        # 新的类别结构: [ (原类别 0 + 原类别 1), 原类别 2, 原类别 3 ]
        # soft_voted_probs 的形状是 (N, 4)
        merged_probs = np.zeros((soft_voted_probs.shape[0], 3))

        # a. 合并原类别 0 和 1 的概率到新类别 0
        merged_probs[:, 0] = soft_voted_probs[:, 0] + soft_voted_probs[:, 1]
        
        # b. 复制原类别 2 的概率到新类别 1
        merged_probs[:, 1] = soft_voted_probs[:, 2]
        
        # c. 复制原类别 3 的概率到新类别 2
        merged_probs[:, 2] = soft_voted_probs[:, 3]
        
        # 2. 计算新的预测标签
        all_preds_array = np.array(soft_voted_preds)
        merged_preds = np.copy(all_preds_array)
        
        # 将所有原始标签 1 映射为新的标签 0
        merged_preds[merged_preds == 1] = 0
        
        # 将所有原始标签 2 映射为新的标签 1
        merged_preds[merged_preds == 2] = 1
        
        # 将所有原始标签 3 映射为新的标签 2
        merged_preds[merged_preds == 3] = 2

        
        # 3. 调整真实标签 (all_labels)
        # 原始标签: 0, 1, 2, 3
        # 目标标签: 0, 0, 1, 2
        
        all_labels_array = np.array(all_labels)
        merged_labels = np.copy(all_labels_array)

        # 将所有原始标签 1 映射为新的标签 0
        merged_labels[merged_labels == 1] = 0
        
        # 将所有原始标签 2 映射为新的标签 1
        merged_labels[merged_labels == 2] = 1
        
        # 将所有原始标签 3 映射为新的标签 2
        merged_labels[merged_labels == 3] = 2
        
        # 4. 替换返回变量
        return merged_labels, merged_preds, merged_probs

    def calc_metrics(self, labels, preds, probs):
        metrics = {}
        # metrics['flops'], metrics['params'] = self.calc_flops_params()
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
        # 保存预测结果（增加文件名）
        output_file = os.path.join(self.output_dir, "test_prediction.csv")
        # 获取所有测试集文件名
        filenames = [os.path.basename(path) for path, _ in self.test_dataset.base_dataset.samples]
        with open(output_file, "w") as f:
            f.write("Filename,Label,Prediction,Probabilities\n")
            for fname, l, p, prob in zip(filenames, labels, preds, probs):
                f.write(f"{fname},{l},{p},{prob.tolist()}\n")

        # 混淆矩阵
        cm = confusion_matrix(labels, preds, labels=range(self.num_classes))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['apical + between the roots', 'buccal', 'lingual'])
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.savefig(os.path.join(self.output_dir, "confusion_matrix.png"))
        plt.close()

        # # ROC 曲线，每个类别一个子图
        # fig, axes = plt.subplots(1, self.num_classes, figsize=(6 * self.num_classes, 5))
        # if self.num_classes == 1:
        #     axes = [axes]
        # for i in range(self.num_classes):
        #     ax = axes[i]
        #     try:
        #         RocCurveDisplay.from_predictions(
        #             (labels == i).astype(int), probs[:, i], name=self.class_names[i], ax=ax
        #         )
        #         ax.set_title(f"ROC Curve: {self.class_names[i]}")
        #     except Exception:
        #         ax.set_title(f"ROC Curve: {self.class_names[i]} (Error)")
        # plt.tight_layout()
        # plt.savefig(os.path.join(self.output_dir, "roc_curves_subplots.png"))
        # plt.close()

        # # PR 曲线，每个类别一个子图
        # fig, axes = plt.subplots(1, self.num_classes, figsize=(6 * self.num_classes, 5))
        # if self.num_classes == 1:
        #     axes = [axes]
        # for i in range(self.num_classes):
        #     ax = axes[i]
        #     try:
        #         PrecisionRecallDisplay.from_predictions(
        #             (labels == i).astype(int), probs[:, i], name=self.class_names[i], ax=ax
        #         )
        #         ax.set_title(f"PR Curve: {self.class_names[i]}")
        #     except Exception:
        #         ax.set_title(f"PR Curve: {self.class_names[i]} (Error)")
        # plt.tight_layout()
        # plt.savefig(os.path.join(self.output_dir, "pr_curves_subplots.png"))
        # plt.close()

        # # 保存指标
        # metrics_file = os.path.join(self.output_dir, "test_metrics.txt")
        # with open(metrics_file, "w") as f:
        #     f.write(f"FLOPs: {metrics['flops']}\n")
        #     f.write(f"Params: {metrics['params']}\n")
        #     f.write(f"Accuracy: {metrics['accuracy']:.3f}\n")
        #     f.write(f"Precision (per class): {metrics['precision']}\n")
        #     f.write(f"Sensitivity/Recall (per class): {metrics['recall']}\n")
        #     f.write(f"Specificity (per class): {metrics['specificity']}\n")
        #     f.write(f"F1 Score (per class): {metrics['f1']}\n")
        #     f.write(f"AUC (per class): {metrics['auc']}\n")

    def run_all(self):
        labels, preds, probs = self.get_preds_labels_probs()
        print(f"类别: {self.class_names}")
        metrics = self.calc_metrics(labels, preds, probs)
        # print(f"FLOPs: {metrics['flops']:.2e}, Params: {metrics['params']:.2e}")
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

    prediction_files= [
        # "/home/yifei/code/Med_CV/MedMamba/logs/Spatial_Task_Swin_All/bs64_ep1000_lr0.0001/2025-10-25-04-06-14/test_prediction.csv",
        "/home/yifei/code/Med_CV/MedMamba/logs/Spatial_Task_Swin_All/bs64_ep1000_lr0.0001/2025-10-25-04-32-00/test_prediction.csv",
        "/home/yifei/code/Med_CV/MedMamba/logs/Spatial_Task_Swin_All/bs64_ep1000_lr0.0001/2025-10-25-04-37-47/test_prediction.csv"
    ]

    tester = Tester(
        model_type=args.model_type,
        num_classes=args.num_classes,
        prediction_files=prediction_files,
        test_root_dir=args.test_root_dir,
        batch_size=args.batch_size,
        pretrained_path=args.pretrained_path
    )
    tester.run_all()
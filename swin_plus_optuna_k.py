from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import argparse
import optuna
import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import StratifiedKFold
from transformers import AutoImageProcessor, AutoModelForImageClassification, ViTImageProcessor, ViTForImageClassification, ConvNextImageProcessor, ConvNextForImageClassification
from tqdm import tqdm

import numpy as np

# ----------- Dataset Wrappers -----------
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

# ----------- Focal Loss -----------
import torch.nn.functional as F
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        # alpha可以是float、list、tensor
        if isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target):
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = logpt.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = pt.gather(1, target.unsqueeze(1)).squeeze(1)
        # 支持多类别alpha
        if isinstance(self.alpha, torch.Tensor):
            if self.alpha.device != input.device:
                self.alpha = self.alpha.to(input.device)
            at = self.alpha[target]
        else:
            at = self.alpha
        loss = -at * (1 - pt) ** self.gamma * logpt
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# ----------- Metric Learning Loss -----------
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0, reduction='mean'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        else:
            return losses
        
# ----------- Trainer -----------
class Trainer:    
    def objective(self, trial):
        # 搜索空间
        alpha = trial.suggest_float('alpha', 0.01, 1.0)
        gamma = trial.suggest_float('gamma', 0.5, 5.0)
        lr = trial.suggest_loguniform('lr', 1e-6, 1e-3)

        # K折交叉验证
        k_folds = 5
        dataset = self.train_dataset
        all_indices = np.arange(len(dataset))
        all_labels = np.array([label for _, label in dataset])
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        acc_list = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(all_indices, all_labels)):
            # 构建子集
            train_subset = torch.utils.data.Subset(dataset, train_idx)
            val_subset = torch.utils.data.Subset(dataset, val_idx)
            train_loader = torch.utils.data.DataLoader(train_subset, batch_size=self.batch_size, shuffle=True, num_workers=2)
            val_loader = torch.utils.data.DataLoader(val_subset, batch_size=self.batch_size, shuffle=False, num_workers=2)

            # 新建模型
            net = self.model_cls().to(self.device)
            loss_function = FocalLoss(alpha=alpha, gamma=gamma)
            optimizer = optim.Adam(net.parameters(), lr=lr)

            # 训练
            for epoch in range(3):  # 每折只训练3个epoch加速
                net.train()
                for images, labels in train_loader:
                    optimizer.zero_grad()
                    outputs = net(images.to(self.device)).logits
                    loss = loss_function(outputs, labels.to(self.device))
                    loss.backward()
                    optimizer.step()

            # 验证
            net.eval()
            acc = 0.0
            total = 0
            with torch.no_grad():
                for val_images, val_labels in val_loader:
                    outputs = net(val_images.to(self.device)).logits
                    predict_y = torch.max(outputs, dim=1)[1]
                    acc += torch.eq(predict_y, val_labels.to(self.device)).sum().item()
                    total += val_labels.size(0)
            acc_list.append(acc / total)
        mean_acc = np.mean(acc_list)
        return 1.0 - mean_acc  # optuna 最小化目标
    
    def __init__(self, model_type, model_name, num_classes, train_root_dir, val_root_dir, test_root_dir,
                 batch_size=32, epochs=100, lr=5e-6, log_dir='./logs', pretrained_path=None, gamma=2):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.model_name = model_name + "_" + "plus"
        self.num_classes = num_classes
        self.train_root_dir = train_root_dir
        self.val_root_dir = val_root_dir
        self.test_root_dir = test_root_dir
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.gamma = gamma

        self.hyperparam_str = f"bs{self.batch_size}_ep{self.epochs}_lr{self.lr}"
        self.time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self.log_dir = os.path.join(log_dir, self.model_name, self.hyperparam_str, self.time_str)
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)

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

        self._prepare_data()
        self._build_model()

    def _prepare_data(self):
        self.train_dataset = ImageFolderDataset(root=self.train_root_dir, processor=self.data_transform)
        self.val_dataset = ImageFolderDataset(root=self.val_root_dir, processor=self.data_transform)
        self.test_dataset = ImageFolderDataset(root=self.test_root_dir, processor=self.data_transform)

        # 对第二类，随机选择 10% 样本
        class_counts = np.bincount([label for _, label in self.train_dataset])
        print(f"Class distribution in training set before balancing: {class_counts}")
        if len(class_counts) > 1 and class_counts[1] > 0:
            indices_class_0 = [i for i, (_, label) in enumerate(self.train_dataset) if label == 0]
            indices_class_1 = [i for i, (_, label) in enumerate(self.train_dataset) if label == 1]
            # np.random.seed(42)
            sampled_indices_class_1 = np.random.choice(indices_class_1, size=int(0.2 * len(indices_class_1)), replace=False)
            balanced_indices = indices_class_0 + sampled_indices_class_1.tolist()
            balanced_subset = torch.utils.data.Subset(self.train_dataset, balanced_indices)
            self.train_dataset = balanced_subset
            class_counts_balanced = np.bincount([self.train_dataset[i][1] for i in range(len(self.train_dataset))])
            print(f"Class distribution in training set after balancing: {class_counts_balanced}")
        


        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4
                                                        # , sampler=sampler
                                                        )
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

        cla_dict = dict((val, key) for key, val in self.val_dataset.base_dataset.class_to_idx.items())
        with open(os.path.join(self.log_dir, 'class_indices.json'), 'w') as json_file:
            json.dump(cla_dict, json_file, indent=4)

        print(f"using {len(self.train_dataset)} images for training, {len(self.val_dataset)} images for validation, {len(self.test_dataset)} images for testing.")

    def _build_model(self):
        self.net = self.model_cls()
        self.net.to(self.device)
        # 可选: focal loss 或 triplet loss
        self.loss_function = FocalLoss(alpha=[10, 1], gamma=self.gamma)
        # self.loss_function = TripletLoss(margin=1.0)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

    def train(self):
        # K折交叉验证
        k_folds = 5
        dataset = self.train_dataset
        all_indices = np.arange(len(dataset))
        all_labels = np.array([label for _, label in dataset])
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        f1_list = []
        for fold, (train_idx, val_idx) in tqdm(enumerate(skf.split(all_indices, all_labels))):
            print(f"Fold {fold+1}/{k_folds}")
            train_subset = torch.utils.data.Subset(dataset, train_idx)
            val_subset = torch.utils.data.Subset(dataset, val_idx)
            train_loader = torch.utils.data.DataLoader(train_subset, batch_size=self.batch_size, shuffle=True, num_workers=2)
            val_loader = torch.utils.data.DataLoader(val_subset, batch_size=self.batch_size, shuffle=False, num_workers=2)

            net = self.model_cls().to(self.device)
            loss_function = FocalLoss(alpha=[10, 1], gamma=self.gamma)
            optimizer = optim.Adam(net.parameters(), lr=self.lr)

            for epoch in range(self.epochs):
                net.train()
                for images, labels in train_loader:
                    optimizer.zero_grad()
                    outputs = net(images.to(self.device)).logits
                    loss = loss_function(outputs, labels.to(self.device))
                    loss.backward()
                    optimizer.step()

            # 验证
            net.eval()
            val_label_all, val_pred_all = [], []
            with torch.no_grad():
                for val_images, val_labels in val_loader:
                    outputs = net(val_images.to(self.device)).logits
                    predict_y = torch.max(outputs, dim=1)[1]
                    val_label_all.extend(val_labels.cpu().numpy())
                    val_pred_all.extend(predict_y.cpu().numpy())
            f1_macro = f1_score(val_label_all, val_pred_all, average='macro', zero_division=0)
            f1_per_class = f1_score(val_label_all, val_pred_all, average=None, zero_division=0)
            print(f"Fold {fold+1} F1(macro): {f1_macro:.4f}")
            for i, f1c in enumerate(f1_per_class):
                print(f"Fold {fold+1} F1(class {i}): {f1c:.4f}")
            f1_list.append(f1_macro)
            if fold == 0:
                f1_class_list = [ [f1c] for f1c in f1_per_class ]
            else:
                for i, f1c in enumerate(f1_per_class):
                    f1_class_list[i].append(f1c)

        mean_f1 = np.mean(f1_list)
        print(f"K折交叉验证平均F1(macro): {mean_f1:.4f}")
        for i, f1c_list in enumerate(f1_class_list):
            print(f"K折交叉验证平均F1(class {i}): {np.mean(f1c_list):.4f}")
        return 1 - mean_f1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True, choices=['resnet', 'vit', 'convnext', 'swin'])
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--num_classes', type=int, required=True)
    parser.add_argument('--train_root_dir', type=str, required=True)
    parser.add_argument('--val_root_dir', type=str, required=True)
    parser.add_argument('--test_root_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=50)
    
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--pretrained_path', type=str, default=None, help='Path to a local pretrained model or model identifier from huggingface.co/models')

    args = parser.parse_args()

    def optuna_objective(trial):
        # 搜索空间
        # gamma = trial.suggest_float('gamma', 1.0, 5.0)
        # lr = trial.suggest_float('lr', 1e-5, 1e-4, log=True)

        trainer = Trainer(
            model_type=args.model_type,
            model_name=args.model_name,
            num_classes=args.num_classes,
            train_root_dir=args.train_root_dir,
            val_root_dir=args.val_root_dir,
            test_root_dir=args.test_root_dir,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            log_dir=args.log_dir,
            pretrained_path=args.pretrained_path,
            gamma=5
        )

        return trainer.train()


    study = optuna.create_study(direction='minimize')
    study.optimize(optuna_objective, n_trials=10)

    print('Best trial:')
    trial = study.best_trial
    print(f'  Value: {trial.value}')
    print('  Params:')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')
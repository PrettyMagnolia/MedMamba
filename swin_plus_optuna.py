import optuna

import os
import sys
import json
import time
import argparse

import torch
import torch.nn as nn
from torchvision import datasets
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset

# 导入 transformers 相关
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    ConvNextImageProcessor,
    ConvNextForImageClassification,
    ViTImageProcessor,
    ViTForImageClassification,
)

from sklearn.metrics import (
    precision_score, recall_score, roc_auc_score, f1_score,
    confusion_matrix, accuracy_score, ConfusionMatrixDisplay,
    RocCurveDisplay, PrecisionRecallDisplay
)

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

        # # 对第二类，随机选择 10% 样本
        # class_counts = np.bincount([label for _, label in self.train_dataset])
        # print(f"Class distribution in training set before balancing: {class_counts}")
        # if len(class_counts) > 1 and class_counts[1] > 0:
        #     indices_class_0 = [i for i, (_, label) in enumerate(self.train_dataset) if label == 0]
        #     indices_class_1 = [i for i, (_, label) in enumerate(self.train_dataset) if label == 1]
        #     # np.random.seed(42)
        #     sampled_indices_class_1 = np.random.choice(indices_class_1, size=int(0.2 * len(indices_class_1)), replace=False)
        #     balanced_indices = indices_class_0 + sampled_indices_class_1.tolist()
        #     balanced_subset = torch.utils.data.Subset(self.train_dataset, balanced_indices)
        #     self.train_dataset = balanced_subset
        #     class_counts_balanced = np.bincount([self.train_dataset[i][1] for i in range(len(self.train_dataset))])
        #     print(f"Class distribution in training set after balancing: {class_counts_balanced}")

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
        # self.loss_function = FocalLoss(alpha=[10, 1], gamma=self.gamma)
        # self.loss_function = TripletLoss(margin=1.0)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

    def train(self):
        best_acc = 0.0
        save_path = os.path.join(self.log_dir, 'best.pth')
        train_steps = len(self.train_loader)

        epoch_bar = tqdm(range(self.epochs), file=sys.stdout)
        for epoch in epoch_bar:
            # train
            self.net.train()
            running_loss = 0.0
            for step, data in enumerate(self.train_loader):
                images, labels = data
                self.optimizer.zero_grad()
                outputs = self.net(images.to(self.device)).logits
                loss = self.loss_function(outputs, labels.to(self.device))
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            # validate
            self.net.eval()
            # acc = 0.0
            # with torch.no_grad():
            #     for val_data in self.val_loader:
            #         val_images, val_labels = val_data
            #         outputs = self.net(val_images.to(self.device)).logits
            #         predict_y = torch.max(outputs, dim=1)[1]
            #         acc += torch.eq(predict_y, val_labels.to(self.device)).sum().item()
            # val_accurate = acc / len(self.val_dataset)

            # test
            test_acc = 0.0
            test_label_all, test_pred_all = [], []
            with torch.no_grad():
                for test_data in self.test_loader:
                    test_images, test_labels = test_data
                    outputs = self.net(test_images.to(self.device)).logits
                    predict_y = torch.max(outputs, dim=1)[1]
                    test_acc += torch.eq(predict_y, test_labels.to(self.device)).sum().item()
                    test_label_all.extend(test_labels.cpu().numpy())
                    test_pred_all.extend(predict_y.cpu().numpy())
            test_accurate = test_acc / len(self.test_dataset)

            # print(f'[epoch {epoch + 1}] train_loss: {running_loss / train_steps:.3f}  val_accuracy: {val_accurate:.3f}  test_accuracy: {test_accurate:.3f}')

            
            self.writer.add_scalar("Train/loss", running_loss / train_steps, epoch + 1)
            # self.writer.add_scalar("Val/accuracy", val_accurate, epoch + 1)
            self.writer.add_scalar("Val/test_accuracy", test_accurate, epoch + 1)

            precision = precision_score(test_label_all, test_pred_all, average=None, zero_division=0)
            recall = recall_score(test_label_all, test_pred_all, average=None, zero_division=0)
            f1 = f1_score(test_label_all, test_pred_all, average=None, zero_division=0)
            for i, p in enumerate(precision):
                self.writer.add_scalar(f'Val/precision_class_{i}', p, epoch + 1)
            self.writer.add_scalar('Val/precision_class_mean', precision.mean(), epoch + 1)
            
            for i, r in enumerate(recall):
                self.writer.add_scalar(f'Val/recall_class_{i}', r, epoch + 1)
            self.writer.add_scalar('Val/recall_class_mean', recall.mean(), epoch + 1)

            for i, f in enumerate(f1):
                self.writer.add_scalar(f'Val/f1_class_{i}', f, epoch + 1)
            self.writer.add_scalar('Val/f1_class_mean', f1.mean(), epoch + 1)

            cm = confusion_matrix(test_label_all, test_pred_all)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[i for i in range(self.num_classes)])
            disp.plot(cmap='Blues')
            self.writer.add_figure("Confusion_Matrix", disp.figure_, global_step=epoch + 1)

            if f1.mean() > best_acc:
                best_acc = f1.mean()
                torch.save(self.net.state_dict(), save_path)

            epoch_bar.set_description(f"train epoch[{epoch + 1}/{self.epochs}] loss:{running_loss / train_steps:.3f}, test_acc:{test_accurate:.3f}, test_f1:{f1.mean():.3f}")

        print('Finished Training')
        self.writer.close()

        return 1 - best_acc

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
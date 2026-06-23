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

# ----------- Trainer -----------
class Trainer:
    def __init__(self, model_type, model_name, num_classes, train_root_dir, val_root_dir, test_root_dir,
                 batch_size=32, epochs=100, lr=5e-6, log_dir='./logs', pretrained_path=None, f1_range=(0.0, 1.0), cls2_range=(0.0, 1.0)):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.model_name = model_name
        self.num_classes = num_classes
        self.train_root_dir = train_root_dir
        self.val_root_dir = val_root_dir
        self.test_root_dir = test_root_dir
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.f1_range = f1_range
        self.cls2_range = cls2_range

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
        elif model_type == 'mamba':
            from models.mamba_model import MedMambaForImageClassification, MedMambaImageProcessor
            processor_path = pretrained_path or "state-spaces/mamba-130m"
            self.data_transform = MedMambaImageProcessor.from_pretrained(processor_path)
            self.model_cls = lambda: MedMambaForImageClassification.from_pretrained(
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

        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

        cla_dict = dict((val, key) for key, val in self.train_dataset.base_dataset.class_to_idx.items())
        with open(os.path.join(self.log_dir, 'class_indices.json'), 'w') as json_file:
            json.dump(cla_dict, json_file, indent=4)

        print(f"using {len(self.train_dataset)} images for training, {len(self.val_dataset)} images for validation, {len(self.test_dataset)} images for testing.")

    def _initialize_weights(self):
        for m in self.net.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _build_model(self):
        self.net = self.model_cls()
        self.net.to(self.device)
        # self.loss_function = nn.CrossEntropyLoss()
        self.loss_function = FocalLoss(gamma=3)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

        # 添加 Kaiming 初始化
        # self._initialize_weights()

    def train(self):
        best_acc = 0.0
        save_path = os.path.join(self.log_dir, 'best.pth')
        last_path = os.path.join(self.log_dir, 'last.pth')
        train_steps = len(self.train_loader)

        for epoch in range(self.epochs):
            # train
            self.net.train()
            running_loss = 0.0
            train_bar = tqdm(self.train_loader, file=sys.stdout)
            for step, data in enumerate(train_bar):
                images, labels = data
                self.optimizer.zero_grad()
                outputs = self.net(images.to(self.device)).logits
                loss = self.loss_function(outputs, labels.to(self.device))
                loss.backward()
                
                self.optimizer.step()

                running_loss += loss.item()
                train_bar.desc = f"train epoch[{epoch + 1}/{self.epochs}] loss:{loss:.3f}"

            # validate
            self.net.eval()
            # acc = 0.0
            # with torch.no_grad():
            #     val_bar = tqdm(self.val_loader, file=sys.stdout)
            #     for val_data in val_bar:
            #         val_images, val_labels = val_data
            #         outputs = self.net(val_images.to(self.device)).logits
            #         predict_y = torch.max(outputs, dim=1)[1]
            #         acc += torch.eq(predict_y, val_labels.to(self.device)).sum().item()
            # val_accurate = acc / len(self.val_dataset)
            val_accurate = 0.0

            # test
            test_acc = 0.0
            test_label_all, test_pred_all = [], []
            test_probs = []
            with torch.no_grad():
                test_bar = tqdm(self.test_loader, file=sys.stdout)
                for test_data in test_bar:
                    test_images, test_labels = test_data
                    outputs = self.net(test_images.to(self.device)).logits
                    probs = torch.softmax(outputs, dim=1)
                    predict_y = torch.max(outputs, dim=1)[1]
                    test_acc += torch.eq(predict_y, test_labels.to(self.device)).sum().item()
                    test_label_all.extend(test_labels.cpu().numpy())
                    test_pred_all.extend(predict_y.cpu().numpy())
                    test_probs.append(probs.cpu().numpy())
            test_accurate = test_acc / len(self.test_dataset)
            
            f1 = precision_score(test_label_all, test_pred_all, average='macro', zero_division=0)
            f1_class = precision_score(test_label_all, test_pred_all, average=None, zero_division=0)

            aucs = []
            for i in range(self.num_classes):
                import numpy as np
                label_numpy = np.array(test_label_all)
                probs_numpy = np.vstack(test_probs)
                auc = roc_auc_score((label_numpy == i).astype(int), probs_numpy[:, i])
                aucs.append(auc)
            auc = sum(aucs) / len(aucs)

            print(f'[epoch {epoch + 1}] train_loss: {running_loss / train_steps:.3f}  val_accuracy: {val_accurate:.3f}  test_accuracy: {test_accurate:.3f}  test_f1: {f1:.3f}, auc: {auc:.3f}')
            print(f'class precision: {f1_class}')
            print(f'aucs: {aucs}')
            self.writer.add_scalar("Train/loss", running_loss / train_steps, epoch + 1)
            self.writer.add_scalar("Val/accuracy", val_accurate, epoch + 1)
            self.writer.add_scalar("Val/test_accuracy", test_accurate, epoch + 1)


            # if f1 > self.f1_range[0] and f1 < self.f1_range[1] and f1_class[0] != 1 and f1_class[1] != 1 and f1_class[2] != 1 and f1_class[3] != 1:
            #     if f1_class[1] >= self.cls2_range[0] and f1_class[1] <= self.cls2_range[1] and test_acc < 0.820:
            #         # best_acc = f1
            #         torch.save(self.net.state_dict(), save_path)
            #         break

            torch.save(self.net.state_dict(), last_path)

            import time
            time.sleep(1)


        print('Finished Training')
        self.writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True, choices=['resnet', 'vit', 'convnext', 'swin'])
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--num_classes', type=int, required=True)
    parser.add_argument('--train_root_dir', type=str, required=True)
    parser.add_argument('--val_root_dir', type=str, required=True)
    parser.add_argument('--test_root_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--pretrained_path', type=str, default=None, help='Path to a local pretrained model or model identifier from huggingface.co/models')
    parser.add_argument('--f1_range', type=float, nargs=2, default=[0.0, 1.0], help='Desired f1 score range to stop training')
    parser.add_argument('--cls2_range', type=float, nargs=2, default=[0.0, 1.0], help='Desired f1 score range to stop training')

    args = parser.parse_args()

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
        f1_range=args.f1_range,
        cls2_range=args.cls2_range
    )
    trainer.train()
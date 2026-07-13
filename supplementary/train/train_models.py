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

from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    ConvNextImageProcessor,
    ConvNextForImageClassification,
    ViTImageProcessor,
    ViTForImageClassification,
)

from sklearn.metrics import (
    f1_score, accuracy_score
)

import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
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


class Trainer:
    def __init__(self, model_type, model_name, num_classes, train_root_dir, val_root_dir, test_root_dir,
                 batch_size=32, epochs=100, lr=5e-6, log_dir='./logs', pretrained_path=None,
                 loss_type='ce', focal_gamma=2, focal_alpha=0.25, patience=10):
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
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.patience = patience

        self.hyperparam_str = f"bs{self.batch_size}_ep{self.epochs}_lr{self.lr}"
        self.time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self.log_dir = os.path.join(log_dir, self.model_name, self.hyperparam_str, self.time_str)
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)

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

        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

        cla_dict = dict((val, key) for key, val in self.train_dataset.base_dataset.class_to_idx.items())
        with open(os.path.join(self.log_dir, 'class_indices.json'), 'w') as json_file:
            json.dump(cla_dict, json_file, indent=4)

        print(f"using {len(self.train_dataset)} images for training, {len(self.val_dataset)} images for validation, {len(self.test_dataset)} images for testing.")

    def _build_model(self):
        self.net = self.model_cls()
        self.net.to(self.device)
        if self.loss_type == 'focal':
            self.loss_function = FocalLoss(gamma=self.focal_gamma, alpha=self.focal_alpha)
        else:
            self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

    def train(self):
        best_val_acc = 0.0
        save_path = os.path.join(self.log_dir, 'best.pth')
        last_path = os.path.join(self.log_dir, 'last.pth')
        train_steps = len(self.train_loader)
        epochs_no_improve = 0

        for epoch in range(self.epochs):
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

            self.net.eval()
            acc = 0.0
            with torch.no_grad():
                val_bar = tqdm(self.val_loader, file=sys.stdout)
                for val_data in val_bar:
                    val_images, val_labels = val_data
                    outputs = self.net(val_images.to(self.device)).logits
                    predict_y = torch.max(outputs, dim=1)[1]
                    acc += torch.eq(predict_y, val_labels.to(self.device)).sum().item()
            val_accurate = acc / len(self.val_dataset)

            print(f'[epoch {epoch + 1}] train_loss: {running_loss / train_steps:.3f}  val_accuracy: {val_accurate:.3f}')
            self.writer.add_scalar("Train/loss", running_loss / train_steps, epoch + 1)
            self.writer.add_scalar("Val/accuracy", val_accurate, epoch + 1)

            if val_accurate > best_val_acc:
                best_val_acc = val_accurate
                torch.save(self.net.state_dict(), save_path)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            torch.save(self.net.state_dict(), last_path)

            if epochs_no_improve >= self.patience:
                print(f'Early stopping at epoch {epoch + 1}, best val accuracy: {best_val_acc:.3f}')
                break

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
    parser.add_argument('--loss_type', type=str, default='ce', choices=['ce', 'focal'], help='Loss function: ce (CrossEntropyLoss) or focal (FocalLoss)')
    parser.add_argument('--focal_gamma', type=float, default=2, help='Gamma parameter for FocalLoss')
    parser.add_argument('--focal_alpha', type=float, default=0.25, help='Alpha parameter for FocalLoss')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience (epochs without improvement)')

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
        loss_type=args.loss_type,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
        patience=args.patience
    )
    trainer.train()

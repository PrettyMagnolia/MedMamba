import os
import sys
import json
import time

import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm

from MedMamba import VSSM as medmamba  # import model
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self, model_name, num_classes, train_root_dir, val_root_dir, test_root_dir,
                 batch_size=32, epochs=100, lr=5e-6, log_dir='./logs'):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.num_classes = num_classes
        self.train_root_dir = train_root_dir
        self.val_root_dir = val_root_dir
        self.test_root_dir = test_root_dir
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr

        self.hyperparam_str = f"bs{self.batch_size}_ep{self.epochs}_lr{self.lr}"
        self.time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self.log_dir = os.path.join(log_dir, self.model_name, self.hyperparam_str, self.time_str)
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.data_transform = {
            "train": transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]),
            "val": transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        }

        self._prepare_data()
        self._build_model()

    def _prepare_data(self):
        self.train_dataset = datasets.ImageFolder(root=self.train_root_dir, transform=self.data_transform["train"])
        self.val_dataset = datasets.ImageFolder(root=self.val_root_dir, transform=self.data_transform["val"])
        self.test_dataset = datasets.ImageFolder(root=self.test_root_dir, transform=self.data_transform["val"])

        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

        # 保存类别索引
        cla_dict = dict((val, key) for key, val in self.train_dataset.class_to_idx.items())
        with open(os.path.join(self.log_dir, 'class_indices.json'), 'w') as json_file:
            json.dump(cla_dict, json_file, indent=4)

        print(f"using {len(self.train_dataset)} images for training, {len(self.val_dataset)} images for validation, {len(self.test_dataset)} images for testing.")

    def _build_model(self):
        self.net = medmamba(num_classes=self.num_classes)
        self.net.to(self.device)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

    def train(self):
        best_acc = 0.0
        save_path = os.path.join(self.log_dir, 'best.pth')
        train_steps = len(self.train_loader)

        for epoch in range(self.epochs):
            # train
            self.net.train()
            running_loss = 0.0
            train_bar = tqdm(self.train_loader, file=sys.stdout)
            for step, data in enumerate(train_bar):
                images, labels = data
                self.optimizer.zero_grad()
                outputs = self.net(images.to(self.device))
                loss = self.loss_function(outputs, labels.to(self.device))
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                train_bar.desc = f"train epoch[{epoch + 1}/{self.epochs}] loss:{loss:.3f}"

            # validate
            self.net.eval()
            acc = 0.0
            with torch.no_grad():
                val_bar = tqdm(self.val_loader, file=sys.stdout)
                for val_data in val_bar:
                    val_images, val_labels = val_data
                    outputs = self.net(val_images.to(self.device))
                    predict_y = torch.max(outputs, dim=1)[1]
                    acc += torch.eq(predict_y, val_labels.to(self.device)).sum().item()
            val_accurate = acc / len(self.val_dataset)

            # test
            test_acc = 0.0
            with torch.no_grad():
                test_bar = tqdm(self.test_loader, file=sys.stdout)
                for test_data in test_bar:
                    test_images, test_labels = test_data
                    outputs = self.net(test_images.to(self.device))
                    predict_y = torch.max(outputs, dim=1)[1]
                    test_acc += torch.eq(predict_y, test_labels.to(self.device)).sum().item()
            test_accurate = test_acc / len(self.test_dataset)

            print(f'[epoch {epoch + 1}] train_loss: {running_loss / train_steps:.3f}  val_accuracy: {val_accurate:.3f}  test_accuracy: {test_accurate:.3f}')
            self.writer.add_scalar("Train/loss", running_loss / train_steps, epoch + 1)
            self.writer.add_scalar("Val/accuracy", val_accurate, epoch + 1)
            self.writer.add_scalar("Val/test_accuracy", test_accurate, epoch + 1)

            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(self.net.state_dict(), save_path)

        print('Finished Training')
        self.writer.close()

if __name__ == '__main__':
    # 训练 Contact 任务
    # trainer = Trainer(
    #     model_name="Contact_Task_MedMamba",
    #     num_classes=2,
    #     train_root_dir="/home/yifei/code/Med_CV/MedMamba/dataset/3_contact_task/train",
    #     val_root_dir="/home/yifei/code/Med_CV/MedMamba/dataset/3_contact_task/val",
    #     test_root_dir="/home/yifei/code/Med_CV/MedMamba/dataset/3_contact_task/test",
    #     batch_size=16,
    #     epochs=100,
    #     lr=1e-7
    # )
    # 训练 Spatial 任务
    trainer = Trainer(
        model_name="Spatial_Task_MedMamba",
        num_classes=4,
        train_root_dir="/home/yifei/code/Med_CV/MedMamba/dataset/4_spatial_task/train",
        val_root_dir="/home/yifei/code/Med_CV/MedMamba/dataset/4_spatial_task/val",
        test_root_dir="/home/yifei/code/Med_CV/MedMamba/dataset/4_spatial_task/test",
        batch_size=32,
        epochs=100,
        lr=5e-6
    )
    trainer.train()
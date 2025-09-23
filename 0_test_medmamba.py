
import os
import torch
from torchvision import transforms, datasets
from MedMamba import VSSM as medmamba
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, confusion_matrix, accuracy_score
import numpy as np

try:
    from thop import profile
except ImportError:
    profile = None

class MedMambaTester:
    def __init__(self, num_classes=4, ckpt_path=None, test_root_dir=None, batch_size=64):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"using {self.device} device.")
        self.num_classes = num_classes
        self.ckpt_path = ckpt_path
        self.test_root_dir = test_root_dir
        self.batch_size = batch_size
        self.data_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self._load_data()
        self._load_model()

    def _load_data(self):
        self.test_dataset = datasets.ImageFolder(root=self.test_root_dir, transform=self.data_transform)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        self.class_names = self.test_dataset.classes

    def _load_model(self):
        self.net = medmamba(num_classes=self.num_classes)
        self.net.to(self.device)
        self.net.eval()
        if self.ckpt_path and os.path.exists(self.ckpt_path):
            self.net.load_state_dict(torch.load(self.ckpt_path, map_location=self.device))
            print(f"Loaded checkpoint from {self.ckpt_path}")
        else:
            raise FileNotFoundError(f"Checkpoint not found at {self.ckpt_path}")

    def get_preds_labels_probs(self):
        all_labels = []
        all_preds = []
        all_probs = []
        with torch.no_grad():
            for images, labels in self.test_loader:
                outputs = self.net(images.to(self.device))
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(probs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        return np.array(all_labels), np.array(all_preds), np.array(all_probs)

    def calc_acc(self, labels, preds):
        acc = accuracy_score(labels, preds)
        print(f"Accuracy: {acc:.3f}")
        return acc

    def calc_precision(self, labels, preds):
        precision = precision_score(labels, preds, average=None, zero_division=0)
        print(f"Precision (per class): {precision}")
        return precision

    def calc_sensitivity(self, labels, preds):
        # Sensitivity = Recall
        se = recall_score(labels, preds, average=None, zero_division=0)
        print(f"Sensitivity/Recall (per class): {se}")
        return se

    def calc_specificity(self, labels, preds):
        # Specificity = TN / (TN + FP)
        cm = confusion_matrix(labels, preds, labels=range(self.num_classes))
        specificity = []
        for i in range(self.num_classes):
            tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
            fp = cm[:, i].sum() - cm[i, i]
            specificity.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
        print(f"Specificity (per class): {specificity}")
        return np.array(specificity)

    def calc_f1(self, labels, preds):
        f1 = f1_score(labels, preds, average=None, zero_division=0)
        print(f"F1 Score (per class): {f1}")
        return f1

    def calc_auc(self, labels, probs):
        aucs = []
        for i in range(self.num_classes):
            try:
                auc = roc_auc_score((labels == i).astype(int), probs[:, i])
            except Exception:
                auc = float('nan')
            aucs.append(auc)
        print(f"AUC (per class): {aucs}")
        return np.array(aucs)

    def calc_flops_params(self):
        if profile is None:
            print("thop 未安装，无法计算 FLOPs 和参数量。请先 pip install thop")
            return None, None
        dummy = torch.randn(1, 3, 224, 224).to(self.device)
        flops, params = profile(self.net, inputs=(dummy,), verbose=False)
        print(f"FLOPs: {flops:.2e}, Params: {params:.2e}")
        return flops, params

    def run_all(self):
        labels, preds, probs = self.get_preds_labels_probs()
        print(f"类别: {self.class_names}")
        self.calc_flops_params()
        self.calc_acc(labels, preds)
        self.calc_precision(labels, preds)
        self.calc_sensitivity(labels, preds)
        self.calc_specificity(labels, preds)
        self.calc_f1(labels, preds)
        self.calc_auc(labels, probs)

if __name__ == '__main__':
    # tester = MedMambaTester(
    #     num_classes=2,
    #     ckpt_path="/home/yifei/code/Med_CV/MedMamba/logs/Contact_Task_MedMamba/bs32_ep100_lr5e-06/2025-09-21-13-59-20/best.pth",
    #     test_root_dir="/home/yifei/code/Med_CV/MedMamba/dataset/3_contact_task/test",
    #     batch_size=64
    # )

    tester = MedMambaTester(
        num_classes=4,
        ckpt_path="/home/yifei/code/Med_CV/MedMamba/logs/Spatial_Task_MedMamba/bs32_ep100_lr5e-06/2025-09-21-15-31-09/best.pth",
        test_root_dir="/home/yifei/code/Med_CV/MedMamba/dataset/4_spatial_task/test",
        batch_size=64
    )
    tester.run_all()
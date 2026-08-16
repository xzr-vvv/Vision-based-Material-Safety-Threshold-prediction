# -*- coding: utf-8 -*-
"""
双流网络（RGB流 + 深度流）多任务训练脚本
RGB流: ResNet18(ImageNet预训练) 提取颜色/纹理特征
深度流: ResNet18(从头训练, 单通道输入) 提取几何/形状特征
两路512维特征拼接后接 分类头(6类) + 力范围回归头(min/max)
"""
import csv
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from depth_utils import estimate_depth_map

BASE_DIR = r"E:\A-触觉机器学习\图像力预测模型"
TRAIN_DIR = os.path.join(BASE_DIR, "train_images")
TRAIN_DEPTH_DIR = os.path.join(BASE_DIR, "train_depth")
TEST_DIR = os.path.join(BASE_DIR, "test_images")
TEST_DEPTH_DIR = os.path.join(BASE_DIR, "test_depth")
LABEL_FILE = os.path.join(BASE_DIR, "force_labels.csv")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "grasp_force_model_dual.pth")
RESULT_FILE = os.path.join(BASE_DIR, "training_results_dual.txt")

CLASSES = ["metal", "hard_plastic", "wood_paper", "leather_textile", "foam_soft", "fragile_glass"]
CLASS_NAMES_CN = {
    "metal": "金属刚体",
    "hard_plastic": "硬塑料",
    "wood_paper": "木材纸板",
    "leather_textile": "皮革织物",
    "foam_soft": "软质泡沫",
    "fragile_glass": "玻璃陶瓷易碎品",
}

BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
IMG_SIZE = 224
DEVICE = torch.device("cpu")

RGB_MEAN = [0.485, 0.456, 0.406]
RGB_STD = [0.229, 0.224, 0.225]


def load_force_labels():
    labels = {}
    with open(LABEL_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row["class_name"]] = (
                float(row["min_grasp_force_N"]),
                float(row["max_safe_force_N"]),
            )
    return labels


class DualStreamDataset(Dataset):
    """RGB + 深度配对数据集；深度图缺失时自动估计并缓存"""

    def __init__(self, rgb_root, depth_root, force_labels, augment=False):
        self.depth_root = depth_root
        self.force_labels = force_labels
        self.augment = augment
        self.samples = []
        for cls_idx, cls_name in enumerate(CLASSES):
            cls_dir = os.path.join(rgb_root, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            min_f, max_f = self.force_labels[cls_name]
            for fname in os.listdir(cls_dir):
                if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                if fname.lower().endswith("_depth.png"):
                    continue
                self.samples.append({
                    "rgb_path": os.path.join(cls_dir, fname),
                    "class_idx": cls_idx,
                    "min_force": min_f,
                    "max_force": max_f,
                })

    def _load_depth(self, rgb_path):
        cls = os.path.basename(os.path.dirname(rgb_path))
        fname = os.path.basename(rgb_path)
        depth_path = os.path.join(self.depth_root, cls, fname)
        if os.path.isfile(depth_path):
            return Image.open(depth_path).convert("L")
        rgb = Image.open(rgb_path).convert("RGB")
        depth = estimate_depth_map(rgb)
        cache_dir = os.path.join(self.depth_root, cls)
        os.makedirs(cache_dir, exist_ok=True)
        depth.save(os.path.join(cache_dir, fname))
        return depth

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        rgb = Image.open(s["rgb_path"]).convert("RGB")
        depth = self._load_depth(s["rgb_path"])

        rgb = TF.resize(rgb, [IMG_SIZE, IMG_SIZE])
        depth = TF.resize(depth, [IMG_SIZE, IMG_SIZE])

        if self.augment:
            if random.random() < 0.5:
                rgb = TF.hflip(rgb)
                depth = TF.hflip(depth)
            if random.random() < 0.2:
                rgb = TF.vflip(rgb)
                depth = TF.vflip(depth)
            angle = random.uniform(-15, 15)
            rgb = TF.rotate(rgb, angle)
            depth = TF.rotate(depth, angle)
            tx = random.uniform(-0.1, 0.1) * IMG_SIZE
            ty = random.uniform(-0.1, 0.1) * IMG_SIZE
            scale = random.uniform(0.9, 1.1)
            rgb = TF.affine(rgb, angle=0, translate=[tx, ty], scale=scale, shear=0)
            depth = TF.affine(depth, angle=0, translate=[tx, ty], scale=scale, shear=0)
            jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
            rgb = jitter(rgb)

        rgb_t = TF.to_tensor(rgb)
        rgb_t = TF.normalize(rgb_t, RGB_MEAN, RGB_STD)
        depth_t = TF.to_tensor(depth) * 2.0 - 1.0  # 归一化到 [-1, 1]

        return {
            "rgb": rgb_t,
            "depth": depth_t,
            "class_idx": torch.tensor(s["class_idx"], dtype=torch.long),
            "min_force": torch.tensor(s["min_force"], dtype=torch.float32),
            "max_force": torch.tensor(s["max_force"], dtype=torch.float32),
        }


class DualStreamGraspForceModel(nn.Module):
    """双流多任务模型：RGB纹理流 + 深度几何流 → 特征融合 → 分类/回归"""

    def __init__(self, num_classes=6, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.rgb_stream = models.resnet18(weights=weights)
        rgb_feat = self.rgb_stream.fc.in_features
        self.rgb_stream.fc = nn.Identity()

        self.depth_stream = models.resnet18(weights=None)
        self.depth_stream.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        depth_feat = self.depth_stream.fc.in_features
        self.depth_stream.fc = nn.Identity()

        fused = rgb_feat + depth_feat

        self.classifier = nn.Sequential(
            nn.Linear(fused, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
        self.regressor = nn.Sequential(
            nn.Linear(fused, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def forward(self, rgb, depth):
        f_rgb = self.rgb_stream(rgb)
        f_depth = self.depth_stream(depth)
        fused = torch.cat([f_rgb, f_depth], dim=1)
        return self.classifier(fused), self.regressor(fused)


def run_epoch(model, loader, criterion_cls, criterion_reg, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss = correct = total = 0
    force_mae = 0.0
    class_correct = [0] * len(CLASSES)
    class_total = [0] * len(CLASSES)

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            rgb = batch["rgb"].to(device)
            depth = batch["depth"].to(device)
            cls_idx = batch["class_idx"].to(device)
            target = torch.stack([batch["min_force"], batch["max_force"]], dim=1).to(device)

            class_logits, force_pred = model(rgb, depth)
            loss = criterion_cls(class_logits, cls_idx) + 0.5 * criterion_reg(force_pred, target)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * rgb.size(0)
            pred = class_logits.argmax(dim=1)
            correct += (pred == cls_idx).sum().item()
            total += rgb.size(0)
            force_mae += torch.mean(torch.abs(force_pred - target)).item() * rgb.size(0)
            for i in range(len(cls_idx)):
                lb = cls_idx[i].item()
                class_correct[lb] += (pred[i] == lb).item()
                class_total[lb] += 1

    per_class = {c: (class_correct[i] / class_total[i] if class_total[i] else 0.0)
                 for i, c in enumerate(CLASSES)}
    return total_loss / total, correct / total, force_mae / total, per_class


def main():
    print("=" * 60)
    print("  双流网络训练：RGB纹理流 + 深度几何流")
    print("=" * 60)

    force_labels = load_force_labels()
    train_set = DualStreamDataset(TRAIN_DIR, TRAIN_DEPTH_DIR, force_labels, augment=True)
    test_set = DualStreamDataset(TEST_DIR, TEST_DEPTH_DIR, force_labels, augment=False)
    print(f"训练集: {len(train_set)} 张 | 测试集: {len(test_set)} 张")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = DualStreamGraspForceModel(num_classes=len(CLASSES)).to(DEVICE)

    # RGB流沿用迁移学习策略：冻结底层只微调layer4；深度流从头训练不冻结
    for param in model.rgb_stream.parameters():
        param.requires_grad = False
    for param in model.rgb_stream.layer4.parameters():
        param.requires_grad = True

    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    print(f"\n开始训练 ({NUM_EPOCHS} epochs)...")
    print(f"{'Epoch':>5} {'TrainLoss':>10} {'TrainAcc':>9} {'TrainMAE':>9} {'TestAcc':>8} {'TestMAE':>8}")
    print("-" * 55)

    best_acc = 0.0
    results = []
    start = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        tr_loss, tr_acc, tr_mae, _ = run_epoch(
            model, train_loader, criterion_cls, criterion_reg, optimizer, DEVICE, train=True)
        te_loss, te_acc, te_mae, per_class = run_epoch(
            model, test_loader, criterion_cls, criterion_reg, optimizer, DEVICE, train=False)
        scheduler.step()
        print(f"{epoch:5d} {tr_loss:10.4f} {tr_acc:8.1%} {tr_mae:9.3f} {te_acc:7.1%} {te_mae:8.3f}")
        results.append((epoch, tr_loss, tr_acc, tr_mae, te_loss, te_acc, te_mae))

        if te_acc > best_acc:
            best_acc = te_acc
            torch.save({
                "epoch": epoch,
                "model_type": "dual_stream",
                "model_state_dict": model.state_dict(),
                "test_acc": te_acc,
                "test_mae": te_mae,
                "classes": CLASSES,
                "class_names_cn": CLASS_NAMES_CN,
                "force_labels": force_labels,
            }, MODEL_SAVE_PATH)

    elapsed = time.time() - start
    ckpt = torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    _, final_acc, final_mae, per_class = run_epoch(
        model, test_loader, criterion_cls, criterion_reg, optimizer, DEVICE, train=False)

    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        f.write("双流网络(RGB+深度)训练结果报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"训练轮数: {NUM_EPOCHS}\n训练耗时: {elapsed:.1f} 秒\n")
        f.write(f"训练集: {len(train_set)} | 测试集: {len(test_set)}\n\n")
        f.write(f"最佳测试集准确率: {best_acc:.2%}\n")
        f.write(f"最终测试集力预测MAE: {final_mae:.3f} N\n\n各类别准确率:\n")
        for cls, acc in per_class.items():
            f.write(f"  {cls:20s} ({CLASS_NAMES_CN[cls]}): {acc:.1%}\n")
        f.write("\n每轮详细数据:\n")
        f.write(f"{'Epoch':>5} {'TrainLoss':>10} {'TrainAcc':>9} {'TrainMAE':>9} {'TestLoss':>9} {'TestAcc':>8} {'TestMAE':>8}\n")
        for r in results:
            f.write(f"{r[0]:5d} {r[1]:10.4f} {r[2]:8.1%} {r[3]:9.3f} {r[4]:9.4f} {r[5]:7.1%} {r[6]:8.3f}\n")

    print(f"\n训练完成！耗时 {elapsed:.1f} 秒 | 最佳准确率 {best_acc:.2%} | 力MAE {final_mae:.3f} N")
    print(f"模型保存: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()

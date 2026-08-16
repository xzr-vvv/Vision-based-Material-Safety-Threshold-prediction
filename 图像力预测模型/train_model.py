# -*- coding: utf-8 -*-
"""
图像分类 + 安全抓力范围预测 多任务训练脚本
使用 ResNet18 迁移学习
输出：类别（6分类） + 最小抓取力 + 最大安全力（回归）
"""
import os
import sys

import csv
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

BASE_DIR = r"E:\A-触觉机器学习\图像力预测模型"
TRAIN_DIR = os.path.join(BASE_DIR, "train_images")
TEST_DIR = os.path.join(BASE_DIR, "test_images")
LABEL_FILE = os.path.join(BASE_DIR, "force_labels.csv")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "grasp_force_model.pth")
RESULT_FILE = os.path.join(BASE_DIR, "training_results.txt")

# 类别列表（顺序即标签编号）
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
NUM_EPOCHS = 40
LEARNING_RATE = 0.001
IMG_SIZE = 224
DEVICE = torch.device("cpu")


def load_force_labels():
    """加载力范围标签"""
    labels = {}
    with open(LABEL_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row["class_name"]] = (
                float(row["min_grasp_force_N"]),
                float(row["max_safe_force_N"])
            )
    return labels


class GraspForceDataset(Dataset):
    """抓取力数据集"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.force_labels = load_force_labels()
        self.samples = []

        for cls_idx, cls_name in enumerate(CLASSES):
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            min_f, max_f = self.force_labels[cls_name]
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append({
                        "path": os.path.join(cls_dir, fname),
                        "class_idx": cls_idx,
                        "min_force": min_f,
                        "max_force": max_f,
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample["path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return {
            "image": img,
            "class_idx": torch.tensor(sample["class_idx"], dtype=torch.long),
            "min_force": torch.tensor(sample["min_force"], dtype=torch.float32),
            "max_force": torch.tensor(sample["max_force"], dtype=torch.float32),
        }


class GraspForceModel(nn.Module):
    """多任务模型：分类 + 力范围回归"""
    def __init__(self, num_classes=6, pretrained=True):
        super().__init__()
        # 加载预训练 ResNet18
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # 去掉原始全连接层

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

        # 回归头（预测 min_force 和 max_force）
        self.regressor = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),  # 输出 [min_force, max_force]
        )

    def forward(self, x):
        features = self.backbone(x)
        class_logits = self.classifier(features)
        force_pred = self.regressor(features)
        return class_logits, force_pred


def train_one_epoch(model, dataloader, criterion_cls, criterion_reg, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    force_mae = 0.0

    for batch in dataloader:
        images = batch["image"].to(device)
        class_idx = batch["class_idx"].to(device)
        min_f = batch["min_force"].to(device)
        max_f = batch["max_force"].to(device)

        optimizer.zero_grad()
        class_logits, force_pred = model(images)

        # 分类损失
        loss_cls = criterion_cls(class_logits, class_idx)

        # 回归损失
        force_target = torch.stack([min_f, max_f], dim=1)
        loss_reg = criterion_reg(force_pred, force_target)

        # 总损失（加权）
        loss = loss_cls + 0.5 * loss_reg

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(class_logits, 1)
        correct += (predicted == class_idx).sum().item()
        total += images.size(0)
        force_mae += torch.mean(torch.abs(force_pred - force_target)).item() * images.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    avg_mae = force_mae / total
    return avg_loss, accuracy, avg_mae


def evaluate(model, dataloader, criterion_cls, criterion_reg, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    force_mae = 0.0
    class_correct = [0] * len(CLASSES)
    class_total = [0] * len(CLASSES)

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            class_idx = batch["class_idx"].to(device)
            min_f = batch["min_force"].to(device)
            max_f = batch["max_force"].to(device)

            class_logits, force_pred = model(images)

            loss_cls = criterion_cls(class_logits, class_idx)
            force_target = torch.stack([min_f, max_f], dim=1)
            loss_reg = criterion_reg(force_pred, force_target)
            loss = loss_cls + 0.5 * loss_reg

            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(class_logits, 1)
            correct += (predicted == class_idx).sum().item()
            total += images.size(0)
            force_mae += torch.mean(torch.abs(force_pred - force_target)).item() * images.size(0)

            for i in range(len(class_idx)):
                label = class_idx[i].item()
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1

    avg_loss = total_loss / total
    accuracy = correct / total
    avg_mae = force_mae / total

    per_class_acc = {}
    for i, cls in enumerate(CLASSES):
        if class_total[i] > 0:
            per_class_acc[cls] = class_correct[i] / class_total[i]
        else:
            per_class_acc[cls] = 0.0

    return avg_loss, accuracy, avg_mae, per_class_acc


def main():
    print("=" * 60)
    print("  机械臂安全抓力范围预测 - 模型训练")
    print("  任务：图像分类(6类) + 力范围回归(min/max)")
    print("=" * 60)

    # 数据增强
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 数据集
    train_dataset = GraspForceDataset(TRAIN_DIR, transform=train_transform)
    test_dataset = GraspForceDataset(TEST_DIR, transform=test_transform)
    print(f"\n训练集: {len(train_dataset)} 张")
    print(f"测试集: {len(test_dataset)} 张")
    print(f"类别数: {len(CLASSES)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 模型
    model = GraspForceModel(num_classes=len(CLASSES), pretrained=True)
    model = model.to(DEVICE)

    # 冻结 backbone 前几层，只训练高层
    for param in model.backbone.parameters():
        param.requires_grad = False
    # 解冻 layer4 和 fc
    for param in model.backbone.layer4.parameters():
        param.requires_grad = True

    # 损失函数和优化器
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    print(f"\n开始训练 ({NUM_EPOCHS} epochs)...")
    print(f"{'Epoch':>5} {'TrainLoss':>10} {'TrainAcc':>9} {'TrainMAE':>9} {'TestLoss':>9} {'TestAcc':>8} {'TestMAE':>8}")
    print("-" * 65)

    best_acc = 0.0
    results = []

    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc, train_mae = train_one_epoch(
            model, train_loader, criterion_cls, criterion_reg, optimizer, DEVICE
        )
        test_loss, test_acc, test_mae, per_class_acc = evaluate(
            model, test_loader, criterion_cls, criterion_reg, DEVICE
        )
        scheduler.step()

        print(f"{epoch:5d} {train_loss:10.4f} {train_acc:8.1%} {train_mae:9.3f} {test_loss:9.4f} {test_acc:7.1%} {test_mae:8.3f}")

        results.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_mae": train_mae,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_mae": test_mae,
        })

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "test_acc": test_acc,
                "test_mae": test_mae,
                "classes": CLASSES,
                "class_names_cn": CLASS_NAMES_CN,
                "force_labels": load_force_labels(),
            }, MODEL_SAVE_PATH)

    elapsed = time.time() - start_time

    # 加载最佳模型做最终评估
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, final_acc, final_mae, per_class_acc = evaluate(
        model, test_loader, criterion_cls, criterion_reg, DEVICE
    )

    # 保存结果
    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("  训练结果报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"训练轮数: {NUM_EPOCHS}\n")
        f.write(f"训练耗时: {elapsed:.1f} 秒\n")
        f.write(f"训练集大小: {len(train_dataset)}\n")
        f.write(f"测试集大小: {len(test_dataset)}\n\n")
        f.write(f"最佳测试集准确率: {best_acc:.2%}\n")
        f.write(f"最终测试集力预测MAE: {final_mae:.3f} N\n\n")
        f.write("各类别准确率:\n")
        for cls, acc in per_class_acc.items():
            f.write(f"  {cls:20s} ({CLASS_NAMES_CN[cls]:10s}): {acc:.1%}\n")
        f.write("\n每轮详细数据:\n")
        f.write(f"{'Epoch':>5} {'TrainLoss':>10} {'TrainAcc':>9} {'TrainMAE':>9} {'TestLoss':>9} {'TestAcc':>8} {'TestMAE':>8}\n")
        for r in results:
            f.write(f"{r['epoch']:5d} {r['train_loss']:10.4f} {r['train_acc']:8.1%} {r['train_mae']:9.3f} "
                   f"{r['test_loss']:9.4f} {r['test_acc']:7.1%} {r['test_mae']:8.3f}\n")

    print(f"\n{'='*60}")
    print(f"  训练完成！耗时 {elapsed:.1f} 秒")
    print(f"  最佳测试准确率: {best_acc:.2%}")
    print(f"  力预测 MAE: {final_mae:.3f} N")
    print(f"  模型保存: {MODEL_SAVE_PATH}")
    print(f"  结果报告: {RESULT_FILE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

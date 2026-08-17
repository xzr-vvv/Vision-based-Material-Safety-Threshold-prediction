# -*- coding: utf-8 -*-
"""Exp-Force 单流(RGB)模型训练 — 准确优先
数据: E:/A-机器学习/ExpForce_images (129 张真实机械臂腕部相机图) + 07 CSV 标注
任务: 图像 -> 类别(刚体/柔性/易碎) + 安全抓力上下限(N, 单一牛顿单位)
策略: ImageNet 预训练 ResNet18, 分层划分 80/20, 强增广, 冻结->解冻两阶段,
      余弦退火, 标签平滑, 按验证集准确率保存最优模型
"""
import csv
import os
import random
import sys
import time
from math import cos, pi

E_LIB = r"E:\Lib\site-packages"
if os.path.isdir(E_LIB) and E_LIB not in sys.path:
    sys.path.insert(0, E_LIB)

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

DATA_CSV = r"E:\A-机器学习\07_ExpForce_安全抓力范围.csv"
IMG_DIR = r"E:\A-机器学习\ExpForce_images"
MODEL_OUT = r"E:\A-触觉机器学习\ExpForce单流模型\expforce_single_stream.pth"
CLASSES = ["刚体", "柔性", "易碎"]
IMG_SIZE = 224
BATCH = 16
EPOCHS = 60
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
MIN_SCALE = 8.0   # 回归目标归一化: min_force/MIN_SCALE
MAX_SCALE = 40.0  # max_force/MAX_SCALE

random.seed(SEED)
torch.manual_seed(SEED)

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.25, 0.25, 0.25, 0.05),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


class ExpForceDataset(Dataset):
    def __init__(self, rows, tf):
        self.rows = rows
        self.tf = tf

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(os.path.join(IMG_DIR, r["image_file"])).convert("RGB")
        x = self.tf(img)
        y_cls = CLASSES.index(r["category"])
        y_force = torch.tensor([
            float(r["min_grasp_force_N"]) / MIN_SCALE,
            float(r["max_safe_force_N"]) / MAX_SCALE,
        ], dtype=torch.float32)
        return x, y_cls, y_force


class SingleStreamModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        feat = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(feat, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
        self.regressor = nn.Sequential(
            nn.Linear(feat, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        f = self.backbone(x)
        return self.classifier(f), self.regressor(f)


def load_rows():
    with open(DATA_CSV, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def stratified_split(rows, val_ratio=0.2):
    by_cls = {}
    for r in rows:
        by_cls.setdefault(r["category"], []).append(r)
    train, val = [], []
    for c, items in by_cls.items():
        items = items[:]
        random.shuffle(items)
        n_val = max(1, round(len(items) * val_ratio))
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    random.shuffle(train)
    random.shuffle(val)
    return train, val


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    err_min = err_max = 0.0
    cm = [[0] * len(CLASSES) for _ in CLASSES]
    for x, y_cls, y_force in loader:
        x, y_cls, y_force = x.to(DEVICE), y_cls.to(DEVICE), y_force.to(DEVICE)
        logits, force = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y_cls).sum().item()
        total += y_cls.size(0)
        for p, t in zip(pred.tolist(), y_cls.tolist()):
            cm[t][p] += 1
        pred_n = torch.cat([
            (force[:, 0:1] * MIN_SCALE),
            (force[:, 1:2] * MAX_SCALE),
        ], dim=1)
        true_n = torch.cat([
            (y_force[:, 0:1] * MIN_SCALE),
            (y_force[:, 1:2] * MAX_SCALE),
        ], dim=1)
        err_min += (pred_n[:, 0] - true_n[:, 0]).abs().sum().item()
        err_max += (pred_n[:, 1] - true_n[:, 1]).abs().sum().item()
    return correct / max(total, 1), err_min / max(total, 1), err_max / max(total, 1), cm


def main():
    rows = load_rows()
    train_rows, val_rows = stratified_split(rows)
    print(f"训练 {len(train_rows)} 张 / 验证 {len(val_rows)} 张, 设备 {DEVICE}")
    for c in CLASSES:
        print(f"  {c}: 训练 {sum(1 for r in train_rows if r['category']==c)}"
              f" / 验证 {sum(1 for r in val_rows if r['category']==c)}")

    train_loader = DataLoader(ExpForceDataset(train_rows, train_tf),
                              batch_size=BATCH, shuffle=True, num_workers=0)
    val_loader = DataLoader(ExpForceDataset(val_rows, val_tf),
                            batch_size=BATCH, shuffle=False, num_workers=0)

    model = SingleStreamModel().to(DEVICE)
    ce = nn.CrossEntropyLoss(label_smoothing=0.1)
    reg = nn.SmoothL1Loss()

    best_acc, best_mae = -1.0, float("inf")
    no_improve = 0
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        # 前5轮冻结骨干只训头部, 之后全部解冻
        freeze = epoch <= 5
        for p in model.backbone.parameters():
            p.requires_grad = not freeze
        lr = 3e-3 if freeze else 3e-4 * (0.5 * (1 + cos(epoch / EPOCHS * pi)))
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        model.train()
        run_loss = n = 0
        for x, y_cls, y_force in train_loader:
            x = x.to(DEVICE)
            y_cls = y_cls.to(DEVICE)
            y_force = y_force.to(DEVICE)
            opt.zero_grad()
            logits, force = model(x)
            loss = ce(logits, y_cls) + 0.5 * reg(force, y_force)
            loss.backward()
            opt.step()
            run_loss += loss.item()
            n += 1

        acc, mae_min, mae_max, cm = evaluate(model, val_loader)
        mark = ""
        if acc > best_acc or (acc == best_acc and mae_min + mae_max < best_mae):
            best_acc, best_mae = acc, mae_min + mae_max
            no_improve = 0
            os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
            torch.save({
                "model": model.state_dict(),
                "classes": CLASSES,
                "min_scale": MIN_SCALE,
                "max_scale": MAX_SCALE,
                "val_acc": acc,
                "val_mae_min": mae_min,
                "val_mae_max": mae_max,
            }, MODEL_OUT)
            mark = " <- 保存最优"
        else:
            no_improve += 1
        print(f"epoch {epoch:3d} | lr {lr:.2e} | loss {run_loss/n:.4f} | "
              f"验证准确率 {acc:.2%} | MAE min {mae_min:.2f}N max {mae_max:.2f}N{mark}")
        if no_improve >= 15:
            print("早停: 验证集 15 轮无提升")
            break

    print(f"\n训练完成, 用时 {(time.time()-t0)/60:.1f} 分钟")
    print(f"最优验证准确率: {best_acc:.2%}, 最优 MAE(min+max)/2: {best_mae/2:.2f} N")
    print("混淆矩阵 (行=真实, 列=预测):")
    print("        " + "  ".join(f"{c:>4s}" for c in CLASSES))
    for i, c in enumerate(CLASSES):
        print(f"{c:>6s}  " + "  ".join(f"{v:4d}" for v in cm[i]))


if __name__ == "__main__":
    main()

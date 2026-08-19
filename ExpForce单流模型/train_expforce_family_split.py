# -*- coding: utf-8 -*-
"""Exp-Force 单流(RGB)模型训练 — 族隔离切分版(合规基线)
与原版区别:
1. 切分按 object_family(同型物体绑定)而非逐行随机, 消除同族泄漏;
2. 回归目标仅 F_min(实测值), 不再回归 max_safe_force_N(k*f_min 估计值, 规范禁止当真值训练);
3. 列名适配规范整改后的 07 CSV(f_min_value)。
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
MODEL_OUT = r"E:\A-触觉机器学习\ExpForce单流模型\expforce_single_family.pth"
CLASSES = ["刚体", "柔性", "易碎"]
IMG_SIZE = 224
BATCH = 16
EPOCHS = 60
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
MIN_SCALE = 8.0  # F_min 归一化

# 族定义: 同一物体/同型多实例必须绑定, 防 train/val 泄漏
FAMILY = {
    "pringle_can": ["E002", "E045"],
    "paper_cup": ["E005", "E006"],
    "plastic_cup": ["E007", "E114"],
    "straw": ["E004", "E034", "E035"],
    "supp_bottle": ["E010", "E011"],
    "marshmallow": ["E013", "E048", "E049"],
    "soda_can": ["E016", "E017", "E043", "E044", "E089", "E099"],
    "roma_tomato": ["E020", "E021"],
    "strawberry": ["E026", "E073", "E074"],
    "orange": ["E027", "E028", "E029", "E103"],
    "lime": ["E009", "E031"],
    "lemon": ["E023", "E065"],
    "apple": ["E024", "E066", "E067", "E068", "E069", "E102"],
    "bell_pepper": ["E025", "E070", "E071", "E072"],
    "mandarin": ["E030", "E075"],
    "grape": ["E032", "E033", "E076", "E077", "E078", "E079"],
    "grape_tomato": ["E080", "E081", "E082", "E083", "E084", "E085"],
    "blackberry": ["E036", "E088"],
    "pocky_box": ["E038", "E110"],
    "bean_paste": ["E046", "E109", "E123"],
    "power_adapter": ["E014", "E051", "E124", "E125", "E126", "E127"],
    "candy_ball": ["E001", "E037"],
    "tape_roll": ["E052", "E101"],
    "gummies": ["E055", "E062", "E096"],
    "egg": ["E060", "E061"],
    "oxiclean": ["E104", "E106"],
    "glassware": ["E041", "E116", "E117", "E118", "E119"],
    "toothpaste_box": ["E059", "E113"],
}
ID2FAM = {oid: fam for fam, ids in FAMILY.items() for oid in ids}

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
        y_fmin = torch.tensor([float(r["f_min_value"]) / MIN_SCALE], dtype=torch.float32)
        return x, y_cls, y_fmin


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
        self.regressor = nn.Sequential(   # 仅回归 F_min
            nn.Linear(feat, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        f = self.backbone(x)
        return self.classifier(f), self.regressor(f)


def load_rows():
    with open(DATA_CSV, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def family_of(r):
    return ID2FAM.get(r["object_id"], "solo_" + r["object_id"])


def family_split(rows, val_ratio=0.2):
    """按类别分层, 以族为单位切分; 返回 train_rows, val_rows, 族清单"""
    fams_by_cls = {}
    fam_rows = {}
    for r in rows:
        fam = family_of(r)
        fam_rows.setdefault(fam, []).append(r)
        fams_by_cls.setdefault(r["category"], set()).add(fam)
    train, val = [], []
    val_fams = []
    for c, fams in fams_by_cls.items():
        fams = sorted(fams)
        random.shuffle(fams)
        n_val = max(1, round(len(fams) * val_ratio))
        chosen = fams[:n_val]
        val_fams += [(f, c) for f in chosen]
        for f in chosen:
            val.extend(fam_rows[f])
        for f in fams[n_val:]:
            train.extend(fam_rows[f])
    random.shuffle(train)
    random.shuffle(val)
    return train, val, val_fams


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    err_min = 0.0
    cm = [[0] * len(CLASSES) for _ in CLASSES]
    for x, y_cls, y_fmin in loader:
        x, y_cls = x.to(DEVICE), y_cls.to(DEVICE)
        logits, fmin = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y_cls).sum().item()
        total += y_cls.size(0)
        for p, t in zip(pred.tolist(), y_cls.tolist()):
            cm[t][p] += 1
        err_min += (fmin.squeeze(-1).cpu() * MIN_SCALE
                    - y_fmin.squeeze(-1) * MIN_SCALE).abs().sum().item()
    return correct / max(total, 1), err_min / max(total, 1), cm


def main():
    rows = load_rows()
    train_rows, val_rows, val_fams = family_split(rows)
    print(f"训练 {len(train_rows)} 张 / 验证 {len(val_rows)} 张, 设备 {DEVICE}")
    for c in CLASSES:
        print(f"  {c}: 训练 {sum(1 for r in train_rows if r['category']==c)}"
              f" / 验证 {sum(1 for r in val_rows if r['category']==c)}")
    tr_fams = {family_of(r) for r in train_rows}
    va_fams = {family_of(r) for r in val_rows}
    assert not tr_fams & va_fams, "族泄漏!"
    print(f"族数: 训练 {len(tr_fams)} / 验证 {len(va_fams)}, 无交集 ✓")
    print("验证集族:", ", ".join(f"{f}({c})" for f, c in val_fams))

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
        freeze = epoch <= 5
        for p in model.backbone.parameters():
            p.requires_grad = not freeze
        lr = 3e-3 if freeze else 3e-4 * (0.5 * (1 + cos(epoch / EPOCHS * pi)))
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        model.train()
        run_loss = n = 0
        for x, y_cls, y_fmin in train_loader:
            x, y_cls, y_fmin = x.to(DEVICE), y_cls.to(DEVICE), y_fmin.to(DEVICE)
            opt.zero_grad()
            logits, fmin = model(x)
            loss = ce(logits, y_cls) + 0.5 * reg(fmin.squeeze(-1), y_fmin.squeeze(-1))
            loss.backward()
            opt.step()
            run_loss += loss.item()
            n += 1

        acc, mae_min, cm = evaluate(model, val_loader)
        mark = ""
        if acc > best_acc or (acc == best_acc and mae_min < best_mae):
            best_acc, best_mae = acc, mae_min
            no_improve = 0
            torch.save({
                "model": model.state_dict(),
                "classes": CLASSES,
                "min_scale": MIN_SCALE,
                "split": "family_isolated",
                "val_families": val_fams,
                "val_acc": acc,
                "val_mae_fmin": mae_min,
            }, MODEL_OUT)
            mark = " <- 保存最优"
        else:
            no_improve += 1
        print(f"epoch {epoch:3d} | lr {lr:.2e} | loss {run_loss/n:.4f} | "
              f"验证准确率 {acc:.2%} | F_min MAE {mae_min:.2f}N{mark}", flush=True)
        if no_improve >= 15:
            print("早停: 验证集 15 轮无提升")
            break

    print(f"\n训练完成, 用时 {(time.time()-t0)/60:.1f} 分钟")
    print(f"=== 族隔离验证结果(诚实基线) ===")
    print(f"最优验证准确率: {best_acc:.2%}  F_min MAE: {best_mae:.2f} N")
    from collections import Counter
    val_dist = Counter(r["category"] for r in val_rows)
    maj = max(val_dist.values()) / len(val_rows)
    print(f"验证集多数类基线(瞎猜上限): {maj:.2%}  (分布 {dict(val_dist)})")
    print("混淆矩阵 (行=真实, 列=预测):")
    print("        " + "  ".join(f"{c:>4s}" for c in CLASSES))
    for i, c in enumerate(CLASSES):
        print(f"{c:>6s}  " + "  ".join(f"{v:4d}" for v in cm[i]))


if __name__ == "__main__":
    main()

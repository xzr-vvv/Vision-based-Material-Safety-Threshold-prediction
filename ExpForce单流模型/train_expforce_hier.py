# -*- coding: utf-8 -*-
"""Exp-Force 分层四类模型（方案 C）— L1 安全门控 + L2 叶分类 + 力回归

架构（对应 docs/成熟视觉模型整合方案）:
  共享 ResNet18 骨干
  L1 门控头: 易碎 vs 非易碎 (二元, 召回优先: 易碎类加权 + 阈值偏移)
  L2 易碎头: 轻脆 vs 重脆 (仅易碎样本训练; 轻脆 = 质量 < 100g)
  L2 非易碎头: 刚体 vs 柔性 (仅非易碎样本训练)
  力回归头: F_min / F_max (牛顿)
  质量头: 质量估计 (辅助特征 + 物理基线)

推理规则:
  P(易碎) > gate_thr → 易碎, 再由易碎头细分轻脆/重脆
  P(易碎) <= gate_thr → 非易碎, 由非易碎头分刚体/柔性
  gate 概率落在 [uncertain_lo, uncertain_hi] → 按易碎保守处理 + unknown 标志
"""
import argparse
import csv
import os
import random
import sys
import time
from collections import Counter
from math import cos, pi

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
DATA_CSV = os.path.join(_REPO_ROOT, "ExpForce数据集", "07_ExpForce_安全抓力范围.csv")
IMG_DIR = os.path.join(_REPO_ROOT, "ExpForce数据集", "ExpForce_images")

GATE_CLASSES = ["非易碎", "易碎"]
FRAG_CLASSES = ["重脆", "轻脆"]      # 轻脆 = 质量 < 100g (薯片/鸡蛋)
RIGID_CLASSES = ["刚体", "柔性"]
LEAF3 = ["刚体", "柔性", "易碎"]     # 与旧三类对齐(对比用)
LIGHT_FRAG_MASS_G = 100.0

IMG_SIZE = 224
BATCH = 16
EPOCHS = 60
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
MIN_SCALE = 8.0
MAX_SCALE = 40.0
MASS_SCALE = 500.0

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


def frag_sub(mass_g: float) -> str:
    return "轻脆" if float(mass_g) < LIGHT_FRAG_MASS_G else "重脆"


class HierDataset(Dataset):
    """行需含: image_file(可绝对), category, frag_sub, mass_g, f_min_value, max_safe_force_N"""

    def __init__(self, rows, tf):
        self.rows = rows
        self.tf = tf

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        p = r["image_file"]
        img = Image.open(p if os.path.isabs(p) else os.path.join(IMG_DIR, p)).convert("RGB")
        x = self.tf(img)
        y_gate = GATE_CLASSES.index("易碎" if r["category"] == "易碎" else "非易碎")
        y_frag = FRAG_CLASSES.index(r["frag_sub"]) if r["category"] == "易碎" else -1
        y_rigid = RIGID_CLASSES.index(r["category"]) if r["category"] in RIGID_CLASSES else -1
        y_force = torch.tensor([
            float(r["f_min_value"]) / MIN_SCALE,
            float(r["max_safe_force_N"]) / MAX_SCALE,
        ], dtype=torch.float32)
        y_mass = torch.tensor(float(r["mass_g"]) / MASS_SCALE, dtype=torch.float32)
        return x, y_gate, y_frag, y_rigid, y_force, y_mass


class HierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        feat = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.gate = nn.Sequential(nn.Linear(feat, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 2))
        self.frag = nn.Sequential(nn.Linear(feat, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 2))
        self.rigid = nn.Sequential(nn.Linear(feat, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 2))
        self.force = nn.Sequential(
            nn.Linear(feat, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 2))
        self.mass = nn.Sequential(nn.Linear(feat, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x):
        f = self.backbone(x)
        return self.gate(f), self.frag(f), self.rigid(f), self.force(f), self.mass(f).squeeze(-1)


def load_main_rows(csv_path):
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["frag_sub"] = frag_sub(r["mass_g"])
    return rows


def load_aug_rows(aug_csv, main_rows):
    """增强行按 object_id 关联主表补全 mass/frag_sub(增强图本身无质量列)"""
    meta = {r["object_id"]: r for r in main_rows}
    with open(aug_csv, newline="", encoding="utf-8-sig") as f:
        raw = [r for r in csv.DictReader(f) if r.get("category") in LEAF3]
    out = []
    for r in raw:
        m = meta.get(r.get("object_id"))
        if not m:
            continue
        r["mass_g"] = m["mass_g"]
        r["frag_sub"] = m["frag_sub"]
        out.append(r)
    return out


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
def evaluate(model, loader, gate_thr=0.5):
    """返回: 三类准确率, 门控易碎召回/精确, 叶(4类)准确率, 力MAE, 混淆矩阵, unknown率"""
    model.eval()
    n = correct3 = leaf_ok = 0
    gate_tp = gate_fn = gate_fp = 0
    unknown = 0
    err_min = err_max = 0.0
    cm = {c: Counter() for c in LEAF3}
    for x, y_gate, y_frag, y_rigid, y_force, y_mass in loader:
        x = x.to(DEVICE)
        y_force = y_force.to(DEVICE)
        logits_gate, logits_frag, logits_rigid, force, _ = model(x)
        p_frag = torch.softmax(logits_gate, dim=1)[:, GATE_CLASSES.index("易碎")]
        pred_leaf = []
        for i in range(x.size(0)):
            pf = p_frag[i].item()
            unknown_flag = 0.4 < pf < 0.6
            if unknown_flag:
                unknown += 1
            if pf > gate_thr:            # 判易碎(保守: 不确定也按易碎)
                sub = FRAG_CLASSES[logits_frag[i].argmax().item()]
                pred = "易碎"
                leaf = sub
            else:
                pred = RIGID_CLASSES[logits_rigid[i].argmax().item()]
                leaf = pred
            truth = "易碎" if y_gate[i].item() == 1 else RIGID_CLASSES[y_rigid[i].item()] \
                if y_rigid[i].item() >= 0 else FRAG_CLASSES[y_frag[i].item()]
            truth3 = "易碎" if y_gate[i].item() == 1 else RIGID_CLASSES[max(y_rigid[i].item(), 0)]
            # 真实叶标签
            if y_gate[i].item() == 1:
                truth_leaf = FRAG_CLASSES[y_frag[i].item()] if y_frag[i].item() >= 0 else "?"
            else:
                truth_leaf = RIGID_CLASSES[y_rigid[i].item()] if y_rigid[i].item() >= 0 else "?"
            correct3 += (pred == truth3)
            leaf_ok += (leaf == truth_leaf)
            cm[truth3][pred] += 1
            # 门控指标(仅按易碎/非易碎)
            is_frag = y_gate[i].item() == 1
            gate_pred_frag = pf > gate_thr
            gate_tp += is_frag and gate_pred_frag
            gate_fn += is_frag and not gate_pred_frag
            gate_fp += (not is_frag) and gate_pred_frag
            n += 1
        pred_n = torch.cat([force[:, 0:1] * MIN_SCALE, force[:, 1:2] * MAX_SCALE], dim=1)
        true_n = torch.cat([y_force[:, 0:1] * MIN_SCALE, y_force[:, 1:2] * MAX_SCALE], dim=1)
        err_min += (pred_n[:, 0] - true_n[:, 0]).abs().sum().item()
        err_max += (pred_n[:, 1] - true_n[:, 1]).abs().sum().item()
    recall = gate_tp / max(gate_tp + gate_fn, 1)
    precision = gate_tp / max(gate_tp + gate_fp, 1)
    return dict(acc3=correct3 / max(n, 1), gate_recall=recall, gate_precision=precision,
                leaf_acc=leaf_ok / max(n, 1), mae_min=err_min / max(n, 1),
                mae_max=err_max / max(n, 1), cm=cm, unknown_rate=unknown / max(n, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DATA_CSV)
    ap.add_argument("--aug-csv", default=None)
    ap.add_argument("--out", default=os.path.join(_HERE, "expforce_hier.pth"))
    ap.add_argument("--gate-weight", type=float, default=2.0, help="门控损失中易碎类权重(召回优先)")
    ap.add_argument("--gate-thr", type=float, default=0.35,
                    help="推理门控阈值(<0.5 偏向易碎=召回优先)")
    args = ap.parse_args()

    main_rows = load_main_rows(args.csv)
    train_rows, val_rows = stratified_split(main_rows)
    if args.aug_csv:
        aug = load_aug_rows(args.aug_csv, main_rows)
        train_rows = train_rows + aug
        random.shuffle(train_rows)
        print(f"增强: +{len(aug)} 张(全部只进训练集)")

    print(f"训练 {len(train_rows)} / 验证 {len(val_rows)} 张, 设备 {DEVICE}")
    print("训练集类别:", dict(Counter(r['category'] for r in train_rows)))
    print("验证集类别:", dict(Counter(r['category'] for r in val_rows)),
          "| 验证易碎细分:", dict(Counter(r['frag_sub'] for r in val_rows if r['category'] == '易碎')))

    train_loader = DataLoader(HierDataset(train_rows, train_tf), batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(HierDataset(val_rows, val_tf), batch_size=BATCH, shuffle=False)

    model = HierModel().to(DEVICE)
    gw = torch.tensor([1.0, args.gate_weight]).to(DEVICE)
    ce_gate = nn.CrossEntropyLoss(weight=gw, label_smoothing=0.1)
    ce_frag = nn.CrossEntropyLoss(label_smoothing=0.1)
    ce_rigid = nn.CrossEntropyLoss(label_smoothing=0.1)
    reg = nn.SmoothL1Loss()
    reg_mass = nn.SmoothL1Loss()

    best_score, no_improve = -1.0, 0
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        freeze = epoch <= 5
        for p in model.backbone.parameters():
            p.requires_grad = not freeze
        lr = 3e-3 if freeze else 3e-4 * (0.5 * (1 + cos(epoch / EPOCHS * pi)))
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        model.train()
        run_loss = nb = 0
        for x, y_gate, y_frag, y_rigid, y_force, y_mass in train_loader:
            x = x.to(DEVICE)
            y_gate = y_gate.to(DEVICE)
            y_force = y_force.to(DEVICE)
            y_mass = y_mass.to(DEVICE)
            opt.zero_grad()
            lg, lf, lr_, fo, ma = model(x)
            loss = ce_gate(lg, y_gate) + 0.5 * reg(fo, y_force) + 0.3 * reg_mass(ma, y_mass)
            # 叶分类: 各分支只用对应样本
            m_frag = y_frag >= 0
            if m_frag.any():
                loss = loss + 0.5 * ce_frag(lf[m_frag], y_frag[m_frag].to(DEVICE))
            m_rigid = y_rigid >= 0
            if m_rigid.any():
                loss = loss + 0.5 * ce_rigid(lr_[m_rigid], y_rigid[m_rigid].to(DEVICE))
            loss.backward()
            opt.step()
            run_loss += loss.item()
            nb += 1

        m = evaluate(model, val_loader, gate_thr=args.gate_thr)
        score = m["acc3"] + 0.5 * m["gate_recall"]  # 易碎召回优先的综合分
        mark = ""
        if score > best_score:
            best_score = score
            no_improve = 0
            torch.save({
                "model": model.state_dict(),
                "gate_classes": GATE_CLASSES, "frag_classes": FRAG_CLASSES,
                "rigid_classes": RIGID_CLASSES, "leaf3": LEAF3,
                "min_scale": MIN_SCALE, "max_scale": MAX_SCALE,
                "mass_scale": MASS_SCALE, "gate_thr": args.gate_thr,
                "val": {k: v for k, v in m.items() if k != "cm"},
            }, args.out)
            mark = " <- 保存最优"
        else:
            no_improve += 1
        print(f"epoch {epoch:3d} | loss {run_loss/nb:.4f} | 三类acc {m['acc3']:.0%} | "
              f"门控召回 {m['gate_recall']:.0%}/精确 {m['gate_precision']:.0%} | "
              f"叶acc {m['leaf_acc']:.0%} | MAE {m['mae_min']:.2f}/{m['mae_max']:.2f}N | "
              f"unknown {m['unknown_rate']:.0%}{mark}")
        if no_improve >= 15:
            print("早停")
            break

    print(f"\n完成, 用时 {(time.time()-t0)/60:.1f} 分钟, 最优综合分 {best_score:.3f}")
    print("混淆矩阵(行=真实, 列=预测):")
    print("        " + "  ".join(f"{c:>4s}" for c in LEAF3))
    for c in LEAF3:
        print(f"{c:>6s}  " + "  ".join(f"{m['cm'][c][p]:4d}" for p in LEAF3))


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""跨相机泛化实测: 用 Exp-Force 同相机训练的单流模型, 评测另一相机拍的 30 张实物图
类别映射: metal/hard_plastic/wood_paper->刚体, foam_soft/leather_textile->柔性, fragile_glass->易碎
"""
import os
import re
import sys

E_LIB = r"E:\Lib\site-packages"
if os.path.isdir(E_LIB) and E_LIB not in sys.path:
    sys.path.insert(0, E_LIB)
sys.path.insert(0, r"E:\A-触觉机器学习\ExpForce单流模型")

import torch
from PIL import Image
from torchvision import transforms
from train_expforce_single import SingleStreamModel, MEAN, STD, IMG_SIZE, DEVICE
from predict_expforce_single import load_model, tf

REAL_DIR = r"E:\A-触觉机器学习\图像力预测模型\real_images"
PREFIX2CLS = {
    "metal": "刚体", "hard_plastic": "刚体", "wood_paper": "刚体",
    "foam_soft": "柔性", "leather_textile": "柔性",
    "fragile_glass": "易碎",
}


def main():
    model, classes, ckpt = load_model()
    print(f"模型: 验证集准确率 {ckpt['val_acc']:.2%} (同相机留出集)\n")

    rows = []
    for f in sorted(os.listdir(REAL_DIR)):
        if not f.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        m = re.match(r"([a-z_]+?)_\d", f)
        truth = PREFIX2CLS.get(m.group(1) if m else "", None)
        if truth is None:
            continue
        img = Image.open(os.path.join(REAL_DIR, f)).convert("RGB")
        x = tf(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits, force = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        conf, idx = probs.max(dim=0)
        pred = classes[idx.item()]
        rows.append((f, truth, pred, conf.item()))

    correct = sum(1 for _, t, p, _ in rows if t == p)
    print(f"跨相机总准确率: {correct}/{len(rows)} = {correct/len(rows):.2%}\n")
    print(f"{'图片':<42s}{'真实':<5s}{'预测':<5s}{'置信度':<7s}结果")
    for f, t, p, c in rows:
        mark = "✓" if t == p else "✗"
        print(f"{f:<42s}{t:<5s}{p:<5s}{c:<7.0%}{mark}")
    print()
    for cls in classes:
        sub = [r for r in rows if r[1] == cls]
        if sub:
            hit = sum(1 for r in sub if r[1] == r[2])
            print(f"  {cls}: {hit}/{len(sub)} = {hit/len(sub):.0%}")
    low = [(f, t, p, c) for f, t, p, c in rows if c < 0.6]
    if low:
        print(f"\n低置信度(<60%) {len(low)} 张:", ", ".join(f.split('_')[0] for f, *_ in low))


if __name__ == "__main__":
    main()

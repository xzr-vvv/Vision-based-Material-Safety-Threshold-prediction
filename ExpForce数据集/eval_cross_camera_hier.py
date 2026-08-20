# -*- coding: utf-8 -*-
"""分层四类模型的跨相机泛化评测
用法: python eval_cross_camera_hier.py --model ../ExpForce单流模型/expforce_hier.pth
"""
import argparse
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "ExpForce单流模型"))

import torch
from PIL import Image
from torchvision import transforms

from train_expforce_hier import (HierModel, GATE_CLASSES, FRAG_CLASSES, RIGID_CLASSES,
                                 LEAF3, MEAN, STD, IMG_SIZE, MIN_SCALE, MAX_SCALE, DEVICE)

REAL_DIR = os.path.join(_REPO_ROOT, "图像力预测模型", "real_images")
PREFIX2CLS = {
    "metal": "刚体", "hard_plastic": "刚体", "wood_paper": "刚体",
    "foam_soft": "柔性", "leather_textile": "柔性",
    "fragile_glass": "易碎",
}

tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    ckpt = torch.load(args.model, map_location=DEVICE)
    gate_thr = ckpt.get("gate_thr", 0.5)
    model = HierModel()
    model.load_state_dict(ckpt["model"])
    model.to(DEVICE).eval()
    v = ckpt.get("val", {})
    print(f"模型(分层四类) 门控阈值 {gate_thr} | 同相机: 三类acc {v.get('acc3', 0):.0%} "
          f"门控召回 {v.get('gate_recall', 0):.0%} MAE {v.get('mae_min', 0):.2f}/{v.get('mae_max', 0):.2f}N\n")

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
            lg, lf, lr_, fo, ma = model(x)
        p_frag = torch.softmax(lg, dim=1)[0, GATE_CLASSES.index("易碎")].item()
        if p_frag > gate_thr:
            pred, leaf = "易碎", FRAG_CLASSES[lf[0].argmax().item()]
        else:
            pred = RIGID_CLASSES[lr_[0].argmax().item()]
            leaf = pred
        unknown = 0.4 < p_frag < 0.6
        rows.append((f, truth, pred, leaf, p_frag, unknown))

    correct = sum(1 for _, t, p, *_ in rows if t == p)
    print(f"跨相机总准确率(三类): {correct}/{len(rows)} = {correct/len(rows):.2%}")
    frag_recall = [r for r in rows if r[1] == "易碎"]
    hit = sum(1 for r in frag_recall if r[2] == "易碎")
    print(f"L1 门控易碎召回(安全关键): {hit}/{len(frag_recall)} = {hit/len(frag_recall):.0%}\n")
    print(f"{'图片':<40s}{'真实':<5s}{'预测':<5s}{'叶':<4s}{'P(易碎)':<8s}{'unk'}")
    for f, t, p, leaf, pf, unk in rows:
        print(f"{f:<40s}{t:<5s}{p:<5s}{leaf:<4s}{pf:<8.2f}{'?' if unk else ''}")
    print()
    for cls in LEAF3:
        sub = [r for r in rows if r[1] == cls]
        if sub:
            h = sum(1 for r in sub if r[1] == r[2])
            print(f"  {cls}: {h}/{len(sub)} = {h/len(sub):.0%}")


if __name__ == "__main__":
    main()

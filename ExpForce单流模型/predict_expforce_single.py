# -*- coding: utf-8 -*-
"""Exp-Force 单流模型预测: 输入图片 -> 类别(刚体/柔性/易碎) + 安全抓力上下限(牛顿)
用法:
  python predict_expforce_single.py 图片路径            # 单张
  python predict_expforce_single.py 文件夹路径          # 批量(递归)
"""
import os
import sys

E_LIB = r"E:\Lib\site-packages"
if os.path.isdir(E_LIB) and E_LIB not in sys.path:
    sys.path.insert(0, E_LIB)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from PIL import Image
from torchvision import transforms

from train_expforce_single import SingleStreamModel, MEAN, STD, IMG_SIZE, DEVICE

MODEL_PATH = r"E:\A-触觉机器学习\ExpForce单流模型\expforce_single_stream.pth"
EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def load_model():
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    classes = ckpt["classes"]
    model = SingleStreamModel(num_classes=len(classes))
    model.load_state_dict(ckpt["model"])
    model.to(DEVICE).eval()
    return model, classes, ckpt


tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def predict(model, classes, ckpt, image_path):
    img = Image.open(image_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits, force = model(x)
    probs = torch.softmax(logits, dim=1)[0]
    conf, idx = probs.max(dim=0)
    min_n = force[0, 0].item() * ckpt["min_scale"]
    max_n = force[0, 1].item() * ckpt["max_scale"]
    min_n = max(0.25, min_n)
    max_n = max(min_n + 0.25, max_n)
    return {
        "class": classes[idx.item()],
        "confidence": conf.item(),
        "min_N": min_n,
        "max_N": max_n,
        "probs": {c: probs[i].item() for i, c in enumerate(classes)},
    }


def collect_images(path):
    if os.path.isfile(path):
        return [path]
    out = []
    for root, _, files in os.walk(path):
        for f in sorted(files):
            if f.lower().endswith(EXTS):
                out.append(os.path.join(root, f))
    return out


def main():
    if len(sys.argv) < 2:
        print("用法: python predict_expforce_single.py <图片或文件夹>")
        return
    model, classes, ckpt = load_model()
    print(f"模型已加载 (验证集准确率 {ckpt['val_acc']:.2%})\n")
    paths = collect_images(sys.argv[1])
    if not paths:
        print("未找到图片")
        return
    correct = 0
    for p in paths:
        r = predict(model, classes, ckpt, p)
        name = os.path.basename(p)
        truth = None
        for part in p.replace("\\", "/").split("/"):
            if part in classes:
                truth = part
        hit = ""
        if truth:
            ok = r["class"] == truth
            correct += ok
            hit = f" | 真实 {truth} {'✓' if ok else '✗'}"
        print(f"{name}\n  -> {r['class']} (置信度 {r['confidence']:.0%}) | "
              f"安全抓力 {r['min_N']:.2f} ~ {r['max_N']:.2f} N{hit}")
    if any(part in classes for p in paths for part in p.replace("\\", "/").split("/")):
        print(f"\n批量准确率: {correct}/{len(paths)} = {correct/len(paths):.2%}")


if __name__ == "__main__":
    main()

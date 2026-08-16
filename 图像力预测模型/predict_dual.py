# -*- coding: utf-8 -*-
"""
双流模型预测脚本：输入图片 → 物体类别 + 安全抓力范围
优先使用同目录下配对的 <文件名>_depth.png 深度图；
没有深度图时自动从RGB估计伪深度（真深度请用RGB-D相机拍摄）
用法:
  python predict_dual.py --image 路径\图片.jpg
  python predict_dual.py --folder 路径\文件夹
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from depth_utils import estimate_depth_map, find_paired_depth

BASE_DIR = r"E:\A-触觉机器学习\图像力预测模型"
MODEL_PATH = os.path.join(BASE_DIR, "grasp_force_model_dual.pth")
DEVICE = torch.device("cpu")
IMG_SIZE = 224
RGB_MEAN = [0.485, 0.456, 0.406]
RGB_STD = [0.229, 0.224, 0.225]


def load_model():
    import sys
    sys.path.insert(0, BASE_DIR)
    from train_model_dual import DualStreamGraspForceModel

    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model = DualStreamGraspForceModel(num_classes=len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE).eval()
    return model, ckpt["classes"], ckpt["class_names_cn"], ckpt["force_labels"]


def predict_image(model, classes, names_cn, image_path, verbose=True):
    rgb = Image.open(image_path).convert("RGB")
    rgb_in = TF.resize(rgb, [IMG_SIZE, IMG_SIZE])

    depth_path = find_paired_depth(image_path)
    if depth_path:
        depth_in = Image.open(depth_path).convert("L").resize((IMG_SIZE, IMG_SIZE))
        depth_src = "配对深度图"
    else:
        depth_in = estimate_depth_map(rgb)
        depth_src = "自动估计伪深度"

    rgb_t = TF.normalize(TF.to_tensor(rgb_in), RGB_MEAN, RGB_STD).unsqueeze(0).to(DEVICE)
    depth_t = (TF.to_tensor(depth_in) * 2.0 - 1.0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits, force_pred = model(rgb_t, depth_t)
    probs = torch.softmax(logits, dim=1)[0]
    conf, idx = probs.max(dim=0)
    cls = classes[idx.item()]

    min_f = max(0.1, force_pred[0, 0].item())
    max_f = max(min_f + 0.1, force_pred[0, 1].item())

    result = {
        "predicted_class": cls,
        "predicted_class_cn": names_cn.get(cls, cls),
        "confidence": conf.item(),
        "min_grasp_force_N": min_f,
        "max_safe_force_N": max_f,
        "depth_source": depth_src,
        "probs": {c: probs[i].item() for i, c in enumerate(classes)},
    }
    if verbose:
        print_image_result(image_path, result, names_cn)
    return result


def print_image_result(path, r, names_cn):
    bar_len = 30
    print("=" * 60)
    print(f"  图片: {os.path.basename(path)}   [深度来源: {r['depth_source']}]")
    print(f"  预测类别: {r['predicted_class_cn']}  置信度 {r['confidence']:.1%}")
    print(f"  最小抓取力: {r['min_grasp_force_N']:.2f} N  (低于此力会掉落)")
    print(f"  最大安全力: {r['max_safe_force_N']:.2f} N  (超过此力会损坏)")
    print(f"  安全力范围: {r['max_safe_force_N'] - r['min_grasp_force_N']:.2f} N")
    print("-" * 60)
    for c, p in sorted(r["probs"].items(), key=lambda kv: -kv[1]):
        filled = int(bar_len * p)
        print(f"    {c:20s} {'█' * filled}{' ' * (bar_len - filled)} {p:5.1%}")


def predict_folder(model, classes, names_cn, folder):
    exts = (".png", ".jpg", ".jpeg")
    files = []
    for root, _dirs, fnames in os.walk(folder):
        for f in sorted(fnames):
            if f.lower().endswith(exts) and "_depth" not in f.lower():
                files.append(os.path.join(root, f))
    files.sort()
    if not files:
        print("文件夹里没有找到图片")
        return

    correct = total = 0
    print(f"\n找到 {len(files)} 张图片，开始批量预测...\n")
    for path in files:
        r = predict_image(model, classes, names_cn, path, verbose=False)
        name = os.path.basename(path).lower()
        gt = next((c for c in classes if c in name), None)
        mark = ""
        if gt:
            total += 1
            ok = r["predicted_class"] == gt
            correct += ok
            mark = "✓" if ok else f"✗(真实:{names_cn[gt]})"
        print(f"  {os.path.basename(path):45s} → {r['predicted_class_cn']:10s}"
              f" [{r['min_grasp_force_N']:5.2f},{r['max_safe_force_N']:6.2f}] N"
              f"  {r['confidence']:5.0%}  {mark}")

    if total:
        print(f"\n  准确率: {correct}/{total} = {correct / total:.1%}")
    print(f"\n共预测 {len(files)} 张图片")


def main():
    parser = argparse.ArgumentParser(description="双流模型预测：物体类别 + 安全抓力范围")
    parser.add_argument("--image", help="单张图片路径")
    parser.add_argument("--folder", help="批量预测文件夹")
    args = parser.parse_args()

    print("加载双流模型...")
    model, classes, names_cn, _ = load_model()
    print(f"模型加载完成，支持 {len(classes)} 个类别（RGB流 + 深度流）\n")

    if args.image:
        predict_image(model, classes, names_cn, args.image)
    elif args.folder:
        predict_folder(model, classes, names_cn, args.folder)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

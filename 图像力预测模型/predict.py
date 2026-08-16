# -*- coding: utf-8 -*-
"""
图像推理脚本：输入一张物体图片，输出物体类别 + 安全抓力范围
用法：
    python predict.py --image path/to/image.png
    python predict.py --folder path/to/folder/  (批量预测)
"""
import os
import sys

import argparse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

BASE_DIR = r"E:\A-触觉机器学习\图像力预测模型"
MODEL_PATH = os.path.join(BASE_DIR, "grasp_force_model.pth")
IMG_SIZE = 224
DEVICE = torch.device("cpu")


class GraspForceModel(nn.Module):
    """多任务模型：分类 + 力范围回归（与训练时结构一致）"""
    def __init__(self, num_classes=6):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
        self.regressor = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        features = self.backbone(x)
        class_logits = self.classifier(features)
        force_pred = self.regressor(features)
        return class_logits, force_pred


def load_model():
    """加载训练好的模型"""
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    classes = checkpoint["classes"]
    class_names_cn = checkpoint["class_names_cn"]
    force_labels = checkpoint["force_labels"]

    model = GraspForceModel(num_classes=len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return model, transform, classes, class_names_cn, force_labels


def predict_image(model, transform, classes, class_names_cn, image_path):
    """对单张图片进行预测"""
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        class_logits, force_pred = model(img_tensor)

    # 分类结果
    probs = torch.softmax(class_logits, dim=1)
    conf, pred_idx = torch.max(probs, dim=1)
    pred_class = classes[pred_idx.item()]
    confidence = conf.item()

    # 力范围预测
    min_force = max(0.1, force_pred[0, 0].item())  # 保证非负
    max_force = max(min_force + 0.1, force_pred[0, 1].item())  # max > min

    return {
        "image": image_path,
        "predicted_class": pred_class,
        "predicted_class_cn": class_names_cn.get(pred_class, pred_class),
        "confidence": confidence,
        "min_grasp_force_N": round(min_force, 2),
        "max_safe_force_N": round(max_force, 2),
        "safe_range_N": round(max_force - min_force, 2),
        "all_probabilities": {cls: round(probs[0, i].item(), 4) for i, cls in enumerate(classes)},
    }


def print_result(result):
    """美观地打印预测结果"""
    print("\n" + "=" * 55)
    print(f"  图片: {os.path.basename(result['image'])}")
    print("=" * 55)
    print(f"  预测类别:  {result['predicted_class_cn']}")
    print(f"  英文标签:  {result['predicted_class']}")
    print(f"  置信度:    {result['confidence']:.1%}")
    print("-" * 55)
    print(f"  最小抓取力: {result['min_grasp_force_N']:>6.2f} N  (低于此力会掉落)")
    print(f"  最大安全力: {result['max_safe_force_N']:>6.2f} N  (超过此力会损坏)")
    print(f"  安全力范围: {result['safe_range_N']:>6.2f} N")
    print("-" * 55)
    print("  各类别概率:")
    for cls, prob in result["all_probabilities"].items():
        bar = "█" * int(prob * 30)
        print(f"    {cls:20s} {bar:30s} {prob:6.1%}")
    print("=" * 55 + "\n")


def main():
    parser = argparse.ArgumentParser(description="图像抓取安全力范围预测")
    parser.add_argument("--image", type=str, help="单张图片路径")
    parser.add_argument("--folder", type=str, help="图片文件夹路径（批量预测）")
    args = parser.parse_args()

    if not args.image and not args.folder:
        # 默认使用测试集第一张图片
        test_dir = os.path.join(BASE_DIR, "test_images", "metal")
        if os.path.isdir(test_dir):
            files = [f for f in os.listdir(test_dir) if f.endswith(".png")]
            if files:
                args.image = os.path.join(test_dir, files[0])
                print(f"未指定图片，使用默认测试图: {args.image}")
            else:
                print("错误：请使用 --image 或 --folder 指定图片")
                return
        else:
            print("错误：请使用 --image 或 --folder 指定图片")
            return

    print("加载模型...")
    model, transform, classes, class_names_cn, force_labels = load_model()
    print(f"模型加载完成，支持 {len(classes)} 个类别")

    if args.image:
        result = predict_image(model, transform, classes, class_names_cn, args.image)
        print_result(result)

    if args.folder:
        image_ext = (".png", ".jpg", ".jpeg", ".bmp", ".gif")
        images = []
        for root, _, files in os.walk(args.folder):
            for f in files:
                if f.lower().endswith(image_ext):
                    images.append(os.path.join(root, f))
        images.sort()
        print(f"\n找到 {len(images)} 张图片，开始批量预测...\n")

        results = []
        correct = 0
        total = 0
        for fpath in images:
            fname = os.path.basename(fpath)
            result = predict_image(model, transform, classes, class_names_cn, fpath)
            results.append(result)

            # 尝试从文件名推断真实类别
            true_cls = None
            for cls in classes:
                if cls in fname.lower():
                    true_cls = cls
                    break
            if true_cls:
                total += 1
                if result["predicted_class"] == true_cls:
                    correct += 1

            print(f"  {fname:30s} → {result['predicted_class_cn']:10s} "
                  f"[{result['min_grasp_force_N']:5.2f}, {result['max_safe_force_N']:5.2f}] N  "
                  f"置信度 {result['confidence']:.0%}")

        if total > 0:
            print(f"\n  准确率: {correct}/{total} = {correct/total:.1%}")
        print(f"\n共预测 {len(results)} 张图片")


if __name__ == "__main__":
    main()

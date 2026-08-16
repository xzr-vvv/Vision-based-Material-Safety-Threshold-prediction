# -*- coding: utf-8 -*-
"""
为 train_images / test_images 里的全部合成图生成配对深度图
输出：train_depth/<类别>/<原名>.png、test_depth/<类别>/<原名>.png
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image

from depth_utils import estimate_depth_map

BASE_DIR = r"E:\A-触觉机器学习\图像力预测模型"

PAIRS = [
    (os.path.join(BASE_DIR, "train_images"), os.path.join(BASE_DIR, "train_depth")),
    (os.path.join(BASE_DIR, "test_images"), os.path.join(BASE_DIR, "test_depth")),
]


def main():
    total = 0
    for src_root, dst_root in PAIRS:
        if not os.path.isdir(src_root):
            print("skip missing:", src_root)
            continue
        for cls in sorted(os.listdir(src_root)):
            cls_dir = os.path.join(src_root, cls)
            if not os.path.isdir(cls_dir):
                continue
            out_dir = os.path.join(dst_root, cls)
            os.makedirs(out_dir, exist_ok=True)
            n = 0
            for fname in sorted(os.listdir(cls_dir)):
                if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                rgb = Image.open(os.path.join(cls_dir, fname)).convert("RGB")
                depth = estimate_depth_map(rgb, is_synthetic=True)
                depth.save(os.path.join(out_dir, fname))
                n += 1
            total += n
            print(f"{dst_root}\\{cls}: {n}")
    print("done, total", total)


if __name__ == "__main__":
    main()

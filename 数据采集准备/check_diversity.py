# -*- coding: utf-8 -*-
"""检查每个物体文件夹内图片的视角多样性:
- 计算每张图 32x32 灰度缩略图的 pairwise L1 距离
- 标记近重复对(距离过小)和整体多样性过低的文件夹
"""
import os
import sys

sys.path.insert(0, r"E:\Lib\site-packages")
import numpy as np
from PIL import Image

RGB = r"E:\A-触觉机器学习\RGB_dataset"
LEAVES = ["轻脆", "重脆", "刚体", "柔性"]


def thumb_vec(path):
    img = Image.open(path).convert("L").resize((32, 32))
    return np.asarray(img, dtype=np.float32) / 255.0


def main():
    print(f"{'物体':<12}{'类别':<6}{'图数':>4}{'均值距离':>10}{'最小距离':>10}{'近重复对':>8}")
    flagged = []
    for leaf in LEAVES:
        leaf_dir = os.path.join(RGB, leaf)
        if not os.path.isdir(leaf_dir):
            continue
        for oid in sorted(os.listdir(leaf_dir)):
            obj_dir = os.path.join(leaf_dir, oid)
            if not os.path.isdir(obj_dir):
                continue
            files = sorted(f for f in os.listdir(obj_dir)
                           if f.lower().endswith((".png", ".jpg", ".jpeg")))
            if len(files) < 2:
                continue
            vecs = [thumb_vec(os.path.join(obj_dir, f)).ravel() for f in files]
            n = len(vecs)
            dists = []
            for i in range(n):
                for j in range(i + 1, n):
                    dists.append(np.abs(vecs[i] - vecs[j]).mean())
            dists = np.array(dists)
            near = int((dists < 0.04).sum())
            mean_d = dists.mean()
            print(f"{oid:<12}{leaf:<6}{n:>4}{mean_d:>10.4f}{dists.min():>10.4f}{near:>8}")
            if near > 0 or mean_d < 0.10:
                flagged.append((oid, leaf, n, mean_d, near))
    print(f"\n近重复/低多样性文件夹数: {len(flagged)}")
    for oid, leaf, n, md, near in flagged:
        print(f"  {oid} ({leaf}) n={n} mean={md:.4f} near_dup={near}")


if __name__ == "__main__":
    main()

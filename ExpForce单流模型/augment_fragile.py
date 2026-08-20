# -*- coding: utf-8 -*-
"""易碎物体背景替换式域增强（P0-a）

目的: 跨相机测试易碎类 0% 的主因是域差(素色灰背景固定俯拍 vs 真实多样背景)。
方法: 不做物体抠图——把"颜色接近背景分布"的区域(背景 + 玻璃透光区)替换为
     程序化生成的多样化背景(木纹/桌面/布纹/渐变/暗色), 物体轮廓与阴影保留
     原像素, 过渡带 alpha 渐变; 再做整图小旋转/裁剪/颜色抖动。
     玻璃透光区自然透出新背景, 符合透明物成像特性。
规范: 增强图只进训练集(训练脚本 --aug-csv 控制), 验证/测试保持原始图。
输出: <OUT_DIR>/fragile/<原图名>_augNN.png + <OUT_DIR>/augmented_labels.csv
"""
import csv
import os
import sys

import numpy as np
from PIL import Image, ImageEnhance

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
DATA_CSV = os.path.join(_REPO_ROOT, "ExpForce数据集", "07_ExpForce_安全抓力范围.csv")
IMG_DIR = os.path.join(_REPO_ROOT, "ExpForce数据集", "ExpForce_images")
OUT_DIR = "/media/jesse/4AEE6803C369DAA3/safetyvtla_A1/augmented"

N_PER_OBJECT = 15   # 每物体变体数
SEED = 42

rng = np.random.default_rng(SEED)


# ---------- 1. 背景统计 ----------

def bg_stats(arr):
    """四边内缩带采样, 稳健估计背景色分布 (μ, σ)"""
    band = 6
    strips = [arr[:band].reshape(-1, 3), arr[-band:].reshape(-1, 3),
              arr[:, :band].reshape(-1, 3), arr[:, -band:].reshape(-1, 3)]
    px = np.concatenate(strips)
    med = np.median(px, axis=0)
    mad = np.median(np.abs(px - med), axis=0) * 1.4826 + 1e-3
    sigma = float(max(mad.max(), 6.0))  # 下限防过严
    return med, sigma


# ---------- 2. 程序化背景 ----------

def _noise(h, w, amp):
    return np.clip(rng.normal(0, amp, (h, w, 1)), -60, 60)


def make_background(w, h):
    kind = int(rng.integers(0, 5))
    if kind == 0:  # 木纹桌面
        base = np.array([150, 110, 75]) + rng.normal(0, 18, 3)
        x = np.linspace(0, 4 * np.pi, w)
        stripe = 10 * np.sin(x * 6 + rng.uniform(0, 6)) + 6 * np.sin(x * 17)
        img = np.tile(base, (h, w, 1)) + stripe[None, :, None] * 0.6
        img += _noise(h, w, 6)
        img[int(h * rng.uniform(0.2, 0.8))] -= 30
    elif kind == 1:  # 柔和纯色桌面
        palette = [(188, 186, 182), (222, 214, 200), (176, 188, 200),
                   (196, 206, 188), (160, 160, 165), (206, 196, 186)]
        base = np.array(palette[int(rng.integers(len(palette)))], float)
        img = np.tile(base, (h, w, 1)) + _noise(h, w, 4)
        img += np.linspace(-12, 12, h)[:, None, None]
    elif kind == 2:  # 布纹
        base = np.array([120, 128, 150]) + rng.normal(0, 25, 3)
        img = np.tile(base, (h, w, 1)) + _noise(h, w, 10)
        grid = np.zeros((h, w, 1))
        grid[::3] += 8
        grid[:, ::3] += 8
        img += grid
    elif kind == 3:  # 双色渐变
        c1 = rng.uniform(60, 210, 3)
        c2 = rng.uniform(60, 210, 3)
        t = np.linspace(0, 1, h)[:, None, None]
        img = c1[None, None] * (1 - t) + c2[None, None] * t + _noise(h, w, 5)
    else:  # 暗色工业台面
        base = np.array([58, 58, 62]) + rng.normal(0, 10, 3)
        img = np.tile(base, (h, w, 1))
        yy, xx = np.mgrid[0:h, 0:w]
        r = np.sqrt(((yy - h / 2) / (h / 2)) ** 2 + ((xx - w / 2) / (w / 2)) ** 2)
        img += (30 * (1 - np.clip(r, 0, 1)))[:, :, None] + _noise(h, w, 5)
    return np.clip(img, 0, 255).astype(np.float32)


# ---------- 3. 背景替换合成 ----------

def compose(src_img):
    """背景替换 + 整图几何/光度扰动, 返回 RGB PIL 图"""
    arr = np.asarray(src_img.convert("RGB")).astype(np.float32)
    h, w = arr.shape[:2]
    med, sigma = bg_stats(arr)
    dist = np.sqrt(((arr - med[None, None]) ** 2).sum(-1))

    # 物体 alpha: 颜色距离线性渐变 (背景=0 物体=255), 物体与阴影保留
    lo, hi = 1.5 * sigma, 2.8 * sigma
    alpha = np.clip((dist - lo) / max(hi - lo, 1e-3), 0, 1)[:, :, None]

    bg = make_background(w, h)
    out = arr * alpha + bg * (1 - alpha)

    # 整图光度扰动
    out = out * rng.uniform(0.88, 1.12) + rng.uniform(-10, 10)
    out = np.clip(out, 0, 255).astype(np.uint8)
    img = Image.fromarray(out, "RGB")

    # 小旋转 + 随机裁剪窗口(等效视角/平移扰动)
    img = img.rotate(rng.uniform(-6, 6), resample=Image.BICUBIC,
                     fillcolor=tuple(int(v) for v in med))
    scale = rng.uniform(0.9, 1.0)
    cw, ch = int(w * scale), int(h * scale)
    x0 = int(rng.integers(0, max(1, w - cw + 1)))
    y0 = int(rng.integers(0, max(1, h - ch + 1)))
    img = img.crop((x0, y0, x0 + cw, y0 + ch)).resize((w, h), Image.LANCZOS)

    img = ImageEnhance.Color(img).enhance(rng.uniform(0.85, 1.12))
    return img


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="全类别增强(默认仅易碎)")
    ap.add_argument("-n", type=int, default=N_PER_OBJECT, help="每物体变体数")
    args = ap.parse_args()

    with open(DATA_CSV, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not args.all:
        rows = [r for r in rows if r["category"] == "易碎"]
    subdir = "all" if args.all else "fragile"
    csv_name = "augmented_all_labels.csv" if args.all else "augmented_labels.csv"
    print(f"增强对象: {'全部 ' + str(len(rows)) if args.all else '易碎 ' + str(len(rows))} 个,"
          f" 每物体 {args.n} 张背景替换变体")

    os.makedirs(os.path.join(OUT_DIR, subdir), exist_ok=True)
    aug_rows = []
    for r in rows:
        src = os.path.join(IMG_DIR, r["image_file"])
        img = Image.open(src).convert("RGB")
        stem = os.path.splitext(r["image_file"])[0]
        for k in range(1, args.n + 1):
            out_path = os.path.join(OUT_DIR, subdir, f"{stem}_aug{k:02d}.png")
            compose(img).save(out_path)
            aug_rows.append({
                "object_id": r["object_id"],
                "object_name": r["object_name"],
                "category": r["category"],
                "f_min_value": r["f_min_value"],
                "max_safe_force_N": r["max_safe_force_N"],
                "image_file": out_path,
                "augmentation": "bg_replace",
            })
        print(f"  {r['object_name'][:32]:<34} +{args.n}")

    csv_path = os.path.join(OUT_DIR, csv_name)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=list(aug_rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(aug_rows)
    print(f"\n完成: {len(aug_rows)} 张增强图 -> {OUT_DIR}/{subdir}/")
    print(f"标签: {csv_path}")


if __name__ == "__main__":
    sys.exit(main())

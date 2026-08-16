# -*- coding: utf-8 -*-
"""
深度图工具：从RGB图像估计深度图
- 合成图（浅色近纯色背景）：按背景色分割物体掩膜
- 真实照片：用"中心先验 + 边缘能量"估计显著性掩膜
掩膜确定后，用腐蚀迭代构造"中心厚边缘薄"的圆顶状深度图
"""
import numpy as np
from PIL import Image, ImageFilter

IMG_SIZE = 224
BG_DEPTH = 30      # 背景深度值（远）
OBJ_DEPTH_BASE = 120  # 物体表面基础深度（近）
DEPTH_MAX = 255


def _mask_from_synthetic(img):
    """合成图：背景为高亮度低饱和的浅灰色"""
    arr = np.asarray(img).astype(np.int16)
    brightness = arr.mean(axis=2)
    saturation = arr.max(axis=2) - arr.min(axis=2)
    bg = (brightness > 210) & (saturation < 14)
    return ~bg


def _mask_from_real(img):
    """真实照片：中心先验 + 边缘能量 + 暗区偏好"""
    gray = img.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.GaussianBlur(6))
    e = np.asarray(edges).astype(np.float32)
    e = (e - e.min()) / (e.max() - e.min() + 1e-6)
    g = np.asarray(gray.filter(ImageFilter.GaussianBlur(4))).astype(np.float32)
    g = (g - g.min()) / (g.max() - g.min() + 1e-6)

    h, w = e.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2, h / 2
    dist = np.sqrt(((xx - cx) / (w / 2)) ** 2 + ((yy - cy) / (h / 2)) ** 2)
    center = np.clip(1.3 - dist, 0, 1) ** 2

    score = center * (0.7 * e + 0.3 * (1.0 - g))
    thr = np.percentile(score, 78)
    mask = score > max(thr, 1e-4)
    m = Image.fromarray((mask * 255).astype(np.uint8))
    m = m.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(7))
    return np.asarray(m) > 127


def _fill_mask_holes(mask):
    """把掩膜内部空洞填上（取每行首尾物体像素之间的区域）"""
    filled = mask.copy()
    h, w = mask.shape
    for y in range(h):
        xs = np.where(mask[y])[0]
        if len(xs) >= 2:
            filled[y, xs[0]:xs[-1] + 1] = True
    return filled


def _dome_depth(mask):
    """腐蚀迭代构造圆顶状厚度：中心存活久→深度大"""
    m = Image.fromarray((mask * 255).astype(np.uint8))
    thickness = np.zeros(mask.shape, dtype=np.float32)
    level = 1.0
    cur = m
    while True:
        alive = np.asarray(cur) > 127
        if not alive.any():
            break
        thickness[alive] = level
        cur = cur.filter(ImageFilter.MinFilter(5))
        level += 1.0
    tmax = thickness.max()
    if tmax > 0:
        thickness /= tmax
    depth = np.full(mask.shape, BG_DEPTH, dtype=np.float32)
    depth[mask] = OBJ_DEPTH_BASE + (DEPTH_MAX - OBJ_DEPTH_BASE) * thickness[mask]
    return Image.fromarray(np.clip(depth, 0, 255).astype(np.uint8))


def estimate_depth_map(rgb_img, is_synthetic=None):
    """输入PIL RGB图，返回224x224的深度图（PIL L模式）"""
    if rgb_img.size != (IMG_SIZE, IMG_SIZE):
        rgb_img = rgb_img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    if is_synthetic is None:
        is_synthetic = _looks_synthetic(img=rgb_img)
    mask = _mask_from_synthetic(rgb_img) if is_synthetic else _mask_from_real(rgb_img)
    mask = _fill_mask_holes(mask)
    return _dome_depth(mask)


def _looks_synthetic(img):
    """合成图四角应接近同一浅灰色"""
    arr = np.asarray(img).astype(np.int16)
    h, w, _ = arr.shape
    k = 12
    corners = [arr[:k, :k], arr[:k, -k:], arr[-k:, :k], arr[-k:, -k:]]
    means = [c.reshape(-1, 3).mean(axis=0) for c in corners]
    bright = all(m.mean() > 210 for m in means)
    consistent = max(
        np.abs(means[i] - means[j]).max() for i in range(4) for j in range(i + 1, 4)
    ) < 15
    return bright and consistent


def find_paired_depth(rgb_path):
    """查找与RGB图片配对的深度图（同目录 <文件名>_depth.png）"""
    import os
    stem, _ = os.path.splitext(rgb_path)
    for cand in (stem + "_depth.png", stem + "_depth.jpg"):
        if os.path.isfile(cand):
            return cand
    return None

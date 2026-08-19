# -*- coding: utf-8 -*-
"""把 ycbv 16bit 深度图转成人眼可看的伪彩色预览, 保存到 E:\A-触觉机器学习\YCB深度图预览
每个场景挑 2 帧, 输出: RGB | 伪彩深度 | 叠加 三联图
"""
import json
import os

import numpy as np
from PIL import Image

TEST = r"E:\A-触觉机器学习\datasets\ycb_rgbd\test"
OUT = r"E:\A-触觉机器学习\YCB深度图预览"
os.makedirs(OUT, exist_ok=True)


def colorize_depth(arr_mm):
    """16bit 毫米深度 -> 伪彩色 (0=黑, 近=暖, 远=冷)"""
    v = arr_mm.astype(np.float64)
    valid = v > 0
    out = np.zeros((v.shape[0], v.shape[1], 3), dtype=np.uint8)
    if valid.sum() == 0:
        return Image.fromarray(out)
    lo, hi = np.percentile(v[valid], [2, 98])
    dn = np.clip((v - lo) / (hi - lo + 1e-6), 0, 1)
    # turbo 风格近似渐变: 蓝(远)->绿->黄->红(近) 反转为近暖远冷
    r = np.clip(1.5 - np.abs(4 * dn - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * dn - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * dn - 1), 0, 1)
    out[..., 0] = (r * 255 * valid).astype(np.uint8)
    out[..., 1] = (g * 255 * valid).astype(np.uint8)
    out[..., 2] = (b * 255 * valid).astype(np.uint8)
    return Image.fromarray(out)


def overlay(rgb, arr_mm, alpha=0.55):
    dep = colorize_depth(arr_mm)
    a = np.array(rgb, dtype=np.float64)
    b = np.array(dep, dtype=np.float64)
    return Image.fromarray((a * (1 - alpha) + b * alpha).astype(np.uint8))


n = 0
for sc in sorted(os.listdir(TEST)):
    sc_dir = os.path.join(TEST, sc)
    if not os.path.isdir(os.path.join(sc_dir, "rgb")):
        continue
    cam = json.load(open(os.path.join(sc_dir, "scene_camera.json"), encoding="utf-8"))
    frames = sorted(f[:-4] for f in os.listdir(os.path.join(sc_dir, "rgb")) if f.endswith(".png"))
    for fr in frames[:: max(len(frames) // 2, 1)][:2]:
        rgb = Image.open(os.path.join(sc_dir, "rgb", fr + ".png")).convert("RGB")
        dep_raw = np.array(Image.open(os.path.join(sc_dir, "depth", fr + ".png")))
        dep_mm = dep_raw * cam[str(int(fr))]["depth_scale"]  # 原始值(0.1mm)×0.1 = 毫米
        # 三联图: RGB | 伪彩深度 | 叠加
        trio = Image.new("RGB", (rgb.width * 3 + 20, rgb.height), (30, 30, 30))
        trio.paste(rgb, (0, 0))
        trio.paste(colorize_depth(dep_mm), (rgb.width + 10, 0))
        trio.paste(overlay(rgb, dep_mm), (rgb.width * 2 + 20, 0))
        trio.save(os.path.join(OUT, f"{sc}_{fr}_RGB_深度_叠加.jpg"), quality=90)
        n += 1
print(f"已保存 {n} 张三联预览图 -> {OUT}")

# -*- coding: utf-8 -*-
"""从 BOP ycbv 测试集抽样生成项目约定配对 (rgb.png + rgb_depth.png, 16bit 毫米)
BOP 深度单位 0.1mm (depth_scale=0.1) -> 项目约定毫米: mm = value * 0.1
"""
import json
import os
import random

import numpy as np
from PIL import Image

SRC = r"E:\A-触觉机器学习\datasets\ycb_rgbd\test"
DST = r"E:\A-触觉机器学习\datasets\ycb_rgbd\pairs"
N_PER_SCENE = 5

random.seed(0)
os.makedirs(DST, exist_ok=True)

scenes = sorted(os.listdir(SRC))
made = 0
for sc in scenes:
    sc_dir = os.path.join(SRC, sc)
    rgb_dir = os.path.join(sc_dir, "rgb")
    if not os.path.isdir(rgb_dir):
        continue
    frames = sorted(f[:-4] for f in os.listdir(rgb_dir) if f.endswith(".png"))
    cam = json.load(open(os.path.join(sc_dir, "scene_camera.json"), encoding="utf-8"))
    for fr in random.sample(frames, min(N_PER_SCENE, len(frames))):
        scale = cam[str(int(fr))]["depth_scale"]  # json 键是无前导零的整数帧号
        rgb = Image.open(os.path.join(rgb_dir, fr + ".png")).convert("RGB")
        dep = np.array(Image.open(os.path.join(sc_dir, "depth", fr + ".png")), dtype=np.float64)
        dep_mm = (dep * scale).round().astype(np.uint16)   # 0.1mm -> mm
        stem = f"ycbv_{sc}_{fr}"
        rgb.save(os.path.join(DST, stem + ".png"))
        Image.fromarray(dep_mm).save(os.path.join(DST, stem + "_depth.png"))
        made += 1
print(f"已生成 {made} 对 -> {DST}")

v = np.array(Image.open(os.path.join(DST, os.listdir(DST)[0].replace('.png','') + '_depth.png')))
nnz = v[v > 0]
print(f"校验: dtype={v.dtype} 有效深度 {nnz.min()}~{nnz.max()} mm (即 {nnz.min()/1000:.2f}~{nnz.max()/1000:.2f} m)")

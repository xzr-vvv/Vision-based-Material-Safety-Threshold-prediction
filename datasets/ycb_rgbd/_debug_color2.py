# -*- coding: utf-8 -*-
"""场景55帧22: 6个实例掩码区域的平均颜色全查 + 与 scene_gt px_count 交叉核对"""
import io
import json
import os
import sys

import numpy as np
from PIL import Image

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

sdir = os.path.join(r"E:\A-触觉机器学习\datasets\ycb_rgbd\test", "000055")
frame = 22
NAMES = {1: "sauce_bottle(红标)", 3: "cracker_box(红盒)", 4: "sugar_box(白蓝)",
         12: "strawberry(红)", 14: "pear(黄绿)", 16: "apple(红)"}

im = np.array(Image.open(os.path.join(sdir, "rgb", f"{frame:06d}.png")).convert("RGB"))
with open(os.path.join(sdir, "scene_gt.json"), encoding="utf-8") as f:
    gt = json.load(f)
with open(os.path.join(sdir, "scene_gt_info.json"), encoding="utf-8") as f:
    info = json.load(f)

anns = gt[str(frame)]
print(f"帧{frame} 标注物体: {[a['obj_id'] for a in anns]}")
for i, a in enumerate(anns):
    mask = np.array(Image.open(os.path.join(sdir, "mask_visib", f"{frame:06d}_{i:06d}.png"))) > 0
    px_img = int(mask.sum())
    px_info = info[str(frame)][i].get("px_count_visib")
    if mask.any():
        mean_rgb = im[mask].mean(axis=0).astype(int)
    else:
        mean_rgb = "-"
    print(f"  实例{i} {NAMES.get(a['obj_id'], a['obj_id'])}: "
          f"掩码像素{px_img} vs info像素{px_info}  区域平均RGB {mean_rgb}")

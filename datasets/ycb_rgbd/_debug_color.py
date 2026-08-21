# -*- coding: utf-8 -*-
"""用掩码区域平均颜色验证 rgb 与标注对齐:
apple 掩码区域应为红色; 另测 banana(黄) 和 wood_block(木色) 作对照"""
import io
import json
import os
import sys

import numpy as np
from PIL import Image

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

TEST = r"E:\A-触觉机器学习\datasets\ycb_rgbd\test"
CASES = [
    ("000055", 22, 5, "apple(应红)"),
    ("000048", None, None, None),  # 后面动态找 banana/wood
]

def mask_region_rgb(scene, frame, idx):
    sdir = os.path.join(TEST, scene)
    im = np.array(Image.open(os.path.join(sdir, "rgb", f"{frame:06d}.png")).convert("RGB"))
    m = np.array(Image.open(os.path.join(sdir, "mask_visib", f"{frame:06d}_{idx:06d}.png"))) > 0
    if not m.any():
        return None
    px = im[m]
    return px.mean(axis=0), m.sum()

# 场景55 帧22 实例5 = apple
r = mask_region_rgb("000055", 22, 5)
print(f"apple 掩码区域平均RGB: {r[0].astype(int)}  像素数{r[1]}  (期望: 红色 R>>G,B)")

# 动态找一个 banana 和一个 wood_block 帧
for scene in sorted(os.listdir(TEST)):
    sdir = os.path.join(TEST, scene)
    gt_p = os.path.join(sdir, "scene_gt.json")
    if not os.path.exists(gt_p):
        continue
    with open(gt_p, encoding="utf-8") as f:
        gt = json.load(f)
    for frame, anns in gt.items():
        ids = [a["obj_id"] for a in anns]
        if 11 in ids and frame == sorted(gt.keys(), key=int)[0]:  # banana
            r = mask_region_rgb(scene, int(frame), ids.index(11))
            if r:
                print(f"banana({scene} 帧{frame}) 掩码区域平均RGB: {r[0].astype(int)}  (期望: 黄色 R,G>>B)")
        if 20 in ids and frame == sorted(gt.keys(), key=int)[0]:  # wood block
            r = mask_region_rgb(scene, int(frame), ids.index(20))
            if r:
                print(f"wood_block({scene} 帧{frame}) 掩码区域平均RGB: {r[0].astype(int)}  (期望: 浅木色 R>G>B)")
    else:
        continue

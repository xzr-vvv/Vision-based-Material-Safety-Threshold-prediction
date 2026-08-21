# -*- coding: utf-8 -*-
"""用 mask_visib 实例掩码验证: apple(帧22, 实例索引5) 的掩码包围盒 vs bbox_visib"""
import io
import os
import sys

import numpy as np
from PIL import Image

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

sdir = r"E:\A-触觉机器学习\datasets\ycb_rgbd\test\000055"
frame = 22
idx = 5  # apple 在 scene_gt 中的索引

mask = np.array(Image.open(os.path.join(sdir, "mask_visib", f"{frame:06d}_{idx:06d}.png")))
ys, xs = np.where(mask > 0)
mb = [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]
print(f"apple 掩码包围盒 [x,y,w,h]: {mb}, 面积 {int((mask > 0).sum())}px")
print(f"scene_gt_info bbox_visib:  [249, 190, 215, 159]")

im = Image.open(os.path.join(sdir, "rgb", f"{frame:06d}.png")).convert("RGB")
W, H = im.size
x, y, w, h = mb
mx, my = int(w * 0.15), int(h * 0.15)
im.crop((max(0, x - mx), max(0, y - my), min(W, x + w + mx), min(H, y + h + my))).save(
    r"E:\A-触觉机器学习\RGB_dataset\_debug_mask_crop.png")
print("mask 包围盒裁剪已保存: _debug_mask_crop.png")
print(f"图像尺寸: {W}x{H}")

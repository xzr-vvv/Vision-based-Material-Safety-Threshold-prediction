# -*- coding: utf-8 -*-
"""诊断 YCB 裁剪错位: 对 YCB16(苹果) 第一候选帧, 用 mask_visib 验证 bbox 配对"""
import io
import json
import os
import sys

import numpy as np
from PIL import Image

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

TEST = r"E:\A-触觉机器学习\datasets\ycb_rgbd\test"

# 找 YCB16 第一候选: 与 collect_local 相同逻辑
for scene in sorted(os.listdir(TEST)):
    sdir = os.path.join(TEST, scene)
    gi = os.path.join(sdir, "scene_gt_info.json")
    if not os.path.exists(gi):
        continue
    with open(gi, encoding="utf-8") as f:
        info = json.load(f)
    with open(os.path.join(sdir, "scene_gt.json"), encoding="utf-8") as f:
        gt = json.load(f)
    for frame in sorted(info.keys(), key=int):
        gt_anns = gt.get(frame, [])
        for i, a in enumerate(info[frame]):
            if i >= len(gt_anns):
                break
            if gt_anns[i]["obj_id"] == 16:
                bb = a.get("bbox_visib")
                if bb and min(bb[2], bb[3]) >= 60:
                    print(f"场景 {scene} 帧 {frame}: apple 在 scene_gt 索引 {i}")
                    print(f"  bbox_visib (按[x,y,w,h]解释): {bb}")
                    print(f"  该帧全部 obj_id: {[g['obj_id'] for g in gt_anns]}")

                    # rgb 文件名核对
                    rgb = os.path.join(sdir, "rgb", f"{int(frame):06d}.png")
                    print(f"  rgb 存在: {os.path.exists(rgb)}  ({rgb})")

                    # mask_visib: 实例i -> 像素值 i+1
                    mask = np.array(Image.open(os.path.join(sdir, "mask_visib", f"{int(frame):06d}.png")))
                    print(f"  mask 唯一值: {sorted(np.unique(mask).tolist())}")
                    m = mask == (i + 1)
                    if m.any():
                        ys, xs = np.where(m)
                        mb = [int(xs.min()), int(ys.min()),
                              int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]
                        print(f"  mask[{i+1}] 的包围盒: {mb}  面积: {m.sum()}px")
                        # 用 mask 包围盒裁一张对照图
                        im = Image.open(rgb).convert("RGB")
                        x, y, w, h = mb
                        mx, my = int(w * 0.15), int(h * 0.15)
                        W, H = im.size
                        im.crop((max(0, x - mx), max(0, y - my),
                                 min(W, x + w + mx), min(H, y + h + my))).save(
                            r"E:\A-触觉机器学习\RGB_dataset\_debug_mask_crop.png")
                        print("  已存 mask 包围盒裁剪: _debug_mask_crop.png")
                    sys.exit(0)

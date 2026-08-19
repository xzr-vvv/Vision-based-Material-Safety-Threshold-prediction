# -*- coding: utf-8 -*-
"""Cornell PCD 点云 -> 配对 16bit 深度图 (毫米)
修正原脚本两处 bug:
  1. float(parts) -> 应逐列 float(parts[0..2])
  2. pts.reshape((h,w)) 对 (N,3) 必炸 -> 应取 pts[:,2] 后 reshape
Cornell pcd*.txt 每行一个像素的 xyz (米), 640x480 行主序
"""
import os
import sys

import numpy as np
from PIL import Image


def pcd_txt_to_depth(pcd_path, depth_path, width=640, height=480):
    pts = []
    with open(pcd_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    pts.append((float(parts[0]), float(parts[1]), float(parts[2])))
                except ValueError:
                    pts.append((0.0, 0.0, 0.0))
    n = len(pts)
    if n < width * height * 0.5:
        return False, f"点数不足: {n}"
    pts = np.array(pts, dtype=np.float64)
    # 有效行补齐到 w*h (部分文件末尾有附加行)
    if n > width * height:
        pts = pts[: width * height]
    elif n < width * height:
        pad = np.zeros((width * height - n, 3))
        pts = np.vstack([pts, pad])
    z = pts[:, 2].reshape(height, width)
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    depth_mm = np.clip(z * 1000.0, 0, 65535).astype(np.uint16)
    Image.fromarray(depth_mm).save(depth_path)
    valid = (depth_mm > 0).mean()
    return True, f"有效像素率 {valid*100:.1f}% 深度范围 {depth_mm[depth_mm>0].min()}~{depth_mm.max()}mm"


def main(root):
    count = ok = 0
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if fn.endswith(".txt") and fn.startswith("pcd"):
                stem = fn[:-4]
                rgb = None
                for cand in (stem + ".png", stem + "r.png"):
                    if os.path.exists(os.path.join(dirpath, cand)):
                        rgb = cand
                        break
                if not rgb:
                    continue
                count += 1
                src = os.path.join(dirpath, fn)
                dst = os.path.join(dirpath, stem + "_depth.png")
                if os.path.exists(dst):
                    ok += 1
                    continue
                good, info = pcd_txt_to_depth(src, dst)
                ok += good
                print(("OK " if good else "跳过") + f" {fn} -> {stem}_depth.png ({info})")
    print(f"\n共 {count} 个 RGB-PCD 对, 转换 {ok} 个")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else r"E:\A-触觉机器学习\datasets\cornell_grasping")

# -*- coding: utf-8 -*-
"""实测 DAv2 CPU 前向速度, 决定预训练规模"""
import sys
import time

sys.path.insert(0, r"E:\A-触觉机器学习\dinov3_dual")
sys.path.insert(0, r"E:\Lib\site-packages")

import numpy as np
import torch
from PIL import Image

from backbones import (extract_depth_feature, get_device,
                       load_depth_encoder, preprocess_depth)

rng = np.random.RandomState(0)
fake = rng.uniform(0.5, 2.0, (480, 640)).astype(np.float32) * rng.choice([0, 1], (480, 640), p=[0.2, 0.8])

dev = get_device()
print("设备:", dev)
enc, dim, name = load_depth_encoder(device=dev)
print("骨干:", name, dim)

img, stats = preprocess_depth(fake)
t0 = time.time()
for i in range(5):
    _ = extract_depth_feature(enc, img, dev)
    print(f"  第{i+1}张: 累计{time.time()-t0:.1f}s", flush=True)
per = (time.time() - t0) / 5
print(f"平均 {per:.2f}s/张 -> 900张×3视图 = {900*3*per/60:.0f} 分钟")

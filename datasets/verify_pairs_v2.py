# -*- coding: utf-8 -*-
"""验证 ycbv pairs 能被 v2 双流代码直接消费"""
import os
import sys

sys.path.insert(0, r"E:\A-触觉机器学习\dinov3_dual")
sys.path.insert(0, r"E:\Lib\site-packages")

import numpy as np

from backbones import find_depth_for, get_device, load_depth_encoder
from backbones import extract_depth_feature, load_depth_array, preprocess_depth

PAIRS = r"E:\A-触觉机器学习\datasets\ycb_rgbd\pairs"
files = sorted(f for f in os.listdir(PAIRS) if not f.endswith("_depth.png"))[:3]

for fn in files:
    rgb = os.path.join(PAIRS, fn)
    dep = find_depth_for(rgb)
    assert dep is not None, f"配对失败: {fn}"
    depth_img, stats = preprocess_depth(load_depth_array(dep))
    print(f"{fn} -> 配对OK | 深度图 {depth_img.size} {depth_img.mode} | 统计 [均值,方差,梯度,空洞率]={np.round(stats,4)}")

enc, dim, name = load_depth_encoder(device=get_device())
depth_img, stats = preprocess_depth(load_depth_array(find_depth_for(os.path.join(PAIRS, files[0]))))
feat = extract_depth_feature(enc, depth_img, get_device())
print(f"DAv2 深度特征: {tuple(feat.shape)} 骨干={name}")
print("结论: ycbv pairs 可直接进入 v2 深度流")

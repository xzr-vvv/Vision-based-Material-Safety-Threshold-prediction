# -*- coding: utf-8 -*-
"""v2 双流管线冒烟测试 — 只验证代码路径, 不产生训练数据
1. 加载 DINOv3 + DAv2 双骨干(应命中本地缓存, 不再下载)
2. 用一张真实 ExpForce RGB 图提取 RGB 特征
3. 用一张程序生成的假深度数组走通深度处理+特征提取(仅冒烟, 不入库)
4. 用 4 个假样本走通 FusionNet 前向+反向一步(证明 train.py 可训)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
E_LIB = r"E:\Lib\site-packages"
if os.path.isdir(E_LIB) and E_LIB not in sys.path:
    sys.path.insert(0, E_LIB)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import config as C
from backbones import (get_device, load_backbone, extract_rgb_feature,
                       load_depth_encoder, extract_depth_feature,
                       load_depth_array, preprocess_depth, find_depth_for)
from train import FusionNet, force_nll

ok = []

device = get_device()
print("设备:", device)

print("\n[1/4] 加载 RGB 骨干 DINOv3...")
model, dim, name = load_backbone(device=device)
print("  ->", name, "维度", dim)
ok.append(("RGB骨干加载", True, name))

print("\n[2/4] 加载深度编码器 DAv2...")
enc, dep_dim, dep_name = load_depth_encoder(device=device)
print("  ->", dep_name, "维度", dep_dim)
ok.append(("深度编码器加载", True, dep_name))

print("\n[3/4] 真实图片特征提取...")
rgb_dir = r"E:\A-机器学习\ExpForce_images"
img_file = sorted(os.listdir(rgb_dir))[0]
img_path = os.path.join(rgb_dir, img_file)
feat = extract_rgb_feature(model, Image.open(img_path), device)
assert feat.shape == (dim,), feat.shape
print(f"  -> {img_file} RGB特征 {tuple(feat.shape)} ✓")

depth_arr = np.clip(np.random.default_rng(0).normal(0.5, 0.1, (240, 320)), 0.02, 1.2).astype(np.float32)
depth_img, stats = preprocess_depth(depth_arr)
df = extract_depth_feature(enc, depth_img, device)
assert df.shape == (dep_dim,), df.shape
print(f"  -> 合成深度数组(仅冒烟) 深度特征 {tuple(df.shape)} 统计向量 {tuple(stats.shape)} ✓")
assert find_depth_for(img_path) is None, "ExpForce图不应有配对深度"
ok.append(("特征提取路径", True, f"RGB{dim}维+Dep{dep_dim}维"))

print("\n[4/4] FusionNet 可训性(前向+反向一步)...")
net = FusionNet(dim, dep_dim).to(device)
rgb_b = torch.randn(4, dim)
dep_b = torch.randn(4, dep_dim)
stt_b = torch.randn(4, 4)
cls_b = torch.tensor([0, 1, 2, 0])
y_b = torch.tensor([2.0, 1.0, 0.5, 3.0])
logits, force, mass = net(rgb_b, dep_b, stt_b)
loss = F.cross_entropy(logits, cls_b) + force_nll(force, y_b) + 0.3 * F.mse_loss(mass.squeeze(-1), torch.zeros(4))
loss.backward()
grads = sum(1 for p in net.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
total_p = sum(p.numel() for p in net.parameters())
opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
opt.step()
print(f"  -> 前向输出 logits{tuple(logits.shape)} force{tuple(force.shape)} | loss={loss.item():.3f}")
print(f"  -> 反向传播: {grads} 个参数张量获得梯度, 共 {total_p/1e6:.2f}M 参数 ✓")
ok.append(("融合网可训性", True, f"{total_p/1e6:.2f}M参数可反传"))

print("\n========== 冒烟测试结论 ==========")
for name_, passed, info in ok:
    print(f"  [{'通过' if passed else '失败'}] {name_}: {info}")
print("\n注: 深度输入为程序生成的随机数组, 仅用于验证代码路径, 不进入任何训练数据。")
print("真实训练仍等 RGB-D 实拍数据(check_dataset.py 当前 0 张)。")

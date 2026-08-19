# -*- coding: utf-8 -*-
"""深度流预训练 (SafetyVTLA 合规: 仅用 ycbv 真实深度图, 无力值标签)

方案: 冻结 DAv2 ViT-L 主干, 对 900 张真实深度图每张生成 原图+3 增强视图,
      提取特征缓存后在特征上做 SimCLR 对比学习, 训练深度适配头 1024->256。
      同一深度图的不同增强互为正样本(视不变), batch 内其他图为负样本。
产物: depth_pretrain/adapter.pt + feats_cache.pt + pretrain_report.txt
接入: v2 正式训练时, depth 特征先过 adapter 再进 FusionNet
"""
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"E:\Lib\site-packages")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from PIL import Image

from backbones import (extract_depth_feature, get_device,
                       load_depth_encoder, preprocess_depth)

TEST_ROOT = r"E:\A-触觉机器学习\datasets\ycb_rgbd\test"
OUT_DIR = os.path.join(os.path.dirname(TEST_ROOT), "depth_pretrain")
N_VIEWS = 4          # 原图 + 3 增强
BATCH = 64
EPOCHS = 120
TEMP = 0.2
SEED = 0

rng = np.random.RandomState(SEED)
torch.manual_seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)


def collect_depth_files():
    files = []
    for sc in sorted(os.listdir(TEST_ROOT)):
        d = os.path.join(TEST_ROOT, sc, "depth")
        if not os.path.isdir(d):
            continue
        cam = json.load(open(os.path.join(TEST_ROOT, sc, "scene_camera.json"), encoding="utf-8"))
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".png"):
                key = str(int(fn[:-4]))
                files.append((os.path.join(d, fn), cam[key]["depth_scale"], sc, fn[:-4]))
    return files


def load_depth_m(p, scale):
    """BOP 0.1mm 制 png -> 米制数组"""
    arr = np.array(Image.open(p), dtype=np.float32)
    return arr * scale / 1000.0


def augment(arr):
    """深度图增强: 视角/距离/空洞/噪声 四类, 返回增强后米制数组"""
    a = arr.copy()
    if rng.rand() < 0.5:                       # 水平翻转
        a = a[:, ::-1]
    if rng.rand() < 0.7:                       # 随机裁剪 70~100% 再 resize
        h, w = a.shape
        r = rng.uniform(0.7, 1.0)
        ch, cw = int(h * r), int(w * r)
        y0, x0 = rng.randint(0, h - ch + 1), rng.randint(0, w - cw + 1)
        a = np.array(Image.fromarray(a[y0:y0 + ch, x0:x0 + cw]).resize((w, h), Image.BILINEAR))
    if rng.rand() < 0.5:                       # 距离缩放(模拟不同相机距离)
        a = a * rng.uniform(0.9, 1.1)
    if rng.rand() < 0.5:                       # 随机矩形空洞(模拟传感器缺失)
        a = a.copy()
        h, w = a.shape
        for _ in range(rng.randint(1, 4)):
            hh, ww = rng.randint(h // 12, h // 4), rng.randint(w // 12, w // 4)
            y0, x0 = rng.randint(0, h - hh), rng.randint(0, w - ww)
            a[y0:y0 + hh, x0:x0 + ww] = 0.0
    if rng.rand() < 0.5:                       # 高斯噪声
        a = a + rng.normal(0, 0.01, a.shape).astype(np.float32) * (a > 0)
    return np.ascontiguousarray(a)


class Adapter(nn.Module):
    """深度适配头: 1024 -> 256 (对比学习空间)"""

    def __init__(self, dim=1024, out=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, out),
        )

    def forward(self, x):
        return Fn.normalize(self.net(x), dim=1)


def build_cache(device):
    """离线提取 900×4 视图特征, 存盘"""
    cache_path = os.path.join(OUT_DIR, "feats_cache.pt")
    if os.path.exists(cache_path):
        print("特征缓存已存在, 跳过提取")
        return torch.load(cache_path, map_location="cpu")

    enc, dim, name = load_depth_encoder(device=device)
    files = collect_depth_files()
    print(f"深度图 {len(files)} 张 × {N_VIEWS} 视图, 骨干 {name}")

    feats = np.zeros((len(files), N_VIEWS, dim), dtype=np.float32)
    stats_all = np.zeros((len(files), N_VIEWS, 4), dtype=np.float32)
    t0 = time.time()
    for i, (p, scale, sc, fr) in enumerate(files):
        arr = load_depth_m(p, scale)
        views = [arr] + [augment(arr) for _ in range(N_VIEWS - 1)]
        for v, view in enumerate(views):
            try:
                img, st = preprocess_depth(view)
                feats[i, v] = extract_depth_feature(enc, img, device).numpy()
                stats_all[i, v] = st.numpy()
            except ValueError:
                feats[i, v] = feats[i, 0] if v > 0 and feats[i, 0].any() else 0
                stats_all[i, v] = stats_all[i, 0] if v > 0 else 0
        if (i + 1) % 50 == 0 or i + 1 == len(files):
            rate = (time.time() - t0) / (i + 1) * N_VIEWS
            eta = (len(files) - i - 1) * rate / N_VIEWS
            print(f"  {i+1}/{len(files)}  剩余 {eta/60:.1f} 分钟", flush=True)

    meta = {"scenes": [f[2] for f in files], "frames": [f[3] for f in files]}
    out = {"feats": torch.from_numpy(feats), "stats": torch.from_numpy(stats_all), "meta": meta}
    torch.save(out, cache_path)
    print(f"缓存完成: {cache_path}  用时 {(time.time()-t0)/60:.1f} 分钟")
    return out


def info_nce(z, temp):
    """SimCLR loss: z=(2B, D), 前半后半互为正样本"""
    b = z.size(0) // 2
    sim = z @ z.T / temp
    sim.fill_diagonal_(-1e9)
    target = torch.arange(b, device=z.device)
    target = torch.cat([target + b, target])
    return Fn.cross_entropy(sim, target)


def train(cache):
    feats = cache["feats"]           # (N, V, D)
    n = feats.size(0)
    adapter = Adapter(feats.size(2))
    opt = torch.optim.AdamW(adapter.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

    best_acc, best_state, hist = 0.0, None, []
    for ep in range(1, EPOCHS + 1):
        adapter.train()
        perm = torch.randperm(n)
        tot_loss = tot_acc = cnt = 0
        for s in range(0, n - 1, BATCH // 2):
            idx = perm[s:s + BATCH // 2]
            v1 = torch.randint(0, N_VIEWS, (len(idx),))
            v2 = (v1 + torch.randint(1, N_VIEWS, (len(idx),))) % N_VIEWS
            z = adapter(torch.cat([feats[idx, v1], feats[idx, v2]]))
            loss = info_nce(z, TEMP)
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                b = z.size(0) // 2
                sim = z[:b] @ z[b:].T
                acc = (sim.argmax(1) == torch.arange(b)).float().mean().item()
            tot_loss += loss.item() * b
            tot_acc += acc * b
            cnt += b
        sched.step()
        hist.append((ep, tot_loss / cnt, tot_acc / cnt))
        if ep % 10 == 0:
            print(f"  epoch {ep:3d} | loss {tot_loss/cnt:.3f} | 视图对齐准确率 {tot_acc/cnt*100:.1f}%  (随机={100/max(cnt,1)*0+100/BATCH:.1f}%)", flush=True)
        if tot_acc / cnt > best_acc:
            best_acc = tot_acc / cnt
            best_state = {k: v.clone() for k, v in adapter.state_dict().items()}

    torch.save({"state_dict": best_state, "in_dim": feats.size(2),
                "out_dim": 256, "epochs": EPOCHS, "best_align_acc": best_acc,
                "temp": TEMP, "n_images": n, "views": N_VIEWS},
               os.path.join(OUT_DIR, "adapter.pt"))
    return adapter, best_state, hist


def evaluate(cache, state):
    """最近邻检索 sanity check: 用原图特征找最近邻, 看是否命中间场景同物体帧"""
    feats = cache["feats"]
    meta = cache["meta"]
    adapter = Adapter(feats.size(2))
    adapter.load_state_dict(state)
    adapter.eval()
    with torch.no_grad():
        z = adapter(feats[:, 0])            # 每图取原图视图
    z = Fn.normalize(z, dim=1)
    sim = z @ z.T
    sim.fill_diagonal_(-1e9)
    nn_idx = sim.argmax(1)
    scenes = meta["scenes"]
    same_scene = sum(1 for i, j in enumerate(nn_idx) if scenes[i] == scenes[j])
    # 增强视图互检: 图 i 的视图0 最近邻应为同图其他视图
    with torch.no_grad():
        zv = adapter(feats.reshape(-1, feats.size(2)))
    zv = Fn.normalize(zv, dim=1).reshape(feats.size(0), N_VIEWS, -1)
    view_hit = 0
    for i in range(feats.size(0)):
        for v in range(N_VIEWS):
            sims = zv[i, v] @ zv.reshape(-1, zv.size(2)).T
            sims[i * N_VIEWS + v] = -1e9
            if sims.argmax().item() // N_VIEWS == i:
                view_hit += 1
    total_v = feats.size(0) * N_VIEWS
    return {
        "n_images": feats.size(0),
        "nn_same_scene": same_scene,
        "nn_same_scene_pct": same_scene / feats.size(0) * 100,
        "view_invariance": view_hit / total_v * 100,
        "n_scenes": len(set(scenes)),
    }


def main():
    device = get_device()
    print("设备:", device)
    cache = build_cache(device)
    print("\n开始对比学习训练适配头 ...")
    t0 = time.time()
    _, best_state, hist = train(cache)
    print(f"训练完成 用时 {(time.time()-t0)/60:.1f} 分钟")

    print("\n评估: 最近邻检索 + 视图不变性")
    ev = evaluate(cache, best_state)
    print(f"  图像数 {ev['n_images']} | 场景数 {ev['n_scenes']}")
    print(f"  原图最近邻同场景率: {ev['nn_same_scene_pct']:.1f}%")
    print(f"  增强视图不变性命中率: {ev['view_invariance']:.1f}%")

    lines = [
        "深度流预训练报告 (ycbv 900 张真实深度图, 无力值标签, SafetyVTLA 合规)",
        f"骨干: DAv2 ViT-L 冻结 | 适配头 1024->256 | SimCLR temp={TEMP}",
        f"训练: {EPOCHS} epoch, 视图对齐最优准确率见 adapter.pt",
        f"原图最近邻同场景率: {ev['nn_same_scene_pct']:.1f}% (12场景随机基线约8%)",
        f"增强视图不变性: {ev['view_invariance']:.1f}%",
        "接入方式: v2 正式训练时 depth 特征先过 adapter.pt 再进 FusionNet",
    ]
    open(os.path.join(OUT_DIR, "pretrain_report.txt"), "w", encoding="utf-8").write("\n".join(lines))
    print("\n报告已写入", os.path.join(OUT_DIR, "pretrain_report.txt"))


if __name__ == "__main__":
    main()

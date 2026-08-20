# -*- coding: utf-8 -*-
"""PaliGemma 视觉塔(SigLIP-So400m)检索式 A1 先验 — 最低成本路径验证

方案①(纯视觉塔 k-NN 检索):
  索引: 129 张 Exp-Force 图 -> SigLIP 图像嵌入 -> 余弦相似度
  推理: 查询图 -> top-k 邻居 -> 类别投票(易碎门控语义先验) + F_min 加权回归
验证: 30 张跨相机图(易碎 5 张此前 P(易碎) 仅 0.04-0.10 全错)
模型: thomas/siglip-so400m-patch14-384 (ModelScope, 与 PaliGemma-3b-pt 视觉塔同架构 SoViT-400m)
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
DATA_CSV = os.path.join(_REPO_ROOT, "ExpForce数据集", "07_ExpForce_安全抓力范围.csv")
IMG_DIR = os.path.join(_REPO_ROOT, "ExpForce数据集", "ExpForce_images")
REAL_DIR = os.path.join(_REPO_ROOT, "图像力预测模型", "real_images")
CACHE_DIR = "/media/jesse/4AEE6803C369DAA3/safetyvtla_A1/features"
MODEL_DIR = "/media/jesse/4AEE6803C369DAA3/safetyvtla_A1/model_cache/siglip-so400m-patch14-384"

CLASSES = ["刚体", "柔性", "易碎"]
PREFIX2CLS = {
    "metal": "刚体", "hard_plastic": "刚体", "wood_paper": "刚体",
    "foam_soft": "柔性", "leather_textile": "柔性",
    "fragile_glass": "易碎",
}

_device = None
_model = None
_proc = None


def load_model(device):
    global _model, _proc, _device
    if _model is not None:
        return _model, _proc
    from transformers import AutoModel, AutoImageProcessor
    _device = device
    _model = AutoModel.from_pretrained(MODEL_DIR).to(device).eval()
    _proc = AutoImageProcessor.from_pretrained(MODEL_DIR)  # 纯图像检索无需 tokenizer
    return _model, _proc


@torch.no_grad()
def embed_images(paths, device, batch=8):
    """SigLIP 图像嵌入(取 pooler_output, 已 L2 归一化的对比空间)"""
    model, proc = load_model(device)
    feats = []
    for i in range(0, len(paths), batch):
        imgs = [Image.open(p).convert("RGB") for p in paths[i:i + batch]]
        inputs = _proc(images=imgs, return_tensors="pt").to(device)
        out = model.get_image_features(pixel_values=inputs["pixel_values"])
        emb = out.pooler_output if hasattr(out, "pooler_output") else out  # 兼容 transformers 5.x
        feats.append(emb.cpu().float())  # (B, 1152)
    f = torch.cat(feats, dim=0)
    f = torch.nn.functional.normalize(f, dim=-1)  # 余弦空间
    return f.numpy()


def build_index(device):
    """编码 129 张 Exp-Force 图并缓存索引"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, "siglip_expforce_index.npz")
    with open(DATA_CSV, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        if len(d["files"]) == len(rows):
            print(f"索引缓存命中: {cache}")
            return rows, d["feats"]
    paths = [os.path.join(IMG_DIR, r["image_file"]) for r in rows]
    print(f"编码 {len(paths)} 张索引图 (SigLIP-So400m)...")
    feats = embed_images(paths, device)
    np.savez(cache, files=np.array([r["image_file"] for r in rows]), feats=feats)
    return rows, feats


def retrieve(query_feat, index_feats, rows, k):
    """top-k 邻居 -> 类别投票 + 力值加权回归"""
    sims = index_feats @ query_feat  # 余弦相似度 (n,)
    order = np.argsort(-sims)[:k]
    votes = {c: 0.0 for c in CLASSES}
    f_min = f_max = 0.0
    wsum = 0.0
    for rank, idx in enumerate(order):
        w = float(sims[idx])  # 相似度加权
        w = max(w, 0.0)
        votes[rows[idx]["category"]] += w
        f_min += w * float(rows[idx]["f_min_value"])
        f_max += w * float(rows[idx]["max_safe_force_N"])
        wsum += w
    pred = max(votes, key=votes.get)
    return {
        "pred": pred,
        "vote_share": {c: v / max(sum(votes.values()), 1e-6) for c, v in votes.items()},
        "f_min": f_min / max(wsum, 1e-6),
        "f_max": f_max / max(wsum, 1e-6),
        "neighbors": [(rows[i]["object_name"], rows[i]["category"], float(sims[i])) for i in order],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5, help="检索邻居数")
    ap.add_argument("--query-dir", default=REAL_DIR, help="查询图目录(默认跨相机 30 张)")
    ap.add_argument("--json-out", default=None, help="结果 JSON 输出路径")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")
    rows, index_feats = build_index(device)

    import re
    queries = []
    for f in sorted(os.listdir(args.query_dir)):
        if not f.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        m = re.match(r"([a-z_]+?)_\d", f)
        truth = PREFIX2CLS.get(m.group(1) if m else "", None)
        if truth:
            queries.append((f, truth))
    print(f"查询图: {len(queries)} 张\n")

    qpaths = [os.path.join(args.query_dir, f) for f, _ in queries]
    qfeats = embed_images(qpaths, device)

    results = []
    correct = 0
    per_cls = {c: [0, 0] for c in CLASSES}
    err_min = []
    for (fname, truth), qf in zip(queries, qfeats):
        r = retrieve(qf, index_feats, rows, args.k)
        ok = r["pred"] == truth
        correct += ok
        per_cls[truth][0] += ok
        per_cls[truth][1] += 1
        results.append({"file": fname, "truth": truth, **r, "correct": ok})
        top = ", ".join(f"{n}({c},{s:.2f})" for n, c, s in r["neighbors"][:3])
        print(f"{fname:<42s}{truth:<4s}->{r['pred']:<4s}{'✓' if ok else '✗'} "
              f"F=[{r['f_min']:.2f},{r['f_max']:.2f}]N | {top}")

    print(f"\n===== 检索式 A1 跨相机结果 (k={args.k}) =====")
    print(f"总准确率: {correct}/{len(queries)} = {correct/len(queries):.0%}")
    for c in CLASSES:
        h, n = per_cls[c]
        if n:
            print(f"  {c}: {h}/{n} = {h/n:.0%}")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=1, default=float)
        print(f"\n结果已存: {args.json_out}")


if __name__ == "__main__":
    main()

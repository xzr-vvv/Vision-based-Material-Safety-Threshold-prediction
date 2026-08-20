# -*- coding: utf-8 -*-
"""融合实验: 分层模型 E(视觉门控) × SigLIP 检索先验 — A1 最终形态验证

融合策略:
  visual   : 仅视觉门控(基线, 模型E gate_thr=0.35)
  retrieve : 仅检索投票
  max      : P(易碎) = max(P_visual, P_retrieval)   ← 召回优先
  weighted : P(易碎) = w·P_semantic + (1-w)·P_visual, w=0.5
  or_gate  : 易碎 if P_visual>0.35 OR P_retrieval>0.5

评测集:
  ① 同相机验证集(25 张, 分层划分的 val; 检索索引只含训练物体 104 张, 防自检索作弊)
  ② 跨相机 30 张(索引用全 129 张)
非易碎细分: 视觉 rigid 头为主, 检索票数对 刚体/柔性 做软融合
"""
import argparse
import csv
import os
import re
import sys

import numpy as np
import torch
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "ExpForce单流模型"))
sys.path.insert(0, _HERE)

from train_expforce_hier import (HierModel, GATE_CLASSES, FRAG_CLASSES, RIGID_CLASSES,
                                 LEAF3, MEAN, STD, IMG_SIZE, DEVICE,
                                 load_main_rows, stratified_split, val_tf, frag_sub)
from siglip_retrieval import embed_images, IMG_DIR, REAL_DIR, PREFIX2CLS

MODEL_PATH = os.path.join(_REPO_ROOT, "ExpForce单流模型", "expforce_hier_aug_all.pth")

STRATEGIES = ["visual", "retrieve", "max", "weighted", "routed"]
GATE_THR = 0.35
DOMAIN_THR = 0.70  # SigLIP top-1 相似度域分离器: 同相机全部>0.70, 跨相机全部<0.70


@torch.no_grad()
def visual_gate(model, paths, batch=16):
    """返回每个 gate/rigid/frag 头的概率"""
    out = []
    for i in range(0, len(paths), batch):
        imgs = [val_tf(Image.open(p).convert("RGB")) for p in paths[i:i + batch]]
        x = torch.stack(imgs).to(DEVICE)
        lg, lf, lr_, fo, ma = model(x)
        pg = torch.softmax(lg, dim=1).cpu().numpy()          # (B,2) 非易碎/易碎
        pr = torch.softmax(lr_, dim=1).cpu().numpy()          # (B,2) 刚体/柔性
        out.append(np.stack([pg[:, 1], pr[:, 0], pr[:, 1]], axis=1))  # P易碎, P刚体, P柔性
    return np.concatenate(out, axis=0)


def retrieval_probs(query_feats, index_feats, index_rows):
    """检索: 返回 [P易碎,P刚体,P柔性] 票数, 力值回归, top-1 相似度(域检测器)"""
    probs, forces, top1s = [], [], []
    order_cls = ["易碎", "刚体", "柔性"]  # 与 pv_row 顺序一致
    for qf in query_feats:
        sims = index_feats @ qf
        top = np.argsort(-sims)[:9]  # k=9 (实验最优)
        votes = np.zeros(3)
        fmin = fmax = wsum = 0.0
        for idx in top:
            w = max(float(sims[idx]), 0.0)
            votes[order_cls.index(index_rows[idx]["category"])] += w
            fmin += w * float(index_rows[idx]["f_min_value"])
            fmax += w * float(index_rows[idx]["max_safe_force_N"])
            wsum += w
        probs.append(votes / max(votes.sum(), 1e-6))
        forces.append((fmin / max(wsum, 1e-6), fmax / max(wsum, 1e-6)))
        top1s.append(float(sims[top[0]]))
    return np.array(probs), np.array(forces), np.array(top1s)


def fuse(pv, ps, strategy):
    """返回融合后的 P(易碎) 与是否判易碎"""
    if strategy == "visual":
        p = pv
        frag = p > GATE_THR
    elif strategy == "retrieve":
        # 检索独立基线由 predict3 特殊处理(argmax), 此处不达
        p, frag = ps, None
    elif strategy == "max":
        p = max(pv, ps)
        frag = p > GATE_THR
    elif strategy == "weighted":
        p = 0.5 * pv + 0.5 * ps
        frag = p > GATE_THR
    elif strategy == "weighted30":
        # 同 weighted 但门控阈值降到 0.30(兜住跨相机玻璃: pv≈0.04, ps≈0.55 → 融合≈0.30)
        p = 0.5 * pv + 0.5 * ps
        frag = p > 0.30
    elif strategy == "or_gate":
        p = max(pv, ps)
        frag = (pv > GATE_THR) or (ps > 0.5)
    return p, frag


def predict3(pv_row, ps_row, strategy, top1=1.0):
    """三类预测。routed: top1>0.70 域内走 weighted 融合, 域外走检索(视觉跨域不可靠)"""
    pv_frag, pv_rigid, pv_soft = pv_row
    ps_frag, ps_rigid, ps_soft = ps_row
    if strategy == "retrieve":
        return ["易碎", "刚体", "柔性"][int(np.argmax(ps_row))]
    if strategy == "routed":
        strategy = "weighted" if top1 > DOMAIN_THR else "retrieve"
        if strategy == "retrieve":
            return ["易碎", "刚体", "柔性"][int(np.argmax(ps_row))]
    _, is_frag = fuse(pv_frag, ps_frag, strategy)
    if is_frag:
        return "易碎"
    r = 0.5 * pv_rigid + 0.5 * ps_rigid
    s = 0.5 * pv_soft + 0.5 * ps_soft
    return "刚体" if r >= s else "柔性"


def eval_set(name, truths, preds, extra=""):
    correct = sum(t == p for t, p in zip(truths, preds))
    print(f"\n===== {name} =====")
    print(f"总准确率: {correct}/{len(truths)} = {correct/len(truths):.0%}{extra}")
    for c in LEAF3:
        idx = [i for i, t in enumerate(truths) if t == c]
        if idx:
            h = sum(preds[i] == c for i in idx)
            print(f"  {c}: {h}/{len(idx)} = {h/len(idx):.0%}")
    return correct / len(truths)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_PATH)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")

    # ---- 视觉模型 ----
    ckpt = torch.load(args.model, map_location=DEVICE)
    model = HierModel()
    model.load_state_dict(ckpt["model"])
    model.to(DEVICE).eval()
    print(f"视觉模型: {os.path.basename(args.model)} (同相机 acc {ckpt.get('val',{}).get('acc3',0):.0%})")

    # ---- 数据: 同相机 val (与训练同 seed 划分) ----
    main_rows = load_main_rows(os.path.join(_REPO_ROOT, "ExpForce数据集",
                                            "07_ExpForce_安全抓力范围.csv"))
    train_rows, val_rows = stratified_split(main_rows)

    results = {}

    # ========== ① 同相机验证集(索引仅含训练物体) ==========
    val_paths = [os.path.join(IMG_DIR, r["image_file"]) for r in val_rows]
    val_truths = [r["category"] for r in val_rows]
    pv_val = visual_gate(model, val_paths)
    tr_feats = embed_images([os.path.join(IMG_DIR, r["image_file"]) for r in train_rows], device)
    va_feats = embed_images(val_paths, device)
    ps_val, _, _ = retrieval_probs(va_feats, tr_feats, train_rows)

    print("\n【同相机验证集 25 张 / 检索索引=训练物体 104 张】")
    for st in STRATEGIES:
        preds = [predict3(pv_val[i], ps_val[i], st) for i in range(len(val_truths))]
        results[("same", st)] = eval_set(f"策略 {st}", val_truths, preds)

    # ========== ② 跨相机 30 张(索引=全 129) ==========
    queries = []
    for f in sorted(os.listdir(REAL_DIR)):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            m = re.match(r"([a-z_]+?)_\d", f)
            t = PREFIX2CLS.get(m.group(1) if m else "", None)
            if t:
                queries.append((f, t))
    q_paths = [os.path.join(REAL_DIR, f) for f, _ in queries]
    q_truths = [t for _, t in queries]
    pv_cc = visual_gate(model, q_paths)
    ps_cc, _, top1_cc = retrieval_probs(
        embed_images(q_paths, device),
        embed_images([os.path.join(IMG_DIR, r["image_file"]) for r in main_rows], device),
        main_rows)

    print("\n【跨相机 30 张 / 检索索引=全 129 物体】")
    for st in STRATEGIES:
        preds = [predict3(pv_cc[i], ps_cc[i], st, top1=float(top1_cc[i])) for i in range(len(q_truths))]
        results[("cross", st)] = eval_set(f"策略 {st}", q_truths, preds)

    # ---- 汇总表 ----
    print("\n════════ 融合策略汇总 ════════")
    print(f"{'策略':<12s}{'同相机':<10s}{'跨相机':<10s}")
    for st in STRATEGIES:
        print(f"{st:<12s}{results[('same',st)]:<10.0%}{results[('cross',st)]:<10.0%}")


if __name__ == "__main__":
    main()

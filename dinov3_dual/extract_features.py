import os
import time

import pandas as pd
import torch
from PIL import Image

import config as C
from backbones import (get_device, load_backbone, extract_rgb_feature,
                       load_depth_encoder, extract_depth_feature,
                       load_depth_array, preprocess_depth, find_depth_for)


def main():
    device = get_device()
    print("设备:", device)
    model, dim, name = load_backbone(device=device)
    print("RGB 骨干:", name, "维度:", dim)
    enc, dep_dim, dep_name = load_depth_encoder(device=device)
    print("深度编码器:", dep_name, "维度:", dep_dim)

    rows = pd.read_csv(C.CSV_PATH)
    print(f"labels: {len(rows)} 条")

    feats = {}
    t0 = time.time()
    for i, r in enumerate(rows.itertuples(index=False), 1):
        rgb_path = os.path.join(C.DATASET_ROOT, r.image_file)
        if not os.path.exists(rgb_path):
            print(f"[跳过] 图片不存在: {r.image_file}")
            continue
        depth_path = find_depth_for(rgb_path)
        if depth_path is None:
            print(f"[跳过] 缺配对深度图: {r.image_file}")
            continue
        try:
            depth_img, stats = preprocess_depth(load_depth_array(depth_path))
        except Exception as e:
            print(f"[跳过] 深度图异常: {r.image_file} ({e})")
            continue

        mass = float(r.mass_g) if pd.notna(getattr(r, "mass_g", None)) else -1.0
        feats[r.image_file] = {
            "rgb": extract_rgb_feature(model, Image.open(rgb_path), device),
            "dep": extract_depth_feature(enc, depth_img, device),
            "stats": stats,
            "cls": C.CLASSES.index(r.category),
            "min_force": float(r.min_grasp_force_N),
            "mass_g": mass,
        }
        if i % 20 == 0 or i == len(rows):
            rate = (time.time() - t0) / i
            print(f"{i}/{len(rows)}  平均 {rate:.2f}s/张  预计剩余 {(len(rows)-i)*rate/60:.1f} 分钟")

    os.makedirs(C.FEATURE_DIR, exist_ok=True)
    torch.save({"rgb_backbone": name, "rgb_dim": dim, "dep_dim": dep_dim,
                "feats": feats}, C.FEAT_FILE)
    print(f"完成: {len(feats)} 个样本特征已缓存 -> {C.FEAT_FILE}")


if __name__ == "__main__":
    main()

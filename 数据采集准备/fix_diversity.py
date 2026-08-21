# -*- coding: utf-8 -*-
"""剔除近重复图片: 贪心保留(与已保留图距离>=阈值), 其余移入隔离区.

- YCB18/YCB21: 同一YCB-Video场景连续帧, 整体低多样性, 全部隔离换网图
- 其余标记文件夹: 阈值0.05贪心剔除
- 同步清理 provenance.csv 中被隔离图片的行
"""
import csv
import os
import shutil
import sys

sys.path.insert(0, r"E:\Lib\site-packages")
import numpy as np
from PIL import Image

RGB = r"E:\A-触觉机器学习\RGB_dataset"
PROV_FILE = os.path.join(RGB, "provenance.csv")
QUAR = os.path.join(RGB, "_quarantine")
THRESH = 0.05

WIPE = {("刚体", "YCB18"), ("刚体", "YCB21")}

FLAGGED = {
    "轻脆": ["FRA017", "FRA047", "FRA050"],
    "重脆": ["E108", "E116", "FRA004", "FRA005", "FRA036", "FRA059"],
    "刚体": ["E016", "E058", "E093", "E101", "E107", "YCB07", "YCB08",
             "YCB10", "YCB18", "YCB19", "YCB20", "YCB21"],
    "柔性": ["YCB13", "YCB14", "YCB16", "YCB17"],
}


def thumb_vec(path):
    img = Image.open(path).convert("L").resize((32, 32))
    return np.asarray(img, dtype=np.float32) / 255.0


def greedy_keep(obj_dir, files):
    vecs = {f: thumb_vec(os.path.join(obj_dir, f)).ravel() for f in files}
    kept = []
    for f in files:
        v = vecs[f]
        if all(np.abs(v - vecs[k]).mean() >= THRESH for k in kept):
            kept.append(f)
    return kept


def main():
    removed = []  # (leaf, oid, filename)
    for leaf, oids in FLAGGED.items():
        for oid in oids:
            obj_dir = os.path.join(RGB, leaf, oid)
            if not os.path.isdir(obj_dir):
                continue
            files = sorted(f for f in os.listdir(obj_dir)
                           if f.lower().endswith((".png", ".jpg", ".jpeg")))
            if (leaf, oid) in WIPE:
                keep = []
            else:
                keep = greedy_keep(obj_dir, files)
            gone = [f for f in files if f not in keep]
            if gone:
                qdir = os.path.join(QUAR, leaf, oid)
                os.makedirs(qdir, exist_ok=True)
                for f in gone:
                    shutil.move(os.path.join(obj_dir, f), os.path.join(qdir, f))
                    removed.append((leaf, oid, f))
                print(f"{leaf}\\{oid}: 保留{len(keep)} 隔离{len(gone)}")
    if not removed:
        print("无需剔除")
        return
    gone_set = {os.path.normpath(os.path.join(leaf, oid, f)) for leaf, oid, f in removed}
    with open(PROV_FILE, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
        fields = list(rows[0].keys())
    keep_rows = []
    dropped = 0
    for r in rows:
        rel = os.path.normpath(r["image_file"].replace("\\", os.sep))
        if rel in gone_set:
            dropped += 1
        else:
            keep_rows.append(r)
    with open(PROV_FILE, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(keep_rows)
    print(f"共隔离 {len(removed)} 张图, 清理 provenance {dropped} 行")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
r"""检查自采 RGB 数据集并生成/更新 labels.csv（L1/L2 分层版, 仅 RGB）

结构约定: RGB_dataset\{叶类}\{物体ID}\s0.png ~ s14.png
力值红线: f_min/f_max 列只允许实测值, 生成时留空
0821: 改用标准库 csv 实现, 去掉 pandas 依赖
"""
import argparse
import csv
import os

import config as C

EXTS = (".png", ".jpg", ".jpeg")
MIN_IMGS_PER_OBJ = 15  # 3 视角 x 5 张

LABEL_COLS = ["image_file", "leaf_class", "category", "l1_gate", "object_id",
              "object_family_id", "mass_g", "f_min_measured_N", "f_max_measured_N", "note"]


def scan(root):
    rows = []
    for leaf in C.LEAVES:
        leaf_dir = os.path.join(root, leaf)
        if not os.path.isdir(leaf_dir):
            continue
        for oid in sorted(os.listdir(leaf_dir)):
            obj_dir = os.path.join(leaf_dir, oid)
            if not os.path.isdir(obj_dir):
                continue
            for n in sorted(os.listdir(obj_dir)):
                if not n.lower().endswith(EXTS):
                    continue
                # 历史遗留的 *_depth.* 一律不计入 RGB 统计
                if n.lower().endswith("_depth" + os.path.splitext(n)[1]):
                    continue
                rows.append({"image_file": os.path.relpath(os.path.join(obj_dir, n), root),
                             "leaf_class": leaf,
                             "object_id": oid})
    return rows


def load_objects():
    """物体清单: object_id -> (leaf, family, mass_g)"""
    if not os.path.exists(C.OBJECTS_CSV):
        return {}
    out = {}
    with open(C.OBJECTS_CSV, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            oid = (r.get("object_id") or "").strip()
            if not oid:
                continue
            leaf = (r.get("leaf_class") or "").strip()
            family = (r.get("object_family_id") or "").strip() or oid
            mass = ""
            mkg = (r.get("mass_kg") or "").strip()
            if mkg:
                try:
                    mass = round(float(mkg) * 1000)
                except ValueError:
                    mass = ""
            out[oid] = (leaf, family, mass)
    return out


def main():
    ap = argparse.ArgumentParser(description="检查 RGB 数据集完整性并生成/更新 labels.csv")
    ap.add_argument("--make-labels", action="store_true", help="生成 labels.csv（力值列留空待实测）")
    args = ap.parse_args()

    if not os.path.isdir(C.DATASET_ROOT):
        os.makedirs(C.DATASET_ROOT, exist_ok=True)
    for leaf in C.LEAVES:  # 缺的叶类文件夹顺手建好
        os.makedirs(os.path.join(C.DATASET_ROOT, leaf), exist_ok=True)

    rows = scan(C.DATASET_ROOT)
    objects = load_objects()

    print(f"数据集根目录: {C.DATASET_ROOT}")
    if not rows:
        print("未扫描到图片。请按结构放图: RGB_dataset\\{叶类}\\{物体ID}\\s0.png ~ s14.png")
        print("物体ID与叶类对照: " + C.OBJECTS_CSV)
        return

    imgs_per_obj = {}
    leaf_objs = {leaf: set() for leaf in C.LEAVES}
    leaf_imgs = {leaf: 0 for leaf in C.LEAVES}
    pairs = set()
    for r in rows:
        imgs_per_obj[r["object_id"]] = imgs_per_obj.get(r["object_id"], 0) + 1
        leaf_objs[r["leaf_class"]].add(r["object_id"])
        leaf_imgs[r["leaf_class"]] += 1
        pairs.add((r["leaf_class"], r["object_id"]))

    print(f"\n扫描到 {len(rows)} 张 RGB 图, {len(imgs_per_obj)} 个物体")

    # --- 每叶统计 + 每物体张数 ---
    print("\n叶类统计（目标: 每叶 ≥%d 物体, 每物体 ≥%d 张 RGB）:" % (C.LEAF_TARGET, MIN_IMGS_PER_OBJ))
    for leaf in C.LEAVES:
        n_obj = len(leaf_objs[leaf])
        mark = "OK" if n_obj >= C.LEAF_TARGET else "不足!"
        print(f"  {leaf}: {n_obj} 物体 / {leaf_imgs[leaf]} 张  [{mark}]")

    short = {oid: n for oid, n in imgs_per_obj.items() if n < MIN_IMGS_PER_OBJ}
    if short:
        print(f"\n[张数不足] {len(short)} 个物体少于 {MIN_IMGS_PER_OBJ} 张:")
        for oid in sorted(short):
            print(f"  {oid}: {short[oid]} 张")

    # --- 清单交叉核对: 文件夹叶类 vs 清单叶类 ---
    if objects:
        mismatch = []
        for leaf, oid in sorted(pairs):
            if oid in objects and objects[oid][0] != leaf:
                mismatch.append(f"  {oid}: 文件夹在[{leaf}], 清单标[{objects[oid][0]}]")
        if mismatch:
            print("\n[叶类不一致] 文件夹位置与物体清单 leaf_class 冲突:")
            print("\n".join(mismatch))

    if args.make_labels:
        old = {}
        if os.path.exists(C.CSV_PATH):
            with open(C.CSV_PATH, encoding="utf-8-sig", newline="") as f:
                for r in csv.DictReader(f):
                    old[r["image_file"]] = r

        out_rows = []
        for r in rows:
            tpl_leaf, family, mass = objects.get(r["object_id"], (r["leaf_class"], r["object_id"], ""))
            keep = old.get(r["image_file"], {})
            fmin = (keep.get("f_min_measured_N") or "").strip()
            fmax = (keep.get("f_max_measured_N") or "").strip()
            note = "已填实测" if fmin != "" else "力值待实测"
            out_rows.append({
                "image_file": r["image_file"],
                "leaf_class": r["leaf_class"],
                "category": C.LEAF2CAT[r["leaf_class"]],
                "l1_gate": C.LEAF2L1[r["leaf_class"]],
                "object_id": r["object_id"],
                "object_family_id": family,
                "mass_g": mass,
                "f_min_measured_N": fmin,
                "f_max_measured_N": fmax,
                "note": note,
            })
        with open(C.CSV_PATH, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=LABEL_COLS)
            w.writeheader()
            w.writerows(out_rows)
        n_filled = sum(1 for r in out_rows if r["f_min_measured_N"] != "")
        print(f"\nlabels.csv 已生成: {C.CSV_PATH}")
        print(f"已填实测 f_min: {n_filled}/{len(out_rows)} 行; 其余留空, 实测后填入再训练")


if __name__ == "__main__":
    main()

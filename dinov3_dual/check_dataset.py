import argparse
import os

import pandas as pd

import config as C

EXTS = (".png", ".jpg", ".jpeg")
DEPTH_EXTS = (".png", ".jpg", ".jpeg", ".npy", ".tiff", ".bmp")


def find_depth(rgb_path):
    base, name = os.path.split(rgb_path)
    stem = os.path.splitext(name)[0]
    root = os.path.dirname(base)
    for ext in DEPTH_EXTS:
        for p in (os.path.join(base, stem + "_depth" + ext),
                  os.path.join(base, "depth", name),
                  os.path.join(root, "depth", name),
                  os.path.join(base, "depth", stem + ext)):
            if os.path.exists(p):
                return p
    return None


def scan(root):
    rows = []
    for cls in C.CLASSES:
        cls_dir = os.path.join(root, cls)
        if not os.path.isdir(cls_dir):
            continue
        for n in sorted(os.listdir(cls_dir)):
            if not n.lower().endswith(EXTS) or n.lower().endswith("_depth" + os.path.splitext(n)[1]):
                continue
            rgb = os.path.join(cls_dir, n)
            rows.append({"image_file": os.path.relpath(rgb, root),
                         "category": cls,
                         "depth_found": find_depth(rgb) is not None})
    return pd.DataFrame(rows)


def expforce_defaults():
    if not os.path.exists(C.EXPFORCE_CSV):
        return {}
    df = pd.read_csv(C.EXPFORCE_CSV)
    out = {}
    for cls, g in df.groupby("category"):
        out[cls] = (round(float(g.min_grasp_force_N.median()), 2),
                    round(float(g.max_safe_force_N.median()), 2),
                    round(float(g.mass_g.median())))
    return out


def main():
    ap = argparse.ArgumentParser(description="检查 RGB-D 数据集配对完整性并生成/更新 labels.csv")
    ap.add_argument("--make-labels", action="store_true", help="生成 labels.csv（力值默认取 Exp-Force 类别中位数，可手改）")
    args = ap.parse_args()

    if not os.path.isdir(C.DATASET_ROOT):
        print(f"数据集目录不存在: {C.DATASET_ROOT}")
        print(f"请先创建: {C.DATASET_ROOT}\\{{刚体,柔性,易碎}}\\  并放入 RGB 与配对深度图")
        return

    df = scan(C.DATASET_ROOT)
    if df.empty:
        print("未扫描到图片。请确认子文件夹名为: " + "、".join(C.CLASSES))
        return

    print(f"扫描到 {len(df)} 张 RGB 图")
    n_no_depth = (~df.depth_found).sum()
    for f in df.loc[~df.depth_found, "image_file"].head(10):
        print("  [缺深度] " + f)
    print(f"配对完整: {len(df) - n_no_depth}/{len(df)}"
          + ("（0.9 起）" if n_no_depth else "全部就绪"))

    print("\n类别分布:")
    print(df.category.value_counts().to_string())

    if args.make_labels:
        defaults = expforce_defaults()
        out_rows = []
        for r in df.itertuples(index=False):
            mn, mx, mass = defaults.get(r.category, (1.0, 3.0, 200))
            out_rows.append({"image_file": r.image_file, "category": r.category,
                             "mass_g": mass, "min_grasp_force_N": mn,
                             "max_safe_force_N": mx,
                             "note": "expforce默认" if defaults else ""})
        out = pd.DataFrame(out_rows)
        if os.path.exists(C.CSV_PATH):
            old = pd.read_csv(C.CSV_PATH)
            keep = old.set_index("image_file")
            for i, r in out.iterrows():
                if r.image_file in keep.index:
                    out.loc[i, ["mass_g", "min_grasp_force_N",
                                "max_safe_force_N"]] = keep.loc[r.image_file,
                                ["mass_g", "min_grasp_force_N", "max_safe_force_N"]].values
                    out.loc[i, "note"] = "已填"
        out.to_csv(C.CSV_PATH, index=False, encoding="utf-8-sig")
        print(f"\nlabels.csv 已生成: {C.CSV_PATH}")
        print("提示: 力值默认取 Exp-Force 同类中位数（标签平移），请按实测替换后再训练")


if __name__ == "__main__":
    main()

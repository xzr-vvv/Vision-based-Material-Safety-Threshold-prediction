# -*- coding: utf-8 -*-
r"""0821 数据集裁剪 (每类减到30, exp-force全保留):
- 轻脆/重脆: E系全留 + FRA 按力值证据等级排序
  (T1 ExpForce同型实测 > T2 同型转移 > T3 近似转移),
  T3 内部按视角多样性(均值L1距离)降序保留, 凑足每类30个物体
- 刚体: 只留 E 系 (YCB 14 个移出)
- 柔性: E 系 + FRA016/039/040 (YCB 7 个移出; FRA016/039 为 T1 同型实测, 保留)
- 移出物体文件夹进 _quarantine\{类}\{物体}, provenance.csv 同步清理对应行
"""
import csv
import os
import re
import shutil

ROOT = r"E:\A-触觉机器学习\RGB_dataset"
QUAR = os.path.join(ROOT, "_quarantine")
FM = os.path.join(ROOT, "force_map.csv")
PROV = os.path.join(ROOT, "provenance.csv")
DIV_REPORT = r"E:\A-触觉机器学习\数据采集准备\diversity_report_final.txt"

TARGET = {"轻脆": 30, "重脆": 30}
KEEP_FRA_SOFT = {"FRA016", "FRA039", "FRA040"}


def load_fra_tier():
    tier = {}
    with open(FM, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            oid = r["object_id"]
            if not oid.startswith("FRA"):
                continue
            src = r["label_source"] or ""
            if "ExpForce同型实测" in src:
                tier[oid] = 1
            elif src.startswith("同型转移"):
                tier[oid] = 2
            else:
                tier[oid] = 3
    return tier


def load_divmean():
    d = {}
    pat = re.compile(r"^([A-Z]+\d+)\s+(轻脆|重脆|刚体|柔性)\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)\s*$")
    with open(DIV_REPORT, encoding="utf-8") as f:
        for line in f:
            m = pat.match(line.strip())
            if m:
                d[m.group(1)] = float(m.group(4))
    return d


def main():
    tier = load_fra_tier()
    div = load_divmean()
    removed = []  # (leaf, oid)

    for leaf in ("轻脆", "重脆"):
        leaf_dir = os.path.join(ROOT, leaf)
        entries = sorted(os.listdir(leaf_dir))
        e_cnt = sum(1 for o in entries if re.match(r"^E\d", o))
        fras = [o for o in entries if o.startswith("FRA")]
        need = TARGET[leaf] - e_cnt
        fras.sort(key=lambda o: (tier.get(o, 3), -div.get(o, 0.0)))
        keep = set(fras[:need])
        drop = [o for o in fras if o not in keep]
        print(f"{leaf}: E {e_cnt} + FRA保留{len(keep)} (T1/T2全保) -> 删FRA {len(drop)}: {', '.join(drop)}")
        for oid in drop:
            removed.append((leaf, oid))

    for leaf, pfx in (("刚体", "YCB"), ("柔性", "YCB")):
        leaf_dir = os.path.join(ROOT, leaf)
        for oid in sorted(os.listdir(leaf_dir)):
            if oid.startswith(pfx):
                removed.append((leaf, oid))

    # FRA016/039/040 属柔性, 不在轻脆/重脆删除逻辑内, 此处无动作
    print(f"\n共移出 {len(removed)} 个物体")

    for leaf, oid in removed:
        src = os.path.join(ROOT, leaf, oid)
        dst = os.path.join(QUAR, leaf, oid)
        if os.path.isdir(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.isdir(dst):
                shutil.rmtree(dst)
            shutil.move(src, dst)

    prefixes = tuple(f"{leaf}\\{oid}\\" for leaf, oid in removed)
    with open(PROV, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    kept = [r for r in body if r and not r[1].startswith(prefixes)]
    print(f"provenance: {len(body)} -> {len(kept)} 行 (移除 {len(body) - len(kept)})")
    with open(PROV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(kept)

    for leaf in ("轻脆", "重脆", "刚体", "柔性"):
        leaf_dir = os.path.join(ROOT, leaf)
        objs = [o for o in os.listdir(leaf_dir) if os.path.isdir(os.path.join(leaf_dir, o))]
        imgs = sum(len([x for x in os.listdir(os.path.join(leaf_dir, o))
                        if x.lower().endswith((".png", ".jpg", ".jpeg"))]) for o in objs)
        print(f"最终 {leaf}: {len(objs)} 物体 / {imgs} 图")


if __name__ == "__main__":
    main()

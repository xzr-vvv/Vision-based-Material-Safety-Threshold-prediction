# -*- coding: utf-8 -*-
r"""fragile_flag / 本地四分类 与 ExpForce 官方分类的一致性比对
层面A: objects_已有安全阈值.csv 手填 fragile_flag/deformable_flag vs ref_category vs ExpForce官方
层面B: RGB_dataset/labels.csv E系物体本地分类(轻脆/重脆/刚体/柔性) vs ExpForce官方
输出: 控制台清单 + flag_consistency_report.csv
"""
import csv
import os
import re

ROOT = r"E:\A-触觉机器学习"
THRESH = os.path.join(ROOT, "数据采集准备", "objects_已有安全阈值.csv")
EXPF = os.path.join(ROOT, "ExpForce数据集", "07_ExpForce_安全抓力范围.csv")
LABELS = os.path.join(ROOT, "RGB_dataset", "labels.csv")
PROV = os.path.join(ROOT, "RGB_dataset", "provenance.csv")
OUT = os.path.join(ROOT, "数据采集准备", "flag_consistency_report.csv")


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def hand_class(frag, defo):
    if frag == "1":
        return "易碎"
    if defo == "1":
        return "柔性"
    return "刚体"


def parse_refs(s):
    out = []
    for tok in re.split(r"[/]", (s or "").strip()):
        tok = tok.strip()
        m = re.match(r"^(E\d+)-(E\d+)$", tok)
        if m:
            a, b = int(m.group(1)[1:]), int(m.group(2)[1:])
            out += [f"E{i:03d}" for i in range(a, b + 1)]
        elif tok:
            out.append(tok)
    return out


def main():
    # ExpForce 官方分类: 按E编号/名称/图片 stem 三索引
    eid2cat, name2cat, stem2cat = {}, {}, {}
    with open(EXPF, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            eid2cat[r["object_id"]] = r["category"]
            name2cat[norm(r["object_name"])] = r["category"]
            stem2cat[norm(os.path.splitext(r["image_file"])[0])] = r["category"]

    report = []

    print("=" * 100)
    print("层面A: objects_已有安全阈值.csv (41行手填) vs ref_category vs ExpForce官方")
    print("=" * 100)
    with open(THRESH, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    n_incons_a = 0
    for r in rows:
        hand = hand_class(r["fragile_flag"], r["deformable_flag"])
        refcat = r["ref_category"]
        refs = parse_refs(r["ref_expforce_id"])
        offcats = sorted({eid2cat.get(x, "?") for x in refs})
        # ref_category 是否忠实于官方
        ref_ok = (not offcats or refcat in offcats)
        hand_ok = (hand == refcat)
        if not hand_ok:
            n_incons_a += 1
        status = "一致" if (hand_ok and ref_ok) else ("手填vs参照不一致" if ref_ok else "ref_category与官方不符")
        print(f"{r['object_id']:8s} {r['object_name'][:16]:18s} 手填={hand} ref_category={refcat} "
              f"官方={','.join(offcats):10s} [{status}]")
        report.append({
            "层面": "A_阈值表手填", "object_id": r["object_id"],
            "object_name": r["object_name"], "本地口径": hand,
            "ref_category": refcat, "ExpForce官方": "/".join(offcats),
            "不一致": "" if (hand_ok and ref_ok) else "是",
            "备注": status,
        })
    print(f"\n层面A不一致: {n_incons_a}/{len(rows)} 行\n")

    print("=" * 100)
    print("层面B: labels.csv E系物体 (129个) 本地四分类 vs ExpForce官方")
    print("=" * 100)
    # provenance: E系 object_id -> ExpForce 原始图名 (仅 ExpForce 源的行)
    oid2stem = {}
    with open(PROV, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r["object_id"].startswith("E") and r["source"].startswith("ExpForce"):
                oid2stem[r["object_id"]] = norm(os.path.splitext(os.path.basename(r["image_file"]))[0])
    with open(LABELS, encoding="utf-8-sig") as f:
        lrows = list(csv.DictReader(f))
    seen, objs = set(), []
    for r in lrows:
        oid = r["object_id"]
        if oid.startswith("E") and oid not in seen:
            seen.add(oid)
            objs.append(r)

    cat_map = {"轻脆": "易碎", "重脆": "易碎", "柔性": "柔性", "刚体": "刚体"}
    n_incons_b = 0
    by_pair = {}
    for r in objs:
        local = cat_map.get(r["category"], r["category"])
        stem = oid2stem.get(r["object_id"], "")
        off = name2cat.get(stem, stem2cat.get(stem, "?"))
        by_pair.setdefault((local, off), []).append(r["object_id"])
        if local != off:
            n_incons_b += 1
            print(f"{r['object_id']:8s} {r.get('object_name','')[:20]:24s} 本地={r['category']}/{r['leaf_class']}"
                  f" 官方={off} [不一致]")
    print(f"\n层面B不一致: {n_incons_b}/{len(objs)} 个物体")
    print("\n本地(映射后) x 官方 混淆分布:")
    for (local, off), ids in sorted(by_pair.items(), key=lambda kv: -len(kv[1])):
        mark = "  <-- 不一致" if local != off else ""
        print(f"  {local:4s} x {off:4s}: {len(ids):3d} 个  {','.join(ids[:12])}{'...' if len(ids) > 12 else ''}{mark}")

    for (local, off), ids in sorted(by_pair.items(), key=lambda kv: -len(kv[1])):
        if local != off:
            for oid in ids:
                lr = next(r for r in lrows if r["object_id"] == oid)
                report.append({
                    "层面": "B_labels本地", "object_id": oid,
                    "object_name": lr.get("object_name", ""), "本地口径": f"{lr['category']}/{lr['leaf_class']}",
                    "ref_category": "", "ExpForce官方": off, "不一致": "是",
                    "备注": f"l1_gate={lr['l1_gate']}, family={lr['object_family_id']}",
                })

    with open(OUT, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(report[0].keys()))
        w.writeheader()
        w.writerows(report)
    print(f"\n已写出: {OUT} ({len(report)} 行)")


if __name__ == "__main__":
    main()

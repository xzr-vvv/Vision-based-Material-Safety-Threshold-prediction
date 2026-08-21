# -*- coding: utf-8 -*-
"""ExpForce 力值汇总: 按类别的 f_min 分布 + 易碎明细 + 族级映射候选"""
import csv
import io
import statistics
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

with open(r"E:\A-触觉机器学习\ExpForce数据集\07_ExpForce_安全抓力范围.csv", newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

cat = {}
for r in rows:
    cat.setdefault(r["category"], []).append(r)

print("== 按类别 f_min 分布 ==")
for c, rs in sorted(cat.items()):
    vals = []
    for r in rs:
        try:
            vals.append(float(r["f_min_value"]))
        except (ValueError, KeyError):
            pass
    if vals:
        print(f"{c}: n={len(vals)}, 中位={statistics.median(vals):.2f}N, "
              f"min={min(vals)}, max={max(vals)}")

print("\n== 易碎明细 ==")
for r in cat.get("易碎", []):
    print(f"{r['object_id']}|{r['object_name']}|f_min={r['f_min_value']}"
          f"|max_safe={r.get('max_safe_force_N','')}|img={r.get('image_file','')}")

print("\n== 刚体明细(前20) ==")
for r in cat.get("刚体", [])[:20]:
    print(f"{r['object_id']}|{r['object_name']}|f_min={r['f_min_value']}")

print("\n== 柔性明细(前20) ==")
for r in cat.get("柔性", [])[:20]:
    print(f"{r['object_id']}|{r['object_name']}|f_min={r['f_min_value']}")

print("\n== 图片字段示例 ==")
for r in rows[:3]:
    print(r.get("image_file", ""), "|", r.get("data_source", ""))

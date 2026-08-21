# -*- coding: utf-8 -*-
r"""0821 force_map 力值清理 (只保留别人实测的值):
- 所有行: f_max 为估算/外推(非实测) -> 值清空, 原值移入 f_max_basis 留档
- FRA 行: f_min 为近似转移(无同类型实测来源) -> 值清空, 原值移入 evidence 留档
- 保留: ExpForce本物实测(E系f_min) / 同型实测与同型转移(FRA f_min) / 文献·国标实测(f_max)
"""
import csv

FM = r"E:\A-触觉机器学习\RGB_dataset\force_map.csv"

n_fmax = n_fmin = 0

with open(FM, encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    cols = reader.fieldnames
    rows = list(reader)

for r in rows:
    sem = r["f_max_semantics"] or ""
    if "估算" in sem:
        old = r["f_max_label_N"]
        note = f"[0821删] 原估算值{old}N({sem})"
        r["f_max_basis"] = (r["f_max_basis"] + "; " if r["f_max_basis"] else "") + note
        r["f_max_label_N"] = ""
        r["f_max_total_N"] = ""
        r["f_max_semantics"] = "未找到实测"
        n_fmax += 1
    if r["object_id"].startswith("FRA") and "近似转移" in (r["label_source"] or ""):
        old = r["f_min_label_N"]
        old_src = r["label_source"]
        r["evidence"] = (r["evidence"] or "") + f" | [0821删] 原近似转移值{old}N({old_src})"
        r["f_min_label_N"] = ""
        r["f_min_single_N"] = ""
        r["f_min_semantics"] = ""
        r["label_source"] = "未找到实测"
        r["value_basis"] = "未找到实测"
        n_fmin += 1

with open(FM, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)

print(f"清理估算 f_max: {n_fmax} 行; 清理 FRA 近似 f_min: {n_fmin} 行; 共 {len(rows)} 行")

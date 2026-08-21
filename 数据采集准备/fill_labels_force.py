# -*- coding: utf-8 -*-
r"""0821 用清理后的 force_map 回填 labels.csv 力值列:
- f_min_measured_N <- f_min_label_N (仅存 ExpForce本物实测 / 同型实测·同型转移)
- f_max_measured_N <- f_max_total_N (仅存文献/国标实测, 两指合力语义)
- note 列写明每列的证据来源, 无实测则明确标注
"""
import csv
import os

ROOT = r"E:\A-触觉机器学习\RGB_dataset"
FM = os.path.join(ROOT, "force_map.csv")
LABELS = os.path.join(ROOT, "labels.csv")

with open(FM, encoding="utf-8-sig", newline="") as f:
    fm = {r["object_id"]: r for r in csv.DictReader(f)}

with open(LABELS, encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    cols = reader.fieldnames
    rows = list(reader)

n_min = n_max = 0
for r in rows:
    m = fm.get(r["object_id"])
    fmin = fmax = ""
    parts = []
    if m:
        fmin = (m["f_min_label_N"] or "").strip()
        fmax = (m["f_max_label_N"] or "").strip()
        fmax = (m["f_max_total_N"] or "").strip() if fmax else ""
        if fmin:
            n_min += 1
            if (m["label_source"] or "").startswith("ExpForce实测"):
                parts.append(f"f_min:ExpForce本物实测({fmin}N,两指合力)")
            else:
                parts.append(f"f_min:{m['label_source']}({fmin}N,两指合力)")
        else:
            parts.append("f_min:无实测")
        if fmax:
            n_max += 1
            sem = m["f_max_semantics"] or ""
            orig = (m["f_max_label_N"] or "").strip()
            if "保守按合力" in sem:
                parts.append(f"f_max:{m['f_max_basis']}({orig}N,夹持实测,单/总未确认保守按合力)")
            else:
                parts.append(f"f_max:文献实测{orig}N({sem})×2换算={fmax}N(两指合力,非直接实测)")
        else:
            parts.append("f_max:无实测")
    else:
        parts.append("force_map无条目")
    r["f_min_measured_N"] = fmin
    r["f_max_measured_N"] = fmax
    r["note"] = " | ".join(parts)

with open(LABELS, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)

print(f"labels.csv {len(rows)} 行: f_min 填 {n_min} 行, f_max 填 {n_max} 行")

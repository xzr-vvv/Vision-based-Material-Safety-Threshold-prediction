# -*- coding: utf-8 -*-
"""导出采购清单: object_id | leaf | 名称 | 族 | source"""
import csv
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

with open(r"E:\A-触觉机器学习\数据采集准备\objects_模板.csv", newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

print("字段:", list(rows[0].keys()))
print()
for r in rows:
    print(f"{r['object_id']}|{r['leaf_class']}|{r['object_name']}|{r['object_family_id']}|{r.get('source','')}")

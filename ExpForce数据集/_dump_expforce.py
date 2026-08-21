# -*- coding: utf-8 -*-
"""导出 ExpForce 全部物体: id|名称|类别|f_min|f_max (用于同族力值转移映射)"""
import csv
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

with open(r"E:\A-触觉机器学习\ExpForce数据集\07_ExpForce_安全抓力范围.csv", newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

print("字段:", list(rows[0].keys()))
print()
for r in rows:
    print(f"{r.get('object_id','')}|{r.get('object_name', r.get('name',''))}|{r.get('category','')}|{r.get('f_min_value', r.get('f_min',''))}|{r.get('f_max_value', r.get('f_max',''))}")

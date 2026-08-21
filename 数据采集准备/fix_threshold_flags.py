# -*- coding: utf-8 -*-
r"""统一 objects_已有安全阈值.csv 的分类 flags (0821):
FRA016/FRA039: fragile 1->0, deformable 0->1, 对齐本地模板/labels/ExpForce官方(柔性) — 过时标记同步
FRA048/SOF053: flags 保留本地口径, 仅在 notes 补充分歧理由
"""
import csv
import shutil

PATH = r"E:\A-触觉机器学习\数据采集准备\objects_已有安全阈值.csv"
BAK = PATH + ".bak_before_flag_unify"

FLAG_FIX = {
    "FRA016": ("[0821改]flags原为易碎,现对齐本地模板/labels(柔性);ExpForce官方亦为柔性;失效模式=压瘪变形非脆断"),
    "FRA039": ("[0821改]flags原为易碎,现对齐本地模板/labels(柔性);ExpForce官方亦为柔性;失效模式=压瘪变形非脆断"),
}
NOTE_APPEND = {
    "FRA048": ("[0821注]保留本地轻脆(安全语义:盒内脆性饼干,压溃即损);ExpForce刚体=其力程分类口径,非损伤口径;两口径并存于本表"),
    "SOF053": ("[0821注]E055参照为瓶装42粒(刚体瓶),本地为袋装(柔性),容器材质不同;保留本地柔性;F_min转移已按同型处理"),
}

shutil.copyfile(PATH, BAK)
with open(PATH, encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    fields = list(reader.fieldnames)
    rows = list(reader)

for r in rows:
    oid = r["object_id"]
    if oid in FLAG_FIX:
        r["fragile_flag"] = "0"
        r["deformable_flag"] = "1"
        r["notes"] = (r["notes"] + ";" if r["notes"] else "") + FLAG_FIX[oid]
    elif oid in NOTE_APPEND:
        r["notes"] = (r["notes"] + ";" if r["notes"] else "") + NOTE_APPEND[oid]

with open(PATH, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

for r in rows:
    if r["object_id"] in FLAG_FIX or r["object_id"] in NOTE_APPEND:
        print(r["object_id"], "fragile=", r["fragile_flag"], "deformable=", r["deformable_flag"])
print("备份:", BAK)

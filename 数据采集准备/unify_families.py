# -*- coding: utf-8 -*-
"""统一 object_family_id: 合并同型异族 (防族分组切分泄漏), 同步 labels.csv 与 objects_模板.csv。

合并依据 (0821, 对齐 objects_已有安全阈值.csv 的同型参照关系):
- 蛋: large_brown_egg/large_white_egg -> egg (csv 中 FRA024/025/051 均为 egg)
- 苹果: golden/green/red_apple -> apple
- 葡萄: red/green_grape -> grape
- 番茄: roma + red/green/yellow_grape_tomato -> tomato (SOF003 参照 E020/E021)
- 椒: mini/yellow_mini_bell_pepper -> bell_pepper (SOF013 参照 E025)
- 柠檬/青柠/橙/橘: large_lemon->lemon, small_lime->lime, cara_cara->orange, mandarins->mandarin
- 棉花糖: large_marshmallow -> marshmallow (SOF030 参照)
- 玻璃杯: glass_tumbler/glass_whiskey_cup -> glass_cup (FRA001/002 参照 E116/E041)
- 高脚杯: wine/margarita/champagne_glass -> glass_stemware (FRA003 参照)
- 玻璃瓶罐: glass_cocacola_bottle/blueberry_jam_jar -> glass_bottle (FRA005/006 参照)
- 陶瓷杯: ceramic_mug -> ceramic_cup (FRA010 参照 E120)
- 脆零食: pocky_snack_box/single_pringles_potato_chip -> brittle_snack (FRA048/FRA019 参照)
"""
import csv
import os
import shutil

ROOT = r"E:\A-触觉机器学习"
LABELS = os.path.join(ROOT, "RGB_dataset", "labels.csv")
TEMPLATE = os.path.join(ROOT, "数据采集准备", "objects_模板.csv")

MERGE = {
    "large_brown_egg": "egg", "large_white_egg": "egg",
    "golden_apple": "apple", "green_apple": "apple", "red_apple": "apple",
    "red_grape": "grape", "green_grape": "grape",
    "roma_tomato": "tomato", "red_grape_tomato": "tomato",
    "green_grape_tomato": "tomato", "yellow_grape_tomato": "tomato",
    "mini_bell_pepper": "bell_pepper", "yellow_mini_bell_pepper": "bell_pepper",
    "large_lemon": "lemon", "small_lime": "lime",
    "cara_cara_orange": "orange", "mandarins": "mandarin",
    "large_marshmallow": "marshmallow",
    "glass_tumbler": "glass_cup", "glass_whiskey_cup": "glass_cup",
    "wine_glass": "glass_stemware", "margarita_glass": "glass_stemware",
    "champagne_glass": "glass_stemware",
    "glass_cocacola_bottle": "glass_bottle", "blueberry_jam_jar": "glass_bottle",
    "ceramic_mug": "ceramic_cup",
    "pocky_snack_box": "brittle_snack", "single_pringles_potato_chip": "brittle_snack",
}


def apply(path, key_field, backup_tag):
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    changed = []
    for r in rows:
        old = r.get("object_family_id", "")
        if old in MERGE:
            r["object_family_id"] = MERGE[old]
            changed.append((r[key_field], old, MERGE[old]))
    shutil.copyfile(path, path + backup_tag)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows, changed


def main():
    rows, changed = apply(LABELS, "object_id", ".bak_before_family_unify")
    fams = {r["object_family_id"] for r in rows}
    print(f"[labels.csv] 图片行 {len(rows)}, 改动物体 {len(changed)} 个, "
          f"族数 128 -> {len(fams)}")
    for oid, old, new in changed:
        print(f"  {oid}: {old} -> {new}")

    rows2, changed2 = apply(TEMPLATE, "object_id", ".bak_before_family_unify")
    fams2 = {r["object_family_id"] for r in rows2}
    print(f"\n[objects_模板.csv] 行 {len(rows2)}, 改动 {len(changed2)} 行, "
          f"族数 -> {len(fams2)}")


if __name__ == "__main__":
    main()

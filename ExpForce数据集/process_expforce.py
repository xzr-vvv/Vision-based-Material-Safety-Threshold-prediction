# -*- coding: utf-8 -*-
"""Exp-Force 数据集一致性处理
输入: E:/A-机器学习/ExpForce_dataset.csv (129 物体, 官方直接发布: 物体名/图片/质量/真值最小抓取力 F*)
处理: 1) 按 刚体/柔性/易碎 三类归类(规则法, 见 FRAGILE/FLEXIBLE 关键词)
      2) 最大安全力 = 类别安全系数 k x 实测最小力 F* (官方未发布损坏阈值, 统一规则补齐)
         k: 易碎=2.0, 柔性=3.0, 刚体=5.0; 结果向上取整到 0.25 N (与官方 F* 精度一致)
输出: E:/A-机器学习/07_ExpForce_安全抓力范围.csv (单项目单一格式, 直接可用于监督学习)
"""
import csv
import math
import os

SRC = r"E:\A-机器学习\ExpForce_dataset.csv"
DST = r"E:\A-机器学习\07_ExpForce_安全抓力范围.csv"

# 易碎: 脆性材料, 受力超过阈值会突然破裂(玻璃/陶瓷/蛋壳/薯片)
FRAGILE_KEYS = [
    "egg", "champagne glass", "wine glass", "margarita glass",
    "whiskey cup", "glass tumbler", "ceramic mug", "glass coca-cola",
    "pringles potato chip", "jam jar",
]
# 柔性: 可大变形/易压伤(软食品/生鲜果蔬/纸杯塑料杯/软包装/吸管)
FLEXIBLE_KEYS = [
    "marshmallow", "banana", "strawberry", "grape", "blueberry", "raspberry",
    "blackberry", "tomato", "pepper", "lemon", "lime", "orange", "mandarin",
    "apple", "pear", "grapefruit", "paper cup", "plastic cup", "seaweed",
    "oreo thin cookie pack", "straw",
]
# 容器词: 命中说明是硬质包装而非生鲜果蔬本身(如 "Sunkist Orange soda" 是易拉罐)
CONTAINER_WORDS = ["soda", "gummi", "bottle", "can", "jar", "box", "adapter", "can"]

K_FACTOR = {"易碎": 2.0, "柔性": 3.0, "刚体": 5.0}


def classify(name: str) -> str:
    n = name.lower()
    if any(k in n for k in FRAGILE_KEYS):
        return "易碎"
    if any(k in n for k in CONTAINER_WORDS):
        return "刚体"
    if any(k in n for k in FLEXIBLE_KEYS):
        return "柔性"
    return "刚体"


def ceil_quarter(x: float) -> float:
    return math.ceil(x / 0.25) * 0.25


def main():
    with open(SRC, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    out = []
    for i, r in enumerate(rows, 1):
        name = r["Object"].strip()
        cat = classify(name)
        fmin = float(r["Gripping Force"])
        fmax = ceil_quarter(fmin * K_FACTOR[cat])
        out.append({
            "object_id": f"E{i:03d}",
            "object_name": name,
            "category": cat,
            "mass_g": float(r["Mass"]),
            "min_grasp_force_N": fmin,
            "max_safe_force_N": fmax,
            "safe_range_N": round(fmax - fmin, 2),
            "unit": "N",
            "image_file": r["Image"].strip(),
            "data_source": "Exp-Force(UT Austin, arXiv:2603.08668)实测最小力; 最大力=类别系数x实测最小力",
            "notes": f"k={K_FACTOR[cat]}",
        })

    with open(DST, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    cnt = {}
    for o in out:
        cnt[o["category"]] = cnt.get(o["category"], 0) + 1
    print("总物体数:", len(out))
    for c in ("刚体", "柔性", "易碎"):
        print(f"  {c}: {cnt.get(c, 0)}")
    img_missing = [o["image_file"] for o in out
                   if not os.path.isfile(os.path.join(r"E:\A-机器学习\ExpForce_images", o["image_file"]))]
    print("图片缺失:", img_missing if img_missing else "无")
    print("\n各类代表物体:")
    for c in ("刚体", "柔性", "易碎"):
        reps = [o["object_name"] for o in out if o["category"] == c][:8]
        print(f"  {c}: {', '.join(reps)} ...")


if __name__ == "__main__":
    main()

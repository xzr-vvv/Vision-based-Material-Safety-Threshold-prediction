# -*- coding: utf-8 -*-
"""升级 force_map.csv: 区分 实测-本物/同型实测/近似转移/估算, 增加取值依据和证据列.

同型实测 = ExpForce(UT Austin) 用同款夹爪(LEGATO+FORTE)在同类商品上实测的夹持力,
          与我们的物体是同一商品类型、不同个体.
"""
import csv
import os

RGB = r"E:\A-触觉机器学习\RGB_dataset"
FORCE_MAP = os.path.join(RGB, "force_map.csv")

EXPFORCE_EV = ("ExpForce_dataset_官方原始.csv (UT Austin; LEGATO两指爪+FORTE触觉指实测夹持力)")

# object_id -> (ExpForce物体名, 实测f_min N)  同一商品类型
SAME_TYPE = {
    "FRA001": ("Glass tumbler", 1.5),
    "FRA003": ("Wine glass", 1.0),
    "FRA006": ("Blueberry jam jar (Empty)", 0.75),
    "FRA010": ("Ceramic mug", 1.5),
    "FRA016": ("Medium paper cup", 0.25),
    "FRA024": ("Large brown egg", 0.5),
    "FRA039": ("Medium plastic cup", 0.25),
    "FRA048": ("Pocky snack box (Long-edge orientation)", 1.0),
    "RIG001": ("Diet Coke can (Red Bull 2.5/Monster 2.0/7UP 2.0/Sunkist 2.0)", 2.0),
    "RIG004": ("Head & Shoulders shampoo bottle", 2.0),
    "RIG005": ("Spam canned meat", 2.5),
    "RIG008": ("Pringles can", 1.5),
    "RIG013": ("Magic tape roll (Duct tape roll 1.0)", 2.0),
    "RIG016": ("Green power bank", 2.0),
    "RIG017": ("TV remote control", 1.0),
    "RIG049": ("Umbrella (Folded)", 1.5),
    "SOF001": ("Banana", 1.0),
    "SOF002": ("Orange 1 (Orange 2 2.0/CaraCara 1.5)", 1.5),
    "SOF003": ("Roma tomato 1 (Roma2 2.0)", 2.0),
    "SOF004": ("Strawberry 1 (large 1.0)", 0.5),
    "SOF008": ("Lemon (Large lemon 0.75)", 0.5),
    "SOF009": ("Red grape tomato 1", 0.5),
    "SOF030": ("Marshmallow 1 (Large marshmallow 0.25)", 0.25),
    "SOF031": ("Green pear", 1.5),
}

# 近似但有 ExpForce 同类参照 (保持估算属性, 但记录参照值)
NEAR_REF = {
    "FRA002": "Glass tumbler 1.5N (薄壁差异)",
    "FRA005": "Glass Coca-Cola bottle 2.5N (满瓶)",
    "FRA025": "Large brown egg 0.5N (皮蛋)",
    "FRA031": "Glass tumbler 1.5N",
    "FRA032": "Blueberry jam jar (Empty) 0.75N",
    "FRA034": "Ceramic mug 1.5N",
    "FRA051": "Large brown egg 0.5N (鸭蛋更大)",
    "FRA056": "Large brown egg 0.5N (空壳更脆)",
    "FRA066": "Glass Coca-Cola bottle 2.5N (空瓶)",
    "RIG006": "StarKist tuna 1.5N/Tomato soup 2.5N (铁罐)",
    "RIG024": "Protein shaker bottle 2.0N",
    "RIG025": "Clorox cleaner bottle 3.25N (喷瓶)",
    "RIG054": "Yellow mustard bottle 3.25N",
    "RIG057": "Muffin mix box 2.0N (纸盒)",
    "RIG058": "Spam canned meat 2.5N (罐装)",
    "SOF005": "Red/Green grapes 0.5N (单粒)",
    "SOF013": "Mini bell peppers 0.5N/Yellow 1.0N",
}


def main():
    with open(FORCE_MAP, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
        fields = list(rows[0].keys())
    for new_col in ("value_basis", "evidence"):
        if new_col not in fields:
            fields.append(new_col)

    n_same, n_near, n_direct = 0, 0, 0
    for r in rows:
        oid = r["object_id"]
        if oid in SAME_TYPE:
            name, fmin = SAME_TYPE[oid]
            r["f_min_label_N"] = str(fmin)
            r["label_source"] = f"ExpForce同型实测({name}; f_min={fmin}N)"
            r["is_measured"] = "0"
            r["value_basis"] = "同型实测"
            r["evidence"] = EXPFORCE_EV
            n_same += 1
        elif oid in NEAR_REF:
            r["value_basis"] = "近似转移"
            r["evidence"] = f"ExpForce同类参照: {NEAR_REF[oid]}"
            n_near += 1
        elif oid.startswith("E") and r.get("is_measured") == "1":
            r["value_basis"] = "本物实测"
            r["evidence"] = EXPFORCE_EV + "; f_max=5×f_min外推(待复核)"
            n_direct += 1
        else:
            r["value_basis"] = "估算" if ("近似" in r["label_source"] or "系数" in r["label_source"]) else "近似转移"
            r["evidence"] = ""
    with open(FORCE_MAP, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"同型实测: {n_same}, 近似(有ExpForce参照): {n_near}, 本物实测: {n_direct}, 总: {len(rows)}")


if __name__ == "__main__":
    main()

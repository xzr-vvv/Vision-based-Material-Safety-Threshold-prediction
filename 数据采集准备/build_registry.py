# -*- coding: utf-8 -*-
"""构建 RGB_dataset 物体注册表:
1. 追加 ExpForce 129 个物体(真实实测力值, 叶类按四类划分)
2. 追加 YCB-Video 21 个物体(同型实测值转移)
3. 给采购清单 186 项填转移力值(同型/近似, 来源逐条记录)
输出: objects_模板.csv 更新 + force_map.csv(供 check_dataset 使用)
"""
import csv
import io
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

OBJ_CSV = r"E:\A-触觉机器学习\数据采集准备\objects_模板.csv"
EXP_CSV = r"E:\A-触觉机器学习\ExpForce数据集\07_ExpForce_安全抓力范围.csv"
FORCE_MAP = r"E:\A-触觉机器学习\RGB_dataset\force_map.csv"

# ---------- ExpForce 易碎 -> 轻/重脆 划分 ----------
EXP_LIGHT = {"E040", "E060", "E061"}          # 薯片/鸡蛋: 轻脆
# 其余易碎(E008,E041,E108,E116-E120) -> 重脆(玻璃/陶瓷)

# ---------- YCB 物体: (名称, 叶类, 转移源E编号, 转移类型) ----------
YCB = {
    1:  ("hmaster_sauce_bottle", "刚体", "E093", "同型转移(芥末酱瓶)"),
    2:  ("master_chef_can", "刚体", "E003", "同型转移(肉罐头)"),
    3:  ("cracker_box", "刚体", "E091", "同型转移(饼干盒)"),
    4:  ("sugar_box", "刚体", "E111", "同型转移(糖盒)"),
    5:  ("tomato_soup_can", "刚体", "E095", "同型转移(番茄汤罐,同型)"),
    6:  ("mustard_bottle", "刚体", "E093", "同型转移(芥末瓶,同型)"),
    7:  ("tuna_fish_can", "刚体", "E100", "同型转移(金枪鱼罐,同型)"),
    8:  ("pudding_box", "刚体", "E112", "近似转移(零食盒)"),
    9:  ("gelatin_box", "刚体", "E112", "近似转移(零食盒)"),
    10: ("potted_meat_can", "刚体", "E003", "同型转移(午餐肉罐,同型)"),
    11: ("banana", "柔性", "E056", "同型转移(香蕉,同型)"),
    12: ("strawberry", "柔性", "E026", "同型转移(草莓,同型)"),
    13: ("peach", "柔性", "E067", "近似转移(苹果)"),
    14: ("pear", "柔性", "E064", "同型转移(梨,同型)"),
    15: ("citrus_orange", "柔性", "E029", "同型转移(橙子,同型)"),
    16: ("apple", "柔性", "E067", "同型转移(苹果,同型)"),
    17: ("plum", "柔性", "E075", "近似转移(柑橘)"),
    18: ("clamp", "刚体", "E053", "近似转移(手动工具)"),
    19: ("power_drill", "刚体", "E129", "同型转移(电钻,同型)"),
    20: ("wood_block", "刚体", "E053", "近似转移(硬质工具)"),
    21: ("scissors", "刚体", "E053", "近似转移(手动工具)"),
}

# ---------- 采购清单物体转移力值: object_id -> (源E编号, 转移类型) ----------
TRANSFER = {
    # 重脆(玻璃/陶瓷 -> ExpForce 玻璃/陶瓷实测)
    "FRA001": ("E116", "同型转移(玻璃杯)"), "FRA002": ("E116", "同型转移(玻璃杯)"),
    "FRA003": ("E119", "同型转移(红酒杯)"), "FRA004": ("E008", "近似转移(玻璃空心器皿)"),
    "FRA005": ("E108", "同型转移(玻璃汽水瓶)"), "FRA006": ("E008", "同型转移(玻璃罐)"),
    "FRA007": ("E008", "近似转移(薄壁玻璃器皿)"), "FRA008": ("E041", "近似转移(厚壁玻璃)"),
    "FRA009": ("E118", "近似转移(薄壁玻璃饰品)"), "FRA010": ("E120", "同型转移(陶瓷马克杯,同型)"),
    "FRA011": ("E120", "同型转移(陶瓷杯)"), "FRA012": ("E120", "近似转移(陶瓷餐具)"),
    "FRA013": ("E120", "近似转移(陶瓷餐具)"), "FRA014": ("E120", "近似转移(陶瓷摆件)"),
    "FRA015": ("E120", "近似转移(陶瓷摆件)"), "FRA031": ("E116", "同型转移(玻璃杯)"),
    "FRA032": ("E008", "同型转移(玻璃罐)"), "FRA033": ("E117", "近似转移(玻璃烛台)"),
    "FRA034": ("E120", "同型转移(陶瓷咖啡杯)"), "FRA035": ("E120", "近似转移(陶瓷碗)"),
    "FRA036": ("E120", "近似转移(陶瓷碟)"), "FRA037": ("E120", "近似转移(陶瓷餐具)"),
    "FRA038": ("E120", "近似转移(陶瓷摆件)"), "FRA058": ("E120", "近似转移(陶瓷餐具)"),
    "FRA059": ("E120", "近似转移(陶瓷片)"), "FRA060": ("E116", "近似转移(玻璃罐装)"),
    "FRA061": ("E008", "近似转移(玻璃容器)"), "FRA062": ("E116", "近似转移(玻璃碗)"),
    "FRA063": ("E120", "近似转移(陶瓷壶)"), "FRA064": ("E120", "近似转移(陶瓷盆)"),
    "FRA065": ("E120", "近似转移(陶瓷罐)"), "FRA066": ("E108", "近似转移(玻璃瓶)"),
    # 轻脆
    "FRA017": ("E060", "近似转移(薄壁空心,待实测复核)"),
    "FRA018": ("E060", "近似转移(薄壁空心壳)"),
    "FRA019": ("E012", "同型转移(密封零食袋)"), "FRA020": ("E091", "同型转移(薄脆饼干盒)"),
    "FRA021": ("E038", "近似转移(脆零食盒)"), "FRA022": ("E012", "近似转移(密封零食袋)"),
    "FRA023": ("E040", "近似转移(脆皮烘焙)"), "FRA024": ("E060", "同型转移(鸡蛋,同型)"),
    "FRA025": ("E060", "同型转移(蛋类)"), "FRA026": ("E060", "近似转移(薄壳)"),
    "FRA027": ("E040", "近似转移(脆性捆扎物)"), "FRA028": ("E060", "近似转移(脆性文具)"),
    "FRA029": ("E060", "近似转移(脆性柱体)"), "FRA030": ("E060", "近似转移(石膏脆性)"),
    "FRA041": ("E091", "同型转移(曲奇盒)"), "FRA042": ("E038", "近似转移(威化条装)"),
    "FRA043": ("E012", "近似转移(密封零食袋)"), "FRA044": ("E040", "近似转移(脆性零食)"),
    "FRA045": ("E038", "近似转移(月饼糕点)"), "FRA046": ("E038", "近似转移(油炸脆点)"),
    "FRA047": ("E040", "近似转移(酥性饼干)"), "FRA048": ("E038", "同型转移(Pocky盒,同型)"),
    "FRA049": ("E040", "近似转移(膨化脆糖)"), "FRA050": ("E001", "近似转移(糖果)"),
    "FRA051": ("E060", "同型转移(蛋类)"), "FRA052": ("E040", "近似转移(干燥脆物)"),
    "FRA053": ("E040", "近似转移(干燥脆物)"), "FRA054": ("E040", "近似转移(脆性捆扎物)"),
    "FRA055": ("E040", "近似转移(干燥脆物)"), "FRA056": ("E060", "同型转移(蛋壳,同型)"),
    "FRA057": ("E060", "近似转移(脆性文具)"),
    # 柔性(自采清单)
    "SOF002": ("E029", "同型转移(橙子,同型)"), "SOF003": ("E020", "同型转移(番茄,同型)"),
    "SOF004": ("E026", "同型转移(草莓,同型)"), "SOF005": ("E032", "同型转移(葡萄,同型)"),
    "SOF006": ("E024", "近似转移(猕猴桃)"), "SOF007": ("E024", "近似转移(牛油果)"),
    "SOF008": ("E023", "同型转移(柠檬,同型)"), "SOF009": ("E080", "同型转移(圣女果,同型)"),
    "SOF010": ("E067", "近似转移(桃)"), "SOF011": ("E020", "近似转移(玉米)"),
    "SOF012": ("E022", "近似转移(西兰花)"), "SOF013": ("E025", "同型转移(彩椒,同型)"),
    "SOF014": ("E026", "近似转移(蘑菇)"), "SOF015": ("E013", "近似转移(软面包)"),
    "SOF016": ("E013", "近似转移(贝果)"), "SOF017": ("E013", "近似转移(纸杯蛋糕)"),
    "SOF018": ("E013", "近似转移(馒头)"), "SOF019": ("E013", "近似转移(麻薯)"),
    "SOF020": ("E013", "近似转移(毛巾)"), "SOF023": ("E013", "近似转移(毛绒)"),
    "SOF024": ("E013", "近似转移(海绵)"), "SOF030": ("E013", "同型转移(棉花糖,同型)"),
    "SOF038": ("E013", "近似转移(甜甜圈)"),
    "SOF001": ("E056", "同型转移(香蕉,同型)"),
    "FRA016": ("E005", "同型转移(纸杯,同型)"), "FRA039": ("E007", "同型转移(塑料杯,同型)"),
    "FRA040": ("E007", "近似转移(薄壁塑料盒)"),
    # 刚体(自采清单)
    "RIG010": ("E053", "近似转移(手动工具)"), "RIG011": ("E053", "近似转移(手动工具)"),
    "RIG014": ("E053", "近似转移(桌面工具)"), "RIG015": ("E015", "近似转移(手机类电子)"),
    "RIG016": ("E019", "同型转移(充电宝,同型)"), "RIG017": ("E050", "同型转移(遥控器,同型)"),
    "RIG019": ("E112", "近似转移(书本)"), "RIG021": ("E121", "近似转移(硬塑料块)"),
    "RIG029": ("E053", "近似转移(手动工具)"), "RIG030": ("E052", "近似转移(卷状物)"),
    "RIG031": ("E053", "近似转移(手动工具)"), "RIG032": ("E053", "近似转移(刀具)"),
    "RIG033": ("E015", "近似转移(桌面电子)"), "RIG034": ("E015", "近似转移(桌面电子)"),
    "RIG035": ("E015", "近似转移(手持电子)"), "RIG038": ("E053", "近似转移(硬质文具)"),
    # 其余刚体清单项(暂不采集也预填,便于后续自采)
    "RIG001": ("E044", "同型转移(易拉罐,同型)"), "RIG002": ("E043", "同型转移(瓶装饮料,同型)"),
    "RIG003": ("E047", "近似转移(硬质瓶壶)"), "RIG004": ("E107", "同型转移(洗发水瓶,同型)"),
    "RIG005": ("E003", "同型转移(罐头,同型)"), "RIG006": ("E097", "同型转移(食品罐,同型)"),
    "RIG007": ("E010", "近似转移(铁罐)"), "RIG008": ("E002", "同型转移(薯片罐,同型)"),
    "RIG009": ("E090", "近似转移(纸盒装)"), "RIG012": ("E053", "近似转移(工具)"),
    "RIG013": ("E052", "同型转移(胶带卷,同型)"), "RIG018": ("E015", "近似转移(桌面电子)"),
    "RIG020": ("E112", "近似转移(硬面本)"), "RIG022": ("E046", "近似转移(杯面)"),
    "RIG023": ("E043", "同型转移(碳酸饮料瓶,同型)"), "RIG024": ("E047", "近似转移(运动水壶)"),
    "RIG025": ("E105", "近似转移(喷瓶)"), "RIG026": ("E097", "近似转移(食品罐)"),
    "RIG027": ("E090", "近似转移(利乐盒)"), "RIG028": ("E090", "近似转移(利乐盒)"),
    "RIG036": ("E015", "近似转移(小型电子)"), "RIG037": ("E014", "近似转移(电子配件)"),
    "RIG039": ("E053", "近似转移(硬质文具)"), "RIG040": ("E052", "近似转移(文具)"),
    "RIG041": ("E052", "近似转移(文具)"), "RIG042": ("E053", "近似转移(文具)"),
    "RIG043": ("E128", "近似转移(收纳硬盒)"), "RIG044": ("E057", "近似转移(个人护理)"),
    "RIG045": ("E113", "近似转移(小盒装)"), "RIG046": ("E051", "近似转移(小型电子)"),
    "RIG047": ("E121", "近似转移(硬壳盒)"), "RIG048": ("E128", "近似转移(卷状硬物)"),
    "RIG049": ("E128", "近似转移(卷状硬物)"), "RIG050": ("E104", "近似转移(硬塑料容器)"),
    "RIG051": ("E053", "近似转移(金属重物)"), "RIG052": ("E104", "近似转移(硬塑料容器)"),
    "RIG053": ("E053", "近似转移(金属容器)"), "RIG054": ("E094", "近似转移(调料瓶)"),
    "RIG055": ("E111", "近似转移(糖盒)"), "RIG056": ("E097", "近似转移(食品铁盒)"),
    "RIG057": ("E090", "近似转移(谷物盒)"), "RIG058": ("E044", "近似转移(罐装食品)"),
    "RIG059": ("E090", "近似转移(咖啡盒)"), "RIG060": ("E094", "近似转移(奶粉罐)"),
}


def fam_of(name):
    """ExpForce 物体名 -> 族: 去括号/尾数字/序号"""
    s = re.sub(r"\(.*?\)", "", name)
    s = re.sub(r"\s+\d+$", "", s.strip())
    s = re.sub(r"[^\w\s]", "", s)
    return "_".join(s.lower().split()) or "unknown"


def main():
    # 读取 ExpForce 全部行
    with open(EXP_CSV, newline="", encoding="utf-8-sig") as f:
        exp_rows = list(csv.DictReader(f))
    exp_by_id = {r["object_id"]: r for r in exp_rows}

    def fmin(eid):
        return float(exp_by_id[eid]["f_min_value"])

    def fmax_of(r):
        v = r.get("max_safe_force_N", "")
        try:
            return float(v)
        except ValueError:
            return round(2 * float(r["f_min_value"]), 2)

    # ---------- 读取现有清单 ----------
    with open(OBJ_CSV, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames)
        rows = list(reader)
    by_id = {r["object_id"]: r for r in rows}

    # ---------- 1. 采购清单填转移力值 ----------
    n_same = n_approx = 0
    for oid, (src, kind) in TRANSFER.items():
        r = by_id.get(oid)
        if r is None:
            continue
        v = fmin(src)
        r["f_min_value"] = v
        r["f_min_unit"] = "N"
        r["force_total_or_single"] = "total"
        r["measurement_protocol"] = f"{kind}: 来源ExpForce {src}({exp_by_id[src]['object_name']})"
        r["gripper_model"] = exp_by_id[src].get("gripper_model", "")
        r["sensor_model"] = exp_by_id[src].get("sensor_model", "")
        if "近似" in kind:
            r["notes"] = (r["notes"] + "; " if r["notes"] else "") + "力值为近似转移,待实测复核"
            n_approx += 1
        else:
            n_same += 1

    # ---------- 2. 追加 YCB 21 物体 ----------
    def blank_row():
        return {k: "" for k in fields}

    for num, (name, leaf, src, kind) in YCB.items():
        oid = f"YCB{num:02d}"
        if oid in by_id:
            continue
        r = blank_row()
        r.update(object_id=oid, object_family_id=f"ycb_{name}", object_name=f"YCB {name}",
                 material_class="", fragile_flag="1" if "脆" in leaf else "0",
                 deformable_flag="1" if leaf == "柔性" else "0", leaf_class=leaf,
                 source="YCB-Video(BOP ycbv test)", f_min_value=fmin(src), f_min_unit="N",
                 force_total_or_single="total",
                 measurement_protocol=f"{kind}: 来源ExpForce {src}({exp_by_id[src]['object_name']})",
                 notes="力值为转移值" + (",近似待实测复核" if "近似" in kind else ""))
        rows.append(r)
        by_id[oid] = r

    # ---------- 3. 追加 ExpForce 129 物体(真实实测) ----------
    for e in exp_rows:
        oid = e["object_id"]
        if oid in by_id:
            continue
        cat = e["category"]
        if cat == "易碎":
            leaf = "轻脆" if oid in EXP_LIGHT else "重脆"
        else:
            leaf = cat
        r = blank_row()
        r.update(object_id=oid, object_family_id=fam_of(e["object_name"]),
                 object_name=e["object_name"], material_class="", mass_kg="",
                 fragile_flag="1" if cat == "易碎" else "0",
                 deformable_flag="1" if cat == "柔性" else "0",
                 leaf_class=leaf, source="ExpForce(UT Austin)实测",
                 f_min_value=e["f_min_value"], f_min_unit="N",
                 force_total_or_single=e.get("force_total_or_single", ""),
                 measurement_protocol="ExpForce原实测值",
                 gripper_model=e.get("gripper_model", ""),
                 sensor_model=e.get("sensor_model", ""),
                 notes="max_safe=" + str(fmax_of(e)))
        rows.append(r)
        by_id[oid] = r

    with open(OBJ_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # ---------- 4. force_map.csv ----------
    with open(FORCE_MAP, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["object_id", "f_min_label_N", "f_max_label_N", "label_source", "is_measured"])
        for r in rows:
            oid = r["object_id"]
            v = r.get("f_min_value", "")
            if v == "" or v is None:
                continue
            if oid.startswith("E"):
                src = "ExpForce实测"
                vmax = fmax_of(exp_by_id[oid])
                measured = "1"
            else:
                proto = r.get("measurement_protocol", "")
                src = proto.split(":")[0] if proto else "转移"
                vmax = round(2 * float(v), 2)
                measured = "0"
            w.writerow([oid, float(v), vmax, src, measured])

    # ---------- 校验输出 ----------
    import collections
    cnt = collections.Counter(r["leaf_class"] for r in rows)
    print(f"注册表总物体: {len(rows)}")
    print(f"叶类分布: {dict(cnt)}")
    print(f"采购清单力值: 同型转移 {n_same}, 近似转移 {n_approx}")
    has_force = sum(1 for r in rows if r.get("f_min_value"))
    print(f"有力值物体: {has_force}/{len(rows)}")


if __name__ == "__main__":
    main()

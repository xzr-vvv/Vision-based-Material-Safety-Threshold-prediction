# -*- coding: utf-8 -*-
"""准备补图阶段: 新增 6 个轻脆物体(FRA067-072) + 为 56 个选中 E 系物体写检索词."""
import csv
import json
import os

PREP = r"E:\A-触觉机器学习\数据采集准备"
RGB = r"E:\A-触觉机器学习\RGB_dataset"
TEMPLATE = os.path.join(PREP, "objects_模板.csv")
TERMS_FILE = os.path.join(PREP, "wikimedia_search_terms.json")
FORCE_MAP = os.path.join(RGB, "force_map.csv")

GRIPPER = "LEGATO linkage-based two-finger gripper + FORTE fin-ray fingers (UT Austin)"
SENSOR = "FORTE tactile fin-ray fingers (force+slip sensing)"

NEW_FRA = [
    ("FRA067", "wafer_cone", "冰淇淋甜筒(空威化筒)", "威化薄壳", "1", "0", "轻脆",
     "超市(食用后留筒)", "0.5", "近似转移(威化薄筒): 参照蛋卷类脆壳",
     "薄壁威化一捏即碎; 力值为近似转移,待实测复核"),
    ("FRA068", "gingerbread", "硬姜饼(块/人形)", "硬质烘焙", "1", "0", "轻脆",
     "超市/烘焙店", "1.0", "近似转移(硬质姜饼): 参照硬质烘焙脆物",
     "干硬脆裂; 力值为近似转移,待实测复核"),
    ("FRA069", "grissini", "意式脆面包棒(Grissini)", "干烤面棒", "1", "0", "轻脆",
     "超市", "1.0", "近似转移(干烤面棒): 参照挂面/干脆面类",
     "细长脆棒易折断; 力值为近似转移,待实测复核"),
    ("FRA070", "rice_cracker", "仙贝/米饼(单片)", "膨化米饼", "1", "0", "轻脆",
     "超市", "0.5", "近似转移(膨化米饼): 参照锅巴/米通类",
     "薄脆易碎; 力值为近似转移,待实测复核"),
    ("FRA071", "peanut_brittle", "花生脆糖(块)", "脆糖", "1", "0", "轻脆",
     "超市", "0.5", "近似转移(脆糖): 参照硬糖/酥糖类",
     "硬脆糖块易崩裂; 力值为近似转移,待实测复核"),
    ("FRA072", "honeycomb_candy", "蜂巢糖(块)", "蜂窝脆糖", "1", "0", "轻脆",
     "自制/糖果店", "0.25", "近似转移(蜂窝脆糖): 参照酥性糖果",
     "多孔酥脆一压即塌; 力值为近似转移,待实测复核"),
]

FORCE_ROWS = [
    ("FRA067", "0.5", "1.0", "近似转移(威化薄筒)", "0"),
    ("FRA068", "1.0", "2.0", "近似转移(硬质姜饼)", "0"),
    ("FRA069", "1.0", "2.0", "近似转移(干烤面棒)", "0"),
    ("FRA070", "0.5", "1.0", "近似转移(膨化米饼)", "0"),
    ("FRA071", "0.5", "1.0", "近似转移(脆糖)", "0"),
    ("FRA072", "0.25", "0.5", "近似转移(蜂窝脆糖)", "0"),
]

E_TERMS = {
    "E002": ("Pringles", ["pringles can", "pringles chips tube", "pringles potato chips"]),
    "E003": ("Spam (food)", ["spam canned meat", "spam can", "canned meat spam"]),
    "E016": ("Monster Energy", ["monster energy drink can", "monster energy can", "energy drink can"]),
    "E017": ("Red Bull", ["red bull can", "red bull energy drink", "red bull soda can"]),
    "E043": ("7 Up", ["7up soda can", "seven up can", "lemon lime soda can"]),
    "E044": ("Diet Coke", ["diet coke can", "coca cola diet can", "diet coke bottle"]),
    "E050": ("Remote controls", ["tv remote control", "television remote", "remote control handset"]),
    "E052": ("Adhesive tape", ["magic tape roll", "scotch tape roll", "sticky tape roll"]),
    "E053": (None, ["vegetable peeler", "kitchen peeler", "potato peeler"]),
    "E058": (None, ["lint roller", "sticky lint roller", "clothes lint remover"]),
    "E090": (None, ["muffin mix box", "baking mix box", "cake mix box"]),
    "E091": (None, ["cheez it crackers", "cheese crackers box", "crackers box"]),
    "E093": (None, ["yellow mustard bottle", "mustard bottle", "french mustard"]),
    "E095": (None, ["tomato soup can", "canned tomato soup", "condensed tomato soup"]),
    "E097": ("Nutella", ["nutella jar", "hazelnut spread jar", "nutella jar glass"]),
    "E098": (None, ["morton salt", "salt container", "table salt canister"]),
    "E100": (None, ["canned tuna", "tuna can", "starkist tuna"]),
    "E101": ("Duct tape", ["duct tape roll", "duck tape roll", "grey duct tape"]),
    "E107": (None, ["shampoo bottle", "head shoulders shampoo", "hair shampoo bottle"]),
    "E110": ("Pocky", ["pocky box", "pocky snack", "chocolate biscuit sticks box"]),
    "E111": (None, ["granulated sugar", "sugar bag packaging", "white sugar package"]),
    "E112": (None, ["swiss roll snack", "little debbie swiss rolls", "cream roll snack"]),
    "E113": ("Toothpaste", ["toothpaste box", "toothpaste packaging", "toothpaste carton"]),
    "E121": (None, ["plastic mug", "plastic cup mug", "plastic coffee mug"]),
    "E128": ("Umbrellas", ["folded umbrella", "compact umbrella", "foldable umbrella"]),
    "E129": (None, ["cordless drill", "handheld power drill", "power drill tool"]),
    "E009": ("Limes", ["lime fruit", "fresh lime", "whole lime"]),
    "E020": (None, ["roma tomato", "plum tomato", "fresh roma tomatoes"]),
    "E022": ("Grapefruits", ["grapefruit", "whole grapefruit", "fresh grapefruit"]),
    "E023": ("Lemons", ["lemon", "whole lemon", "fresh lemons"]),
    "E024": ("Apples", ["red apple", "whole apple", "fresh apples"]),
    "E025": ("Bell peppers", ["yellow bell pepper", "bell pepper", "sweet pepper"]),
    "E026": ("Strawberries", ["strawberry", "fresh strawberries", "whole strawberry"]),
    "E027": (None, ["cara cara orange", "navel orange", "whole oranges"]),
    "E029": ("Oranges", ["orange fruit", "whole orange", "fresh oranges"]),
    "E030": (None, ["mandarin orange", "mandarins", "clementine fruit"]),
    "E031": ("Limes", ["green lime", "lime citrus", "whole limes"]),
    "E032": ("Grapes", ["red grapes", "grape bunch", "red globe grapes"]),
    "E033": ("Grapes", ["red table grapes", "grapes cluster", "red seedless grapes"]),
    "E036": ("Blackberries", ["blackberry", "fresh blackberries", "whole blackberries"]),
    "E039": ("Oreo", ["oreo cookies pack", "oreo thin", "oreo package"]),
    "E048": ("Marshmallows", ["marshmallow", "white marshmallows", "plain marshmallow"]),
    "E056": ("Bananas", ["banana", "whole banana", "yellow banana fruit"]),
    "E064": ("Pears", ["green pear", "whole pear", "fresh pear"]),
    "E065": ("Lemons", ["large lemon", "big lemon", "lemon isolated"]),
    "E066": (None, ["golden delicious apple", "yellow apple", "golden apple"]),
    "E067": (None, ["red apple", "fresh red apple", "red delicious apple"]),
    "E068": (None, ["green apple", "granny smith apple", "fresh green apple"]),
    "E073": ("Strawberries", ["fresh strawberry", "strawberry fruit", "ripe strawberries"]),
    "E074": ("Strawberries", ["large strawberry", "big strawberry", "strawberry closeup"]),
    "E075": (None, ["mandarin oranges", "clementines", "mandarin fruits"]),
    "E077": ("Grapes", ["green grapes", "white grapes", "green seedless grapes"]),
    "E080": (None, ["grape tomatoes", "cherry tomatoes", "red grape tomato"]),
    "E086": ("Blueberries", ["blueberry", "fresh blueberries", "whole blueberries"]),
    "E087": ("Raspberries", ["raspberry", "fresh raspberries", "whole raspberry"]),
    "E122": ("Marshmallows", ["large marshmallow", "big marshmallows", "jumbo marshmallow"]),
}

FRA_TERMS = {
    "FRA067": ("Ice cream cones", ["ice cream cone", "wafer cone", "waffle ice cream cone"]),
    "FRA068": ("Gingerbread", ["gingerbread", "gingerbread cookies", "gingerbread man"]),
    "FRA069": (None, ["grissini", "italian breadsticks", "bread sticks"]),
    "FRA070": ("Senbei", ["senbei", "rice crackers", "japanese rice crackers"]),
    "FRA071": ("Peanut brittle", ["peanut brittle", "brittle candy", "peanut brittle candy"]),
    "FRA072": ("Honeycomb toffee", ["honeycomb candy", "cinder toffee", "honeycomb toffee"]),
}


def main():
    with open(TEMPLATE, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys())
    existing = {r["object_id"] for r in rows}
    added = 0
    for (oid, fam, name, mat, frag, defo, leaf, src, fmin, proto,
         notes) in NEW_FRA:
        if oid in existing:
            print(f"skip {oid} (already in template)")
            continue
        rows.append({
            "object_id": oid, "object_family_id": fam, "object_name": name,
            "material_class": mat, "mass_kg": "", "dimensions_m": "",
            "fragile_flag": frag, "deformable_flag": defo, "leaf_class": leaf,
            "source": src, "f_min_value": fmin, "f_min_unit": "N",
            "f_min_semantics": "", "force_total_or_single": "total",
            "measurement_protocol": proto, "trial_count": "", "trial_values": "",
            "selected_value": "", "gripper_model": GRIPPER,
            "sensor_model": SENSOR, "calibration_id": "", "notes": notes,
        })
        added += 1
    if added:
        with open(TEMPLATE, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
    print(f"template: +{added} rows, total {len(rows)}")

    with open(FORCE_MAP, encoding="utf-8-sig", newline="") as f:
        frows = list(csv.DictReader(f))
        ffields = list(frows[0].keys())
        fids = {r["object_id"] for r in frows}
    fadded = 0
    for oid, fmin, fmax, src, measured in FORCE_ROWS:
        if oid in fids:
            continue
        frows.append({"object_id": oid, "f_min_label_N": fmin,
                      "f_max_label_N": fmax, "label_source": src,
                      "is_measured": measured})
        fadded += 1
    if fadded:
        with open(FORCE_MAP, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=ffields)
            w.writeheader()
            w.writerows(frows)
    print(f"force_map: +{fadded} rows, total {len(frows)}")

    with open(TERMS_FILE, encoding="utf-8") as f:
        terms = json.load(f)
    tadded = 0
    for oid, (cat, ts) in list(E_TERMS.items()) + list(FRA_TERMS.items()):
        if oid not in terms:
            terms[oid] = {"cat": cat, "terms": ts}
            tadded += 1
        else:
            terms[oid]["cat"] = cat or terms[oid].get("cat")
            terms[oid]["terms"] = ts
            tadded += 1
    with open(TERMS_FILE, "w", encoding="utf-8") as f:
        json.dump(terms, f, ensure_ascii=False, indent=1)
    print(f"terms: {tadded} entries updated, total {len(terms)}")


if __name__ == "__main__":
    main()

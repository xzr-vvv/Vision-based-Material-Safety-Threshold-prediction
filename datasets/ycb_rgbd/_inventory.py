# -*- coding: utf-8 -*-
"""盘点 YCB 12 个测试场景: 每个物体在多少帧中可见 + 每物体可裁剪的包围盒统计"""
import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

TEST = r"E:\A-触觉机器学习\datasets\ycb_rgbd\test"

# BOP ycbv 模型ID -> 名称
NAMES = {
    1: "hmaster_sauce_bottle", 2: "master_chef_can", 3: "cracker_box", 4: "sugar_box",
    5: "tomato_soup_can", 6: "mustard_bottle", 7: "tuna_fish_can", 8: "pudding_box",
    9: "gelatin_box", 10: "potted_meat_can", 11: "banana", 12: "strawberry",
    13: "peach", 14: "pear", 15: "citrus_orange", 16: "apple", 17: "plum",
    18: "clamp", 19: "power_drill", 20: "wood_block", 21: "scissors",
}
# 视觉大类映射(叶类按材质语义)
LEAF = {
    "hmaster_sauce_bottle": ("刚体", "sauce_bottle"), "master_chef_can": ("刚体", "food_can"),
    "cracker_box": ("刚体", "cracker_box"), "sugar_box": ("刚体", "sugar_box"),
    "tomato_soup_can": ("刚体", "food_can"), "mustard_bottle": ("刚体", "mustard_bottle"),
    "tuna_fish_can": ("刚体", "food_can"), "pudding_box": ("刚体", "pudding_box"),
    "gelatin_box": ("刚体", "gelatin_box"), "potted_meat_can": ("刚体", "food_can"),
    "banana": ("柔性", "banana"), "strawberry": ("刚体", "strawberry"),
    "peach": ("刚体", "peach"), "pear": ("刚体", "pear"),
    "citrus_orange": ("刚体", "orange"), "apple": ("刚体", "apple"),
    "plum": ("刚体", "plum"), "clamp": ("刚体", "clamp"),
    "power_drill": ("刚体", "power_drill"), "wood_block": ("刚体", "wood_block"),
    "scissors": ("刚体", "scissors"),
}

stats = {}  # obj_id -> {frames: n, scenes: set, min_w, min_h}
for scene in sorted(os.listdir(TEST)):
    sdir = os.path.join(TEST, scene)
    gt_info = os.path.join(sdir, "scene_gt_info.json")
    if not os.path.exists(gt_info):
        continue
    with open(gt_info, "r", encoding="utf-8") as f:
        info = json.load(f)
    with open(os.path.join(sdir, "scene_gt.json"), "r", encoding="utf-8") as f:
        gt = json.load(f)
    for frame, anns in info.items():
        gt_anns = gt.get(frame, [])
        for i, a in enumerate(anns):
            if i >= len(gt_anns):
                break
            oid = gt_anns[i]["obj_id"]
            bb = a.get("bbox_visib")  # [x,y,w,h]
            if not bb or bb[2] < 40 or bb[3] < 40:
                continue
            st = stats.setdefault(oid, {"frames": 0, "scenes": set(), "min_side": 9999})
            st["frames"] += 1
            st["scenes"].add(scene)
            st["min_side"] = min(st["min_side"], min(bb[2], bb[3]))

print(f"{'ID':>3} {'名称':<22} {'叶类':<4} {'可见帧数':>6} {'场景数':>4} {'最小边px':>7}")
for oid in sorted(stats):
    st = stats[oid]
    name = NAMES.get(oid, "?")
    leaf = LEAF.get(name, ("?", "?"))[0]
    print(f"{oid:>3} {name:<22} {leaf:<4} {st['frames']:>7} {len(st['scenes']):>5} {st['min_side']:>8}")
print(f"\n合计可用物体: {len(stats)}")

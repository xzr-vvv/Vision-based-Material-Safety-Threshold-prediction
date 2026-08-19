# -*- coding: utf-8 -*-
"""统计 ycbv 测试集 12 个场景里出现的 YCB 物体类别"""
import json
import os
from collections import Counter

root = r"E:\A-触觉机器学习\datasets\ycb_rgbd\test"

# BOP ycbv 官方 obj_id -> YCB 物体名
NAMES = {
    1: "002_master_chef_can 咖啡罐", 2: "003_cracker_box 饼干盒", 3: "004_sugar_box 糖盒",
    4: "005_tomato_soup_can 汤罐", 5: "006_mustard_bottle 芥末瓶", 6: "007_tuna_fish_can 金枪鱼罐",
    7: "008_pudding_box 布丁盒", 8: "009_gelatin_box 果冻盒", 9: "010_potted_meat_can 肉罐",
    10: "011_banana 香蕉", 11: "019_pitcher_base 水壶", 12: "021_bleach_cleanser 漂白剂瓶",
    13: "022_windex_bottle 清洁剂瓶", 14: "024_bowl 碗", 15: "025_mug 马克杯",
    16: "035_power_drill 电钻", 17: "036_wood_block 木块", 18: "037_scissors 剪刀",
    19: "040_large_marker 马克笔", 20: "051_large_clamp 大夹钳", 21: "052_extra_large_clamp 特大夹钳",
}

objs = Counter()
for sc in sorted(os.listdir(root)):
    gt = os.path.join(root, sc, "scene_gt.json")
    if not os.path.exists(gt):
        continue
    d = json.load(open(gt, encoding="utf-8"))
    for fr, anns in d.items():
        for a in anns:
            objs[a["obj_id"]] += 1

print(f"标注总实例数: {sum(objs.values())}, 物体类别数: {len(objs)}\n")
for oid in sorted(objs.keys()):
    print(f"  {oid:2d}  {NAMES.get(oid, 'ID-%d' % oid):38s} {objs[oid]} 次")

# -*- coding: utf-8 -*-
"""为缺图物体补充/修正 Wikimedia 检索词:
- E052/E112/E039 原检索词太窄, 候选池不足
- YCB18/YCB21 被多样性修复清空, 10 个缺图 YCB 物体无词条
"""
import json

TERMS_FILE = r"E:\A-触觉机器学习\数据采集准备\wikimedia_search_terms.json"

UPDATES = {
    "E052": {"cat": "Scotch Tape",
             "terms": ["scotch tape", "sellotape", "cellophane tape", "sticky tape"]},
    "E112": {"cat": None,
             "terms": ["swiss roll", "swiss roll cake", "jelly roll",
                       "chocolate cream roll"]},
    "E039": {"cat": None,
             "terms": ["oreo", "oreo cookie", "oreo pack",
                       "chocolate sandwich cookie"]},
    "YCB07": {"cat": None, "terms": ["canned tuna", "tuna can", "tuna fish can"]},
    "YCB08": {"cat": None,
              "terms": ["pudding box", "chocolate pudding snack", "pudding mix box"]},
    "YCB10": {"cat": None, "terms": ["potted meat can", "canned meat", "spam can"]},
    "YCB13": {"cat": None, "terms": ["peach fruit", "fresh peach", "peaches"]},
    "YCB14": {"cat": "Pears", "terms": ["pear fruit", "fresh pear", "pears"]},
    "YCB16": {"cat": None, "terms": ["red apple", "apple fruit", "fresh apple"]},
    "YCB17": {"cat": None, "terms": ["plums", "plum fruit", "fresh plum"]},
    "YCB18": {"cat": "Clamps",
              "terms": ["metal clamp", "spring clamp", "bar clamp", "clamp tool"]},
    "YCB19": {"cat": None, "terms": ["power drill", "cordless drill", "electric drill"]},
    "YCB20": {"cat": None, "terms": ["wood block", "wooden block", "wood cube"]},
    "YCB21": {"cat": "Scissors",
              "terms": ["scissors", "kitchen scissors", "sewing scissors"]},
}

with open(TERMS_FILE, encoding="utf-8") as f:
    data = json.load(f)
for k, v in UPDATES.items():
    data[k] = v
with open(TERMS_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=1)
print(f"updated {len(UPDATES)} entries")

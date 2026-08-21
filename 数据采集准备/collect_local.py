# -*- coding: utf-8 -*-
"""1) 复制 ExpForce 129 张实测图到 RGB_dataset\{叶类}\{E编号}\
2) 从 YCB-Video 900 帧裁剪 21 物体 x 15 张多视角图到 RGB_dataset\{叶类}\{YCBxx}\
均写 provenance.csv"""
import csv
import io
import json
import os
import shutil
import sys

from PIL import Image

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

OBJ_CSV = r"E:\A-触觉机器学习\数据采集准备\objects_模板.csv"
EXP_CSV = r"E:\A-触觉机器学习\ExpForce数据集\07_ExpForce_安全抓力范围.csv"
EXP_IMG = r"E:\A-触觉机器学习\ExpForce数据集\ExpForce_images"
YCB_TEST = r"E:\A-触觉机器学习\datasets\ycb_rgbd\test"
ROOT = r"E:\A-触觉机器学习\RGB_dataset"
PROV = os.path.join(ROOT, "provenance.csv")
NEED = 15


def load_registry():
    with open(OBJ_CSV, newline="", encoding="utf-8-sig") as f:
        return {r["object_id"]: r for r in csv.DictReader(f)}


def prov_write(rows):
    exists = os.path.exists(PROV)
    with open(PROV, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["object_id", "image_file", "source", "source_url", "license"])
        w.writerows(rows)


def copy_expforce(reg):
    with open(EXP_CSV, newline="", encoding="utf-8-sig") as f:
        exp_rows = list(csv.DictReader(f))
    n = 0
    prov = []
    for e in exp_rows:
        oid = e["object_id"]
        leaf = reg[oid]["leaf_class"]
        img = e.get("image_file", "").strip()
        src = os.path.join(EXP_IMG, img)
        if not img or not os.path.exists(src):
            print(f"  [缺图] {oid} {img}")
            continue
        dst_dir = os.path.join(ROOT, leaf, oid)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, img)
        shutil.copy2(src, dst)
        prov.append([oid, os.path.relpath(dst, ROOT), "ExpForce(UT Austin, arXiv:2603.08668)",
                     "", "ExpForce数据集许可"])
        n += 1
    prov_write(prov)
    print(f"ExpForce 复制完成: {n} 张")


def collect_ycb(reg):
    """每物体从全部可见帧中均匀取 15 帧, mask_visib 裁剪(扩 15% 边距)"""
    NAME = {1: "hmaster_sauce_bottle", 2: "master_chef_can", 3: "cracker_box", 4: "sugar_box",
            5: "tomato_soup_can", 6: "mustard_bottle", 7: "tuna_fish_can", 8: "pudding_box",
            9: "gelatin_box", 10: "potted_meat_can", 11: "banana", 12: "strawberry",
            13: "peach", 14: "pear", 15: "citrus_orange", 16: "apple", 17: "plum",
            18: "clamp", 19: "power_drill", 20: "wood_block", 21: "scissors"}
    cands = {}  # oid -> [(scene, frame, bbox)]
    for scene in sorted(os.listdir(YCB_TEST)):
        sdir = os.path.join(YCB_TEST, scene)
        gi = os.path.join(sdir, "scene_gt_info.json")
        if not os.path.exists(gi):
            continue
        with open(gi, encoding="utf-8") as f:
            info = json.load(f)
        with open(os.path.join(sdir, "scene_gt.json"), encoding="utf-8") as f:
            gt = json.load(f)
        for frame in sorted(info.keys(), key=int):
            gt_anns = gt.get(frame, [])
            for i, a in enumerate(info[frame]):
                if i >= len(gt_anns):
                    break
                oid = gt_anns[i]["obj_id"]
                bb = a.get("bbox_visib")
                if not bb or min(bb[2], bb[3]) < 60:
                    continue
                cands.setdefault(oid, []).append((scene, int(frame), bb))

    prov = []
    for oid, items in sorted(cands.items()):
        name = NAME[oid]
        reg_id = f"YCB{oid:02d}"
        leaf = reg[reg_id]["leaf_class"]
        dst_dir = os.path.join(ROOT, leaf, reg_id)
        os.makedirs(dst_dir, exist_ok=True)
        # 均匀取 15 帧(覆盖不同场景/视角)
        step = max(1, len(items) // NEED)
        picked = items[::step][:NEED]
        for k, (scene, frame, bb) in enumerate(picked):
            img_path = os.path.join(YCB_TEST, scene, "rgb", f"{frame:06d}.png")
            im = Image.open(img_path).convert("RGB")
            W, H = im.size
            x, y, w, h = bb
            mx, my = int(w * 0.15), int(h * 0.15)
            box = (max(0, x - mx), max(0, y - my),
                   min(W, x + w + mx), min(H, y + h + my))
            crop = im.crop(box)
            out = os.path.join(dst_dir, f"s{k}.png")
            crop.save(out)
            prov.append([reg_id, os.path.relpath(out, ROOT),
                         f"YCB-Video(BOP ycbv test) scene {scene} frame {frame:06d} 裁剪",
                         "", "CC BY 4.0-ish(BOP/YCB许可)"])
        print(f"  {reg_id} {name}: {len(picked)} 张 -> {leaf}")
    prov_write(prov)


if __name__ == "__main__":
    reg = load_registry()
    print("== 复制 ExpForce ==")
    copy_expforce(reg)
    print("== 裁剪 YCB ==")
    collect_ycb(reg)

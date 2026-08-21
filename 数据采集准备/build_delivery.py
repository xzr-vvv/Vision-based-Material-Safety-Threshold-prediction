# -*- coding: utf-8 -*-
r"""组装 SafetyVTLA 第一批交付包 (A级 visual_prior)
按 SAFETYVTLA_DATA_REQUIREMENTS_V1.md §1.1 目录/字段, §4 第一批, §5 结构, §6 自检。

输出: E:\A-触觉机器学习\safety_vtla_data_delivery_20260821\
  visual_prior\images\<object_id>\<原文件名>.png   (1847 张原始 RGB)
  visual_prior\objects.csv    (181 行, §1.1 必填字段 + 力值标签全字段)
  visual_prior\captures.csv   (1847 行, 缺失采集元数据留空不编造)
  visual_prior\splits.json    (按 object_family_id 隔离切分)
  README.md                   (来源/许可证/单位/缺失字段/弱先验声明/自检)
  checksums.sha256
"""
import csv
import hashlib
import json
import os
import shutil
import struct

ROOT = r"E:\A-触觉机器学习"
PKG = os.path.join(ROOT, "safety_vtla_data_delivery_20260821")
VP = os.path.join(PKG, "visual_prior")
IMG = os.path.join(VP, "images")

EXPF_SRC = "ExpForce(UT Austin, arXiv:2603.08668)"
GRIPPER = "LEGATO linkage-based two-finger gripper + FORTE fin-ray fingers (UT Austin)"
SENSOR = "FORTE tactile fin-ray fingers (force+slip sensing)"
CALIB = "ExpForce_official_v1"


def load(p):
    with open(p, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def png_size(path):
    with open(path, "rb") as f:
        head = f.read(24)
    w, h = struct.unpack(">II", head[16:24])
    return w, h


def main():
    labels = load(os.path.join(ROOT, "RGB_dataset", "labels.csv"))
    prov = {(r["object_id"], os.path.basename(r["image_file"])): r
            for r in load(os.path.join(ROOT, "RGB_dataset", "provenance.csv"))}
    tpl = {r["object_id"]: r for r in load(os.path.join(ROOT, "数据采集准备", "objects_模板.csv"))}
    thr = {r["object_id"]: r for r in load(os.path.join(ROOT, "数据采集准备", "objects_已有安全阈值.csv"))}
    e07 = {r["object_id"]: r for r in load(os.path.join(ROOT, "ExpForce数据集", "07_ExpForce_安全抓力范围.csv"))}
    fmap = {r["object_id"]: r for r in load(os.path.join(ROOT, "RGB_dataset", "force_map.csv"))}

    objs = {}
    for r in labels:
        oid = r["object_id"]
        d = objs.setdefault(oid, {
            "family": r["object_family_id"], "category": r["category"],
            "leaf": r["leaf_class"], "l1": r["l1_gate"],
            "fmin": r["f_min_measured_N"], "fmax": r["f_max_measured_N"],
            "note": r["note"], "n": 0})
        d["n"] += 1

    # ---- 切分: 按统一族 md5 确定性哈希 70/15/15, 保证每个 split 覆盖全部 L2 叶类 ----
    fam2objs = {}
    for oid, d in objs.items():
        fam2objs.setdefault(d["family"], []).append(oid)
    fam2leaf = {f: objs[v[0]]["leaf"] for f, v in fam2objs.items()}

    def fam_bucket(f):
        h = int(hashlib.md5(f.encode("utf-8")).hexdigest()[:8], 16) % 100
        return "train" if h < 70 else ("calibration" if h < 85 else "test")

    fam2split = {f: fam_bucket(f) for f in fam2objs}
    moved = []
    for sp in ("train", "calibration", "test"):
        for leaf in ("轻脆", "重脆", "刚体", "柔性"):
            if not any(fam2leaf[f] == leaf for f, s in fam2split.items() if s == sp):
                donors = [f for f, s in fam2split.items() if s != sp and fam2leaf[f] == leaf]
                donor = min(donors, key=lambda f: len(fam2objs[f]))
                fam2split[donor] = sp
                moved.append((donor, leaf, sp))
    obj2split = {oid: fam2split[d["family"]] for oid, d in objs.items()}

    # ---- objects.csv ----
    ofields = [
        "object_id", "object_family_id", "object_name", "material_class",
        "mass_kg", "dimensions_m", "fragile_flag", "deformable_flag", "source", "notes",
        "f_min_value", "f_min_unit", "f_min_semantics", "force_total_or_single",
        "f_min_single_finger_N", "f_min_source", "measurement_protocol",
        "trial_count", "trial_values", "selected_value",
        "gripper_model", "sensor_model", "calibration_id",
        "f_max_measured_N", "f_max_semantics", "f_max_source", "force_note",
        "f_min_evidence", "f_max_evidence",
        "expforce_object_name", "category", "leaf_class", "l1_gate", "n_images",
        "mass_source", "dimensions_basis", "volume_m3", "density_kg_m3",
        "phys_est_basis", "ref_expforce_id", "ref_object_name",
    ]
    orows = []
    flag_mismatch = []
    for oid in sorted(objs):
        d = objs[oid]
        t = tpl.get(oid, {})
        e = e07.get(oid)
        th = thr.get(oid)
        fm = fmap.get(oid, {})
        row = {k: "" for k in ofields}
        row.update({
            "object_id": oid, "object_family_id": d["family"],
            "object_name": t.get("object_name", ""), "material_class": t.get("material_class", ""),
            "fragile_flag": t.get("fragile_flag", ""), "deformable_flag": t.get("deformable_flag", ""),
            "source": EXPF_SRC if e else t.get("source", ""),
            "notes": t.get("notes", ""),
            "category": d["category"], "leaf_class": d["leaf"], "l1_gate": d["l1"],
            "n_images": d["n"], "force_note": d["note"],
        })
        if e:
            # flags 一致性核对 (本地 vs ExpForce 官方)
            off = {"易碎": ("1", "0"), "柔性": ("0", "1"), "刚体": ("0", "0")}[e["category"]]
            if (t.get("fragile_flag"), t.get("deformable_flag")) != off:
                flag_mismatch.append((oid, e["category"]))
            row.update({
                "mass_kg": f"{float(e['mass_g']) / 1000.0:.4f}",
                "mass_source": "ExpForce本物实测",
                "expforce_object_name": e["object_name"],
                "f_min_value": e["f_min_value"], "f_min_unit": e["f_min_unit"],
                "f_min_semantics": e["f_min_semantics"],
                "force_total_or_single": e["force_total_or_single"],
                "f_min_single_finger_N": e["f_min_single_finger_N"],
                "f_min_source": "ExpForce本物实测",
                "measurement_protocol": e["measurement_protocol"],
                "trial_count": e["trial_count"], "trial_values": e["trial_values"],
                "selected_value": e["selected_value"],
                "gripper_model": e["gripper_model"], "sensor_model": e["sensor_model"],
                "calibration_id": e["calibration_id"],
            })
        else:
            if th and th.get("mass_kg"):
                row["mass_kg"] = th["mass_kg"]
                row["mass_source"] = "ExpForce同型参照均值(估算)"
            if th:
                row.update({
                    "dimensions_m": th.get("dimensions_m", ""),
                    "dimensions_basis": "典型值估算(±30%)" if th.get("dimensions_m") else "",
                    "volume_m3": th.get("volume_m3", ""),
                    "density_kg_m3": th.get("density_kg_m3", ""),
                    "phys_est_basis": th.get("phys_est_basis", ""),
                    "ref_expforce_id": th.get("ref_expforce_id", ""),
                    "ref_object_name": th.get("ref_object_name", ""),
                })
            if d["fmin"]:
                row.update({
                    "f_min_value": d["fmin"], "f_min_unit": "N",
                    "f_min_semantics": fm.get("f_min_semantics", "两指法向力之和"),
                    "force_total_or_single": "total",
                    "f_min_single_finger_N": fm.get("f_min_single_N", ""),
                    "f_min_source": "同型转移(非本物实测): " + fm.get("label_source", ""),
                    "measurement_protocol": ("同型转移: " + fm.get("label_source", "")
                                             + " | 参照协议=ExpForce自适应力控(0.25N起步,滑落收紧,"
                                               "每物体3次试验取中位数,举升5cm@0.5cm/s验证不滑)"),
                    "trial_count": "3(参照)", "trial_values": "参照官方未发布逐次原始值(仅中位数)",
                    "selected_value": "median(参照转移)",
                    "gripper_model": GRIPPER, "sensor_model": SENSOR, "calibration_id": CALIB,
                })
            else:
                row["f_min_source"] = "无实测值(仅分类用途)"
        if e and th and th.get("dimensions_m"):
            row.update({
                "dimensions_m": th["dimensions_m"],
                "dimensions_basis": "典型值估算(±30%)",
                "volume_m3": th.get("volume_m3", ""),
                "density_kg_m3": th.get("density_kg_m3", ""),
                "phys_est_basis": th.get("phys_est_basis", ""),
            })
        if not row["mass_source"]:
            row["mass_source"] = "缺失"
        if d["fmax"]:
            fmax_sem = fm.get("f_max_semantics", "")
            if fm.get("f_max_total_N"):
                fmax_sem += f";换算两指合力={fm['f_max_total_N']}N"
            row.update({
                "f_max_measured_N": d["fmax"],
                "f_max_semantics": fmax_sem,
                "f_max_source": fm.get("f_max_basis", ""),
            })
        row["f_min_evidence"] = fm.get("evidence", "") if d["fmin"] else ""
        row["f_max_evidence"] = fm.get("evidence", "") if d["fmax"] else ""
        orows.append(row)

    # ---- captures.csv + 图片复制 ----
    cfields = ["capture_id", "object_id", "object_family_id", "session_id", "camera_id",
               "image_path", "width_px", "height_px", "timestamp_ns",
               "distance_m", "view_yaw_deg", "view_pitch_deg", "lighting_id",
               "background_id", "target_visible", "occluded", "split",
               "source", "source_url", "license"]
    crows = []
    for r in labels:
        oid = r["object_id"]
        fn = os.path.basename(r["image_file"])
        src = os.path.join(ROOT, "RGB_dataset", r["image_file"])
        dstdir = os.path.join(IMG, oid)
        os.makedirs(dstdir, exist_ok=True)
        shutil.copyfile(src, os.path.join(dstdir, fn))
        w, h = png_size(src)
        p = prov.get((oid, fn), {})
        crows.append({
            "capture_id": f"{oid}_{os.path.splitext(fn)[0]}", "object_id": oid,
            "object_family_id": r["object_family_id"], "session_id": "", "camera_id": "",
            "image_path": f"visual_prior/images/{oid}/{fn}",
            "width_px": w, "height_px": h, "timestamp_ns": "",
            "distance_m": "", "view_yaw_deg": "", "view_pitch_deg": "", "lighting_id": "",
            "background_id": "", "target_visible": "", "occluded": "",
            "split": obj2split[oid],
            "source": p.get("source", ""), "source_url": p.get("source_url", ""),
            "license": p.get("license", ""),
        })

    # ---- 写 CSV / JSON ----
    os.makedirs(VP, exist_ok=True)
    with open(os.path.join(VP, "objects.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w_ = csv.DictWriter(f, fieldnames=ofields)
        w_.writeheader()
        w_.writerows(orows)
    with open(os.path.join(VP, "captures.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w_ = csv.DictWriter(f, fieldnames=cfields)
        w_.writeheader()
        w_.writerows(crows)

    sp_fams = {sp: sorted(f for f, s in fam2split.items() if s == sp)
               for sp in ("train", "calibration", "test")}
    sp_objs = {sp: sum(1 for o, s in obj2split.items() if s == sp) for sp in sp_fams}
    sp_imgs = {sp: sum(1 for c in crows if c["split"] == sp) for sp in sp_fams}
    splits_doc = {
        "method": ("按 object_family_id 隔离的确定性切分(md5(family) 70/15/15), "
                   "并强制每个 split 覆盖全部 4 个 L2 叶类(轻脆/重脆/刚体/柔性); "
                   "无采集 session 元数据, 故仅按族隔离"),
        "rule": "同一物体/同一族的所有图片必然进入同一 split; 不按图片随机切分",
        "coverage_fixes": [{"family": f, "leaf": leaf, "moved_to": sp} for f, leaf, sp in moved],
        "splits": sp_fams,
        "object_counts": sp_objs,
        "image_counts": sp_imgs,
        "leaf_class_coverage": {sp: sorted({fam2leaf[f] for f in fams})
                                for sp, fams in sp_fams.items()},
        "notes": "训练/校准/测试互不共享任何 object_family_id; 单图物体(62个)已包含",
    }
    with open(os.path.join(VP, "splits.json"), "w", encoding="utf-8") as f:
        json.dump(splits_doc, f, ensure_ascii=False, indent=2)

    # ---- 统计 ----
    n_e = sum(1 for o in objs if o.startswith("E"))
    n_fmin = sum(1 for d in objs.values() if d["fmin"])
    n_fmax = sum(1 for d in objs.values() if d["fmax"])
    n_mass = sum(1 for r in orows if r["mass_kg"])
    n_dim = sum(1 for r in orows if r["dimensions_m"])
    n_single = sum(1 for d in objs.values() if d["n"] == 1)
    stats = dict(objs=len(objs), imgs=len(crows), fams=len(fam2objs), n_e=n_e,
                 n_fmin=n_fmin, n_fmax=n_fmax, n_mass=n_mass, n_dim=n_dim,
                 n_single=n_single, sp_objs=sp_objs, sp_imgs=sp_imgs,
                 flag_mismatch=flag_mismatch)
    print(json.dumps({k: v for k, v in stats.items() if k != "flag_mismatch"},
                     ensure_ascii=False, indent=1))
    print("flags与ExpForce官方不一致(应为0):", flag_mismatch)
    return stats


if __name__ == "__main__":
    main()

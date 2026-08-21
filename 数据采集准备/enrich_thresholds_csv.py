# -*- coding: utf-8 -*-
"""按 SafetyVTLA_缺口与可用资源整理_20260821.md §4.1/§5 (用户已确认口径) 补齐
objects_已有安全阈值.csv:
1. mass_kg: ExpForce 同型参照实测质量均值 (refs -> Object -> Mass)
2. dimensions_m: 常见典型尺寸 (±30% 浮动, 逐物体)
3. volume_m3: 按形状公式 (cyl=pi/4*d^2*h, ell=pi/6*L*W*H, box=f*L*W*H, bag=0.45*L*W*H)
4. density_kg_m3 = mass_kg / volume_m3 (有效密度, 含空心)
5. F_max 评测参考列 (全部不写入 SafetyVTLA 真值表, 真值等 Gen3 同构实测):
   优先级: 本物文献换算 > 同型参照文献换算 > 同族文献换算(取最小,保守) > 类别系数占位
   类别系数: 易碎x2 / 柔性x3 / 刚体x5 (mlproject 原口径, 仅评 gap 上界)
6. 输出 family_crosswalk.csv: csv 物体 -> 参照 -> 参照所在统一族 (跨文件泄漏对照)
"""
import csv
import math
import os
import re
import shutil

ROOT = r"E:\A-触觉机器学习"
THRESH = os.path.join(ROOT, "数据采集准备", "objects_已有安全阈值.csv")
EXPFORCE = os.path.join(ROOT, "ExpForce数据集", "ExpForce_dataset_官方原始.csv")
PROV = os.path.join(ROOT, "RGB_dataset", "provenance.csv")
LABELS = os.path.join(ROOT, "RGB_dataset", "labels.csv")
CROSSWALK = os.path.join(ROOT, "数据采集准备", "family_crosswalk.csv")

# object_id -> (L, W, H, shape, fill) 尺寸为常见典型值, 口径 ±30%
DIMS = {
    "FRA001": (0.080, 0.080, 0.110, "cyl", 1.0),
    "FRA002": (0.075, 0.075, 0.110, "cyl", 1.0),
    "FRA003": (0.070, 0.070, 0.210, "cyl", 1.0),
    "FRA005": (0.065, 0.065, 0.240, "cyl", 1.0),
    "FRA006": (0.080, 0.080, 0.120, "cyl", 1.0),
    "FRA010": (0.095, 0.095, 0.100, "cyl", 1.0),
    "FRA016": (0.065, 0.065, 0.090, "cyl", 1.0),
    "FRA019": (0.200, 0.150, 0.060, "bag", 0.45),
    "FRA024": (0.060, 0.045, 0.045, "ell", 1.0),
    "FRA025": (0.055, 0.042, 0.042, "ell", 1.0),
    "FRA032": (0.070, 0.070, 0.130, "cyl", 1.0),
    "FRA039": (0.070, 0.070, 0.100, "cyl", 1.0),
    "FRA048": (0.140, 0.065, 0.210, "box", 1.0),
    "FRA051": (0.060, 0.045, 0.045, "ell", 1.0),
    "SOF001": (0.200, 0.045, 0.045, "ell", 1.0),
    "SOF002": (0.080, 0.080, 0.080, "ell", 1.0),
    "SOF003": (0.070, 0.060, 0.060, "ell", 1.0),
    "SOF004": (0.040, 0.035, 0.035, "ell", 1.0),
    "SOF005": (0.150, 0.100, 0.060, "bag", 0.45),
    "SOF008": (0.080, 0.055, 0.055, "ell", 1.0),
    "SOF009": (0.035, 0.030, 0.030, "ell", 1.0),
    "SOF013": (0.090, 0.070, 0.070, "ell", 1.0),
    "SOF030": (0.032, 0.032, 0.035, "cyl", 1.0),
    "SOF031": (0.090, 0.075, 0.075, "ell", 1.0),
    "SOF053": (0.150, 0.100, 0.050, "bag", 0.45),
    "RIG001": (0.066, 0.066, 0.125, "cyl", 1.0),
    "RIG004": (0.070, 0.050, 0.180, "box", 0.9),
    "RIG005": (0.075, 0.075, 0.100, "cyl", 1.0),
    "RIG006": (0.075, 0.075, 0.110, "cyl", 1.0),
    "RIG008": (0.075, 0.075, 0.240, "cyl", 1.0),
    "RIG013": (0.110, 0.110, 0.025, "cyl", 1.0),
    "RIG016": (0.100, 0.065, 0.025, "box", 1.0),
    "RIG017": (0.160, 0.050, 0.020, "box", 1.0),
    "RIG024": (0.070, 0.070, 0.220, "cyl", 1.0),
    "RIG025": (0.090, 0.060, 0.250, "box", 0.9),
    "RIG049": (0.060, 0.060, 0.300, "cyl", 1.0),
    "RIG054": (0.070, 0.050, 0.160, "box", 0.9),
    "RIG055": (0.110, 0.080, 0.130, "box", 1.0),
    "RIG057": (0.200, 0.070, 0.280, "box", 1.0),
    "RIG058": (0.070, 0.070, 0.100, "cyl", 1.0),
    "RIG059": (0.100, 0.060, 0.140, "box", 1.0),
}

COEF = {"易碎": 2.0, "柔性": 3.0, "刚体": 5.0}

# 参照与实物状态不匹配, 追加到 phys_est_basis (质量口径警示)
STATE_NOTES = {
    "FRA019": "参照E040为单片薯片(2g),袋装整体质量待实测",
    "SOF005": "参照为单粒葡萄(10g),整串质量待实测",
    "RIG024": "参照E047为空瓶(130g),满壶质量待实测",
    "SOF013": "参照E025为迷你椒(45g),常规青椒质量待实测",
}


def volume(l, w, h, shape, fill):
    if shape == "cyl":
        return math.pi / 4.0 * w * w * h
    if shape == "ell":
        return math.pi / 6.0 * l * w * h
    return fill * l * w * h


def parse_refs(s):
    out = []
    for tok in re.split(r"[/]", s.strip()):
        tok = tok.strip()
        if not tok:
            continue
        m = re.match(r"^(E\d+)-(E\d+)$", tok)
        if m:
            a, b = int(m.group(1)[1:]), int(m.group(2)[1:])
            out += [f"E{i:03d}" for i in range(a, b + 1)]
        else:
            out.append(tok)
    return out


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def main():
    with open(EXPFORCE, encoding="utf-8-sig") as f:
        name2mass = {r["Object"]: float(r["Mass"]) for r in csv.DictReader(f)
                     if r["Mass"]}
    norm2name = {norm(k): k for k in name2mass}

    eid2name = {}
    unmatched = []
    with open(PROV, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["object_id"].startswith("E") and r["source"].startswith("ExpForce"):
                stem = os.path.splitext(os.path.basename(r["image_file"]))[0]
                ns = norm(stem)
                if ns in norm2name:
                    eid2name[r["object_id"]] = norm2name[ns]
                    continue
                hit = next((k for k in norm2name
                            if ns.startswith(k) or k.startswith(ns)), None)
                if hit is None:
                    import difflib
                    close = difflib.get_close_matches(ns, list(norm2name), n=1, cutoff=0.8)
                    hit = close[0] if close else None
                if hit:
                    eid2name[r["object_id"]] = norm2name[hit]
                else:
                    unmatched.append((r["object_id"], stem))
    if unmatched:
        print("未能匹配 ExpForce 名称: " + ", ".join(f"{a}({b})" for a, b in unmatched))

    obj2fam, obj2fmax = {}, {}
    with open(LABELS, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            obj2fam[r["object_id"]] = r["object_family_id"]
            if r["f_max_measured_N"]:
                obj2fmax[r["object_id"]] = float(r["f_max_measured_N"])
    fam2objs = {}
    for oid, fam in obj2fam.items():
        fam2objs.setdefault(fam, set()).add(oid)

    if not os.path.exists(THRESH + ".bak_before_enrich"):
        shutil.copyfile(THRESH, THRESH + ".bak_before_enrich")
    with open(THRESH, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames)
        rows = list(reader)
    for col in ["volume_m3", "density_kg_m3", "ref_f_max_eval_total_N",
                "ref_f_max_eval_single_N", "ref_f_max_eval_basis", "phys_est_basis"]:
        if col not in fields:
            fields.append(col)

    cw_rows = []
    for r in rows:
        oid = r["object_id"]
        refs = parse_refs(r["ref_expforce_id"]) if r["ref_expforce_id"] else []

        masses = [name2mass[eid2name[x]] for x in refs
                  if x in eid2name and eid2name.get(x) in name2mass]
        if masses:
            r["mass_kg"] = f"{sum(masses) / len(masses) / 1000.0:.4f}"
            mass_note = f"ExpForce同型实测均值({len(masses)}个参照)"
        else:
            mass_note = "无参照质量"

        if oid in DIMS:
            l, w, h, shape, fill = DIMS[oid]
            r["dimensions_m"] = f"{l:.3f}x{w:.3f}x{h:.3f}"
            v = volume(l, w, h, shape, fill)
            r["volume_m3"] = f"{v:.6f}"
            if masses and v > 0:
                dens = (sum(masses) / len(masses) / 1000.0) / v
                r["density_kg_m3"] = f"{dens:.1f}"
            r["phys_est_basis"] = (
                f"尺寸=典型值±30%;体积={shape}公式;质量={mass_note};"
                f"密度=有效密度(空心按外接体积)"
                + (f";{STATE_NOTES[oid]}" if oid in STATE_NOTES else ""))
        elif r["mass_kg"]:
            r["phys_est_basis"] = f"质量={mass_note};尺寸/体积待补"

        fmax, basis = None, None
        if oid in obj2fmax:
            fmax, basis = obj2fmax[oid], f"文献实测换算(本物,{obj2fmax[oid]}N两指合力)"
        else:
            for x in refs:
                if x in obj2fmax:
                    fmax, basis = obj2fmax[x], f"文献实测换算(同型参照{x},{obj2fmax[x]}N两指合力)"
                    break
        if fmax is None:
            fams = {obj2fam[x] for x in refs if x in obj2fam}
            lits = [(obj2fmax[m], m) for fam in fams
                    for m in fam2objs.get(fam, ()) if m in obj2fmax]
            if lits:
                v, m = min(lits)
                fmax, basis = v, f"文献实测换算(同族{obj2fam[m]}族{m},取最小{v}N保守)"
        if fmax is None:
            cat = r["ref_category"]
            coef = COEF.get(cat, 3.0)
            fmin = float(r["ref_f_min_total_N"]) if r["ref_f_min_total_N"] else 0.0
            fmax = round(coef * fmin, 2)
            basis = f"类别系数占位({cat}x{coef:g}),仅评gap上界,非实测"
        r["ref_f_max_eval_total_N"] = f"{fmax:g}"
        r["ref_f_max_eval_single_N"] = f"{fmax / 2.0:g}"
        r["ref_f_max_eval_basis"] = basis

        ref_fams = sorted({obj2fam[x] for x in refs if x in obj2fam})
        cw_rows.append({
            "object_id": oid, "csv_family": r["object_family_id"],
            "refs": "/".join(refs), "ref_unified_families": "/".join(ref_fams),
            "f_max_eval_basis": basis,
        })

    with open(THRESH, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with open(CROSSWALK, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["object_id", "csv_family", "refs",
                                               "ref_unified_families", "f_max_eval_basis"])
        writer.writeheader()
        writer.writerows(cw_rows)

    lit = sum(1 for r in rows if "占位" not in r["ref_f_max_eval_basis"])
    print(f"objects_已有安全阈值.csv: {len(rows)} 行已补齐物理量与F_max评测参考")
    print(f"  文献换算值: {lit} 行 | 类别系数占位: {len(rows) - lit} 行")
    print(f"  质量解析: {sum(1 for r in rows if r['mass_kg'])}/{len(rows)} 行")
    print(f"  体积/密度: {sum(1 for r in rows if r['volume_m3'])}/{len(rows)} 行")
    print(f"family_crosswalk.csv 已生成: {len(cw_rows)} 行")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""SafetyVTLA 视觉力先验离线评测 (独立版, 0821)。

按《SafetyVTLA_缺口与可用资源整理_20260821.md》§5 A 类任务实现;
因 SafetyVTLA 仓库 (scripts/predict_visual_grasp_force.py) 本机缺失,
预测器采用可插拔接口 + 三个基线实现。仓库找回后实现 SafetyVTLPredictor
适配器接入即可, 评测协议不变。

协议:
- 数据: objects_已有安全阈值.csv (41 物体, 16 族)
- 真值: ref_f_min_total_N (ExpForce F*, 两指合力) -> 单指 F/2 (SafetyVTLA 单指语义)
- 切分: 按族留一 (leave-one-family-out), 无同型泄漏
- 指标: per-family MAE (单指 N), 总体 MAE, 最差族, F_min/F_max 覆盖率

红线: 本评测为离线参照分析, 不代表 Gen3 端到端结果;
F_max 评测参考列含 24 行类别系数占位 (仅 gap 上界, 非真值)。
"""
import csv
import os

ROOT = r"E:\A-触觉机器学习"
THRESH = os.path.join(ROOT, "数据采集准备", "objects_已有安全阈值.csv")
LOG_DIR = os.path.join(ROOT, "training_log")
G = 9.8


class GlobalMean:
    name = "全局均值"

    def fit(self, rows):
        self.v = sum(r["y"] for r in rows) / len(rows)

    def predict(self, row):
        return self.v


class CategoryMean:
    name = "类别均值(易碎/柔性/刚体)"

    def fit(self, rows):
        d = {}
        for r in rows:
            d.setdefault(r["cat"], []).append(r["y"])
        self.d = {k: sum(v) / len(v) for k, v in d.items()}

    def predict(self, row):
        return self.d.get(row["cat"], sum(self.d.values()) / len(self.d))


class PhysicsFitted:
    name = "物理先验(F=mg/mu, mu拟合)"

    def fit(self, rows):
        num = den = 0.0
        for r in rows:
            mg = r["mass_kg"] * G
            num += mg * r["y"]
            den += mg * mg
        self.mu = num / den if den > 0 else 1.0

    def predict(self, row):
        return row["mass_kg"] * G / self.mu

    def describe(self):
        return f"拟合有效摩擦系数 mu={self.mu:.2f}"


def main():
    with open(THRESH, encoding="utf-8-sig", newline="") as f:
        raw = list(csv.DictReader(f))

    rows = []
    for r in raw:
        if not r["ref_f_min_total_N"]:
            continue
        rows.append({
            "oid": r["object_id"], "fam": r["object_family_id"],
            "cat": r["ref_category"], "mass_kg": float(r["mass_kg"] or 0),
            "y": float(r["ref_f_min_total_N"]) / 2.0,  # 单指语义
            "fmax": r["ref_f_max_eval_total_N"],
            "fmax_basis": r["ref_f_max_eval_basis"],
        })
    fams = sorted({r["fam"] for r in rows})

    predictors = [GlobalMean(), CategoryMean(), PhysicsFitted()]
    results = {}
    for p in predictors:
        fam_mae = {}
        for fam in fams:
            train = [r for r in rows if r["fam"] != fam]
            test = [r for r in rows if r["fam"] == fam]
            p.fit(train)
            errs = [abs(p.predict(r) - r["y"]) for r in test]
            fam_mae[fam] = (sum(errs) / len(errs), len(test))
        overall = sum(m * n for m, n in fam_mae.values()) / len(rows)
        results[p.name] = (fam_mae, overall, p.describe() if hasattr(p, "describe") else "")

    lit = sum(1 for r in rows if "占位" not in r["fmax_basis"])
    lines = [
        "# 视觉力先验离线评测 (基线版) — 2026-08-21",
        "",
        "数据: objects_已有安全阈值.csv, "
        f"{len(rows)} 物体 / {len(fams)} 族; 真值 = ExpForce 参照 F_min"
        " (两指合力, 单指 = F/2, SafetyVTLA 单指语义); 切分 = 按族留一 (LOFO)。",
        "",
        "**红线声明**: SafetyVTLA 仓库 predict_visual_grasp_force.py 本机缺失,",
        "本报告为基线预测器离线参照分析, **不代表 Gen3 视觉→F_min 结果**;",
        f"F_max 评测参考列: 文献换算 {lit} 行 + 类别系数占位 {len(rows) - lit} 行"
        " (占位仅评 gap 上界, 不作真值)。",
        "",
        "## 总体结果 (单指 MAE, N)",
        "",
        "| 预测器 | 总体MAE | 说明 |",
        "|---|---|---|",
    ]
    for name, (_, overall, desc) in results.items():
        lines.append(f"| {name} | {overall:.3f} | {desc} |")

    lines += ["", "## Per-family MAE (单指 N, 按最差基线排序)", "",
              "| 族 | 物体数 | " + " | ".join(n for n in results) + " |",
              "|---|---|" + "---|" * len(results)]
    worst = max(results.items(), key=lambda kv: kv[1][1])[0]
    for fam in sorted(fams, key=lambda f: -max(results[n][0][f][0] for n in results)):
        cells = " | ".join(f"{results[n][0][fam][0]:.3f}" for n in results)
        lines.append(f"| {fam} | {results[worst][0][fam][1]} | {cells} |")
    wname, (wmae, _) = max(results[worst][0].items(), key=lambda kv: kv[1][0])
    lines += ["", f"**最差族**: {wname} (MAE {wmae:.3f} N, 预测器={worst})"]

    lines += ["", "## 覆盖率", "",
              f"- F_min: {len(rows)}/{len(raw)} 行有参照值 (全部为 ExpForce 同型参照, "
              "本物实测 0 个, 待用户实测)",
              f"- F_max 评测参考: {lit}/{len(rows)} 文献换算 + {len(rows) - lit} 占位",
              f"- 物理量: mass/volume/density {sum(1 for r in raw if r.get('volume_m3'))}/{len(raw)}"]

    os.makedirs(LOG_DIR, exist_ok=True)
    out = os.path.join(LOG_DIR, "visual_force_prior_offline_20260821.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"评测完成: {len(rows)} 物体 / {len(fams)} 族 / {len(predictors)} 基线")
    for name, (_, overall, desc) in results.items():
        print(f"  {name}: 总体单指MAE = {overall:.3f} N  {desc}")
    print(f"最差族: {wname} ({wmae:.3f} N)")
    print(f"报告: {out}")


if __name__ == "__main__":
    main()

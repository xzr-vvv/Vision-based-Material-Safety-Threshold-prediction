# -*- coding: utf-8 -*-
"""应用第三批文献力值(两指夹持语义优先) + 交叉验证证据."""
import csv
import os

RGB = r"E:\A-触觉机器学习\RGB_dataset"
FORCE_MAP = os.path.join(RGB, "force_map.csv")

MAO_EV = "Mao et al. 2024, Advanced Intelligent Systems 6(8) DOI:10.1002/aisy.202400175, 仿生光纤触觉指实测最大损伤力; https://onlinelibrary.wiley.com/doi/full/10.1002/aisy.202400175"
GBT_EV = "GB/T 27590-2011 纸杯杯身挺度(径向双板对压至9.5mm变形): <=250mL 合格品>=2.10N/一等品>=2.60N/优等品>=3.00N; http://www.xktest.com/news/s57233.html"

# oid -> (f_max, f_max_basis, f_max_semantics, 是否单指语义换算, evidence)
UPDATES = {
    "E040": ("2.4", "文献实测(薯片最大损伤力)",
     "单指法向力(压头加载)", True, MAO_EV),
    "FRA019": ("2.4", "文献近似(单片薯片损伤力,密封袋近似)",
     "单指法向力(压头加载)", True, MAO_EV + "; 袋装受力经袋分布, 取单片阈值为保守值"),
    "E086": ("0.4", "文献实测(蓝莓最大损伤力)",
     "单指法向力(压头加载)", True, MAO_EV),
    "E005": ("2.1", "国标实测(纸杯杯身挺度下限)",
     "单指法向力(双板对压)", True, GBT_EV),
    "E006": ("2.1", "国标实测(纸杯杯身挺度下限)",
     "单指法向力(双板对压)", True, GBT_EV),
}

# 交叉验证/参照证据(不改动数值)
EV_APPEND = {
    "SOF004": "交叉验证: Mao2024测草莓损伤0.5N, 与钟燚0.6N一致",
    "E026": "交叉验证: Mao2024测草莓损伤0.5N, 与钟燚0.6N一致",
    "E073": "交叉验证: Mao2024测草莓损伤0.5N, 与钟燚0.6N一致",
    "E074": "交叉验证: Mao2024测草莓损伤0.5N, 与钟燚0.6N一致",
    "FRA024": "参照: 软指夹爪以0.12N成功夹持鸡蛋(Mun2024 SORB DOI:10.1089/soro.2023.0068, 安全下界); FORTE软指0-8N范围抓薯片/树莓98.6%成功(arXiv:2506.18960)",
    "E060": "参照: 软指夹爪以0.12N成功夹持鸡蛋(Mun2024, 安全下界)",
    "E061": "参照: 软指夹爪以0.12N成功夹持鸡蛋(Mun2024, 安全下界)",
    "E030": "参照: 灵巧手柑橘指尖力6.39-11.65N无损伤(2026, 安全上界参考); https://airobotsight.com/design-and-experimental-analysis-of-a-dexterous-robotic-hand-for-citrus-manipulation/",
    "E075": "参照: 同E030 柑橘指尖力安全范围",
    "E080": "参照: 千禧小番茄驱动载荷11-14.56N时损伤度0(华南农大2022, 驱动载荷非接触力); https://journal.scau.edu.cn/article/doi/10.7671/j.issn.1001-411X.202212021",
    "E081": "参照: 同E080 小番茄驱动载荷损伤度0",
}


def to_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def main():
    with open(FORCE_MAP, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
        fields = list(rows[0].keys())
    n_upd = 0
    for r in rows:
        oid = r["object_id"]
        if oid in UPDATES:
            val, basis, sem, conv, ev = UPDATES[oid]
            r["f_max_label_N"] = val
            r["f_max_basis"] = basis
            r["f_max_semantics"] = sem
            v = to_float(val)
            r["f_max_total_N"] = f"{v * 2:g}" if (conv and v) else val
            if conv:
                r["conversion_rule"] = "F_total = 2×F_single (SAFETYVTLA默认反算)"
            base = r.get("evidence", "")
            r["evidence"] = (base + " | " if base else "") + ev
            n_upd += 1
        if oid in EV_APPEND:
            base = r.get("evidence", "")
            tag = EV_APPEND[oid].split(":")[0]
            if tag not in base:
                r["evidence"] = (base + " | " if base else "") + EV_APPEND[oid]
    with open(FORCE_MAP, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    n_lit = sum(1 for r in rows if r.get("f_max_basis"))
    print(f"更新: {n_upd}, 有文献/国标f_max: {n_lit}, 总行: {len(rows)}")


if __name__ == "__main__":
    main()

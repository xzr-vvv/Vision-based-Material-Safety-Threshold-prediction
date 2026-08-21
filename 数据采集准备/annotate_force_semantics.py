# -*- coding: utf-8 -*-
"""force_map.csv 力值语义标注 (SAFETYVTLA V1 要求).

规则:
- f_min 全部目标语义 = 两指法向力之和 (ExpForce 论文定义: sum of contact normal
  forces of two parallel gripper fingers; arXiv:2603.08668)
- f_max 文献值多为平板压缩/单点加载 = 单指法向力语义, 按文件默认规则
  F_total = 2 x F_single 显式换算, 保留原始值不覆盖
- 语义无法确认的值(草莓/GraspSense): 不做换算, 保守按合力采用并标注
"""
import csv
import os

RGB = r"E:\A-触觉机器学习\RGB_dataset"
FORCE_MAP = os.path.join(RGB, "force_map.csv")

TWO_F = "两指法向力之和"

# f_max_basis -> (f_max_semantics, 是否单指语义可换算)
BASIS_SEM = {
    "文献实测(平板压缩破碎力下限)": ("单指法向力(平板压缩)", True),
    "文献实测(整粒压缩破裂力下限)": ("单指法向力(平板压缩)", True),
    "文献实测(纵向挤压破裂力)": ("单指法向力(平板挤压)", True),
    "文献实测(压缩损伤阈值)": ("单指法向力(平板压缩)", True),
    "文献实测(横向极限载荷)": ("单指法向力(平板压缩)", True),
    "文献实测(压缩50%形变最大力)": ("单指法向力(平板压缩)", True),
    "文献实测(蜜橘压缩力下限)": ("单指法向力(平板压缩)", True),
    "文献实测(非脆性品种压缩力)": ("单指法向力(平板压缩)", True),
    "文献实测(果皮破裂力下限)": ("单指法向力(平板压缩)", True),
    "文献实测(准静态压缩压伤最大力)": ("单指法向力(平板压缩)", True),
    "文献实测(拇指挤压可见压伤)": ("单指法向力(单点按压)", True),
    "文献实测(空罐侧壁屈曲下限)": ("单指法向力(侧向双板挤压)", True),
    "文献实测(夹持损伤阈值,两指语义)": ("夹持法向力(单指/合力未确认,保守按合力)", False),
}

# 修正: 草莓论文力语义无法确认(agent 核查结论), 改标注
FIX_BASIS = {
    "文献实测(夹持损伤阈值,两指语义)": "文献实测(夹持损伤阈值,单指/合力未确认)",
}

NEW_F_MAX = {
    "FRA016": ("2.1", "国标实测(杯身挺度,径向对压9.5mm变形,合格品下限)",
     "单指法向力(双板对压)",
     "GB/T 27590-2011 纸杯: <=250mL合格品>=2.10N/一等品>=2.60N/优等品>=3.00N; http://www.xktest.com/news/s57233.html"),
    "FRA003": ("69.0", "文献实测(GraspSense玻璃高脚杯杯碗接触最大安全夹持力)",
     "夹持安全力(总/单指未确认,保守按合力)",
     "GraspSense arXiv:2604.05697: 玻璃高脚杯杯碗接触69N/杯茎接触565N; https://arxiv.org/html/2604.05697v1"),
    "FRA002": ("69.0", "文献近似(GraspSense玻璃杯碗69N,薄壁直筒杯近似)",
     "夹持安全力(总/单指未确认,保守按合力)",
     "GraspSense arXiv:2604.05697: 玻璃高脚杯杯碗69N; 薄壁玻璃杯近似采用; https://arxiv.org/html/2604.05697v1"),
}

EV_APPEND = {
    "FRA001": "文献参照: GraspSense玻璃高脚杯杯碗安全夹持69N(arXiv:2604.05697), 厚壁直筒杯更强",
    "FRA005": "文献参照: 小玻璃瓶径向挤压>=1067N(Read Consulting 2010); https://readconsulting.com/glass-expert-discusses-glass-bottle-strength/",
    "FRA066": "文献参照: 同FRA005 小玻璃瓶径向>=1067N; 汽水瓶/奶瓶未单独实测",
    "FRA010": "文献参照: Pi0以5.8N平均力成功夹持陶瓷杯(94%成功率,CSDN 2026,安全下界非损伤阈值); https://blog.csdn.net/weixin_35459464/article/details/158020767",
    "RIG008": "文献参照: Pringles罐轴向压缩92.3N(Alibaba LifeTips 2026,弱来源,侧压未找到)",
    "FRA017": "文献参照: 罐壁球体动态撞击最大力4.89-8.01N(LS-DYNA,动态非准静态); https://lsdyna.ansys.com/wp-content/uploads/attachments/session10-3.pdf",
    "FRA039": "塑料杯侧捏溃力: 未找到文献实测值(仅有顶压堆叠强度26-975N,语义不符)",
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
    for col in ("f_min_semantics", "f_min_single_N", "f_max_semantics",
                "f_max_total_N", "conversion_rule"):
        if col not in fields:
            fields.append(col)

    for r in rows:
        oid = r["object_id"]
        # --- f_min 语义: 全部两指合力 (ExpForce 定义) ---
        fmin = to_float(r.get("f_min_label_N"))
        r["f_min_semantics"] = TWO_F
        if fmin is not None:
            r["f_min_single_N"] = f"{fmin / 2:g}"
        r["conversion_rule"] = "F_single = F_total/2 (SAFETYVTLA默认)"

        # --- f_max 语义 ---
        basis = r.get("f_max_basis", "")
        if oid in NEW_F_MAX:
            val, nbasis, sem, ev = NEW_F_MAX[oid]
            r["f_max_label_N"] = val
            r["f_max_basis"] = nbasis
            r["f_max_semantics"] = sem
            base = r.get("evidence", "")
            r["evidence"] = (base + " | " if base else "") + ev
            v = to_float(val)
            if sem.startswith("单指"):
                r["f_max_total_N"] = f"{v * 2:g}" if v else ""
            else:
                r["f_max_total_N"] = val  # 未确认语义, 保守按合力不换算
        elif basis in BASIS_SEM:
            sem, convertible = BASIS_SEM[basis]
            r["f_max_semantics"] = sem
            v = to_float(r.get("f_max_label_N"))
            if convertible and v is not None:
                r["f_max_total_N"] = f"{v * 2:g}"
                r["conversion_rule"] = "F_total = 2×F_single (SAFETYVTLA默认反算)"
            else:
                r["f_max_total_N"] = r.get("f_max_label_N", "")  # 保守按合力
        elif to_float(r.get("f_max_label_N")) is not None:
            # 无文献依据: E系5x外推 或 FRA/RIG/SOF估算, 均沿用 f_min 的两指合力单位
            if oid.startswith("E"):
                r["f_max_semantics"] = f"{TWO_F}(5×f_min外推,估算)"
            else:
                r["f_max_semantics"] = f"{TWO_F}(估算)"
            r["f_max_total_N"] = r.get("f_max_label_N", "")
        else:
            r["f_max_semantics"] = ""
            r["f_max_total_N"] = ""

        # --- basis 文本修正(草莓) ---
        if r.get("f_max_basis") in FIX_BASIS:
            r["f_max_basis"] = FIX_BASIS[r["f_max_basis"]]

        # --- 证据补充 ---
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
    n_two = sum(1 for r in rows if (r.get("f_max_total_N") or "") != "")
    print(f"总行: {len(rows)}, 有文献f_max: {n_lit}, 有两指合力f_max: {n_two}")


if __name__ == "__main__":
    main()

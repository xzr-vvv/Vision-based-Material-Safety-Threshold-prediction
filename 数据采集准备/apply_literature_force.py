# -*- coding: utf-8 -*-
"""将文献实测力值写入 force_map.csv (f_max 为主), 并补 SOF031 梨."""
import csv
import os

RGB = r"E:\A-触觉机器学习\RGB_dataset"
FORCE_MAP = os.path.join(RGB, "force_map.csv")

# 直接可用的实测损伤/破坏阈值 -> (f_max, basis, evidence)
F_MAX_LIT = {
    "SOF001": (194.0, "文献实测(准静态压缩压伤最大力)",
     "Jahanbakhshi et al. 2020, Int J Fruit Sci 20(3), 香蕉准静态压缩; https://www.tandfonline.com/doi/full/10.1080/15538362.2019.1633723"),
    "SOF002": (163.0, "文献实测(横向极限载荷)",
     "脐橙挤压损伤力学特性, 农业工程学报, 横向163.46N/纵向196.15N; http://www.tcsae.org/cn/article/id/687bd121-9deb-4e40-8783-d19f0865852c"),
    "SOF004": (0.6, "文献实测(夹持损伤阈值,两指语义)",
     "钟燚等 2025, 农业工程学报41(21), 草莓夹持0.6-1N损伤率55%; http://tcsae.org/article/doi/10.11975/j.issn.1002-6819.202411048"),
    "SOF005": (4.0, "文献实测(整粒压缩破裂力下限)",
     "Inouye et al. 2013, HortScience 48(9), 葡萄整粒最大力4.0-13.9N; https://journals.ashs.org/hortsci/view/journals/hortsci/48/9/article-p1130.xml"),
    "SOF007": (10.0, "文献实测(拇指挤压可见压伤)",
     "Australian Avocado Industry 2018, Talking Avocados 28(4), 10N轻度挤压压伤; https://avocado.org.au/public-articles/tav28n4_bruising/"),
    "SOF008": (50.57, "文献实测(压缩50%形变最大力)",
     "Jentzsch et al. 2022, Frontiers in Materials 9, 柠檬准静态压缩; https://www.frontiersin.org/journals/materials/articles/10.3389/fmats.2022.979151/full"),
    "SOF009": (21.42, "文献实测(纵向挤压破裂力)",
     "熊征等 2020, 现代农业装备41(3), 樱桃番茄纵向21.42N/横向37.59N; https://www.aeeisp.com/xdnyzb/cn/article/pdf/preview/ccd641a3-9bc1-4640-b72a-da69cf956a52.pdf"),
    "SOF031": (42.0, "文献实测(果皮破裂力下限)",
     "Mechanical behavior of Deveci pear cultivar, 破裂力41.95-52.69N; https://www.researchgate.net/publication/338448198"),
    "RIG001": (635.0, "文献实测(空罐侧壁屈曲下限)",
     "LS-DYNA/ANSYS: Sidewall Indentation and Buckling of Aluminum Beverage Cans, 635-985N; https://lsdyna.ansys.com/wp-content/uploads/attachments/session10-3.pdf"),
    "E030": (13.3, "文献实测(蜜橘压缩力下限)",
     "W. Murcott mandarin 物性研究, 压缩1.351-1.650kgf(13.3-16.2N); https://pdfs.semanticscholar.org/b01a/03a4a88e945678728184a76bc386b8695c7e.pdf"),
    "E075": (13.3, "文献实测(蜜橘压缩力下限)",
     "同E030来源"),
    "E036": (8.0, "文献实测(非脆性品种压缩力)",
     "Univ. of Arkansas blackberry breeding, HortScience 51(5), 非脆性8.0N/脆性11.8N; https://journals.ashs.org/view/journals/hortsci/51/5/article-p468.xml"),
    "E088": (8.0, "文献实测(非脆性品种压缩力)",
     "同E036来源"),
}

# 文献有参照值但不作为 f_max (加载方式差异大/只有极限载荷) -> 仅补 evidence
EV_ONLY = {
    "SOF003": "文献参照: 樱桃番茄挤压破裂21-64N(熊征2020, 品种差异大); 苹果损伤阈值40N/压伤54N(Zhu 2022; Wang 2024 Processes 12:543)",
    "SOF013": "文献参照: 96份甜椒资源破裂力1.26-38.37N/坚实度15-35N(中华农业科学2025); https://www.chinaagrisci.com/EN/10.3864/j.issn.0578-1752.2025.14.011",
    "E086": "文献参照: 蓝莓bioyield力0.491N, 仪器压缩上限2.2N(USDA/PMC); https://www.ars.usda.gov/ARSUserFiles/30200525/PortableInstrumentsPA999.pdf",
    "FRA017": "文献参照: 空罐侧壁整体屈曲635-985N(LS-DYNA), 局部凹陷阈值远低于此, 无实测凹陷值",
    "FRA016": "文献参照: 纸杯轴向抗压344-480N(上海质检院2026, 顶压堆叠语义, 非侧捏); https://m.gmw.cn/2026-02/10/content_1304338843.htm",
    "FRA039": "文献参照: 一次性塑料杯抗压PP/PS/PET 26-975N, PLA 26-151N(同上, 顶压语义)",
    "RIG002": "文献参照: PET瓶压缩1300N出现屈曲(数值研究); https://www.researchgate.net/publication/264187009",
    "RIG023": "文献参照: 同RIG002, PET瓶屈曲1300N",
    "RIG006": "文献参照: 空罐轴向压溃679-2400N(Wang 2024 Materials); https://pmc.ncbi.nlm.nih.gov/articles/PMC11174009/",
    "RIG058": "文献参照: 同RIG006, 金属罐轴向压溃679-2400N",
}

NEW_ROWS = [
    {"object_id": "SOF031", "f_min_label_N": "1.5", "f_max_label_N": "42.0",
     "label_source": "ExpForce同型实测(Green pear; f_min=1.5N)", "is_measured": "0",
     "value_basis": "同型实测",
     "evidence": "ExpForce_dataset_官方原始.csv (UT Austin; LEGATO两指爪+FORTE触觉指实测夹持力)"},
]


def main():
    with open(FORCE_MAP, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
        fields = list(rows[0].keys())
    ids = {r["object_id"] for r in rows}
    for r in NEW_ROWS:
        if r["object_id"] not in ids:
            rows.append(r)
    n_max, n_ev = 0, 0
    for r in rows:
        oid = r["object_id"]
        if oid in F_MAX_LIT:
            fmax, basis, ev = F_MAX_LIT[oid]
            r["f_max_label_N"] = str(fmax)
            r["value_basis"] = r.get("value_basis", "")
            r["f_max_basis"] = basis
            if "文献" in ev or "http" in ev:
                base_ev = r.get("evidence", "")
                r["evidence"] = (base_ev + " | " if base_ev else "") + ev
            n_max += 1
        elif oid in EV_ONLY:
            base_ev = r.get("evidence", "")
            r["evidence"] = (base_ev + " | " if base_ev else "") + EV_ONLY[oid]
            n_ev += 1
    if "f_max_basis" not in fields:
        fields.append("f_max_basis")
    with open(FORCE_MAP, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"f_max文献实测更新: {n_max}, 仅补证据: {n_ev}, 新增行: {len(NEW_ROWS)}, 总行: {len(rows)}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""应用食品断裂力文献实测值到 force_map.csv (第二批: 鸡蛋/水果/薯片等)."""
import csv
import os

RGB = r"E:\A-触觉机器学习\RGB_dataset"
FORCE_MAP = os.path.join(RGB, "force_map.csv")

EGG_EV = ("鸡蛋壳平板压缩破碎力42-51N: De Ketelaere2002 Br Poult Sci 43:238; "
          "Altuntas&Sekeroglu2008 DOI:10.1111/j.1745-4530.2008.00263.x; "
          "Sutanto2025 Commun Phys 8:182 DOI:10.1038/s42005-025-02087-0")
STRAW_EV = "钟燚等2025 农业工程学报41(21) 草莓夹持0.6-1N损伤率55%; http://tcsae.org/article/doi/10.11975/j.issn.1002-6819.202411048"
GRAPE_EV = "Inouye2013 HortScience 48(9) 葡萄整粒压缩4.0-13.9N; https://journals.ashs.org/hortsci/view/journals/hortsci/48/9/article-p1130.xml"
CTOM_EV = "熊征等2020 现代农业装备41(3) 樱桃番茄纵向21.42N/横向37.59N; https://www.aeeisp.com/xdnyzb/cn/article/pdf/preview/ccd641a3-9bc1-4640-b72a-da69cf956a52.pdf"
BANANA_EV = "Jahanbakhshi2020 Int J Fruit Sci 20(3) 香蕉准静态压缩; https://www.tandfonline.com/doi/full/10.1080/15538362.2019.1633723"
APPLE_EV = "Zhu2022 Food Sci Technol Res 28(1) 苹果压缩损伤阈值40N; Wang2024 Processes 12:543 压伤阈值54N; https://mdpi-res.com/d_attachment/processes/processes-12-00543/article_deploy/processes-12-00543.pdf"
ORANGE_EV = "脐橙挤压损伤力学 农业工程学报 横向163.46N/纵向196.15N; http://www.tcsae.org/cn/article/id/687bd121-9deb-4e40-8783-d19f0865852c"
LEMON_EV = "Jentzsch2022 Frontiers in Materials 9 柠檬压缩50%形变50.57N; https://www.frontiersin.org/journals/materials/articles/10.3389/fmats.2022.979151/full"
PEAR_EV = "Deveci梨破裂力41.95-52.69N; https://www.researchgate.net/publication/338448198"

F_MAX_LIT = {
    "FRA024": (42.0, "文献实测(平板压缩破碎力下限)", EGG_EV),
    "E060": (42.0, "文献实测(平板压缩破碎力下限)", EGG_EV),
    "E061": (42.0, "文献实测(平板压缩破碎力下限)", EGG_EV),
    "E026": (0.6, "文献实测(夹持损伤阈值,两指语义)", STRAW_EV),
    "E073": (0.6, "文献实测(夹持损伤阈值,两指语义)", STRAW_EV),
    "E074": (0.6, "文献实测(夹持损伤阈值,两指语义)", STRAW_EV),
    "E032": (4.0, "文献实测(整粒压缩破裂力下限)", GRAPE_EV),
    "E033": (4.0, "文献实测(整粒压缩破裂力下限)", GRAPE_EV),
    "E076": (4.0, "文献实测(整粒压缩破裂力下限)", GRAPE_EV),
    "E077": (4.0, "文献实测(整粒压缩破裂力下限)", GRAPE_EV),
    "E078": (4.0, "文献实测(整粒压缩破裂力下限)", GRAPE_EV),
    "E079": (4.0, "文献实测(整粒压缩破裂力下限)", GRAPE_EV),
    "E080": (21.42, "文献实测(纵向挤压破裂力)", CTOM_EV),
    "E081": (21.42, "文献实测(纵向挤压破裂力)", CTOM_EV),
    "E082": (21.42, "文献实测(纵向挤压破裂力)", CTOM_EV),
    "E083": (21.42, "文献实测(纵向挤压破裂力)", CTOM_EV),
    "E084": (21.42, "文献实测(纵向挤压破裂力)", CTOM_EV),
    "E085": (21.42, "文献实测(纵向挤压破裂力)", CTOM_EV),
    "E056": (194.0, "文献实测(准静态压缩压伤最大力)", BANANA_EV),
    "E024": (40.0, "文献实测(压缩损伤阈值)", APPLE_EV),
    "E066": (40.0, "文献实测(压缩损伤阈值)", APPLE_EV),
    "E067": (40.0, "文献实测(压缩损伤阈值)", APPLE_EV),
    "E068": (40.0, "文献实测(压缩损伤阈值)", APPLE_EV),
    "E069": (40.0, "文献实测(压缩损伤阈值)", APPLE_EV),
    "E102": (40.0, "文献实测(压缩损伤阈值)", APPLE_EV),
    "E027": (163.0, "文献实测(横向极限载荷)", ORANGE_EV),
    "E028": (163.0, "文献实测(横向极限载荷)", ORANGE_EV),
    "E029": (163.0, "文献实测(横向极限载荷)", ORANGE_EV),
    "E103": (163.0, "文献实测(横向极限载荷)", ORANGE_EV),
    "E023": (50.57, "文献实测(压缩50%形变最大力)", LEMON_EV),
    "E065": (50.57, "文献实测(压缩50%形变最大力)", LEMON_EV),
    "E064": (42.0, "文献实测(果皮破裂力下限)", PEAR_EV),
}

PEPPER_EV = "文献参照: 96份甜椒资源破裂力1.26-38.37N(中华农业科学2025); https://www.chinaagrisci.com/EN/10.3864/j.issn.0578-1752.2025.14.011"
EV_ONLY = {
    "FRA025": "文献参照: 鸡蛋壳压缩42-51N(见FRA024来源), 皮蛋壳未单独实测",
    "FRA051": "文献参照: 鸡蛋壳压缩42-51N, 鸭蛋壳未单独实测",
    "FRA056": "文献参照: 掏空蛋壳无可核验实测值(988N为模拟值, 不采用)",
    "FRA019": "文献参照: 单片薯片三点弯曲断裂0.55-2.86N, 压入测试6.1-8.5N(Xu&Kerr2012 LWT DOI:10.1016/j.lwt.2012.02.019); 整袋受压分布不同",
    "FRA027": "文献参照: 干面条单根三点弯曲断裂0.30-0.73N(Cai2023 Foods 12:55 DOI:10.3390/foods12010055); 整束未实测",
    "FRA054": "文献参照: 同FRA027干面条单根断裂力; 粉丝整捆未实测",
    "E022": "文献参照: 柑橘类准静态压缩 柚40.32N/柠檬50.57N(Jentzsch2022); 西柚未单独实测",
    "E009": "文献参照: 柠檬压缩50.57N(Jentzsch2022); 青柠未单独实测",
    "E020": "文献参照: 樱桃番茄挤压破裂21-64N(熊征2020); 普通番茄未单独实测",
    "E021": "文献参照: 同E020樱桃番茄挤压破裂21-64N",
    "E025": PEPPER_EV,
    "E070": PEPPER_EV,
    "E071": PEPPER_EV,
    "E072": PEPPER_EV,
    "E087": "树莓: 未找到文献实测值",
    "E086": "蓝莓: bioyield力0.491N, 仪器压缩上限2.2N(USDA); 加载语义弱, 不作f_max",
    "E013": "棉花糖: 未找到文献实测值",
    "E048": "棉花糖: 未找到文献实测值",
    "E049": "棉花糖: 未找到文献实测值",
    "E122": "棉花糖: 未找到文献实测值",
}


def main():
    with open(FORCE_MAP, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
        fields = list(rows[0].keys())
    if "f_max_basis" not in fields:
        fields.append("f_max_basis")
    n_max, n_ev = 0, 0
    for r in rows:
        oid = r["object_id"]
        if oid in F_MAX_LIT:
            fmax, basis, ev = F_MAX_LIT[oid]
            r["f_max_label_N"] = str(fmax)
            r["f_max_basis"] = basis
            base_ev = r.get("evidence", "")
            r["evidence"] = (base_ev + " | " if base_ev else "") + ev
            n_max += 1
        elif oid in EV_ONLY:
            base_ev = r.get("evidence", "")
            tag = EV_ONLY[oid]
            if tag.split(":")[0] not in base_ev:
                r["evidence"] = (base_ev + " | " if base_ev else "") + tag
            n_ev += 1
    with open(FORCE_MAP, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"f_max文献实测更新: {n_max}, 证据补充: {n_ev}, 总行: {len(rows)}")


if __name__ == "__main__":
    main()

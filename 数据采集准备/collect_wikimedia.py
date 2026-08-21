# -*- coding: utf-8 -*-
"""从 Wikimedia Commons 下载物体图片, 补充 RGB_dataset.

用法:
  E:\\python.exe collect_wikimedia.py --phase fra_light [--limit N]

阶段:
  fra_light  轻脆 FRA 31 物体 (新建 s0-s14)
  fra_heavy  重脆 FRA 32 物体 (新建 s0-s14)
  e_fragile  E 系易碎 11 物体 (原已有 1 张, 补 s1-s14)
  fra_soft   FRA 柔性 3 物体 (FRA016/039/040)
  sof        SOF 柔性 24 物体 (有 force_map 力值条目的)
  rig        RIG 刚体 60 物体
  e_rest     E 系非易碎 118 物体 (检索词由 ExpForce 名称自动推导)

候选来源(精度从高到低):
  1. Commons 分类成员 (人工策展, +15 分)
  2. 全文搜索 (按标题词重叠打分)
过滤: 仅 JPEG/PNG, 原图>=400x300, 标题含负面词扣分, 要求总分>0,
      JPEG(真实照片概率高)加分。
图片: 640px 缩略图 -> RGB PNG -> s{N}.png, 每物体共 15 张
记录: provenance.csv 逐张追加; wikimedia_state.json 记录已用标题(全局去重)
"""
import argparse
import csv
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, r"E:\Lib\site-packages")
import numpy as np
from PIL import Image

ROOT = r"E:\A-触觉机器学习"
RGB = os.path.join(ROOT, "RGB_dataset")
PREP = os.path.join(ROOT, "数据采集准备")
STATE_FILE = os.path.join(PREP, "wikimedia_state.json")
TERMS_FILE = os.path.join(PREP, "wikimedia_search_terms.json")
PROV_FILE = os.path.join(RGB, "provenance.csv")
TEMPLATE = os.path.join(PREP, "objects_模板.csv")
EXPFORCE_CSV = os.path.join(ROOT, "ExpForce数据集", "ExpForce_dataset_官方原始.csv")
API = "https://commons.wikimedia.org/w/api.php"
UA = ("V-MaST-DataCollector/1.2 "
      "(academic robot-grasp research; contact: vtla.datacollect@outlook.com)")

E_FRAGILE = ["E008", "E040", "E041", "E060", "E061", "E108", "E116", "E117", "E118", "E119", "E120"]

E_RIGID_FILL = ["E002", "E003", "E016", "E017", "E043", "E044", "E050", "E052", "E053",
                "E058", "E090", "E091", "E093", "E095", "E097", "E098", "E100", "E101",
                "E107", "E110", "E111", "E112", "E113", "E121", "E128", "E129"]
E_SOFT_FILL = ["E009", "E020", "E022", "E023", "E024", "E025", "E026", "E027", "E029",
               "E030", "E031", "E032", "E033", "E036", "E039", "E048", "E056", "E064",
               "E065", "E066", "E067", "E068", "E073", "E074", "E075", "E077", "E080",
               "E086", "E087", "E122"]
SOF_WITH_FORCE = ["SOF001", "SOF002", "SOF003", "SOF004", "SOF005", "SOF006", "SOF007", "SOF008",
                  "SOF009", "SOF010", "SOF011", "SOF012", "SOF013", "SOF014", "SOF015", "SOF016",
                  "SOF017", "SOF018", "SOF019", "SOF020", "SOF023", "SOF024", "SOF030", "SOF038"]

BAD_TITLE = re.compile(
    r"\b(barnstar|award|medal|trophy|tournament|championship|player|players|"
    r"match|matches|team|game|competition|club|map|diagram|chart|logo|drawing|"
    r"illustration|coat of arms|flag|icon|graph|poster|screenshot|svg|clipart|"
    r"emblem|seal|symbol|sign|machine|automat|vending|shop|store|supermarket|"
    r"shelf|factory|production)\b", re.I)
SFILE_RE = re.compile(r"^s(\d+)\.png$")
IMG_EXT = (".png", ".jpg", ".jpeg")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_template():
    d = {}
    with open(TEMPLATE, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            d[row["object_id"]] = (row["leaf_class"], row["object_name"])
    return d


def load_terms():
    with open(TERMS_FILE, encoding="utf-8") as f:
        return json.load(f)


def load_expforce_names():
    img2name = {}
    with open(EXPFORCE_CSV, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            img2name[row["Image"]] = row["Object"]
    e2name = {}
    with open(PROV_FILE, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["object_id"].startswith("E") and row["source"].startswith("ExpForce"):
                fn = os.path.basename(row["image_file"].replace("\\", "/"))
                if fn in img2name:
                    e2name[row["object_id"]] = img2name[fn]
    return e2name


def derive_terms(name):
    words = name.replace("_", " ").split()
    terms = [" ".join(words)]
    if len(words) >= 3:
        terms.append(" ".join(words[-2:]))
        terms.append(words[-1])
    elif len(words) == 2:
        terms.append(words[-1])
    return [t for t in terms if len(t) > 2]


def api_get(params):
    url = API + "?" + urllib.parse.urlencode(params)
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception as e:
            if attempt == 2:
                log(f"    api error: {e}")
                return {}
            time.sleep(1.5 * (attempt + 1))


def api_search(term, limit=50):
    return api_get({
        "action": "query", "format": "json",
        "generator": "search", "gsrsearch": term, "gsrnamespace": "6",
        "gsrlimit": str(limit),
        "prop": "imageinfo", "iiprop": "url|size|mime|extmetadata",
        "iiurlwidth": "640",
    })


def api_category(cat, limit=40):
    return api_get({
        "action": "query", "format": "json",
        "generator": "categorymembers", "gcmtitle": f"Category:{cat}",
        "gcmtype": "file", "gcmlimit": str(limit),
        "prop": "imageinfo", "iiprop": "url|size|mime|extmetadata",
        "iiurlwidth": "640",
    })


def download_bytes(url):
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 30 * (attempt + 1)
                log(f"    429 rate-limited, waiting {wait}s")
                time.sleep(wait)
                continue
            if attempt == 3:
                log(f"    download error: HTTP {e.code}")
                return None
            time.sleep(2.0 * (attempt + 1))
        except Exception as e:
            if attempt == 3:
                log(f"    download error: {e}")
                return None
            time.sleep(2.0 * (attempt + 1))
    return None


def thumb_vec(img):
    g = img.convert("L").resize((32, 32))
    return np.asarray(g, dtype=np.float32).ravel() / 255.0


def title_score(title, terms):
    words = set(re.findall(r"[a-z]+", title.lower()))
    best = 0
    for term in terms:
        tw = [w for w in re.findall(r"[a-z]+", term.lower()) if len(w) > 2]
        best = max(best, sum(1 for w in tw if w in words))
    score = best * 10
    if BAD_TITLE.search(title):
        score -= 30
    return score


def save_state(used_titles):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump({"used_titles": sorted(used_titles)}, f, ensure_ascii=False)


def collect_object(oid, leaf, cat, terms, used_titles, target=15):
    obj_dir = os.path.join(RGB, leaf, oid)
    os.makedirs(obj_dir, exist_ok=True)
    files = os.listdir(obj_dir)
    sfiles = [f for f in files if SFILE_RE.match(f)]
    others = [f for f in files if f.lower().endswith(IMG_EXT) and not SFILE_RE.match(f)]
    have = len(sfiles) + len(others)
    if sfiles:
        start = max(int(SFILE_RE.match(f).group(1)) for f in sfiles) + 1
    else:
        start = 1 if others else 0
    needed = target - have
    if needed <= 0:
        return 0, have

    best = {}

    def add_pages(data, bonus, source_label):
        pages = data.get("query", {}).get("pages", {})
        got = 0
        for p in pages.values():
            title = p.get("title", "")
            if not title or title in used_titles:
                continue
            ii_list = p.get("imageinfo")
            if not ii_list:
                continue
            ii = ii_list[0]
            if ii.get("mime") not in ("image/jpeg", "image/png"):
                continue
            if ii.get("width", 0) < 400 or ii.get("height", 0) < 300:
                continue
            thumb = ii.get("thumburl")
            if not thumb:
                continue
            em = ii.get("extmetadata", {})
            lic = em.get("LicenseShortName", {}).get("value", "unknown")
            descurl = ii.get("descriptionurl", "")
            sc = title_score(title, terms) + bonus
            if ii.get("mime") == "image/jpeg":
                sc += 8
            if sc <= 0:
                continue
            if title not in best or sc > best[title][0]:
                best[title] = (sc, title, thumb, descurl, lic)
            got += 1
        log(f"    {source_label} -> {got} candidates (pool {len(best)})")

    if cat:
        add_pages(api_category(cat), 15, f"cat '{cat}'")
        time.sleep(0.25)
    for term in terms:
        if len(best) >= needed * 3:
            break
        add_pages(api_search(term), 0, f"search '{term}'")
        time.sleep(0.25)

    ranked = sorted(best.values(), key=lambda x: -x[0])
    kept_vecs = []
    for f in sorted(os.listdir(obj_dir)):
        if f.lower().endswith(IMG_EXT):
            try:
                with Image.open(os.path.join(obj_dir, f)) as im:
                    kept_vecs.append(thumb_vec(im))
            except Exception:
                pass
    n = 0
    skipped_dup = 0
    for sc, title, thumb, descurl, lic in ranked:
        if n >= needed:
            break
        data = download_bytes(thumb)
        if not data:
            continue
        try:
            img = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception as e:
            log(f"    bad image {title}: {e}")
            continue
        if img.width < 200 or img.height < 150:
            continue
        # 存图时即时近重复检测, 避免补图后再被多样性修复隔离
        v = thumb_vec(img)
        if kept_vecs and min(np.abs(v - k).mean() for k in kept_vecs) < 0.045:
            skipped_dup += 1
            continue
        fn = f"s{start + n}.png"
        img.save(os.path.join(obj_dir, fn), "PNG")
        with open(PROV_FILE, "a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(
                [oid, f"{leaf}\\{oid}\\{fn}", "Wikimedia Commons", descurl, lic])
        used_titles.add(title)
        save_state(used_titles)
        kept_vecs.append(v)
        n += 1
        time.sleep(0.5)
    if skipped_dup:
        log(f"    skipped {skipped_dup} near-duplicate candidates")
    return n, have + n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    help="逗号分隔多阶段: fra_light,fra_heavy,e_fragile,fra_soft,sof,rig,e_rest,e_rigid_fill,e_soft_fill")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个物体")
    ap.add_argument("--objects", default="",
                    help="逗号分隔物体ID, 直接指定采集对象(覆盖phase)")
    args = ap.parse_args()
    phase_list = [p.strip() for p in args.phase.split(",") if p.strip()]
    for ph in phase_list:
        if ph not in ("fra_light", "fra_heavy", "e_fragile", "fra_soft", "sof", "rig",
                      "e_rest", "e_rigid_fill", "e_soft_fill", "custom"):
            ap.error(f"未知阶段: {ph}")

    template = load_template()
    terms_map = load_terms()

    fra_light = sorted(k for k, (leaf, _) in template.items()
                       if k.startswith("FRA") and leaf == "轻脆")
    fra_heavy = sorted(k for k, (leaf, _) in template.items()
                       if k.startswith("FRA") and leaf == "重脆")
    fra_soft = ["FRA016", "FRA039", "FRA040"]
    rig = [f"RIG{i:03d}" for i in range(1, 61)]
    e_all = [k for k in sorted(template) if re.match(r"^E\d+$", k)]
    e_rest = [e for e in e_all if e not in E_FRAGILE]

    phase_map = {
        "fra_light": fra_light, "fra_heavy": fra_heavy, "e_fragile": E_FRAGILE,
        "fra_soft": fra_soft, "sof": SOF_WITH_FORCE, "rig": rig, "e_rest": e_rest,
        "e_rigid_fill": E_RIGID_FILL, "e_soft_fill": E_SOFT_FILL,
    }
    if args.objects:
        ids = [o.strip() for o in args.objects.split(",") if o.strip()]
        phase_map["custom"] = ids

    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, encoding="utf-8") as f:
            state = json.load(f)
        used_titles = set(state.get("used_titles", []))
    else:
        used_titles = set()

    grand_total = 0
    e2name = None
    for args_phase in phase_list:
        phase_objects = list(phase_map[args_phase])
        if args.limit > 0:
            phase_objects = phase_objects[: args.limit]
        log(f"phase={args_phase} objects={len(phase_objects)} used_titles={len(used_titles)}")
        total_new = 0
        for i, oid in enumerate(phase_objects, 1):
            leaf, name = template[oid]
            entry = terms_map.get(oid)
            if entry:
                cat, terms = entry.get("cat"), entry["terms"]
            elif oid.startswith("E"):
                if e2name is None:
                    e2name = load_expforce_names()
                nm = e2name.get(oid)
                cat, terms = None, (derive_terms(nm) if nm else [])
            else:
                cat, terms = None, []
            if not terms:
                log(f"[{i}/{len(phase_objects)}] {oid} {name}: 无检索词, 跳过")
                continue
            log(f"[{i}/{len(phase_objects)}] {oid} {name} ({leaf}) cat={cat} terms={terms[0]}...")
            n, total = collect_object(oid, leaf, cat, terms, used_titles)
            total_new += n
            log(f"    saved {n}, folder total {total}")
            save_state(used_titles)
        log(f"DONE phase={args_phase} new_images={total_new}")
        grand_total += total_new
    log(f"ALL DONE phases={phase_list} total_new_images={grand_total}")


if __name__ == "__main__":
    main()

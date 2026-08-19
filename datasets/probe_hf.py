# -*- coding: utf-8 -*-
"""探测 HuggingFace 上的 Cornell / YCB-Video 数据集镜像"""
import json
import socket
import urllib.request

socket.setdefaulttimeout(40)


def get(u):
    req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r:
        return r.read()


print("=== HF 搜索: cornell grasping ===")
try:
    d = json.loads(get("https://huggingface.co/api/datasets?search=cornell+grasp"))
    for it in d[:10]:
        print("  ", it.get("id"), "| downloads:", it.get("downloads"))
except Exception as e:
    print("ERR", str(e)[:120])

print("=== HF 搜索: ycb video / bop ===")
for q in ("ycbv", "ycb-video", "bop ycbv"):
    try:
        d = json.loads(get("https://huggingface.co/api/datasets?search=" + q.replace(" ", "+")))
        for it in d[:8]:
            print(f"  [{q}]", it.get("id"), "| downloads:", it.get("downloads"))
    except Exception as e:
        print("ERR", q, str(e)[:100])

print("=== HF 数据集文件清单 ===")
for repo in ("JEFFREY-VERDIERE/Cornell_Grasping_Dataset", "bop-benchmark/ycbv"):
    try:
        d = json.loads(get(f"https://huggingface.co/api/datasets/{repo}/tree/main"))
        print(f"--- {repo} ---")
        for it in d[:25]:
            sz = it.get("size", 0)
            print(f"  {it.get('path')}  {sz/1e6:.2f}MB" if sz > 1e6 else f"  {it.get('path')}  {sz/1e3:.1f}KB")
    except Exception as e:
        print("ERR", repo, str(e)[:120])

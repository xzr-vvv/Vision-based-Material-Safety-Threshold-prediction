# -*- coding: utf-8 -*-
"""archive.org 主站搜索 Cornell 数据集副本"""
import json
import socket
import urllib.parse
import urllib.request

socket.setdefaulttimeout(40)

for q in ["cornell grasping", "cornell grasp dataset", "cornell robot learning grasping"]:
    u = ("https://archive.org/advancedsearch.php?q="
         + urllib.parse.quote('"' + q + '"')
         + "&fl%5B%5D=identifier&fl%5B%5D=title&fl%5B%5D=item_size&rows=10&output=json")
    req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
    try:
        d = json.load(urllib.request.urlopen(req))
        docs = d.get("response", {}).get("docs", [])
        print(f"[{q}] {len(docs)}条")
        for doc in docs:
            print("  ", doc.get("identifier"), "|", str(doc.get("title"))[:60], "|", doc.get("item_size"))
    except Exception as e:
        print("ERR", q, str(e)[:100])

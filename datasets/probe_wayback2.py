# -*- coding: utf-8 -*-
"""查 processedData.zip 大小 + rect_data 单文件存档情况"""
import json
import socket
import urllib.request

socket.setdefaulttimeout(90)
UA = {"User-Agent": "Mozilla/5.0"}

print("=== processedData.zip 大小 ===")
u = "http://web.archive.org/web/20130608025029/http://pr.cs.cornell.edu/grasping/rect_data/processedData.zip"
req = urllib.request.Request(u, headers=UA, method="HEAD")
try:
    with urllib.request.urlopen(req) as r:
        print("status", r.status, "| size", r.headers.get("Content-Length"),
              "|", r.url[:120])
except Exception as e:
    print("ERR", str(e)[:150])

print("=== rect_data/data/ 单文件存档 ===")
u2 = ("https://web.archive.org/cdx/search/cdx?url=pr.cs.cornell.edu/grasping/rect_data/data/*"
      "&output=json&limit=60&collapse=urlkey")
req = urllib.request.Request(u2, headers=UA)
with urllib.request.urlopen(req) as r:
    rows = json.load(r)
print("命中", len(rows) - 1, "个 URL:")
for row in rows[1:40]:
    print(f"  {row[1]} {row[3][:20]} {row[2]}")

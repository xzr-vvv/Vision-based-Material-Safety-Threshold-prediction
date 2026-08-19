# -*- coding: utf-8 -*-
"""在 Wayback CDX 里找 Cornell 数据集的 zip/数据文件存档"""
import json
import socket
import urllib.request

socket.setdefaulttimeout(90)

u = ("https://web.archive.org/cdx/search/cdx?url=pr.cs.cornell.edu/grasping/*"
     "&output=json&limit=2000&collapse=urlkey&filter=mimetype:application/x-zip-compressed|application/octet-stream|application/zip")
req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req) as r:
    rows = json.load(r)
print("命中", len(rows) - 1, "条数据文件:")
for row in rows[1:]:
    url, ts, mime = row[2], row[1], row[3]
    status = row[4] if len(row) > 4 else "?"
    print(f"{ts} [{mime}] {status} {url}")

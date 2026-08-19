# -*- coding: utf-8 -*-
"""探测 YCB S3 镜像目录结构，找真实 RGB-D 数据"""
import re
import socket
import sys
import urllib.request

socket.setdefaulttimeout(40)


def ls(u):
    try:
        req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as r:
            html = r.read().decode("utf-8", "ignore")
        links = re.findall(r'href="([^"]+)"', html)
        seen = []
        for l in links:
            if l.startswith(("?", "//", "#")) or l in ("..", "/", "../"):
                continue
            if l not in seen:
                seen.append(l)
        return seen[:80]
    except Exception as e:
        print("ERR", u, str(e)[:120])
        return []


base = "https://rll.eecs.berkeley.edu/ycb/"
print("=== 根目录原始 HTML ===")
req = urllib.request.Request(base, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req) as r:
    print(r.status, r.url)
    html = r.read().decode("utf-8", "ignore")
print(html[:3000])

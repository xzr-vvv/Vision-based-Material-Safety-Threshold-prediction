# -*- coding: utf-8 -*-
"""下载 BOP ycbv 真实 RGB-D 测试集 (660MB, 真实传感器 color+depth)"""
import os
import socket
import sys
import time
import urllib.request

socket.setdefaulttimeout(60)
URL = "https://huggingface.co/datasets/bop-benchmark/ycbv/resolve/main/ycbv_test_bop19.zip"
DST = r"E:\A-触觉机器学习\datasets\ycb_rgbd\ycbv_test_bop19.zip"
os.makedirs(os.path.dirname(DST), exist_ok=True)

req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
t0 = time.time()
with urllib.request.urlopen(req) as r, open(DST, "wb") as f:
    total = int(r.headers.get("Content-Length", 0))
    done = 0
    while True:
        chunk = r.read(1 << 20)
        if not chunk:
            break
        f.write(chunk)
        done += len(chunk)
        if done % (50 << 20) < (1 << 20):
            pct = done / total * 100 if total else 0
            mbps = done / 1e6 / (time.time() - t0 + 1)
            print(f"{done/1e6:.0f}/{total/1e6:.0f}MB ({pct:.0f}%) {mbps:.1f}MB/s", flush=True)
print(f"完成: {DST} {done/1e6:.0f}MB 用时{time.time()-t0:.0f}s")

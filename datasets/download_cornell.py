# -*- coding: utf-8 -*-
"""从 Wayback Machine 下载 Cornell processedData.zip（官方处理版 RGB-D 数据）"""
import os
import socket
import time
import urllib.request

socket.setdefaulttimeout(90)
UA = {"User-Agent": "Mozilla/5.0"}

URL = ("http://web.archive.org/web/20130608025029id_/"
       "http://pr.cs.cornell.edu/grasping/rect_data/processedData.zip")
DST = r"E:\A-触觉机器学习\datasets\cornell_grasping\processedData.zip"
os.makedirs(os.path.dirname(DST), exist_ok=True)

req = urllib.request.Request(URL, headers=UA)
t0 = time.time()
done = 0
with urllib.request.urlopen(req) as r, open(DST, "wb") as f:
    total = int(r.headers.get("Content-Length") or 0)
    print("总大小:", total / 1e6 if total else "未知(流式)", "| 状态", r.status, flush=True)
    while True:
        chunk = r.read(1 << 20)
        if not chunk:
            break
        f.write(chunk)
        done += len(chunk)
        if done % (20 << 20) < (1 << 20):
            mbps = done / 1e6 / (time.time() - t0 + 1)
            eta = (total - done) / 1e6 / mbps if total and mbps > 0 else 0
            print(f"{done/1e6:.0f}MB {mbps:.2f}MB/s ETA {eta/60:.0f}min", flush=True)
dur = time.time() - t0
print(f"完成: {DST} {done/1e6:.1f}MB 用时{dur/60:.1f}min 平均{done/1e6/dur*60:.2f}MB/s")

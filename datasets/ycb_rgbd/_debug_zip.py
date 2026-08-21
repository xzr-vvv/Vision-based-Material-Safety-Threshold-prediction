# -*- coding: utf-8 -*-
"""对比 zip 原始文件与解压文件夹: 场景000055 帧22 的 rgb 是否同一张图"""
import hashlib
import io
import os
import sys
import zipfile

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ZIP = r"E:\A-触觉机器学习\datasets\ycb_rgbd\ycbv_test_bop19.zip"
FOLDER = r"E:\A-触觉机器学习\datasets\ycb_rgbd\test"

z = zipfile.ZipFile(ZIP)
names = z.namelist()
print(f"zip 内文件数: {len(names)}")
z55 = sorted(n for n in names if "/000055/rgb/" in n.replace("\\", "/"))[:6]
print("zip 场景55 rgb 前6个:", [os.path.basename(n) for n in z55])
print("文件夹 场景55 rgb 前6个:", sorted(os.listdir(os.path.join(FOLDER, "000055", "rgb")))[:6])

def md5(b):
    return hashlib.md5(b).hexdigest()[:10]

# zip 内 000022.png
zn = [n for n in names if n.replace("\\", "/").endswith("000055/rgb/000022.png")]
if zn:
    zdata = z.read(zn[0])
    fdata = open(os.path.join(FOLDER, "000055", "rgb", "000022.png"), "rb").read()
    print(f"\nzip 000022.png md5: {md5(zdata)}  大小 {len(zdata)}")
    print(f"文件夹 000022.png md5: {md5(fdata)}  大小 {len(fdata)}")
    print("同一张图:", zdata == fdata)
    # zip 里有没有哪张图和文件夹的 000022.png 相同?
    hit = None
    for n in names:
        if "/000055/rgb/" in n.replace("\\", "/"):
            if md5(z.read(n)) == md5(fdata):
                hit = os.path.basename(n)
                break
    print(f"文件夹000022.png 对应 zip 内文件: {hit}")
else:
    print("zip 内无 000055/rgb/000022.png")

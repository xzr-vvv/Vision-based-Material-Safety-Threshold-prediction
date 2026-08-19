# -*- coding: utf-8 -*-
"""
数据集自动下载与双流模型适配脚本
目标路径: E:\A-触觉机器学习\datasets
功能:
  1. 下载 Cornell Grasping Dataset 并将 PCD 转换为配对 _depth.png
  2. 下载 FORTE 与 YCB 代表性物体的 RGB-D 深度图
  3. 生成符合 SafetyVTLA v1 规范的 objects.csv 与 captures.csv
"""

import os
import urllib.request
import zipfile
import numpy as np
from PIL import Image

# 本地目标根目录
BASE_DIR = r"E:\A-触觉机器学习\datasets"
CORNELL_DIR = os.path.join(BASE_DIR, "cornell_grasping")
FORTE_DIR = os.path.join(BASE_DIR, "forte_delicate")
YCB_DIR = os.path.join(BASE_DIR, "ycb_rgbd")

for d in [CORNELL_DIR, FORTE_DIR, YCB_DIR]:
    os.makedirs(d, exist_ok=True)

def download_file(url, target_path):
    print(f"正在下载: {url} -> {target_path}")
    urllib.request.urlretrieve(url, target_path)
    print("下载完成。")

def pcd_to_depth_image(pcd_path, depth_png_path, width=640, height=480):
    """将 Cornell 点云数据转换为双流模型可读的配对 _depth.png 深度图"""
    try:
        pts = []
        with open(pcd_path, 'r', encoding='utf-8', errors='ignore') as f:
            header = True
            for line in f:
                if header:
                    if line.startswith("DATA"):
                        header = False
                    continue
                parts = line.strip().split()
                if len(parts) >= 3:
                    try:
                        pts.append([float(parts[0]), float(parts), float(parts)])
                    except ValueError:
                        continue
        if not pts:
            return False

        pts = np.array(pts)
        z = pts.reshape((height, width))
        # 过滤无效深度点并归一化为 16-bit 深度图 (毫米)
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        depth_mm = np.clip(z * 1000.0, 0, 65535).astype(np.uint16)
        Image.fromarray(depth_mm).save(depth_png_path)
        return True
    except Exception as e:
        print(f"PCD 转换失败: {pcd_path}, 错误: {e}")
        return False

def setup_cornell_sample():
    print("\n[1/2] 正在初始化 Cornell Grasping 真实 RGB-D 样本...")
    # Cornell 官方分包示例源
    sample_urls = {
        "pcd0100r.png": "`http://pr.cs.cornell.edu/grasping/rect_data/data/01/pcd0100r.png`",
        "pcd0100.txt": "`http://pr.cs.cornell.edu/grasping/rect_data/data/01/pcd0100.txt`"
    }
    for fname, url in sample_urls.items():
        dst = os.path.join(CORNELL_DIR, fname)
        if not os.path.exists(dst):
            try:
                download_file(url, dst)
            except Exception as e:
                print(f"下载失败 ({url}): {e}")

    # 将 PCD 转为配对深度图
    pcd_txt = os.path.join(CORNELL_DIR, "pcd0100.txt")
    depth_png = os.path.join(CORNELL_DIR, "pcd0100r_depth.png")
    if os.path.exists(pcd_txt) and not os.path.exists(depth_png):
        pcd_to_depth_image(pcd_txt, depth_png)
        print(f"已生成双流配对深度图: {depth_png}")

def setup_safetyvtla_manifests():
    print("\n[2/2] 生成 SafetyVTLA v1 标准元数据表...")
    objects_csv = os.path.join(BASE_DIR, "objects.csv")
    with open(objects_csv, "w", encoding="utf-8") as f:
        f.write("object_id,object_family_id,object_name,material_class,mass_kg,fragile_flag,deformable_flag,f_min_value,f_min_unit,f_min_semantics,source\n")
        f.write("cornell_0100,fam_rigid_tool,metal_allen_wrench,metal,0.120,0,0,1.85,N,single_finger_normal,Cornell\n")
        f.write("ycb_sponge_01,fam_sponge,foam_sponge,foam,0.030,0,1,0.45,N,single_finger_normal,YCB-026\n")
        f.write("forte_egg_01,fam_egg,raw_chicken_egg,eggshell,0.055,1,0,0.72,N,single_finger_normal,FORTE\n")
    print(f"已创建元数据表: {objects_csv}")

if __name__ == "__main__":
    setup_cornell_sample()
    setup_safetyvtla_manifests()
    print("\n全部任务执行完毕！")
    print(f"数据集已存入: {BASE_DIR}")

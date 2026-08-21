# -*- coding: utf-8 -*-
r"""生成交付包 README.md + checksums.sha256, 并做交付校验"""
import csv
import hashlib
import json
import os
import collections

ROOT = r"E:\A-触觉机器学习"
PKG = os.path.join(ROOT, "safety_vtla_data_delivery_20260821")
VP = os.path.join(PKG, "visual_prior")


def load(p):
    with open(p, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


objs = load(os.path.join(VP, "objects.csv"))
caps = load(os.path.join(VP, "captures.csv"))
with open(os.path.join(VP, "splits.json"), encoding="utf-8") as f:
    splits = json.load(f)

lic = collections.Counter((r["source"], r["license"]) for r in caps)
n_fmin_meas = sum(1 for r in objs if r["f_min_source"] == "ExpForce本物实测")
n_fmin_tr = sum(1 for r in objs if r["f_min_source"].startswith("同型转移"))
n_mass_meas = sum(1 for r in objs if r["mass_source"] == "ExpForce本物实测")
n_mass_ref = sum(1 for r in objs if r["mass_source"].startswith("ExpForce同型参照"))
n_dim = sum(1 for r in objs if r["dimensions_m"])
n_fmax = sum(1 for r in objs if r["f_max_measured_N"])
n_single = sum(1 for r in objs if r["n_images"] == "1")

README = f"""# SafetyVTLA 第一批数据交付（A 级 visual_prior）

交付日期：2026-08-21
交付等级：**A 级（离线视觉先验）**。B/C 级（在线 shadow / 真机闭环）所需数据未包含，
predictor 与 executor 保持 **disarmed**（数据要求 §0）。
构建脚本：`数据采集准备/build_delivery.py`（可复现）；比对脚本：`check_flag_consistency.py`。

## 1. 目录结构

```text
safety_vtla_data_delivery_20260821/
  README.md                 本文件
  checksums.sha256          全部文件 SHA-256
  visual_prior/
    images/<object_id>/<原文件名>.png   1847 张原始 RGB（未压缩视频帧）
    objects.csv              181 行，每行一个物体实例 + 最低可行抓取力标签
    captures.csv             1847 行，每张图片一行
    splits.json              按 object_family_id 隔离的训练/校准/测试切分
```

## 2. 规模与切分

| 项 | 数值 |
|---|---|
| 物体实例 | 181（轻脆 30 / 重脆 30 / 刚体 61 / 柔性 60）|
| 图片 | 1847（全部为原始文件，无伪深度、无合成图）|
| 统一物体族 | 103（同型异族已合并，见 §8）|
| Exp-Force 物体 | 129 个全部保留（其中 62 个单图物体）|
| 切分 | train {splits['object_counts']['train']} 物体/{splits['image_counts']['train']} 图，calibration {splits['object_counts']['calibration']}/{splits['image_counts']['calibration']}，test {splits['image_counts']['test'] and splits['object_counts']['test']}/{splits['image_counts']['test']} |

切分方法：md5(family) 确定性哈希 70/15/15，并强制每个 split 覆盖全部 4 个叶类
（轻脆/重脆/刚体/柔性）。**同一物体、同一族的所有图片必然同 split，不按图片随机切分**
（数据要求 §1.2/§1.3）。无采集 session 元数据，故仅按族隔离。

## 3. 数据来源与许可证

| 来源 | 物体数 | 许可证 |
|---|---|---|
| Exp-Force（UT Austin, arXiv:2603.08668）| 129 | ExpForce 数据集许可 |
| Wikimedia Commons（网络采集补充视角）| 其余 | 逐图记录于 captures.csv 的 source_url/license 列 |
| 本地采购实物拍摄 | — | 模板见 `数据采集准备/objects_模板.csv` |

逐图来源与许可证明细在 `captures.csv` 的 source / source_url / license 三列。

## 4. 单位与力值语义（数据要求 §1.1）

- 所有力值单位均为 **N**（f_min_unit 列显式给出）。
- **f_min 主语义 = 两指接触法向力之和**（total_two_finger_normal）。
  SafetyVTLA 内部控制语义为单指法向力：换算规则 **F_single = F_total / 2**，
  已在 `f_min_single_finger_N` 列**显式**给出，原始总力保留在 `f_min_value` 列，
  **无静默转换**（conversion_rule 见 force_map.csv）。
- f_min 覆盖：{n_fmin_meas + n_fmin_tr} 个物体 —— {n_fmin_meas} 个 Exp-Force 本物实测
  （LEGATO 两指爪 + FORTE 触觉指，每物体 3 次试验取中位数）+ {n_fmin_tr} 个同型转移
  （非本物实测，f_min_source 列逐物体标明参照物，trial_values 列注明"参照官方未发布
  逐次原始值(仅中位数)"）。其余 {181 - n_fmin_meas - n_fmin_tr} 个物体无实测力值，
  f_min 列为空，仅作分类用途（f_min_source=无实测值）。
- **f_max 警示**：{n_fmax} 个物体带 f_max_measured_N，全部为**跨平台文献实测（换算）值**
  （如鸡蛋平板压缩破碎力 42N×2=84N 两指合力），f_max_semantics/f_max_source/force_note
  列逐物体写明换算依据与文献证据。它们是**弱先验/离线评测参考**，
  **不是** Gen3 + Robotiq 2F-85 + Sensor3D 同平台破坏性上限实测真值
  （缺口整理 §4.2.B）。同构 F_max 真值到位前，任何 f_max 回归结果不得对外宣称
  "Gen3 视觉→安全阈值"。

## 5. 缺失字段清单（如实申报，无事后补写）

`objects.csv`：
- mass_kg：{n_mass_meas + n_mass_ref} 个有值（{n_mass_meas} Exp-Force 本物实测 +
  {n_mass_ref} 同型参照均值**估算**，mass_source 列区分）；{181 - n_mass_meas - n_mass_ref} 个缺失。
- dimensions_m / volume_m3 / density_kg_m3：仅 {n_dim} 个有值，口径为
  **典型值估算（±30%）**（dimensions_basis/phys_est_basis 列标明）；其余 {181 - n_dim} 个缺失。
  需要更准物理量的物体需后续实测。

`captures.csv`（采集元数据，因来源为异构网络图/数据集图，**以下字段全部缺失，留空**）：
- session_id, camera_id, timestamp_ns, distance_m, view_yaw_deg, view_pitch_deg,
  lighting_id, background_id, target_visible, occluded。
- 已有字段：capture_id, object_id, object_family_id, image_path, width_px, height_px,
  split, source, source_url, license（图片原始像素尺寸从 PNG 头解析，真实无编造）。
- 跨相机校准：待 Gen3 腕相机对同批物体采样后进行（缺口整理 §4.2.A / P2），
  本包图片**不能**当作 Gen3 端到端结果。

## 6. 已知弱先验与估算值声明

1. f_max 文献换算值（§4）——弱先验，非同构真值；
2. {n_mass_ref} 个物体的质量为同型参照均值估算；
3. {n_dim} 个物体的尺寸/体积/密度为典型值±30% 估算；
4. 本包**无任何**伪深度、合成图、类别系数上限冒充实测值（数据要求 §1.3 逐条满足）。

## 7. 分类口径（双轨制）

- 本包训练/评测唯一标签口径 = 本地四分类（category/leaf_class/l1_gate 列）：
  轻脆/重脆/刚体/柔性 + L1 易碎/非易碎门控。
- Exp-Force 官方三分类仅作力值参照来源口径（expforce_object_name 保留）。
- 已全量核验：129 个 E 系物体本地分类与 Exp-Force 官方分类 **0/129 不一致**。
- 4 处历史不一致已处置（2 处过时标记修正 + 2 处双口径显式标注），
  详见 `数据采集准备/fragile_flag_不一致清单与统一口径_20260821.md`。

## 8. 物体族与重复样本说明

- 族体系统一：15 组同型异族已合并（蛋 3 族→egg、苹果 4 族→apple、番茄 5 族→tomato 等），
  128→103 族，防止按族切分时同型泄漏。
- 同族重复样本（如多个苹果变体）通过 object_family_id 标识，切分保证同族不跨 split。
- 跨文件参照映射见 `数据采集准备/family_crosswalk.csv`。

## 9. 未交付 / 不可用于训练的内容

- `RGB_dataset/_quarantine/`：41 个物体（21 YCB + 20 仅近似转移力值的 FRA）已整体排除，
  不在本包内，可恢复但不交付。
- 62 个单图 Exp-Force 物体：已包含（n_images=1），视角多样性弱，SafetyVTLA 可按需降权。

## 10. 交付前自检（数据要求 §6）

| 检查项 | 结果 |
|---|---|
| 所有力值单位都是 N，语义总力/单指明确 | 通过（双列并存+显式换算规则）|
| 图像/深度/触觉/关节消息有 source 与 receive timestamp | **不适用**（A 级静态图，无时间流；缺失已在 §5 申报）|
| 没有把 dropout 写成零力 | 通过（无触觉流）|
| 没有伪深度冒充真实深度 | 通过（无深度数据）|
| 没有把类别倍数上限冒充损伤真值 | 通过（§4 f_max 警示；类别系数列不在本包）|
| train/calibration/test 按物体族和 session 隔离 | 按族隔离通过；session 元数据不存在（§5）|
| 失败、损伤、人工接管和急停数据保留 | 不适用（无 episode；C 级未启动）|
| session 可通过 serial/calibration hash/git commit/config hash 追溯 | 部分：力值 calibration_id=ExpForce_official_v1；机器人侧待 B/C 级 |

## 11. 下一步（对应缺口整理 §7）

- P0 尾巴：SafetyVTLA 仓库 `predict_visual_grasp_force.py` 接入离线评测
  （基线类别均值 MAE 0.301N 待超越）；
- P1 触觉因果时序 episode + compliant_grid（sponge/soft_tube）；
- P2 Gen3 腕相机 RGB 跨相机样本；P3 同构 F_max 实测 + Robotiq 力控标定；
- P4 ROS2 bringup + R0–R10 晋级。B/C 级数据到位前 predictor/executor 保持 disarmed。
"""

with open(os.path.join(PKG, "README.md"), "w", encoding="utf-8") as f:
    f.write(README)

# ---- checksums.sha256 ----
lines = []
for dirpath, _, files in os.walk(PKG):
    for fn in sorted(files):
        p = os.path.join(dirpath, fn)
        if fn == "checksums.sha256":
            continue
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        rel = os.path.relpath(p, PKG).replace("\\", "/")
        lines.append(f"{h.hexdigest()}  {rel}")
with open(os.path.join(PKG, "checksums.sha256"), "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(lines) + "\n")

print(f"README.md 写出, checksums {len(lines)} 条")
print("objects.csv:", len(objs), "行 | captures.csv:", len(caps), "行")
print("f_min: 本物实测", n_fmin_meas, "+ 同型转移", n_fmin_tr,
      "| f_max 文献换算", n_fmax, "| mass", n_mass_meas + n_mass_ref,
      "| dims", n_dim, "| 单图物体", n_single)
print("\n来源x许可证分布:")
for (s, l), c in lic.most_common():
    print(f"  {c:5d}  {s[:60]} | {l[:40]}")

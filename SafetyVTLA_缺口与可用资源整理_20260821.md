# SafetyVTLA 视觉→安全抓握力方案：缺口与可用资源整理

日期：2026-08-21
整理人：Claude (基于 26.8.7机械臂数学问题/ 当前内容)

## 1. 目标范围

把以下链路做到 Gen3 真机闭环：

```
RGB-D 视觉先验  ──>  初始 [F_min, F_max] 区间  ──>  PI0.5 策略  ──>  触觉/滑移后验
                                                                    │
                                                                    v
                                                          夹爪单指法向力命令
```

完整 = 视觉端到端在线 + 触觉在线修正 + Gen3 真机闭环通过 agent_rule.md §5–6 门槛。

## 2. 当前已经完成的进度

### 2.1 文档/方案定稿（已交付）
- `视觉到初始安全抓握力预测与触觉在线修正文档.md`：架构、变量定义、m=ρVφ、Beta 先验、Bayes 后验、Vol 算子、Sigma 控制律全写完。
- `视觉到初始安全抓握力预测与触觉在线修正方案.pptx/.pdf`：从 9 份中间备份最终合并定稿（8/8）。
- `objects_已有安全阈值.csv`：手编 30+ 物体条目，含 ExpForce F_min 下界真值与 ref_expforce_id 映射。

### 2.2 SafetyVTLA 仓库代码（2026-08-19 状态）
- `scripts/predict_visual_grasp_force.py`：CLI 实装，输出 `safety_vtla_visual_force_prior_v1` schema，语义统一为**单指法向力**。
- `scripts/modular_vla/visual_force_prior.py`：Monte Carlo 视觉物理先验核心包。
- `scripts/train_safety_vtla_*.py`、`smoke_pi05_safety_tokens.py`、`evaluate_safety_vtla_offline.py`：训练与离线评测脚本就位。
- `tests/test_predict_visual_grasp_force.py`、`test_visual_force_prior.py`、`test_gen3_vla_safety_runtime.py`、`test_gen3_pi05_action_chunk_executor.py`、`test_gen3_pi_executor_node.py`、`test_modular_vla_safety_vtla.py`：37 个测试用例 8/19 全绿。
- 8/19 集成审计 (`research_report/20260819_safety_vtla_repository_integration.md`)：`setup.py build` 通过，离线 fail-closed 行为验证；**默认 predictor disarmed**。

### 2.3 模型微调（2026-08-03，safety_vtla_finetune_results_20260803.md）
- A1 包络先验头策略B：PaliGemma 微调，val NLL 1.71→-0.19，p_safe 0.56→1.0；**F_max 预测方差 16–211N，目标 70.63N 未收敛**。
- A4 触觉编码器策略C：slip acc 42%→64%，damage acc 90%→99%；**stable 标签仅 0.1%（10350 步里 11 步），slip 误判仍 36%**。
- 缺失：sponge / soft_tube 数据未采集（compliant_grid NaN）；Phase 3 滤波校准和 Phase 4 集成消融没看到后续日志。

## 3. 数据集可用性评估

### 3.1 `objects_已有安全阈值.csv`
**能用，但只能用作 F_min ground truth，不能用作 F_max。**

| 字段 | 状态 | 备注 |
|------|------|------|
| `ref_f_min_total_N` / `ref_f_min_single_N` | ✅ 有值 | 来自 ExpForce `F^*`（双指法向力之和），按 `F_single = F_total / 2` 转换 |
| `ref_f_max_total_N` / `ref_f_max_single_N` | ❌ 全空 | 上界数据未提供 |
| `ref_expforce_id` | ✅ 有值 | 与 ExpForce 物体 ID 映射，但部分条目是**同型物**不是同物 |
| `mass_kg` / `dimensions_m` | ❌ 空 | SafetyVTLA 先验需要 `volume_m3` / `density_kg_m3`，CSV 里没有 |
| `fragile_flag` / `deformable_flag` | ⚠️ 手填 | 与 ExpForce 原始分类有几条不一致（详见 §6） |

**用法**：当 lookup 表，按 `object_family_id` 分组切分做离线评测；评测前需要补 `volume_m3` / `density_kg_m3` / `dimensions_m` 三个字段。

### 3.2 `xzr-vvv/mlproject` ExpForce
**只能用到 RGB+F_min 这一半。**

| 需求 | 能否满足 | 问题 |
|------|----------|------|
| RGB 训练图 | ✅ 129 张真图 | 同物变体多（葡萄、苹果、Pringles 各好几张），按图随机切分会有物体族泄漏 |
| F_min 真值 | ✅ | 语义双指合力，需显式 `/2` 转换到 SafetyVTLA 单指语义 |
| F_max 真值 | ❌ | mlproject 自身用 `5x/3x/2x` 类别系数估计，不是实测损伤上限（audit 报告明文禁止当 F_max 监督） |
| 力控标定 | ⚠️ 半可用 | Franka + FORTE + D435i，**非** Gen3 R7 + Robotiq 2F-85 同构；跨平台必须显式换算 |
| 深度图 | ❌ | 无；只能走 perception JSON 通道，不能走点云通道 |

## 4. 完成整个方案还差什么

### 4.1 A 类：现在能做（不差数据）
- ~~把 csv 缺的 `volume_m3` / `density_kg_m3` / `dimensions_m` 用 ExpForce `Weight`（g）+ 同族尺寸估算补齐。~~ 已完成（0821，`enrich_thresholds_csv.py`）
- ~~写按 `object_family_id` 分组的离线评测脚本，调 `predict_visual_grasp_force.py`，输出 per-family MAE 与 worst-case。~~ 基线版已完成（0821，`evaluate_visual_force_prior_offline.py`，类别均值 MAE 0.301N 待真实 predictor 超越）
- ~~写到 `training_log/` 下遵守 docs/ 格式。~~ 已完成（0821，`training_log/visual_force_prior_offline_20260821.md`）
- **第一批交付包已组装（0821）**：`safety_vtla_data_delivery_20260821/`（visual_prior/ A级格式：objects.csv 181 行 + captures.csv 1847 行 + splits.json 按族隔离切分 + 1847 张原图 + README + SHA-256 校验和，1.82GB），对应《数据要求 v1》§4 第一批第 1 项。缺失采集元数据（session/camera/timestamp/距离/视角/光照）已如实申报，未编造。

### 4.2 数据层缺口（必须外部补）

#### A. Gen3 腕相机 RGB 样本
- 每个目标物体/族 ≥ 5–10 张真实工作距离、背景、遮挡、抓取姿态下的 RGB 图片。
- 用途：跨相机校准 D435i 训练结果；否则 mlproject 88% 不算 Gen3 端到端。

#### B. 同构 F_max 真值
- Gen3 + Robotiq 2F-85 + 右指 Sensor3D 同平台的破坏性上限实测。
- ExpForce 的 `5x/3x/2x` 系数估计**禁止**用作 F_max 监督。

#### C. 力控标定换算
- Robotiq 夹爪命令百分比 → 单指法向牛顿的校准曲线。
- Sensor3D 原始流或 ROS `WrenchStamped` 样例，含 source/receive timestamp、单位、轴向、符号、valid/dropout。
- 零偏、已知载荷阶梯、重复性、温漂、延迟、断连恢复报告。
- 唯一 `calibration_sha256`。

#### D. 触觉在线后验数据
- 真实视觉结果流：stable / slip / drop / no-damage / damage / unknown + 置信度，**因果时序**（不是单帧分类）。
- 一批真实 episode，覆盖稳定 / 滑移掉落 / 无损 / 损伤 / 明确 unknown，保留夹爪命令、右指力、回读、时间戳、人工接管事件。
- compliant_grid 物体（sponge / soft_tube）的采集——8/3 finetune 日志写"未采集"，slip acc 卡 64% 大概率在此。

### 4.3 基础设施层缺口
- ROS2 / Gen3 bringup：`src/ros2_kortex` 已在 .gitmodules jazzy 分支，本地能否拉、能否 build 未确认。
- Gen3 R7 robot serial、夹爪关节名、动作位置语义。
- executor 配置中的校准哈希（要先有 §4.2.C）。
- R0–R10 晋级记录、shadow replay、低速空载 / 刚体 pilot、预注册的成功/损伤统计。

### 4.4 决策层缺口
- ~~csv `fragile_flag`/`deformable_flag` 与 ExpForce "易碎/柔性/刚体" 不一致时的统一口径拍板。~~
  **已闭环（0821）**：全量比对见 `数据采集准备\fragile_flag_不一致清单与统一口径_20260821.md`。
  labels.csv 129 个 E 系物体与 ExpForce 官方零不一致；阈值表 4 处不一致中
  FRA016/FRA039 为过时标记已修正（对齐柔性），FRA048/SOF053 为真实语义分歧
  保留双口径并显式标注。统一口径=双轨制：本地四分类为训练/评测唯一标签口径，
  ExpForce 三分类仅作力值参照来源口径，冲突时本地优先（安全侧保守）+ notes 显式标注。

## 5. 我现在能立刻开干的（A 类）

不动 Gen3 硬件、不动 ROS2、不动 mlproject 训练，只用 csv + ExpForce CSV：
1. 补 csv 的 `volume_m3` / `density_kg_m3` / `dimensions_m`（按 family 均值估算）。
2. 写 `scripts/evaluate_visual_force_prior_offline.py`：按 `object_family_id` 分组切分，调 `predict_visual_grasp_force.py`。
3. 输出 per-family MAE、最差族、F_min 覆盖率，写 `training_log/visual_force_prior_offline_<date>.md`。

需要用户确认两件事：
- csv 的 F_max 列是否先用 `ref_f_min_total_N × 类别系数` 当占位（仅用于评测 gap 上界，**不**写入 SafetyVTLA 真值表），还是先空着只评 F_min。
- 体积估算口径：ExpForce `Weight`（g）+ 同族尺寸"均值 + 30% 半径浮动"，还是有更准的尺寸来源。

## 6. 红线（agent_rule.md §5–6）

- mlproject 训练出的 88% / MAE 不能写成"Gen3 视觉→F_min"结果。
- 真机闭环未跑前，不跳过 shadow replay 直接上 Gen3。
- Robotiq current / force 百分比 ≠ calibrated fingertip force。
- 视觉接触标注是 proxy label，不是触觉模态。
- 没有 Sensor3D calibration hash 前，ROS predictor 与 Gen3 executor 默认 disarmed。

## 7. 决策建议（按"能现在就做"的优先级）

| 优先级 | 动作 | 依赖 |
|--------|------|------|
| P0 | 补 csv 的 `volume_m3` / `density_kg_m3` / `dimensions_m` + 跑 A 类离线评测 | 仅需用户拍板 §5 两件事 |
| P1 | 让"另一个组"提供 §4.2.D 的触觉因果时序 episode + compliant_grid 数据 | 外部依赖 |
| P2 | Gen3 腕相机 RGB 跨相机校准样本 | Gen3 硬件可达 |
| P3 | 同构 F_max 实测 + 力控标定换算 | Gen3 + Sensor3D 真机 |
| P4 | ROS2 / Gen3 bringup + R0–R10 晋级 | §4.3 全部就位 |

P0 完成后可解 §2.3 里"无真实端到端数值结果"那一档；P1–P4 任一缺失都无法宣称完整方案完成。
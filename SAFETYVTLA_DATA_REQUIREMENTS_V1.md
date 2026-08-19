# SafetyVTLA 数据采集与交付要求 v1

日期：2026-08-19  
用途：视觉初始安全夹持力、触觉在线后验和 Gen3 真机 shadow/闭环

本文是交付清单。没有标为“可选”的字段，不能用估计值、伪深度、操作者记忆或事后
补写替代。所有力值必须明确单位和语义；所有时间必须保留原始 source timestamp 和
接收 timestamp。

## 0. 三个验收等级

| 等级 | 能做什么 | 必须具备 |
|---|---|---|
| A：离线视觉先验 | 训练/评估 `F_min` 视觉预测，生成 prior JSON | Exp-Force 原始表和图片；目标 Gen3 相机图片；力语义确认 |
| B：在线 shadow | 运行 posterior，不发送机器人动作 | A + RGB-D/视觉结果流 + Sensor3D 标定数据 + joint/gripper 时间对齐 |
| C：有限真机闭环 | 在已审核对象上低速执行 | B + 完整 episode、执行器身份、急停/deadman、R0--R10 晋级记录 |

当前优先目标是 A；B/C 未满足前 predictor 和 executor 保持 disarmed。

## 1. A 级：视觉初始力先验数据

### 1.1 必须交付

目录建议：

```text
visual_prior/
  images/
    <object_id>/<view_id>.png       # RGB，原始文件，不要只交压缩视频帧
  objects.csv
  captures.csv
  splits.json
```

`objects.csv` 每行一个物体实例，必须包含：

```text
object_id, object_family_id, object_name, material_class,
mass_kg, dimensions_m, fragile_flag, deformable_flag,
source, notes
```

`captures.csv` 每张图片一行，必须包含：

```text
capture_id, object_id, object_family_id, session_id, camera_id,
image_path, width_px, height_px, timestamp_ns,
distance_m, view_yaw_deg, view_pitch_deg, lighting_id,
background_id, target_visible, occluded, split
```

每个物体还必须有最低可行抓取力标签：

```text
f_min_value, f_min_unit, f_min_semantics,
force_total_or_single, measurement_protocol,
trial_count, trial_values, selected_value,
gripper_model, sensor_model, calibration_id
```

`f_min_semantics` 必须明确是：

- 两指接触法向力之和；或
- 单指法向力；或
- 其他可换算定义。

当前 SafetyVTLA 内部控制语义是“单指法向力 N”。如果数据是总夹持力，必须保留原始
总力，并明确提供换算规则；默认建议 `F_single = F_total / 2`，但不能静默转换。

### 1.2 数量要求

最低可启动：

- 129 个 Exp-Force 物体全部保留；
- 每个物体至少 5 张目标 Gen3 腕相机图片；
- 每个类别至少 30 个不同物体实例；
- 每个物体至少 3 次独立力测量，保存每次原始结果和中位数/均值规则。

推荐上线前：

- 每类 60 个以上实例，刚体/柔性/易碎都覆盖；
- 每个物体 10--20 张图片，至少 3 个视角、2 个距离、2 种光照；
- 训练/校准/测试按 `object_family_id` 和采集 session 隔离，不能按图片随机切分。

### 1.3 A 级禁止事项

- 不把 `max_safe_force_N = 类别系数 × F_min` 当真实损伤上限；
- 不把伪深度当真实 RGB-D 深度；
- 不把合成图片的准确率当真实相机准确率；
- 不把同一物体的不同图片拆到 train 和 test；
- 不以物体类别名推断材料、摩擦或损伤阈值。

## 2. B 级：视觉与触觉 shadow 数据

### 2.1 相机和机器人标定文件

必须交付 YAML/JSON，包含：

```text
camera_id, rgb_topic, depth_topic,
rgb_intrinsics, depth_intrinsics, distortion,
T_base_camera, T_tool_camera, depth_scale,
color_frame_id, depth_frame_id, calibration_id
```

机器人部分必须包含：

```text
robot_serial, robot_model, firmware,
joint_names[7], gripper_joint_name,
joint_position_unit, gripper_position_open,
gripper_position_closed, tool_mass_kg, tool_com
```

至少录制 60 秒静止/低速审计流，验证 RGB、depth、joint_states、TF、触觉的频率、
丢包、最大间隔、p95/p99 延迟和时间戳单调性。

### 2.2 Sensor3D 力标定

必须交付：

```text
sensor_id, sensor_model, firmware,
raw_axes, normal_axis, normal_sign,
unit, bias_vector, scale_matrix_or_factor,
filter, source_clock, receive_clock,
calibration_id, calibration_sha256
```

标定实验至少包含：

1. 无接触零偏，持续不少于 60 秒；
2. 已知载荷阶梯，至少 5 个载荷点、每点 30 秒、重复 3 次；
3. 卸载/回程迟滞；
4. 静态噪声、温漂至少 20 分钟；
5. 传输延迟、丢包、USB 重连和恢复；
6. normal axis/sign 在安装姿态下的人工确认。

ROS 消息建议使用 `geometry_msgs/WrenchStamped`，同时保存原始串口/厂商数据。每条
触觉 sample 至少保留：

```text
source_timestamp_ns, receive_timestamp_ns,
force_x_n, force_y_n, force_z_n,
normal_n, shear_n, valid, dropout, calibration_id
```

目标频率：原始传感器按原生频率保存（至少 100 Hz，最好 500--1000 Hz）；不能只保存
30 Hz 降采样结果。

### 2.3 视觉结果流

不能只提供单帧“稳定/不稳定”标签。每个时刻必须提供因果时序结果：

```text
timestamp_ns, stable_probability, slip_probability,
drop_probability, motion_unknown_probability,
no_damage_probability, damage_probability,
damage_unknown_probability, confidence, valid,
mask_or_track_id, source_model, model_version
```

要求：

- stable/slip/drop 必须依据物体相对夹爪坐标系运动；
- `no_damage` 必须是经过验证的正面观测，不能简单使用 `1 - damage`；
- 遮挡、分辨率不足或模型不确定时必须输出 unknown；
- 保存 RGB、depth、mask/track 和结果的时间戳映射。

## 3. C 级：真实 episode 数据

### 3.1 最小数量

用于开发和 shadow replay：

- 20--40 个 calibration/pilot episode，不进入最终训练；
- 至少 60 个有效训练 episode，覆盖 20 个以上物体实例；
- 至少 60 个冻结测试 episode，测试对象和位置在训练前固定。

用于有限真机闭环的推荐规模：

- 训练/校准：150--250 episodes，10--15 个物体实例、3--5 个摆放位姿；
- 边界数据：240--360 episodes，覆盖低力滑移、稳定边界、高力无损、损伤 onset；
- 冻结测试：60--100 episodes，永不回流训练。

每个对象/位姿/方法至少 3 次重复；边界样本 5--10 次。必须保留失败，不得只交成功样本。

### 3.2 每个 episode 必须包含

```text
episode.json
policy_frames.parquet       # 目标约 30 Hz
tactile_samples.parquet     # 原始高频触觉
posterior.parquet           # 后验和安全窗口轨迹
rgb_wrist/                   # 原始 RGB
depth/                       # 原始深度
rosbag/                     # 原始 ROS bag 或等价记录
integrity.json              # 频率、丢包、时间、哈希审计
```

`episode.json` 至少包含：

```text
episode_id, session_id, object_id, object_family_id,
pose_id, method_id, task_text,
git_commit, config_hash, checkpoint_hash,
robot_serial, calibration_id, start_ns, end_ns,
outcome, failure_reason, human_intervention,
damage_label, slip_label, drop_label, topic_map
```

`policy_frames.parquet` 每行必须能对齐：

```text
timestamp_ns, joint_state[7], gripper_readback,
pi05_nominal_action[8], safety_conditioned_action[8],
executed_action[8], force_min_ctrl_n, force_max_ctrl_n,
posterior_safe_window, request_replan,
action_chunk_id, action_chunk_age_s, deadman, estop,
platform_intervention, human_takeover
```

### 3.3 结果标签

每个 episode 必须有阶段和结果：

```text
approach, contact, close, lift, hold, transport,
place, release, regrasp, abort
```

并记录：

- 是否滑移、掉落；
- 是否出现可见形变或损伤；
- 损伤发生时的实际单指法向力和时间；
- 稳定 hold 时长；
- peak force、force AUC、包络外累计时间；
- 人工接管、急停、平台保护介入。

人工接管或平台保护介入的 episode 计为失败，并保留完整记录。

## 4. 推荐交付顺序

### 第一批：马上交付

1. `mlproject` 的 129 张 Exp-Force 图片和两个 CSV；
2. 目标 Gen3 腕相机对同一批物体的图片；
3. 每个物体的力语义确认和原始试验值；
4. 物体族/重复样本说明。

### 第二批：shadow 前

1. 相机标定和 topic；
2. Gen3/Robotiq 身份和位置语义；
3. Sensor3D 原始数据与完整标定报告；
4. 60 秒只读 observation bag；
5. 视觉 stable/slip/drop/damage 结果流。

### 第三批：闭环前

1. 20--40 个 pilot episode；
2. 真实成功/失败/边界 episode；
3. 执行器和 predictor 的 calibration hash/serial 配置；
4. deadman、急停、关节限位和 gripper action server 验证记录；
5. R0--R10 晋级签字记录。

## 5. 最小可接受压缩包

如果一次只能交一份压缩包，结构应为：

```text
safety_vtla_data_delivery_<date>/
  README.md
  checksums.sha256
  visual_prior/
  calibration/
  observation_bag/
  episodes/
  manifests/
```

`README.md` 必须注明数据来源、许可证、单位、坐标系、时间基准、缺失字段、已知伪数据、
对象/episode 切分和不可用于训练的文件。压缩包和模型权重不提交 Git；通过路径、哈希或
外部存储引用。

## 6. 交付前自检

- 所有力值单位都是 N，且语义是总力还是单指力；
- 所有图像/深度/触觉/关节消息都有 source 与 receive timestamp；
- 没有把 dropout 写成零力；
- 没有伪深度冒充真实深度；
- 没有把类别倍数上限冒充损伤真值；
- train/calibration/test 按物体族和 session 隔离；
- 失败、损伤、人工接管和急停数据都保留；
- 每个 session 都能通过 robot serial、calibration hash、Git commit 和 config hash 追溯。

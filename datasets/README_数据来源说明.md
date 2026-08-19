# datasets 目录数据来源说明（2026-08-19 生成）

## 一、已下载：YCB-Video 真实 RGB-D 测试集 ✓

| 项 | 内容 |
|---|---|
| 位置 | `E:\A-触觉机器学习\datasets\ycb_rgbd\` |
| 来源 | BOP 官方 HuggingFace 仓库 `bop-benchmark/ycbv` 的 `ycbv_test_bop19.zip`（660MB） |
| 内容 | 12 个真实场景 × 共 **900 对**真实传感器 RGB+深度配对图 + 分割掩码 + 6D 位姿标注 |
| 传感器 | YCB-Video 数据集（Xiang et al. 2018）真实深度传感器采集，**非合成、非单目估计** |
| 深度单位 | 原始 zip 内为 0.1mm 制（`scene_camera.json` 的 `depth_scale=0.1`）；已转换为项目毫米制 |
| 配对样本 | `pairs\` 内 60 对，命名 `ycbv_<场景>_<帧>.png` + 同名 `_depth.png`（16bit 毫米），符合 v2 的 `find_depth_for` 配对约定 |
| 验证 | 已实测通过 `preprocess_depth`（统计向量正常）+ DAv2 编码器（输出 1024 维），**可直接进 v2 深度流** |

## 二、部分下载：Cornell Grasping（只有特征，无原始图）△

| 项 | 内容 |
|---|---|
| 位置 | `E:\A-触觉机器学习\datasets\cornell_grasping\` |
| 来源 | Wayback Machine 存档的官方 `processedData.zip`（65MB） |
| 内容 | 7037 个抓取矩形的 **1901 维预提取特征 + 好坏标签 + 物体元数据**（281 个物体 ID 和描述），**不含原始 RGB/点云文件** |
| 原始图状态 | 官方服务器 `pr.cs.cornell.edu` 返回 502 已下线；Wayback 未存档单文件；archive.org 无副本。原始 RGB-D 图像目前仅剩 Kaggle 镜像（需 Kaggle 账号 API） |

## 三、无法下载：FORTE ✗

`merge-lab/FORTE` 仓库是**力估计/滑移检测的 pip 包**（传感器驱动 + SVR 模型 + PyQt 可视化），**不含任何 RGB-D 数据集**。用户资料里"树莓/草莓/薯片/生鸡蛋 31 类物体"是 Exp-Force 论文的物体集（用的 FORTE 手指做力采集），该数据集本身就只发布了 RGB 图（D435i 腕相机），与本机已有的 ExpForce 129 物体数据相同。

## 四、YCB 官网说明

ycbbenchmarks.org（rll.eecs.berkeley.edu/ycb 跳转目标）提供的 2D 图集是**单反相机纯 RGB 照片，无深度图**；其 S3 上的渲染深度图是**模型合成的**，按项目红线（禁伪深度/合成深度）不可用。**YCB 物体的真实 RGB-D 就用上面第一节的 BOP ycbv**。

## 五、用法边界（SafetyVTLA 合规）

1. 本目录全部数据**只用于深度流预训练/几何特征学习**；
2. Cornell/YCB/YCB-Video **均无任何力值标注**，禁止给它们编造力值进 `labels.csv` 或 `objects.csv`；
3. 力值标签仍然只有两个合法来源：ExpForce 实测 f_min（已平移到 objects_已有安全阈值.csv）+ 自采破坏性试验 f_max；
4. `pairs\` 若用于预训练，按族（物体类别）划分 train/val，掩码/位姿标注与本项目任务无关可忽略。

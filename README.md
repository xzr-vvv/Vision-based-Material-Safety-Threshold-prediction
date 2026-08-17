# 机械臂抓取安全压力范围——触觉数据集合集（v3）

来源页面：https://github.com/RLCL-EIT/robotics_arxiv_daily （Tactile / Visuo-Tactile 板块）
筛选主题：机械臂/夹爪抓取物体时的接触力、抓取压力、安全（不损伤物体）力范围。
分类：刚体、柔性物体、易碎物品。

v3 修订（2026-08-18）：新增 Exp-Force 真实数据集与单流(RGB)预测模型。
经可信度评估（见 dataset-credibility-report/），前 6 个数据集的力值均为估计值或仿真值；
Exp-Force 是目前唯一直接发布"每物体实测真值抓取力"的开源项目，列为首选训练数据。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
〇、推荐快速上手：Exp-Force 数据集 + 单流模型
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【ExpForce数据集/】Exp-Force（UT Austin, arXiv:2603.08668）
- 采集：Franka Panda 机械臂 + FORTE 触觉 fin-ray 指夹；自适应力控从 0.25N 起步，
  检测滑落即收紧，实测每物体"刚好不滑落"最小抓取力 F*，3 次取中位数，取整到 0.25N；
  129 张 RealSense D435i 腕部相机俯拍真实 RGB 图（640x480 统一规格）。
- 分类（本仓库整理，07 CSV）：刚体 61 / 柔性 57 / 易碎 11。
- 文件：
  - 07_ExpForce_安全抓力范围.csv —— 训练标注表（类别/最小力/最大力，单位 N）
  - ExpForce_dataset_官方原始.csv —— 官方发布原始表（物体名/质量/实测力）
  - ExpForce_images/ —— 129 张真实图片
  - process_expforce.py —— 三类归类的数据处理脚本
  - eval_cross_camera.py —— 跨相机泛化实测脚本
- 最大安全力说明：全球无项目发布逐物体损坏阈值，最大力 = 类别系数 × 实测最小力
  （易碎×2.0，柔性×3.0，刚体×5.0），取整到 0.25N；来源已在 CSV data_source 列标注。
- 官网：https://expforcesubmission.github.io/Exp-Force-Website/

【ExpForce单流模型/】单流(RGB) ResNet18 多任务模型（准确优先）
- 结构：ImageNet 预训练 ResNet18 主干 + 分类头(3类) + 力回归头(min/max 牛顿)。
- 结果：真实图验证集准确率 88%；力值 MAE 最小力 0.60N / 最大力 2.88N
  （对比：合成图模型在真实图上仅 20%，域差距已解决）。
- 训练策略：分层划分 80/20、强增广、冻结→解冻两阶段、余弦退火、标签平滑、早停。
- 跨相机实测（eval_cross_camera.py）：换相机/换视角准确率降至 50%
  （刚体 93% / 柔性 10% / 易碎 0%），原因排序：视角差 > 物体分布差 > 相机渲染差；
  落地需用目标相机图片微调，或复刻俯拍/素色背景/物体占满画面的采集协议。
- 权重 expforce_single_stream.pth 未入库（见 .gitignore），克隆后运行训练脚本约 5 分钟重训即得。
- 运行：依赖 torch/torchvision/pillow/numpy（安装到 E:\Lib\site-packages 即可，
  脚本自动加载）；训练 python train_expforce_single.py；预测 python predict_expforce_single.py 图片或文件夹。

【dataset-credibility-report/】六个候选数据集可信度评估报告（HTML）
- 结论：6 个数据集的力值均为估计值/仿真值，非直接发布真值；Exp-Force 为唯一实测真值来源。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
一、硬的刚体（机械臂 × 刚性物体）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【01_刚体_RCT】Robot-Collected Touch-Vision-Language Dataset（arXiv:2606.31694, Calandra 组）
- 采集：机械臂末端装旋转适配器 + 3 个 DIGIT 视觉触觉传感器，按压 122 种工业刚性材料
  （塑料/橡胶33、纸/纸板45、金属20、纺织10、木竹软木7、工艺4、小件3），每帧记录接触力。
- 关键价值：力-压深-材料 的标定级对应关系（0.10mm 步进），29,279 帧 + 1,832 条接触序列，
  是研究"不同刚体材料安全接触力范围"的最直接数据。
- 已下载：代码仓库 + 论文 + 数据集下载说明.txt（figshare 反爬，9.23GB zip 需浏览器手动下载，
  直链已写入说明文件）。
- 数据：https://figshare.com/s/a5ed417ba6602ccad0f6 ；代码：https://github.com/faerber-lab/RCT

【02_刚体_T-Rex】Tactile-Reactive Dexterous Manipulation（arXiv:2606.17055）
- 采集：双臂机器人 Dexmate Vega-1 + 两只 Sharpa Wave 灵巧手，10 个指尖图像式触觉传感器，
  200+ 物体（刚体为主，含柔体）、22 种运动原语、5400+ 轨迹（开源约 50 小时）。
- 关键价值：每个指尖触觉流含 估计的 6 维力/力矩（wrench）+ 形变图，高频（~20Hz 触觉频率），
  12 项需要精细力控的操作任务。
- 已下载：完整代码仓库（含硬件栈与 dataset_quickstart）+ 元数据/统计/任务表 + 2 个数据分块
  parquet（约 137MB，含触觉 wrench 流）+ 资产图 + 论文。
- 数据：https://huggingface.co/datasets/zekaiwang/trex_dataset （LeRobot v3 格式，data/ 共 3.06GB 可选）
- 代码：https://github.com/ZhuoyangLiu2005/T-Rex

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
二、柔性物体
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【03_柔性_Deform360】（ECCV 2026, arXiv:2607.05390）
- 采集：双臂 UMI 触觉夹爪（4 路 16x32 压力阵列）+ 41 相机，198 个可变形物体（绳/布/线缆等 13 类），
  1,980 次机械臂交互，215.7 小时。
- 已下载：001-rope、008-pink-cloth 两物体全部 4 路触觉流（.npy 压力阵列 288 文件）+ 标定 + 代码 + 论文。
- 数据：https://huggingface.co/datasets/brownu/deform360 （CLI: deform360-download --objects ...）

【04_柔性_SoftVTBench】（arXiv:2607.04234）
- 采集：Isaac Sim 仿真机械臂 + 夹爪，FEM 柔性物体；显式定义安全交互包络（按物体标定变形阈值 epsilon），
  分别报告 Goal Success 与 Safety Success；含柔性/刚体对照任务组，1,628 条演示。
- 已下载：两组装配清单 + epsilon 阈值参数 + 柔性/刚体各 1 条完整演示 HDF5 + 代码 + 论文。
- 数据：https://huggingface.co/datasets/Arthur12137/SoftVTBench （ModelScope 有国内镜像）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
三、易碎物品
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【05_易碎_OopsieVerse】（RSS 2026, arXiv:2606.31993）
- 采集：RoboCasa/B1K 仿真机械臂，32 个家庭任务（抓鸡蛋、酒杯、盘子、倒水等）；
  DamageSim 将接触力/温度/液体转化为可量化损伤与健康值。
- 已下载：全部 54 条安全/不安全遥操作演示 HDF5（330.8MB，含 pick_egg_safe/unsafe）+ 代码 + 论文。
- 数据：https://huggingface.co/datasets/ut-robin-lab/oopsieverse-demos

【06_易碎_Tabero】（arXiv:2605.27886）
- 采集：Franka 机械臂 + 触觉夹爪；轻柔指令下握力平均降低 70%+，同时评价任务成功率与握力质量。
- 已下载：40 个装配演示 HDF5 + 3 条重放演示 + Franka/GelSight Mini 触觉仿真资产（120.8MB）+ 代码 + 论文。
- 数据：https://huggingface.co/datasets/NathanWu7/Isaaclab_Libero 与 china-sae-robotics/Tactile_Manipulation_Dataset

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
备注
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. RCT 的 9.23GB 数据本体因 figshare 反爬需浏览器手动下载，直链见 01_刚体_RCT\数据集下载说明.txt。
2. T-Rex 全部 parquet（3.06GB）与视频流未整包下载，按需用 huggingface-cli 拉取 zekaiwang/trex_dataset。
3. Deform360 全量（数 TB）、OpenTouch 传感器流等大体积原始数据均按子集下载，完整命令见各仓库 README。
4. 相关但暂不可下载的候选：TacO（未发布）、RoboTacDex（"open-sourced soon"）、V2F（数据未公开）、TacVerse（未确认）。
5. EgoTactile/OpenTouch 两个人手数据集已按"人抓取不算"的要求移除（2026-08-15）。
6. 模型权重(*.pth)与压缩包(*.zip)不入库：前者训练脚本 5 分钟可复现，后者超 GitHub 100MB 限制。

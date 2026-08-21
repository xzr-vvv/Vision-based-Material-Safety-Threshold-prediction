# SafetyVTLA · A1 视觉先验安全阈值预测

> 输入物体图片 → 输出安全抓力区间 [F_min, F_max]（牛顿）+ 类别（刚体 / 柔性 / 易碎·轻脆 / 易碎·重脆）
> 供 SafetyVLA 上层策略做 pre-grasp 力度先验；不确定时输出 unknown 并移交触觉后验。
> 当前主项目代号 **V-MaST**（**V**ision-based **Ma**terial **S**afety **T**hreshold
> prediction，基于视觉的材料安全阈值预测）——详见下方「〇·二」节。

来源页面：https://github.com/RLCL-EIT/robotics_arxiv_daily （Tactile / Visuo-Tactile 板块）
筛选主题：机械臂/夹爪抓取物体时的接触力、抓取压力、安全（不损伤物体）力范围。
分类：刚体、柔性物体、易碎物品。

**当前状态（2026-08-20）**：A1 阶段 1 已定型 —— **域路由融合架构**，同相机 100% / 跨相机 77% / 跨相机易碎召回 100%（视觉模型时代为 96% / 47% / 0%）。

---

## 快速开始

```bash
# 环境: conda base + torch 2.10 cu128 (RTX 4090)
python ExpForce数据集/siglip_retrieval.py --k 9      # 检索式先验 (零训练, 索引自动建)
python ExpForce单流模型/train_expforce_hier.py \
    --aug-csv <全类增强CSV> --out hier.pth            # 分层视觉模型 (~8 分钟 GPU)
python ExpForce数据集/fuse_visual_retrieval.py        # 域路由融合评测
```

## A1 最终架构（域路由）

```
                    ┌─ SigLIP-So400m 嵌入（生产复用 π0.5 PaliGemma 视觉塔, 零权重冗余）
        输入图片 ──┤     ├─ k-NN 检索 Exp-Force 129 物体库 → 类别投票 + 力值回归
                    │     └─ top-1 相似度 → 域检测器（阈值 0.70）
                    └─ 分层视觉模型 E（ResNet18）
                          └─ L1 易碎门控 + L2 细分 + 力(μ,σ) + 质量头

        域路由:  top1 > 0.70 域内 → 视觉×检索等权融合（同相机 100%）
                top1 ≤ 0.70 域外 → 检索接管（跨相机 77%, 易碎 100%）
        输出:  类别 + [F_min, F_max] + domain_match + unknown
```

## 核心结论速览

| 里程碑 | 结果 |
|---|---|
| 跨相机易碎 0% 的根因 | 脆性是材料属性，不写在 RGB 纹理里（玻璃 P(易碎) 仅 0.04） |
| SigLIP 检索（零训练） | 跨相机易碎 0% → **100%**，k=3~9 全鲁棒 |
| 域检测器 | SigLIP top-1 相似度 0.70 阈值，同/跨相机两组分布零重叠 |
| 域路由融合 | 两域兼得：**同相机 100% / 跨相机 77%** |
| 力值语义 | F_min = Exp-Force 实测真值（两指总法向力）；F_max = k·F_min 推算，禁当损伤阈值 |

---

## 目录导览

### 数据

| 目录 | 内容 |
|---|---|
| [ExpForce数据集/](ExpForce数据集) | **首选训练数据**（唯一逐物体实测抓力真值，[官网](https://expforcesubmission.github.io/Exp-Force-Website/)）：Franka Panda + FORTE 触觉指夹自适应力控实测"刚好不滑落"最小抓力；129 物体（刚体 61/柔性 57/易碎 11）× D435i 腕部俯拍 RGB；`07_安全抓力范围.csv` 训练标注表；`siglip_retrieval.py` 检索先验；`fuse_visual_retrieval.py` 域路由融合；`eval_cross_camera*.py` 跨相机评测 |
| [ExpForce单流模型/](ExpForce单流模型) | 训练脚本：`train_expforce_single.py`（三类基线）/ `train_expforce_hier.py`（分层四类）/ `augment_fragile.py`（背景替换增强）；权重不入库，GPU 数分钟可复现 |
| [数据采集准备/](数据采集准备) | RGB-D 自采清单：三类各 60 候选（易碎优先），首批 90 个约 ¥570 |
| [图像力预测模型/](图像力预测模型) | 早期合成图模型 + 30 张跨相机实拍测试集（real_images/） |

### 模型

| 目录/文件 | 内容 |
|---|---|
| [dinov3_dual/](dinov3_dual) | 双流模型（DINOv3 + Depth Anything V2，骨干全冻结）——阶段 2 激活，深度流任务=空洞率门控/壁厚/质量估计 |
| 4T 盘 `safetyvtla_A1/` | model_cache（SigLIP 3.3GB）/ augmented（2100 张增强图）/ features（嵌入索引） |

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
〇·二、V-MaST 自采数据集与单流模型（当前工作重心）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

项目代号 **V-MaST**（**V**ision-based **Ma**terial **S**afety **T**hreshold
prediction，基于视觉的材料安全阈值预测）：以 RGB 视觉先验（DINOv2 特征 +
PaliGemma VLM 语义检索）预测物体材料类别与抓取安全力阈值（f_min / f_max），
集成于 π0.5 VLA。数据集：RGB_dataset/（结构见其 README_数据集结构说明.md）。
（2026-08-20 决策：双流方案废弃，仅建 DINO 单流模型。）

【数据采集准备/】RGB 自采物体清单（2026-08-19）
- 物体清单.html —— 三类物体各 60 个候选（易碎优先，公开 RGB-D 数据为零），
  每类前 30 个为首批必采（满足规范 A 级"最低可启动"线）；含分类边界速查、
  采购渠道与预算（全部 180 个约 ¥1100，首批约 ¥570）；浏览器打开可勾选追踪收集进度。
- objects_模板.csv —— 180 行预填模板（object_id/family_id/材质/渠道），质量尺寸与
  力值列待实测后填写；family_id 用于训练/测试隔离切分，禁止按图片随机切分。

### 文档（docs/）

| 文档 | 一句话 |
|---|---|
| [A1 实施路径](docs/A1_视觉先验安全阈值上下界预测_实施路径.md) | Exp-Force 实测盘点：只有 RGB+安全阈值 → 单流正确；三要素数据集不存在 |
| [成熟视觉模型整合方案](docs/成熟视觉模型整合方案_分类体系与端到端架构.md) | 分类体系选型：分层四类（易碎门控+轻脆/重脆+刚体/柔性），不合并刚体柔性 |
| [增强扩充与分层架构实验报告](docs/增强扩充与分层架构实验报告.md) | 7 组对照（A-G）：背景替换增强同相机 96-100%；易碎偏置增强跨相机召回 60-80% |
| [易碎物体安全阈值预测专项调研](docs/易碎物体安全阈值预测_专项调研.md) | Tier1 五项目：GraspSense 力图 / V2F 物理残差 / OMRON 可牺牲物 / 草莓实测 |
| [易碎与刚体视觉区分难题专项调研](docs/易碎与刚体视觉区分难题_专项调研.md) | 根因诊断 + 四条解决路线（物理传感/纯RGB/VLM语义先验/已有信号） |
| [SigLIP 检索式 A1 先验实验报告](docs/SigLIP检索式A1先验_实验报告.md) | 零训练检索：跨相机易碎 0%→100%，验证 PaliGemma 视觉塔复用可行 |
| [A1 域路由融合最终实验报告](docs/A1域路由融合_最终实验报告.md) | **A1 定稿**：top-1 域检测器 + 双策略路由，两域兼得 |

### 调研报告（HTML）

| 报告 | 主题 |
|---|---|
| dataset-credibility-report | 六候选数据集可信度评估（结论：力值均为估计/仿真，Exp-Force 唯一真值） |
| rgbd-image-datasets | RGB-D 图片数据集复检（GraspNet / TransCG / YCB-Video / UW） |
| rgbd-grasp-force-survey | RGB-D 抓力预测顶会调研（ForceSight / Hoi! / DeliGrasp / V2F） |
| vision-backbone-survey | 视觉骨干选型（DINOv3 单流首选 / DINOv3+DAv2 双流） |
| proposal-insights / project-progress-report | 方案洞察与进度汇报 |

### 候选数据集（已盘点归档，详细采集协议见 dataset-credibility-report）

| 类别 | 数据集 | 关键信息 | 状态 |
|---|---|---|---|
| 刚体 | RCT（arXiv:2606.31694） | DIGIT×3 按压 122 种工业材料，力-压深标定级对应（0.10mm 步进），29,279 帧 | 代码+论文已下载；9.23GB 数据 figshare 反爬需手动 |
| 刚体 | T-Rex（arXiv:2606.17055） | 双臂灵巧手 10 指尖触觉，200+ 物体 5400+ 轨迹，~20Hz wrench 流 | 代码+137MB 子集已下载 |
| 柔性 | Deform360（ECCV 2026） | 198 可变形物（绳/布/缆 13 类），4 路压力阵列+41 相机，215.7h | 2 物体触觉流子集已下载 |
| 柔性 | SoftVTBench（arXiv:2607.04234） | Isaac Sim FEM 柔性体，显式安全包络（按物体标定变形阈值 ε） | 清单+阈值参数+演示已下载 |
| 易碎 | OopsieVerse（RSS 2026） | DamageSim 将接触力转化为损伤/健康值，32 个家庭任务（含抓鸡蛋/酒杯） | 54 条演示 HDF5 已下载 |
| 易碎 | Tabero（arXiv:2605.27886） | Franka+触觉夹爪，轻柔指令下握力降 70%+ | 40 条演示+仿真资产已下载 |

> 可信度评估结论：以上 6 个数据集力值均为估计/仿真值；Exp-Force 是唯一逐物体实测真值来源，故列为首选训练数据。

---

## 力值规范（全链路统一）

- **F_min**：实测真值 = "刚好不滑落"最小两指总法向力，3 次取中位数，0.25N 取整
- **F_single** = F_total / 2（单指换算列并存，禁止静默转换）
- **F_max** = k × F_min（易碎 2.0 / 柔性 3.0 / 刚体 5.0）——工程推算值，**禁止当真实损伤阈值**
- 切分按 object_id / family_id 隔离；增强图只进训练集

## 环境与硬件

- conda base（Python 3.14）+ torch 2.10.0+cu128 + transformers
- RTX 4090 24GB（sudo 绕沙箱）；SigLIP 索引编码 ~10 秒，分层模型训练 0.3-8 分钟
- 模型权重 / 大文件不入库；4T 盘存权重缓存与增强图

## 路线图

- [x] 阶段 1：单流 RGB + 域路由检索（**已完成**）
- [ ] 索引库扩充：TransCG 玻璃物 + 自采柔性物入库（补库外泛化短板）
- [ ] 阶段 2：自采 RGB-D → 激活深度流（空洞率门控 / 壁厚容许力 / 质量估计）
- [ ] 阶段 3：易碎破损阈值实测（可牺牲物协议）→ F_max 实测化

---

## 备注

1. RCT 9.23GB 数据本体 figshare 反爬，直链见 `01_刚体_RCT/数据集下载说明.txt`
2. T-Rex / Deform360 大体积原始数据按需用 huggingface-cli 拉取
3. 暂不可下载候选：TacO（未发布）、RoboTacDex（soon）、V2F（未公开）
4. EgoTactile / OpenTouch 已按"人抓取不算"移除（2026-08-15）
5. 模型权重（*.pth）与压缩包不入库；训练脚本 GPU 数分钟可复现

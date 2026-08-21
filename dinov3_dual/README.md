# 模块一：视觉先验（真双流：DINOv3 ViT-L + Depth Anything V2 ViT-L）

对应技术方案中的**第一模块**（唯一任务）：输入 **RGB-D 配对图片**，输出**物体类别 + 初始安全抓力区间**。
不使用伪深度——深度图由你的 RGB-D 相机实测提供。

> **当前阶段（0820 起）：只采 RGB，不采深度图。** 自采数据放
> `E:\A-触觉机器学习\RGB_dataset\{叶类}\{物体ID}\`，训练路径为单流
> DINOv2 冻结 + L1/L2 分层头；本模块的双流流程等配对真实深度数据到位后再启用。

## 架构

```
RGB 图片  ──> DINOv3 ViT-L（冻结）──────────> 1024 维特征 ─┐
真实深度图 ──> DAv2 ViT-L 编码器（冻结）────> 1024 维特征 ─┼─> 融合 MLP ─> 分类头(3类)
深度统计(均值/方差/梯度/空洞率) ───────────────────────────┘         ├─> 力回归头(μ, σ)
                                                                    └─> 质量头(g)
物理基线: F = m·g / (2μ_类别)  与网络预测互为印证
```

- 两个 ViT-L 骨干全部**冻结**，只训练融合层和输出头（约 130 万参数）→ CPU 可训练
- 力回归输出 (μ, σ)，推荐抓力 = μ + z·σ（防滑高分位），封顶 = k·μ（损坏上限）
- 评估含掉落违规率、80% 区间覆盖率（来自方案报告的评价思想）

## 数据集格式（当前阶段仅 RGB，你负责拍照，程序负责其余）

```
E:\A-触觉机器学习\RGB_dataset\
├─ 轻脆\{物体ID}\s0.png ~ s14.png   （每物体 3 视角 × ≥5 张）
├─ 重脆\{物体ID}\...
├─ 刚体\{物体ID}\...
├─ 柔性\{物体ID}\...
└─ labels.csv   （由 check_dataset.py 自动生成，力值列留空待实测）
```

深度图当前不采集。未来若为 v2 双流补采：与 RGB 同目录同名加 `_depth` 后缀，
必须为 RGB-D 相机直出的 16-bit PNG 毫米制（禁伪深度）。

## 环境配置（一次性）

前提：E 盘 Python 环境（torch/torchvision/pillow/transformers 已装好）。

```powershell
# 模型权重预下载（约 2.5GB，存本项目 model_cache，不进 C 盘）
cd E:\A-触觉机器学习\dinov3_dual
set HF_ENDPOINT=https://hf-mirror.com
python download_models.py
```
DINOv3 走 GitHub，卡住再开 VPN；失败会自动回退 DINOv2（日志明确提示）。

## 使用步骤（5 条命令）

```powershell
cd E:\A-触觉机器学习\dinov3_dual

# ① 拍好图放入 RGB_dataset 后，检查完整性并生成 labels.csv
python check_dataset.py --make-labels
#    然后打开 labels.csv，填入实测力值（生成时留空，禁止填估计值）

# ②~④ 为双流(v2)流程，需配对深度图，当前阶段暂缓，仅作未来参考
# ② 提取双流特征并缓存（每张约 2-5 秒 CPU / <0.5 秒 GPU，只跑一次）
python extract_features.py

# ③ 训练融合头（分钟级）
python train.py

# ④ 推理
python predict.py --image "RGB_dataset\重脆\FRA001\s0.png"      # 单张（自动找 s0_depth.png）
python predict.py --image xx.png --depth xx_depth.png          # 显式指定深度图
python predict.py --folder "E:\某个文件夹"                       # 批量（仅配有深度的图）
python predict.py --image xx.png --risk 0.95                   # 更保守的防滑水平
```

## 输出示例

```
图片: glass1.png
  类别: 易碎（置信度 96.2%）
  质量估计: ~218 g   物理基线力: 1.12 N
  最小抓力: 1.05 ± 0.11 N
  => 推荐初始抓力 1.19 N，安全区间 [1.19, 2.10] N
```

## 耗时与显存

| 操作 | CPU | GPU(8GB) |
|---|---|---|
| 特征提取（首次） | 约 2-5 秒/张 | <0.5 秒/张 |
| 训练头部 | 分钟级 | <1 分钟 |
| 单张推理 | 5-30 秒 | <0.5 秒 |

双骨干推理峰值显存约 3-3.5GB。无 GPU 全自动回退 CPU。

## 目录说明

| 文件 | 作用 |
|---|---|
| config.py | 路径与超参数（换模型型号改这里） |
| backbones.py | 双骨干加载 + 真深度预处理（自动识别 16bit/8bit/npy） |
| check_dataset.py | 数据集配对校验 + labels.csv 生成 |
| extract_features.py | 双流特征缓存 |
| train.py | 融合网络训练（含覆盖率/违规率评估） |
| predict.py | 推理 |
| model_cache/ | 模型权重（E 盘） |

## 诚实声明

- 深度编码器 DAv2 ViT-L 的预训练输入是 RGB（单目深度估计），此处迁移用作真实深度图特征提取器——冻结骨干 + 可训练融合头会自动适配其特征空间，属常见迁移用法；若追求原生 RGB-D 预训练骨干可换 DFormer（留待后续）
- 透明物（玻璃）深度图可能有大片空洞，预处理会统计空洞率供模型参考；建议拍摄时避免强反光
- labels.csv 的力值若沿用 Exp-Force 平移默认值，为类别中位数估计；有条件请按自家夹爪实测替换

# 模块一：视觉先验（真双流：DINOv3 ViT-L + Depth Anything V2 ViT-L）

对应技术方案中的**第一模块**（唯一任务）：输入 **RGB-D 配对图片**，输出**物体类别 + 初始安全抓力区间**。
不使用伪深度——深度图由你的 RGB-D 相机实测提供。

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

## 数据集格式（你负责拍照，程序负责其余）

```
E:\A-机器学习\RGBD_dataset\
├─ 刚体\   obj001.png  obj001_depth.png   （RGB 与深度同目录同名，深度加 _depth 后缀）
├─ 柔性\   ...
├─ 易碎\   ...
└─ labels.csv   （由 check_dataset.py 自动生成，力值默认取 Exp-Force 同类中位数，可手改）
```

深度图也接受 `depth\` 子目录同名文件；支持 16-bit PNG（毫米）、8-bit 灰度、`.npy`（米）。
16-bit PNG（RealSense/D435 默认导出）直接放入即可，程序自动识别单位。

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

# ① 拍好图放入 RGBD_dataset 后，检查配对并生成 labels.csv
python check_dataset.py --make-labels
#    然后打开 labels.csv，把力值换成实测值（或保留 Exp-Force 平移默认值）

# ② 提取双流特征并缓存（每张约 2-5 秒 CPU / <0.5 秒 GPU，只跑一次）
python extract_features.py

# ③ 训练融合头（分钟级）
python train.py

# ④ 推理
python predict.py --image "RGBD_dataset\易碎\glass1.png"        # 单张（自动找 glass1_depth.png）
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

# Report Plan

## Meta
- **Type**: 项目进度汇报报告
- **Topic**: RGB-D 安全抓持力双流预测模型 —— 当前进度与完成情况（截至 2026-08-20）
- **Audience**: 项目负责人本人 / 汇报对象（导师、合作方）
- **Language**: 中文

## Design System
### Palette (工程蓝, Solid 模式)
- --bg: #f6f8fb / --bg2: #ffffff / --ink: #17233b / --muted: #5b6b84 / --rule: #dfe6f2 / --accent: #1e56c8 / --accent2: #0e9f76

### Typography
- 标题/正文: WorkSans + 系统中文回退(微软雅黑); 数字/代码: JetBrainsMono
- 标题样式: 加粗左对齐 + 章节编号; 正文 15px / 1.7
- 布局: 960px 居中单栏; h2 底部 2px accent 边框; 卡片 bg2+1px rule; 表格极简行边框

## Structure
1. 项目总览与当前状态 — 状态卡片仪表盘(已完成/进行中/阻塞)
2. 系统架构与模型版本 — Mermaid 双流架构图; v1/v2 定位; DINOv3→DINOv2 回退说明
3. 数据资产盘点 — 六项资产表格 + 合规红线
4. 训练与验证成果 — v1 族隔离训练曲线(ECharts) + 混淆矩阵; v2 冒烟 4/4; 深度流预训练曲线(ECharts 双轴)
5. 数据合规整改记录 — 伪深度清理 / 编造力值清理 / 族隔离 / ExpForce 无深度确认
6. 阻塞项与下一步 — 硬件清单表; 用户待办 vs 我方待办; 时间线

## Visuals
| Visual | Type | Tool |
|--------|------|------|
| 双流架构图 | flowchart | Mermaid |
| v1 族隔离训练曲线(准确率+MAE) | line 双轴 | ECharts |
| 深度流预训练曲线(对齐准确率+loss) | line 双轴 | ECharts |

## Key Facts (全部来自本次会话实测)
- v1: 111训/18验, 55/14族无交集, best 88.89% / MAE 0.95N, 早停41轮, 3.1分钟, 多数类基线50%
- 混淆: 刚体9/9, 柔性6/8, 易碎0/1
- v2 冒烟 4/4, FusionNet 1.18M 参数
- 预训练: 900×4视图, 120ep, 对齐91.7%(随机1.6%), 最近邻同场景99.6%
- YCB: 900对/12场景/21类/4125标注, 深度0.71-2.70m
- 资产: ExpForce 129(纯RGB+实测f_min), 阈值CSV 41物体, RGBD_dataset 0张
- GitHub: xzr-vvv/mlproject @17bbeb5(8-19), 其后产物待同步

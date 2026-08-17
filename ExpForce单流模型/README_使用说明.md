# Exp-Force 单流(RGB)安全抓力预测模型 — 使用说明

## 数据来源（单一开源项目，格式统一）
- 项目: Exp-Force (UT Austin, arXiv:2603.08668, 2026.03)
- 官网/仓库: https://expforcesubmission.github.io/Exp-Force-Website/
- 采集: Franka Panda 机械臂 + FORTE 触觉指夹, 自适应力控实测每物体"刚好不滑落"最小抓取力 F*,
  3 次取中位数, 向上取整到 0.25 N; 129 张 RealSense D435i 腕部相机真实 RGB 图
- 本地文件:
  - 标注: E:\A-机器学习\07_ExpForce_安全抓力范围.csv (129 行, 刚体61/柔性57/易碎11)
  - 图片: E:\A-机器学习\ExpForce_images\ (129 张真实图)
  - 原始官方表: E:\A-机器学习\ExpForce_dataset.csv
- 单位说明: 最小力为论文实测真值(牛顿); 最大安全力 = 类别系数 × 实测最小力
  (易碎×2.0, 柔性×3.0, 刚体×5.0), 同样取整到 0.25 N。无任何单位换算, 全表统一 N。
  最大力是政策性估计(全球无项目发布逐物体损坏阈值), data_source 列已标注。

## 模型与结果
- 结构: 单流 ResNet18(ImageNet 预训练) + 分类头(3类) + 力回归头(min/max, 牛顿)
- 训练: 129 张真实图, 分层划分 80/20 (训练104/验证25), 强增广, 冻结→解冻两阶段,
  余弦退火 + 标签平滑, 早停
- 结果: 验证集准确率 88%, 力 MAE: 最小力 0.60 N / 最大力 2.88 N
- 权重: expforce_single_stream.pth

## 运行环境（全部在 E 盘）
torch/torchvision/pillow/numpy 已安装到 E:\Lib\site-packages, 脚本内自动加载, 无需设置环境变量。

## 使用
训练（重跑）:
    python train_expforce_single.py
预测（单张图片或整个文件夹）:
    python predict_expforce_single.py E:\A-机器学习\ExpForce_images\Ceramic_mug.png
    python predict_expforce_single.py 某个文件夹路径

示例输出:
    Ceramic_mug.png -> 易碎 (置信度 98%) | 安全抓力 0.81 ~ 1.83 N

## 已知限制
- 易碎类样本最少(11 个), 该类验证仅 2 张, 是当前最弱类别; 增补易碎物体图片可优先提升
- 重物(如电钻 7N)的最大力存在低估倾向, 因训练集中高力值样本少
- 若换用其他相机拍摄的新图片, 建议先用几十张新图微调(fine-tune)再上线

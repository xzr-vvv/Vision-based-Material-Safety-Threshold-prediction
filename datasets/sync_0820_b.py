# -*- coding: utf-8 -*-
"""同步批次2: 修改的代码文件 + 文档与报告"""
import sys

sys.path.insert(0, r"E:\Lib\site-packages")
sys.path.insert(0, r"E:\A-触觉机器学习\datasets")

from git_ops import commit

commit({
    "add": [
        # 代码修复(backbones 的 tuple/BackboneOutput 适配, config 的 DAV2-HF 仓库修正等)
        r"dinov3_dual\backbones.py",
        r"dinov3_dual\config.py",
        r"dinov3_dual\download_models.py",
        # ExpForce 标签列变更(规范整改)
        r"ExpForce数据集\07_ExpForce_安全抓力范围.csv",
        # 数据来源说明
        r"datasets\README_数据来源说明.md",
    ],
    "message": "修改: backbones适配transformers5输出/DAV2-HF仓库, ExpForce标签列规范整改, 新增datasets来源说明README",
})

commit({
    "add": [
        # 调研产物
        r"rgbd-grasp-force-survey\ForceSight_对比表.csv",
        r"rgbd-grasp-force-survey\ForceSight_项目档案.csv",
        # 采集准备
        r"数据采集准备\objects_已有安全阈值.csv.xlsx",
        r"物体清单.html",
        # 进度报告
        r"progress-report-0820.html",
        r"progress-report-0820\plan.md",
        r"progress-report-0820\progress-report-0820.html",
        r"progress-report-0820\assets\charts.js",
    ],
    "message": "新增: ForceSight调研CSV, 物体清单(0819最终版), 0820项目进度报告(HTML+图表)",
})

commit({
    "add": [
        # 深度流预训练产物(小权重+报告; feats_cache可再生成不同步)
        r"datasets\ycb_rgbd\depth_pretrain\adapter.pt",
        r"datasets\ycb_rgbd\depth_pretrain\pretrain_report.txt",
        # 同步工具自身
        r"datasets\git_ops.py",
        r"datasets\sync_0820_a.py",
        r"datasets\sync_0820_b.py",
    ],
    "message": "新增: 深度流预训练adapter权重与报告(视图对齐91.7%/近邻同场景99.6%), git同步工具",
})

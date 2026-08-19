# -*- coding: utf-8 -*-
"""同步批次1: 代码与脚本类新增/修改"""
import sys

sys.path.insert(0, r"E:\Lib\site-packages")
sys.path.insert(0, r"E:\A-触觉机器学习\datasets")

from git_ops import commit

commit({
    "add": [
        # v2 代码修复与新增
        r"dinov3_dual\pretrain_depth.py",
        r"dinov3_dual\smoke_test.py",
        # v1 族隔离重训脚本
        r"ExpForce单流模型\train_expforce_family_split.py",
        # datasets 工具脚本(不含大数据)
        r"datasets\download_ycbv.py",
        r"datasets\ycbv_make_pairs.py",
        r"datasets\cornell_to_depth.py",
        r"datasets\make_depth_preview.py",
        r"datasets\verify_pairs_v2.py",
        r"datasets\stat_ycbv_objects.py",
        r"datasets\probe_hf.py",
        r"datasets\probe_wayback.py",
        r"datasets\download_cornell.py",
        r"datasets\download_datasets.py",
        r"datasets\bench_dav2.py",
        r"datasets\probe_ycb.py",
        r"datasets\probe_wayback2.py",
        r"datasets\probe_archive_org.py",
        r"datasets\git_ops.py",
        r"datasets\sync_0820_a.py",
    ],
    "message": "新增: v2管线修复(smoke_test/pretrain_depth), v1族隔离重训脚本, datasets下载转换验证脚本集",
})

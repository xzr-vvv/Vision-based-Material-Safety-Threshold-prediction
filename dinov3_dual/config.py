import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# E 盘 Python 环境的库路径
if os.path.isdir(r"E:\Lib\site-packages"):
    p = r"E:\Lib\site-packages"
    if p not in sys.path:
        sys.path.insert(0, p)

# 模型权重与缓存全部放 E 盘
os.environ.setdefault("TORCH_HOME", os.path.join(BASE_DIR, "model_cache", "torch"))
os.environ.setdefault("HF_HOME", os.path.join(BASE_DIR, "model_cache", "hf"))
# 国内下载 HuggingFace 模型：先在终端执行 set HF_ENDPOINT=https://hf-mirror.com

# ===== 你的自建数据集（当前阶段: 仅 RGB, 0820 决定不采深度图）=====
# 结构: RGB_dataset\{叶类}\{物体ID}\s0.png ~ s14.png（详见 RGB_dataset\README_数据集结构说明.md）
DATASET_ROOT = r"E:\A-触觉机器学习\RGB_dataset"
CSV_PATH = os.path.join(DATASET_ROOT, "labels.csv")
# 采集物体清单（object_id/object_family_id/leaf_class 关联用）
OBJECTS_CSV = r"E:\A-触觉机器学习\数据采集准备\objects_模板.csv"
# Exp-Force 标签库（对照参考，不作自采数据的力值来源）
EXPFORCE_CSV = r"E:\A-触觉机器学习\ExpForce数据集\07_ExpForce_安全抓力范围.csv"

MODEL_SIZE = "l"          # l = ViT-L（推荐）
# 注意: 必须用 -hf 后缀的 transformers 兼容版; 原版仓库 config 缺 model_type 无法加载
DAV2_HF_ID = "depth-anything/Depth-Anything-V2-Large-hf"

IMG_SIZE = 224
BATCH = 16
EPOCHS = 200
LR = 3e-4
SEED = 42
VAL_RATIO = 0.2

FEATURE_DIR = os.path.join(BASE_DIR, "rgb_features")
FEAT_FILE = os.path.join(FEATURE_DIR, "features.pt")
CKPT_PATH = os.path.join(BASE_DIR, "dinov3_dual_head.pth")

CLASSES = ["刚体", "柔性", "易碎"]

# L1/L2 分层分类: 叶类是文件夹与 labels.csv 的基本单位
LEAVES = ["轻脆", "重脆", "刚体", "柔性"]
LEAF2CAT = {"轻脆": "易碎", "重脆": "易碎", "刚体": "刚体", "柔性": "柔性"}
LEAF2L1 = {"轻脆": "易碎", "重脆": "易碎", "刚体": "非易碎", "柔性": "非易碎"}
LEAF_TARGET = 30  # 每叶最少物体数

# 物理基线参数（Exp-Force 129 物体反推）
MU_CLASS = {"刚体": 0.66, "柔性": 0.22, "易碎": 0.95}
K_MAX = {"刚体": 5.0, "柔性": 3.0, "易碎": 2.0}
G = 9.81
FORCE_RES = 0.25

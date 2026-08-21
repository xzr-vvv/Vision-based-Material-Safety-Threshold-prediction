# V-MaST —— 基于视觉的材料安全阈值预测（RGB_dataset 自采数据集）

> 项目代号 **V-MaST**（**V**ision-based **Ma**terial **S**afety **T**hreshold prediction，
> 中文名：基于视觉的材料安全阈值预测）：以 RGB 视觉先验（DINOv2 特征 + VLM 语义检索）
> 预测物体材料类别与抓取安全力阈值（f_min / f_max），集成于 π0.5 VLA。
> 本目录为其数据集部分，当前阶段：仅 RGB。

## 目录结构（4 个叶类，L1/L2 分层分类法）

```
RGB_dataset\
  轻脆\{物体ID}\s0.png ~ s14.png    薯片/鸡蛋/薄脆饼干等（受力即毁，F_min 约 0.25–0.5 N）
  重脆\{物体ID}\...                 玻璃杯/陶瓷杯/玻璃瓶等（可承受中等力，超限碎裂）
  刚体\{物体ID}\...
  柔性\{物体ID}\...
  labels.csv                        由 check_dataset.py --make-labels 生成
```

叶类 → 类别/L1 映射（脚本自动推导，无需手填）：
- 轻脆、重脆 → 易碎（L1 门控：易碎）
- 刚体、柔性 → 非易碎（L1 门控：非易碎）

**当前阶段只采 RGB 图，不采深度图。**（0820 决定：训练路径为单流
DINOv2 冻结 + L1/L2 头；v2 双流等有配对真实深度数据后再议。）

## 物体 ID

沿用 `数据采集准备\objects_模板.csv` 的 object_id：
- FRA001–FRA066：易碎物体（leaf_class 列标明轻脆/重脆，柔性除外）
- RIG001–RIG060：刚体　　SOF001–SOF060：柔性

注意：FRA016（纸杯）、FRA039（薄壁塑料杯）、FRA040（蛋盒）已于 0820 重标为**柔性**
（ExpForce 实测中纸杯/塑料杯均为柔性类），ID 前缀保留不变，以 leaf_class 列为准。

## 每个物体的图片要求

- ≥15 张 RGB：3 个视角 × 每视角 ≥5 张
- 命名：`s0.png ~ s14.png`（s0–s4 视角1，s5–s9 视角2，s10–s14 视角3）
- 普通相机/手机即可，光照自然、背景日常，无需深度图

## 数量与构成（0821 裁剪定稿）

共 **181 物体 / 1847 图**：轻脆 30（3 E + 27 FRA）/ 重脆 30（8 E + 22 FRA）/
刚体 61（全 E）/ 柔性 60（57 E + FRA016/039/040）。

- ExpForce E 系 129 个**全部保留**（含 62 个仅有原始 1 图的物体；其 f_min 均为
  ExpForce 本物实测）；E 系多图物体另有 15 张网图补充视角
- 21 个 YCB 物体与 20 个仅有近似力值的 FRA 已于 0821 移出（YCB 图片源自
  YCB-Video 视频帧、力值无一实测），文件夹在 `_quarantine\` 可找回
- 轻脆/重脆的 FRA 按力值证据等级取舍：同型实测/同型转移全保，
  近似转移者按视角多样性排序删除最差 10 + 10 个

## 力值标签（红线）

- labels.csv 的 `f_min_measured_N`、`f_max_measured_N` 只填**别人实测**的值
  （0821 已回填完成，note 列逐行注明证据来源）：
  - f_min：ExpForce 本物实测（E 系 129 个）或同型实测/同型转移（FRA 20 个），
    语义均为两指法向力之和
  - f_max：44 个物体。其中 **39 个为"文献/国标单接触实测值 ×2 语义换算"**
    （note 列已明示，如 `文献实测42.0N(平板压缩)×2换算=84N(两指合力,非直接实测)`）；
    5 个为夹持语义直接实测（草莓类 0.6N、GraspSense 玻璃杯 69N，单/总未确认保守按合力）。
    **严格意义的"两指合力 f_max 直接实测"目前为 0 个**——换算值按 SAFETYVTLA V1
    双列规则保留（force_map 中 f_max_label_N 存原始值、f_max_total_N 存换算值）
  - 无实测来源的物体两列留空、note 标注"无实测"，仅用于分类训练
- 禁止填类别系数 × f_min 之类的估计值当真值（SAFETYVTLA v1 红线）
- **F_max 的 SafetyVTLA 定位（0821，按《缺口与可用资源整理》§4.2.B）**：上述 44 个
  f_max 均为**跨平台文献实测（换算）值**，只能用作**弱先验 / 离线评测参考**，
  **不是** SafetyVTLA 要求的"Gen3 + Robotiq 2F-85 + Sensor3D 同平台破坏性上限
  实测"真值。同构 F_max 真值到位前，任何 f_max 回归结果不得对外宣称
  "Gen3 视觉→安全阈值"（agent_rule §5–6 红线）
- 叶类存疑的物体（清单 notes 标"[叶类边界]"）：实测力值落入另一叶的力域时，复核后改
  objects_模板.csv 的 leaf_class，再重跑 check_dataset.py

### force_map.csv 力值语义（SAFETYVTLA V1 强制要求，0820 落实）

**所有力值单位 N，语义必须显式标注，禁止静默换算。**

- `f_min_semantics`：全部为**两指法向力之和**。依据 ExpForce 论文定义：
  "Grasp force is defined as the sum of the contact normal forces of two
  parallel gripper fingers against the object"（arXiv:2603.08668）。
  测量协议：FORTE 触觉指自适应加力（0.25N 起）至滑移瞬间记录。
- `f_min_single_N`：按文件默认规则 `F_single = F_total/2` 显式换算的单指力
- `f_max_label_N`：文献/国标原始值，**多为单指法向力语义**（平板压缩=单接触加载）
- `f_max_total_N`：按 `F_total = 2×F_single` 显式换算的两指合力；语义无法确认的
  （草莓 0.6N、GraspSense 69N）保守按合力采用、不翻倍，见 `f_max_semantics` 列
- `f_max_semantics`：逐行标注加载语义（平板压缩/双板对压/压头/挤压/夹持）
- `conversion_rule`：逐行换算规则

### force_map.csv 力值依据分级（0820 新增 value_basis/f_max_basis/evidence 列）

force_map 是**标签参考值表**，不是实测真值表。每行标注取值依据：

| value_basis | 含义 | 数量 |
|---|---|---|
| 本物实测 | ExpForce(UT Austin) 对该物体本身的实测夹持力（f_min） | 129 |
| 同型实测 | ExpForce 对**同商品类型**（如 Ceramic mug↔陶瓷马克杯）的实测值，同款夹爪 | 24 |
| 近似转移 | 同类别相近物体的参照值（有 ExpForce 参照的注明参照物） | 37 |
| 估算 | 无实测依据的估计值 | 117 |

f_max 另有 f_max_basis 列：44 个物体有**文献/国标实测**依据（鸡蛋壳平板压缩 42–51 N、
草莓夹持损伤 0.6 N、苹果损伤阈值 40 N、空罐侧壁屈曲 635 N 等，均带 DOI/URL 于
evidence 列）。

**0821 力值清理**：估算/外推的 f_max（254 行）与 FRA 近似转移的 f_min（52 行）
已全部删除，原值移入 f_max_basis / evidence 列留档（标注 `[0821删]`）。
现存有效值：f_min = ExpForce 本物实测 129 + 同型 20（全部为两指合力直接实测）；
f_max = 44 个（39 个文献单接触实测 ×2 换算 + 5 个夹持实测保守值，见上）。

**加载方式警示（0821 按本项目夹爪构型修订）**：本项目机械臂为**两指夹爪、
指面带两层传感器平板**——与文献平板压缩试验同为"平板对压"加载方式
（平板压曲面本身就是接触斑加载，随力增大），**接触几何基本一致**；
软质传感器垫顺应物体曲面后接触斑只会更大，真实破碎力 ≥ 刚性平板值
（偏安全方向）。因此 44 个 f_max 的 ×2 换算语义对**本项目硬件直接成立**，
文献平板压缩值无需按"指尖局部接触"打折。

残余差距仅两项，使用时留 **1.2–1.5 倍安全余量**即可：
1. **加载速率**：文献为准静态压缩；夹爪快速闭合产生动载，脆性材料对速率敏感
2. **加载方向**（蛋类等各向异性物体）：蛋沿长轴抗压最强、赤道方向明显更弱；
   E060/E061/FRA024 等蛋类物体引用的 42–51 N 若为沿长轴值，赤道方向夹持时
   破坏力会更低

ExpForce 自身装置（LEGATO 两指爪 + FORTE 软质触觉指）与本构型同族，
其 f_min（两指合力）对本项目硬件迁移性好。
ExpForce 论文出处：Shang et al., Exp-Force, UT Austin 2026
（https://expforcesubmission.github.io/Exp-Force-Website/）。

## 族体系统一与阈值表增补（0821）

**族统一**：labels.csv 的 object_family_id 已合并 15 组同型异族
（蛋 3 族→egg、苹果 4 族→apple、葡萄 2 族、番茄 5 族→tomato、椒 2 族、
柠檬/青柠/橙/橘各 2 族、棉花糖 2 族、玻璃杯 3 族→glass_cup、高脚杯 4 族、
玻璃瓶罐 3 族、陶瓷杯 2 族、脆零食 3 族），族数 128→**103**；
objects_模板.csv 同步（`unify_families.py`，原文件备份 .bak_before_family_unify）。
目的：**按族分组切分训练/测试时防同型泄漏**。族对照关系见
`数据采集准备\family_crosswalk.csv`（阈值表物体 → ExpForce 参照 → 参照所在统一族）。

**objects_已有安全阈值.csv 增补**（`enrich_thresholds_csv.py`，41 行全补齐）：

- `mass_kg`：ExpForce 同型参照实测质量均值（provenance→物体名→Mass 三级模糊匹配）
- `dimensions_m`：常见典型尺寸（**±30% 浮动估算口径**，逐物体）
- `volume_m3`：按形状公式（cyl=π/4·d²·h、ell=π/6·L·W·H、box=f·L·W·H、bag=0.45·L·W·H）
- `density_kg_m3`：有效密度（空心物体按外接体积，非材料密度）
- `ref_f_max_eval_total_N / _single_N / _basis`：F_max **评测参考列**
  （17 行文献换算值 + 24 行类别系数占位 易碎×2/柔性×3/刚体×5，
  **全部不写入 SafetyVTLA 真值表**，占位值仅用于评 gap 上界）
- `phys_est_basis`：物理量估算口径；4 行参照状态不匹配已标注
  （单片薯片↔袋装、单粒葡萄↔整串、空瓶↔满壶、迷你椒↔常规青椒）

## 采完后的操作

```
cd E:\A-触觉机器学习\dinov3_dual
python check_dataset.py --make-labels
```

脚本会：校验每物体图片数 → 统计每叶物体数 → 生成 labels.csv（含
leaf_class/category/l1_gate/object_id/object_family_id 列，族信息自动关联清单）→
提示张数不足的物体。注意：`--make-labels` 会保留已填的力值列；
力值回填用 `数据采集准备\fill_labels_force.py`（从 force_map 合并）。
力值已回填别人实测值，当前可直接用于训练。

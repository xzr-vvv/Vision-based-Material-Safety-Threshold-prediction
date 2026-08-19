import numpy as np
import torch
import torch.nn.functional as F

import config as C


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _freeze(m):
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def _cls_feature(out):
    if isinstance(out, dict):
        for k in ("x_norm_clstoken", "last_hidden_state"):
            if k in out:
                v = out[k]
                return v[:, 0] if v.dim() == 3 else v
        v = next(iter(out.values()))
        return v[:, 0] if v.dim() == 3 else v
    if torch.is_tensor(out):
        return out[:, 0] if out.dim() == 3 else out
    raise RuntimeError("无法从骨干输出中提取特征: " + str(type(out)))


@torch.no_grad()
def load_backbone(size=None, device=None):
    """RGB 流: DINOv3，失败自动回退 DINOv2 / timm。返回 (model, feat_dim, name)"""
    size = size or C.MODEL_SIZE
    device = device or get_device()
    errors = []
    try:
        m = torch.hub.load("facebookresearch/dinov3", f"dinov3_vit{size}",
                           pretrained=True, trust_repo=True)
        _freeze(m).to(device)
        return m, m.embed_dim, f"DINOv3 ViT-{size.upper()}"
    except Exception as e:
        errors.append(f"DINOv3 加载失败: {e}")
    try:
        name = {"s": "dinov2_vits14", "b": "dinov2_vitb14", "l": "dinov2_vitl14"}[size]
        m = torch.hub.load("facebookresearch/dinov2", name, pretrained=True, trust_repo=True)
        _freeze(m).to(device)
        return m, m.embed_dim, f"DINOv2 {name}（回退）"
    except Exception as e:
        errors.append(f"DINOv2 加载失败: {e}")
    try:
        import timm
        name = {"s": "vit_small_patch14_dinov2.lvd142m",
                "b": "vit_base_patch14_dinov2.lvd142m",
                "l": "vit_large_patch14_dinov2.lvd142m"}[size]
        m = _freeze(timm.create_model(name, pretrained=True, num_classes=0)).to(device)
        return m, m.num_features, f"timm {name}（回退）"
    except Exception as e:
        errors.append(f"timm 加载失败: {e}")
    raise RuntimeError("所有 RGB 骨干加载失败:\n" + "\n".join(errors)
                       + "\n请检查网络（GitHub 可能需要 VPN），或设置 HF_ENDPOINT=https://hf-mirror.com")


@torch.no_grad()
def extract_rgb_feature(model, pil_img, device=None):
    """PIL RGB 图片 -> CLS 特征 (D,)，CPU tensor"""
    from torchvision.transforms import functional as TF
    device = device or next(model.parameters()).device
    x = TF.resize(pil_img.convert("RGB"), [C.IMG_SIZE, C.IMG_SIZE])
    t = TF.normalize(TF.to_tensor(x), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    out = model(t.unsqueeze(0).to(device))
    return _cls_feature(out).squeeze(0).float().cpu()


# ===================== 深度流（真实 RGB-D 深度图） =====================

def load_depth_array(path):
    """读深度文件 -> 米制 float32 数组（支持 16bit png 毫米 / 8bit 灰度 / .npy 米）"""
    if path.lower().endswith(".npy"):
        return np.load(path).astype(np.float32)
    from PIL import Image
    with Image.open(path) as im:
        arr = np.array(im)
        is_metric = arr.dtype in (np.uint16, np.int32)   # 16/32bit PNG = 毫米（RealSense 等约定）
    arr = arr.astype(np.float32)
    if is_metric:
        return arr / 1000.0
    if arr.max() <= 1.0:          # 已归一化
        return arr
    return arr / 255.0 * 2.5      # 8bit 相对深度，缩放到米量级


def preprocess_depth(arr):
    """米制深度数组 -> (归一化深度 PIL 图 224x224, 统计向量 [均值,方差,梯度,空洞率])"""
    from PIL import Image
    valid = (arr > 0.01) & np.isfinite(arr)
    if valid.sum() < 100:
        raise ValueError("有效深度像素过少，深度图疑似为空")
    v = arr[valid]
    lo, hi = np.percentile(v, [5, 95])
    dn = np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1)
    dn[~valid] = 0.0

    img = Image.fromarray((dn * 255).astype(np.uint8)).resize(
        (C.IMG_SIZE, C.IMG_SIZE), Image.LANCZOS)

    t = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0)
    gx = t[1:, :] - t[:-1, :]
    gy = t[:, 1:] - t[:, :-1]
    stats = torch.stack([t.mean(), t.std(),
                         (gx.abs().mean() + gy.abs().mean()) / 2,
                         torch.tensor(1.0 - valid.mean(), dtype=torch.float32)])
    return img, stats


@torch.no_grad()
def load_depth_encoder(device=None):
    """深度流: Depth Anything V2 ViT-L 编码器（冻结），作为深度图特征提取器。
    注: 该编码器预训练输入为 RGB（单目深度估计），此处迁移用于真实深度图特征提取，
    属常见迁移用法；融合头会学习适配其特征空间。"""
    from transformers import AutoModelForDepthEstimation
    device = device or get_device()
    full = AutoModelForDepthEstimation.from_pretrained(C.DAV2_HF_ID)
    enc = getattr(full, "backbone", None) or getattr(full.model, "backbone", full.model)
    _freeze(enc).to(device)
    dim = getattr(enc, "hidden_size", None) or 1024
    return enc, dim, "DAv2 ViT-L encoder"


@torch.no_grad()
def extract_depth_feature(enc, depth_pil, device=None):
    """归一化深度 PIL 图 -> CLS 特征 (D,)，CPU tensor"""
    from torchvision.transforms import functional as TF
    device = device or next(enc.parameters()).device
    t = TF.to_tensor(depth_pil.convert("L")).repeat(3, 1, 1)     # 复制3通道对齐预训练输入
    t = TF.normalize(t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    try:
        out = enc(t.unsqueeze(0).to(device), interpolate_pos_encoding=True)
    except TypeError:
        out = enc(t.unsqueeze(0).to(device))
    return _cls_feature(out).squeeze(0).float().cpu()


def find_depth_for(rgb_path):
    """RGB 路径 -> 配对深度图路径；找不到返回 None。
    约定: 同目录 xxx_depth.* 或 depth 子目录同名"""
    import os
    base, name = os.path.split(rgb_path)
    stem = os.path.splitext(name)[0]
    root = os.path.dirname(base)
    cands = []
    for ext in (".png", ".jpg", ".jpeg", ".npy", ".tiff", ".bmp"):
        cands.append(os.path.join(base, stem + "_depth" + ext))
        cands.append(os.path.join(base, "depth", name))
        cands.append(os.path.join(root, "depth", name))
        cands.append(os.path.join(base, "depth", stem + ext))
    for p in cands:
        if os.path.exists(p):
            return p
    return None


Z_TABLE = {0.80: 0.84, 0.90: 1.28, 0.95: 1.64, 0.99: 2.33}


def z_of_risk(risk):
    if risk in Z_TABLE:
        return Z_TABLE[risk]
    raise ValueError(f"risk 仅支持 {list(Z_TABLE)}，收到 {risk}")


def force_nll(pred, y, mask=None):
    mu = pred[:, 0]
    sigma = 0.05 + F.softplus(pred[:, 1])
    nll = (y - mu) ** 2 / (2 * sigma ** 2) + torch.log(sigma)
    if mask is not None:
        return (nll * mask).sum() / mask.sum().clamp(min=1)
    return nll.mean()

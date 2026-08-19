import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

import config as C
from backbones import get_device, force_nll


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class FusionNet(nn.Module):
    """双流融合: DINOv3 RGB 特征 + DAv2 深度特征 + 深度统计 -> 三头"""

    def __init__(self, rgb_dim, dep_dim, n_classes=3):
        super().__init__()
        in_dim = rgb_dim + dep_dim + 4
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, 512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.2),
        )
        self.head_cls = nn.Linear(256, n_classes)
        self.head_force = nn.Linear(256, 2)   # (μ, raw σ)
        self.head_mass = nn.Linear(256, 1)    # log1p(质量/g)

    def forward(self, rgb_feat, dep_feat, stats):
        z = self.trunk(torch.cat([rgb_feat, dep_feat, stats], dim=1))
        return self.head_cls(z), self.head_force(z), self.head_mass(z)


def load_tensors():
    data = torch.load(C.FEAT_FILE, map_location="cpu")
    feats = data["feats"]
    rgb, dep, stt, cls, y, mass, mask, names = [], [], [], [], [], [], [], []
    for name, rec in feats.items():
        rgb.append(rec["rgb"])
        dep.append(rec["dep"])
        stt.append(rec["stats"])
        cls.append(rec["cls"])
        y.append(rec["min_force"])
        mass.append(np.log1p(max(rec["mass_g"], 0.0)) if rec["mass_g"] > 0 else 0.0)
        mask.append(1.0 if rec["mass_g"] > 0 else 0.0)
        names.append(name)
    return (torch.stack(rgb), torch.stack(dep), torch.stack(stt),
            torch.tensor(cls), torch.tensor(y, dtype=torch.float32),
            torch.tensor(mass, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32), names,
            data["rgb_dim"], data["dep_dim"])


def stratified_split(cls, ratio, seed):
    rng = random.Random(seed)
    tr, va = [], []
    for c in sorted(set(cls.tolist())):
        idx = [i for i, v in enumerate(cls.tolist()) if v == c]
        rng.shuffle(idx)
        k = max(1, int(len(idx) * ratio))
        va += idx[:k]
        tr += idx[k:]
    return tr, va


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    n = correct = 0
    errs, drops, cov80 = [], 0, 0
    per_sigma = {c: [] for c in range(len(C.CLASSES))}
    for rgb, dep, stt, cls, y, _m in loader:
        rgb, dep, stt = rgb.to(device), dep.to(device), stt.to(device)
        logits, force, _ = model(rgb, dep, stt)
        pred = logits.argmax(1).cpu()
        mu = force[:, 0].cpu()
        sigma = 0.05 + F.softplus(force[:, 1]).cpu()
        correct += (pred == cls).sum().item()
        n += len(cls)
        errs += (mu - y).abs().tolist()
        drops += (mu > y + C.FORCE_RES).sum().item()
        cov80 += (y >= mu - 1.28 * sigma).sum().item()
        for c, s in zip(cls.tolist(), sigma.tolist()):
            per_sigma[c].append(s)
    return {
        "acc": correct / n,
        "mae": float(np.mean(errs)),
        "drop_rate": drops / n,
        "cov80": cov80 / n,
        "sigma": {C.CLASSES[c]: float(np.mean(v)) for c, v in per_sigma.items() if v},
    }


def main():
    set_seed(C.SEED)
    device = get_device()
    print("设备:", device)
    rgb, dep, stt, cls, y, mass, mask, names, rgb_dim, dep_dim = load_tensors()
    print(f"样本 {len(names)}  RGB {rgb_dim} 维  Depth {dep_dim} 维")

    tr, va = stratified_split(cls, C.VAL_RATIO, C.SEED)
    print(f"训练 {len(tr)} / 验证 {len(va)}（按类别分层）")
    if len(tr) < 10:
        print("训练样本过少，请先补充数据集图片")

    counts = torch.bincount(cls[tr], minlength=len(C.CLASSES)).float()
    w = (1.0 / counts.clamp(min=1).sqrt())
    w = (w / w.sum() * len(C.CLASSES)).to(device)
    sampler_w = counts[cls[tr]].clamp(min=1)
    sampler = WeightedRandomSampler(1.0 / sampler_w, num_samples=len(tr), replacement=True)

    def mk(idx):
        return TensorDataset(rgb[idx], dep[idx], stt[idx], cls[idx], y[idx], mass[idx])

    tr_loader = DataLoader(mk(torch.tensor(tr)), batch_size=C.BATCH, sampler=sampler)
    va_loader = DataLoader(mk(torch.tensor(va)), batch_size=C.BATCH)

    model = FusionNet(rgb_dim, dep_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=C.EPOCHS)

    best = -1
    best_state = None
    for ep in range(1, C.EPOCHS + 1):
        model.train()
        for rgb_b, dep_b, stt_b, cls_b, y_b, m_b in tr_loader:
            rgb_b, dep_b, stt_b = rgb_b.to(device), dep_b.to(device), stt_b.to(device)
            cls_b, y_b, m_b = cls_b.to(device), y_b.to(device), m_b.to(device)
            logits, force, mass_pred = model(rgb_b, dep_b, stt_b)
            loss = (F.cross_entropy(logits, cls_b, weight=w)
                    + force_nll(force, y_b)
                    + 0.3 * F.mse_loss(mass_pred.squeeze(-1), m_b))
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        if ep % 10 == 0 or ep == C.EPOCHS:
            m = evaluate(model, va_loader, device)
            score = m["acc"] + m["cov80"] - m["mae"] / 10
            print(f"epoch {ep:3d}  acc {m['acc']:.3f}  MAE {m['mae']:.2f}N  "
                  f"掉落违规 {m['drop_rate']:.3f}  80%覆盖 {m['cov80']:.3f}")
            if score > best:
                best = score
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    final = evaluate(model, va_loader, device)
    print("\n===== 最终验证结果 =====")
    print(f"分类准确率: {final['acc']*100:.1f}%")
    print(f"最小力 MAE: {final['mae']:.2f} N")
    print(f"掉落违规率(预测下限偏高): {final['drop_rate']*100:.1f}%")
    print(f"80% 区间覆盖率: {final['cov80']*100:.1f}%")
    print("各类预测不确定度 σ:", {k: f"{v:.2f}N" for k, v in final["sigma"].items()})

    torch.save({"state": model.state_dict(), "rgb_dim": rgb_dim, "dep_dim": dep_dim,
                "classes": C.CLASSES}, C.CKPT_PATH)
    print("模型已保存:", C.CKPT_PATH)


if __name__ == "__main__":
    main()

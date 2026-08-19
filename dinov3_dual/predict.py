import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import config as C
from backbones import (get_device, load_backbone, extract_rgb_feature,
                       load_depth_encoder, extract_depth_feature,
                       load_depth_array, preprocess_depth, find_depth_for, z_of_risk)
from train import FusionNet

EXTS = (".png", ".jpg", ".jpeg")


class Predictor:
    def __init__(self, risk=0.90):
        self.device = get_device()
        ckpt = torch.load(C.CKPT_PATH, map_location="cpu")
        self.model = FusionNet(ckpt["rgb_dim"], ckpt["dep_dim"])
        self.model.load_state_dict(ckpt["state"])
        self.model.eval().to(self.device)
        self.z = z_of_risk(risk)
        self.backbone = None
        self.depth_enc = None

    def _ensure_models(self):
        if self.backbone is None:
            print("首次推理，加载模型（约 1-3 分钟）...")
            self.backbone, _, _ = load_backbone(device=self.device)
            self.depth_enc, _, _ = load_depth_encoder(device=self.device)

    @torch.no_grad()
    def predict_path(self, rgb_path, depth_path=None):
        depth_path = depth_path or find_depth_for(rgb_path)
        if depth_path is None:
            raise FileNotFoundError(
                f"未找到配对深度图: {rgb_path}\n"
                f"约定: 同目录 xxx_depth.png，或 depth\\ 子目录同名文件。本模型为真双流，必须提供真实深度图")

        self._ensure_models()
        depth_img, stats = preprocess_depth(load_depth_array(depth_path))
        rgb_feat = extract_rgb_feature(self.backbone, Image.open(rgb_path), self.device)
        dep_feat = extract_depth_feature(self.depth_enc, depth_img, self.device)

        logits, force, mass_pred = self.model(
            rgb_feat.unsqueeze(0).to(self.device),
            dep_feat.unsqueeze(0).to(self.device),
            stats.unsqueeze(0).to(self.device),
        )
        probs = F.softmax(logits, dim=1)[0]
        conf, idx = probs.max(0)
        cls = C.CLASSES[idx.item()]
        mu = float(force[0, 0])
        sigma = float(0.05 + F.softplus(force[0, 1]))
        mass_g = float(torch.expm1(mass_pred[0, 0]))
        k = C.K_MAX[cls]

        cmd = min(mu + self.z * sigma, mu * k)
        phys = max(0.05, mass_g / 1000 * C.G / (2 * C.MU_CLASS[cls]))
        return {
            "file": os.path.basename(rgb_path), "cls": cls, "conf": float(conf),
            "mu": mu, "sigma": sigma, "mass_g": mass_g,
            "f_min": max(0.1, cmd), "f_max": max(mu * k, 0.2),
            "phys": phys, "probs": {c: float(probs[i]) for i, c in enumerate(C.CLASSES)},
        }


def print_result(r):
    print(f"\n图片: {r['file']}")
    print(f"  类别: {r['cls']}（置信度 {r['conf']*100:.1f}%）")
    for c, p in r["probs"].items():
        print(f"    {c}: {p*100:5.1f}%")
    print(f"  质量估计: ~{r['mass_g']:.0f} g   物理基线力: {r['phys']:.2f} N")
    print(f"  最小抓力: {r['mu']:.2f} ± {r['sigma']:.2f} N")
    print(f"  => 推荐初始抓力 {r['f_min']:.2f} N，安全区间 [{r['f_min']:.2f}, {r['f_max']:.2f}] N")


def main():
    ap = argparse.ArgumentParser(description="模块一推理：RGB-D 图片 -> 类别 + 初始安全抓力区间")
    ap.add_argument("--image", help="RGB 图片路径（深度图按约定自动查找）")
    ap.add_argument("--depth", help="显式指定深度图路径（可选）")
    ap.add_argument("--folder", help="文件夹路径（递归预测有配对深度图的图片）")
    ap.add_argument("--risk", type=float, default=0.90,
                    help="防滑置信水平 0.80/0.90/0.95/0.99，默认 0.90")
    args = ap.parse_args()

    pred = Predictor(risk=args.risk)

    if args.image:
        print_result(pred.predict_path(args.image, args.depth))
        return

    if args.folder:
        paths = []
        for root, _, names in os.walk(args.folder):
            for n in sorted(names):
                if n.lower().endswith(EXTS) and "_depth" not in n.lower():
                    p = os.path.join(root, n)
                    if find_depth_for(p):
                        paths.append(p)
        if not paths:
            print("文件夹中没有找到带配对深度图的 RGB 图片")
            return
        print(f"共 {len(paths)} 张有配对深度图的图片")
        for p in paths:
            print_result(pred.predict_path(p))
        return

    ap.print_help()


if __name__ == "__main__":
    main()

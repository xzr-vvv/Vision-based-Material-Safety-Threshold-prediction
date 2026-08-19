import config as C
from backbones import load_backbone, load_depth_model, get_device

if __name__ == "__main__":
    device = get_device()
    print("设备:", device)
    print("正在下载/加载 RGB 骨干（首次约 1.2GB，存至", C.BASE_DIR, "model_cache）...")
    _, dim, name = load_backbone(device=device)
    print("RGB 骨干就绪:", name, "特征维度", dim)
    print("正在下载/加载深度模型（首次约 1.3GB）...")
    _, proc = load_depth_model(device=device)
    print("深度模型就绪:", C.DAV2_HF_ID)
    print("全部完成。若下载失败: 开 VPN，或先执行  set HF_ENDPOINT=https://hf-mirror.com  再重跑")

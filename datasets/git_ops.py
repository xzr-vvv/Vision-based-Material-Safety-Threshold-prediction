# -*- coding: utf-8 -*-
"""dulwich git 操作: 状态 / 提交 / 推送"""
import os
import sys

sys.path.insert(0, r"E:\Lib\site-packages")

from datetime import datetime

from dulwich import porcelain

ROOT = r"E:\A-触觉机器学习"

# 不同步的大文件/目录(现有 .gitignore 已覆盖 *.zip *.pth, 此处明确列出以便核对)
SKIP_DIRS = {
    r"datasets\ycb_rgbd\test",            # 900对原始RGB-D ~700MB(README已附HF链接)
    r"datasets\ycb_rgbd\pairs",           # 可由脚本再生
    r"datasets\ycb_rgbd\ycbv_test_bop19.zip",  # 630MB zip(gitignore已挡)
    r"YCB深度图预览",                      # 24张预览图(可由脚本再生)
    r"dinov3_dual\model_cache",           # 骨干权重缓存 ~2.4GB
    r"ForceSight样本",                     # 临时调试
    r"datasets\cornell_grasping",         # 特征包 260MB(README已附Wayback链接)
    r"datasets\ycb_rgbd\depth_pretrain\feats_cache.pt",  # 14MB 特征缓存可再生
}


def path_ok(rel):
    rel = rel.replace("/", "\\")
    for s in SKIP_DIRS:
        s2 = s.replace("/", "\\")
        if rel == s2 or rel.startswith(s2 + "\\"):
            return False
    return True


def commit(paths_by_label):
    repo = porcelain.open_repo(ROOT)

    add_files = []
    for rel in paths_by_label["add"]:
        if path_ok(rel):
            add_files.append(rel.replace("\\", "/"))
    if add_files:
        porcelain.add(repo=repo, paths=add_files)
    for rel in paths_by_label.get("rm", []):
        porcelain.rm(repo=repo, paths=[rel.replace("\\", "/")])

    print(f"暂存: 新增 {len(add_files)} 个, 删除 {len(paths_by_label.get('rm', []))} 个")
    sha = porcelain.commit(
        repo=repo,
        message=paths_by_label["message"].encode("utf-8"),
        author=b"xzr-vvv <270125681+xzr-vvv@users.noreply.github.com>",
        committer=b"xzr-vvv <270125681+xzr-vvv@users.noreply.github.com>",
    )
    print("commit:", sha.decode()[:9])
    return sha


def push():
    repo = porcelain.open_repo(ROOT)
    res = porcelain.push(repo, "https://github.com/xzr-vvv/mlproject.git", "refs/heads/main")
    print("push 结果:", res)


def log(n=5):
    repo = porcelain.open_repo(ROOT)
    print("HEAD:", repo.head().decode()[:9])
    for e in repo.get_walker(max_entries=n):
        c = e.commit
        print(c.id.decode()[:7], datetime.fromtimestamp(c.commit_time).strftime("%m-%d %H:%M"),
              c.message.decode(errors="ignore").splitlines()[0][:66])


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "log"
    if cmd == "log":
        log(int(sys.argv[2]) if len(sys.argv) > 2 else 5)
    elif cmd == "status":
        repo = porcelain.open_repo(ROOT)
        st = porcelain.status(repo)
        print("未跟踪:")
        for f in st.untracked:
            print("  ", f.decode())
        print("变更:")
        for f in st.unstaged:
            print("  ", f.decode())

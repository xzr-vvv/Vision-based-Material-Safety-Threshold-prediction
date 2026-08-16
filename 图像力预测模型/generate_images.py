# -*- coding: utf-8 -*-
"""
生成6类物体的合成训练图片
每类30张训练图 + 5张测试图
"""
import os
import sys
import math
import random
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

import numpy as np

random.seed(42)
np.random.seed(42)

IMG_SIZE = 224
BASE_DIR = r"E:\A-触觉机器学习\图像力预测模型"
TRAIN_DIR = os.path.join(BASE_DIR, "train_images")
TEST_DIR = os.path.join(BASE_DIR, "test_images")

# 6个类别 + 对应的安全力范围（单位：牛顿）
CLASS_FORCE_RANGE = {
    "metal":            (2.0, 20.0),
    "hard_plastic":     (1.5, 15.0),
    "wood_paper":       (1.0, 12.0),
    "leather_textile":  (0.8, 8.0),
    "foam_soft":        (0.3, 4.0),
    "fragile_glass":    (0.5, 2.5),
}

CLASS_NAMES_CN = {
    "metal": "金属刚体",
    "hard_plastic": "硬塑料",
    "wood_paper": "木材纸板",
    "leather_textile": "皮革织物",
    "foam_soft": "软质泡沫",
    "fragile_glass": "玻璃陶瓷易碎品",
}


def draw_metal(draw, img, cx, cy, size, color_variation=0):
    base_color = (180 + color_variation, 185 + color_variation, 195 + color_variation)
    highlight = (230, 235, 245)
    shadow = (120, 125, 135)
    shape = random.choice(["rect", "circle", "hexagon"])
    if shape == "rect":
        w, h = int(size * 1.2), int(size * 0.7)
        x1, y1 = cx - w//2, cy - h//2
        x2, y2 = cx + w//2, cy + h//2
        draw.rectangle([x1, y1, x2, y2], fill=base_color, outline=shadow, width=2)
        draw.rectangle([x1+5, y1+5, x2-5, y1+h//4], fill=highlight)
        draw.rectangle([x1+5, y2-h//4, x2-5, y2-5], fill=shadow)
    elif shape == "circle":
        r = int(size * 0.5)
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=base_color, outline=shadow, width=2)
        draw.ellipse([cx-r//2, cy-r//2-r//3, cx+r//3, cy-r//6], fill=highlight)
    else:
        r = int(size * 0.55)
        pts = []
        for i in range(6):
            angle = math.radians(60 * i - 30)
            pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
        draw.polygon(pts, fill=base_color, outline=shadow)
        draw.polygon([pts[0], pts[1], (cx, cy)], fill=highlight)
    arr = np.array(img)
    noise = np.random.randint(-8, 8, arr.shape[:2], dtype=np.int16)
    arr = np.clip(arr.astype(np.int16) + noise[:, :, np.newaxis], 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def draw_hard_plastic(draw, img, cx, cy, size, color_variation=0):
    colors = [
        (220, 50, 50), (50, 100, 220), (50, 180, 80),
        (240, 180, 30), (150, 80, 180), (230, 120, 30),
    ]
    base_color = colors[color_variation % len(colors)]
    highlight = tuple(min(255, c + 60) for c in base_color)
    shadow = tuple(max(0, c - 50) for c in base_color)
    shape = random.choice(["rect_rounded", "cylinder", "cube_3d"])
    if shape == "rect_rounded":
        w, h = int(size * 1.0), int(size * 0.8)
        x1, y1 = cx - w//2, cy - h//2
        x2, y2 = cx + w//2, cy + h//2
        draw.rounded_rectangle([x1, y1, x2, y2], radius=15, fill=base_color, outline=shadow, width=2)
        draw.rounded_rectangle([x1+8, y1+8, x2-8, y1+h//3], radius=7, fill=highlight)
    elif shape == "cylinder":
        w, h = int(size * 0.6), int(size * 1.0)
        x1, y1 = cx - w//2, cy - h//2
        x2, y2 = cx + w//2, cy + h//2
        draw.rectangle([x1, y1+h//6, x2, y2-h//6], fill=base_color)
        draw.ellipse([x1, y1, x2, y1+h//3], fill=highlight, outline=shadow)
        draw.ellipse([x1, y2-h//3, x2, y2], fill=shadow, outline=shadow)
    else:
        s = int(size * 0.5)
        draw.rectangle([cx-s, cy-s, cx+s, cy+s], fill=base_color, outline=shadow, width=2)
        offset = int(s * 0.4)
        draw.polygon([(cx-s, cy-s), (cx-s+offset, cy-s-offset),
                      (cx+s+offset, cy-s-offset), (cx+s, cy-s)],
                     fill=highlight, outline=shadow)
        draw.polygon([(cx+s, cy-s), (cx+s+offset, cy-s-offset),
                      (cx+s+offset, cy+s-offset), (cx+s, cy+s)],
                     fill=shadow, outline=shadow)
    return img


def draw_wood_paper(draw, img, cx, cy, size, color_variation=0):
    base_colors = [
        (180, 140, 90), (160, 120, 70),
        (200, 160, 100), (140, 100, 60),
    ]
    base_color = base_colors[color_variation % len(base_colors)]
    grain_color = tuple(max(0, c - 30) for c in base_color)
    w, h = int(size * 1.3), int(size * 0.6)
    x1, y1 = cx - w//2, cy - h//2
    x2, y2 = cx + w//2, cy + h//2
    draw.rectangle([x1, y1, x2, y2], fill=base_color, outline=grain_color, width=2)
    for i in range(5, h-5, max(3, h // 8)):
        y = y1 + i
        offset = random.randint(-2, 2)
        draw.line([x1+3, y+offset, x2-3, y+offset], fill=grain_color, width=1)
    if random.random() > 0.5:
        for i in range(0, w, 10):
            x = x1 + i
            draw.line([x, y1+2, x+5, y1+5], fill=grain_color, width=1)
            draw.line([x+5, y1+5, x+10, y1+2], fill=grain_color, width=1)
    return img


def draw_leather_textile(draw, img, cx, cy, size, color_variation=0):
    colors = [
        (139, 90, 60), (180, 140, 100), (60, 60, 60),
        (100, 80, 150), (80, 120, 80), (170, 70, 70),
    ]
    base_color = colors[color_variation % len(colors)]
    texture_color = tuple(max(0, c - 25) for c in base_color)
    highlight = tuple(min(255, c + 20) for c in base_color)
    shape = random.choice(["rect_fabric", "folded", "round_patch"])
    if shape == "rect_fabric":
        w, h = int(size * 1.1), int(size * 0.9)
        x1, y1 = cx - w//2, cy - h//2
        x2, y2 = cx + w//2, cy + h//2
        draw.rectangle([x1, y1, x2, y2], fill=base_color)
        for i in range(0, w, 8):
            draw.line([x1+i, y1, x1+i, y2], fill=texture_color, width=1)
        for j in range(0, h, 8):
            draw.line([x1, y1+j, x2, y1+j], fill=highlight, width=1)
    elif shape == "folded":
        pts = [
            (cx - size//2, cy - size//3),
            (cx + size//2, cy - size//2),
            (cx + size//3, cy + size//2),
            (cx - size//2, cy + size//3),
        ]
        draw.polygon(pts, fill=base_color, outline=texture_color)
        draw.line([(cx - size//4, cy - size//3), (cx - size//6, cy + size//3)],
                  fill=texture_color, width=2)
    else:
        r = int(size * 0.45)
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=base_color, outline=texture_color, width=2)
        for _ in range(20):
            dx = random.randint(-r+5, r-5)
            dy = random.randint(-r+5, r-5)
            if dx*dx + dy*dy < (r-5)**2:
                draw.ellipse([cx+dx-1, cy+dy-1, cx+dx+1, cy+dy+1], fill=texture_color)
    return img


def draw_foam_soft(draw, img, cx, cy, size, color_variation=0):
    colors = [
        (255, 220, 200), (255, 240, 200), (220, 230, 240),
        (230, 220, 240), (240, 240, 240), (255, 200, 180),
    ]
    base_color = colors[color_variation % len(colors)]
    hole_color = tuple(max(0, c - 30) for c in base_color)
    shape = random.choice(["cloud", "sponge_cube", "pastry_round"])
    if shape == "cloud":
        r = int(size * 0.3)
        draw.ellipse([cx-r, cy-r//2, cx+r, cy+r//2], fill=base_color)
        draw.ellipse([cx-r*2//3, cy-r, cx+r*2//3, cy], fill=base_color)
        draw.ellipse([cx-r//2, cy-r*2//3, cx+r*3//2, cy+r//3], fill=base_color)
        draw.ellipse([cx-r*3//2, cy-r//3, cx-r//4, cy+r//2], fill=base_color)
    elif shape == "sponge_cube":
        s = int(size * 0.5)
        draw.rounded_rectangle([cx-s, cy-s, cx+s, cy+s], radius=15, fill=base_color,
                               outline=tuple(max(0,c-20) for c in base_color), width=2)
    else:
        r = int(size * 0.45)
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=base_color,
                     outline=tuple(max(0,c-20) for c in base_color), width=2)
        draw.ellipse([cx-r//2, cy-r*2//3, cx+r//2, cy-r//4], fill=tuple(min(255,c+20) for c in base_color))
    arr = np.array(img)
    gray = np.mean(arr, axis=2)
    mask = np.abs(gray - np.mean(gray)) > 3
    holes = np.random.random(arr.shape[:2]) < 0.02
    holes = holes & mask
    arr[holes] = hole_color
    return Image.fromarray(arr)


def draw_fragile_glass(draw, img, cx, cy, size, color_variation=0):
    colors = [
        (200, 230, 240), (220, 220, 220), (200, 180, 160),
        (180, 200, 220), (240, 220, 200),
    ]
    base_color = colors[color_variation % len(colors)]
    highlight = (255, 255, 255)
    edge = tuple(max(0, c - 40) for c in base_color)
    shape = random.choice(["wine_glass", "mug", "egg", "bottle", "plate"])
    if shape == "wine_glass":
        bw = int(size * 0.3)
        bh = int(size * 0.4)
        bx, by = cx - bw//2, cy - size//2
        draw.polygon([(bx, by), (bx+bw, by), (bx+bw*3//4, by+bh), (bx+bw//4, by+bh)],
                     fill=base_color, outline=edge)
        draw.rectangle([cx-3, by+bh, cx+3, by+bh+size//3], fill=edge)
        draw.ellipse([cx-size//4, by+bh+size//3, cx+size//4, by+bh+size//3+8], fill=base_color, outline=edge)
        draw.line([(bx+bw//4, by+5), (bx+bw//3, by+bh-5)], fill=highlight, width=2)
    elif shape == "mug":
        w, h = int(size * 0.5), int(size * 0.7)
        x1, y1 = cx - w//2, cy - h//2
        x2, y2 = cx + w//2, cy + h//2
        draw.rectangle([x1, y1, x2, y2], fill=base_color, outline=edge, width=2)
        draw.ellipse([x1, y1-5, x2, y1+8], fill=highlight, outline=edge)
        draw.arc([x2, y1+h//4, x2+size//3, y2-h//4], start=-90, end=90, fill=base_color, width=8)
        draw.arc([x2, y1+h//4, x2+size//3, y2-h//4], start=-90, end=90, fill=edge, width=2)
    elif shape == "egg":
        rx, ry = int(size * 0.3), int(size * 0.4)
        draw.ellipse([cx-rx, cy-ry, cx+rx, cy+ry], fill=base_color, outline=edge, width=2)
        draw.ellipse([cx-rx//2, cy-ry*2//3, cx-rx//6, cy-ry//3], fill=highlight)
    elif shape == "bottle":
        bw = int(size * 0.35)
        bh = int(size * 0.8)
        bx, by = cx - bw//2, cy - bh//2
        draw.rounded_rectangle([bx, by+bh//4, bx+bw, by+bh], radius=5, fill=base_color, outline=edge)
        draw.rectangle([cx-bw//4, by, cx+bw//4, by+bh//3], fill=base_color, outline=edge)
        draw.rectangle([cx-bw//3, by-5, cx+bw//3, by+5], fill=edge)
        draw.line([(bx+bw//4, by+bh//3), (bx+bw//4, by+bh-5)], fill=highlight, width=2)
    else:
        rx, ry = int(size * 0.5), int(size * 0.2)
        draw.ellipse([cx-rx, cy-ry, cx+rx, cy+ry], fill=base_color, outline=edge, width=3)
        draw.ellipse([cx-rx*3//4, cy-ry*2//3, cx+rx*3//4, cy+ry*2//3], fill=highlight, outline=edge)
    return img


DRAW_FUNCTIONS = {
    "metal": draw_metal,
    "hard_plastic": draw_hard_plastic,
    "wood_paper": draw_wood_paper,
    "leather_textile": draw_leather_textile,
    "foam_soft": draw_foam_soft,
    "fragile_glass": draw_fragile_glass,
}


def generate_image(cls, idx, output_dir, is_test=False):
    bg_choices = [
        (245, 245, 245), (230, 230, 225), (250, 248, 245),
        (240, 240, 245), (235, 235, 230),
    ]
    bg = random.choice(bg_choices)
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), bg)
    draw = ImageDraw.Draw(img)
    arr = np.array(img)
    noise = np.random.randint(-5, 5, arr.shape[:2], dtype=np.int16)
    arr = np.clip(arr.astype(np.int16) + noise[:, :, np.newaxis], 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)

    cx = IMG_SIZE // 2 + random.randint(-20, 20)
    cy = IMG_SIZE // 2 + random.randint(-20, 20)
    size = random.randint(90, 140)

    color_var = idx if not is_test else idx + 100
    func = DRAW_FUNCTIONS[cls]
    img = func(draw, img, cx, cy, size, color_var)

    img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.8)))
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.9, 1.1))
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(random.uniform(0.9, 1.15))

    filename = f"{cls}_{idx:03d}.png"
    filepath = os.path.join(output_dir, cls, filename)
    img.save(filepath)
    return filepath


def main():
    for cls in CLASS_FORCE_RANGE:
        os.makedirs(os.path.join(TRAIN_DIR, cls), exist_ok=True)
        for i in range(30):
            generate_image(cls, i, TRAIN_DIR, is_test=False)
        print(f"  {cls} (train): 30")

    for cls in CLASS_FORCE_RANGE:
        os.makedirs(os.path.join(TEST_DIR, cls), exist_ok=True)
        for i in range(5):
            generate_image(cls, i, TEST_DIR, is_test=True)
        print(f"  {cls} (test): 5")

    import csv
    label_file = os.path.join(BASE_DIR, "force_labels.csv")
    with open(label_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_name", "class_name_cn", "min_grasp_force_N", "max_safe_force_N", "safe_range_N"])
        for cls, (min_f, max_f) in CLASS_FORCE_RANGE.items():
            writer.writerow([cls, CLASS_NAMES_CN[cls], min_f, max_f, max_f - min_f])

    print(f"\nDone! Total: 180 train + 30 test = 210 images")
    print(f"Labels: {label_file}")


if __name__ == "__main__":
    main()

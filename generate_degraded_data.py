"""
Generate degraded NeRF Synthetic datasets for 3DGS experiments.

Usage:
    cd ~/autodl-tmp/gaussian-splatting
    python generate_degraded_data.py

Output structure:
    data/nerf_synthetic_degraded/
        lego_down2/     lego_down4/     lego_down8/
        lego_noise10/   lego_noise25/   lego_noise50/
        lego_blur5/     lego_blur15/    lego_blur25/
        lego_jpeg10/    lego_jpeg30/    lego_jpeg50/
        lego_mixed/
        (same for chair, hotdog, mic)
"""

import os
import sys
import json
import shutil
import cv2
import numpy as np
from pathlib import Path


SCENES = ["lego", "chair", "hotdog", "mic"]
SRC_ROOT = "data/nerf_synthetic"
DST_ROOT = "data/nerf_synthetic_degraded"


def downsample(img, scale):
    h, w = img.shape[:2]
    return cv2.resize(img, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)


def add_gaussian_noise(img, sigma):
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def add_motion_blur(img, kernel_size):
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, :] = 1.0 / kernel_size
    return cv2.filter2D(img, -1, kernel)


def jpeg_compress(img, quality):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode(".jpg", img, encode_param)
    return cv2.imdecode(encimg, cv2.IMREAD_UNCHANGED)


def apply_degradation(rgba_img, deg_type, deg_param):
    rgb = rgba_img[:, :, :3]
    alpha = rgba_img[:, :, 3]

    if deg_type == "down":
        rgb_deg = downsample(rgb, deg_param)
        h, w = rgb_deg.shape[:2]
        alpha_deg = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_CUBIC)
        return np.dstack([rgb_deg, alpha_deg])

    if deg_type == "noise":
        rgb_deg = add_gaussian_noise(rgb, deg_param)
        return np.dstack([rgb_deg, alpha])

    if deg_type == "blur":
        rgb_deg = add_motion_blur(rgb, deg_param)
        return np.dstack([rgb_deg, alpha])

    if deg_type == "jpeg":
        rgb_deg = jpeg_compress(rgb, deg_param)
        return np.dstack([rgb_deg, alpha])

    if deg_type == "mixed":
        scale, sigma = deg_param
        rgb_deg = downsample(rgb, scale)
        h, w = rgb_deg.shape[:2]
        alpha_deg = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_CUBIC)
        rgb_deg = add_gaussian_noise(rgb_deg, sigma)
        return np.dstack([rgb_deg, alpha_deg])


def create_degraded_scene(scene, deg_type, deg_param, label):
    src_dir = os.path.join(SRC_ROOT, scene)
    dst_dir = os.path.join(DST_ROOT, f"{scene}_{label}")

    if os.path.exists(dst_dir):
        print(f"  [skip] {dst_dir} already exists")
        return

    os.makedirs(os.path.join(dst_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(dst_dir, "test"), exist_ok=True)

    for json_name in ["transforms_train.json", "transforms_test.json"]:
        src_json = os.path.join(src_dir, json_name)
        if os.path.exists(src_json):
            shutil.copy2(src_json, os.path.join(dst_dir, json_name))

    if os.path.exists(os.path.join(src_dir, "transforms_val.json")):
        shutil.copy2(
            os.path.join(src_dir, "transforms_val.json"),
            os.path.join(dst_dir, "transforms_val.json"),
        )

    train_dir = os.path.join(src_dir, "train")
    train_files = sorted([f for f in os.listdir(train_dir) if f.endswith(".png")])
    for fname in train_files:
        img = cv2.imread(os.path.join(train_dir, fname), cv2.IMREAD_UNCHANGED)
        img_deg = apply_degradation(img, deg_type, deg_param)
        cv2.imwrite(os.path.join(dst_dir, "train", fname), img_deg)

    test_dir = os.path.join(src_dir, "test")
    if os.path.exists(test_dir):
        test_files = sorted([f for f in os.listdir(test_dir) if f.endswith(".png")])
        for fname in test_files:
            shutil.copy2(
                os.path.join(test_dir, fname), os.path.join(dst_dir, "test", fname)
            )

    print(f"  [done] {dst_dir}  ({len(train_files)} train, {len(test_files) if os.path.exists(test_dir) else 0} test)")


DEGRADATIONS = [
    ("down", 2, "down2"),
    ("down", 4, "down4"),
    ("down", 8, "down8"),
    ("noise", 10, "noise10"),
    ("noise", 25, "noise25"),
    ("noise", 50, "noise50"),
    ("blur", 5, "blur5"),
    ("blur", 15, "blur15"),
    ("blur", 25, "blur25"),
    ("jpeg", 50, "jpeg50"),
    ("jpeg", 30, "jpeg30"),
    ("jpeg", 10, "jpeg10"),
    ("mixed", (4, 15), "mixed"),
]


if __name__ == "__main__":
    if not os.path.exists(SRC_ROOT):
        print(f"Error: {SRC_ROOT} not found. Run from gaussian-splatting root directory.")
        sys.exit(1)

    os.makedirs(DST_ROOT, exist_ok=True)

    for scene in SCENES:
        if not os.path.exists(os.path.join(SRC_ROOT, scene)):
            print(f"Warning: {scene} not found, skipping.")
            continue
        print(f"\n=== {scene} ===")
        for deg_type, deg_param, label in DEGRADATIONS:
            create_degraded_scene(scene, deg_type, deg_param, label)

    total = len(SCENES) * len(DEGRADATIONS)
    print(f"\nDone. Generated {total} degraded datasets in {DST_ROOT}/")

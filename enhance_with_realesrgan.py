"""
Enhance degraded NeRF Synthetic training images using Real-ESRGAN.

This script processes degraded datasets (from generate_degraded_data.py)
with Real-ESRGAN super-resolution, producing enhanced training images
for the Chapter 5 comparison experiments.

Usage:
    cd ~/autodl-tmp/gaussian-splatting
    python enhance_with_realesrgan.py

Output structure:
    data/nerf_synthetic_enhanced/
        lego_down4_esrgan/
            train/          (enhanced images)
            test/           (original test images, copied)
            transforms_*.json (copied from degraded)
        chair_down4_esrgan/
        hotdog_down4_esrgan/
        mic_down4_esrgan/
"""

import os
import sys
import glob
import shutil
import cv2
import numpy as np
from pathlib import Path

SCENES = ["lego", "chair", "hotdog", "mic"]
DEGRADED_ROOT = "data/nerf_synthetic_degraded"
ENHANCED_ROOT = "data/nerf_synthetic_enhanced"
ORIGINAL_ROOT = "data/nerf_synthetic"

DEGRADATION_CONFIGS = [
    ("down4", 4),
]


def enhance_scene(scene, deg_label, upscale_factor):
    """Enhance a degraded scene's training images with Real-ESRGAN."""
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    import torch

    src_dir = os.path.join(DEGRADED_ROOT, f"{scene}_{deg_label}")
    dst_dir = os.path.join(ENHANCED_ROOT, f"{scene}_{deg_label}_esrgan")
    orig_dir = os.path.join(ORIGINAL_ROOT, scene)

    if not os.path.exists(src_dir):
        print(f"  [skip] {src_dir} not found")
        return

    if os.path.exists(dst_dir):
        print(f"  [skip] {dst_dir} already exists")
        return

    os.makedirs(os.path.join(dst_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(dst_dir, "test"), exist_ok=True)

    model = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64,
        num_block=23, num_grow_ch=32, scale=4
    )
    model_path = os.path.join(
        os.path.expanduser("~"), "autodl-tmp", "Real-ESRGAN",
        "weights", "RealESRGAN_x4plus.pth"
    )
    if not os.path.exists(model_path):
        model_path = "weights/RealESRGAN_x4plus.pth"

    upsampler = RealESRGANer(
        scale=4, model_path=model_path, model=model,
        tile=0, tile_pad=10, pre_pad=0, half=True
    )

    for json_name in ["transforms_train.json", "transforms_test.json", "transforms_val.json"]:
        src_json = os.path.join(src_dir, json_name)
        if os.path.exists(src_json):
            shutil.copy2(src_json, os.path.join(dst_dir, json_name))

    train_dir = os.path.join(src_dir, "train")
    train_files = sorted([f for f in os.listdir(train_dir) if f.endswith(".png")])
    orig_h, orig_w = 800, 800

    print(f"  Enhancing {len(train_files)} training images...")
    for i, fname in enumerate(train_files):
        img = cv2.imread(os.path.join(train_dir, fname), cv2.IMREAD_UNCHANGED)
        has_alpha = img.shape[2] == 4

        if has_alpha:
            rgb = img[:, :, :3]
            alpha = img[:, :, 3]
        else:
            rgb = img

        output, _ = upsampler.enhance(rgb, outscale=upscale_factor)

        target_h, target_w = orig_h, orig_w
        if output.shape[0] != target_h or output.shape[1] != target_w:
            output = cv2.resize(output, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

        if has_alpha:
            alpha_resized = cv2.resize(alpha, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
            output = np.dstack([output, alpha_resized])

        cv2.imwrite(os.path.join(dst_dir, "train", fname), output)

        if (i + 1) % 20 == 0 or i == 0:
            print(f"    [{i+1}/{len(train_files)}] {fname}")

    test_src = os.path.join(orig_dir, "test")
    if os.path.exists(test_src):
        test_files = sorted([f for f in os.listdir(test_src) if f.endswith(".png")])
        for fname in test_files:
            shutil.copy2(
                os.path.join(test_src, fname),
                os.path.join(dst_dir, "test", fname)
            )
        print(f"  Copied {len(test_files)} test images from original dataset")

    print(f"  [done] {dst_dir}")


if __name__ == "__main__":
    if not os.path.exists(DEGRADED_ROOT):
        print(f"Error: {DEGRADED_ROOT} not found.")
        sys.exit(1)

    os.makedirs(ENHANCED_ROOT, exist_ok=True)

    for scene in SCENES:
        print(f"\n=== {scene} ===")
        for deg_label, upscale in DEGRADATION_CONFIGS:
            enhance_scene(scene, deg_label, upscale)

    print(f"\nDone. Enhanced datasets saved to {ENHANCED_ROOT}/")

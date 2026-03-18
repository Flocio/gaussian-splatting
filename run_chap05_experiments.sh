#!/bin/bash
# Chapter 5 experiments: Comparison experiments
# Run this after enhance_with_realesrgan.py has completed.
#
# Usage:
#   cd ~/autodl-tmp/gaussian-splatting
#   screen -S chap05
#   bash run_chap05_experiments.sh 2>&1 | tee chap05_results.txt
#
# Experiments:
#   1. Preprocess+3DGS: Enhanced images -> standard 3DGS training
#   (Original 3DGS on low-quality and Original data baseline already done in Chapter 3)

set -e
cd ~/autodl-tmp/gaussian-splatting
export OMP_NUM_THREADS=4

RESULTS_FILE="chap05_results.txt"

echo "=== Chapter 5 Experiment Results ===" >> $RESULTS_FILE
echo "Started: $(date)" >> $RESULTS_FILE

scenes=("lego" "chair" "hotdog" "mic")

# ============================================================
# Step 1: Preprocess + 3DGS (Real-ESRGAN enhanced x4 -> 3DGS)
# ============================================================
echo "" >> $RESULTS_FILE
echo "=== Preprocess + 3DGS (down4 enhanced with Real-ESRGAN) ===" >> $RESULTS_FILE

for scene in "${scenes[@]}"; do
    echo "" >> $RESULTS_FILE
    echo "--- ${scene} preprocess+3DGS ---" >> $RESULTS_FILE

    python train.py \
        -s data/nerf_synthetic_enhanced/${scene}_down4_esrgan \
        -m output/${scene}_down4_preprocess \
        --eval --white_background

    find output/ -name ".DS_Store" -delete

    python render.py -m output/${scene}_down4_preprocess

    python metrics.py -m output/${scene}_down4_preprocess >> $RESULTS_FILE 2>&1

    echo "--- ${scene} preprocess+3DGS done at $(date) ---" >> $RESULTS_FILE
done

echo "" >> $RESULTS_FILE
echo "=== All Chapter 5 comparison experiments completed at $(date) ===" >> $RESULTS_FILE
echo "Done!"

#!/bin/bash
set -e
cd ~/autodl-tmp/gaussian-splatting
export OMP_NUM_THREADS=4

RESULTS_FILE="noise_robust_results.txt"

echo "=== Noise Robustness Experiment Results ===" >> $RESULTS_FILE
echo "Started: $(date)" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE
echo "Data: noise_25 degraded (NO preprocessing)" >> $RESULTS_FILE
echo "Method: Perceptual Loss + Improved Density Control (train_ours.py)" >> $RESULTS_FILE
echo "lambda_lpips=0.05, densify_grad_threshold=0.0005, opacity_reset_interval=1500" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE

scenes=("lego" "chair" "hotdog" "mic")

for scene in "${scenes[@]}"; do
    echo "" >> $RESULTS_FILE
    echo "--- ${scene} noise25_robust ---" >> $RESULTS_FILE
    echo "[$(date)] Training ${scene}..."

    python train_ours.py \
        -s data/nerf_synthetic_degraded/${scene}_noise25 \
        -m output/${scene}_noise25_robust \
        --eval --white_background \
        --lambda_lpips 0.05 \
        --densify_grad_threshold_override 0.0005 \
        --opacity_reset_interval_override 1500

    find output/ -name ".DS_Store" -delete
    python render.py -m output/${scene}_noise25_robust
    python metrics.py -m output/${scene}_noise25_robust >> $RESULTS_FILE 2>&1

    echo "--- ${scene} noise25_robust done at $(date) ---" >> $RESULTS_FILE
    echo "[$(date)] ${scene} done."
done

echo "" >> $RESULTS_FILE
echo "=== All noise robustness experiments completed at $(date) ===" >> $RESULTS_FILE
echo "All done! Results in $RESULTS_FILE"

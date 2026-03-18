#!/bin/bash
set -e
cd ~/autodl-tmp/gaussian-splatting
export OMP_NUM_THREADS=4

RESULTS_FILE="ours_v2_results.txt"

echo "=== Our Method v2 Experiment Results ===" >> $RESULTS_FILE
echo "Started: $(date)" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE
echo "Method: Preprocess(Real-ESRGAN) + Perceptual Loss + Improved Density Control" >> $RESULTS_FILE
echo "lambda_lpips=0.05, densify_grad_threshold=0.0005, opacity_reset_interval=1500" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE

scenes=("lego" "chair" "hotdog" "mic")

for scene in "${scenes[@]}"; do
    echo "" >> $RESULTS_FILE
    echo "--- ${scene} ours_v2 ---" >> $RESULTS_FILE
    echo "[$(date)] Training ${scene}..."

    python train_ours.py \
        -s data/nerf_synthetic_enhanced/${scene}_down4_esrgan \
        -m output/${scene}_down4_ours_v2 \
        --eval --white_background \
        --lambda_lpips 0.05 \
        --densify_grad_threshold_override 0.0005 \
        --opacity_reset_interval_override 1500

    find output/ -name ".DS_Store" -delete
    python render.py -m output/${scene}_down4_ours_v2
    python metrics.py -m output/${scene}_down4_ours_v2 >> $RESULTS_FILE 2>&1

    echo "--- ${scene} ours_v2 done at $(date) ---" >> $RESULTS_FILE
    echo "[$(date)] ${scene} done."
done

echo "" >> $RESULTS_FILE
echo "=== All v2 experiments completed at $(date) ===" >> $RESULTS_FILE
echo "All done! Results in $RESULTS_FILE"

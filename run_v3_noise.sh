#!/bin/bash
set -e
cd ~/autodl-tmp/gaussian-splatting
export OMP_NUM_THREADS=4

RESULTS_FILE="v3_noise_results.txt"

echo "=== V3 Noise Experiment Results ===" >> $RESULTS_FILE
echo "Started: $(date)" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE
echo "Data: noise_25 degraded (NO preprocessing)" >> $RESULTS_FILE
echo "Method: Perceptual Loss + TV Reg + Freq Anneal + Improved Density Control" >> $RESULTS_FILE
echo "lambda_lpips=0.01, lambda_tv=0.01, freq_anneal_until=10000, max_down=2.0" >> $RESULTS_FILE
echo "densify_grad_threshold=0.0003, opacity_reset_interval=2000" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE

scenes=("lego" "chair" "hotdog" "mic")

for scene in "${scenes[@]}"; do
    echo "" >> $RESULTS_FILE
    echo "--- ${scene} v3_noise ---" >> $RESULTS_FILE
    echo "[$(date)] Training ${scene}..."

    python train_v3.py \
        -s data/nerf_synthetic_degraded/${scene}_noise25 \
        -m output/${scene}_noise25_v3 \
        --eval --white_background \
        --lambda_lpips 0.01 \
        --lambda_tv 0.01 \
        --freq_anneal_until 10000 \
        --freq_anneal_max_down 2.0 \
        --densify_grad_threshold_override 0.0003 \
        --opacity_reset_interval_override 2000

    find output/ -name ".DS_Store" -delete
    python render.py -m output/${scene}_noise25_v3
    python metrics.py -m output/${scene}_noise25_v3 >> $RESULTS_FILE 2>&1

    echo "--- ${scene} v3_noise done at $(date) ---" >> $RESULTS_FILE
    echo "[$(date)] ${scene} done."
done

echo "" >> $RESULTS_FILE
echo "=== All V3 noise experiments completed at $(date) ===" >> $RESULTS_FILE
echo "All done! Results in $RESULTS_FILE"

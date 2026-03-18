#!/bin/bash
# Chapter 3 experiments: degradation impact analysis
# Run with: screen -S chap03 bash run_chap03_experiments.sh
# Check progress: screen -r chap03
# Detach: Ctrl+A then D

cd ~/autodl-tmp/gaussian-splatting
source /root/miniconda3/bin/activate
conda activate gaussian_splatting
export OMP_NUM_THREADS=4

SCENES=("lego" "chair" "hotdog" "mic")
LOG_FILE="chap03_results.txt"
echo "=== Chapter 3 Experiment Results ===" > $LOG_FILE
echo "Started: $(date)" >> $LOG_FILE

run_experiment() {
    local data_path=$1
    local output_name=$2
    local label=$3

    echo ""
    echo "============================================"
    echo "Running: $label"
    echo "  Data: $data_path"
    echo "  Output: output/$output_name"
    echo "============================================"

    python train.py -s "$data_path" -m "output/$output_name" --eval --white_background
    find output/ -name ".DS_Store" -delete
    python render.py -m "output/$output_name"
    python metrics.py -m "output/$output_name" 2>&1 | tee -a $LOG_FILE

    echo "--- $label done at $(date) ---" >> $LOG_FILE
    echo ""
}

# === Resolution degradation (Table 3.1) ===
echo "" >> $LOG_FILE
echo "=== Resolution Degradation ===" >> $LOG_FILE
for scene in "${SCENES[@]}"; do
    run_experiment "data/nerf_synthetic_degraded/${scene}_down2" "${scene}_down2" "${scene} downsample x2"
    run_experiment "data/nerf_synthetic_degraded/${scene}_down4" "${scene}_down4" "${scene} downsample x4"
    run_experiment "data/nerf_synthetic_degraded/${scene}_down8" "${scene}_down8" "${scene} downsample x8"
done

# === Noise degradation (Table 3.2) ===
echo "" >> $LOG_FILE
echo "=== Noise Degradation ===" >> $LOG_FILE
for scene in "${SCENES[@]}"; do
    run_experiment "data/nerf_synthetic_degraded/${scene}_noise10" "${scene}_noise10" "${scene} noise sigma=10"
    run_experiment "data/nerf_synthetic_degraded/${scene}_noise25" "${scene}_noise25" "${scene} noise sigma=25"
    run_experiment "data/nerf_synthetic_degraded/${scene}_noise50" "${scene}_noise50" "${scene} noise sigma=50"
done

# === Motion blur degradation (Table 3.3) ===
echo "" >> $LOG_FILE
echo "=== Motion Blur Degradation ===" >> $LOG_FILE
for scene in "${SCENES[@]}"; do
    run_experiment "data/nerf_synthetic_degraded/${scene}_blur5" "${scene}_blur5" "${scene} blur k=5"
    run_experiment "data/nerf_synthetic_degraded/${scene}_blur15" "${scene}_blur15" "${scene} blur k=15"
    run_experiment "data/nerf_synthetic_degraded/${scene}_blur25" "${scene}_blur25" "${scene} blur k=25"
done

# === JPEG compression degradation (Table 3.4) ===
echo "" >> $LOG_FILE
echo "=== JPEG Compression Degradation ===" >> $LOG_FILE
for scene in "${SCENES[@]}"; do
    run_experiment "data/nerf_synthetic_degraded/${scene}_jpeg50" "${scene}_jpeg50" "${scene} jpeg q=50"
    run_experiment "data/nerf_synthetic_degraded/${scene}_jpeg30" "${scene}_jpeg30" "${scene} jpeg q=30"
    run_experiment "data/nerf_synthetic_degraded/${scene}_jpeg10" "${scene}_jpeg10" "${scene} jpeg q=10"
done

echo "" >> $LOG_FILE
echo "=== All experiments completed at $(date) ===" >> $LOG_FILE
echo ""
echo "All done! Results saved to $LOG_FILE"

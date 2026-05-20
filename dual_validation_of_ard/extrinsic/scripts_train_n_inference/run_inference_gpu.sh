#!/bin/bash
#SBATCH --job-name=cloud-inference-stratified
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --gpus-per-node=1
#SBATCH --output=inference/inference_stratified_%j.out
#SBATCH --error=inference/inference_stratified_%j.err

echo "=========================================="
echo "RUNNING STRATIFIED INFERENCE ON ALL MODELS"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

# --- Environment Setup (Baldo HPC) ---
source /home/username/anaconda3/etc/profile.d/conda.sh
conda activate prithvi_py311
module purge
module load CUDA/11.8.0

# Change to working directory
WORK_DIR="/home/username/prithvi_work/prov_cloud_imputation"
cd "$WORK_DIR"

# --- Configuration Paths (Updated from ls output) ---
EXPERIMENTS_DIR="$WORK_DIR/experiments_unified"
OUTPUT_BASE="inference" #relative path for rocrate lib

# Assuming these two external paths are still correct based on your setup
ZARR_PATH="/MOD09GA_165_184_2021.zarr"
PRITHVI_WEIGHTS="/Prithvi-EO-2.0-300M/Prithvi_EO_V2_300M.pt"

# Updated to point directly to your working directory
DIFFICULTY_CSV="$WORK_DIR/ard_results_parallel/dataset_difficulty_metrics_2021.csv"


# 1. --- Run Trivial Baselines (Oracle & Mosaicking) ---
echo "--- Running Inference: Trivial Baselines ---"
python evaluate_stratified_inference.py \
    --model_type "trivial" \
    --zarr_path "$ZARR_PATH" \
    --difficulty_csv "$DIFFICULTY_CSV" \
    --output_dir "$OUTPUT_BASE/trivial" \
    --num_vis 20

# 2. --- Run Inference for 3D U-Net Baseline ---
echo "--- Running Inference: 3D U-Net Baseline ---"
python evaluate_stratified_inference.py \
    --checkpoint_path "$EXPERIMENTS_DIR/Baseline_3D_UNet_final.pth" \
    --model_type "unet" \
    --zarr_path "$ZARR_PATH" \
    --difficulty_csv "$DIFFICULTY_CSV" \
    --output_dir "$OUTPUT_BASE/unet" \
    --num_vis 20

# 3. --- Run Inference for Prithvi (Zero-Shot) ---
echo "--- Running Inference: Prithvi (Zero-Shot) ---"
python evaluate_stratified_inference.py \
    --model_type "prithvi_zeroshot" \
    --zarr_path "$ZARR_PATH" \
    --difficulty_csv "$DIFFICULTY_CSV" \
    --output_dir "$OUTPUT_BASE/prithvi_zeroshot" \
    --prithvi_weights "$PRITHVI_WEIGHTS" \
    --num_vis 20

# 4. --- Run Inference for Prithvi (Frozen) ---
echo "--- Running Inference: Prithvi (Frozen) ---"
python evaluate_stratified_inference.py \
    --checkpoint_path "$EXPERIMENTS_DIR/Prithvi_Frozen_final.pth" \
    --model_type "prithvi_frozen" \
    --zarr_path "$ZARR_PATH" \
    --difficulty_csv "$DIFFICULTY_CSV" \
    --output_dir "$OUTPUT_BASE/prithvi_frozen" \
    --prithvi_weights "$PRITHVI_WEIGHTS" \
    --num_vis 20

# 5. --- Run Inference for Prithvi (Partial Fine-Tune) ---
echo "--- Running Inference: Prithvi (Partial Fine-Tune) ---"
python evaluate_stratified_inference.py \
    --checkpoint_path "$EXPERIMENTS_DIR/Prithvi_Partial_Finetune_final.pth" \
    --model_type "prithvi_partial" \
    --zarr_path "$ZARR_PATH" \
    --difficulty_csv "$DIFFICULTY_CSV" \
    --output_dir "$OUTPUT_BASE/prithvi_partial" \
    --prithvi_weights "$PRITHVI_WEIGHTS" \
    --num_vis 20

echo "=========================================="
echo "All stratified inference runs complete."
echo "Results saved in: $OUTPUT_BASE/"
echo "=========================================="

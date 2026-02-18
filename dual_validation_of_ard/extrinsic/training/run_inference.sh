#!/bin/bash
#SBATCH --job-name=cloud-inference
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/inference_timeseries-%j.out
#SBATCH --error=logs/inference_timeseries-%j.err

echo "=========================================="
echo "RUNNING INFERENCE ON ALL MODELS"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

# --- Environment Setup ---
source /home/tadiebirihan.medimem/anaconda3/etc/profile.d/conda.sh
conda activate prithvi_py311
module purge
module load CUDA/11.8.0

# --- Configuration ---
WORK_DIR="/home/tadiebirihan.medimem/prithvi_work/cloud_imputation"
TRAINING_RUN_1_DIR="$WORK_DIR/experiments_20251009_074242"
RESUME_RUN_DIR="$WORK_DIR/experiments_20251010_092758"
ZARR_PATH="/home/tadiebirihan.medimem/prithvi_work/mydata/MOD09GA_CentralEurope_ROI.zarr"
PRITHVI_WEIGHTS="/home/tadiebirihan.medimem/prithvi_work/Prithvi-EO-2.0-300M/Prithvi_EO_V2_300M.pt"

cd "$WORK_DIR"

# --- Run Inference for 3D U-Net Baseline ---
echo "--- Running Inference: 3D U-Net Baseline ---"
python inference3.py \
    --checkpoint_path "$TRAINING_RUN_1_DIR/baseline/baseline_unet.pth" \
    --model_type "unet" \
    --zarr_path "$ZARR_PATH" \
    --output_dir "$TRAINING_RUN_1_DIR/baseline/inference_results3"\
    --num_vis 50

# --- Run Inference for Prithvi (Zero-Shot) ---
echo "--- Running Inference: Prithvi (Zero-Shot) ---"
python inference3.py \
    --checkpoint_path "$TRAINING_RUN_1_DIR/prithvi/prithvi_zeroshot.pth" \
    --model_type "prithvi_zeroshot" \
    --zarr_path "$ZARR_PATH" \
    --output_dir "$TRAINING_RUN_1_DIR/prithvi/inference_results_zeroshot3" \
    --prithvi_weights "$PRITHVI_WEIGHTS"\
    --num_vis 50

# --- Run Inference for Prithvi (Frozen) ---
echo "--- Running Inference: Prithvi (Frozen) ---"
python inference3.py \
    --checkpoint_path "$TRAINING_RUN_1_DIR/prithvi/prithvi_frozen.pth" \
    --model_type "prithvi_frozen" \
    --zarr_path "$ZARR_PATH" \
    --output_dir "$TRAINING_RUN_1_DIR/prithvi/inference_results_frozen3" \
    --prithvi_weights "$PRITHVI_WEIGHTS"\
    --num_vis 50

# --- Run Inference for Prithvi (Partial Fine-Tune) ---
echo "--- Running Inference: Prithvi (Partial Fine-Tune) ---"
python inference3.py \
    --checkpoint_path "$RESUME_RUN_DIR/prithvi/prithvi_partial_finetune.pth" \
    --model_type "prithvi_partial" \
    --zarr_path "$ZARR_PATH" \
    --output_dir "$RESUME_RUN_DIR/prithvi/inference_results_partial3" \
    --prithvi_weights "$PRITHVI_WEIGHTS"\
    --num_vis 50

# --- Run Trivial Baselines ---
echo "--- Running Evaluation: Trivial Baselines ---"
python evaluate_trivial_baselines3.py \
    --zarr_path "$ZARR_PATH" \
    --num_vis 50

echo "=========================================="
echo "All inference runs complete."
echo "=========================================="
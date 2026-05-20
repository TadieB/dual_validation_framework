#!/bin/bash
#SBATCH --job-name=generate_vis_patches
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --gpus-per-node=1
#SBATCH --output=inference/generate_vis_%j.out
#SBATCH --error=inference/generate_vis_%j.err

echo "=========================================="
echo "GENERATING PPTX VISUALIZATION PATCHES"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

source /home/username/anaconda3/etc/profile.d/conda.sh
conda activate prithvi_py311
module purge
module load CUDA/11.8.0

WORK_DIR="/home/username/prithvi_work/prov_cloud_imputation"
cd "$WORK_DIR"

EXPERIMENTS_DIR="$WORK_DIR/experiments_unified"
OUTPUT_BASE="inference/pptx" 
ZARR_PATH="$WORK_DIR/ard_results_parallel/MOD09GA_165_184_2021.zarr"
PRITHVI_WEIGHTS="/Prithvi-EO-2.0-300M/Prithvi_EO_V2_300M.pt"
DIFFICULTY_CSV="$WORK_DIR/ard_results_parallel/dataset_difficulty_metrics_2021.csv"

SAMPLES=10

echo "--- 1. Mosaicking ---"
python panel_of_visualization.py --model_type "mosaicking" --zarr_path "$ZARR_PATH" --difficulty_csv "$DIFFICULTY_CSV" --output_dir "$OUTPUT_BASE" --samples_per_stratum $SAMPLES

echo "--- 2. 3D U-Net ---"
python panel_of_visualization.py --model_type "unet" --checkpoint_path "$EXPERIMENTS_DIR/Baseline_3D_UNet_final.pth" --zarr_path "$ZARR_PATH" --difficulty_csv "$DIFFICULTY_CSV" --output_dir "$OUTPUT_BASE" --samples_per_stratum $SAMPLES

echo "--- 3. Prithvi (Frozen) ---"
python panel_of_visualization.py --model_type "prithvi_frozen" --checkpoint_path "$EXPERIMENTS_DIR/Prithvi_Frozen_final.pth" --prithvi_weights "$PRITHVI_WEIGHTS" --zarr_path "$ZARR_PATH" --difficulty_csv "$DIFFICULTY_CSV" --output_dir "$OUTPUT_BASE" --samples_per_stratum $SAMPLES

echo "--- 4. Prithvi (Full FT) ---"
python panel_of_visualization.py --model_type "prithvi_partial" --checkpoint_path "$EXPERIMENTS_DIR/Prithvi_Partial_Finetune_final.pth" --prithvi_weights "$PRITHVI_WEIGHTS" --zarr_path "$ZARR_PATH" --difficulty_csv "$DIFFICULTY_CSV" --output_dir "$OUTPUT_BASE" --samples_per_stratum $SAMPLES

echo "=========================================="
echo "Visualizations saved to $OUTPUT_BASE/pptx_visualizations"
echo "=========================================="

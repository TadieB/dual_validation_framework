#!/bin/bash
#SBATCH --job-name=cloud-imputation
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00 #1 days
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/resume_cloud_imputation_partial-%j.out
#SBATCH --error=logs/resume_cloud_imputation_partial-%j.err


echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start Time: $(date)"
echo "=========================================="

# Environment setup
source ~/.bashrc
conda activate prithvi_py311
echo "Conda environment activated"

# Check GPU
nvidia-smi
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# --- Paths configuration ---
WORK_DIR="/home/tadiebirihan.medimem/prithvi_work/cloud_imputation"
# IMPORTANT: UPDATE these two paths
ZARR_PATH="/home/tadiebirihan.medimem/prithvi_work/mydata/MOD09GA_CentralEurope_ROI.zarr"
PRITHVI_WEIGHTS="/home/tadiebirihan.medimem/prithvi_work/Prithvi-EO-2.0-300M/Prithvi_EO_V2_300M.pt"
OUTPUT_DIR="$WORK_DIR/experiments_$(date +%Y%m%d_%H%M%S)"

cd "$WORK_DIR"
mkdir -p logs
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Configuration:"
echo "Working directory: $WORK_DIR"
echo "Zarr data path: $ZARR_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="

# Run training
# Both baseline and Prithvi models will be trained and evaluated
python train_cloud_imputation.py \
    --zarr_path "$ZARR_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --train_baseline \
    --baseline_type unet \
    --baseline_epochs 50 \
    --train_prithvi \
    --prithvi_weights "$PRITHVI_WEIGHTS" \
    --prithvi_epochs 50 \
    --batch_size 4 \
    --num_workers 8

EXIT_CODE=$?

echo "=========================================="
echo "Exit Code: $EXIT_CODE"
echo "End Time: $(date)"
echo "=========================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
    echo "Results saved in: $OUTPUT_DIR"
else
    echo "Training failed with exit code: $EXIT_CODE"
fi

exit $EXIT_CODE

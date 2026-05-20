#!/bin/bash
#SBATCH --job-name=cloud-ddp-train
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --gpus-per-node=4
#SBATCH --output=logs/ddp_training_full%j.out
#SBATCH --error=logs/ddp_training_full%j.err

echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Starting DDP Training on 4 GPUs"
echo "=========================================="

# --- Environment Setup (Baldo HPC) ---
source /home/username/anaconda3/etc/profile.d/conda.sh
conda activate prithvi_py311
module purge
module load CUDA/11.8.0

# Change to working directory
cd /home/username/prithvi_work/prov_cloud_imputation

# Run PyTorch Native DDP Launcher,
torchrun --nproc_per_node=4 train_unified_gpu_ddp_all_carbon.py

echo "=========================================="
echo "Training completed and tracked via yProv4ML!"
echo "=========================================="

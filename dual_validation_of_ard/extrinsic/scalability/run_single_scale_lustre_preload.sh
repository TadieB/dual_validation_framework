#!/bin/bash
#SBATCH --job-name=scaling
#SBATCH --account=cli900
#SBATCH --partition=batch
#SBATCH --time=01:00:00
#SBATCH -o log_lustre_preload_bcast/scale-%j.out  #final.......
#SBATCH -e log_lustre_preload_bcast/scale-%j.err

MODEL=${1:-unet}
NODES=${2:-$SLURM_JOB_NUM_NODES}

# Environment Setup
unset SLURM_EXPORT_ENV
module purge
module load PrgEnv-gnu/8.6.0 cray-mpich/8.1.31 rocm/6.2.4 craype-accel-amd-gfx90a miniforge3/23.11.0-0

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /lustre/orion/world-shared/cli900/users/tadie/torch_rocm6_v3
PY_EXE=$(which python3)

# MIOpen Cache
export MIOPEN_USER_DB_PATH="/tmp/miopen-$USER-$$"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH"
export MIOPEN_FIND_MODE=1
export MIOPEN_DISABLE_CACHE=0

# Network
MASTER_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$(getent hosts "$MASTER_NODE" | awk '{print $1}' | head -n 1)
export MASTER_PORT=29501
export NCCL_SOCKET_IFNAME=hsn0
export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_CROSS_NIC=1
export TORCH_DISTRIBUTED_USE_IPV6=0
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=3600

# =============================================================================
# Data paths (Lustre - single copy, all nodes access)
# =============================================================================
ZARR_PATH="/lustre/orion/world-shared/cli900/users/tadie/zarr_netcdf_pipeline/final_datacubes2/MOD09GA_CentralEurope_ROI.zarr"
PRITHVI_PATH="/lustre/orion/world-shared/cli900/users/tadie/prithvi_eo/prithvi_ExO_2/300m_params/Prithvi_EO_V2_300M.pt"

# =============================================================================
# Benchmark Configuration
# =============================================================================
GPUS=$((NODES * 8))
OUTPUT_CSV="ieee_results.csv"

echo ""
echo "=========================================="
echo "🔬 IEEE-STYLE BENCHMARK"
echo "=========================================="
echo "Model:     $MODEL"
echo "Nodes:     $NODES"
echo "GPUs:      $GPUS"
echo "Master:    $MASTER_ADDR"
echo "Data:      Lustre (shared storage)"
echo ""
echo "Strategy:"
echo "  - Pre-load batches into GPU memory"
echo "  - Benchmark measures COMPUTE ONLY"
echo "  - Lustre I/O happens BEFORE timing"
echo ""
echo "IEEE Methodology:"
echo "  - Steps per run: 15"
echo "  - Repeats: 5"
echo "  - Average and report std dev"
echo "=========================================="
echo ""

# Fixed steps/repeats per IEEE paper
STEPS=15
REPEATS=5

# =============================================================================
# RUN BENCHMARK
# =============================================================================

if [ "$MODEL" == "prithvi" ]; then
    srun --unbuffered \
         -N${NODES} \
         -n${GPUS} \
         --cpus-per-task=7 \
         --ntasks-per-node=8 \
         --gpu-bind=closest \
         bash -c "$PY_EXE -u ddp_ieee_lustre_preloading_fix_batch.py \
             --model_type prithvi \
             --zarr_path '$ZARR_PATH' \
             --prithvi_weights '$PRITHVI_PATH' \
             --batch_size_per_gpu 4 \
             --num_steps $STEPS \
             --num_repeats $REPEATS \
             --preload_batches 30 \
             --output_csv '$OUTPUT_CSV'"
else
    srun --unbuffered \
         -N${NODES} \
         -n${GPUS} \
         --cpus-per-task=7 \
         --ntasks-per-node=8 \
         --gpu-bind=closest \
         bash -c "$PY_EXE -u ddp_ieee_lustre_preloading_fix_batch.py \
             --model_type unet \
             --zarr_path '$ZARR_PATH' \
             --batch_size_per_gpu 4 \
             --num_steps $STEPS \
             --num_repeats $REPEATS \
             --preload_batches 30 \
             --output_csv '$OUTPUT_CSV'"
fi

echo ""
echo "=========================================="
echo "✅ Benchmark Complete"
echo "=========================================="
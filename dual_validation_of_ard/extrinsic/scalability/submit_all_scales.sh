#!/bin/bash

# Submit all scaling jobs as separate SLURM jobs
# Each runs independently, avoiding time limit issues

mkdir -p log_lustre_preload_bcast

# UNet scaling
for NODES in 1 2 4 8 16; do
    GPUS=$((NODES * 8))
    echo "Submitting UNet on $GPUS GPUs ($NODES nodes)..."
    sbatch --nodes=$NODES \
           --gpus-per-node=8 \
           --ntasks-per-node=8 \
           run_single_scale_lustre_preload.sh unet $NODES
    sleep 2
done

# Prithvi scaling
for NODES in 1 2 4 8 16; do
    GPUS=$((NODES * 8))
    echo "Submitting Prithvi on $GPUS GPUs ($NODES nodes)..."
    sbatch --nodes=$NODES \
           --gpus-per-node=8 \
           --ntasks-per-node=8 \
           run_single_scale_lustre_preload.sh prithvi $NODES
    sleep 2
done

echo ""
echo "=========================================="
echo "All 10 jobs submitted!"
echo "Check status with: squeue -u $USER"
echo "=========================================="
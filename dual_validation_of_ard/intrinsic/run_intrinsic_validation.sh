#!/bin/bash
#SBATCH --job-name=intrinsic_validation
#SBATCH --output=logs/intrinsic_validation_geo_px%j.out
#SBATCH --error=logs/intrinsic_validation_geo_px%j.err
#SBATCH --partition=batch
#SBATCH --account=cli900
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:30:00

# --- Environment Setup ---
mkdir -p ./logs
module load miniforge3/23.11.0
source activate /lustre/orion/world-shared/cli900/users/tadie/myconda

# Set environment paths
export CONDA_PREFIX=/lustre/orion/world-shared/cli900/users/tadie/myconda
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PATH="$CONDA_PREFIX/bin:$PATH"
CONDA_PYTHON="$CONDA_PREFIX/bin/python"

# --- Stability Settings for Numerical Libraries ---
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# --- Dask Memory Management Settings ---
export DASK_DISTRIBUTED__WORKER__MEMORY__TARGET=0.6
export DASK_DISTRIBUTED__WORKER__MEMORY__SPILL=0.7
export DASK_DISTRIBUTED__WORKER__MEMORY__PAUSE=0.8
export DASK_DISTRIBUTED__WORKER__MEMORY__TERMINATE=0.9

echo "========================================================="
echo "Starting Intrinsic Validation Job | Job ID: $SLURM_JOB_ID"
echo "========================================================="

# --- Run the Python Script ---
$CONDA_PYTHON main_intrinsic_validation.py

exit_code=$?
if [ $exit_code -eq 0 ]; then
    echo "SUCCESS: Intrinsic validation completed."
else
    echo "FAILED: Script failed with exit code $exit_code"
fi
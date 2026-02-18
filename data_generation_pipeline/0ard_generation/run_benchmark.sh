#!/bin/bash
#SBATCH --job-name=create_modis_datacube
#SBATCH --output=logs_final2/benchmark%j.out
#SBATCH --error=logs_final2/benchmark%j.err
#SBATCH --partition=batch
#SBATCH --account=cli900
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00

# disable HDF5 file locking for this job (must happen before Python imports)
export HDF5_USE_FILE_LOCKING=FALSE

# --- Environment Setup ---
mkdir -p ./logs_final2
module load miniforge3/23.11.0
source activate /lustre/orion/world-shared/cli900/users/tadie/myconda

# CRITICAL: Set environment paths based on your proven script
export CONDA_PREFIX=/lustre/orion/world-shared/cli900/users/tadie/myconda
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PATH="$CONDA_PREFIX/bin:$PATH"
CONDA_PYTHON="$CONDA_PREFIX/bin/python"

# --- Stability Settings from Your Production Script ---

# Prevent segmentation faults from scientific libraries
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# Disable GPU libraries (CPU-only processing)
export CUDA_VISIBLE_DEVICES=""

# Dask memory settings using environment variables
export DASK_DISTRIBUTED__WORKER__MEMORY__TARGET=0.6
export DASK_DISTRIBUTED__WORKER__MEMORY__SPILL=0.7
export DASK_DISTRIBUTED__WORKER__MEMORY__PAUSE=0.8
export DASK_DISTRIBUTED__WORKER__MEMORY__TERMINATE=0.9

echo "========================================================="
echo "Starting Data Cube Creation Job | Job ID: $SLURM_JOB_ID"
echo "========================================================="

# --- Run the Script ---
$CONDA_PYTHON main_benchmark_final2.py

exit_code=$?
if [ $exit_code -eq 0 ]; then
    echo "SUCCESS: Data cube creation completed."
else
    echo "FAILED: Script failed with exit code $exit_code"
fi

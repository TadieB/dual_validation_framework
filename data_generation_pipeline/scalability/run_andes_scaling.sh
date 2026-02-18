#!/bin/bash
#SBATCH --job-name=scalability_weak
#SBATCH --output=logs_scaling/benchmark_weak-%j.out
#SBATCH --error=logs_scaling/benchmark_weak-%j.err
#SBATCH --partition=batch
#SBATCH --account=cli900
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00

# ==============================================================================
# ENVIRONMENT SETUP (Specific to OLCF Andes)
# ==============================================================================

# disable HDF5 file locking (Must be before imports)
export HDF5_USE_FILE_LOCKING=FALSE

# Load Modules & Activate Environment
mkdir -p ./logs_scaling
module load miniforge3/23.11.0
source activate /lustre/orion/world-shared/cli900/users/tadie/myconda

# CRITICAL: Export exact paths
export CONDA_PREFIX=/lustre/orion/world-shared/cli900/users/tadie/myconda
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PATH="$CONDA_PREFIX/bin:$PATH"
CONDA_PYTHON="$CONDA_PREFIX/bin/python"

# --- Stability Settings ---
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export CUDA_VISIBLE_DEVICES="" # CPU-only

# Dask memory settings
export DASK_DISTRIBUTED__WORKER__MEMORY__TARGET=0.6
export DASK_DISTRIBUTED__WORKER__MEMORY__SPILL=0.7
export DASK_DISTRIBUTED__WORKER__MEMORY__PAUSE=0.8
export DASK_DISTRIBUTED__WORKER__MEMORY__TERMINATE=0.9

echo "========================================================="
echo "Starting Sandro's Weak Scaling Experiment (8 -> 128 Cores)"
echo "Orchestrator Job ID: $SLURM_JOB_ID"
echo "========================================================="

# ==============================================================================
# SCALING LOGIC
# ==============================================================================

TEMPLATE="config_benchmark.yaml"
OUTPUT_CSV="scaling_results_real.csv"
OUTPUT_JSON="scaling_results.json"
MEASURER="measure_throughput.py"

# Initialize outputs
echo "Worker_Count,Days,Data_Size_MB,Throughput_MBs" > $OUTPUT_CSV
if [ ! -f "$OUTPUT_JSON" ]; then echo "[]" > $OUTPUT_JSON; fi

# --- 5 SCALING STEPS (Weak Scaling) ---
# Format: "NODES CORES_PER_NODE DAYS"

# Step 1: 1 Node, 8 Cores active -> 1 Day
EXP_1="1 8 1"
# Step 2: 1 Node, 16 Cores active -> 2 Days
EXP_2="1 16 2"
# Step 3: 1 Node, 32 Cores active (Full Node) -> 4 Days
EXP_3="1 32 4"
# Step 4: 2 Nodes, 32 Cores/Node (64 Total) -> 8 Days
EXP_4="2 32 8"
# Step 5: 4 Nodes, 32 Cores/Node (128 Total) -> 16 Days
EXP_5="4 32 16"

for exp in "$EXP_1" "$EXP_2" "$EXP_3" "$EXP_4" "$EXP_5"; do
    read -r NODES CORES DAYS <<< "$exp"
    TOTAL_CORES=$((NODES * CORES))
    
    echo "----------------------------------------------------------"
    echo "Running Step: $NODES Node(s) x $CORES Cores | $DAYS Day(s)"
    echo "----------------------------------------------------------"
    
    # 1. Update Config Template
    cp $TEMPLATE config_temp.yaml
    sed -i "s/slurm_jobs: [0-9]*/slurm_jobs: $NODES/" config_temp.yaml
    sed -i "s/cores: [0-9]*/cores: $CORES/" config_temp.yaml
    sed -i "s/expected_days: [0-9]*/expected_days: $DAYS/" config_temp.yaml
    
    # Extract Output Path to clean up before run
    OUT_DIR=$(grep "output_dir:" config_temp.yaml | awk '{print $2}' | tr -d '"')
    ZARR_NAME=$(grep "zarr_store_name:" config_temp.yaml | awk '{print $2}' | tr -d '"')
    FULL_ZARR_PATH="${OUT_DIR%/}/${ZARR_NAME}"
    rm -rf "$FULL_ZARR_PATH"
    
    # 2. Run Pipeline & Time It
    START_TIME=$(date +%s.%N)
    
    # Use exact python path
    $CONDA_PYTHON main_pipeline_optimized.py --config config_temp.yaml --format zarr
    
    END_TIME=$(date +%s.%N)
    
    # Use python to calculate duration to avoid float errors in bash
    DURATION=$($CONDA_PYTHON -c "print($END_TIME - $START_TIME)")
    echo "Pipeline finished in $DURATION seconds."
    
    # 3. Measure & Log (Using your updated Python script)
    # This prints the CSV line to stdout (captured by RESULT) and saves JSON silently
    RESULT=$($CONDA_PYTHON $MEASURER "$FULL_ZARR_PATH" "$DURATION" "$TOTAL_CORES" "$DAYS" "$OUTPUT_JSON")
    
    echo "Recorded: $RESULT"
    echo "$RESULT" >> $OUTPUT_CSV
    
    # Cleanup to save space for next run
    rm -rf "$FULL_ZARR_PATH"
done

echo "========================================================="
echo "Experiment Complete. Data in $OUTPUT_CSV and $OUTPUT_JSON"
echo "========================================================="
# A Dual Validation Framework for Analysis-Ready Satellite Data

This repository contains the source code, models, and high-performance computing (HPC) deployment scripts for the paper: **"A Dual Validation Framework for Analysis-Ready Satellite Data: A Scalable Pipeline and Stratified Performance Analysis."**

## 📌 Overview
This framework provides an end-to-end pipeline to acquire, validate, and train on large-scale Earth Observation datasets. It introduces a **Difficulty Index (DI)** for stratified model evaluation and rigorously benchmarks the scalability of 3D U-Net and Prithvi Vision Transformer architectures on extreme-scale supercomputers (OLCF Frontier).

---

## 📂 Repository Structure & Scripts

### 1. `data_generation_pipeline/`
Handles the parallel acquisition and construction of the multitemporal data cube.
* `[data_acquisition.py]`: Parallel LPDAAC data acquisition using Parsl and Globus.
* `[main_benchmark_final2.py]`: Preprocessing and chunking of the dataset into a `.zarr` cube.

### 2. `dual_validation/`
The core scientific framework quantifying data quality and model utility.
* **`intrinsic/`**: 
  *  Calculates the composite Difficulty Index (spatial heterogeneity, temporal variability, and scarcity).
  *  Radiometric sanity and temporal consistency checks.
* **`extrinsic/`**: 
  * Evaluates downstream model performance across different DI strata.

### 3. `models/`
Architectures and data ingestion pipelines.
* `baseline_3d_unet.py`: The 3D Convolutional baseline model.
* `prithvi_mae.py`: The Prithvi foundation model (Vision Transformer) backbone.
* `prithvi_cloud_imputation_model_bcast.py`: Decoder adaptations for the cloud-gap imputation task.
* `cloud_imputation_dataloader_original.py`: Custom PyTorch dataloader featuring Just-In-Time (JIT) quality filtering and NaN sanitization.

### 4. `hpc_deployment/`
SLURM scripts and DDP wrappers used to execute runs across computing environments.
* `ddp_ieee_lustre_preloading_fix_batch.py`: The PyTorch Distributed Data Parallel (DDP) wrapper script that handles pre-loading data into accelerator memory to prevent I/O bottlenecks during benchmarking.
* `submit_all_scales.sh`: Master execution script that automatically submits sequential jobs for 8, 16, 32, 64, and 128 GPUs.
* `run_single_scale_lustre_preload.sh`: The worker SLURM script configured for AMD MI250X GPUs on OLCF Frontier.

## 🚀 Running the Scalability Benchmarks (OLCF Frontier)

To reproduce the strong scaling results showing 3D U-Net's high parallel efficiency (88.1%) vs. Prithvi ViT's communication bottlenecks (15.6%):

1. **Submit the Jobs:**
   ```bash
   cd hpc_deployment
   bash submit_all_scales.sh

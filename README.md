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

### 🚀 Installation & Environments

Due to the diverse hardware used in this framework, we provide distinct environment configurations.

**1. Data Pipeline (CPU)**
```bash
conda create -n pipeline_env python=3.11 -y
conda activate pipeline_env
pip install -r requirements/andes_pipeline.txt
```

**2. Production Training (NVIDIA GPU)**
```bash
conda create -n train_env python=3.11 -y
conda activate train_env
pip install torch==2.4.0 torchvision==0.19.0 --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
pip install -r requirements/baldo_training.txt
```

**3. Extreme-Scale Benchmarking (AMD ROCm)**
```bash
# Ensure the base ROCm module is loaded per OLCF Frontier documentation
pip install -r requirements/frontier_scaling.txt
```

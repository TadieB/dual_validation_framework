#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODIS Data Cube Benchmark: Zarr vs NetCDF
A hybrid script combining the proven data processing logic of main_benchmark_fix.py
with the advanced benchmarking and reporting of main_benchmark_final.py.
"""

import os
import sys
import time
import yaml
import logging
import json
import argparse
import glob
import warnings
import platform
import psutil
from datetime import datetime
from collections import defaultdict

import xarray as xr
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import dask
from dask.delayed import delayed
from dask_jobqueue import SLURMCluster
from dask.distributed import Client

warnings.filterwarnings("ignore", category=UserWarning, module="zarr")
from mod09ga_tile_processor_multimask import process_tile, extract_julian_day

# --- PYTORCH DATASET CLASS ---
class SatelliteDataset(Dataset):
    def __init__(self, dataset_path, patch_size, timesteps=4, file_format='zarr'):
        self.dataset_path = dataset_path
        self.patch_size = patch_size
        self.timesteps = timesteps
        self.file_format = file_format
        self.engine = 'zarr' if self.file_format == 'zarr' else 'h5netcdf'
        
        with xr.open_dataset(self.dataset_path, engine=self.engine) as ds:
            self.data_shape = ds['reflectance_and_mask'].shape
            self.dtype = ds['reflectance_and_mask'].dtype
            self.num_patches_y = self.data_shape[2] // self.patch_size
            self.num_patches_x = self.data_shape[3] // self.patch_size
            self.num_time_windows = self.data_shape[0] - self.timesteps + 1
            if self.num_time_windows < 1:
                raise ValueError("Not enough timesteps in dataset to create a patch.")
        
        self.total_patches = self.num_patches_y * self.num_patches_x * self.num_time_windows
        self.data = None
    
    def __len__(self):
        return self.total_patches
    
    def __getitem__(self, idx):
        if self.data is None:
            self.data = xr.open_dataset(self.dataset_path, engine=self.engine)['reflectance_and_mask']
        
        time_idx = idx % self.num_time_windows
        spatial_idx = idx // self.num_time_windows
        y_idx = spatial_idx // self.num_patches_x
        x_idx = spatial_idx % self.num_patches_x
        y_start = y_idx * self.patch_size
        x_start = x_idx * self.patch_size
        
        patch = self.data[time_idx:time_idx+self.timesteps, :, y_start:y_start+self.patch_size, x_start:x_start+self.patch_size].values
        return torch.from_numpy(patch)

# --- ADVANCED BENCHMARKING FUNCTIONS ---

def run_simple_read_benchmark(filepath, config, file_format):
    logging.info(f"  Simple Read Test: Reading one patch from {file_format.upper()}...")
    patch_size = config.get('patch_size', 224)
    timesteps = config.get('timesteps_for_benchmark', 4)
    engine = 'zarr' if file_format == 'zarr' else 'h5netcdf'
    
    try:
        start_time = time.time()
        with xr.open_dataset(filepath, engine=engine, chunks={}) as ds:
            data_slice = ds['reflectance_and_mask'][0:timesteps, :, 0:patch_size, 0:patch_size]
            data_slice.load()
        read_time = time.time() - start_time
        data_mb = data_slice.nbytes / (1024**2)
        throughput = data_mb / read_time if read_time > 0 else 0
        result = {'read_time_s': round(read_time, 4), 'data_read_mb': round(data_mb, 2), 'throughput_mb_per_s': round(throughput, 2), 'status': 'Success'}
        logging.info(f"    Result: {result['throughput_mb_per_s']} MB/s")
        return result
    except Exception as e:
        logging.error(f"  Simple Read Test for {file_format.upper()} failed: {e}", exc_info=True)
        return {'status': f'Failed: {str(e)}', 'read_time_s': -1}


def run_pytorch_benchmark(filepath, config, file_format='zarr', num_workers=0):
    logging.info(f"  PyTorch DataLoader Test: {file_format.upper()} with {num_workers} workers...")
    patch_size = config.get('patch_size', 224)
    num_batches = config.get('num_batches_to_test', 10)
    batch_size = config.get('batch_size', 8)
    timesteps = config.get('timesteps_for_benchmark', 4)
    
    fail_result = {'num_workers': num_workers, 'status': 'Failed', 'total_time_s': -1, 'throughput_mb_per_s': 0, 'samples_per_second': 0}
    
    try:
        dataset = SatelliteDataset(filepath, patch_size, timesteps, file_format)
        persistent = num_workers > 0
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=persistent)
        
        logging.info("    Warming up DataLoader workers...")
        _ = next(iter(dataloader))
        logging.info("    Warm-up successful.")
        
        start_time = time.time()
        batches_loaded, total_samples = 0, 0
        for batch in dataloader:
            batches_loaded += 1
            total_samples += batch.shape[0]
            if batches_loaded >= num_batches: break
        
        total_time = time.time() - start_time
        
        if batches_loaded > 0:
            patch_bytes = timesteps * dataset.data_shape[1] * patch_size * patch_size * np.dtype(dataset.dtype).itemsize
            total_data_mb = (patch_bytes * total_samples) / (1024**2)
            result = {
                'num_workers': num_workers, 'batches_loaded': batches_loaded, 'samples_loaded': total_samples, 'total_data_mb': round(total_data_mb, 2),
                'total_time_s': round(total_time, 4), 'throughput_mb_per_s': round(total_data_mb / total_time, 2) if total_time > 0 else 0,
                'samples_per_second': round(total_samples / total_time, 2) if total_time > 0 else 0, 'status': 'Success'
            }
            logging.info(f"    Result: {result['throughput_mb_per_s']} MB/s, {result['samples_per_second']} samples/s")
            return result
        else:
            raise ValueError("No batches were loaded.")
    except Exception as e:
        logging.error(f"  PyTorch benchmark for {file_format.upper()} failed: {e}", exc_info=True)
        fail_result['status'] = f'Failed: {str(e)}'
        return fail_result

# --- UTILITY AND METADATA FUNCTIONS ---

def get_system_specs(client):
    try:
        wi = list(client.scheduler_info()['workers'].values())[0]
        cores, mem = wi['nthreads'], f"{wi['memory_limit']/1e9:.2f} GB"
    except (IndexError, KeyError):
        cores, mem = "N/A", "N/A"
    return {
        "Platform": {"System": platform.system(), "Release": platform.release()},
        "Hardware": {"CPU_Physical_Cores": psutil.cpu_count(logical=False), "CPU_Logical_Cores": psutil.cpu_count(logical=True), "Total_RAM_GB": f"{psutil.virtual_memory().total/(1024**3):.2f}"},
        "Dask_Cluster": {"Workers": len(client.scheduler_info().get('workers', {})), "Cores_per_Worker": cores, "Memory_per_Worker": mem},
        "Software": {"python": platform.python_version(), "xarray": xr.__version__, "torch": torch.__version__, "dask": dask.__version__}
    }


def generate_paper_ready_report(timings, output_dir, config, processing_stats, benchmark_results, system_specs):
    logging.info("Generating paper-ready JSON report...")
    def safe_div(num, den):
        if den is None or den == 0: return 0
        return num / den

    nc_path = os.path.join(output_dir, config['netcdf_filename'])
    zarr_path = os.path.join(output_dir, config['zarr_store_name'])
    try: nc_size_gb = os.path.getsize(nc_path) / (1024**3)
    except FileNotFoundError: nc_size_gb = -1
    try:
        zarr_size_bytes = sum(os.path.getsize(os.path.join(dp, fn)) for dp, _, fns in os.walk(zarr_path) for fn in fns)
        zarr_size_gb = zarr_size_bytes / (1024**3)
    except Exception: zarr_size_gb = -1

    nc_wt = timings.get('netcdf_write_time', -1)
    z_wt = timings.get('zarr_write_time', -1)
    
    nc_simple, z_simple = benchmark_results.get('netcdf_simple_read', {}), benchmark_results.get('zarr_simple_read', {})
    nc_torch_0, z_torch_0, z_torch_4 = benchmark_results.get('netcdf_pytorch_0', {}), benchmark_results.get('zarr_pytorch_0', {}), benchmark_results.get('zarr_pytorch_4', {})
    
    z_vs_nc_fair_read_speedup = safe_div(z_torch_0.get('throughput_mb_per_s'), nc_torch_0.get('throughput_mb_per_s'))
    z_scaling_factor = safe_div(z_torch_4.get('throughput_mb_per_s'), z_torch_0.get('throughput_mb_per_s'))
    
    report = {
        "Pipeline_Summary": {"Total_Execution_Time_s": round(timings.get('total_pipeline_time', 0), 2), "Complete_Mosaics_Created": processing_stats.get('complete_mosaics_created', 0), "Days_In_Datacube": len(processing_stats.get('datacube_days', []))},
        "Write_Performance": {
            "NetCDF4": {"write_time_s": round(nc_wt, 2), "size_gb": round(nc_size_gb, 3), "throughput_mb_per_s": round(safe_div(nc_size_gb * 1024, nc_wt), 2)},
            "Zarr": {"write_time_s": round(z_wt, 2), "size_gb": round(zarr_size_gb, 3), "throughput_mb_per_s": round(safe_div(zarr_size_gb * 1024, z_wt), 2)},
            "Comparison": {"write_speedup_zarr_vs_netcdf": round(safe_div(nc_wt, z_wt), 2), "size_ratio_zarr_vs_netcdf": round(safe_div(zarr_size_gb, nc_size_gb), 2)}
        },
        "Read_Performance": {
            "Simple_Read_Single_Patch": {"NetCDF4": nc_simple, "Zarr": z_simple, "Speedup_Zarr_vs_NetCDF": round(safe_div(z_simple.get('throughput_mb_per_s'), nc_simple.get('throughput_mb_per_s')), 2)},
            "PyTorch_DataLoader_Single_Worker": {"NetCDF4_workers_0": nc_torch_0, "Zarr_workers_0": z_torch_0, "Speedup_Zarr_vs_NetCDF": round(z_vs_nc_fair_read_speedup, 2)},
            "PyTorch_DataLoader_Zarr_Scaling": {"Zarr_workers_0": z_torch_0, "Zarr_workers_4": z_torch_4, "Scaling_Factor": round(z_scaling_factor, 2), "Parallel_Efficiency_Percent": round(safe_div(z_scaling_factor, 4) * 100, 1)}
        },
        "System_Specifications": system_specs
    }
    
    json_path = os.path.join(output_dir, 'benchmark_report.json')
    with open(json_path, 'w') as f: json.dump(report, f, indent=4)
    logging.info(f"JSON report saved to: {json_path}")
    return report

def setup_logging():
    os.makedirs("./logs", exist_ok=True)
    log_file = f"./logs/benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)])
    logging.info(f"Logging initialized. Log file: {log_file}")

def setup_dask_cluster(config):
    logging.info("Setting up Dask SLURMCluster...")
    cluster_config = config.copy()
    num_jobs = cluster_config.pop('slurm_jobs', 4)
    cluster = SLURMCluster(**cluster_config)
    client = Client(cluster)
    cluster.scale(jobs=num_jobs)
    workers_per_job = config.get('processes', 1)
    expected_workers = num_jobs * workers_per_job
    logging.info(f"Waiting for {expected_workers} Dask workers...")
    client.wait_for_workers(n_workers=expected_workers, timeout=900)
    logging.info(f"Dask cluster ready with {len(client.scheduler_info()['workers'])} workers.")
    return cluster, client

# --- NEW METADATA HELPER FUNCTIONS ---

def _extract_tile_ids(files_by_day, complete_days):
    """Extracts unique MODIS tile IDs (e.g., h18v04) from file paths."""
    if not complete_days:
        return set()
    # Use the file paths from the first complete day to find the tile IDs
    first_day = complete_days[0]
    tile_paths_for_day = files_by_day[first_day]
    # Assumes filename format like 'MOD09GA.A2024153.h18v04.061.2024155040156.hdf'
    tile_ids = {os.path.basename(p).split('.')[2] for p in tile_paths_for_day}
    return tile_ids

def add_cf_metadata(ds, included_days, tile_ids):
    """Adds comprehensive, CF-compliant metadata to the final data cube."""
    logging.info("Adding detailed CF-compliant metadata...")

    # --- Define Band Information ---
    band_long_names = "Red, NIR, Blue, Green, SWIR 1, SWIR 2, SWIR 3, Usability Mask"
    band_wavelengths = "620-670 nm, 841-876 nm, 459-479 nm, 545-565 nm, 1230-1250 nm, 1628-1652 nm, 2105-2155 nm, N/A"

    # --- Global Attributes ---
    ds.attrs = {
        'Conventions': 'CF-1.11',
        'title': 'MODIS MOD09GA Daily Surface Reflectance Multitemporal Multispectral Data Cube',
        'institution': 'University of Trento',
        'source': 'MODIS/Terra Surface Reflectance Daily L2G Global 500m SIN Grid V061',
        'history': f'Created {datetime.utcnow().isoformat()}Z by the benchmark script.',
        'source_tiles': ','.join(sorted(list(tile_ids))),
        'included_julian_days': ','.join(sorted(included_days)),
        'total_days_processed': len(included_days),
        'usability_mask_key': "0=Pristine, 1=Shadow, 2=Hazy (from aerosol), 3=Contaminated (from cirrus or cloud adjacency), 4=Cloudy, 255=Fill"
    }

    # --- Coordinate Attributes ---
    ds['time'].attrs = {'long_name': 'time', 'standard_name': 'time'}  
    ds['y'].attrs = {'long_name': 'y coordinate of projection', 'standard_name': 'projection_y_coordinate', 'units': 'm'}
    ds['x'].attrs = {'long_name': 'x coordinate of projection', 'standard_name': 'projection_x_coordinate', 'units': 'm'}
    
    ds['band'].attrs = {
        'long_name': 'Surface reflectance bands and usability mask',
        'band_names': band_long_names,
        'band_wavelengths_nm': band_wavelengths
    }

    # --- Data Variable Attributes ---
    ds['reflectance_and_mask'].attrs = {
        'long_name': 'Surface reflectance for bands 1-7 and usability mask for band 8',
        'units': '1', # Dimensionless quantity
        'grid_mapping': 'sinusoidal',
        'notes': 'Bands 1-7 are float32 scaled reflectance with a fill value of NaN. Band 8 is an uint8 integer usability mask. See usability_mask_key global attribute.'
    }
    
    return ds

def validate_config(config):
    for key in ['input_dir', 'output_dir', 'required_bands', 'tiles_per_day']:
        if key not in config: raise ValueError(f"Missing required configuration key: '{key}'")
    if not os.path.isdir(config['input_dir']):
        raise FileNotFoundError(f"Input directory not found: {config['input_dir']}")
    logging.info("Configuration file validated successfully.")

# --- MAIN WORKFLOW ---

def main(config_path):
    setup_logging()
    cluster, client = None, None
    
    try:
        with open(config_path, 'r') as f: config = yaml.safe_load(f)
        validate_config(config)
        os.makedirs(config['output_dir'], exist_ok=True)
        
        cluster, client = setup_dask_cluster(config['dask_cluster'])
        
        timings = {}
        processing_stats = {
            'complete_mosaics_created': 0, 'datacube_days': [], 'datacube_created': False
        }
        pipeline_start_time = time.time()

        # === PHASES 1-3: Data Discovery, Processing, and Aggregation (PROVEN WORKING LOGIC) ===
        logging.info("="*60 + "\nPHASE 1: FILE DISCOVERY\n" + "="*60)
        all_files = sorted(glob.glob(os.path.join(config['input_dir'], "**", "*.hdf"), recursive=True))
        if not all_files: raise RuntimeError(f"No HDF files found in {config['input_dir']}")
        
        days = sorted(list({extract_julian_day(os.path.basename(f)) for f in all_files if extract_julian_day(os.path.basename(f)) is not None}))
        days_to_process = days[:config.get('expected_days', len(days))]
        logging.info(f"Found {len(all_files)} files across {len(days)} days. Processing up to {len(days_to_process)} days.")
        
        files_by_day = defaultdict(list)
        for f in all_files:
            day = extract_julian_day(os.path.basename(f))
            if day in days_to_process: files_by_day[day].append(f)
        
        complete_days = [d for d in days_to_process if len(files_by_day[d]) == config['tiles_per_day']]
        if not complete_days: raise RuntimeError("No days with complete tile coverage found.")
        logging.info(f"Found {len(complete_days)} days with complete tile coverage.")
        
        logging.info("="*60 + "\nPHASE 2: PARALLEL PROCESSING\n" + "="*60)
        @delayed
        def process_day_with_validation(day_id, tile_paths, conf):
            processed_tiles = [process_tile(p, conf) for p in tile_paths]
            valid_tiles = [t for t in processed_tiles if t is not None]
            if len(valid_tiles) == len(tile_paths):
                try: return {'day': day_id, 'mosaic': xr.combine_by_coords(valid_tiles), 'success': True}
                except Exception as e:
                    logging.error(f"Day {day_id}: FAILED to combine mosaic: {e}")
                    return {'success': False}
            return {'success': False}
        
        delayed_tasks = [process_day_with_validation(day, files_by_day[day], config) for day in complete_days]
        computed_results = dask.compute(*delayed_tasks)
        
        logging.info("="*60 + "\nPHASE 3: AGGREGATION\n" + "="*60)
        successful_mosaics = {r['day']: r['mosaic'] for r in computed_results if r and r.get('success')}
        processing_stats.update({'complete_mosaics_created': len(successful_mosaics), 'datacube_days': list(successful_mosaics.keys())})
        if not successful_mosaics: raise RuntimeError("No complete mosaics were created after processing.")

        # === PHASE 4: Data Cube Creation (PROVEN WORKING LOGIC) ===
        logging.info("="*60 + "\nPHASE 4: WRITING DATACUBES\n" + "="*60)
        
        # Extract tile IDs for metadata before creating the final cube
        tile_ids = _extract_tile_ids(files_by_day, complete_days)
        
        sorted_days = sorted(successful_mosaics.keys())
        time_coords = pd.to_datetime([f'2024-{day}' for day in sorted_days], format='%Y-%j')
        final_datacube = xr.concat([successful_mosaics[day] for day in sorted_days], dim=pd.Index(time_coords, name='time'))
        final_ds = final_datacube.to_dataset(name='reflectance_and_mask')
        
        # Add the final, most descriptive metadata
        final_ds = add_cf_metadata(final_ds, list(successful_mosaics.keys()), tile_ids)
        
        final_ds['band'] = final_ds['band'].astype('O')
        
        nc_path = os.path.join(config['output_dir'], config['netcdf_filename'])
        zarr_path = os.path.join(config['output_dir'], config['zarr_store_name'])
        
        logging.info(f"Saving NetCDF to: {nc_path} with zlib compression...")
        nc_write_start = time.time()

        netcdf_encoding = {
            'reflectance_and_mask': {
                'zlib': True, 'complevel': 4, 'shuffle': True
            }
        }
        final_ds.to_netcdf(nc_path, engine='h5netcdf', encoding=netcdf_encoding)
        os.sync() 

        timings['netcdf_write_time'] = time.time() - nc_write_start
        
        logging.info(f"Saving Zarr to: {zarr_path}")
        zarr_write_start = time.time()
        final_ds.to_zarr(zarr_path, mode='w')
        timings['zarr_write_time'] = time.time() - zarr_write_start
        processing_stats['datacube_created'] = True
        
        # === PHASE 5: REFINED BENCHMARKING & REPORTING ===
        logging.info("="*60 + "\nPHASE 5: BENCHMARKING & REPORTING\n" + "="*60)
        system_specs = get_system_specs(client)
        
        benchmark_results = {}
        logging.info("--- Running NetCDF Benchmarks ---")
        benchmark_results['netcdf_simple_read'] = run_simple_read_benchmark(nc_path, config, 'netcdf')
        benchmark_results['netcdf_pytorch_0'] = run_pytorch_benchmark(nc_path, config, 'netcdf', num_workers=0)
        
        logging.info("--- Running Zarr Benchmarks ---")
        benchmark_results['zarr_simple_read'] = run_simple_read_benchmark(zarr_path, config, 'zarr')
        benchmark_results['zarr_pytorch_0'] = run_pytorch_benchmark(zarr_path, config, 'zarr', num_workers=0)
        benchmark_results['zarr_pytorch_4'] = run_pytorch_benchmark(zarr_path, config, 'zarr', num_workers=4)

        timings['total_pipeline_time'] = time.time() - pipeline_start_time
        report = generate_paper_ready_report(timings, config['output_dir'], config, processing_stats, benchmark_results, system_specs)
        
        logging.info("="*60 + "\nPIPELINE COMPLETED\n" + "="*60)
        logging.info(f"Final report saved to: {os.path.join(config['output_dir'], 'benchmark_report.json')}")
        
        wp, rp_fair, rp_scale = report['Write_Performance']['Comparison'], report['Read_Performance']['PyTorch_DataLoader_Single_Worker'], report['Read_Performance']['PyTorch_DataLoader_Zarr_Scaling']
        logging.info("\n--- KEY FINDINGS ---")
        logging.info(f"  Write Speed: Zarr was {wp['write_speedup_zarr_vs_netcdf']:.2f}x faster to write than NetCDF.")
        logging.info(f"  File Size: Zarr was {wp['size_ratio_zarr_vs_netcdf']:.2f}x the size of NetCDF.")
        logging.info(f"  Fair Read Speed (1 worker): Zarr was {rp_fair['Speedup_Zarr_vs_NetCDF']:.2f}x faster than NetCDF.")
        logging.info(f"  Zarr Parallel Scaling (4 workers): Achieved a {rp_scale['Scaling_Factor']:.2f}x speedup ({rp_scale['Parallel_Efficiency_Percent']:.1f}% efficiency).")

    except Exception as e:
        logging.error(f"PIPELINE FAILED: {e}", exc_info=True)
        raise
        
    finally:
        if client: client.close()
        if cluster: cluster.close()
        logging.info("Dask cluster has been shut down.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create and benchmark a MODIS data cube.")
    parser.add_argument('--config', default='config_benchmark.yaml', help='Path to the YAML configuration file.')
    args = parser.parse_args()
    
    try:
        main(config_path=args.config)
        sys.exit(0)
    except (Exception, SystemExit) as e:
        if isinstance(e, SystemExit) and e.code == 0: pass
        else:
            logging.error("Exiting due to an unrecoverable error.")
            sys.exit(1)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MODIS Data Cube Generation & Write Benchmark - OUT-OF-CORE OPTIMIZED
====================================================================
Optimized Zarr path to avoid the serial client memory bottleneck by using lazy aggregation.
NetCDF path is preserved for serial benchmarking.
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
from typing import Dict, Optional, List

import xarray as xr
import numpy as np
import pandas as pd
import dask
from dask.delayed import delayed
from dask_jobqueue import SLURMCluster
from dask.distributed import Client

# Assuming mod09ga_tile_processor_multimask.py is in the environment
# NOTE: process_tile and extract_julian_day are imported from that module
from mod09ga_tile_processor_multimask import process_tile, extract_julian_day 

warnings.filterwarnings("ignore", category=UserWarning, module="zarr")
warnings.filterwarnings("ignore", category=UserWarning, module="xarray")

# --- NEW LAZY AGGREGATION WRAPPER (Fix 1) ---

@delayed
def get_lazy_mosaic(day_id, tile_paths, conf):
    """
    Wrapper function that performs parallel tile processing for a day
    and returns a single *computed* mosaic as an Xarray DataArray.
    (Used by the NetCDF path, which requires computed results).
    """
    processed_tiles = [process_tile(p, conf) for p in tile_paths]
    valid_tiles = [t for t in processed_tiles if t is not None]
    if len(valid_tiles) == conf['tiles_per_day']:
        try:
            # xr.combine_by_coords here combines computed results into a single day's mosaic
            mosaic = xr.combine_by_coords(valid_tiles)
            return {'day': day_id, 'mosaic': mosaic, 'success': True, 'processed_tiles': len(valid_tiles)}
        except Exception as e:
            logging.error(f"Day {day_id}: FAILED to combine mosaic: {e}")
            return {'success': False}
    return {'success': False}


# --- UTILITY FUNCTIONS (Completed Body) ---

def get_system_specs(client):
    """Gets Dask cluster specs."""
    try:
        wi = list(client.scheduler_info()['workers'].values())[0]
        cores, mem = wi['nthreads'], f"{wi['memory_limit']/1e9:.2f} GB"
        num_workers = len(client.scheduler_info().get('workers', {}))
    except (IndexError, KeyError):
        cores, mem, num_workers = "N/A", "N/A", 0
    return {
        "Platform": {"System": platform.system(), "Release": platform.release()},
        "Hardware": {"CPU_Physical_Cores": psutil.cpu_count(logical=False), "CPU_Logical_Cores": psutil.cpu_count(logical=True), "Total_RAM_GB": f"{psutil.virtual_memory().total/(1024**3):.2f}"},
        "Dask_Cluster": {"Workers": num_workers, "Cores_per_Worker": cores, "Memory_per_Worker": mem},
        "Software": {"python": platform.python_version(), "xarray": xr.__version__, "dask": dask.__version__}
    }

def generate_paper_ready_report(timings, output_dir, config, processing_stats, system_specs, output_format):
    """Generates a JSON report for WRITE performance only."""
    logging.info(f"Generating paper-ready JSON report (Format: {output_format})...")
    
    def safe_div(num, den):
        if den is None or den == 0 or num is None: return None
        return num / den

    # --- Write Performance ---
    nc_path = os.path.join(output_dir, config['netcdf_filename'])
    zarr_path = os.path.join(output_dir, config['zarr_store_name'])
    
    try: nc_size_gb = os.path.getsize(nc_path) / (1024**3)
    except (FileNotFoundError, TypeError): nc_size_gb = None
    
    try:
        zarr_size_bytes = sum(os.path.getsize(os.path.join(dp, fn)) for dp, _, fns in os.walk(zarr_path) for fn in fns)
        zarr_size_gb = zarr_size_bytes / (1024**3)
    except Exception: zarr_size_gb = None

    nc_wt = timings.get('netcdf_write_time')
    z_wt = timings.get('zarr_write_time')
    
    nc_throughput = round(safe_div(nc_size_gb * 1024, nc_wt), 2) if nc_wt and nc_size_gb else None
    
    z_throughput = round(safe_div(zarr_size_gb * 1024, z_wt), 2) if z_wt and zarr_size_gb else None

    write_performance = {
        "NetCDF4": {
            "write_time_s": round(nc_wt, 2) if nc_wt is not None else None,
            "size_gb": round(nc_size_gb, 3) if nc_size_gb is not None else None,
            "throughput_mb_per_s": nc_throughput
        },
        "Zarr": {
            "write_time_s": round(z_wt, 2) if z_wt is not None else None,
            "size_gb": round(zarr_size_gb, 3) if zarr_size_gb is not None else None,
            "throughput_mb_per_s": z_throughput
        },
        "Comparison": {
            "write_speedup_zarr_vs_netcdf": round(safe_div(nc_wt, z_wt), 2) if nc_wt and z_wt else None,
            "size_ratio_zarr_vs_netcdf": round(safe_div(zarr_size_gb, nc_size_gb), 2) if nc_size_gb and zarr_size_gb else None
        }
    }
    
    # --- Assemble Final Report ---
    dask_workers = system_specs.get("Dask_Cluster", {}).get("Workers", 0)
    report = {
        "Pipeline_Summary": {
            "Total_Execution_Time_s": round(timings.get('total_pipeline_time', 0), 2),
            "Benchmark_Format": output_format,
            "Dask_Workers_Used": dask_workers,
            "Complete_Mosaics_Created": processing_stats.get('complete_mosaics_created', 0), 
            "Days_In_Datacube": len(processing_stats.get('datacube_days', []))
        },
        "Pipeline_Write_Performance": write_performance,
        "System_Specifications": system_specs
    }
    
    json_path = os.path.join(output_dir, f'benchmark_report_{output_format}_{dask_workers}_workers.json')
    with open(json_path, 'w') as f: json.dump(report, f, indent=4)
    logging.info(f"JSON report saved to: {json_path}")
    return report

def setup_logging():
    os.makedirs("./logs", exist_ok=True)
    log_file = f"./logs/benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)])
    logging.info(f"Logging initialized. Log file: {log_file}")

def setup_dask_cluster(config):
    """Sets up and scales the Dask SLURMCluster."""
    logging.info("Setting up Dask SLURMCluster...")
    cluster_config = config.copy()
    num_jobs = cluster_config.pop('slurm_jobs', 1)
    cluster = SLURMCluster(**cluster_config)
    client = Client(cluster)
    cluster.scale(jobs=num_jobs)
    workers_per_job = config.get('processes', 1)
    expected_workers = num_jobs * workers_per_job
    logging.info(f"Waiting for {expected_workers} Dask workers...")
    client.wait_for_workers(n_workers=expected_workers, timeout=900)
    logging.info(f"Dask cluster ready with {len(client.scheduler_info()['workers'])} workers.")
    return cluster, client

def _extract_tile_ids(files_by_day, complete_days):
    """Extracts unique MODIS tile IDs (e.g., h18v04) from file paths."""
    if not complete_days: return set()
    first_day = complete_days[0]
    tile_paths_for_day = files_by_day[first_day]
    tile_ids = {os.path.basename(p).split('.')[2] for p in tile_paths_for_day}
    return tile_ids

def add_cf_metadata(ds, included_days, tile_ids):
    """Adds comprehensive, CF-compliant metadata to the final data cube."""
    logging.info("Adding detailed CF-compliant metadata...")
    
    band_long_names = "Red, NIR, Blue, Green, SWIR 1, SWIR 2, SWIR 3, Usability Mask"
    band_wavelengths = "620-670 nm, 841-876 nm, 459-479 nm, 545-565 nm, 1230-1250 nm, 1628-1652 nm, 2105-2155 nm, N/A"

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
    ds['time'].attrs = {'long_name': 'time', 'standard_name': 'time'}
    ds['y'].attrs = {'long_name': 'y coordinate of projection', 'standard_name': 'projection_y_coordinate', 'units': 'm'}
    ds['x'].attrs = {'long_name': 'x coordinate of projection', 'standard_name': 'projection_x_coordinate', 'units': 'm'}
    ds['band'].attrs = {
        'long_name': 'Surface reflectance bands and usability mask',
        'band_names': band_long_names,
        'band_wavelengths_nm': band_wavelengths
    }
    ds['reflectance_and_mask'].attrs = {
        'long_name': 'Surface reflectance for bands 1-7 and usability mask for band 8',
        'units': '1',
        'grid_mapping': 'sinusoidal',
        'notes': 'Bands 1-7 are float32 scaled reflectance with a fill value of NaN. Band 8 is an uint8 integer usability mask. See usability_mask_key global attribute.'
    }
    
    return ds

def validate_config(config):
    """Validates required keys in the configuration dictionary."""
    for key in ['input_dir', 'output_dir', 'required_bands', 'tiles_per_day']:
        if key not in config: raise ValueError(f"Missing required configuration key: '{key}'")
    if not os.path.isdir(config['input_dir']):
        raise FileNotFoundError(f"Input directory not found: {config['input_dir']}")
    logging.info("Configuration file validated successfully.")


# --- MAIN WORKFLOW (HEAVY MODIFICATIONS) ---
def main(config_path, output_format):
    setup_logging()
    cluster, client = None, None
    
    try:
        with open(config_path, 'r') as f: config = yaml.safe_load(f)
        validate_config(config)
        os.makedirs(config['output_dir'], exist_ok=True)
        
        logging.info(f"Benchmark starting for output_format: {output_format}")
        
        cluster, client = setup_dask_cluster(config['dask_cluster'])
        
        timings = {}
        processing_stats = {
            'complete_mosaics_created': 0, 'datacube_days': [], 'datacube_created': False
        }
        pipeline_start_time = time.time()

        # === PHASES 1-3: Data Discovery, Processing, and Lazy Aggregation ===
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
        
        # --- PHASE 2/3: Lazy Graph Construction (CORE CHANGE) ---
        logging.info("="*60 + "\nPHASE 2/3: LAZY GRAPH CONSTRUCTION\n" + "="*60)
        
        # 1. Create a list of delayed tasks (lazily generated daily mosaics)
        lazy_mosaics = [get_lazy_mosaic(day, files_by_day[day], config) for day in complete_days]
        
        # --- Compute the metadata only for success check and sorting ---
        logging.info("Triggering small compute to check mosaic success for alignment...")
        # meta_results = client.compute(lazy_mosaics).result()
        meta_results = client.gather(client.compute(lazy_mosaics))
        
        successful_meta_mosaics = [r for r in meta_results if r and r.get('success')]
        successful_mosaics_dict = {r['day']: r['mosaic'] for r in successful_meta_mosaics}
        
        processing_stats.update({'complete_mosaics_created': len(successful_meta_mosaics), 'datacube_days': list(successful_mosaics_dict.keys())})
        if not successful_meta_mosaics: raise RuntimeError("No complete mosaics were created after processing.")
        
        # 3. Create a final, LAZY, consolidated data structure
        sorted_days = sorted(successful_mosaics_dict.keys())
        tile_ids = _extract_tile_ids(files_by_day, complete_days)
        time_coords = pd.to_datetime([f'2024-{day}' for day in sorted_days], format='%Y-%j')
        
        # NOTE: final_datacube is NOW LAZY (Dask graph remains open)
        final_datacube = xr.concat([successful_mosaics_dict[day] for day in sorted_days], dim=pd.Index(time_coords, name='time'))
        final_ds_lazy = final_datacube.to_dataset(name='reflectance_and_mask')
        
        final_ds_lazy = add_cf_metadata(final_ds_lazy, list(successful_mosaics_dict.keys()), tile_ids)
        final_ds_lazy['band'] = final_ds_lazy['band'].astype('O')

        
        # === PHASE 4: WRITING DATACUBES (MODIFIED) ===
        logging.info("="*60 + "\nPHASE 4: WRITING DATACUBES\n" + "="*60)
        
        nc_path = os.path.join(config['output_dir'], config['netcdf_filename'])
        zarr_path = os.path.join(config['output_dir'], config['zarr_store_name'])
        
        timings['netcdf_write_time'] = None
        timings['zarr_write_time'] = None

        
        # --- NETCDF PATH (SERIAL) ---
        if output_format == 'netcdf':
            logging.info(f"Saving NetCDF to: {nc_path} (SERIAL WRITE - IN-CORE GATHER)...")
            nc_write_start = time.time()
            
            # --- Trigger the full serial compute and gather for NetCDF ---
            final_ds_computed = final_ds_lazy.compute().load()
            
            netcdf_encoding = {'reflectance_and_mask': {'zlib': True, 'complevel': 4, 'shuffle': True}}
            final_ds_computed.to_netcdf(nc_path, engine='h5netcdf', encoding=netcdf_encoding)
            os.sync()
            timings['netcdf_write_time'] = time.time() - nc_write_start
            logging.info(f"NetCDF write complete in {timings['netcdf_write_time']:.2f} s")
        
        
        # --- ZARR PATH (OUT-OF-CORE PARALLEL) ---
        elif output_format == 'zarr':
            logging.info(f"Saving Zarr to: {zarr_path} (OUT-OF-CORE PARALLEL WRITE)...")
            zarr_write_start = time.time()
            
            # 1. Re-chunk the lazy dataset 
            final_ds_chunked_lazy = final_ds_lazy.chunk(config['zarr_chunks'])
            
            # 2. Call .to_zarr() with compute=True. Triggers full graph execution (Out-of-Core).
            zarr_job = final_ds_chunked_lazy.to_zarr(zarr_path, mode='w', compute=True)
            
            timings['zarr_write_time'] = time.time() - zarr_write_start
            logging.info(f"Zarr write complete in {timings['zarr_write_time']:.2f} s")
            
        processing_stats['datacube_created'] = True
        
        
        # =========================================================================
        # === PHASE 5: BENCHMARKING & REPORTING ===
        # =========================================================================
        logging.info("="*60 + "\nPHASE 5: BENCHMARKING & REPORTING\n" + "="*60)
        system_specs = get_system_specs(client)
        
        timings['total_pipeline_time'] = time.time() - pipeline_start_time
        
        report = generate_paper_ready_report(timings, config['output_dir'], config, processing_stats, system_specs, output_format)
        
        logging.info("="*60 + "\nPIPELINE COMPLETED\n" + "="*60)

        # Optional: Print a summary from the new report structure
        try:
            logging.info("\n--- KEY FINDINGS ---")
            summary = report['Pipeline_Summary']
            logging.info(f"  Benchmark Format: {summary['Benchmark_Format']}")
            logging.info(f"  Total Time ({summary['Dask_Workers_Used']} workers): {summary['Total_Execution_Time_s']:.2f} s")
            
            if summary['Benchmark_Format'] == 'netcdf':
                wp_nc = report['Pipeline_Write_Performance']['NetCDF4']
                logging.info(f"  NetCDF Write Time: {wp_nc['write_time_s']:.2f} s")
                logging.info(f"  NetCDF Throughput: {wp_nc['throughput_mb_per_s']} MB/s")
            elif summary['Benchmark_Format'] == 'zarr':
                wp_z = report['Pipeline_Write_Performance']['Zarr']
                logging.info(f"  Zarr Write Time: {wp_z['write_time_s']:.2f} s")
                logging.info(f"  Zarr Throughput: {wp_z['throughput_mb_per_s']} MB/s")
            
        except Exception as e:
            logging.warning(f"Could not print final summary: {e}")

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
    parser.add_argument('--format',
                        choices=['netcdf', 'zarr'],
                        required=True,
                        help='The output format to benchmark (netcdf or zarr).')
    args = parser.parse_args()
    
    try:
        main(config_path=args.config, output_format=args.format)
        sys.exit(0)
    except (Exception, SystemExit) as e:
        if isinstance(e, SystemExit) and e.code == 0: pass
        else:
            logging.error("Exiting due to an unrecoverable error.")
            sys.exit(1)
import os
import yaml
import xarray as xr
from dask.distributed import Client
from dask_jobqueue import SLURMCluster

# Updated to include geo_coords, and pixel grid /coord.......
from intrinsic_validation_logic_geo_pixel import (
    prepare_land_cover_map,
    create_stratified_subset,
    run_radiometric_checks,
    run_temporal_consistency_checks,
    run_difficulty_analysis,
)

def main():
    """
    Orchestrates the full intrinsic validation workflow on an HPC.
    """
    print("--- Starting Intrinsic Validation ---")

    # Initialize variables for cleanup
    cluster = None
    client = None

    try:
        # 1. Load Configurations
        with open("config_intrinsic.yaml", 'r') as f:
            config = yaml.safe_load(f)
        with open("config_hpc_validation_cluster.yaml", 'r') as f:
            hpc_config = yaml.safe_load(f)

        # Path setup from intrinsic config
        output_dir = config['output_dir']
        zarr_store_path = config['zarr_store_path']
        land_cover_path = config['land_cover_path']
        os.makedirs(output_dir, exist_ok=True)

        # 2. Setup Dask SLURMCluster for the HPC
        print("Setting up Dask SLURMCluster for HPC processing...")
        num_jobs = hpc_config.pop('slurm_jobs', 2)

        cluster = SLURMCluster(**hpc_config)
        cluster.scale(jobs=num_jobs)
        client = Client(cluster)

        expected_workers = num_jobs * hpc_config.get('processes', 1)
        print(f"Waiting for {expected_workers} workers... check job status with 'squeue -u $USER'")
        client.wait_for_workers(expected_workers)
        print(f"Cluster is ready. Found {len(client.scheduler_info()['workers'])} workers.")

        # 3. Load Main Dataset (lazily)
        print(f"Lazily opening Zarr dataset from: {zarr_store_path}")
        full_dataset = xr.open_zarr(zarr_store_path, chunks='auto')
        print("Dataset Schema:")
        print(full_dataset)
        
        # 4. Prepare Land Cover Map (Reproject then Mosaic)
        aligned_lc_output_path = os.path.join(output_dir, "aligned_landcover_map.tif")
        aligned_lc_map = prepare_land_cover_map(land_cover_path, full_dataset, aligned_lc_output_path)

        # --- Start Intrinsic Validation Analysis ---
        print("\n--- Starting Intrinsic Validation Analysis ---")

        # Create a representative subset
        subset_df = create_stratified_subset(
            aligned_lc_map,
            config['sampling']['land_cover_classes'],
            config['sampling']['num_samples_per_class'],
            config['patch_size']
        )
        subset_df.to_csv(os.path.join(output_dir, "subset_patch_coordinates.csv"))
        print(f"Generated a representative subset of {len(subset_df)} patches.")

        # Run Radiometric Checks
        run_radiometric_checks(
            full_dataset,
            subset_df,
            config['patch_size'],
            config['band_names'],
            output_dir,
            config['land_cover_map'] # Pass the new land cover map config
        )
        
        # Run Temporal Consistency Checks
        run_temporal_consistency_checks(
            full_dataset,
            aligned_lc_map,
            config['stable_target_classes'],
            output_dir
        )
        
        # Run Difficulty Analysis
        print("Starting dataset difficulty analysis...")
        run_difficulty_analysis(
            full_dataset,
            aligned_lc_map,
            config['band_names'],
            config['patch_size'],
            output_dir
        )

        print("\n--- Intrinsic Validation Complete ---")
        print(f"All plots and metrics have been saved to: {output_dir}")

    finally:
        print("Shutting down Dask client and cluster.")
        if client:
            client.close()
        if cluster:
            cluster.close()

if __name__ == "__main__":
    main()

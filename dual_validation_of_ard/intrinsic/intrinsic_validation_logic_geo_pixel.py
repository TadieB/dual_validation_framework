# Add random seed........ for reproducibility???
import os
import sys
import rioxarray
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm
from skimage.util import view_as_windows
import glob
from rioxarray.merge import merge_arrays

# --- HPC Adaptation ---
TQDM_DISABLE = not sys.stdout.isatty()

def prepare_land_cover_map(lc_glob_path, dataset_xr, output_path):
    """Performs a memory-efficient "reproject-then-mosaic" operation."""
    if os.path.exists(output_path):
        print(f"Loading existing aligned land cover map from {output_path}")
        return rioxarray.open_rasterio(output_path, chunks='auto')

    print("Starting memory-efficient land cover preparation (reproject then mosaic)...")
    tiff_files = glob.glob(lc_glob_path)
    if not tiff_files:
        raise FileNotFoundError(f"No files found matching the pattern: {lc_glob_path}")
    print(f"Found {len(tiff_files)} tiles to process individually.")

    reprojected_tiles = []
    for tile_path in tqdm(tiff_files, desc="Reprojecting Tiles", disable=TQDM_DISABLE):
        tile = rioxarray.open_rasterio(tile_path, chunks='auto')
        reprojected_tile = tile.rio.reproject_match(dataset_xr)
        reprojected_tiles.append(reprojected_tile)

    print("Mosaicking the reprojected tiles...")
    final_aligned_map = merge_arrays(reprojected_tiles)
    final_aligned_map = final_aligned_map.astype('uint8')
    final_aligned_map.rio.to_raster(output_path)
    print(f"Saved final aligned land cover map to {output_path}")
    return final_aligned_map

def create_stratified_subset(aligned_lc_map, classes_to_sample, num_samples_per_class, patch_size):
    """Selects a balanced set of data patch coordinates."""
    print(f"Sampling {num_samples_per_class} patches for each specified land cover type...")
    all_coords = []
    window_shape = (patch_size, patch_size)
    windows = view_as_windows(aligned_lc_map.squeeze().values, window_shape, step=patch_size)
    patch_rows, patch_cols, _, _ = windows.shape

    n_patches = patch_rows * patch_cols
    flat_patches = windows.reshape(n_patches, -1)
    dominant_classes_flat = stats.mode(flat_patches, axis=1, keepdims=False)[0]
    dominant_classes = dominant_classes_flat.reshape(patch_rows, patch_cols)

    for lc_class in tqdm(classes_to_sample, desc="Sampling Classes", disable=TQDM_DISABLE):
        class_indices_y, class_indices_x = np.where(dominant_classes == lc_class)
        if len(class_indices_y) > 0:
            num_available = len(class_indices_y)
            sample_count = min(num_samples_per_class, num_available)
            random_indices = np.random.choice(num_available, size=sample_count, replace=False)
            sampled_y = class_indices_y[random_indices] * patch_size
            sampled_x = class_indices_x[random_indices] * patch_size
            for y, x in zip(sampled_y, sampled_x):
                all_coords.append({'y': y, 'x': x, 'class': lc_class})
    return pd.DataFrame(all_coords)

def run_radiometric_checks(dataset_xr, subset_df, patch_size, band_names_config, output_dir, land_cover_config):
    """Generates plots with official names/colors and saves data to a CSV."""
    print("Running radiometric checks on the subset...")
    
    data_bands_only = dataset_xr.sel(band=[b for b in dataset_xr.band.values if "mask" not in str(b).lower()])
    
    pixel_data = []
    for _, patch_info in tqdm(subset_df.iterrows(), total=len(subset_df), desc="Extracting pixel data", disable=TQDM_DISABLE):
        y, x, lc_class = patch_info['y'], patch_info['x'], patch_info['class']
        patch = data_bands_only.isel(y=slice(y, y + patch_size), x=slice(x, x + patch_size))
        median_patch = patch.median(dim='time', skipna=True).compute()
        df = median_patch['reflectance_and_mask'].to_dataframe(name='reflectance').reset_index()
        df['land_cover_id'] = lc_class
        pixel_data.append(df)
    full_pixel_df = pd.concat(pixel_data, ignore_index=True)

    id_to_name = {k: v['name'] for k, v in land_cover_config.items()}
    name_to_color = {v['name']: v['color'] for k, v in land_cover_config.items()}
    full_pixel_df['land_cover_name'] = full_pixel_df['land_cover_id'].map(id_to_name)

    print("  - Generating per-band histograms by land cover...")
    g = sns.displot(data=full_pixel_df, x='reflectance', col='band', hue='land_cover_name',
                    kind='kde', col_wrap=4, palette=name_to_color,
                    facet_kws=dict(sharex=False, sharey=False))
    g.fig.suptitle("Per-Band Reflectance Distribution by Land Cover", y=1.02)
    plt.savefig(os.path.join(output_dir, "radiometric_histograms.png"), bbox_inches='tight')
    plt.close()

    print("  - Generating inter-band correlation heatmap...")
    corr_df = full_pixel_df.pivot_table(index=['y', 'x', 'land_cover_name'], columns='band', values='reflectance').reset_index()
    correlation_matrix = corr_df[list(data_bands_only.coords['band'].values)].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Inter-Band Correlation Matrix")
    plt.savefig(os.path.join(output_dir, "inter_band_correlation.png"), bbox_inches='tight')
    plt.close()

    print("  - Generating Red vs. NIR scatter plot...")
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=corr_df.sample(n=min(5000, len(corr_df))),
                    x=band_names_config['red'], y=band_names_config['nir'],
                    hue='land_cover_name', palette=name_to_color, alpha=0.7)
    plt.title("Red vs. NIR Reflectance by Land Cover")
    plt.savefig(os.path.join(output_dir, "red_vs_nir_scatter.png"), bbox_inches='tight')
    plt.close()

    csv_path = os.path.join(output_dir, "radiometric_checks_data.csv")
    corr_df.to_csv(csv_path, index=False)
    print(f"  - Saved radiometric pixel data to: {csv_path}")

def run_temporal_consistency_checks(dataset_xr, aligned_lc_map, stable_classes, output_dir):
    """Calculates temporal std dev and saves a summary to a CSV file."""
    print("Calculating temporal standard deviation on stable targets...")
    stable_mask = xr.zeros_like(aligned_lc_map.squeeze(), dtype=bool)
    for lc_class in stable_classes:
        stable_mask = stable_mask | (aligned_lc_map.squeeze() == lc_class)
    
    cloud_mask_band_name = next((b for b in dataset_xr['band'].values if 'mask' in str(b).lower()), None)
    if not cloud_mask_band_name:
        raise ValueError("Could not find a 'usability_mask' band in the dataset.")
        
    usability_mask = dataset_xr.sel(band=cloud_mask_band_name)
    clear_only_data = dataset_xr.where(usability_mask == 0) # 0 = Pristine
    stable_clear_data = clear_only_data.where(stable_mask)
    temporal_std = stable_clear_data.std(dim='time', skipna=True).compute()
    
    print("Mean Temporal Standard Deviation across stable, clear-sky areas:")
    summary_data = []
    for band_name in dataset_xr.coords['band'].values:
        if 'mask' in str(band_name).lower(): continue
        band_std_map = temporal_std.sel(band=band_name)
        band_std_map.rio.to_raster(os.path.join(output_dir, f"temporal_std_dev_{band_name}.tif"))
        mean_std = band_std_map['reflectance_and_mask'].mean(skipna=True).item()
        summary_data.append({'band': band_name, 'mean_temporal_std_dev': mean_std})
        print(f"  - Band {band_name}: {mean_std:.5f}")

    summary_df = pd.DataFrame(summary_data)
    csv_path = os.path.join(output_dir, "temporal_consistency_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"  - Saved temporal consistency summary to: {csv_path}")

def _calculate_shannon_diversity(data):
    """Helper to calculate Shannon's Diversity Index on a 2D patch."""
    _, counts = np.unique(data, return_counts=True)
    probabilities = counts / counts.sum()
    shannon_index = -np.sum(probabilities * np.log2(probabilities))
    return shannon_index

def run_difficulty_analysis(dataset_xr, aligned_lc_map, band_names_config, patch_size, output_dir):
    """Calculates difficulty metrics and saves them with BOTH coordinate systems."""
    print("Quantifying dataset difficulty across the entire ROI...")
    
    print("  - Calculating Land Cover Heterogeneity...")
    window_shape = (patch_size, patch_size)
    lc_windows = view_as_windows(aligned_lc_map.squeeze().values, window_shape, step=patch_size)
    patch_rows, patch_cols, _, _ = lc_windows.shape
    heterogeneity_map = np.zeros((patch_rows, patch_cols), dtype=np.float32)
    for r in tqdm(range(patch_rows), desc="Shannon Diversity", disable=TQDM_DISABLE):
        for c in range(patch_cols):
            heterogeneity_map[r, c] = _calculate_shannon_diversity(lc_windows[r, c])
    
    print("  - Calculating Phenological Variability (NDVI std dev)...")
    red = dataset_xr.sel(band=band_names_config['red'])['reflectance_and_mask']
    nir = dataset_xr.sel(band=band_names_config['nir'])['reflectance_and_mask']
    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = ((nir - red) / (nir + red)).chunk({'time': -1, 'y': 512, 'x': 512})
    
    ndvi_std = ndvi.std(dim='time', skipna=True)
    phenology_map = ndvi_std.coarsen(y=patch_size, x=patch_size, boundary='trim').mean().compute()
    
    print("  - Calculating Cloud Persistence...")
    cloud_mask_band_name = next((b for b in dataset_xr['band'].values if 'mask' in str(b).lower()), None)
    usability_mask = dataset_xr.sel(band=cloud_mask_band_name)['reflectance_and_mask'].chunk({'time': -1, 'y': 512, 'x': 512})
    binary_cloud_mask = (usability_mask == 4).astype('uint8') # 4 = Cloudy
    cloud_persistence = binary_cloud_mask.mean(dim='time', skipna=True)
    persistence_map = cloud_persistence.coarsen(y=patch_size, x=patch_size, boundary='trim').mean().compute()

    print("  - Combining metrics and saving to CSV...")
    def normalize(arr):
        min_val, max_val = np.nanmin(arr), np.nanmax(arr)
        if max_val == min_val: return np.zeros_like(arr)
        return (arr - min_val) / (max_val - min_val)
    
    h_norm = normalize(heterogeneity_map)
    v_norm = normalize(phenology_map.values)
    p_norm = normalize(persistence_map.values)
    
    difficulty_index = (0.33 * h_norm) + (0.33 * v_norm) + (0.33 * p_norm)
    
    # Create lists for both coordinate systems ===
    # Geographic coordinates (floats) from the coarsened map
    geo_y_coords = phenology_map.coords['y'].values
    geo_x_coords = phenology_map.coords['x'].values
    
    # Pixel coordinates (integers) corresponding to the top-left of each patch
    pixel_y_coords = np.arange(patch_rows) * patch_size
    pixel_x_coords = np.arange(patch_cols) * patch_size

    metrics_df = pd.DataFrame({
        'geo_y': np.repeat(geo_y_coords, patch_cols),
        'geo_x': np.tile(geo_x_coords, patch_rows),
        'y_coord': np.repeat(pixel_y_coords, patch_cols),
        'x_coord': np.tile(pixel_x_coords, patch_rows),
        'heterogeneity': heterogeneity_map.flatten(),
        'phenology_variability': phenology_map.values.flatten(),
        'cloud_persistence': persistence_map.values.flatten(),
        'difficulty_index': difficulty_index.flatten()
    })
    
    # Drop rows with NaN values that might have been created during calculations
    metrics_df.dropna(inplace=True)
    
    csv_path = os.path.join(output_dir, "dataset_difficulty_metrics.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"  - Saved patch-level difficulty metrics (with both coordinate systems) to: {csv_path}")

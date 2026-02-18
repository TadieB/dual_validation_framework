#!/usr/bin/env python3
"""
MODIS Tile Processor 
=====================================
"""

import os
import re
import logging
from typing import Dict, Optional

import numpy as np
import xarray as xr
import rioxarray
from rasterio.enums import Resampling

def extract_julian_day(filename: str) -> Optional[str]:
    """Extract Julian day from MODIS filename."""
    match = re.search(r'A\d{4}(\d{3})', filename)
    return match.group(1) if match else None

def create_band_quality_masks(qa_arr: xr.DataArray, required_bands: list) -> Dict[int, xr.DataArray]:
    """Create quality masks for each band."""
    band_quality_masks = {}
    for band in required_bands:
        bit_offset = 2 + 4 * (band - 1)
        band_quality = (qa_arr >> bit_offset) & 0b1111
        band_quality_masks[band] = (band_quality == 0)
    return band_quality_masks

def create_usability_mask(state_arr: xr.DataArray) -> xr.DataArray:
    """Create usability mask from state QA layer."""
    usability = xr.full_like(state_arr, 4, dtype=np.uint8)
    cloud_state = (state_arr >> 0) & 0b11
    cloud_shadow = (state_arr >> 2) & 0b1
    aerosol_quantity = (state_arr >> 6) & 0b11
    cirrus_detected = (state_arr >> 8) & 0b11
    adjacent_to_cloud = (state_arr >> 13) & 0b1
    
    is_cloudy = (cloud_state == 0b01) | (cloud_state == 0b10)
    is_contaminated = (adjacent_to_cloud == 1) | (cirrus_detected != 0b00)
    is_hazy = (aerosol_quantity == 0b11)
    is_shadow = (cloud_shadow == 1)
    is_clear = ~is_cloudy

    usability = usability.where(~(is_clear & ~is_shadow & ~is_hazy & ~is_contaminated), 0)
    usability = usability.where(~(is_clear & is_shadow), 1)
    usability = usability.where(~(is_clear & is_hazy), 2)
    usability = usability.where(~(is_clear & is_contaminated), 3)
    usability = usability.where(state_arr.data != state_arr.rio.nodata, 255)
    
    usability = usability.rio.set_nodata(255)
    return usability

def assess_quality_fraction(band_quality_masks: Dict[int, xr.DataArray], min_valid_fraction: float) -> bool:
    """Check if bands meet minimum quality fraction."""
    masks_to_check = list(band_quality_masks.values())
    total_valid = sum(mask.sum().item() for mask in masks_to_check)
    total_pixels = sum(mask.size for mask in masks_to_check)
    fraction = total_valid / total_pixels if total_pixels > 0 else 0.0
    return fraction >= min_valid_fraction

def process_tile(hdf_path: str, config: Dict) -> Optional[xr.DataArray]:
    """Process a single MODIS tile."""
    try:
        grid_500m = "MODIS_Grid_500m_2D"
        grid_1km = "MODIS_Grid_1km_2D"
        required_bands = config["required_bands"]
        
        band_sds = [f'HDF4_EOS:EOS_GRID:"{hdf_path}":{grid_500m}:sur_refl_b0{b}_1' for b in required_bands]
        qc_sds = f'HDF4_EOS:EOS_GRID:"{hdf_path}":{grid_500m}:QC_500m_1'
        state_sds = f'HDF4_EOS:EOS_GRID:"{hdf_path}":{grid_1km}:state_1km_1'

        # Load data
        refl_arrays = [rioxarray.open_rasterio(sd).squeeze('band', drop=True) for sd in band_sds]
        qc_500m_arr = rioxarray.open_rasterio(qc_sds).squeeze('band', drop=True)
        state_1km_arr = rioxarray.open_rasterio(state_sds).squeeze('band', drop=True)

        # Quality assessment
        band_quality_masks = create_band_quality_masks(qc_500m_arr, required_bands)
        if not assess_quality_fraction(band_quality_masks, config["min_valid_fraction"]):
            logging.warning(
                f"REJECTING TILE: {os.path.basename(hdf_path)} - "
                f"Less than {config['min_valid_fraction']:.0%} quality pixels"
            )
            return None
            
        # Process bands
        processed_bands = []
        for i, band_da in enumerate(refl_arrays):
            band_num = required_bands[i]
            valid_range = config['valid_range']
            scaled = band_da * config['scale_factor']
            clean = scaled.where((scaled >= valid_range[0]) & (scaled <= valid_range[1]))
            masked = clean.where(band_quality_masks[band_num])
            processed_bands.append(masked)
        
        # Process usability mask
        usability_mask_1km = create_usability_mask(state_1km_arr)
        usability_mask_500m = usability_mask_1km.rio.reproject_match(
            processed_bands[0], 
            resampling=Resampling.nearest,
            nodata=255
        )
        
        # Combine
        final_da = xr.concat(processed_bands + [usability_mask_500m], dim="band")
        band_names = [f'B{b:02d}' for b in required_bands] + ['usability_mask']
        final_da = final_da.assign_coords(band=band_names)
        
        return final_da

    except Exception as e:
        logging.error(f"Failed to process tile {os.path.basename(hdf_path)}: {e}")
        return None

import numpy as np
import xarray as xr
import torch
import random
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import os
import json

# =================================================================================
# FULLY CORRECTED DATALOADER (v3 - Spatial Split)
#
# 1. DATA LEAK (FIXED): Implements a spatial split. Patches for train, val,
#    and test are from different geographic regions, ensuring zero data leak.
# 2. AUGMENTATION (FIXED): Augmentation is now applied *before* target selection,
#    guaranteeing inputs and targets are always aligned.
# 3. NORMALIZATION (CORRECT): Scales true physical range [-0.01, 1.6] to [0, 1].
# 4. NaN HANDLING (CORRECT): Replaces NaNs with -1.0 for model input.
# 5. MASKED LOSS (CORRECT): Creates a per-band (C, H, W) "True Pristine" mask
#    that checks both 1km QA (Pristine) and 500m QA (not NaN).
# 6. SPLIT SAVING (ADDED): Saves JSON files for indices AND summary counts.
# =================================================================================


def filter_collate_fn(batch):
    """
    Custom collate function that filters out None values from a batch.
    Required to handle samples discarded by the JIT quality check.
    """
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def _parse_usability_mask_key(key_string):
    """Parse mask encoding from Zarr metadata."""
    mapping = {}
    parts = key_string.split(', ')
    for part in parts:
        value, name = part.split('=')
        mapping[name.strip()] = int(value)
    return mapping


class CloudGapDataset(Dataset):
    """
    High-performance, leak-free dataset for cloud-gap imputation.
    Splits data spatially, normalizes, sanitizes, and quality-checks on-the-fly.
    """

    def __init__(
        self,
        zarr_path,
        split='train',
        temporal_window=4,
        spatial_size=224,
        train_split=0.7,
        val_split=0.15,
        random_seed=42,
        augment=True,
        min_pristine_fraction=0.85,
    ):
        self.zarr_path = zarr_path
        self.split = split
        self.temporal_window = temporal_window
        self.spatial_size = spatial_size
        self.augment = augment and (split == 'train')
        self.min_pristine_fraction = min_pristine_fraction

        # --- Define the true physical range of MODIS data (Scaled) ---
        self.PHYSICAL_MIN = -0.01
        self.PHYSICAL_MAX = 1.6
        self.PHYSICAL_RANGE = self.PHYSICAL_MAX - self.PHYSICAL_MIN
        
        self.zarr_store = xr.open_zarr(zarr_path, chunks='auto')
        self.data = self.zarr_store['reflectance_and_mask']

        mask_key_str = self.data.attrs.get('usability_mask_key', '0=Pristine, 1=Shadow, 2=Hazy, 3=Contaminated, 4=Cloud')
        self.mask_mapping = _parse_usability_mask_key(mask_key_str)
        self.pristine_value = self.mask_mapping.get('Pristine', 0)
        self.bad_mask_values = [v for k, v in self.mask_mapping.items() if k != 'Pristine']

        self.num_times, self.num_bands, self.height, self.width = self.data.shape
        
        # --- START OF THE "SPATIAL SPLIT" DATA LEAK FIX ---
        
        # 1. Create a master list of *spatial patches only*
        spatial_indices = [(y, x) for y in range(0, self.height - self.spatial_size + 1, self.spatial_size) for x in range(0, self.width - self.spatial_size + 1, self.spatial_size)]
        
        # 2. Shuffle this *spatial* list
        random.Random(random_seed).shuffle(spatial_indices)
        
        # 3. Split the *spatial list* into train, val, and test sets
        num_spatial_patches = len(spatial_indices)
        train_end = int(num_spatial_patches * train_split)
        val_end = train_end + int(num_spatial_patches * val_split)

        # Get the unique spatial patches for this split
        if split == 'train':
            split_spatial_patches = spatial_indices[:train_end]
        elif split == 'val':
            split_spatial_patches = spatial_indices[train_end:val_end]
        else: # 'test'
            split_spatial_patches = spatial_indices[val_end:]
        
        # 4. Now, generate temporal samples *only* from the assigned spatial patches
        self.indices = []
        if split == 'train':
            # Training set uses all temporal windows (stride=1) for max data
            temporal_starts = list(range(0, self.num_times - self.temporal_window + 1, 1))
            self.indices = [(y, x, t_start) for (y, x) in split_spatial_patches for t_start in temporal_starts]
        else:
            # Val/Test sets use non-overlapping temporal windows (stride=4) for speed
            temporal_starts = list(range(0, self.num_times - self.temporal_window + 1, self.temporal_window))
            self.indices = [(y, x, t_start) for (y, x) in split_spatial_patches for t_start in temporal_starts]
        
        # 5. Shuffle the final list of (y, x, t_start) tuples
        random.Random(random_seed).shuffle(self.indices) # Shuffle again to mix spatial/temporal
        
        # --- END OF THE DATA LEAK FIX ---
        
        print(f"[{split.upper()}] Zarr shape: T={self.num_times}, H={self.height}, W={self.width}. Created {len(self.indices)} samples from {len(split_spatial_patches)} unique spatial patches.")

    def __len__(self):
        return len(self.indices)

    def _select_target_frame(self, usability_mask_window):
        # Finds index of frame with fewest "bad" (non-pristine) pixels
        bad_pixel_fractions = np.mean(np.isin(usability_mask_window, self.bad_mask_values), axis=(1, 2))
        return np.argmin(bad_pixel_fractions)

    def _apply_augmentation(self, reflectance_window, mask_window):
        # Applies the same geometric augmentations to both data and masks
        
        # Horizontal Flip
        if random.random() < 0.5:
            reflectance_window = np.ascontiguousarray(reflectance_window[:, :, :, ::-1])
            mask_window = np.ascontiguousarray(mask_window[:, :, ::-1])
        # Vertical Flip
        if random.random() < 0.5:
            reflectance_window = np.ascontiguousarray(reflectance_window[:, :, ::-1, :])
            mask_window = np.ascontiguousarray(mask_window[:, ::-1, :])
        
        # Rotation (0, 90, 180, 270 degrees)
        k = random.randint(0, 3)
        if k > 0:
            reflectance_window = np.ascontiguousarray(np.rot90(reflectance_window, k, axes=(2, 3)))
            mask_window = np.ascontiguousarray(np.rot90(mask_window, k, axes=(1, 2)))
        
        return reflectance_window, mask_window

    def __getitem__(self, idx):
        y, x, t_start = self.indices[idx]

        # 1. Load the full cube slice
        cube = self.data.isel(y=slice(y, y + self.spatial_size), x=slice(x, x + self.spatial_size)).values

        t_end = t_start + self.temporal_window
        
        # 2. Get original data *with NaNs* (6 bands)
        # Shape: (T, C, H, W) where T=temporal_window, C=6
        reflectance_window_with_nans = cube[t_start:t_end, :6, :, :]
        # Get 1km usability mask
        # Shape: (T, H, W)
        mask_window = cube[t_start:t_end, -1, :, :]
        
        # 3. Apply Augmentation (if train) *BEFORE* target selection
        if self.augment:
            reflectance_window_with_nans, mask_window = self._apply_augmentation(reflectance_window_with_nans, mask_window)

        # 4. Select best target frame index *after* augmentation
        target_idx_in_window = self._select_target_frame(mask_window)
        # Get the 2D (H, W) usability mask for the chosen target
        target_usability_mask = mask_window[target_idx_in_window]
        
        # 5. Check quality of this target frame
        pristine_fraction = np.mean(target_usability_mask == self.pristine_value)
        if pristine_fraction < self.min_pristine_fraction:
            return None # Skip this sample

        # --- DATA NORMALIZATION & SANITIZATION ---
        
        # 6. Normalize all *valid* data to [0, 1]. NaNs will stay NaN.
        normalized_window = (reflectance_window_with_nans - self.PHYSICAL_MIN) / self.PHYSICAL_RANGE
        
        # 7. Create the MODEL INPUT TENSOR (T, C, H, W)
        # Now, replace remaining NaNs with -1.0
        reflectance_window_input = np.nan_to_num(normalized_window, nan=-1.0)

        # 8. Get the normalized TARGET (C, H, W) - *with NaNs still in it*
        ground_truth_normalized_with_nans = normalized_window[target_idx_in_window]
        
        # --- REFINED MASK AND TARGET ---

        # 9. Create the "TRUE PRISTINE" MASK (C, H, W)
        # Condition 1: 1km usability mask must be "Pristine" (broadcasts to all channels)
        pristine_cond_3d = np.expand_dims(target_usability_mask == self.pristine_value, axis=0)
        # Condition 2: 500m original data must *not* be NaN (per-band check)
        nan_cond_3d = ~np.isnan(ground_truth_normalized_with_nans)
        
        # Final mask is True ONLY for pixels that pass BOTH checks
        valid_mask = pristine_cond_3d & nan_cond_3d # Shape (C, H, W)

        # 10. Create final tensors
        # Input: (C, T, H, W) - Valid data is [0, 1], Missing is -1
        # Transpose from (T, C, H, W) to (C, T, H, W) for models
        input_tensor = torch.from_numpy(np.ascontiguousarray(np.transpose(reflectance_window_input, (1, 0, 2, 3)))).float()
        
        # Target: (C, H, W) - Replace NaN with -1.0 (for consistency)
        ground_truth_final = np.nan_to_num(ground_truth_normalized_with_nans, nan=-1.0) 
        target_tensor = torch.from_numpy(np.ascontiguousarray(ground_truth_final)).float()
        
        # Loss Mask: (C, H, W) - The "True Pristine" mask
        valid_mask_tensor = torch.from_numpy(valid_mask).bool()

        # --- FINAL STEP: Create the return dictionary ---
            
        # This is the base dictionary for all splits
        return_dict = {
            'input': input_tensor,
            'target': target_tensor,
            'valid_mask': valid_mask_tensor,
            'pristine_fraction': pristine_fraction,
        }

        # Add coordinates ONLY for val and test splits (for inference/analysis)
        if self.split != 'train':
            return_dict['y_coord'] = y
            return_dict['x_coord'] = x
            
        return return_dict


class CloudGapDataModule:
    def __init__(self, zarr_path, batch_size=4, num_workers=4, **kwargs):
        self.zarr_path = zarr_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_kwargs = kwargs

    def setup(self):
        # Pop 'output_dir' from kwargs. We need it for saving splits,
        # but CloudGapDataset's constructor doesn't accept it.
        output_dir = self.dataset_kwargs.pop('output_dir', '.')
        
        self.train_dataset = CloudGapDataset(self.zarr_path, split='train', augment=True, **self.dataset_kwargs)
        self.val_dataset = CloudGapDataset(self.zarr_path, split='val', augment=False, **self.dataset_kwargs)
        self.test_dataset = CloudGapDataset(self.zarr_path, split='test', augment=False, **self.dataset_kwargs)

        # --- SAVE SPLITS LOGIC ---
        splits_dir = os.path.join(output_dir, 'data_splits')
        os.makedirs(splits_dir, exist_ok=True)
        
        # Define file paths
        train_path = os.path.join(splits_dir, 'train_indices.json')
        val_path = os.path.join(splits_dir, 'val_indices.json')
        test_path = os.path.join(splits_dir, 'test_indices.json')
        summary_path = os.path.join(splits_dir, 'data_splits_summary.json')

        # Save only if files don't already exist
        if not os.path.exists(train_path):
            print(f"Saving data splits to {splits_dir}...")
            # Save index lists
            with open(train_path, 'w') as f:
                json.dump(self.train_dataset.indices, f)
            with open(val_path, 'w') as f:
                json.dump(self.val_dataset.indices, f)
            with open(test_path, 'w') as f:
                json.dump(self.test_dataset.indices, f)
            
            # Save summary file
            summary = {
                'train_samples': len(self.train_dataset.indices),
                'val_samples': len(self.val_dataset.indices),
                'test_samples': len(self.test_dataset.indices),
                'total_samples': len(self.train_dataset.indices) + len(self.val_dataset.indices) + len(self.test_dataset.indices)
            }
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=4)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, collate_fn=filter_collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True, collate_fn=filter_collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True, collate_fn=filter_collate_fn
        )


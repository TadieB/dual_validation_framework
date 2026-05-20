import numpy as np
import pandas as pd
import xarray as xr
import torch
import random
from torch.utils.data import Dataset, DataLoader
import os
import json

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
        difficulty_csv_path=None # ADDED for Inference
    ):
        self.zarr_path = zarr_path
        self.split = split
        self.temporal_window = temporal_window
        self.spatial_size = spatial_size
        self.augment = augment and (split == 'train')
        self.min_pristine_fraction = min_pristine_fraction

        # --- Physical Normalization Bounds ---
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
        
        # --- SPATIAL SPLIT (DATA LEAK FIX) ---
        spatial_indices = [(y, x) for y in range(0, self.height - self.spatial_size + 1, self.spatial_size) 
                           for x in range(0, self.width - self.spatial_size + 1, self.spatial_size)]
        
        random.Random(random_seed).shuffle(spatial_indices)
        
        num_spatial_patches = len(spatial_indices)
        train_end = int(num_spatial_patches * train_split)
        val_end = train_end + int(num_spatial_patches * val_split)

        if split == 'train':
            split_spatial_patches = spatial_indices[:train_end]
        elif split == 'val':
            split_spatial_patches = spatial_indices[train_end:val_end]
        else: 
            split_spatial_patches = spatial_indices[val_end:]
        
        # Generate Temporal Samples
        self.indices = []
        if split == 'train':
            temporal_starts = list(range(0, self.num_times - self.temporal_window + 1, 1))
        else:
            temporal_starts = list(range(0, self.num_times - self.temporal_window + 1, self.temporal_window))
            
        self.indices = [(y, x, t_start) for (y, x) in split_spatial_patches for t_start in temporal_starts]
        random.Random(random_seed).shuffle(self.indices) 

        # --- STRATIFIED EVALUATION LOGIC ---
        self.difficulty_map = {}
        if self.split != 'train' and difficulty_csv_path is not None and os.path.exists(difficulty_csv_path):
            print(f"[{split.upper()}] Loading difficulty metrics from {difficulty_csv_path}...")
            df_diff = pd.read_csv(difficulty_csv_path)
            self.difficulty_map = {(row['y_coord'], row['x_coord']): row['stratum'] for _, row in df_diff.iterrows()}
        
        print(f"[{split.upper()}] Zarr shape: T={self.num_times}, H={self.height}, W={self.width}. Created {len(self.indices)} samples.")

    def __len__(self):
        return len(self.indices)

    def _select_target_frame(self, usability_mask_window):
        bad_pixel_fractions = np.mean(np.isin(usability_mask_window, self.bad_mask_values), axis=(1, 2))
        return np.argmin(bad_pixel_fractions)

    def _apply_augmentation(self, reflectance_window, mask_window):
        if random.random() < 0.5:
            reflectance_window = np.ascontiguousarray(reflectance_window[:, :, :, ::-1])
            mask_window = np.ascontiguousarray(mask_window[:, :, ::-1])
        if random.random() < 0.5:
            reflectance_window = np.ascontiguousarray(reflectance_window[:, :, ::-1, :])
            mask_window = np.ascontiguousarray(mask_window[:, ::-1, :])
        k = random.randint(0, 3)
        if k > 0:
            reflectance_window = np.ascontiguousarray(np.rot90(reflectance_window, k, axes=(2, 3)))
            mask_window = np.ascontiguousarray(np.rot90(mask_window, k, axes=(1, 2)))
        return reflectance_window, mask_window

    def __getitem__(self, idx):
        y, x, t_start = self.indices[idx]
        cube = self.data.isel(y=slice(y, y + self.spatial_size), x=slice(x, x + self.spatial_size)).values

        t_end = t_start + self.temporal_window
        reflectance_window_with_nans = cube[t_start:t_end, :6, :, :]
        mask_window = cube[t_start:t_end, -1, :, :]
        
        if self.augment:
            reflectance_window_with_nans, mask_window = self._apply_augmentation(reflectance_window_with_nans, mask_window)

        target_idx_in_window = self._select_target_frame(mask_window)
        target_usability_mask = mask_window[target_idx_in_window]
        
        pristine_fraction = np.mean(target_usability_mask == self.pristine_value)
        if pristine_fraction < self.min_pristine_fraction:
            return None

        # --- NORMALIZATION & SANITIZATION ---
        normalized_window = (reflectance_window_with_nans - self.PHYSICAL_MIN) / self.PHYSICAL_RANGE
        reflectance_window_input = np.nan_to_num(normalized_window, nan=0.0) 
        ground_truth_normalized_with_nans = normalized_window[target_idx_in_window]

        # --- SYNTHETIC REALISTIC CLOUD MASKING ---
        # 1. Get indices of the other time steps in this loaded patch
        other_indices = [i for i in range(self.temporal_window) if i != target_idx_in_window]
        
        if other_indices:
            # 2. Randomly pick one of those other days to be the "donor" mask
            donor_idx = random.choice(other_indices)
            donor_mask = mask_window[donor_idx]
            
            # 3. Find where the donor mask is not pristine (i.e., clouds, shadows)
            synthetic_clouds = (donor_mask != self.pristine_value)
            
            # 4. Zero out those pixels across all 6 bands in the target input frame
            reflectance_window_input[target_idx_in_window, :, synthetic_clouds] = 0.0
        # -----------------------------------------
        
        pristine_cond_3d = np.expand_dims(target_usability_mask == self.pristine_value, axis=0)
        nan_cond_3d = ~np.isnan(ground_truth_normalized_with_nans)
        valid_mask = pristine_cond_3d & nan_cond_3d

        input_tensor = torch.from_numpy(np.ascontiguousarray(np.transpose(reflectance_window_input, (1, 0, 2, 3)))).float()
        ground_truth_final = np.nan_to_num(ground_truth_normalized_with_nans, nan=0.0) 
        target_tensor = torch.from_numpy(np.ascontiguousarray(ground_truth_final)).float()
        valid_mask_tensor = torch.from_numpy(valid_mask).bool()

        return_dict = {
            'input': input_tensor,
            'target': target_tensor,
            'valid_mask': valid_mask_tensor,
            'pristine_fraction': pristine_fraction,
        }

        # --- ADD INFERENCE METADATA ---
        if self.split != 'train':
            return_dict['y_coord'] = y
            return_dict['x_coord'] = x
            # Append difficulty stratum if the CSV was loaded
            if self.difficulty_map:
                return_dict['difficulty_stratum'] = self.difficulty_map.get((y, x), 'Unknown')
            
        return return_dict


class CloudGapDataModule:
    def __init__(self, zarr_path, batch_size=4, num_workers=4, **kwargs):
        self.zarr_path = zarr_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_kwargs = kwargs

    def setup(self):
        output_dir = self.dataset_kwargs.pop('output_dir', '.')
        # Pass the difficulty CSV to all, but the Dataset class will ignore it for 'train'
        difficulty_csv = self.dataset_kwargs.get('difficulty_csv_path', None)
        
        self.train_dataset = CloudGapDataset(self.zarr_path, split='train', augment=True, **self.dataset_kwargs)
        self.val_dataset = CloudGapDataset(self.zarr_path, split='val', augment=False, **self.dataset_kwargs)
        self.test_dataset = CloudGapDataset(self.zarr_path, split='test', augment=False, **self.dataset_kwargs)

        # Save Splits Logic
        splits_dir = os.path.join(output_dir, 'data_splits')
        os.makedirs(splits_dir, exist_ok=True)
        
        train_path = os.path.join(splits_dir, 'train_indices.json')
        val_path = os.path.join(splits_dir, 'val_indices.json')
        test_path = os.path.join(splits_dir, 'test_indices.json')
        summary_path = os.path.join(splits_dir, 'data_splits_summary.json')

        if not os.path.exists(train_path):
            print(f"Saving leak-proof data splits to {splits_dir}...")
            with open(train_path, 'w') as f: json.dump(self.train_dataset.indices, f)
            with open(val_path, 'w') as f: json.dump(self.val_dataset.indices, f)
            with open(test_path, 'w') as f: json.dump(self.test_dataset.indices, f)
            
            summary = {
                'train_samples': len(self.train_dataset.indices),
                'val_samples': len(self.val_dataset.indices),
                'test_samples': len(self.test_dataset.indices),
            }
            with open(summary_path, 'w') as f: json.dump(summary, f, indent=4)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True, collate_fn=filter_collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True, collate_fn=filter_collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True, collate_fn=filter_collate_fn)

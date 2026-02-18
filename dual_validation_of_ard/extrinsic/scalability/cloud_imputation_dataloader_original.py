import numpy as np
import xarray as xr
import torch
import random
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm

# =================================================================================
# FINAL DATALOADER FEATURES:
# 1. FAST INITIALIZATION: Uses "Just-in-Time" (JIT) quality filtering in `__getitem__`
#    to make the initial script startup instantaneous.
# 2. DATA SANITIZATION (NaN FIX): Forcefully cleans the data before it enters the
#    model by replacing any `NaN` values with 0 and clipping the data to a
#    strict [0, 1] range. This is the definitive fix for the `loss=nan` error.
# 3. REQUIRED COLLATE FUNCTION: Includes the `filter_collate_fn` which must be used
#    in the DataLoader to handle samples discarded by the JIT quality check.
# 4. UNIFIED DATAMODULE: Manages train/val/test splits from a single Zarr cube.
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
    High-performance dataset for cloud-gap imputation.
    Loads, sanitizes, and quality-checks spatio-temporal windows on-the-fly.
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

        self.zarr_store = xr.open_zarr(zarr_path, chunks='auto')
        self.data = self.zarr_store['reflectance_and_mask']

        mask_key_str = self.data.attrs.get('usability_mask_key', '0=Pristine, 1=Shadow, 2=Hazy, 3=Contaminated, 4=Cloud')
        self.mask_mapping = _parse_usability_mask_key(mask_key_str)
        self.pristine_value = self.mask_mapping.get('Pristine', 0)
        self.bad_mask_values = [v for k, v in self.mask_mapping.items() if k != 'Pristine']

        self.num_times, self.num_bands, self.height, self.width = self.data.shape
        
        spatial_indices = [(y, x) for y in range(0, self.height - self.spatial_size + 1, self.spatial_size) for x in range(0, self.width - self.spatial_size + 1, self.spatial_size)]
        temporal_stride = 1 if split == 'train' else self.temporal_window
        temporal_starts = list(range(0, self.num_times - self.temporal_window + 1, temporal_stride))
        all_indices = [(y, x, t_start) for (y, x) in spatial_indices for t_start in temporal_starts]
        
        random.Random(random_seed).shuffle(all_indices)
        
        num_samples = len(all_indices)
        train_end = int(num_samples * train_split)
        val_end = train_end + int(num_samples * val_split)

        if split == 'train':
            self.indices = all_indices[:train_end]
        elif split == 'val':
            self.indices = all_indices[train_end:val_end]
        else:
            self.indices = all_indices[val_end:]
        
        print(f"[{split.upper()}] Zarr shape: T={self.num_times}, H={self.height}, W={self.width}. Created {len(self.indices)} potential samples.")

    def __len__(self):
        return len(self.indices)

    def _select_target_frame(self, usability_mask_window):
        bad_pixel_fractions = np.mean(np.isin(usability_mask_window, self.bad_mask_values), axis=(1, 2))
        return np.argmin(bad_pixel_fractions)

    def _apply_augmentation(self, reflectance, mask):
        if random.random() < 0.5:
            reflectance = np.ascontiguousarray(reflectance[:, :, :, ::-1])
            mask = np.ascontiguousarray(mask[:, :, ::-1])
        if random.random() < 0.5:
            reflectance = np.ascontiguousarray(reflectance[:, :, ::-1, :])
            mask = np.ascontiguousarray(mask[:, ::-1, :])
        
        k = random.randint(0, 3)
        if k > 0:
            reflectance = np.ascontiguousarray(np.rot90(reflectance, k, axes=(2, 3)))
            mask = np.ascontiguousarray(np.rot90(mask, k, axes=(1, 2)))
        
        return reflectance, mask

    def __getitem__(self, idx):
        y, x, t_start = self.indices[idx]

        cube = self.data.isel(y=slice(y, y + self.spatial_size), x=slice(x, x + self.spatial_size)).values

        t_end = t_start + self.temporal_window
        reflectance_window = cube[t_start:t_end, :-1, :, :]
        mask_window = cube[t_start:t_end, -1, :, :]
        
        target_idx_in_window = self._select_target_frame(mask_window)
        target_mask = mask_window[target_idx_in_window]
        pristine_fraction = np.mean(target_mask == self.pristine_value)

        if pristine_fraction < self.min_pristine_fraction:
            return None

        # --- DATA SANITIZATION  ---
        # Step 1: Replace any potential NaN values with 0.0
        reflectance_window = np.nan_to_num(reflectance_window, nan=0.0)
        # Step 2: Clip all values to be strictly within the [0, 1] range.
        reflectance_window = np.clip(reflectance_window, 0.0, 1.0)

        reflectance_window = reflectance_window[:, :6, :, :]

        if self.augment:
            reflectance_window, mask_window = self._apply_augmentation(reflectance_window, mask_window)
            target_idx_in_window = self._select_target_frame(mask_window)
            target_mask = mask_window[target_idx_in_window]

        input_sequence = np.transpose(reflectance_window, (1, 0, 2, 3))
        ground_truth = reflectance_window[target_idx_in_window]
        valid_mask = (target_mask == self.pristine_value)

        input_tensor = torch.from_numpy(np.ascontiguousarray(input_sequence)).float()
        target_tensor = torch.from_numpy(np.ascontiguousarray(ground_truth)).float()
        valid_mask_tensor = torch.from_numpy(valid_mask).bool()

        return {
            'input': input_tensor,
            'target': target_tensor,
            'valid_mask': valid_mask_tensor,
            'pristine_fraction': pristine_fraction,
        }


class CloudGapDataModule:
    def __init__(self, zarr_path, batch_size=4, num_workers=4, **kwargs):
        self.zarr_path = zarr_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_kwargs = kwargs

    def setup(self):
        self.train_dataset = CloudGapDataset(self.zarr_path, split='train', augment=True, **self.dataset_kwargs)
        self.val_dataset = CloudGapDataset(self.zarr_path, split='val', augment=False, **self.dataset_kwargs)
        self.test_dataset = CloudGapDataset(self.zarr_path, split='test', augment=False, **self.dataset_kwargs)

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

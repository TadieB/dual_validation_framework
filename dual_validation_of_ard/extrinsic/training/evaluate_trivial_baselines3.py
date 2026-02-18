import torch
import numpy as np
from tqdm import tqdm
import argparse
import pandas as pd
import os
import random

import sys
sys.path.insert(0, os.getcwd())

from inference_dataloader import CloudGapDataModule
from inference_metric3 import ImputationMetrics, save_visualization

def evaluate_true_least_cloudy(batch):
    """Selects the input frame closest to ground truth for each sample."""
    inputs = batch['input'].numpy()
    targets = batch['target'].numpy()
    best_preds = []
    for i in range(inputs.shape[0]):
        sample_inputs, sample_target = inputs[i], targets[i]
        # Calculate MSE for each temporal frame against ground truth
        mses = np.mean((sample_inputs - sample_target[:, np.newaxis, :, :])**2, axis=(0, 2, 3))
        best_input_idx = np.argmin(mses)
        best_preds.append(sample_inputs[:, best_input_idx, :, :])
    return torch.from_numpy(np.array(best_preds))

def evaluate_mosaicking(batch):
    """Takes median across all temporal frames."""
    input_sequences = batch['input'].numpy()
    return torch.from_numpy(np.median(input_sequences, axis=2))

def main(args):
    
    random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    vis_dir = os.path.join(args.output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    datamodule = CloudGapDataModule(
        zarr_path=args.zarr_path, 
        batch_size=args.batch_size,
        num_workers=args.num_workers, 
        random_seed=42
    )
    datamodule.setup()
    test_loader = datamodule.test_dataloader()
    print(f"Test set size: {len(datamodule.test_dataset)}")

    total_samples = len(datamodule.test_dataset)
    num_vis = min(args.num_vis, total_samples)
    vis_indices = sorted(random.sample(range(total_samples), num_vis))
    print(f"\nWill save {num_vis} visualizations at sample indices:")
    print(f"{vis_indices[:10]}{'...' if len(vis_indices) > 10 else ''}\n")

    results_lc = []
    results_mosaic = []
    sample_count = 0
    vis_count = 0

    print("Starting evaluation on the test set...")
    for i, batch in enumerate(tqdm(test_loader, desc="Evaluating Baselines")):
        if batch is None: 
            continue
        
        inputs, targets = batch['input'], batch['target']
        valid_masks = batch['valid_mask']
        y_coords, x_coords = batch['y_coord'], batch['x_coord']

        pred_lc = evaluate_true_least_cloudy(batch)
        pred_mosaic = evaluate_mosaicking(batch)

        for j in range(pred_lc.shape[0]):
            mask_j = valid_masks[j].to(device)
            
            y, x = y_coords[j].item(), x_coords[j].item()
            
            if mask_j.sum() > 0:
                metrics_lc_sample = ImputationMetrics.compute_all(
                    pred_lc[j].to(device), targets[j].to(device), mask_j
                )
                results_lc.append({'y_coord': y, 'x_coord': x, **metrics_lc_sample})
                
                metrics_mosaic_sample = ImputationMetrics.compute_all(
                    pred_mosaic[j].to(device), targets[j].to(device), mask_j
                )
                results_mosaic.append({'y_coord': y, 'x_coord': x, **metrics_mosaic_sample})
            
            if sample_count in vis_indices:
                lc_path = os.path.join(vis_dir, f"lc_sample_{sample_count:04d}_y{y}_x{x}.png")
                mosaic_path = os.path.join(vis_dir, f"mosaic_sample_{sample_count:04d}_y{y}_x{x}.png")
                
                save_visualization(lc_path, inputs[j], targets[j], pred_lc[j], "Least Cloudy")
                save_visualization(mosaic_path, inputs[j], targets[j], pred_mosaic[j], "Mosaicking")
                vis_count += 1
            
            sample_count += 1
    
    df_lc = pd.DataFrame(results_lc)
    df_mosaic = pd.DataFrame(results_mosaic)
    
    df_lc.to_csv(os.path.join(args.output_dir, "least_cloudy_results.csv"), index=False)
    df_mosaic.to_csv(os.path.join(args.output_dir, "mosaicking_results.csv"), index=False)
    
    print("\n" + "="*50)
    print("Trivial Baseline Evaluation Complete")
    print(f"Results saved in: {args.output_dir}")
    print(f"Total samples processed: {sample_count}")
    print(f"Total visualizations saved: {vis_count}/{num_vis}")
    print("\nOverall Performance (Least Cloudy):")
    print(df_lc.mean())
    print("\nOverall Performance (Mosaicking):")
    print(df_mosaic.mean())
    print("="*50 + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate trivial baselines for cloud imputation.")
    parser.add_argument('--zarr_path', type=str, required=True,
                        help='Path to test dataset zarr file')
    parser.add_argument('--output_dir', type=str, default='./trivial_baseline_results',
                        help='Directory to save results and visualizations')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    parser.add_argument('--num_vis', type=int, default=20,
                        help='Total number of visualizations to save (randomly sampled from test set)')
    
    args = parser.parse_args()
    main(args)

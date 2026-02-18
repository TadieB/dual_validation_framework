import os
import argparse
import torch
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.getcwd())

from inference_dataloader import CloudGapDataModule
from baseline_3d_unet import UNet3D
from prithvi_cloud_imputation_model import PrithviCloudImputer
from inference_metric3 import ImputationMetrics, save_visualization

def main(args):
   
    random.seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} for model: {args.model_type}")

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

    if 'unet' in args.model_type:
        model = UNet3D(in_channels=6, out_channels=6, base_features=32)
    elif 'prithvi' in args.model_type:
        model = PrithviCloudImputer(
            backbone_weights_path=args.prithvi_weights,
            num_frames=4, in_chans=6, output_bands=6
        )
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    print(f"Loading checkpoint from: {args.checkpoint_path}")
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    total_samples = len(datamodule.test_dataset)
    num_vis = min(args.num_vis, total_samples)
    vis_indices = sorted(random.sample(range(total_samples), num_vis))
    print(f"\nWill save {num_vis} visualizations at sample indices:")
    print(f"{vis_indices[:10]}{'...' if len(vis_indices) > 10 else ''}\n")

    results_list = []
    sample_count = 0
    vis_count = 0
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc=f"Inferencing {args.model_type}")):
            if batch is None: 
                continue

            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            valid_masks = batch['valid_mask'].to(device)
            y_coords = batch['y_coord']
            x_coords = batch['x_coord']

            outputs = model(inputs)

            for j in range(inputs.shape[0]):
                metrics = ImputationMetrics.compute_all(outputs[j], targets[j], valid_masks[j])
                
                result = {
                    'y_coord': y_coords[j].item(),
                    'x_coord': x_coords[j].item(),
                    'rmse': metrics['rmse'],
                    'psnr': metrics['psnr'],
                    'ssim': metrics['ssim'],
                    'sam': metrics['sam']
                }
                results_list.append(result)

                if sample_count in vis_indices:
                    y, x = y_coords[j].item(), x_coords[j].item()
                    save_path = os.path.join(vis_dir, f"sample_{sample_count:04d}_y{y}_x{x}.png")
                    save_visualization(save_path, inputs[j], targets[j], outputs[j], args.model_type)
                    vis_count += 1

                sample_count += 1

    df = pd.DataFrame(results_list)
    results_csv_path = os.path.join(args.output_dir, "quantitative_results.csv")
    df.to_csv(results_csv_path, index=False)
    
    print("\n" + "="*50)
    print(f"Inference complete for {args.model_type}")
    print(f"Quantitative results saved to: {results_csv_path}")
    print(f"Visualizations saved in: {vis_dir}")
    print(f"Total samples processed: {sample_count}")
    print(f"Total visualizations saved: {vis_count}/{num_vis}")
    print("\nOverall Performance:")
    print(df.mean())
    print("="*50 + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference and evaluation on trained models.")
    
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, required=True, 
                        choices=['unet', 'prithvi_zeroshot', 'prithvi_frozen', 'prithvi_partial'],
                        help='Type of model to evaluate')
    parser.add_argument('--zarr_path', type=str, required=True,
                        help='Path to test dataset zarr file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save results and visualizations')
    
    parser.add_argument('--prithvi_weights', type=str, default=None,
                        help='Path to Prithvi backbone weights (required for Prithvi models)')
    
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    parser.add_argument('--num_vis', type=int, default=20,
                        help='Total number of visualizations to save (randomly sampled from test set)')
    
    args = parser.parse_args()
    
    if 'prithvi' in args.model_type and not args.prithvi_weights:
        raise ValueError("--prithvi_weights is required for Prithvi models.")
        
    main(args)

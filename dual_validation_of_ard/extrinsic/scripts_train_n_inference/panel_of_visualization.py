import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

import random

sys.path.insert(0, os.getcwd())
from unified_cloud_imputation_dataloader_syn import CloudGapDataModule
from baseline_3d_unet_gn import UNet3D
from prithvi_cloud_imputation_model_bcast import PrithviCloudImputer

def save_visualization(output_dir, input_seq, target, prediction, model_name, y, x, difficulty_stratum):
    os.makedirs(output_dir, exist_ok=True)
    
    input_vis_seq = input_seq.cpu().numpy()
    target_vis = target.cpu().numpy()
    pred_vis = prediction.cpu().numpy()

    def to_rgb(img):
        return np.transpose(img, (1, 2, 0))[:, :, [0, 3, 2]]

    t1, t2, t3, t4 = [to_rgb(input_vis_seq[i]) for i in range(4)]
    gt, pred = to_rgb(target_vis), to_rgb(pred_vis)
    
    # 1. Calculate stretching percentiles ONLY from Ground Truth (clear sky reference)
    p2_ref, p98_ref = np.percentile(gt.flatten(), [2, 98])
    
    def apply_stretch(img, p2, p98):
        return np.clip((img - p2) / (p98 - p2 + 1e-8), 0, 1) if p98 > p2 else img

    base_name = f"diff_{difficulty_stratum}_y{y}_x{x}"
    safe_model = model_name.replace(' ', '_').replace('(', '').replace(')', '')
    
    # 2. Apply the same reference stretch to EVERYTHING for visual consistency
    plt.imsave(os.path.join(output_dir, f"{base_name}_Input_T1.png"), apply_stretch(t1, p2_ref, p98_ref))
    plt.imsave(os.path.join(output_dir, f"{base_name}_Input_T2.png"), apply_stretch(t2, p2_ref, p98_ref))
    plt.imsave(os.path.join(output_dir, f"{base_name}_Input_T3.png"), apply_stretch(t3, p2_ref, p98_ref))
    plt.imsave(os.path.join(output_dir, f"{base_name}_Input_T4.png"), apply_stretch(t4, p2_ref, p98_ref))
    plt.imsave(os.path.join(output_dir, f"{base_name}_GT.png"), apply_stretch(gt, p2_ref, p98_ref))
    plt.imsave(os.path.join(output_dir, f"{base_name}_Pred_{safe_model}.png"), apply_stretch(pred, p2_ref, p98_ref))

def get_temporal_median(inputs):
    return torch.median(inputs, dim=2).values

def main(args):
    # Mandatory block for identical results across every run
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    # Optional but recommended for extreme precision
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Generating PPTX Visualizations for: {args.model_type}")
    
    vis_dir = os.path.join(args.output_dir, "pptx_visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    df_diff = pd.read_csv(args.difficulty_csv)
    diff_map = df_diff.set_index(['y_coord', 'x_coord'])['stratum'].to_dict()
    
    datamodule = CloudGapDataModule(zarr_path=args.zarr_path, batch_size=4, num_workers=4, difficulty_csv_path=args.difficulty_csv)
    datamodule.setup()
    test_loader = datamodule.test_dataloader()

    if args.model_type == 'mosaicking':
        model, pretty_name = None, "Mosaicking"
    elif args.model_type == 'unet':
        model = UNet3D(in_channels=6, out_channels=6, base_features=32).to(device)
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
        model.eval()
        pretty_name = "3D_U-Net"
    elif args.model_type in ['prithvi_frozen', 'prithvi_partial']:
        model = PrithviCloudImputer(backbone_weights_path=args.prithvi_weights).to(device)
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
        model.eval()
        pretty_name = 'Prithvi_Frozen' if args.model_type == 'prithvi_frozen' else 'Prithvi_Full_FT'

    samples_collected = {'Low': 0, 'Medium': 0, 'High': 0}
    target_samples = args.samples_per_stratum

    with torch.no_grad():
        for batch in test_loader:
            if batch is None: continue
            inputs, targets = batch['input'].to(device), batch['target'].to(device)
            y_coords, x_coords = batch['y_coord'], batch['x_coord']
            
            outputs = get_temporal_median(inputs) if model is None else model(inputs)
            
            for i in range(inputs.size(0)):
                y, x = y_coords[i].item(), x_coords[i].item()
                stratum = diff_map.get((y, x), "unknown")
                
                if stratum in samples_collected and samples_collected[stratum] < target_samples:
                    save_visualization(vis_dir, inputs[i], targets[i], outputs[i], pretty_name, y, x, stratum)
                    samples_collected[stratum] += 1
            
            if all(count >= target_samples for count in samples_collected.values()):
                print(f"✅ Collected {target_samples} patches for all strata!")
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True, choices=['mosaicking', 'unet', 'prithvi_frozen', 'prithvi_partial'])
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--zarr_path', type=str, required=True)
    parser.add_argument('--difficulty_csv', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--prithvi_weights', type=str, default=None)
    parser.add_argument('--samples_per_stratum', type=int, default=10)
    main(parser.parse_args())

import os
import sys
import argparse
import random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.metrics import structural_similarity as skimage_ssim
import yprov4ml as prov4ml 

sys.path.insert(0, os.getcwd())

from unified_cloud_imputation_dataloader_syn import CloudGapDataModule # adding realistic cloud mask to clearest input
from baseline_3d_unet_gn import UNet3D
from prithvi_cloud_imputation_model_bcast import PrithviCloudImputer

# ==========================================
# VISUALIZATION & METRICS (Restored to 3D)
# ==========================================
def save_visualization(output_path, input_seq, target, prediction, model_name):
    input_vis_seq = input_seq.cpu().numpy()
    target_vis = target.cpu().numpy()
    pred_vis = prediction.cpu().numpy()

    def to_rgb(img):
        return np.transpose(img, (1, 2, 0))[:, :, [0, 3, 2]]

    t1_rgb, t2_rgb, t3_rgb, t4_rgb = [to_rgb(input_vis_seq[i]) for i in range(4)]
    target_rgb = to_rgb(target_vis)
    pred_rgb = to_rgb(pred_vis)
    
    input_pixels = np.concatenate([t1_rgb.flatten(), t2_rgb.flatten(), t3_rgb.flatten(), t4_rgb.flatten()])
    p2_in, p98_in = np.percentile(input_pixels, [2, 98])
    
    output_pixels = np.concatenate([target_rgb.flatten(), pred_rgb.flatten()])
    p2_out, p98_out = np.percentile(output_pixels, [2, 98])
    
    def apply_stretch(img_rgb, p2, p98):
        if p98 > p2: img_stretched = (img_rgb - p2) / (p98 - p2)
        else: img_stretched = img_rgb
        return np.clip(img_stretched, 0, 1)

    t1_stretched = apply_stretch(t1_rgb, p2_in, p98_in)
    t2_stretched = apply_stretch(t2_rgb, p2_in, p98_in)
    t3_stretched = apply_stretch(t3_rgb, p2_in, p98_in)
    t4_stretched = apply_stretch(t4_rgb, p2_in, p98_in)
    target_stretched = apply_stretch(target_rgb, p2_out, p98_out)
    pred_stretched = apply_stretch(pred_rgb, p2_out, p98_out)
   
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f"Model: {model_name} | Sample: {os.path.basename(output_path)}", fontsize=16)

    axes[0, 0].imshow(t1_stretched); axes[0, 0].set_title("Input T1", fontsize=14); axes[0, 0].axis('off')
    axes[0, 1].imshow(t2_stretched); axes[0, 1].set_title("Input T2", fontsize=14); axes[0, 1].axis('off')
    axes[0, 2].imshow(t3_stretched); axes[0, 2].set_title("Input T3", fontsize=14); axes[0, 2].axis('off')
    axes[0, 3].imshow(t4_stretched); axes[0, 3].set_title("Input T4", fontsize=14); axes[0, 3].axis('off')
    axes[1, 0].imshow(target_stretched); axes[1, 0].set_title("Ground Truth", fontsize=14, fontweight='bold'); axes[1, 0].axis('off')
    axes[1, 1].imshow(pred_stretched); axes[1, 1].set_title("Model Prediction", fontsize=14, fontweight='bold'); axes[1, 1].axis('off')
    axes[1, 2].axis('off'); axes[1, 3].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)

class ImputationMetrics:
    @staticmethod
    def rmse(pred, target, mask=None):
        if mask is not None and mask.sum() > 0:
            if mask.ndim < pred.ndim: mask = mask.unsqueeze(0).expand_as(pred)
            return torch.sqrt(torch.mean((pred[mask.bool()] - target[mask.bool()])**2))
        return torch.sqrt(torch.mean((pred - target) ** 2))

    @staticmethod
    def psnr(pred, target, mask=None, data_range=1.0):
        if mask is not None and mask.sum() > 0:
            if mask.ndim < pred.ndim: mask = mask.unsqueeze(0).expand_as(pred)
            mse = torch.mean((pred[mask.bool()] - target[mask.bool()])**2)
        else:
            mse = torch.mean((pred - target) ** 2)
        
        if mse == 0: return torch.tensor(float('inf'), device=pred.device)
        return 20 * torch.log10(data_range / torch.sqrt(mse))

    @staticmethod
    def ssim(pred, target, mask=None, data_range=1.0):
        # Transpose 3D tensor (C, H, W) -> (H, W, C) for skimage
        pred_np = pred.detach().cpu().numpy().transpose(1, 2, 0)
        target_np = target.detach().cpu().numpy().transpose(1, 2, 0)
        try:
            val = skimage_ssim(pred_np, target_np, data_range=data_range, channel_axis=-1, win_size=7)
        except TypeError: 
            val = skimage_ssim(pred_np, target_np, data_range=data_range, multichannel=True, win_size=7)
        return val

    @staticmethod
    def sam(pred, target, mask=None):
        if mask is not None and mask.sum() == 0: return 0.0
        
        C = pred.shape[0]
        # Reshape (C, H, W) -> (C, H*W)
        pred_flat = pred.reshape(C, -1)
        target_flat = target.reshape(C, -1)
        
        if mask is not None:
            if mask.ndim == 3:
                # Collapse mask channels: pixel is valid if all channels are valid
                pixel_mask = mask.all(dim=0)
            else:
                pixel_mask = mask
            
            mask_flat = pixel_mask.reshape(-1).bool()
            pred_flat = pred_flat[:, mask_flat]
            target_flat = target_flat[:, mask_flat]

        if pred_flat.shape[1] == 0: return 0.0

        dot_product = (pred_flat * target_flat).sum(dim=0)
        pred_norm = torch.norm(pred_flat, p=2, dim=0)
        target_norm = torch.norm(target_flat, p=2, dim=0)
        
        cos_angle = dot_product / (pred_norm * target_norm + 1e-8)
        angle = torch.acos(torch.clamp(cos_angle, -1.0, 1.0))
        return torch.mean(angle).item()

    @staticmethod
    def compute_all(pred, target, mask=None, data_range=1.0):
        return {
            'rmse': ImputationMetrics.rmse(pred, target, mask).item(),
            'psnr': ImputationMetrics.psnr(pred, target, mask, data_range).item(),
            'ssim': ImputationMetrics.ssim(pred, target, mask, data_range),
            'sam': ImputationMetrics.sam(pred, target, mask)
        }

# ==========================================
# TRIVIAL BASELINE LOGIC
# ==========================================
def get_least_cloudy_oracle(inputs, targets):
    best_preds = []
    for i in range(inputs.size(0)):
        diff = inputs[i] - targets[i].unsqueeze(1)
        mses = torch.mean(diff**2, dim=(0, 2, 3))
        best_t = torch.argmin(mses)
        best_preds.append(inputs[i, :, best_t, :, :])
    return torch.stack(best_preds)

def get_temporal_median(inputs):
    return torch.median(inputs, dim=2).values

# ==========================================
# EVALUATION ENGINE
# ==========================================
def evaluate_model(model_name, model, dataloader, device, args, vis_dir):
    print(f"\n{'='*50}\nEvaluating: {model_name}\n{'='*50}")
    if model is not None: model.eval()
    
    patch_results = []
    total_batches = len(dataloader)
    vis_batches = random.sample(range(total_batches), min(args.num_vis, total_batches))
    vis_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Testing {model_name}")):
            if batch is None: continue
            
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            masks = batch['valid_mask'].to(device)
            y_coords = batch['y_coord']
            x_coords = batch['x_coord']
            
            if model_name == "Mosaicking":
                outputs = get_temporal_median(inputs)
            elif model_name == "Prithvi Zero-Shot Oracle":
                outputs = get_least_cloudy_oracle(inputs, targets)
            else:
                outputs = model(inputs)
            
            for i in range(inputs.size(0)):
                y = y_coords[i].item()
                x = x_coords[i].item()
                
                # NO UNSQUEEZE - Pass 3D tensors (C, H, W) matching the original paper
                pred_i = outputs[i]
                target_i = targets[i]
                mask_i = masks[i]
                
                metrics = ImputationMetrics.compute_all(pred_i, target_i, mask_i)
                
                patch_results.append({
                    'Model': model_name,
                    'y_coord': y,
                    'x_coord': x,
                    'rmse': metrics['rmse'],
                    'psnr': metrics['psnr'],
                    'ssim': metrics['ssim'],
                    'sam': metrics['sam']
                })
                    
            if batch_idx in vis_batches and vis_count < args.num_vis:
                sample_idx = 0
                y, x = y_coords[sample_idx].item(), x_coords[sample_idx].item()
                out_path = os.path.join(vis_dir, f"{model_name.replace(' ', '_')}_y{y}_x{x}.png")
                
                save_visualization(out_path, inputs[sample_idx], targets[sample_idx], outputs[sample_idx], model_name)
                vis_count += 1
                
    return pd.DataFrame(patch_results)

def generate_stratified_tables(df_results, difficulty_csv, output_dir, model_name):
    print(f"\n--- Generating Stratified Tables for {model_name} ---")
    df_diff = pd.read_csv(difficulty_csv)
    df_merged = pd.merge(df_results, df_diff, on=['y_coord', 'x_coord'], how='inner')
    
    safe_model_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
    df_merged.to_csv(os.path.join(output_dir, f"{safe_model_name}_detailed_patch_metrics.csv"), index=False)
    
    # --- 1. OVERALL PERFORMANCE OVERRIDE ---
    overall_summary = df_merged[['psnr', 'ssim', 'sam', 'rmse']].mean().to_frame().T
    overall_summary.insert(0, 'Model', model_name)
    
    # FIX: Recalculate global PSNR from mean RMSE
    avg_rmse = overall_summary.loc[0, 'rmse']
    if avg_rmse > 0:
        overall_summary.loc[0, 'psnr'] = 20 * np.log10(1.0 / avg_rmse)
    else:
        overall_summary.loc[0, 'psnr'] = float('inf')
        
    print("\nTable: Overall Performance")
    print(overall_summary.to_string(index=False))
    overall_summary.to_csv(os.path.join(output_dir, f"{safe_model_name}_overall_performance.csv"), index=False)

    strata_columns = {
        'Overall Difficulty': 'stratum',
        'Spatial Heterogeneity': 'spatial_stratum',
        'Phenology Variability': 'phenology_stratum',
        'Cloud Persistence': 'cloud_stratum'
    }
    
    # --- 2. STRATIFIED PERFORMANCE OVERRIDE ---
    for table_title, col_name in strata_columns.items():
        if col_name in df_merged.columns:
            # Added observed=False to silence pandas deprecation warnings
            summary = df_merged.groupby(col_name, observed=False)[['psnr', 'ssim', 'sam', 'rmse']].mean().reset_index()
            summary.insert(1, 'Model', model_name)
            summary.rename(columns={col_name: 'Stratum'}, inplace=True)
            
            # FIX: Recalculate PSNR for every stratum row
            for idx in summary.index:
                avg_rmse_strat = summary.loc[idx, 'rmse']
                if avg_rmse_strat > 0:
                    summary.loc[idx, 'psnr'] = 20 * np.log10(1.0 / avg_rmse_strat)
                else:
                    summary.loc[idx, 'psnr'] = float('inf')
            
            print(f"\nTable: {table_title}")
            print(summary.to_string(index=False))
            out_name = table_title.replace(" ", "_").lower()
            summary.to_csv(os.path.join(output_dir, f"{safe_model_name}_{out_name}.csv"), index=False)

# ==========================================
# MAIN PIPELINE WITH PROVENANCE
# ==========================================
def main(args):
    random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} for model: {args.model_type}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    vis_dir = os.path.join(args.output_dir, "visualizations")
    prov_dir = os.path.join(args.output_dir, "prov_logs") 
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(prov_dir, exist_ok=True)
    
    # --- 1. START PROVENANCE ---
    safe_model_name = args.model_type.replace(" ", "_").replace("(", "").replace(")", "")
    prov4ml.start_run(
        prov_user_namespace="cloud_imputation", 
        experiment_name=f"Inference_{safe_model_name}", 
        provenance_save_dir=prov_dir
    )
    
    prov4ml.log_param(key="model_type", value=args.model_type)
    prov4ml.log_param(key="checkpoint_path", value=str(args.checkpoint_path))
    prov4ml.log_param(key="zarr_path", value=args.zarr_path)
    prov4ml.log_param(key="difficulty_csv", value=args.difficulty_csv)
    prov4ml.log_param(key="num_vis", value=args.num_vis)

    # --- 2. SETUP DATA & INFERENCE ---
    datamodule = CloudGapDataModule(
        zarr_path=args.zarr_path,
        batch_size=8,
        num_workers=4,
        difficulty_csv_path=args.difficulty_csv 
    )
    datamodule.setup()
    test_loader = datamodule.test_dataloader()
    
    if args.model_type == 'trivial':
        df_res1 = evaluate_model("Mosaicking", None, test_loader, device, args, vis_dir)
        generate_stratified_tables(df_res1, args.difficulty_csv, args.output_dir, "Mosaicking")
        df_res2 = evaluate_model("Prithvi Zero-Shot Oracle", None, test_loader, device, args, vis_dir)
        generate_stratified_tables(df_res2, args.difficulty_csv, args.output_dir, "Zero-Shot_Oracle")
        
    elif args.model_type == 'unet':
        model = UNet3D(in_channels=6, out_channels=6, base_features=32).to(device)
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
        df_res = evaluate_model("3D U-Net", model, test_loader, device, args, vis_dir)
        generate_stratified_tables(df_res, args.difficulty_csv, args.output_dir, "3D U-Net")
        
    elif 'prithvi' in args.model_type:
        model = PrithviCloudImputer(backbone_weights_path=args.prithvi_weights).to(device)
        
        if args.model_type != 'prithvi_zeroshot' and args.checkpoint_path and os.path.exists(args.checkpoint_path):
            model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
        
        paper_names = {
            'prithvi_zeroshot': 'Prithvi_Zero-Shot',
            'prithvi_frozen': 'Prithvi_Frozen',
            'prithvi_partial': 'Prithvi_Full_FT'
        }
        pretty_name = paper_names.get(args.model_type, args.model_type)
        
        df_res = evaluate_model(pretty_name, model, test_loader, device, args, vis_dir)
        generate_stratified_tables(df_res, args.difficulty_csv, args.output_dir, pretty_name)
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    # --- 3. END PROVENANCE ---
    try:
        overall_csv_path = os.path.join(args.output_dir, f"{safe_model_name}_overall_performance.csv")
        if os.path.exists(overall_csv_path):
            df_overall = pd.read_csv(overall_csv_path)
            prov4ml.log_metric(key="test_rmse", value=df_overall['rmse'].iloc[0], context=prov4ml.Context.TESTING, step=0)
            prov4ml.log_metric(key="test_psnr", value=df_overall['psnr'].iloc[0], context=prov4ml.Context.TESTING, step=0)
            prov4ml.log_metric(key="test_ssim", value=df_overall['ssim'].iloc[0], context=prov4ml.Context.TESTING, step=0)
            
            # THE FIX: Convert absolute path to relative path for the artifact logger
            rel_csv_path = os.path.relpath(overall_csv_path)
            prov4ml.log_artifact(artifact_name=f"{safe_model_name}_results_csv", artifact_path=rel_csv_path, context=prov4ml.Context.TESTING, step=0)

        prov4ml.end_run(create_graph=True, create_svg=True, crate_ro_crate=True)
        print(f"✅ W3C Provenance Graph successfully generated for Inference: {safe_model_name}")
    except Exception as e:
        print(f"❌ PROVENANCE ERROR during finalization: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stratified Inference Engine")
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['trivial', 'unet', 'prithvi_zeroshot', 'prithvi_frozen', 'prithvi_partial'])
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--zarr_path', type=str, required=True)
    parser.add_argument('--difficulty_csv', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--prithvi_weights', type=str, default=None)
    parser.add_argument('--num_vis', type=int, default=20)
    args = parser.parse_args()
    main(args)

    

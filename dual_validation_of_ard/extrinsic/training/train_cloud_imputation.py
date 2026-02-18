import os
import argparse
import json
import torch
torch.autograd.set_detect_anomaly(True)

import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import numpy as np

import sys
sys.path.insert(0, os.getcwd())

from cloud_imputation_dataloader import CloudGapDataModule
from prithvi_cloud_imputation_model import PrithviCloudImputer, MaskedLoss
from baseline_3d_unet import UNet3D, SimpleBaseline3D, ImputationMetrics

# =================================================================================
# SCRIPT UPDATES:
# - RESTORED 3 EXPERIMENTS: The `train_prithvi_experiments` function now correctly
#   runs and saves results for three separate strategies as requested:
#   1. Zero-Shot
#   2. Fully Frozen Backbone
#   3. Partial Fine-Tune (Warmup + Full FT)
# - All stability fixes (gradient clipping, AdamW, lower LR) are kept.
# =================================================================================


def evaluate_model(model, dataloader, criterion, device, metrics_to_compute=['rmse', 'psnr']):
    model.eval()
    total_loss = 0.0
    all_metrics = {key: [] for key in metrics_to_compute}
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            if batch is None: continue
            num_batches += 1
            inputs, targets, valid_mask = batch['input'].to(device), batch['target'].to(device), batch['valid_mask'].to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets, valid_mask)
            total_loss += loss.item()
            
            for i in range(outputs.shape[0]):
                mask_i = valid_mask[i]
                if mask_i.sum() > 0:
                    metrics = ImputationMetrics.compute_all(outputs[i], targets[i], mask_i)
                    for k in metrics_to_compute:
                        all_metrics[k].append(metrics.get(k, 0.0))
    
    results = {'loss': total_loss / num_batches if num_batches > 0 else 0}
    for key, values in all_metrics.items():
        if values:
            results[f'{key}_mean'] = np.mean(values)
            results[f'{key}_std'] = np.std(values)
    
    return results


def train_baseline(datamodule, args, device, output_dir):
    print("\n" + "="*60 + f"\nTRAINING 3D BASELINE: {args.baseline_type}\n" + "="*60)
    os.makedirs(output_dir, exist_ok=True)
    
    model = (UNet3D(in_channels=6, out_channels=6, base_features=32) if args.baseline_type == 'unet'
             else SimpleBaseline3D(in_channels=6, out_channels=6, hidden_dim=64)).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    criterion = MaskedLoss(loss_fn='l1')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    best_loss = float('inf')
    
    for epoch in range(args.baseline_epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(datamodule.train_dataloader(), desc=f"Epoch {epoch+1}/{args.baseline_epochs}")
        for batch in pbar:
            if batch is None: continue
            inputs, targets, valid_mask = batch['input'].to(device), batch['target'].to(device), batch['valid_mask'].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets, valid_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm)
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        val_results = evaluate_model(model, datamodule.val_dataloader(), criterion, device)
        val_loss = val_results.get('loss', float('inf'))
        
        print(f"Epoch {epoch+1}: Train Loss={(train_loss/len(pbar)):.4f}, Val Loss={val_loss:.4f}, Val RMSE={val_results.get('rmse_mean', 0):.4f}")
        
        scheduler.step(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, f'baseline_{args.baseline_type}.pth'))
            print("  → Saved best model")
    
    print("\nFinal validation results for baseline:")
    return evaluate_model(model, datamodule.val_dataloader(), criterion, device)


def train_prithvi_experiments(datamodule, model_config, args, device, output_dir):
    print("\n" + "="*60 + "\nPRITHVI EXPERIMENTS\n" + "="*60)
    results = {}
    os.makedirs(output_dir, exist_ok=True)
    criterion = MaskedLoss(loss_fn='l1')
    val_loader = datamodule.val_dataloader()

    # === Experiment A: Zero-shot ===
    print("\n--- Experiment A: Zero-Shot Evaluation ---\n")
    model = PrithviCloudImputer(**model_config).to(device)
    val_results = evaluate_model(model, val_loader, criterion, device)
    results['prithvi_zeroshot_val'] = val_results
    print(f"Zero-shot - Val Loss: {val_results.get('loss',0):.4f}, RMSE: {val_results.get('rmse_mean',0):.4f}")
    torch.save(model.state_dict(), os.path.join(output_dir, 'prithvi_zeroshot.pth'))
    del model # Free memory

    # === Experiment B: Fully Frozen Backbone ===
    print(f"\n--- Experiment B: Fully Frozen Backbone Training ({args.prithvi_epochs} epochs) ---\n")
    model = PrithviCloudImputer(**model_config).to(device)
    model.freeze_encoder()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    best_loss_frozen = float('inf')
    for epoch in range(args.prithvi_epochs):
        model.train()
        pbar = tqdm(datamodule.train_dataloader(), desc=f"[FROZEN] Epoch {epoch+1}/{args.prithvi_epochs}")
        for batch in pbar:
            if batch is None: continue
            inputs, targets, valid_mask = batch['input'].to(device), batch['target'].to(device), batch['valid_mask'].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets, valid_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm)
            optimizer.step()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        val_results = evaluate_model(model, val_loader, criterion, device)
        val_loss = val_results.get('loss', float('inf'))
        print(f"Epoch {epoch+1}: Val Loss={val_loss:.4f}, RMSE={val_results.get('rmse_mean', 0):.4f}")

        if val_loss < best_loss_frozen:
            best_loss_frozen = val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, 'prithvi_frozen.pth'))
            print("  → Saved best model")
    
    results['prithvi_frozen_val'] = evaluate_model(model, val_loader, criterion, device)
    del model # Free memory

    # === Experiment C: Partial Fine-Tune ===
    print(f"\n--- Experiment C: Partial Fine-tuning ({args.prithvi_epochs} total epochs) ---\n")
    model = PrithviCloudImputer(**model_config).to(device)
    
    # Phase 1: Warmup
    freeze_epochs = max(1, int(args.prithvi_epochs * 0.4))
    print(f"Phase 1: Warming up decoder for {freeze_epochs} epochs...")
    model.freeze_encoder()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    for epoch in range(freeze_epochs):
        model.train()
        pbar = tqdm(datamodule.train_dataloader(), desc=f"[WARMUP] Epoch {epoch+1}/{freeze_epochs}")
        # (Training loop for warmup)
        for batch in pbar:
            if batch is None: continue
            inputs, targets, valid_mask = batch['input'].to(device), batch['target'].to(device), batch['valid_mask'].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets, valid_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm)
            optimizer.step()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Phase 2: Fine-tune
    finetune_epochs = args.prithvi_epochs - freeze_epochs
    print(f"\nPhase 2: Fine-tuning entire model for {finetune_epochs} epochs...")
    model.unfreeze_encoder()
    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': args.learning_rate / 10},
        {'params': model.decoder.parameters(), 'lr': args.learning_rate}
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=finetune_epochs, eta_min=1e-6)
    
    best_loss_partial = float('inf')
    for epoch in range(finetune_epochs):
        model.train()
        pbar = tqdm(datamodule.train_dataloader(), desc=f"[FINETUNE] Epoch {epoch+1}/{finetune_epochs}")
        # (Training loop for fine-tuning)
        for batch in pbar:
            if batch is None: continue
            inputs, targets, valid_mask = batch['input'].to(device), batch['target'].to(device), batch['valid_mask'].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets, valid_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm)
            optimizer.step()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        val_results = evaluate_model(model, val_loader, criterion, device)
        val_loss = val_results.get('loss', float('inf'))
        print(f"Epoch {epoch+1}: Val Loss={val_loss:.4f}, RMSE={val_results.get('rmse_mean', 0):.4f}")
        
        scheduler.step()
        if val_loss < best_loss_partial:
            best_loss_partial = val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, 'prithvi_partial_finetune.pth'))
            print("  → Saved best model")
    
    results['prithvi_partial_finetune_val'] = evaluate_model(model, val_loader, criterion, device)
    return results

def main(args):
    #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available(): print("WARNING: CUDA not available, using CPU.")
    print(f"Using device: {device}")
    
    datamodule = CloudGapDataModule(zarr_path=args.zarr_path, batch_size=args.batch_size, num_workers=args.num_workers)
    datamodule.setup()
    
    all_results = {}
    
    if args.train_baseline:
        all_results['baseline_val'] = train_baseline(datamodule, args, device, os.path.join(args.output_dir, 'baseline'))
    
    if args.train_prithvi:
        prithvi_config = {'backbone_weights_path': args.prithvi_weights, 'num_frames': 4, 'in_chans': 6, 'output_bands': 6, 'decoder_type': 'upernet'}
        prithvi_results = train_prithvi_experiments(datamodule, prithvi_config, args, device, os.path.join(args.output_dir, 'prithvi'))
        all_results.update(prithvi_results)

    print("\n" + "="*60 + "\nFINAL TEST SET EVALUATION\n" + "="*60)
    test_loader = datamodule.test_dataloader()
    criterion = MaskedLoss(loss_fn='l1')
    test_metrics = ['rmse', 'psnr', 'ssim', 'sam']
    
    # 
    
    results_path = os.path.join(args.output_dir, 'final_results.json')
    with open(results_path, 'w') as f: json.dump(all_results, f, indent=4)
    print("\n" + "="*60 + "\nFINAL RESULTS SUMMARY\n" + "="*60)
    print(json.dumps(all_results, indent=4))
    print(f"\nAll results saved to: {results_path}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Train and evaluate cloud imputation models.")
    parser.add_argument('--zarr_path', type=str, required=True, help='Path to the single Zarr data cube.')
    parser.add_argument('--output_dir', type=str, default='./experiments', help='Directory for outputs.')
    parser.add_argument('--train_baseline', action='store_true', help='Train 3D baseline model.')
    parser.add_argument('--train_prithvi', action='store_true', help='Train Prithvi model.')
    parser.add_argument('--prithvi_weights', type=str, help='Path to Prithvi pretrained weights.')
    parser.add_argument('--baseline_epochs', type=int, default=30)
    parser.add_argument('--prithvi_epochs', type=int, default=25, help="Total epochs for BOTH the frozen and partial fine-tune experiments.")
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--clip_grad_norm', type=float, default=1.0)
    parser.add_argument('--baseline_type', type=str, default='unet', choices=['unet', 'simple'])
    args = parser.parse_args()
    if not args.train_baseline and not args.train_prithvi: raise ValueError("Must specify at least --train_baseline or --train_prithvi")
    if args.train_prithvi and not args.prithvi_weights: raise ValueError("--prithvi_weights is required when using --train_prithvi")
    main(args)
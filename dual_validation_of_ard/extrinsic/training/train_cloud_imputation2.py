# part - 2 , Training...
import os
import argparse
import json
import torch
# torch.autograd.set_detect_anomaly(True) # Keep this commented out unless debugging
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import numpy as np

import sys
sys.path.insert(0, os.getcwd())

from cloud_imputation_dataloader import CloudGapDataModule
from prithvi_cloud_imputation_model import PrithviCloudImputer, MaskedLoss
from baseline_3d_unet import UNet3D, SimpleBaseline3D, ImputationMetrics

def evaluate_model(model, dataloader, criterion, device, metrics_to_compute=['rmse', 'psnr']):
    # This function remains the same
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
    # This function remains the same, but will not be called from main()
    print("\n" + "="*60 + f"\nTRAINING 3D BASELINE: {args.baseline_type}\n" + "="*60)
    # ... (function content is unchanged) ...
    # ...
    print("\nFinal validation results for baseline:")
    return evaluate_model(model, datamodule.val_dataloader(), criterion, device)


def train_prithvi_experiments(datamodule, model_config, args, device, output_dir):
    print("\n" + "="*60 + "\nPRITHVI EXPERIMENTS (RESUMING PARTIAL TUNE)\n" + "="*60)
    results = {}
    os.makedirs(output_dir, exist_ok=True)
    criterion = MaskedLoss(loss_fn='l1')
    val_loader = datamodule.val_dataloader()

    # === Experiment A: Zero-shot (Already completed) ===
    # print("\n--- Experiment A: Zero-Shot Evaluation ---\n")
    # model = PrithviCloudImputer(**model_config).to(device)
    # val_results = evaluate_model(model, val_loader, criterion, device)
    # results['prithvi_zeroshot_val'] = val_results
    # print(f"Zero-shot - Val Loss: {val_results.get('loss',0):.4f}, RMSE: {val_results.get('rmse_mean',0):.4f}")
    # torch.save(model.state_dict(), os.path.join(output_dir, 'prithvi_zeroshot.pth'))
    # del model

    # === Experiment B: Fully Frozen Backbone (Already completed) ===
    # print(f"\n--- Experiment B: Fully Frozen Backbone Training ({args.prithvi_epochs} epochs) ---\n")
    # model = PrithviCloudImputer(**model_config).to(device)
    # model.freeze_encoder()
    # optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)
    # ... (training loop for frozen model) ...
    # results['prithvi_frozen_val'] = evaluate_model(model, val_loader, criterion, device)
    # del model

    # === Experiment C: Partial Fine-Tune (This is the one we run in part-2) ===
    print(f"\n--- RESUMING Experiment C: Partial Fine-tuning ({args.prithvi_epochs} total epochs) ---\n")
    model = PrithviCloudImputer(**model_config).to(device)
    
    # Phase 1: Warmup
    freeze_epochs = max(1, int(args.prithvi_epochs * 0.4))
    print(f"Phase 1: Warming up decoder for {freeze_epochs} epochs...")
    model.freeze_encoder()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    for epoch in range(freeze_epochs):
        model.train()
        pbar = tqdm(datamodule.train_dataloader(), desc=f"[WARMUP] Epoch {epoch+1}/{freeze_epochs}")
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available(): print("WARNING: CUDA not available, using CPU.")
    print(f"Using device: {device}")
    
    datamodule = CloudGapDataModule(zarr_path=args.zarr_path, batch_size=args.batch_size, num_workers=args.num_workers)
    datamodule.setup()
    
    all_results = {}
    
    # Comment out the call to train_baseline ===
    # if args.train_baseline:
    #     all_results['baseline_val'] = train_baseline(datamodule, args, device, os.path.join(args.output_dir, 'baseline'))
    
    if args.train_prithvi:
        prithvi_config = {'backbone_weights_path': args.prithvi_weights, 'num_frames': 4, 'in_chans': 6, 'output_bands': 6, 'decoder_type': 'upernet'}
        prithvi_results = train_prithvi_experiments(datamodule, prithvi_config, args, device, os.path.join(args.output_dir, 'prithvi'))
        all_results.update(prithvi_results)

    # === We can comment out the final test evaluation for now to save time ===
    # print("\n" + "="*60 + "\nFINAL TEST SET EVALUATION\n" + "="*60)
    # test_loader = datamodule.test_dataloader()
    # ... (rest of test evaluation logic) ...
    
    results_path = os.path.join(args.output_dir, 'resumed_prithvi_results.json') # Save to a new file
    with open(results_path, 'w') as f: json.dump(all_results, f, indent=4)
    print("\n" + "="*60 + "\nFINAL RESULTS SUMMARY\n" + "="*60)
    print(json.dumps(all_results, indent=4))
    print(f"\nAll results saved to: {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate cloud imputation models.")
    parser.add_argument('--zarr_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./experiments')
    
    # We can disable the baseline argument check for this run ===
    # parser.add_argument('--train_baseline', action='store_true', help='Train 3D baseline model.')
    parser.add_argument('--train_prithvi', action='store_true', help='Train Prithvi model.')
    
    parser.add_argument('--prithvi_weights', type=str)
    # parser.add_argument('--baseline_epochs', type=int, default=30)
    parser.add_argument('--prithvi_epochs', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--clip_grad_norm', type=float, default=1.0)
    # parser.add_argument('--baseline_type', type=str, default='unet', choices=['unet', 'simple'])
    
    args = parser.parse_args()
    
    # We are only running prithvi, so we only need to check for that
    if not args.train_prithvi:
        raise ValueError("Must specify --train_prithvi for this run.")
    if not args.prithvi_weights:
        raise ValueError("--prithvi_weights is required when using --train_prithvi")
    
    main(args)

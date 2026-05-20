import os
import time
import yaml
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from unified_cloud_imputation_dataloader_syn import CloudGapDataset, filter_collate_fn #adding mask to clearest input
from baseline_3d_unet_gn import UNet3D
from prithvi_cloud_imputation_model_bcast import PrithviCloudImputer, MaskedLoss
import yprov4ml as prov4ml

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, int(os.environ["RANK"])

def train_model(model_name, model, train_dataset, val_dataset, config, local_rank, global_rank, test_epochs=None, strategy="standard"):
    #
    rel_output_dir = os.path.relpath(config['output_dir'])
    prov_dir = os.path.join(rel_output_dir, "prov_logs") 
    error_log_path = os.path.join(rel_output_dir, f"{model_name}_runtime_errors.log")

    if global_rank == 0:
        print(f"\n{'='*60}\nStarting DDP Training: {model_name} [{strategy.upper()}]\n{'='*60}")
        
        os.makedirs(prov_dir, exist_ok=True)
        
        prov4ml.start_run(
            prov_user_namespace="cloud_imputation", 
            experiment_name=model_name, 
            provenance_save_dir=prov_dir
        )
        for k, v in config.items():
            prov4ml.log_param(key=str(k), value=str(v))
        prov4ml.log_param(key="strategy", value=strategy)

    model = model.cuda(local_rank)
    
    # DDP
    use_find_unused = (strategy == "partial" or strategy == "frozen")
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=use_find_unused)
    
    # Explicitly set loss to MSE as specified in the manuscript
    criterion = MaskedLoss(loss_fn='mse').cuda(local_rank)
    
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], sampler=val_sampler, 
                            num_workers=config['num_workers'], pin_memory=True, collate_fn=filter_collate_fn)

    # ---------------------------------------------------------
    # 1. ZERO-SHOT EVALUATION
    # ---------------------------------------------------------
    if strategy == "zeroshot":
        model.eval()
        val_loss = torch.zeros(1).cuda(local_rank)
        
        start_eval = time.time() # ADDED: For throughput
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Zero-Shot Eval") if global_rank == 0 else val_loader:
                has_data = torch.tensor([1 if batch is not None else 0], dtype=torch.int, device=local_rank)
                dist.all_reduce(has_data, op=dist.ReduceOp.MIN)
                if has_data.item() == 0: continue
                    
                inputs, targets = batch['input'].cuda(local_rank, non_blocking=True), batch['target'].cuda(local_rank, non_blocking=True)
                valid_mask = batch['valid_mask'].cuda(local_rank, non_blocking=True)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets, valid_mask).mean().item()
                
        dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
        avg_val_loss = (val_loss.item() / dist.get_world_size()) / len(val_loader)

        if global_rank == 0:
            eval_time = time.time() - start_eval # ADDED: Throughput math
            throughput = len(val_dataset) / eval_time
            
            print(f"{model_name} [Zero-Shot] | Val MSE Loss: {avg_val_loss:.4f} | Throughput: {throughput:.2f} s/it")
            try:
                # 1. FIXED: Using official Context Enum
                prov4ml.log_metric(key="val_loss", value=avg_val_loss, context=prov4ml.Context.TESTING, step=0)
                prov4ml.log_metric(key="throughput", value=throughput, context=prov4ml.Context.TESTING, step=0) # ADDED
                
                # 2. FIXED: The true working library parameter 'crate_ro_crate'
                prov4ml.end_run(create_graph=True, create_svg=True, crate_ro_crate=True) 
                print(f" W3C Provenance Graph and RO-Crate successfully generated for {model_name}!")
            except Exception as e: 
                print(f" PROVENANCE ERROR in {model_name}: {e}")
                with open(error_log_path, "a") as f: f.write(f"ZeroShot Prov Error: {str(e)}\n") # ADDED
        return

    # ---------------------------------------------------------
    # 2. STANDARD / FROZEN / PARTIAL TRAINING SETUP
    # ---------------------------------------------------------
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], sampler=train_sampler, 
                              num_workers=config['num_workers'], pin_memory=True, collate_fn=filter_collate_fn)
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                                  lr=config['learning_rate'], weight_decay=config['weight_decay'])

    if test_epochs is not None:
        epochs = test_epochs
        freeze_epochs = 1 if test_epochs > 1 else 0
    else:
        epochs = config['baseline_epochs'] if 'unet' in model_name.lower() else config['prithvi_epochs']
        freeze_epochs = max(1, int(epochs * 0.4)) # Exactly 20 epochs warmup if total is 50

    if strategy in ["standard", "frozen"]:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    else:
        scheduler = None

    baseline_epoch_time = None # ADDED: For speedup math

    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        epoch_start_time = time.time() # ADDED: For throughput math
        
        # --- PARTIAL FINE-TUNE: PHASE 2 UNFREEZE TRIGGER ---
        if strategy == "partial" and epoch == freeze_epochs:
            if global_rank == 0: 
                print(f"\n--- Phase 2: Unfreezing Prithvi Encoder at Epoch {epoch+1} ---")
            
            model.module.unfreeze_encoder()
            
            optimizer = torch.optim.AdamW([
                {'params': model.module.backbone.parameters(), 'lr': config['learning_rate'] / 10},
                {'params': model.module.decoder.parameters(), 'lr': config['learning_rate']}
            ], weight_decay=config['weight_decay'])
            
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(epochs - freeze_epochs), eta_min=1e-6)

        model.train()
        train_loss = torch.zeros(1).cuda(local_rank)
        
        pbar_desc = f"Epoch {epoch+1}/{epochs} [Train Phase 2]" if (strategy=="partial" and epoch>=freeze_epochs) else f"Epoch {epoch+1}/{epochs} [Train]"
        pbar = tqdm(train_loader, desc=pbar_desc) if global_rank == 0 else train_loader
        
        for batch in pbar:
            has_data = torch.tensor([1 if batch is not None else 0], dtype=torch.int, device=local_rank)
            dist.all_reduce(has_data, op=dist.ReduceOp.MIN)
            if has_data.item() == 0: continue 

            inputs, targets = batch['input'].cuda(local_rank, non_blocking=True), batch['target'].cuda(local_rank, non_blocking=True)
            valid_mask = batch['valid_mask'].cuda(local_rank, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets, valid_mask).mean()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad_norm'])
            optimizer.step()
            
            train_loss += loss.item()
            
        dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
        avg_train_loss = (train_loss.item() / dist.get_world_size()) / len(train_loader)
        
        model.eval()
        val_loss = torch.zeros(1).cuda(local_rank)
        with torch.no_grad():
            for batch in val_loader:
                has_data = torch.tensor([1 if batch is not None else 0], dtype=torch.int, device=local_rank)
                dist.all_reduce(has_data, op=dist.ReduceOp.MIN)
                if has_data.item() == 0: continue
                    
                inputs, targets = batch['input'].cuda(local_rank, non_blocking=True), batch['target'].cuda(local_rank, non_blocking=True)
                valid_mask = batch['valid_mask'].cuda(local_rank, non_blocking=True)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets, valid_mask).mean().item()
                
        dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
        avg_val_loss = (val_loss.item() / dist.get_world_size()) / len(val_loader)
        
        if scheduler is not None:
            scheduler.step()

        if global_rank == 0:
            # ADDED: Performance Math
            epoch_duration = time.time() - epoch_start_time
            throughput = len(train_dataset) / epoch_duration
            if baseline_epoch_time is None: baseline_epoch_time = epoch_duration
            speedup = baseline_epoch_time / epoch_duration

            print(f"Epoch {epoch+1} | Train MSE Loss: {avg_train_loss:.4f} | Val MSE Loss: {avg_val_loss:.4f}")
            try:
                # 3. FIXED: Explicit keyword mapping with proper official Context Enums
                prov4ml.log_metric(key="train_loss", value=avg_train_loss, context=prov4ml.Context.TRAINING, step=epoch)
                prov4ml.log_metric(key="val_loss", value=avg_val_loss, context=prov4ml.Context.VALIDATION, step=epoch)
                if scheduler is not None:
                    prov4ml.log_metric(key="learning_rate", value=optimizer.param_groups[0]['lr'], context=prov4ml.Context.TRAINING, step=epoch)
                
                # ADDED: New Metrics
                prov4ml.log_metric(key="throughput", value=throughput, context=prov4ml.Context.TRAINING, step=epoch)
                prov4ml.log_metric(key="speedup", value=speedup, context=prov4ml.Context.TRAINING, step=epoch)
                prov4ml.log_system_metrics(context=prov4ml.Context.TRAINING, step=epoch)
                prov4ml.log_carbon_metrics(context=prov4ml.Context.TRAINING, step=epoch)

            except Exception as e: 
                # ADDED: Proper error logging instead of 'pass'
                with open(error_log_path, "a") as f: f.write(f"Epoch {epoch} Prov Error: {str(e)}\n")
    
    if global_rank == 0:
        # FIXED: Using relative path for the artifact to prevent .zip crash
        save_path = os.path.join(rel_output_dir, f"{model_name}_final.pth")
        torch.save(model.module.state_dict(), save_path)
        print(f"Model weights successfully saved to {save_path}")
        try:
            # 4. FIXED: step=(epochs - 1) handles the off-by-one bug
            prov4ml.log_artifact(
                artifact_name=f"{model_name}_weights",
                artifact_path=save_path, 
                context=prov4ml.Context.TRAINING, 
                step=(epochs - 1)
            )
            
            # 5. FIXED: The true working library parameter 'crate_ro_crate'
            prov4ml.end_run(create_graph=True, create_svg=True, crate_ro_crate=True)
            print(" W3C Provenance Graph and RO-Crate successfully generated!")
        except Exception as e: 
            print(f"PROVENANCE ERROR: {e}") 
            with open(error_log_path, "a") as f: f.write(f"Final Prov Error: {str(e)}\n") # ADDED

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_epochs", type=int, default=None, help="Override YAML epochs for a quick test")
    args = parser.parse_args()

    local_rank, global_rank = setup_ddp()
    
    with open("config_train.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    if global_rank == 0:
        os.makedirs(config['output_dir'], exist_ok=True)
        
    train_dataset = CloudGapDataset(config['zarr_path'], split='train', augment=True)
    val_dataset = CloudGapDataset(config['zarr_path'], split='val', augment=False)
    
    # 1. Train Baseline 3D U-Net
    unet = UNet3D(in_channels=6, out_channels=6, base_features=32)
    train_model("Baseline_3D_UNet", unet, train_dataset, val_dataset, config, local_rank, global_rank, test_epochs=args.test_epochs, strategy="standard")

    # 2. Prithvi Zero-Shot (Inference Only)
    prithvi_zs = PrithviCloudImputer(backbone_weights_path=config['prithvi_weights'])
    train_model("Prithvi_ZeroShot", prithvi_zs, train_dataset, val_dataset, config, local_rank, global_rank, test_epochs=args.test_epochs, strategy="zeroshot")

    # 3. Prithvi Frozen Encoder (Decoder Only)
    prithvi_frozen = PrithviCloudImputer(backbone_weights_path=config['prithvi_weights'])
    prithvi_frozen.freeze_encoder()
    train_model("Prithvi_Frozen", prithvi_frozen, train_dataset, val_dataset, config, local_rank, global_rank, test_epochs=args.test_epochs, strategy="frozen")

    # 4. Prithvi Partial Fine-Tune (Warmup -> Full Finetune)
    prithvi_partial = PrithviCloudImputer(backbone_weights_path=config['prithvi_weights'])
    prithvi_partial.freeze_encoder() # Freeze for the Phase 1 warmup
    train_model("Prithvi_Partial_Finetune", prithvi_partial, train_dataset, val_dataset, config, local_rank, global_rank, test_epochs=args.test_epochs, strategy="partial")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()

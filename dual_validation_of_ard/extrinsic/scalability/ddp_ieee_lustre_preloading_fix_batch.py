#!/usr/bin/env python3
"""
IEEE-Style Scaling Benchmark (DDP)
Preserves proven SLURM/DDP config and common logic.
Only changes: robust padding/trimming of batches and safe dummy-target creation.
"""

import os, sys, time, datetime, argparse, warnings, torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np

# Stability Blockade (prevent rasterio import issues)
class MockRasterio:
    def __getattr__(self, name): raise ImportError("blocked")
sys.modules['rasterio'] = MockRasterio()
os.environ["XARRAY_BACKENDS_SERIES"] = "zarr"

from prithvi_cloud_imputation_model_bcast import PrithviCloudImputer
from baseline_3d_unet import UNet3D
from cloud_imputation_dataloader_original import CloudGapDataset, filter_collate_fn

def setup():
    """Slurm-native DDP setup (keeps proven working choices)."""
    rank = int(os.environ["SLURM_PROCID"])
    local_rank = int(os.environ["SLURM_LOCALID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['NCCL_SOCKET_FAMILY'] = 'AF_INET'
    
    # Proven working config: launcher/gpu-bind makes assigned GPU appear as cuda:0
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    
    timeout = datetime.timedelta(hours=2)
    dist.init_process_group(backend='nccl', init_method='env://', timeout=timeout)
    
    if rank == 0:
        print(f"✓ Initialized: {world_size} GPUs across {max(1, world_size//8)} nodes", flush=True)
    
    # Lightweight sanity log: what GPU does this process see?
    try:
        print(f"[rank {rank}] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}, "
              f"torch.cuda.current_device()={torch.cuda.current_device()}", flush=True)
    except Exception:
        pass

    return rank, local_rank, world_size, device

def _make_target_from_input(input_tensor):
    """
    Create a sensible target from input when target is missing.
    - If input is (B, T, C, H, W) -> return last time step as (B, 1, C, H, W)
    - If input is (B, C, H, W)   -> return unsqueezed (B, 1, C, H, W)
    - If input is (B, T*C, H, W) -> attempt to extract last time's channels (heuristic)
    Adjust as needed if your dataloader uses a different layout.
    """
    if input_tensor.dim() == 5:
        # (B, T, C, H, W) -> keep last frame with time dim = 1
        return input_tensor[:, -1:, ...].clone()
    elif input_tensor.dim() == 4:
        B, C_or_TC, H, W = input_tensor.shape
        BAND_COUNT = 6  # your spectral bands per time-frame
        if C_or_TC % BAND_COUNT == 0 and (C_or_TC // BAND_COUNT) > 1:
            # treat as (B, T*C, H, W)
            T = C_or_TC // BAND_COUNT
            last_channels = input_tensor[:, (T-1)*BAND_COUNT : T*BAND_COUNT, :, :]
            # return with a time dim for consistency: (B, 1, C, H, W)
            return last_channels.unsqueeze(1).clone()
        else:
            # treat as (B, C, H, W) -> (B, 1, C, H, W)
            return input_tensor.unsqueeze(1).clone()
    else:
        raise ValueError(f"Unexpected input tensor shape: {tuple(input_tensor.shape)}")

def pad_batch_to_target_size(batch, target_size, device):
    """
    Ensure batch['input'] and batch['target'] are aligned and have batch dimension == target_size.
    - Pads by repeating complete (input, target) samples when smaller.
    - Trims when larger.
    - If 'target' missing, create from input (last time step).
    - Always returns tensors moved to `device`.
    """
    if 'input' not in batch:
        raise ValueError("batch missing 'input' key")
    input_tensor = batch['input']
    if not torch.is_tensor(input_tensor):
        raise ValueError("'input' must be a torch tensor")
    current_size = input_tensor.size(0)
    if current_size == 0:
        raise ValueError("batch has zero samples")
    
    # Determine or build target tensor (do not blindly clone full input)
    target_tensor = batch.get('target', None)
    if target_tensor is None:
        target_tensor = _make_target_from_input(input_tensor)
    
    # Trim case
    if current_size >= target_size:
        in_tensor = input_tensor[:target_size].to(device, non_blocking=True)
        tgt_tensor = target_tensor[:target_size].to(device, non_blocking=True)
        return {'input': in_tensor, 'target': tgt_tensor}
    
    # Pad case: repeat the first samples (or repeat up to needed count)
    need = target_size - current_size
    repeats = []
    repeats_tgt = []

    if need <= current_size:
        pad_input = input_tensor[:need]
        pad_target = target_tensor[:need]
    else:
        # need > current_size: tile as many times as necessary then slice
        times = (need // current_size) + 1
        pad_input = input_tensor.repeat(times, *([1] * (input_tensor.dim()-1)))[:need]
        pad_target = target_tensor.repeat(times, *([1] * (target_tensor.dim()-1)))[:need]
    in_tensor = torch.cat([input_tensor, pad_input], dim=0).to(device, non_blocking=True)
    tgt_tensor = torch.cat([target_tensor, pad_target], dim=0).to(device, non_blocking=True)
    return {'input': in_tensor, 'target': tgt_tensor}

def preload_data_batches(dataloader, num_batches, batch_size_per_gpu, device, rank, world_size):
    """
    Pre-load batches into GPU memory (NOT timed).
    Each returned batch is guaranteed to have batch_size_per_gpu samples.
    """
    if rank == 0:
        print(f"\n📦 PRE-LOADING {num_batches} batches to GPU memory...", flush=True)
        print(f"   (This I/O is NOT part of benchmark timing)", flush=True)
    
    batches = []
    data_iter = iter(dataloader)
    load_start = time.time()
    attempts = 0
    max_attempts = num_batches * 5  # Try up to 5x
    
    while len(batches) < num_batches and attempts < max_attempts:
        try:
            batch = next(data_iter)
            if batch is not None:
                gpu_batch = pad_batch_to_target_size(batch, batch_size_per_gpu, device)
                batches.append(gpu_batch)
            attempts += 1
        except StopIteration:
            data_iter = iter(dataloader)
            attempts += 1
    
    torch.cuda.synchronize()
    dist.barrier()
    load_time = time.time() - load_start
    
    batches_loaded = torch.tensor([len(batches)], device=device, dtype=torch.int32)
    dist.all_reduce(batches_loaded, op=dist.ReduceOp.MIN)
    min_batches = batches_loaded.item()
    
    if rank == 0:
        print(f"✅ Pre-loaded {len(batches)} batches in {load_time:.1f}s", flush=True)
        if len(batches) > 0:
            print(f"   Batch shape (input): {tuple(batches[0]['input'].shape)}", flush=True)
            print(f"   Batch shape (target): {tuple(batches[0]['target'].shape)}", flush=True)
        if min_batches < num_batches:
            print(f"   ⚠️  Warning: Minimum across ranks is {min_batches}/{num_batches} batches", flush=True)
            print(f"   ℹ️  This is OK - batches will cycle during benchmark", flush=True)
    
    if len(batches) == 0:
        raise RuntimeError(f"Rank {rank}: Failed to load any batches!")
    return batches

def precompile_kernels(model, device, rank, local_rank, sample_batch):
    """
    Compile kernels (NOT timed).
    """
    node_id = rank // 8
    if local_rank == 0:
        print(f"[Node {node_id}] Compiling kernels...", flush=True)
    model.train()
    dummy_input = sample_batch['input']
    with torch.no_grad():
        for _ in range(2):
            _ = model(dummy_input)
    dummy_target = torch.randn_like(model(dummy_input), device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for _ in range(3):
        optimizer.zero_grad()
        out = model(dummy_input)
        torch.nn.functional.mse_loss(out, dummy_target).backward()
        optimizer.step()
    torch.cuda.synchronize()
    dist.barrier()
    if rank == 0:
        print("✅ All nodes compiled\n", flush=True)

def benchmark_ieee_style(model, preloaded_batches, device, rank, local_rank, num_steps=15, num_repeats=5):
    """
    IEEE-style benchmark (timed runs; excludes compilation).
    """
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    if rank == 0:
        print(f"{'='*60}")
        print(f"🔬 BENCHMARK: Measuring Compute Performance")
        print(f"{'='*60}")
        print(f"Method:  IEEE Paper (pure compute)")
        print(f"Steps:   {num_steps} per run")
        print(f"Repeats: {num_repeats}")
        print(f"{'='*60}\n")
    precompile_kernels(model, device, rank, local_rank, preloaded_batches[0])
    if rank == 0:
        print(f"🔥 Burn-in run (compilation, NOT counted)...", flush=True)
    for step in range(num_steps):
        batch = preloaded_batches[step % len(preloaded_batches)]
        optimizer.zero_grad()
        out = model(batch['input'])
        loss = torch.nn.functional.mse_loss(out, batch['target'])
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    dist.barrier()
    if rank == 0:
        print(f"✅ Burn-in complete - starting measured runs\n", flush=True)
    all_throughputs = []
    for repeat in range(num_repeats):
        if rank == 0:
            print(f"🔄 Run {repeat+1}/{num_repeats}...", flush=True)
        total_samples = 0
        for i in range(2):  # small warmup
            batch = preloaded_batches[i % len(preloaded_batches)]
            optimizer.zero_grad()
            out = model(batch['input'])
            loss = torch.nn.functional.mse_loss(out, batch['target'])
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        dist.barrier()
        start_time = time.time()
        for step in range(num_steps):
            batch = preloaded_batches[step % len(preloaded_batches)]
            optimizer.zero_grad()
            out = model(batch['input'])
            loss = torch.nn.functional.mse_loss(out, batch['target'])
            loss.backward()
            optimizer.step()
            total_samples += batch['input'].size(0)
        torch.cuda.synchronize()
        elapsed = time.time() - start_time
        samples_tensor = torch.tensor([total_samples], device=device, dtype=torch.float32)
        time_tensor = torch.tensor([elapsed], device=device, dtype=torch.float32)
        dist.all_reduce(samples_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(time_tensor, op=dist.ReduceOp.MAX)
        throughput = samples_tensor.item() / time_tensor.item()
        all_throughputs.append(throughput)
        if rank == 0:
            print(f"  Run {repeat+1}: {throughput:.2f} samples/sec ({elapsed:.2f}s)", flush=True)
    avg_throughput = np.mean(all_throughputs)
    std_throughput = np.std(all_throughputs)
    if rank == 0:
        print(f"\n✅ Average: {avg_throughput:.2f} ± {std_throughput:.2f} samples/sec\n", flush=True)
    return avg_throughput, std_throughput

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zarr_path', type=str, required=True)
    parser.add_argument('--prithvi_weights', type=str)
    parser.add_argument('--model_type', type=str, choices=['prithvi', 'unet'], required=True)
    parser.add_argument('--batch_size_per_gpu', type=int, default=4,
                        help='Batch size per GPU (use 4 for consistency)')
    parser.add_argument('--output_csv', type=str, default='ieee_results.csv')
    parser.add_argument('--num_steps', type=int, default=15,
                        help='Steps per run (IEEE paper: 15)')
    parser.add_argument('--num_repeats', type=int, default=5,
                        help='Number of repeats (IEEE paper: 5)')
    parser.add_argument('--preload_batches', type=int, default=30,
                        help='Batches to pre-load (should be >= num_steps + burn-in)')
    parser.add_argument('--debug', action='store_true', help='Enable extra debug logs')
    args = parser.parse_args()

    rank, local_rank, world_size, device = setup()
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"🔬 IEEE-Style Scaling Benchmark")
        print(f"{'='*60}")
        print(f"Model:      {args.model_type.upper()}")
        print(f"GPUs:       {world_size}")
        print(f"Strategy:   Lustre + Pre-loading")
        print(f"Steps:      {args.num_steps}")
        print(f"Repeats:    {args.num_repeats}")
        print(f"{'='*60}\n")

    meta_dir = "scaling_metadata"
    if rank == 0:
        os.makedirs(meta_dir, exist_ok=True)
    dist.barrier()
    baseline_file = f"{meta_dir}/baseline_{args.model_type}_ieee.txt"

    if rank == 0:
        print(f"🔨 Building model...", flush=True)
    if args.model_type == 'unet':
        model = UNet3D(in_channels=6, out_channels=6, base_features=32).to(device)
    else:
        model = PrithviCloudImputer(
            backbone_weights_path=args.prithvi_weights,
            num_frames=4, in_chans=6, output_bands=6
        ).to(device)

    model = DDP(model, device_ids=[0], find_unused_parameters=False)
    if rank == 0:
        param_count = sum(p.numel() for p in model.parameters())
        print(f"✓ Model: {param_count:,} parameters", flush=True)

    # Data loader (not timed)
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"📦 DATA LOADING (NOT TIMED)")
        print(f"{'='*60}")
    dataset = CloudGapDataset(args.zarr_path, split='train')
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    loader = DataLoader(dataset, batch_size=args.batch_size_per_gpu, sampler=sampler,
                        num_workers=0, collate_fn=filter_collate_fn, pin_memory=False)

    preloaded_batches = preload_data_batches(loader, args.preload_batches, args.batch_size_per_gpu, device, rank, world_size)
    dist.barrier()
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"✅ DATA LOADING COMPLETE")
        print(f"{'='*60}\n")

    try:
        avg_throughput, std_throughput = benchmark_ieee_style(
            model, preloaded_batches, device, rank, local_rank,
            num_steps=args.num_steps, num_repeats=args.num_repeats
        )
    except Exception as e:
        print(f"FATAL [Rank {rank}]: {e}", flush=True)
        import traceback
        traceback.print_exc()
        dist.destroy_process_group()
        raise

    if rank == 0:
        throughput_per_gpu = avg_throughput / world_size
        if world_size == 8:
            with open(baseline_file, 'w') as f:
                f.write(f"{avg_throughput},{std_throughput}")
            baseline_throughput = avg_throughput
            baseline_std = std_throughput
        else:
            if os.path.exists(baseline_file):
                with open(baseline_file) as f:
                    baseline_throughput, baseline_std = map(float, f.read().split(','))
            else:
                baseline_throughput = avg_throughput / (world_size / 8)
                baseline_std = 0.0
        speedup = avg_throughput / baseline_throughput
        efficiency = (speedup / (world_size/8)) * 100
        print(f"\n{'='*60}")
        print(f"📊 FINAL RESULTS")
        print(f"{'='*60}")
        print(f"Model:               {args.model_type}")
        print(f"GPUs:                {world_size}")
        print(f"Throughput (total):  {avg_throughput:.2f} ± {std_throughput:.2f} samples/sec")
        print(f"Throughput per GPU:  {throughput_per_gpu:.2f} samples/sec/GPU")
        print(f"Speedup vs 8 GPUs:   {speedup:.2f}x")
        print(f"Scaling Efficiency:  {efficiency:.1f}%")
        print(f"{'='*60}\n")
        import csv
        write_header = not os.path.exists(args.output_csv)
        with open(args.output_csv, 'a') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(['model', 'gpus', 'throughput_total', 'throughput_per_gpu', 'std', 'speedup', 'efficiency_pct'])
            writer.writerow([args.model_type, world_size, f"{avg_throughput:.2f}", f"{throughput_per_gpu:.2f}", f"{std_throughput:.2f}", f"{speedup:.2f}", f"{efficiency:.1f}"])

    dist.destroy_process_group()

if __name__ == "__main__":
    main()


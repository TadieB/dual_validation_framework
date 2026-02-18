import torch
import numpy as np
from skimage.metrics import structural_similarity as skimage_ssim
import matplotlib.pyplot as plt
import os

def save_visualization(output_path, input_seq, target, prediction, model_name):
    """
    Saves a visual comparison using separate contrast stretches for inputs and outputs
    to ensure clarity for publications.
    """
    input_vis_seq = input_seq.cpu().numpy()
    target_vis = target.cpu().numpy()
    pred_vis = prediction.cpu().numpy()

    def to_rgb(img):
        """Converts to RGB: MODIS bands 1(R), 4(G), 3(B) → indices [0, 3, 2]"""
        img_rgb = np.transpose(img, (1, 2, 0))[:, :, [0, 3, 2]]
        return img_rgb

    # Extract all 4 temporal frames
    t1_rgb = to_rgb(input_vis_seq[0])
    t2_rgb = to_rgb(input_vis_seq[1])
    t3_rgb = to_rgb(input_vis_seq[2])
    t4_rgb = to_rgb(input_vis_seq[3])
    target_rgb = to_rgb(target_vis)
    pred_rgb = to_rgb(pred_vis)
    
    # --- STRETCH 1: Calculate percentiles for INPUTS only ---
    input_pixels = np.concatenate([
        t1_rgb.flatten(), t2_rgb.flatten(), t3_rgb.flatten(), t4_rgb.flatten()
    ])
    p2_in, p98_in = np.percentile(input_pixels, [2, 98])
    
    # --- STRETCH 2: Calculate percentiles for OUTPUTS only ---
    output_pixels = np.concatenate([target_rgb.flatten(), pred_rgb.flatten()])
    p2_out, p98_out = np.percentile(output_pixels, [2, 98])
    
    # --- Apply the separate stretches ---
    def apply_stretch(img_rgb, p2, p98):
        if p98 > p2:
            img_stretched = (img_rgb - p2) / (p98 - p2)
        else:
            img_stretched = img_rgb
        return np.clip(img_stretched, 0, 1)

    # Apply input stretch to input images
    t1_stretched = apply_stretch(t1_rgb, p2_in, p98_in)
    t2_stretched = apply_stretch(t2_rgb, p2_in, p98_in)
    t3_stretched = apply_stretch(t3_rgb, p2_in, p98_in)
    t4_stretched = apply_stretch(t4_rgb, p2_in, p98_in)

    # Apply output stretch to ground truth and prediction
    target_stretched = apply_stretch(target_rgb, p2_out, p98_out)
    pred_stretched = apply_stretch(pred_rgb, p2_out, p98_out)
   
    # Plot with 2 rows: [T1, T2, T3, T4] and [Target, Prediction, (empty)]
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f"Model: {model_name} | Sample: {os.path.basename(output_path)}", fontsize=16)

    # Row 1: All 4 temporal input frames
    axes[0, 0].imshow(t1_stretched)
    axes[0, 0].set_title("Input T1", fontsize=14)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(t2_stretched)
    axes[0, 1].set_title("Input T2", fontsize=14)
    axes[0, 1].axis('off')

    axes[0, 2].imshow(t3_stretched)
    axes[0, 2].set_title("Input T3", fontsize=14)
    axes[0, 2].axis('off')

    axes[0, 3].imshow(t4_stretched)
    axes[0, 3].set_title("Input T4", fontsize=14)
    axes[0, 3].axis('off')

    # Row 2: Ground truth and prediction
    axes[1, 0].imshow(target_stretched)
    axes[1, 0].set_title("Ground Truth", fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(pred_stretched)
    axes[1, 1].set_title("Model Prediction", fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')

    # Hide the remaining subplots in row 2
    axes[1, 2].axis('off')
    axes[1, 3].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


class ImputationMetrics:
    # ... (rest of the class is unchanged) ...
    @staticmethod
    def rmse(pred, target, mask=None):
        if mask is not None:
            if mask.sum() > 0:
                if mask.ndim > pred.ndim - 1: mask = mask.squeeze()
                mask = mask.bool().unsqueeze(0).expand_as(pred)
                return torch.sqrt(torch.mean((pred[mask] - target[mask])**2))
            else:
                return torch.tensor(0.0, device=pred.device)
        return torch.sqrt(torch.mean((pred - target) ** 2))

    @staticmethod
    def psnr(pred, target, mask=None, data_range=1.0):
        if mask is not None:
            if mask.sum() > 0:
                if mask.ndim > pred.ndim - 1: mask = mask.squeeze()
                mask = mask.bool().unsqueeze(0).expand_as(pred)
                mse = torch.mean((pred[mask] - target[mask])**2)
            else:
                mse = torch.tensor(0.0, device=pred.device)
        else:
            mse = torch.mean((pred - target) ** 2)
        
        if mse == 0:
            return torch.tensor(float('inf'), device=pred.device)
        return 20 * torch.log10(data_range / torch.sqrt(mse))

    @staticmethod
    def ssim(pred, target, mask=None, data_range=1.0):
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        
        if pred_np.ndim == 3:
            pred_np = np.transpose(pred_np, (1, 2, 0))
            target_np = np.transpose(target_np, (1, 2, 0))

        try:
            val = skimage_ssim(pred_np, target_np, data_range=data_range, channel_axis=-1, win_size=7)
        except TypeError: 
            val = skimage_ssim(pred_np, target_np, data_range=data_range, multichannel=True, win_size=7)
        return val

    @staticmethod
    def sam(pred, target, mask=None):
        if mask is not None and mask.sum() == 0: return 0.0
        
        pred_flat = pred.reshape(pred.shape[0], -1)
        target_flat = target.reshape(target.shape[0], -1)
        
        if mask is not None:
            mask_flat = mask.flatten().bool()
            pred_flat = pred_flat[:, mask_flat]
            target_flat = target_flat[:, mask_flat]

        if pred_flat.shape[1] == 0: return 0.0

        dot_product = (pred_flat * target_flat).sum(dim=0)
        pred_norm = torch.norm(pred_flat, p=2, dim=0)
        target_norm = torch.norm(target_flat, p=2, dim=0)
        
        cos_angle = dot_product / (pred_norm * target_norm + 1e-8)
        angle = torch.acos(torch.clamp(cos_angle, -1, 1))
        return torch.mean(angle).item()

    @staticmethod
    def compute_all(pred, target, mask=None, data_range=1.0):
        return {
            'rmse': ImputationMetrics.rmse(pred, target, mask).item(),
            'psnr': ImputationMetrics.psnr(pred, target, mask, data_range).item(),
            'ssim': ImputationMetrics.ssim(pred, target, mask, data_range),
            'sam': ImputationMetrics.sam(pred, target, mask),
        }

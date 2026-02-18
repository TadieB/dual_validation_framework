import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as skimage_ssim

class Conv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=False), # Removed bias, often better with BatchNorm
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels=6, out_channels=6, base_features=32):
        super().__init__()
        
        features = base_features
        
        self.encoder1 = Conv3DBlock(in_channels, features)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        self.encoder2 = Conv3DBlock(features, features * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        self.encoder3 = Conv3DBlock(features * 2, features * 4)
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        self.bottleneck = Conv3DBlock(features * 4, features * 8)
        
        self.upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.decoder3 = Conv3DBlock(features * 8, features * 4)
        
        self.upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.decoder2 = Conv3DBlock(features * 4, features * 2)
        
        self.upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.decoder1 = Conv3DBlock(features * 2, features)
        
        # === THE FIX IS HERE ===
        # Replace the unstable Conv3d with a stable pooling operation
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))
        self.final_conv = nn.Conv2d(features, out_channels, kernel_size=1)
    
    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        
        bottleneck = self.bottleneck(self.pool3(enc3))
        
        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        # === AND THE FORWARD PASS FIX IS HERE ===
        # Apply the stable temporal pooling and squeeze the time dimension
        x = self.temporal_pool(dec1).squeeze(2)
        
        return self.final_conv(x)

# --- SimpleBaseline3D and ImputationMetrics remain the same ---

class SimpleBaseline3D(nn.Module):
    def __init__(self, in_channels=6, out_channels=6, hidden_dim=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dim, 3, padding=1), nn.ReLU(),
            nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, None, None)),
        )
        self.final_conv = nn.Conv2d(hidden_dim, out_channels, 1)

    def forward(self, x):
        x = self.model(x).squeeze(2)
        return self.final_conv(x)

class ImputationMetrics:
    @staticmethod
    def rmse(pred, target, mask=None):
        if mask is not None:
            if mask.sum() > 0:
                return torch.sqrt(torch.mean((pred[mask.unsqueeze(0).expand_as(pred)] - target[mask.unsqueeze(0).expand_as(target)])**2))
            else:
                return torch.tensor(0.0)
        return torch.sqrt(torch.mean((pred - target) ** 2))

    @staticmethod
    def psnr(pred, target, mask=None, data_range=1.0):
        if mask is not None:
            if mask.sum() > 0:
                mse = torch.mean((pred[mask.unsqueeze(0).expand_as(pred)] - target[mask.unsqueeze(0).expand_as(target)])**2)
            else:
                mse = 0
        else:
            mse = torch.mean((pred - target) ** 2)
        if mse == 0: return float('inf')
        return 20 * torch.log10(data_range / torch.sqrt(mse))

    @staticmethod
    def ssim(pred, target, mask=None, data_range=1.0):
        # Move to numpy for scikit-image
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        
        # Transpose to (H, W, C) for skimage
        if pred_np.ndim == 3:
            pred_np = np.transpose(pred_np, (1, 2, 0))
            target_np = np.transpose(target_np, (1, 2, 0))

        # SSIM is calculated on the whole image patch, as it relies on local windows
        return skimage_ssim(pred_np, target_np, data_range=data_range, channel_axis=-1, win_size=7)
        
    @staticmethod
    def sam(pred, target, mask=None):
        if mask is not None and mask.sum() == 0: return 0.0
        
        # Flatten spatial dimensions and apply mask
        pred_flat = pred.reshape(pred.shape[0], -1)
        target_flat = target.reshape(target.shape[0], -1)
        
        if mask is not None:
            mask_flat = mask.flatten()
            pred_flat = pred_flat[:, mask_flat]
            target_flat = target_flat[:, mask_flat]

        dot_product = (pred_flat * target_flat).sum(dim=0)
        pred_norm = torch.norm(pred_flat, dim=0)
        target_norm = torch.norm(target_flat, dim=0)
        
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
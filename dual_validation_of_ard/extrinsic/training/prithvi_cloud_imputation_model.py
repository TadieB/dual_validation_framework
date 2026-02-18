import torch
import torch.nn as nn
import torch.nn.functional as F
from prithvi_mae import PrithviViT

class UPerNetDecoder(nn.Module):
    """
    UPerNet-style decoder for dense prediction.
    Uses GroupNorm for stability instead of BatchNorm.
    """
    def __init__(self,
                 embed_dim=1024,
                 num_frames=4,
                 num_classes=6,
                 feature_indices=[5, 11, 17, 23]):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_frames = num_frames

        # Lateral convolutions to process features from the backbone
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, 512, kernel_size=1, bias=False),
                nn.GroupNorm(32, 512), # Using GroupNorm
                nn.ReLU(inplace=True)
            ) for _ in feature_indices
        ])

        # Pyramid Pooling Module (PSP)
        self.psp_modules = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size),
                nn.Conv2d(512, 128, kernel_size=1, bias=False),
                nn.GroupNorm(32, 128), # Using GroupNorm
                nn.ReLU(inplace=True)
            ) for output_size in [1, 2, 3, 6]
        ])

        # Fusion convolution
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(512 + (4 * 128), 512, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, 512), # Using GroupNorm
            nn.ReLU(inplace=True)
        )

        # Final prediction head
        self.final_conv = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, 256), # Using GroupNorm
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(16, 128), # Using GroupNorm
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )

    def _reshape_features(self, features, B, T, H_patch, W_patch):
        patch_tokens = features[:, 1:, :]
        # Reshape to (B, C, H, W), averaging over the time dimension
        return patch_tokens.reshape(B, T, H_patch, W_patch, self.embed_dim).permute(0, 1, 4, 2, 3).mean(dim=1)

    def forward(self, feature_pyramid, input_shape):
        B, _, T, H, W = input_shape
        H_patch, W_patch = 14, 14

        # We use the features from the last transformer block
        features = self._reshape_features(feature_pyramid[-1], B, T, H_patch, W_patch)
        lateral_output = self.lateral_convs[-1](features)

        # Pyramid Pooling
        psp_outs = [lateral_output]
        for psp_module in self.psp_modules:
            pooled = psp_module(lateral_output)
            upsampled = F.interpolate(pooled, size=lateral_output.shape[2:], mode='bilinear', align_corners=False)
            psp_outs.append(upsampled)

        psp_out = torch.cat(psp_outs, dim=1)
        fused = self.fusion_conv(psp_out)
        output = self.final_conv(fused)

        return output

class PrithviCloudImputer(nn.Module):
    def __init__(self, 
                 backbone_weights_path,
                 num_frames=4,
                 in_chans=6,
                 output_bands=6,
                 decoder_type='upernet'):
        super().__init__()
        
        self.backbone = PrithviViT(
            img_size=224, patch_size=(1, 16, 16), num_frames=num_frames,
            in_chans=in_chans, embed_dim=1024, depth=24, num_heads=16,
            mlp_ratio=4.0, norm_layer=nn.LayerNorm
        )
        
        self._load_pretrained_weights(backbone_weights_path, num_frames)
        
        if decoder_type == 'upernet':
            self.decoder = UPerNetDecoder(
                embed_dim=1024, num_frames=num_frames, num_classes=output_bands
            )
        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}")
    
    def _load_pretrained_weights(self, weights_path, num_frames):
        print(f"Loading Prithvi weights from: {weights_path}")
        # Set weights_only=True for security, as recommended by the warning.
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
        
        encoder_state_dict = {}
        for key, value in checkpoint.items():
            if key.startswith('encoder.'):
                new_key = key.replace('encoder.', '', 1)
                encoder_state_dict[new_key] = value
        
        # Handle positional embedding interpolation
        if 'pos_embed' in encoder_state_dict:
            pos_embed_checkpoint = encoder_state_dict['pos_embed']
            expected_num_patches = self.backbone.patch_embed.num_patches
            
            if pos_embed_checkpoint.shape[1] != expected_num_patches + 1:
                print(f"Interpolating positional embedding from {pos_embed_checkpoint.shape[1]-1} to {expected_num_patches} patches.")
                cls_pos_embed = pos_embed_checkpoint[:, 0:1, :]
                patch_pos_embed = pos_embed_checkpoint[:, 1:, :]
                
                # Original shape: (1, 4*14*14, 1024) -> (1, 1024, 4, 14, 14)
                patch_pos_embed = patch_pos_embed.permute(0, 2, 1).reshape(1, 1024, 4, 14, 14)
                
                # Interpolate along the time dimension
                patch_pos_embed = F.interpolate(
                    patch_pos_embed,
                    size=(num_frames, 14, 14),
                    mode='trilinear',
                    align_corners=False
                )
                
                # Reshape back: (1, 1024, T*14*14) -> (1, T*14*14, 1024)
                patch_pos_embed = patch_pos_embed.reshape(1, 1024, -1).permute(0, 2, 1)
                
                encoder_state_dict['pos_embed'] = torch.cat((cls_pos_embed, patch_pos_embed), dim=1)
                
        self.backbone.load_state_dict(encoder_state_dict, strict=False)

    def freeze_encoder(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        feature_pyramid = self.backbone.forward_features(x)
        output = self.decoder(feature_pyramid, x.shape)
        return output


class MaskedLoss(nn.Module):
    def __init__(self, loss_fn='mse'):
        super().__init__()
        if loss_fn == 'mse':
            self.base_loss = nn.MSELoss(reduction='none')
        elif loss_fn == 'l1':
            self.base_loss = nn.L1Loss(reduction='none')
        else:
            raise ValueError(f"Unknown loss: {loss_fn}")
    
    def forward(self, pred, target, valid_mask):
        loss = self.base_loss(pred, target)
        valid_mask = valid_mask.unsqueeze(1)
        masked_loss = loss * valid_mask
        
        num_valid = valid_mask.sum()
        if num_valid > 0:
            return masked_loss.sum() / num_valid
        else:
            return masked_loss.sum()
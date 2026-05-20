import torch
import torch.nn as nn

class Conv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        # Using GroupNorm(8) for DDP stability
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.GroupNorm(8, out_channels), 
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.GroupNorm(8, out_channels),
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
        
        # Stable temporal pooling to crush the 3D sequence into a 2D image
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
        
        # Apply the stable temporal pooling and squeeze the time dimension
        x = self.temporal_pool(dec1).squeeze(2)
        
        return self.final_conv(x)


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
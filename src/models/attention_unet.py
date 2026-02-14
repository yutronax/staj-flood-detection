import torch
import torch.nn as nn
from .modules import ConvBlock, AttentionBlock

# [MODEL_ATTENTION_UNET] Attention U-Net Mimarisi | segmentation, attention, unet
# 6 kanallı girdi (RGB Öncesi + RGB Sonrası) için tasarlanmıştır.
class AttentionUNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=1):
        super(AttentionUNet, self).__init__()

        # [ENCODER] Contracting Path
        self.enc1 = ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc4 = ConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # [BOTTLENECK] En alt katman
        self.bottleneck = ConvBlock(512, 1024)

        # [DECODER] Expansive Path with Attention Gates
        
        # Decoder 4
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.att4 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        self.dec4 = ConvBlock(1024, 512)
        
        # Decoder 3
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att3 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.dec3 = ConvBlock(512, 256)
        
        # Decoder 2
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.dec2 = ConvBlock(256, 128)
        
        # Decoder 1
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att1 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.dec1 = ConvBlock(128, 64)

        # [OUTPUT] Çıktı katmanı
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)
        
        b = self.bottleneck(p4)
        
        # Decoder with Attention
        d4 = self.up4(b)
        x4 = self.att4(g=d4, x=e4) # Attention filtering
        d4 = torch.cat((x4, d4), dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        x3 = self.att3(g=d3, x=e3)
        d3 = torch.cat((x3, d3), dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        x2 = self.att2(g=d2, x=e2)
        d2 = torch.cat((x2, d2), dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        x1 = self.att1(g=d1, x=e1)
        d1 = torch.cat((x1, d1), dim=1)
        d1 = self.dec1(d1)
        
        return self.out_conv(d1)

# [TEST] Model Doğrulama | main, verify, shapes
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # xBD veri seti için 6 kanal (RGB Öncesi + RGB Sonrası)
    model = AttentionUNet(in_channels=6, out_channels=1).to(device)
    
    # Dummy input: Batch=1, Channels=6 (Pre+Post), H=256, W=256
    x = torch.randn(1, 6, 256, 256).to(device)
    y = model(x)
    
    print(f"Model Architecture: Attention U-Net")
    print(f"Input Shape: {x.shape}")
    print(f"Output Shape: {y.shape}")
    
    assert y.shape == (1, 1, 256, 256), "Output shape mismatch!"
    print("Test Passed: Output shape is correct.")

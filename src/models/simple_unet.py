import torch
import torch.nn as nn

# [MODEL_UNET_BASIC] Basit bir U-Net mimarisi | segmentation, cnn, unet, torch
class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(SimpleUNet, self).__init__()
        
        # [ENCODER] Görüntüyü kodlayan kısım (Contracting Path) | conv, relu, maxpool
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        
        # [POOL] Boyut azaltma | max_pool
        self.pool = nn.MaxPool2d(2)
        
        # [DECODER] Görüntüyü yeniden oluşturan kısım (Expansive Path) | upsample, conv
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128) # 256 çünkü skip connection ile birleşiyor
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        
        # [OUTPUT] Çıktı katmanı (maske üretir) | output, conv
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

    # [HELPER] Tekrarlayan konvolüsyon bloğu | helper, conv, batchnorm
    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    # [FORWARD] Verinin model içindeki akışı | forward, pass
    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool(e2)
        
        # Bottom
        e3 = self.enc3(p2)
        
        # Decoder
        d2 = self.up2(e3)
        # [SKIP] Skip connection ile özellikleri birleştirme | cat, skip
        # Boyut uyuşmazlığı olursa crop gerekebilir, burada basitlik için padding=1 varsayıyoruz
        d2 = torch.cat([e2, d2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([e1, d1], dim=1)
        d1 = self.dec1(d1)
        
        return self.out(d1)

# [TEST] Modelin çalıştığını doğrulayan basit test kodu | main, test
if __name__ == "__main__":
    model = SimpleUNet()
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    # Beklenen çıktı: [1, 1, 256, 256]

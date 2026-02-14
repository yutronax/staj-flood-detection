import torch
import torch.nn as nn
import torch.nn.functional as F

# [MODULE_CONV] Standart Çift Konvolüsyon Bloğu | conv, relu, batchnorm
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# [MODULE_ATTENTION] Attention Gate (Dikkat Kapısı) | attention, gate, skip_connection
# Şekil 4.3 referans alınarak oluşturulmuştur.
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        """
        Args:
            F_g: Gating signal channels (Decoder'dan gelen özellik haritası)
            F_l: Low-level feature channels (Encoder'dan gelen özellik haritası)
            F_int: Intermediate channels (Ara katman kanalları)
        """
        super(AttentionBlock, self).__init__()
        
        # [W_g] Gating sinyali için dönüşüm | conv1x1
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        # [W_x] Encoder özellikleri için dönüşüm | conv1x1
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        # [PSI] Aktivasyon ve ağırlık hesaplama | relu, sigmoid, conv1x1
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g: Decoder feature (gating signal)
        # x: Encoder feature (low-level feature)
        
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # [ADD] Özelliklerin birleştirilmesi | element_wise_add
        psi = self.relu(g1 + x1)
        
        # [ATTENTION_MAP] Dikkat haritasının oluşturulması | attention_map
        psi = self.psi(psi)
        
        # [SCALE] Giriş özellikleri ile dikkat haritasının çarpımı | multiply
        return x * psi

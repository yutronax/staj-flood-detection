# Afet Sonrası Hasar ve Nesne Tespiti Projesi - Final Raporu

Bu proje, derin öğrenme (Attention U-Net) kullanarak hava görüntülerinden otomatik afet analizi yapmaktadır. Çalışma iki ana aşamadan oluşmaktadır:

## 1. Aşama: Bina Hasar Tespiti (xBD Veri Seti)
**Hedef:** Afet öncesi ve sonrası görüntüleri karşılaştırarak binalardaki hasarı ikili (pixel-wise binary) olarak tespit etmek.

- **Mimari:** 6 Kanallı Giriş (Pre+Post RGB) -> Attention U-Net.
- **Güçlendirme (v2):** BCE + Dice Loss kombinasyonu ve Veri Artırımı (Augmentation) eklendi.
- **Sonuç:** Bina sınırları yüksek doğrulukla yakalandı.
- **Çıktılar:**
  - Eğitim Grafikleri: `results/Phase1_BuildingDetection/loss_curve_v2_strengthened.png`
  - Tahmin Örnekleri: `results/Phase1_BuildingDetection/prediction_X.png`

## 2. Aşama: Çok Sınıflı Afet Analizi (FloodNet Veri Seti)
**Hedef:** Tek bir görüntüden Sel (Flooding), Araçlar (Vehicles), Hasarlı ve Sağlam Binaları aynı anda tespit etmek.

- **Mimari:** 3 Kanallı RGB Giriş -> Multi-Class Attention U-Net (10 Sınıf).
- **Sınıflar:** 
  - 🔵 Mavi: Sel / Su
  - 🟡 Sarı: Araçlar (Vehicles)
  - 🔴 Kırmızı: Hasarlı Binalar
  - 🟢 Yeşil: Sağlam Binalar
- **Sonuç:** Model, afet bölgesindeki karmaşık nesneleri ve durumları başarıyla ayrıştırdı.
- **Çıktılar:**
  - Eğitim Grafiği: `results/Phase2_MultiClassDisaster/loss_curve_floodnet.png`
  - Tahmin Örnekleri: `results/Phase2_MultiClassDisaster/floodnet_test_X.png`

---
**Geliştiren:** YUSUF ÇİNAR  
**Teknolojiler:** PyTorch, Attention U-Net, OpenCV, Matplotlib.

# Katkıda Bulunma ve Proje Kuralları (Contributing & Rules)

## 🚨 GELİŞTİRİCİ ANAYASASI
1.  **Soru Sorma, İş Yap:** Teyit almak yok. İnisiyatif al ve uygula.
2.  **Önce Kontrol, Sonra Üretim:** Dosya oluşturmadan önce klasör yapısını kontrol et.

## 1. Kodlama Standartları
### Akıllı Yorum Satırları (Smart Comments)
Kodun semantik olarak aranabilirliğini artırmak için aşağıdaki formatı zorunlu kılarız:

**Format:** `# [ETİKET] Açıklama | anahtar, kelimeler`

**Örnekler:**
*   `# [MODEL_UNET] Basit U-Net mimarisi | segmentation, cnn, flood`
*   `# [DATA_LOAD] Görüntü ve maske yükleyici | dataset, dataloader, image`

## 2. Dosya Organizasyonu
*   **Segmentasyon Projesi:**
    *   `src/models`: Model tanımları.
    *   `src/training`: Eğitim döngüleri (train loop).
    *   `src/utils`: Yardımcı fonksiyonlar (metrikler, görselleştirme).

## 3. Otomasyon
*   `scripts/auto_sync.ps1` her saat başı çalışır.

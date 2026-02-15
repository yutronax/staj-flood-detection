# Proje Hafızası: Dinamik Durum (Dynamic State)

## 1. Son Yapılan Değişiklikler (Last Changes)
*   **[2026-02-14]:** GitHub reposu oluşturuldu ve Colab otomasyonu tamamlandı.
*   **[2026-02-15]:** **Eğitim Başlatıldı:** Colab (GPU) üzerinde xBD verisiyle 20 epoch'luk eğitim devam ediyor.
*   **[2026-02-15]:** **Model Güçlendirildi (v2) Tamamlandı:** 
    *   **20 Epoch** eğitim başarıyla bitti (Combined BCE+Dice).
    *   Final Loss: **0.1420**. v1'e göre bina sınırlarını daha keskin yakalayan bir model elde edildi.
*   **[2026-02-15]:** **Analiz:** v2 grafik analizi (`test/plot_loss_v2.py`) tamamlandı.

*   **[2026-02-15]:** **Model Güçlendirildi (v2) Tamamlandı ve Test Edildi:** 
    *   20 Epoch eğitim başarıyla bitti (Combined BCE+Dice).
    *   Final model yerel test edildi ve tahminler `results/v2_predictions` klasörüne kaydedildi.
*   **[2026-02-15]:** **Analiz:** v2 grafik analizi (`test/plot_loss_v2.py`) tamamlandı.
*   **[2026-02-15]:** **FloodNet Multi-Class Training Tamamlandı:** 
    *   **50 Epoch** eğitim başarıyla bitti.
    *   Sınıflar: Sel (Flooded), Araçlar (Vehicles), Binalar, Yollar vb. (10 farklı nesne).
    *   Final Loss: **0.3158** (Train) / **0.3931** (Val).
    *   Model artık hem afet tespiti hem de nesne tespiti yeteneğine sahip.
*   **[2026-02-15]:** **Analiz:** FloodNet özel loss grafiği (`test/plot_loss_floodnet.py`) hazırlandı.

## 2. Aktif Görevler (Current Tasks)
*   [x] FloodNet tabanlı çok sınıflı modelin eğitilmesi.
*   [ ] Final checkpoint (`floodnet_epoch_50.pth`) indirilip yerel test edilmesi.
*   [ ] Araç ve sel tespiti sonuçlarının görselleştirilmesi.
## 3. Proje Durumu
*   **Faz:** Eğitim (Training).
*   **Hedef:** 20. epoch sonunda modeli kaydedip test verileri üzerinde görselleştirme yapmak.

## 4. Bilinen Sorunlar (Known Issues)
*   Windows `pip` Long Path sorunu (`.venv` ile aşıldı).
*   Colab/Kaggle dosya yolu limitleri (Manuel yol tanımlama ile çözüldü).

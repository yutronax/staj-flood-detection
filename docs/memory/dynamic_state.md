# Proje Hafızası: Dinamik Durum (Dynamic State)

## 1. Son Yapılan Değişiklikler (Last Changes)
*   **[2026-02-14]:** GitHub reposu oluşturuldu ve Colab otomasyonu tamamlandı.
*   **[2026-02-15]:** **Eğitim Başlatıldı:** Colab (GPU) üzerinde xBD verisiyle 20 epoch'luk eğitim devam ediyor.
*   **[2026-02-15]:** **Model Güçlendirildi (v2):** 
    *   **Loss:** Sadece BCE yerine **BCE + Dice Loss** (Combined) kullanılmaya başlandı.
    *   **Augmentation:** Eğitim setine **Random Horizontal/Vertical Flip** eklenerek veri çeşitliliği artırıldı.
    *   **Regularization:** Overfitting'e karşı **Weight Decay** (L2) optimizer'a eklendi.
*   **[2026-02-15]:** **Metrikler:** 15. epoch modeli yerel test edildi ve başarılı sonuçlar alındı. 

## 2. Aktif Görevler (Current Tasks)
*   [x] Model güçlendirme (v2) güncellemelerinin GitHub'a gönderilmesi.
*   [ ] Güçlendirilmiş modelin Colab'da tekrar eğitilmesi (Öneri).
## 3. Proje Durumu
*   **Faz:** Eğitim (Training).
*   **Hedef:** 20. epoch sonunda modeli kaydedip test verileri üzerinde görselleştirme yapmak.

## 4. Bilinen Sorunlar (Known Issues)
*   Windows `pip` Long Path sorunu (`.venv` ile aşıldı).
*   Colab/Kaggle dosya yolu limitleri (Manuel yol tanımlama ile çözüldü).

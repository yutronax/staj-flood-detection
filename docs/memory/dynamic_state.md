# Proje Hafızası: Dinamik Durum (Dynamic State)

## 1. Son Yapılan Değişiklikler (Last Changes)
*   **[2026-02-14]:** **xBD (xView2)** veri setine geçiş yapıldı (Segmentation/Maske desteği nedeniyle).
*   **[2026-02-14]:** `AttentionUNet` tekrar 6 kanal (RGB Pre + RGB Post) desteğine çekildi.
*   **[2026-02-14]:** `src/data/xbd_loader.py` ve `src/utils/mask_utils.py` ile JSON'dan maske üretimi entegre edildi.

## 2. Aktif Görevler (Current Tasks)
*   [ ] Eğitim (Train) döngüsünün (`src/train.py`) yazılması.
*   [ ] Modelin xBD verisi üzerinde eğitimi.

## 3. Proje Durumu
*   **Faz:** Hazırlık / Eğitim Başlangıcı.
*   **Hedef:** İlk epoch eğitimini tamamlamak.

## 4. Bilinen Sorunlar (Known Issues)
*   Windows `pip` Long Path sorunu (`.venv` ile aşıldı).
*   xBD test verilerinde etiket yok, validation için train set bölünmeli.

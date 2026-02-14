# Proje Hafızası: Statik Bağlam (Static Context)

Bu dosya, projenin değişmeyen temel kurallarını, mimarisini ve hedeflerini içerir.

## 🚨 KRİTİK SİSTEM KURALLARI (ANAYASA)
1.  **ASLA** "Doğru anladım mı?", "Devam edeyim mi?" gibi teyit soruları sorma. Anladığını özetle ve **İŞLEMİ YAP**. Yanlışsa kullanıcı düzeltir.
2.  **Klasör Kontrolü:** `write_to_file` kullanmadan önce MUTLAKA `list_dir` veya `run_command` ile klasörün varlığını kontrol et. Yoksa `mkdir` ile oluştur.
3.  **Akıllı Yorumlar:** Her dosyada `# [ETİKET] Açıklama | anahtar, kelime` formatını kullan.

## 1. Proje Tanımı
*   **Amaç:** Sel tespiti için görüntü segmentasyonu (Flood Detection Segmentation).
*   **Temel Prensip:** Basitten karmaşığa giden, modüler ve kendi kendini yöneten yapı.

## 2. Hafıza Mimarisi
Proje hafızası iki katmandan oluşur:
1.  **Statik Bağlam (`docs/memory/static_context.md`):** Bu dosya. Kurallar ve mimari.
2.  **Dinamik Durum (`docs/memory/dynamic_state.md`):** Anlık proje durumu, aktif görevler ve son değişiklikler.

## 3. Kodlama ve Dosya Standartları
*   **Dosya Yönetimi:** Tüm kodlar `src/` altında kategorize edilmelidir.
*   **Segmentasyon:** Modeller `src/models/`, veri işleme `src/data/`, eğitim kodları `src/training/` altında olmalıdır.

## 4. Klasör Yapısı
*   `src/models/`: Model mimarileri (U-Net vb.).
*   `src/data/`: Veri yükleme ve ön işleme.
*   `docs/`: Dokümantasyon ve hafıza.
*   `scripts/`: Otomasyon araçları.

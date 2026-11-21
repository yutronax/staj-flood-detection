# Afet Görüntülerinde Segmentasyon

Hava görüntülerinde bina ve afet alanlarının piksel düzeyinde incelenmesi için PyTorch deneyleri. Depo, xBD üzerinde ikili segmentasyon ve FloodNet üzerinde 10 sınıflı segmentasyon çalışmalarını içerir. Kullanılan mimariler arasında Attention U-Net ve SegFormer bulunur.

## Depo içeriği

- `train.py`: xBD eğitim akışı.
- `train_floodnet.py`: FloodNet için Attention U-Net eğitim akışı.
- `train_transformer.py`: transformer tabanlı eğitim denemesi.
- `src/data/`, `src/models/`: veri yükleyicileri ve model kodları.
- `results/`: kayıp eğrileri ve örnek tahmin görselleri.
- [Final raporu](FINAL_REPORT.md): deneylerin kısa özeti.

## Sonuçlar ve sınırlar

Depoda örnek tahminler ve eğitim grafikleri bulunuyor: [xBD örnekleri](results/Phase1_BuildingDetection/) ve [FloodNet örnekleri](results/Phase2_MultiClassDisaster/). Doğrulanabilir sayısal değerlendirme tablosu, veri bölme bilgileri ve eğitilmiş ağırlıklar bu depoda gösterilmiyor; bu nedenle model başarısı hakkında nicel bir iddia yapılmıyor. Sonuç görselleri genelleme performansını tek başına kanıtlamaz.

Bu depodaki veri setleri ve mimariler, CV'de belirtilen Sentinel-2 çalışmasıyla eşleştirilmiş değildir. Bunların aynı araştırma olup olmadığı ayrıca doğrulanmalıdır.

## FloodNet eğitimine başlangıç

Python ortamında `pip install -r requirements.txt` ile bağımlılıkları kurun. Veri setini ayrıca temin edin; `FloodNetDataset`, verilen klasör altında `images/*.jpg` ve karşılık gelen `masks/*.png` dosyalarını bekler. Veri seti kullanım koşullarını ve lisansını sağlayıcısından kontrol edin.

```bash
python train_floodnet.py --data_dir /path/to/floodnet --batch_size 16 --epochs 20
```

Kod veriyi %80 eğitim ve %20 doğrulama olarak ayırır; yalnızca eğitim kümesine veri artırma uygular. Bölme için varsayılan tohum 42'dir; `--seed` ile değiştirilebilir. Bu, bölmeyi sabitler ancak tüm eğitim işleminin deterministik olduğunu garanti etmez. Ağırlıklar her beş epoch'ta `checkpoints_floodnet/` altına kaydedilir. Komutlar bu inceleme sırasında veri seti ve bağımlılıklarla çalıştırılarak doğrulanmamıştır.

## Demo durumu

`streamlit run app.py` ile arayüz başlatılabilir. Otomatik analiz için ilgili model ağırlıkları ayrıca gereklidir; dosya yoksa uygulama sonuç üretmek yerine hata gösterir. SAM, ViLT ve BLIP modelleri ilk kullanımda harici kaynaktan indirilir. Uygulama ve model uyumluluğu bu incelemede uçtan uca doğrulanmamıştır; afet müdahalesinde karar verme amacıyla kullanılmamalıdır.

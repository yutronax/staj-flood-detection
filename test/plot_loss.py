import matplotlib.pyplot as plt

# Paylaşılan loglardan çıkarılan veriler
epochs = list(range(1, 20))
train_losses = [
    0.3852, 0.2766, 0.2283, 0.1924, 0.1661, 0.1444, 0.1297, 0.1177, 0.1064, 
    0.0983, 0.0917, 0.0856, 0.0814, 0.0761, 0.0729, 0.0727, 0.0673, 0.0657, 0.0645
]
val_losses = [
    0.3076, 0.2388, 0.2061, 0.1709, 0.1570, 0.1426, 0.1310, 0.1119, 0.1054, 
    0.0955, 0.0921, 0.0892, 0.0823, 0.0819, 0.0859, 0.0769, 0.0772, 0.0726, 0.0715
]

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_losses, label='Eğitim Kaybı (Train Loss)', marker='o', color='#1f77b4', linewidth=2)
plt.plot(epochs, val_losses, label='Doğrulama Kaybı (Val Loss)', marker='s', color='#ff7f0e', linewidth=2)

plt.title('Attention U-Net: Eğitim Süreci Analizi', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss (Kayıp)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=11)

# Estetik dokunuşlar
plt.xticks(epochs)
plt.style.use('seaborn-v0_8-muted') # Mevcut ise estetik tema

# Kaydet
plt.savefig('test/loss_graph.png', dpi=300, bbox_inches='tight')
print("Grafik başarıyla 'test/loss_graph.png' olarak kaydedildi.")

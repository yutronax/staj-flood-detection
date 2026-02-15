import matplotlib.pyplot as plt

# Paylaşılan loglardan çıkarılan veriler (Strengthened v2: BCE + Dice + Augmentation)
epochs = list(range(1, 21))
train_losses = [
    0.6021, 0.5048, 0.4538, 0.4081, 0.3642, 0.3301, 0.2972, 0.2686, 0.2449, 0.2253,
    0.2166, 0.2015, 0.1905, 0.1779, 0.1705, 0.1642, 0.1551, 0.1511, 0.1473, 0.1420
]
val_losses = [
    0.5321, 0.4770, 0.4466, 0.3898, 0.3490, 0.3164, 0.2860, 0.2530, 0.2327, 0.2188,
    0.2147, 0.1949, 0.1860, 0.1794, 0.1701, 0.1730, 0.1556, 0.1544, 0.1551, 0.1521
]

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_losses, label='Eğitim Kaybı (Train Loss)', marker='o', color='#2ca02c', linewidth=2)
plt.plot(epochs, val_losses, label='Doğrulama Kaybı (Val Loss)', marker='s', color='#d62728', linewidth=2)

plt.title('Attention U-Net v2 (BCE+Dice): Eğitim Süreci Analizi', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Combined Loss', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=11)

plt.xticks(epochs)
plt.style.use('seaborn-v0_8-muted')

# Kaydet
plt.savefig('test/loss_graph_v2.png', dpi=300, bbox_inches='tight')
print("V2 Grafiği başarıyla 'test/loss_graph_v2.png' olarak kaydedildi.")

import matplotlib.pyplot as plt
import os

# FloodNet Training Logs (50 Epochs)
epochs = list(range(1, 51))
train_losses = [
    2.1578, 1.7986, 1.5975, 1.4749, 1.3968, 1.3196, 1.2564, 1.1935, 1.1531, 1.0851,
    1.0791, 1.0515, 1.0170, 0.9640, 0.9025, 0.8873, 0.8329, 0.7997, 0.7632, 0.7458,
    0.7236, 0.7196, 0.6682, 0.6394, 0.6214, 0.5911, 0.6021, 0.5478, 0.5417, 0.5375,
    0.5477, 0.5603, 0.5112, 0.4786, 0.4364, 0.4136, 0.3996, 0.4167, 0.4315, 0.3974,
    0.3606, 0.3889, 0.3807, 0.3654, 0.3754, 0.3356, 0.3462, 0.3343, 0.3107, 0.3158
]
val_losses = [
    2.2280, 1.8544, 1.5301, 1.3525, 1.3882, 1.3628, 1.2404, 1.1528, 1.1685, 1.0430,
    0.9720, 0.9322, 0.9808, 0.8848, 0.8492, 0.8380, 0.8620, 0.8025, 0.8601, 0.7972,
    0.7335, 0.7482, 0.6809, 0.6112, 0.6287, 0.5760, 0.5953, 0.5900, 0.5547, 0.6426,
    0.7406, 0.5560, 0.5326, 0.5156, 0.4904, 0.4734, 0.4914, 0.4581, 0.4823, 0.4493,
    0.4525, 0.4643, 0.4224, 0.4454, 0.4405, 0.4609, 0.4347, 0.3881, 0.4192, 0.3931
]

plt.figure(figsize=(12, 7))
plt.plot(epochs, train_losses, label='Train Loss (CrossEntropy)', color='#1f77b4', linewidth=2)
plt.plot(epochs, val_losses, label='Val Loss (CrossEntropy)', color='#ff7f0e', linewidth=2)

plt.title('FloodNet Multi-Class Training: Loss Curve (50 Epochs)', fontsize=15, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.style.use('seaborn-v0_8-whitegrid')

# Save
os.makedirs('results/floodnet', exist_ok=True)
plt.savefig('results/floodnet/loss_curve_floodnet.png', dpi=300, bbox_inches='tight')
print("FloodNet loss curve saved to 'results/floodnet/loss_curve_floodnet.png'")

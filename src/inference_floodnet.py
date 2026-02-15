import torch
import matplotlib.pyplot as plt
import numpy as np
from src.models.attention_unet import AttentionUNet
from src.data.floodnet_loader import FloodNetDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import os

# FloodNet Sınıf Renkleri (Görselleştirme için)
COLOR_MAP = np.array([
    [0, 0, 0],       # 0: Background - Siyah
    [255, 0, 0],     # 1: Building-Flooded - Kırmızı (HASARLI)
    [0, 255, 0],     # 2: Building-Non-Flooded - Yeşil (SAĞLAM)
    [0, 0, 255],     # 3: Road-Flooded - Mavi (SEL/SU)
    [128, 128, 128], # 4: Road-Non-Flooded - Gri
    [0, 255, 255],   # 5: Water - Cam Göbeği
    [128, 0, 0],     # 6: Tree - Kahverengi
    [255, 255, 0],   # 7: Vehicle - SARI (ARAÇ)
    [255, 0, 255],   # 8: Pool - Mor
    [0, 128, 0]      # 9: Grass - Koyu Yeşil
])

def decode_mask(mask):
    """Sınıf ID'lerini RGB renklerine dönüştürür."""
    rgb = COLOR_MAP[mask]
    return rgb.astype(np.uint8)

def visualize_floodnet(model_path, data_dir, num_samples=5, output_dir='results/floodnet_predictions', device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    os.makedirs(output_dir, exist_ok=True)
    
    model = AttentionUNet(in_channels=3, out_channels=10).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = T_viz = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )

    dataset = FloodNetDataset(root_dir=data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    for i in range(num_samples):
        img, mask = next(iter(loader))
        img = img.to(device)

        with torch.no_grad():
            output = model(img)
            # (1, 10, 256, 256) -> Argmax ile en yüksek olasılıklı sınıfı seç
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

        img_show = inv_normalize(img.squeeze(0)).permute(1, 2, 0).cpu().numpy()
        img_show = np.clip(img_show, 0, 1)
        
        gt_show = decode_mask(mask.squeeze(0).numpy())
        pred_show = decode_mask(pred)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(img_show)
        axes[0].set_title("Original Aerial Image")
        axes[1].imshow(gt_show)
        axes[1].set_title("Ground Truth (All Objects/States)")
        axes[2].imshow(pred_show)
        axes[2].set_title("Prediction (Vehicles, Floods, Buildings)")
        
        for ax in axes: ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'floodnet_test_{i+1}.png'))
        plt.close()
        print(f"Saved inference: {output_dir}/floodnet_test_{i+1}.png")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    args = parser.parse_args()
    visualize_floodnet(args.model_path, args.data_dir)

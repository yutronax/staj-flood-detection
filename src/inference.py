import torch
import matplotlib.pyplot as plt
import numpy as np
from src.models.attention_unet import AttentionUNet
from src.data.xbd_loader import XBDDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import os

def visualize_inference(model_path, data_dir, device='cuda'):
    # [DEVICE]
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # [MODEL] Load
    model = AttentionUNet(in_channels=6, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # [DATA] Load
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Unnormalized transform for visualization
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )

    dataset = XBDDataset(root_dir=data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Get one sample
    images, mask = next(iter(loader))
    images, mask = images.to(device), mask.to(device)

    # Inference
    with torch.no_grad():
        output = model(images)
        pred = torch.sigmoid(output) > 0.5
        pred = pred.float()

    # Convert to CPU/Numpy for plotting
    images = images.cpu().squeeze(0)
    pre_img = inv_normalize(images[:3]).permute(1, 2, 0).numpy()
    post_img = inv_normalize(images[3:]).permute(1, 2, 0).numpy()
    gt_mask = mask.cpu().squeeze().numpy()
    pred_mask = pred.cpu().squeeze().numpy()

    # Clip for safe plotting
    pre_img = np.clip(pre_img, 0, 1)
    post_img = np.clip(post_img, 0, 1)

    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(pre_img)
    axes[0].set_title("Pre-Disaster RGB")
    axes[0].axis('off')

    axes[1].imshow(post_img)
    axes[1].set_title("Post-Disaster RGB")
    axes[1].axis('off')

    axes[2].imshow(gt_mask, cmap='gray')
    axes[2].set_title("Ground Truth Mask")
    axes[2].axis('off')

    axes[3].imshow(pred_mask, cmap='jet')
    axes[3].set_title("Predicted Mask (Attention U-Net)")
    axes[3].axis('off')

    plt.tight_layout()
    plt.show()
    plt.savefig('docs/inference_result.png')
    print("Inference completed and saved to docs/inference_result.png")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument('--data_dir', type=str, default='data/xbd/train')
    args = parser.parse_args()

    os.makedirs('docs', exist_ok=True)
    visualize_inference(args.model_path, args.data_dir)

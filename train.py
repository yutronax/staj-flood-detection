import os
import sys
# [COLAB] Python path ayarı | colab, path
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from src.models.attention_unet import AttentionUNet
from src.data.xbd_loader import XBDDataset
import argparse

def train_model(args):
    # [DEVICE] GPU/CPU Seçimi
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # [DATA] Hazırlık
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    from src.utils.losses import CombinedLoss
    
    # Train set için augmentasyonu aktifleştiriyoruz
    full_dataset = XBDDataset(root_dir=args.data_dir, transform=transform, augment=True)
    print(f"Dataset found: {len(full_dataset)} samples.")
    
    if len(full_dataset) == 0:
        print("ERROR: Dataset is empty. Check data_dir.")
        return

    # Train/Val Split (%80 - %20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Validation set'te augmentasyonu kapatıyoruz (Güvenli yöntem)
    val_dataset.dataset.augment = False # Dikket: random_split sonrası ana dataset'e erişiyoruz

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print("Training loop starting with Augmentation and BCE+Dice Loss...")

    # [MODEL] Başlatma
    model = AttentionUNet(in_channels=6, out_channels=1).to(device)
    
    # Gelişmiş Loss ve Optimizer
    criterion = CombinedLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # [LOOP] Eğitim Döngüsü
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")

        # Val Loss
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        
        print(f"==> Epoch {epoch+1} Finished. Avg Train Loss: {train_loss/len(train_loader):.4f}, Avg Val Loss: {val_loss/len(val_loader):.4f}")

        # [SAVE] Checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/xbd/train')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    os.makedirs('checkpoints', exist_ok=True)
    train_model(args)

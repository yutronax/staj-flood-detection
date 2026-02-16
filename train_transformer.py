import os
import sys
# [COLAB] path ayarı
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from src.models.transformer_models import get_segformer_model
from src.data.floodnet_loader import FloodNetDataset
import argparse

def train_transformer(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # SegFormer Pre-trained Normalizasyon Değerleri (ImageNet)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = FloodNetDataset(root_dir=args.data_dir, transform=transform, augment=True)
    print(f"Dataset found: {len(full_dataset)} samples.")
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset.augment = False

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Training SegFormer ({args.model_name}) starting...")

    # [MODEL]
    model = get_segformer_model(num_labels=10, model_name=args.model_name).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr) # Transformers için AdamW önerilir
    
    # Bilgi: SegFormer logits (N, C, H/4, W/4) döner.
    # CrossEntropyLoss için orijinal maske boyutuna upsample edeceğiz.
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images).logits
            
            # Logits'i maske boyutuna upsample et (Interpolation: Bilinear)
            upsampled_logits = nn.functional.interpolate(
                outputs, 
                size=masks.shape[-2:], # (256, 256)
                mode="bilinear", 
                align_corners=False
            )
            
            loss = criterion(upsampled_logits, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")

        # Val
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images).logits
                upsampled_logits = nn.functional.interpolate(
                    outputs, size=masks.shape[-2:], mode="bilinear", align_corners=False
                )
                loss = criterion(upsampled_logits, masks)
                val_loss += loss.item()
        
        print(f"==> Epoch {epoch+1} Finished. Avg Train Loss: {train_loss/len(train_loader):.4f}, Avg Val Loss: {val_loss/len(val_loader):.4f}")

        # [SAVE]
        if (epoch + 1) % 5 == 0:
            os.makedirs('checkpoints_transformer', exist_ok=True)
            save_path = f"checkpoints_transformer/segformer_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"Checkpoint saved: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8) # Transformers bellek dostu ama dikkatli olunmalı
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=6e-5) # SegFormer için daha düşük LR iyidir
    parser.add_argument('--model_name', type=str, default='nvidia/mit-b0')
    args = parser.parse_args()

    train_transformer(args)

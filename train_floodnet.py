import os
import sys
# [COLAB] Python path ayarı
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from src.models.attention_unet import AttentionUNet
from src.data.floodnet_loader import FloodNetDataset
import argparse

def train_floodnet(args):
    # [DEVICE]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # [DATA] 
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    full_dataset = FloodNetDataset(root_dir=args.data_dir, transform=transform, augment=True)
    print(f"Dataset found: {len(full_dataset)} samples.")
    
    if len(full_dataset) < 2:
        raise ValueError("At least two matched images and masks are required.")

    # Split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    indices = torch.randperm(len(full_dataset), generator=torch.Generator().manual_seed(args.seed)).tolist()
    validation_source = FloodNetDataset(root_dir=args.data_dir, transform=transform, augment=False)
    train_dataset = Subset(full_dataset, indices[:train_size])
    val_dataset = Subset(validation_source, indices[train_size:])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Training FloodNet (10 classes) starting...")

    # [MODEL] Multi-class output = 10
    model = AttentionUNet(in_channels=3, out_channels=10).to(device)
    
    # Loss: CrossEntropy (Multi-class için standart)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # [LOOP]
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            
            # CrossEntropy expects (N, C, H, W) vs (N, H, W) target
            loss = criterion(outputs, masks)
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
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        
        print(f"==> Epoch {epoch+1} Finished. Avg Train Loss: {train_loss/len(train_loader):.4f}, Avg Val Loss: {val_loss/len(val_loader):.4f}")

        # [SAVE]
        if (epoch + 1) % 5 == 0:
            os.makedirs('checkpoints_floodnet', exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints_floodnet/floodnet_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help="Path to unzipped FloodNet folder")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42, help="Seed for the train/validation split")
    args = parser.parse_args()

    train_floodnet(args)


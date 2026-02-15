import os
import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import torchvision.transforms as T

class FloodNetDataset(data.Dataset):
    """
    FloodNet Çok Sınıflı Veri Seti Yükleyicisi
    Sınıflar: Background, Flooded Building, Non-Flooded Building, Flooded Road, etc.
    """
    def __init__(self, root_dir, transform=None, augment=False):
        self.root_dir = root_dir
        self.transform = transform
        self.augment = augment
        self.samples = self._load_samples()
        
        # FloodNet Sınıf Haritası
        self.class_names = {
            0: 'Background',
            1: 'Building-Flooded',
            2: 'Building-Non-Flooded',
            3: 'Road-Flooded',
            4: 'Road-Non-Flooded',
            5: 'Water',
            6: 'Tree',
            7: 'Vehicle',
            8: 'Pool',
            9: 'Grass'
        }

    def _load_samples(self):
        samples = []
        img_dir = os.path.join(self.root_dir, 'images')
        mask_dir = os.path.join(self.root_dir, 'masks')
        
        if not os.path.exists(img_dir):
            return []

        for img_name in os.listdir(img_dir):
            if img_name.endswith('.jpg'):
                mask_name = img_name.replace('.jpg', '.png')
                mask_path = os.path.join(mask_dir, mask_name)
                if os.path.exists(mask_path):
                    samples.append({
                        'image': os.path.join(img_dir, img_name),
                        'mask': mask_path
                    })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        
        image = Image.open(sample['image']).convert('RGB')
        mask = Image.open(sample['mask']).convert('L')
        
        if self.transform:
            image = self.transform(image)
            
            # Maske boyutlandırma (NEAREST interpolation)
            if hasattr(self.transform, 'transforms'):
                for t in self.transform.transforms:
                    if isinstance(t, T.Resize):
                        mask = T.Resize(t.size, interpolation=T.InterpolationMode.NEAREST)(mask)
        
        mask = torch.from_numpy(np.array(mask)).long()

        # [AUGMENTATION]
        if self.augment:
            import random
            if random.random() > 0.5:
                image = torch.flip(image, dims=[2])
                mask = torch.flip(mask, dims=[1])
            if random.random() > 0.5:
                image = torch.flip(image, dims=[1])
                mask = torch.flip(mask, dims=[0])
                
        return image, mask

# [TEST] Loader Doğrulama
if __name__ == "__main__":
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    dataset = FloodNetDataset(root_dir='data/floodnet', transform=transform)
    if len(dataset) > 0:
        x, y = dataset[0]
        print(f"FloodNet Sample Loaded Successfully!")
        print(f"Input Shape: {x.shape}")
        print(f"Mask Shape: {y.shape}")
        print(f"Unique Class IDs: {torch.unique(y)}")
    else:
        print("No samples found in data/floodnet")

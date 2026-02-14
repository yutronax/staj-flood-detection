import os
import torch
import torch.utils.data as data
from PIL import Image
from src.utils.mask_utils import json_to_mask
import torchvision.transforms as T

class XBDDataset(data.Dataset):
    """
    xBD Veri Seti Yükleyicisi | pre, post, damage, segmentation
    
    Afet öncesi ve sonrası görüntüleri birleştirerek 6 kanallı giriş oluşturur.
    """
    def __init__(self, root_dir, transform=None, mask_binary=True):
        self.root_dir = root_dir
        self.transform = transform
        self.mask_binary = mask_binary
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        img_dir = os.path.join(self.root_dir, 'images')
        label_dir = os.path.join(self.root_dir, 'labels')
        
        if os.path.exists(img_dir):
            all_imgs = os.listdir(img_dir)
            pre_imgs = [f for f in all_imgs if 'pre_disaster' in f]
            
            for pre_img in pre_imgs:
                post_img = pre_img.replace('pre_disaster', 'post_disaster')
                if post_img in all_imgs:
                    # Görüntü isminden JSON ismine geçiş
                    label_file = post_img.replace('.png', '.json')
                    label_path = os.path.join(label_dir, label_file)
                    
                    if os.path.exists(label_path):
                        samples.append({
                            'pre': os.path.join(img_dir, pre_img),
                            'post': os.path.join(img_dir, post_img),
                            'label': label_path
                        })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        
        # Görüntüleri yükle
        pre_img = Image.open(sample['pre']).convert('RGB')
        post_img = Image.open(sample['post']).convert('RGB')
        
        # Maskeyi oluştur (JSON'dan)
        mask_np = json_to_mask(sample['label'], image_size=pre_img.size[::-1], binary=self.mask_binary)
        mask = Image.fromarray(mask_np)

        if self.transform:
            # Görüntüleri dönüştür
            pre_img = self.transform(pre_img)
            post_img = self.transform(post_img)
            
            # Maskeyi görüntüyle aynı boyuta getir (Nearest Neighbor interpolation önemli!)
            # self.transform içinde Resize varsa onu kullanmalıyız.
            # Şimdilik basitçe transform'un beklediği boyuta çekiyoruz.
            if hasattr(self.transform, 'transforms'):
                for t in self.transform.transforms:
                    if isinstance(t, T.Resize):
                        mask = T.Resize(t.size, interpolation=T.InterpolationMode.NEAREST)(mask)
            
            mask = T.ToTensor()(mask)
            if self.mask_binary:
                mask = (mask > 0).float() # Kesin 0 veya 1
            
        # Öncesi ve Sonrası görüntüleri birleştir (6 kanal)
        x = torch.cat([pre_img, post_img], dim=0)
        
        return x, mask

# [TEST] Loader Doğrulama
if __name__ == "__main__":
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    dataset = XBDDataset(root_dir='data/xbd/train', transform=transform)
    if len(dataset) > 0:
        x, y = dataset[0]
        print(f"Sample Loaded Successfully!")
        print(f"Input Shape: {x.shape}")
        print(f"Mask Shape: {y.shape}")
        print(f"Unique Label Values: {torch.unique(y)}")
    else:
        print("No samples found in data/xbd/train")

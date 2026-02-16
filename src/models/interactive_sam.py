from transformers import SamModel, SamProcessor
import torch
import numpy as np
from PIL import Image

class DisasterSAM:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = "facebook/sam-vit-base"
        print(f"Loading SAM ({self.model_name})...")
        self.processor = SamProcessor.from_pretrained(self.model_name)
        self.model = SamModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def segment_with_click(self, image, click_coords):
        """
        image: PIL Image
        click_coords: [x, y]
        """
        # Formata hazırla
        input_points = [[click_coords]] # list of list of list for SAM input format
        
        inputs = self.processor(image, input_points=input_points, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Maskeleri al (Orijinal resim boyutuna geri getir)
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), 
            inputs["original_sizes"].cpu(), 
            inputs["reshaped_input_sizes"].cpu()
        )
        
        # En güvenilir maskeyi seç (genelde 3 tane üretir: nesne, parça, detay)
        best_mask = masks[0][0][0].numpy() # İlk maske, ilk batch, 0. skor
        
        return best_mask

if __name__ == "__main__":
    # Test
    # sam = DisasterSAM()
    # print("SAM Ready")
    pass

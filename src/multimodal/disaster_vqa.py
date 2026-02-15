import torch
from PIL import Image
from transformers import ViltProcessor, ViltForQuestionAnswering, BlipProcessor, BlipForConditionalGeneration
import os

class DisasterMultimodalAssistant:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing NON-GPT Multimodal Assistant on {self.device}...")

        # 1. QUESTION ANSWERING: ViLT (Vision-and-Language Transformer)
        # Bu model GPT DEĞİLDİR. Tamamen encoder-only multimodal bir mimaridir.
        self.vqa_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        self.vqa_model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa").to(self.device)

        # 2. IMAGE CAPTIONING: BLIP (ViT Based)
        # Bu model de GPT DEĞİLDİR. Vision Transformer (ViT) tabanlı bir kodlayıcı kullanır.
        self.cap_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.cap_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)

    def generate_report(self, image_path):
        """Görüntüyü analiz eder ve saf bir betimleme (caption) üretir."""
        raw_image = Image.open(image_path).convert('RGB')
        
        inputs = self.cap_processor(raw_image, return_tensors="pt").to(self.device)
        out = self.cap_model.generate(**inputs, max_length=50)
        report = self.cap_processor.decode(out[0], skip_special_tokens=True)
        
        return report

    def ask_assistant(self, image_path, question):
        """Görüntü hakkında teknik bir soru cevaplar (ViLT mimarisi ile)."""
        raw_image = Image.open(image_path).convert('RGB')
        
        # ViLT metin ve görüntüyü aynı katmanda (joint embedding) işler
        inputs = self.vqa_processor(raw_image, question, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.vqa_model(**inputs)
        
        # En yüksek olasılıklı cevabı sınıflandır
        idx = outputs.logits.argmax(-1).item()
        answer = self.vqa_model.config.id2label[idx]
        
        return answer

if __name__ == "__main__":
    # Test Senaryosu
    assistant = DisasterMultimodalAssistant()
    
    # Mevcut sonuçlardan bir resim bulmaya çalışalım
    possible_paths = [
        "results/Phase2_MultiClassDisaster/floodnet_test_1.png",
        "results/Phase1_BuildingDetection/prediction_1.png",
        "data/floodnet/images/10165.jpg" # Yerel veri seti yolu
    ]
    
    test_image = None
    for path in possible_paths:
        if os.path.exists(path):
            test_image = path
            break

    if test_image:
        print(f"\n[TEST]: Testing with image: {test_image}")
        # 1. Otomatik Betimleme
        print("\n[ANALİZ]: Görüntü Tanımlanıyor...")
        desc = assistant.generate_report(test_image)
        print(f"Model Açıklaması: {desc}")
        
        # 2. Teknik Soru-Cevap
        print("\n[SORU-CEVAP]: Teknik Detaylar Sorgulanıyor (ViT+BERT Yapısı)...")
        test_questions = [
            "What is in the image?",
            "Is there any water?",
            "Are there buildings?",
            "What is the weather like?"
        ]
        
        for q in test_questions:
            ans = assistant.ask_assistant(test_image, q)
            print(f"Soru: {q} -> Cevap: {ans}")
    else:
        print("Test görüntüsü bulunamadı. Lütfen önce Phase 2 sonuçlarını üretin.")

from transformers import SegformerForSemanticSegmentation, SegformerConfig
import torch.nn as nn

def get_segformer_model(num_labels=10, model_name="nvidia/mit-b0"):
    """
    Hugging Face SegFormer modelini döner.
    Model: nvidia/mit-b0 (Lightweight) veya nvidia/mit-b5 (Heavyweight)
    """
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True # Pre-trained ağırlıkları yüklerken sınıf sayısı farklıysa hata vermez
    )
    return model

if __name__ == "__main__":
    import torch
    model = get_segformer_model()
    # SegFormer genelde 512x512 veya 256x256 giriş bekler
    x = torch.randn(1, 3, 256, 256)
    outputs = model(x)
    print(f"SegFormer Output Shape: {outputs.logits.shape}")

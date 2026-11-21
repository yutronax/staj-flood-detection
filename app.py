import streamlit as st
import torch
import numpy as np
from PIL import Image
import os
import sys
import matplotlib.pyplot as plt
import torch.nn as nn
from streamlit_image_coordinates import streamlit_image_coordinates

# Proje dizinini ekle
sys.path.append(os.getcwd())

from src.models.attention_unet import AttentionUNet
from src.models.transformer_models import get_segformer_model
from src.models.interactive_sam import DisasterSAM
from src.multimodal.disaster_vqa import DisasterMultimodalAssistant
from src.inference_floodnet import COLOR_MAP, decode_mask
from torchvision import transforms

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Yutronax AI Ultra Pro - Afet Analiz Merkezi",
    page_icon="🌊",
    layout="wide",
)

# --- CSS FOR PREMIUM LOOK ---
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
    }
    .stSidebar {
        background-color: #161b22;
    }
    h1, h2, h3 {
        color: #ff4b4b;
    }
    .interactive-text {
        color: #00ff00;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# --- CACHED MODELS ---
@st.cache_resource
def load_any_model(model_path, mode="SegFormer", out_channels=10, in_channels=3):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model ağırlıkları bulunamadı: {model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if mode == "SegFormer":
        model = get_segformer_model(num_labels=out_channels).to(device)
    else:
        model = AttentionUNet(in_channels=in_channels, out_channels=out_channels).to(device)
        
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model, device

@st.cache_resource
def load_sam():
    return DisasterSAM()

@st.cache_resource
def load_multimodal_ai():
    return DisasterMultimodalAssistant()

# --- APP HEADER ---
st.title("🌊 Yutronax AI Ultra Pro: İnteraktif Afet Yönetimi")
st.markdown("SegFormer, Attention U-Net ve SAM ile deneysel görüntü analizi.")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Mimari Seçimi")
    app_mode = st.radio("Seçenekler", ["Otomatik Analiz", "İnteraktif Segmentasyon (SAM)"])
    
    if app_mode == "Otomatik Analiz":
        model_choice = st.selectbox(
            "Yapay Zeka Modeli",
            ["SegFormer (Transformer)", "Attention U-Net"]
        )
    
    st.markdown("---")
    uploaded_file = st.file_uploader(
        "Afet Görüntüsü Yükleyin", 
        type=["jpg", "png", "jpeg", "webp", "tiff", "tif", "bmp"]
    )
    
    st.info("SAM (Segment Anything Model), seçtiğiniz bir noktadaki nesneyi (bina, sel, araç) anında ayırmanıza olanak tanır.")

# --- MAIN DASHBOARD ---
if uploaded_file is not None:
    original_image = Image.open(uploaded_file).convert('RGB')
    orig_w, orig_h = original_image.size
    
    col1, col2 = st.columns([1, 1])
    
    if app_mode == "Otomatik Analiz":
        with col1:
            st.subheader("📸 Orijinal Görüntü")
            st.image(original_image, use_container_width=True)
            
        # --- INFERENCE CONFIG ---
        if "SegFormer" in model_choice:
            model_path = "checkpoints_transformer/segformer_epoch_20.pth"
            mode = "SegFormer"
        else:
            model_path = "checkpoints_floodnet/floodnet_epoch_20.pth"
            mode = "AttentionUNet"

        if st.button("Analizi Başlat 🚀"):
            with st.spinner(f"{mode} Modeli Pikselleri Hesaplanıyor..."):
                try:
                    model, device = load_any_model(model_path, mode=mode)
                except (FileNotFoundError, RuntimeError, ValueError) as exc:
                    st.error(f"Model yüklenemedi: {exc}")
                    st.stop()
                input_size = (256, 256)
                transform = transforms.Compose([
                    transforms.Resize(input_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                img_input = transform(original_image).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    if mode == "SegFormer":
                        outputs = model(img_input).logits
                        upsampled_logits = nn.functional.interpolate(
                            outputs, size=input_size, mode="bilinear", align_corners=False
                        )
                        pred = torch.argmax(upsampled_logits, dim=1).squeeze(0).cpu().numpy()
                    else:
                        output = model(img_input)
                        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
                    
                    pred_viz_small = decode_mask(pred)
                    pred_viz = Image.fromarray(pred_viz_small).resize((orig_w, orig_h), Image.Resampling.NEAREST)

                with col2:
                    st.subheader(f"🗺️ {mode} Analiz Haritası")
                    st.image(pred_viz, use_container_width=True)

    else: # INTERACTIVE SAM MODE
        with col1:
            st.subheader("🖱️ Nesne Seçmek İçin Resme Tıklayın")
            # Tıklama koordinatlarını yakala
            coords = streamlit_image_coordinates(original_image, use_column_width=True)
            
            if coords:
                x, y = coords["x"], coords["y"]
                st.write(f"Seçilen Nokta: **X={x}, Y={y}**")
                
                sam = load_sam()
                with st.spinner("Meta SAM Nesneyi Ayıklıyor..."):
                    mask = sam.segment_with_click(original_image, [x, y])
                    
                    # Maskeyi renklendir
                    mask_viz = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
                    mask_viz[mask > 0] = [255, 0, 0, 160] # Kırmız yarı saydam maske
                    
                    # Görüntü üzerine bindir
                    overlay = Image.alpha_composite(original_image.convert("RGBA"), Image.fromarray(mask_viz))
                    
                    with col2:
                        st.subheader("🎯 Nesne Segmentasyonu (SAM)")
                        st.image(overlay, use_container_width=True, caption="Seçilen Nesnenin Sınırları")
                        st.success("Nesne başarıyla izole edildi!")

    st.markdown("---")
    # MULTIMODAL AI
    st.subheader("🤖 AI Akıllı Asistan")
    multimodal = load_multimodal_ai()
    temp_path = "temp_ui_image.jpg"
    original_image.save(temp_path)
    
    ai_col1, ai_col2 = st.columns([1, 1])
    with ai_col1:
        caption = multimodal.generate_report(temp_path)
        st.info(f"**AI Raporu:** {caption}")
    with ai_col2:
        user_question = st.text_input("Görüntüye dair soru sor:")
        if user_question:
            answer = multimodal.ask_assistant(temp_path, user_question)
            st.success(f"**AI Cevabı:** {answer}")
else:
    st.warning("👈 Lütfen soldan bir görüntü yükleyerek başlayın.")


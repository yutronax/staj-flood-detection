# Research Summary: Flood Detection & Segmentation

## 1. Architecture: Attention U-Net
Based on `32 Şekil 4.3. Attention mekanizmalı U-Net mimarisi.jpg`:

The proposed model varies from the standard U-Net by incorporating **Attention Gates (AGs)** in the skip connections.

### Key Components:
1.  **Encoder (Contracting Path):**
    - **Backbone:** likely ResNet-34 (as seen in Sec 3.8 experiment) or standard CNN.
    - **Input Strategy:** **6-Channel Input** (Pre-disaster + Post-disaster images concatenated depth-wise).
    - **Dataset:** **xBD (xView2)**. Building damage assessment with multi-temporal RGB imagery.
        - Source: *Page 23 (Section 3.8)* explicitly mentions: *"İki görüntü kanal bazında birleştirilerek modele 6 kanallı tek bir girdi sunulmuştur."*
    - Downsampling via Max Pooling.
2.  **Decoder (Expanding Path):**
    - Upsampling layers.
    - **Attention Blocks:** Features from encoder are filtered via Attention Gates before concatenation.
3.  **Attention Mechanism (Fig 4.3):**
    - Filters encoder features to focus on relevant changes (flood).
    - Components: 1x1 Convs, Sigmoid activation, Resampling.

## 2. Training Details (Extracted from Sec 3.8 & 4)
- **Loss Function:** Binary Cross-Entropy (BCE).
- **Optimizer:** Adam.
- **Metrics:** Dice Score, IoU (Intersection over Union).
- **Training Strategy:** Mixed Precision training, Early Stopping based on Val Dice.

## 3. Bibliography (Extracted)

1.  **Joyce, K. E., et al. (2009).** A review of the status of satellite remote sensing... *Progress in Physical Geography*.
2.  **Xia, H., et al. (2023).** A deep learning application for building damage assessment... *International Journal of Disaster Risk Science*.
3.  **Wu, C., et al. (2021).** Building damage detection using U-Net with attention mechanism... *Remote Sensing*. (Critical Reference)
4.  **Sha, Z., et al. (2022).** MITFormer...
... (and others)

## 4. Implementation Plan
- **Goal:** Implement **Attention U-Net** with **6-channel input**.
- **Files:**
    - `src/models/modules.py`: `AttentionBlock`, `ConvBlock`.
    - `src/models/attention_unet.py`: Main model taking `in_channels=6`.


# Landslide Contrastive Learning (SwAV + U-Net)

Reproducible implementation of a self-supervised learning and segmentation pipeline for landslide detection, following Ghorbanzadeh et al. (2024) and SwAV.

## Overview

1. **SwAV Pre-training (Self-Supervised):**
   - Backbone: ResNet-18 modified for multiband images (14 channels).
   - Remove avgpool and fully-connected layers to obtain 2D feature maps.
   - Apply global average pooling and a projection head (MLP) followed by L2 normalization.
   - SwAV prototypes: linear layer without bias, output K (number of prototypes, e.g. 128).
   - SwAV loss with balanced assignment via Sinkhorn-Knopp.
   - Multi-crop augmentation: 2 global views (128x128) and 2 local views (96x96 resized to 128x128) per image.

2. **U-Net Fine-tuning (Supervised):**
   - Encoder: Pretrained ResNet-18 (frozen).
   - Decoder: U-Net style, with skip connections and progressive upsampling.
   - Loss: Binary Cross-Entropy (BCE).
   - Training with a small labeled set (1%-10% of the data).

3. **Evaluation:**
   - Metrics: Precision, Recall, F1-score per pixel.
   - Visualization of predicted masks vs. ground truth.


## Repository structure

- `models/swav_model.py`: SwAV model (ResNet-18 + projection + prototypes)
- `models/unet_resnet.py`: U-Net with ResNet-18 encoder
- `augmentations.py`: Multi-crop augmentations for SwAV
- `swav_loss.py`: SwAV loss with Sinkhorn
- `train_swav.py`: Self-supervised training
- `train_unet.py`: Supervised fine-tuning
- `eval_unet.py`: Evaluation and visualization


## TODO

- [x] Implement multiband ResNet-18 backbone and SwAV head
- [x] Implement multi-crop augmentations
- [x] Implement SwAV loss (Sinkhorn)
- [x] Implement U-Net with frozen encoder
- [x] SwAV and U-Net training scripts
- [x] Evaluation and visualization script
- [ ] Adapt reading of multispectral images (>3 channels) in datasets
- [ ] Add support for different band combinations (RGB, RGB+NIR+Slope, etc.)
- [ ] Improve result visualization (overlays, batch metrics)
- [ ] Add validation and early stopping
- [ ] Document dependencies and environment (requirements.txt)
- [ ] Add usage examples and expected results

## References

- Ghorbanzadeh et al., 2024. [uwaterloo.ca]
- Caron et al., SwAV. [arxiv.org/abs/2006.09882]
- Lightly SwAV Example. [docs.lightly.ai]

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
- `l4s_dataset.py`: Dataset utilities for the Landslide4Sense HDF5 files

## Dataset

The `Landslide4Sense` dataset consists of multispectral image patches stored as
HDF5 files. Download the training images and masks from the official release or
the HuggingFace mirror and organize them as:

```
data/
  TrainData/
    img/
      image_1.h5
      ...
    mask/
      image_1.h5
      ...
```

`l4s_dataset.py` provides `L4SUnlabeledDataset` and `L4SSegmentationDataset`
to load these files for SwAV pre-training and U-Net fine-tuning respectively.

## Usage

1. **Pre-train the encoder**

   ```bash
   python train_swav.py --img_dir data/TrainData/img --epochs 200 --batch_size 32
   ```

   This saves `swav_checkpoint_final.pth` with the learned weights.

2. **Fine-tune the U-Net**

   ```bash
   python train_unet.py \
       --img_dir data/TrainData/img \
       --mask_dir data/TrainData/mask \
       --checkpoint swav_checkpoint_final.pth \
       --epochs 1000 --batch_size 16
   ```

   Only the decoder parameters are trained while the encoder is frozen.


## TODO

- [x] Implement multiband ResNet-18 backbone and SwAV head
- [x] Implement multi-crop augmentations
- [x] Implement SwAV loss (Sinkhorn)
- [x] Implement U-Net with frozen encoder
- [x] SwAV and U-Net training scripts
- [x] Evaluation and visualization script
- [x] Adapt reading of multispectral images (>3 channels) in datasets
- [ ] Add support for different band combinations (RGB, RGB+NIR+Slope, etc.)
- [ ] Improve result visualization (overlays, batch metrics)
- [ ] Add validation and early stopping
- [ ] Document dependencies and environment (requirements.txt)
- [x] Add usage examples and expected results

## References

- Ghorbanzadeh et al., 2024. [uwaterloo.ca]
- Caron et al., SwAV. [arxiv.org/abs/2006.09882]
- Lightly SwAV Example. [docs.lightly.ai]
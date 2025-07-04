import torch
import numpy as np
from models.unet_resnet import UNetResNet18
from PIL import Image
import os
import matplotlib.pyplot as plt

def threshold_mask(mask, thresh=0.5):
    return (mask > thresh).astype(np.uint8)

def compute_metrics(pred, gt):
    TP = np.logical_and(pred == 1, gt == 1).sum()
    FP = np.logical_and(pred == 1, gt == 0).sum()
    FN = np.logical_and(pred == 0, gt == 1).sum()
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1

def visualize(img, pred_mask, gt_mask, save_path=None):
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.imshow(img.transpose(1,2,0))
    plt.title('Imagen')
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.imshow(pred_mask, cmap='gray')
    plt.title('Predicción')
    plt.axis('off')
    plt.subplot(1,3,3)
    plt.imshow(gt_mask, cmap='gray')
    plt.title('Ground Truth')
    plt.axis('off')
    if save_path:
        plt.savefig(save_path)
    plt.show()

def main():
    in_channels = 14
    model = UNetResNet18(in_channels=in_channels, freeze_encoder=True)
    model.load_state_dict(torch.load('unet_checkpoint_final.pth', map_location='cpu'))
    model.eval()
    img_dir = 'data/val/images/'
    mask_dir = 'data/val/masks/'
    imgs = [f for f in os.listdir(img_dir) if f.endswith('.png') or f.endswith('.jpg')]
    all_prec, all_rec, all_f1 = [], [], []
    for fname in imgs:
        img = Image.open(os.path.join(img_dir, fname)).convert('RGB')
        mask = Image.open(os.path.join(mask_dir, fname)).convert('L')
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.tensor(img_np).permute(2,0,1).unsqueeze(0)
        with torch.no_grad():
            pred = model(img_tensor).squeeze().cpu().numpy()
        pred_bin = threshold_mask(pred)
        gt_bin = threshold_mask(np.array(mask).astype(np.float32)/255.0)
        precision, recall, f1 = compute_metrics(pred_bin, gt_bin)
        all_prec.append(precision)
        all_rec.append(recall)
        all_f1.append(f1)
        visualize(img_np.transpose(2,0,1), pred_bin, gt_bin)
        print(f"{fname}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
    print(f"\nPromedio: Precision={np.mean(all_prec):.3f}, Recall={np.mean(all_rec):.3f}, F1={np.mean(all_f1):.3f}")

if __name__ == '__main__':
    main()

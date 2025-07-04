import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from models.unet_resnet import UNetResNet18
from l4s_dataset import L4SSegmentationDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune U-Net on L4S")
    parser.add_argument("--img_dir", required=True, help="Directory with HDF5 images")
    parser.add_argument("--mask_dir", required=True, help="Directory with HDF5 masks")
    parser.add_argument("--checkpoint", required=True, help="Pretrained SwAV weights")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()

    dataset = L4SSegmentationDataset(args.img_dir, args.mask_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNetResNet18(in_channels=14, freeze_encoder=True).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.encoder.load_state_dict(state_dict, strict=False)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    criterion = torch.nn.BCELoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            preds = model(imgs)
            loss = criterion(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {total_loss/len(loader):.4f}")
        for g in optimizer.param_groups:
            g["lr"] *= 0.95
        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(), f"unet_checkpoint_{epoch+1}.pth")

    torch.save(model.state_dict(), "unet_checkpoint_final.pth")


if __name__ == "__main__":
    main()

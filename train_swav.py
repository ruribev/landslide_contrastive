import argparse
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from models.swav_model import SwaVModel
from augmentations import multi_crop_augment
from swav_loss import swav_loss_func
from l4s_dataset import L4SUnlabeledDataset


def parse_args():
    parser = argparse.ArgumentParser(description="SwAV pre-training on L4S")
    parser.add_argument("--img_dir", required=True, help="Directory with HDF5 images")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--proj_dim", type=int, default=128)
    parser.add_argument("--n_prototypes", type=int, default=128)
    return parser.parse_args()


def collate_fn(batch):
    views = []
    for img in batch:
        views.extend(multi_crop_augment(img))
    return torch.stack(views)


def main():
    args = parse_args()

    dataset = L4SUnlabeledDataset(args.img_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SwaVModel(in_channels=14, proj_dim=args.proj_dim, n_prototypes=args.n_prototypes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for views in loader:
            views = views.to(device)
            logits, _ = model(views)
            loss = swav_loss_func(logits)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {total_loss/len(loader):.4f}")
        for g in optimizer.param_groups:
            g["lr"] *= 0.95
        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), f"swav_checkpoint_{epoch+1}.pth")

    torch.save(model.state_dict(), "swav_checkpoint_final.pth")


if __name__ == "__main__":
    main()

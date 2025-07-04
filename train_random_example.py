import torch
from torch.utils.data import Dataset, DataLoader
from models.unet_resnet import UNetResNet18

class RandomSegmentationDataset(Dataset):
    """Generates random images and masks for quick training tests."""
    def __init__(self, num_samples=50, in_channels=14, img_size=128):
        self.num_samples = num_samples
        self.in_channels = in_channels
        self.img_size = img_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = torch.rand(self.in_channels, self.img_size, self.img_size)
        mask = torch.randint(0, 2, (1, self.img_size, self.img_size)).float()
        return img, mask


def main():
    # Parameters
    in_channels = 14
    batch_size = 4
    num_epochs = 2

    dataset = RandomSegmentationDataset(num_samples=16, in_channels=in_channels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = UNetResNet18(in_channels=in_channels, freeze_encoder=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
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
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss/len(loader):.4f}")


if __name__ == "__main__":
    main()

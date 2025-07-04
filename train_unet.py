import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from models.unet_resnet import UNetResNet18
import os

# Ejemplo de Dataset para segmentación (adapta a tu estructura)
class LandslideSegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.imgs = [f for f in os.listdir(img_dir) if f.endswith('.png') or f.endswith('.jpg')]
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.img_dir, self.imgs[idx])).convert('RGB')
        mask = Image.open(os.path.join(self.mask_dir, self.imgs[idx])).convert('L')
        img = torch.tensor(np.array(img)).permute(2,0,1).float() / 255.0
        mask = torch.tensor(np.array(mask)).unsqueeze(0).float() / 255.0
        return img, mask

# Hiperparámetros
in_channels = 14
batch_size = 16
num_epochs = 1000
img_dir = 'data/labeled/images/'  # Cambia a tu ruta
mask_dir = 'data/labeled/masks/'  # Cambia a tu ruta

# Carga modelo y pesos preentrenados
model = UNetResNet18(in_channels=in_channels, freeze_encoder=True)
model.encoder.load_state_dict(torch.load('swav_checkpoint_final.pth'), strict=False)
model = model.cuda() if torch.cuda.is_available() else model
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
criterion = torch.nn.BCELoss()

dataset = LandslideSegmentationDataset(img_dir, mask_dir)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for imgs, masks in loader:
        imgs = imgs.cuda() if torch.cuda.is_available() else imgs
        masks = masks.cuda() if torch.cuda.is_available() else masks
        preds = model(imgs)
        loss = criterion(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss/len(loader):.4f}")
    for g in optimizer.param_groups:
        g['lr'] *= 0.95
    if (epoch+1) % 100 == 0:
        torch.save(model.state_dict(), f'unet_checkpoint_{epoch+1}.pth')

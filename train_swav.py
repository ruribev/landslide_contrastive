import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from models.swav_model import SwaVModel
from augmentations import multi_crop_augment
from swav_loss import swav_loss_func
import os

# Ejemplo de Dataset (debes adaptar a tu estructura de datos)
class LandslideUnlabeledDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.imgs = [f for f in os.listdir(img_dir) if f.endswith('.png') or f.endswith('.jpg')]
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.img_dir, self.imgs[idx])).convert('RGB')
        views = multi_crop_augment(img)
        return views  # lista de 4 tensores

# Hiperparámetros
in_channels = 14
proj_dim = 128
n_prototypes = 128
batch_size = 32
num_epochs = 200
img_dir = 'data/unlabeled/'  # Cambia a tu ruta

dataset = LandslideUnlabeledDataset(img_dir)
def collate_fn(batch):
    # batch: lista de listas de 4 tensores
    views = [v for sample in batch for v in sample]
    return torch.stack(views)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

model = SwaVModel(in_channels, proj_dim, n_prototypes)
model = model.cuda() if torch.cuda.is_available() else model
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for views in loader:
        views = views.cuda() if torch.cuda.is_available() else views
        logits, _ = model(views)
        loss = swav_loss_func(logits)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss/len(loader):.4f}")
    # Decaimiento LR
    for g in optimizer.param_groups:
        g['lr'] *= 0.95
    # Guardar checkpoint
    if (epoch+1) % 50 == 0:
        torch.save(model.state_dict(), f'swav_checkpoint_{epoch+1}.pth')

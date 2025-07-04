from torchvision import transforms
from PIL import Image

# Augmentaciones multi-crop para SwAV

global_transforms = transforms.Compose([
    transforms.RandomResizedCrop(128, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.ToTensor(),
])

local_transforms = transforms.Compose([
    transforms.RandomResizedCrop(96, scale=(0.2, 0.5)),
    transforms.Resize(128),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.ToTensor(),
])

def multi_crop_augment(img):
    """
    Dada una imagen PIL, retorna 2 vistas globales y 2 locales (todas 128x128)
    """
    views = [global_transforms(img) for _ in range(2)]
    views += [local_transforms(img) for _ in range(2)]
    return views

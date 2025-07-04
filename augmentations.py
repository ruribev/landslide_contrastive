import torch
import torch.nn.functional as F


def random_flip(tensor: torch.Tensor) -> torch.Tensor:
    """Random horizontal flip."""
    if torch.rand(1) < 0.5:
        tensor = tensor.flip(-1)
    return tensor


def random_crop(tensor: torch.Tensor, size: int) -> torch.Tensor:
    """Random square crop of given size."""
    _, h, w = tensor.shape
    if h == size and w == size:
        return tensor
    top = torch.randint(0, h - size + 1, (1,)).item()
    left = torch.randint(0, w - size + 1, (1,)).item()
    return tensor[:, top : top + size, left : left + size]


def resize(tensor: torch.Tensor, size: int) -> torch.Tensor:
    return F.interpolate(tensor.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False).squeeze(0)


def multi_crop_augment(img: torch.Tensor) -> list:
    """Return two global and two local crops from a tensor image."""
    g1 = random_flip(img.clone())
    g2 = random_flip(img.clone())
    l1 = random_flip(resize(random_crop(img, 96), 128))
    l2 = random_flip(resize(random_crop(img, 96), 128))
    return [g1, g2, l1, l2]

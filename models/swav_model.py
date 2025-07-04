import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class ResNet18Backbone(nn.Module):
    def __init__(self, in_channels=14):
        super().__init__()
        resnet = torchvision.models.resnet18(pretrained=False)
        resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
    def forward(self, x):
        return self.backbone(x)  # [B, 512, 4, 4] para input 128x128

class SwaVProjectionHead(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=512, out_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x):
        return self.mlp(x)

class SwaVPrototypes(nn.Module):
    def __init__(self, in_dim, n_prototypes):
        super().__init__()
        self.prototypes = nn.Linear(in_dim, n_prototypes, bias=False)
    def forward(self, x):
        return self.prototypes(x)

class SwaVModel(nn.Module):
    def __init__(self, in_channels=14, proj_dim=128, n_prototypes=128):
        super().__init__()
        self.encoder = ResNet18Backbone(in_channels)
        self.proj_head = SwaVProjectionHead(512, 512, proj_dim)
        self.prototypes = SwaVPrototypes(proj_dim, n_prototypes)
    def forward(self, x):
        feats = self.encoder(x)  # [B, 512, 4, 4]
        pooled = feats.mean([-2, -1])  # global avg pool -> [B, 512]
        proj = self.proj_head(pooled)  # [B, proj_dim]
        proj = F.normalize(proj, dim=1)
        logits = self.prototypes(proj)  # [B, n_prototypes]
        return logits, proj

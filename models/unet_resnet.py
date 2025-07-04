import torch
import torch.nn as nn
import torchvision

class ResNet18Encoder(nn.Module):
    def __init__(self, in_channels=14):
        super().__init__()
        resnet = torchvision.models.resnet18(pretrained=False)
        resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
    def forward(self, x):
        enc1 = self.relu(self.bn1(self.conv1(x)))  # 64x64
        enc2 = self.layer1(self.maxpool(enc1))     # 32x32
        enc3 = self.layer2(enc2)                   # 16x16
        enc4 = self.layer3(enc3)                   # 8x8
        enc5 = self.layer4(enc4)                   # 4x4
        return {"enc1": enc1, "enc2": enc2, "enc3": enc3, "enc4": enc4, "enc5": enc5}

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class UNetResNet18(nn.Module):
    def __init__(self, in_channels=14, freeze_encoder=True):
        super().__init__()
        self.encoder = ResNet18Encoder(in_channels)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.up4 = UpBlock(512, 256, 256)  # 4x4 -> 8x8
        self.up3 = UpBlock(256, 128, 128)  # 8x8 -> 16x16
        self.up2 = UpBlock(128, 64, 64)    # 16x16 -> 32x32
        self.up1 = UpBlock(64, 64, 64)     # 32x32 -> 64x64
        self.up0 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)  # 64x64 -> 128x128
        self.final = nn.Conv2d(32, 1, kernel_size=1)
    def forward(self, x):
        feats = self.encoder(x)
        x = self.up4(feats["enc5"], feats["enc4"])
        x = self.up3(x, feats["enc3"])
        x = self.up2(x, feats["enc2"])
        x = self.up1(x, feats["enc1"])
        x = self.up0(x)
        x = self.final(x)
        return torch.sigmoid(x)

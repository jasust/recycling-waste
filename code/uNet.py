import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, base:int = 32, bilinear:bool = False):
        super(UNet, self).__init__()
        self.bilinear = bilinear

        self.inc = (DoubleConv(3, base))
        self.down1 = (Down(base, base*2))
        self.down2 = (Down(base*2, base*4))
        self.down3 = (Down(base*4, base*8))
        factor = 2 if bilinear else 1
        self.down4 = (Down(base*8, base*16 // factor))
        self.up1 = (Up(base*16, base*8 // factor, bilinear))
        self.up2 = (Up(base*8, base*4 // factor, bilinear))
        self.up3 = (Up(base*4, base*2 // factor, bilinear))
        self.up4 = (Up(base*2, base, bilinear))
        self.outc = (OutConv(base, 1))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


def dice_coeff(input: Tensor, target: Tensor, epsilon: float = 1e-6):
    assert input.size() == target.size()

    sum_dim = (-1, -2) # (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()

def dice_numpy(input: np.ndarray, target: np.ndarray, epsilon: float = 1e-6):

    target = target.reshape((target.shape[0],target.shape[1]))
    union = np.sum(input) + np.sum(target) + epsilon
    inter = 2*np.sum(np.multiply(input, target)) + epsilon

    return inter/union

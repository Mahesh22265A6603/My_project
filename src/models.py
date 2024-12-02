import torch
import torch.nn as nn
import torch.nn.functional as F

# Attention Block (Enhances important regions of the image)
class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super(AttentionBlock, self).__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(channels, channels // 8, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention = self.attn(x)
        return x * attention  # Element-wise multiplication with input feature map

# Enhanced CNN for Deblurring with Color Preservation
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Depth-wise Separable Convolution Layers
        self.conv1 = nn.Conv2d(3, 128, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.attn2 = AttentionBlock(64)

        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.attn3 = AttentionBlock(32)

        self.conv4 = nn.Conv2d(32, 3, kernel_size=1)

        # Skip Connection Layer for Color Consistency
        self.skip_conv = nn.Conv2d(3, 3, kernel_size=1)

        # Parametric ReLU for adaptive activation
        self.prelu = nn.PReLU()

    def forward(self, x):
        residual = self.skip_conv(x)
        x = self.prelu(self.bn1(self.conv1(x)))
        x = self.attn2(self.prelu(self.bn2(self.conv2(x))))
        x = self.attn3(self.prelu(self.bn3(self.conv3(x))))
        x = self.conv4(x) + residual  # Skip connection to preserve colors
        return torch.clamp(x, 0, 1)  # Ensuring valid image range

# Enhanced Autoencoder (SimpleAE) with Color Distortion Mitigation
class SimpleAE(nn.Module):
    def __init__(self):
        super(SimpleAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            AttentionBlock(128)
        )

        # Decoder with Skip Connections and Color Preservation
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=5, stride=1, padding=2),
            nn.Sigmoid()  # Sigmoid to maintain output within [0,1]
        )

    def forward(self, x):
        enc_output = self.encoder(x)
        dec_output = self.decoder(enc_output)
        return dec_output

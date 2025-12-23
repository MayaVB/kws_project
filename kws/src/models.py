import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicDSCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        # Standard 2D convolution (feature extractor starter)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(64)
        # Depthwise convolution (separate filter per channel)
        # Each channel has its own filter → no mixing between channels
        self.dw2   = nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64)
        self.bn2   = nn.BatchNorm2d(64)
        # Pointwise convolution (1x1) — mixes channels after depthwise
        self.pw2   = nn.Conv2d(64, 64, kernel_size=1)
        self.bn3   = nn.BatchNorm2d(64)
        # Downsampling to reduce feature map size
        self.pool2 = nn.MaxPool2d(2)

        # Second Depthwise + Pointwise block
        self.dw3   = nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64)
        self.bn4   = nn.BatchNorm2d(64)
        self.pw3   = nn.Conv2d(64, 128, kernel_size=1)
        self.bn5   = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)

        # Global Average Pooling — converts feature map into one vector
        self.gap   = nn.AdaptiveAvgPool2d(1)

        # Final classifier (Logits vector which goes to softmax)
        self.fc_out = nn.Linear(128, num_classes)

    def forward(self, x):
        # Initial conv
        x = F.relu(self.bn1(self.conv1(x)))

        # Block 1: Depthwise + Pointwise
        x = F.relu(self.bn2(self.dw2(x)))
        x = F.relu(self.bn3(self.pw2(x)))
        x = self.pool2(x)  # downsample

        # Block 2: Depthwise + Pointwise
        x = F.relu(self.bn4(self.dw3(x)))
        x = F.relu(self.bn5(self.pw3(x)))
        x = self.pool3(x)  # downsample

        # Pool to a single vector
        x = self.gap(x)
        x = torch.flatten(x, 1)

        # Output layer
        x = self.fc_out(x)
        return x


class ImprovedDSCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        # Initial conv + batch norm (same as Basic)s
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(64)

        # Block 1: Depthwise + Pointwise + MaxPool + Dropout
        self.dw2   = nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64)
        self.bn2   = nn.BatchNorm2d(64)
        self.pw2   = nn.Conv2d(64, 64, kernel_size=1)
        self.bn3   = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.drop2 = nn.Dropout(0.3) # regularization

        # Block 2: Depthwise + Pointwise + MaxPool + Dropout
        self.dw3   = nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64)
        self.bn4   = nn.BatchNorm2d(64)
        self.pw3   = nn.Conv2d(64, 128, kernel_size=1)
        self.bn5   = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)
        self.drop3 = nn.Dropout(0.3)

        # Global Average Pooling
        self.gap   = nn.AdaptiveAvgPool2d(1)

        # Extra fully-connected layer for more capacity + dropout
        self.fc1   = nn.Linear(128, 128)
        self.drop_fc = nn.Dropout(0.5)
        self.fc_out  = nn.Linear(128, num_classes)

    def forward(self, x):
        # Input: (B, 1, T, F)
        print("in", x.shape)
        x = F.relu(self.bn1(self.conv1(x)))

        x = F.relu(self.bn2(self.dw2(x)))
        x = F.relu(self.bn3(self.pw2(x)))
        x = self.pool2(x)
        x = self.drop2(x)

        x = F.relu(self.bn4(self.dw3(x)))
        x = F.relu(self.bn5(self.pw3(x)))
        x = self.pool3(x)
        x = self.drop3(x)

        x = self.gap(x)
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.drop_fc(x)
        x = self.fc_out(x)
        return x

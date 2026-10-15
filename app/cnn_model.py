"""Part 1 — CNN architecture (Assignment 2).

Spec:
    Input: RGB image, 64x64x3
    Conv2d: 16 filters, 3x3, stride 1, padding 1 -> ReLU
    MaxPool2d: 2x2, stride 2
    Conv2d: 32 filters, 3x3, stride 1, padding 1 -> ReLU
    MaxPool2d: 2x2, stride 2
    Flatten
    Linear: 100 units -> ReLU
    Linear: 10 units (10 output classes)

Shape trace (single image):
    (3, 64, 64) -> conv1 -> (16, 64, 64) -> pool -> (16, 32, 32)
    -> conv2 -> (32, 32, 32) -> pool -> (32, 16, 16)
    -> flatten -> 8192 -> fc1 -> 100 -> fc2 -> 10
"""

import torch
import torch.nn as nn


class CNNClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(32 * 16 * 16, 100)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    model = CNNClassifier()
    dummy = torch.randn(4, 3, 64, 64)  # batch of 4 RGB images
    out = model(dummy)
    print('input :', tuple(dummy.shape))
    print('output:', tuple(out.shape))
    assert out.shape == (4, 10), 'output shape mismatch'
    print('OK - architecture matches the spec')

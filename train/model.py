# Copyright (c) 2026 Eoin Jordan
# SPDX-License-Identifier: Apache-2.0
"""TinyDigitNet: a ~14k parameter CNN for 28x28 grayscale digits.

Kept deliberately small so the fp32 portable ExecuTorch kernels run it in a
few milliseconds on the ESP32-S3 (Xtensa LX7, 240 MHz) with no delegate.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# MNIST normalisation constants. The firmware applies the same transform to
# camera frames, so keep these in sync with firmware/src/main.cpp.
MEAN = 0.1307
STD = 0.3081

INPUT_SHAPE = (1, 1, 28, 28)
NUM_CLASSES = 10


class TinyDigitNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3)   # 28 -> 26
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)  # 13 -> 11
        self.fc1 = nn.Linear(16 * 5 * 5, 32)
        self.fc2 = nn.Linear(32, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)  # 26 -> 13
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # 11 -> 5
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# SPDX-FileCopyrightText: 2025 The Despair Authors
# SPDX-License-Identifier: MIT
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseEdgeDetector
from models.model_manager import ModelManager


# TEED Architecture (simplified version)
class DoubleFusion(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleFusion, self).__init__()
        self.DWconv1 = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch)
        self.PWconv1 = nn.Conv2d(in_ch, out_ch // 2, 1)
        self.DWconv2 = nn.Conv2d(in_ch, in_ch, 5, padding=2, groups=in_ch)
        self.PWconv2 = nn.Conv2d(in_ch, out_ch // 2, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.relu(self.PWconv1(self.DWconv1(x)))
        x2 = self.relu(self.PWconv2(self.DWconv2(x)))
        return torch.cat([x1, x2], dim=1)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class TEEDNet(nn.Module):
    """Simplified TEED architecture"""

    def __init__(self):
        super(TEEDNet, self).__init__()

        # Encoder
        self.conv1 = ConvBlock(3, 16)
        self.conv2 = ConvBlock(16, 32)
        self.conv3 = ConvBlock(32, 64)
        self.conv4 = ConvBlock(64, 128)

        # Fusion blocks
        self.fusion1 = DoubleFusion(32, 32)
        self.fusion2 = DoubleFusion(64, 64)
        self.fusion3 = DoubleFusion(128, 128)

        # Output layers
        self.out1 = nn.Conv2d(32, 1, 1)
        self.out2 = nn.Conv2d(64, 1, 1)
        self.out3 = nn.Conv2d(128, 1, 1)

    def forward(self, x):
        # Encoder path
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)

        # Apply fusion
        f2 = self.fusion1(x2)
        f3 = self.fusion2(x3)
        f4 = self.fusion3(x4)

        # Generate outputs at multiple scales
        out2 = self.out1(f2)
        out3 = self.out2(f3)
        out4 = self.out3(f4)

        # Upsample to original size
        out3 = F.interpolate(
            out3, size=out2.shape[2:], mode="bilinear", align_corners=False
        )
        out4 = F.interpolate(
            out4, size=out2.shape[2:], mode="bilinear", align_corners=False
        )

        # Combine outputs
        out = torch.sigmoid((out2 + out3 + out4) / 3)

        return out


class TEEDModel(BaseEdgeDetector):
    def __init__(self, device="cuda"):
        super().__init__(device)
        self.model_manager = ModelManager()

    def load_model(self):
        """Load TEED model"""
        # For this demo, we use a simplified TEED architecture
        # In production, load the actual TEED model from GitHub
        self.model = TEEDNet()
        self.model.to(self.device)
        self.model.eval()

        # Try to load pretrained weights if available
        checkpoint_path = "checkpoints/teed_simplified.pth"
        try:
            if Path(checkpoint_path).exists():
                state_dict = torch.load(
                    checkpoint_path, map_location=self.device
                )  # nosec B614
                self.model.load_state_dict(state_dict)
                logging.info("Loaded TEED checkpoint")
        except Exception:
            logging.info("Using random initialized TEED (demo mode)")

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for TEED"""
        # Convert to RGB if grayscale
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0

        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)

        return tensor

    def postprocess(self, output: torch.Tensor) -> np.ndarray:
        """Postprocess TEED output"""
        # Get edge map
        edge_map = output.squeeze().cpu().numpy()

        # Scale to [0, 255]
        edge_map = (edge_map * 255).astype(np.uint8)

        # Invert (white background, black lines)
        edge_map = 255 - edge_map

        return edge_map

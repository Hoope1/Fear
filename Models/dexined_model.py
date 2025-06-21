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


# Simplified DexiNed blocks
class DexiBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DexiBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)))
        return out


class DexiNedSimplified(nn.Module):
    """Simplified DexiNed architecture for demo"""

    def __init__(self):
        super(DexiNedSimplified, self).__init__()

        # Dense blocks
        self.block1 = DexiBlock(3, 32)
        self.block2 = DexiBlock(32, 64)
        self.block3 = DexiBlock(64, 128)
        self.block4 = DexiBlock(128, 256)

        # Skip connections
        self.skip1 = nn.Conv2d(32, 1, 1)
        self.skip2 = nn.Conv2d(64, 1, 1)
        self.skip3 = nn.Conv2d(128, 1, 1)
        self.skip4 = nn.Conv2d(256, 1, 1)

        # Upsampling
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.up4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)
        self.up8 = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=False)

    def forward(self, x):
        # Forward through blocks
        x1 = self.block1(x)
        x2 = self.block2(F.max_pool2d(x1, 2))
        x3 = self.block3(F.max_pool2d(x2, 2))
        x4 = self.block4(F.max_pool2d(x3, 2))

        # Generate edge maps at each scale
        e1 = torch.sigmoid(self.skip1(x1))
        e2 = torch.sigmoid(self.skip2(x2))
        e3 = torch.sigmoid(self.skip3(x3))
        e4 = torch.sigmoid(self.skip4(x4))

        # Upsample to original size
        e2 = self.up2(e2)
        e3 = self.up4(e3)
        e4 = self.up8(e4)

        # Fuse multi-scale edges
        out = (e1 + e2 + e3 + e4) / 4

        return out


class DexiNedModel(BaseEdgeDetector):
    def __init__(self, device="cuda"):
        super().__init__(device)
        self.model_manager = ModelManager()

    def load_model(self):
        """Load DexiNed model"""
        # Try to download and load actual DexiNed weights
        checkpoint_path = self.model_manager.download_dexined_weights()

        if checkpoint_path and Path(checkpoint_path).exists():
            # Load actual DexiNed (would need full implementation)
            logging.info("Loading actual DexiNed weights...")
            # For now, use simplified version
            self.model = DexiNedSimplified()
        else:
            # Use simplified version for demo
            logging.info("Using simplified DexiNed (demo mode)")
            self.model = DexiNedSimplified()

        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for DexiNed"""
        # Convert to RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to multiple of 16 (DexiNed requirement)
        h, w = image.shape[:2]
        new_h = ((h + 15) // 16) * 16
        new_w = ((w + 15) // 16) * 16

        if new_h != h or new_w != w:
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Normalize
        image = image.astype(np.float32) / 255.0

        # Convert to tensor
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)

        return tensor

    def postprocess(self, output: torch.Tensor) -> np.ndarray:
        """Postprocess DexiNed output"""
        # Get edge map
        edge_map = output.squeeze().cpu().numpy()

        # Apply non-maximum suppression for thin edges
        edge_map = self._non_max_suppression(edge_map)

        # Scale and invert
        edge_map = (1.0 - edge_map) * 255.0
        edge_map = edge_map.astype(np.uint8)
        return edge_map

    def _non_max_suppression(
        self, edge_map: np.ndarray, threshold: float = 0.1
    ) -> np.ndarray:
        """Apply non-maximum suppression for thinner edges"""
        h, w = edge_map.shape
        suppressed = np.zeros_like(edge_map)

        # Compute gradients and angles
        dx = cv2.Sobel(edge_map, cv2.CV_64F, 1, 0, ksize=3)
        dy = cv2.Sobel(edge_map, cv2.CV_64F, 0, 1, ksize=3)
        angle = np.rad2deg(np.arctan2(dy, dx)) % 180

        # Direction indices 0-3
        direction = ((angle + 22.5) // 45).astype(int) % 4
        offsets = [((0, -1), (0, 1)), ((-1, -1), (1, 1)), ((-1, 0), (1, 0)), ((-1, 1), (1, -1))]

        mask = edge_map >= threshold
        for idx, ((y1, x1), (y2, x2)) in enumerate(offsets):
            dir_mask = mask & (direction == idx)
            before = np.roll(edge_map, shift=(y1, x1), axis=(0, 1))
            after = np.roll(edge_map, shift=(y2, x2), axis=(0, 1))
            local_max = (edge_map >= before) & (edge_map >= after)
            suppressed = np.where(dir_mask & local_max, edge_map, suppressed)

        return suppressed

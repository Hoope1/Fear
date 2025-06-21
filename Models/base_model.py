# SPDX-FileCopyrightText: 2025 The Despair Authors
# SPDX-License-Identifier: MIT
from abc import ABC, abstractmethod

import numpy as np
import torch


class BaseEdgeDetector(ABC):
    """Base class for edge detection models"""

    def __init__(self, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None

    @abstractmethod
    def load_model(self):
        """Load the model architecture and weights"""
        pass

    @abstractmethod
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input"""
        pass

    @abstractmethod
    def postprocess(self, output: torch.Tensor) -> np.ndarray:
        """Postprocess model output to edge map"""
        pass

    def process_large_image(
        self, image: np.ndarray, tile_size: int = 1024, overlap: int = 128
    ) -> np.ndarray:
        """Process large images using tiling to avoid OOM"""
        if tile_size <= overlap:
            raise ValueError("tile_size must be larger than overlap")
        h, w = image.shape[:2]

        # If image is small enough, process directly
        if h <= tile_size and w <= tile_size:
            return self.process_image(image)

        stride = tile_size - overlap

        # Initialize output
        output = np.zeros((h, w), dtype=np.float32)
        weight_map = np.zeros((h, w), dtype=np.float32)

        # Process each tile
        y_start = 0
        while y_start < h:
            x_start = 0
            y_end = min(y_start + tile_size, h)
            while x_start < w:
                x_end = min(x_start + tile_size, w)

                # Extract and process tile
                tile = image[y_start:y_end, x_start:x_end]
                tile_output = self.process_image(tile)

                # Add to output with blending
                output[y_start:y_end, x_start:x_end] += tile_output
                weight_map[y_start:y_end, x_start:x_end] += 1
                x_start += stride
            y_start += stride

        output = output / np.maximum(weight_map, 1)

        return output

    def process_image(self, image: np.ndarray) -> np.ndarray:
        """Process a single image"""
        with torch.no_grad():
            # Preprocess
            input_tensor = self.preprocess(image)
            input_tensor = input_tensor.to(self.device)

            # Forward pass
            output = self.model(input_tensor)

            # Postprocess
            edge_map = self.postprocess(output)

        return edge_map

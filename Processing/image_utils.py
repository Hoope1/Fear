# SPDX-FileCopyrightText: 2025 The Despair Authors
# SPDX-License-Identifier: MIT
import logging
from pathlib import Path

import cv2
import numpy as np


def load_image(image_path):
    """Load image from file"""
    try:
        # Read image
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

        if image is None:
            logging.error("Failed to load image: %s", image_path)
            return None

        return image

    except Exception as e:
        logging.error("Error loading image %s: %s", image_path, e)
        return None


def save_edge_map(edge_map, output_path, original_size=None):
    """Save edge detection result"""
    try:
        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Resize back to original size if needed
        if original_size and (edge_map.shape[1], edge_map.shape[0]) != original_size:
            edge_map = cv2.resize(
                edge_map, original_size, interpolation=cv2.INTER_LINEAR
            )

        # Ensure proper format
        if edge_map.dtype != np.uint8 and edge_map.max() <= 1.0:
            edge_map = (edge_map * 255).astype(np.uint8)

        # Save image
        cv2.imwrite(str(output_path), edge_map)

    except Exception as e:
        logging.error("Error saving edge map: %s", e)

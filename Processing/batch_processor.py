# SPDX-FileCopyrightText: 2025 The Despair Authors
# SPDX-License-Identifier: MIT
import os
import traceback
from pathlib import Path

import torch
from PyQt6.QtCore import QThread, pyqtSignal

from models.dexined_model import DexiNedModel
from models.teed_model import TEEDModel
from processing.image_utils import load_image, save_edge_map


class BatchProcessor(QThread):
    """Process images in background thread"""

    progress_update = pyqtSignal(int)
    log_message = pyqtSignal(str)
    processing_complete = pyqtSignal()
    current_file_update = pyqtSignal(str)

    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = folder_path
        self.models = {}

    def run(self):
        try:
            # Initialize models
            self.log_message.emit("Initializing models...")
            self._initialize_models()

            # Get image files
            image_files = self._get_image_files()
            if not image_files:
                self.log_message.emit("No image files found in selected folder")
                self.processing_complete.emit()
                return

            self.log_message.emit(f"Found {len(image_files)} images to process")

            # Process each image
            total_steps = len(image_files) * len(self.models)
            current_step = 0

            for img_path in image_files:
                filename = os.path.basename(img_path)
                self.current_file_update.emit(filename)

                # Load image
                image = load_image(img_path)
                if image is None:
                    self.log_message.emit(f"Failed to load: {filename}")
                    continue

                self.log_message.emit(
                    f"Processing: {filename} ({image.shape[1]}x{image.shape[0]})"
                )

                # Process with each model
                for model_name, model in self.models.items():
                    try:
                        # Process image (handles large images automatically)
                        if max(image.shape[:2]) > 2048:
                            self.log_message.emit(
                                f"  Using tiled processing for {model_name}"
                            )
                            edge_map = model.process_large_image(
                                image, tile_size=1536, overlap=256
                            )
                        else:
                            edge_map = model.process_image(image)

                        # Save result
                        output_path = (
                            Path("output")
                            / model_name
                            / f"{Path(filename).stem}_edges.png"
                        )
                        save_edge_map(
                            edge_map,
                            output_path,
                            original_size=(image.shape[1], image.shape[0]),
                        )

                    except Exception as e:
                        self.log_message.emit(f"  Error with {model_name}: {str(e)}")
                        traceback.print_exc()

                    current_step += 1
                    progress = int((current_step / total_steps) * 100)
                    self.progress_update.emit(progress)

            self.progress_update.emit(100)
            self.processing_complete.emit()

        except Exception as e:
            self.log_message.emit(f"Processing error: {str(e)}")
            traceback.print_exc()
            self.processing_complete.emit()

    def _initialize_models(self):
        """Initialize edge detection models"""
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Initialize TEED
        try:
            self.log_message.emit("Loading TEED model...")
            teed = TEEDModel()
            teed.load_model()
            self.models["TEED"] = teed
            self.log_message.emit("✓ TEED model loaded")
        except Exception as e:
            self.log_message.emit(f"Failed to load TEED: {str(e)}")

        # Initialize DexiNed
        try:
            self.log_message.emit("Loading DexiNed model...")
            dexined = DexiNedModel()
            dexined.load_model()
            self.models["DexiNed"] = dexined
            self.log_message.emit("✓ DexiNed model loaded")
        except Exception as e:
            self.log_message.emit(f"Failed to load DexiNed: {str(e)}")

        if not self.models:
            raise Exception("No models could be loaded!")

    def _get_image_files(self):
        """Get list of image files in folder"""
        supported_formats = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        image_files = []

        for file in Path(self.folder_path).rglob("*"):
            if file.suffix.lower() in supported_formats:
                image_files.append(str(file))

        return sorted(image_files)

# SPDX-FileCopyrightText: 2025 The Despair Authors
# SPDX-License-Identifier: MIT
import logging
from pathlib import Path
from typing import Optional

try:
    import gdown
except Exception:  # pragma: no cover - optional dependency
    gdown = None


class ModelManager:
    """Manages model downloads and loading"""

    def __init__(self, cache_dir="checkpoints"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def _download_weights(self, url: str, filename: str) -> Optional[str]:
        """Download weights from *url* if needed."""
        output_path = self.cache_dir / filename

        if output_path.exists():
            logging.info("%s already downloaded", filename)
            return str(output_path)

        try:
            if gdown is None:
                raise ImportError("gdown not available")
            logging.info("Downloading %s...", filename)
            gdown.download(url, str(output_path), quiet=False)
            logging.info("%s downloaded successfully", filename)
            return str(output_path)
        except Exception as e:
            logging.error("Failed to download %s: %s", filename, e)
            return None

    def download_dexined_weights(self) -> Optional[str]:
        """Download DexiNed pretrained weights from Google Drive."""
        from config import DEXINED_URL

        return self._download_weights(DEXINED_URL, "dexined_checkpoint.pth")

    def download_teed_weights(self) -> Optional[str]:
        """Download TEED weights from Google Drive."""
        from config import TEED_URL

        return self._download_weights(TEED_URL, "teed_simplified.pth")

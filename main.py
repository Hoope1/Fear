# SPDX-FileCopyrightText: 2025 The Despair Authors
# SPDX-License-Identifier: MIT
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Fix for PyQt6 DLL issues on Windows
if sys.platform == "win32":
    # Add Qt plugin path
    plugin_path = os.path.join(
        os.path.dirname(sys.executable), "..", "Lib", "site-packages", "PyQt6", "Qt6", "plugins"
    )
    if os.path.exists(plugin_path):
        os.environ["QT_PLUGIN_PATH"] = plugin_path

    # Add Qt bin directory to PATH
    qt_bin_path = os.path.join(
        os.path.dirname(sys.executable), "..", "Lib", "site-packages", "PyQt6", "Qt6", "bin"
    )
    if os.path.exists(qt_bin_path):
        os.environ["PATH"] = qt_bin_path + os.pathsep + os.environ.get("PATH", "")

try:
    from PyQt6.QtCore import QCoreApplication, Qt
    from PyQt6.QtWidgets import QApplication

    # Set application attributes before creating QApplication
    QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)

    from gui.main_window import EdgeDetectionApp
except Exception as e:  # pragma: no cover - optional GUI support
    QApplication = None  # type: ignore
    EdgeDetectionApp = None  # type: ignore
    logging.error("ERROR: Failed to import PyQt6: %s", e)
    logging.error("Please ensure PyQt6 is properly installed:")
    logging.error("1. Run: setup_rye_windows.bat")
    logging.error("2. Or manually: rye run pip install --force-reinstall PyQt6==6.6.0")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # Create necessary directories
    Path("checkpoints").mkdir(exist_ok=True)
    Path("output").mkdir(exist_ok=True)
    Path("output/TEED").mkdir(parents=True, exist_ok=True)
    Path("output/DexiNed").mkdir(parents=True, exist_ok=True)

    # Check CUDA availability
    try:
        import torch

        if torch.cuda.is_available():
            logging.info("CUDA available: %s", torch.cuda.get_device_name(0))
            logging.info("CUDA version: %s", torch.version.cuda)
        else:
            logging.warning("CUDA not available, using CPU (will be slower)")
    except Exception as e:
        logging.warning("Could not check CUDA availability: %s", e)

    if QApplication is None or EdgeDetectionApp is None:
        logging.error("PyQt6 is not available. GUI cannot be started.")
        return

    # Start GUI application
    app = QApplication(sys.argv)

    # Set application metadata
    app.setApplicationName("Edge Detection App")
    app.setOrganizationName("Despair")

    window = EdgeDetectionApp()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

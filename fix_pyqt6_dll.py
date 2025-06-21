# SPDX-FileCopyrightText: 2025 The Despair Authors
# SPDX-License-Identifier: MIT
#!/usr/bin/env python
"""
Fix PyQt6 DLL issues on Windows
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path


def fix_pyqt6_windows() -> bool:
    """Fix PyQt6 DLL loading issues on Windows."""

    if platform.system() != "Windows":
        print("This script is only needed on Windows.")
        return False

    print("Fixing PyQt6 DLL issues on Windows...")
    print("=" * 60)

    # 1. Uninstall existing PyQt6
    print("\n1. Uninstalling existing PyQt6 packages...")
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "PyQt6", "PyQt6-Qt6", "PyQt6-sip"],
        capture_output=True,
    )

    # 2. Clear pip cache
    print("\n2. Clearing pip cache...")
    subprocess.run([sys.executable, "-m", "pip", "cache", "purge"], capture_output=True)

    # 3. Install Microsoft Visual C++ Redistributables check
    print("\n3. Checking Visual C++ Redistributables...")
    print("   NOTE: PyQt6 requires Microsoft Visual C++ Redistributables")
    print("   Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe")

    # 4. Reinstall PyQt6 with specific versions
    print("\n4. Installing PyQt6 with specific versions...")
    packages = ["PyQt6-sip==13.6.0", "PyQt6-Qt6==6.6.0", "PyQt6==6.6.0"]

    for package in packages:
        print(f"   Installing {package}...")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--force-reinstall",
                "--no-cache-dir",
                package,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"   ERROR: Failed to install {package}")
            print(f"   {result.stderr}")
        else:
            print(f"   ✓ {package} installed successfully")

    # 5. Test PyQt6 import
    print("\n5. Testing PyQt6 import...")
    try:
        from PyQt6.QtCore import PYQT_VERSION_STR, QT_VERSION_STR

        print("   ✓ PyQt6 imported successfully!")

        # Get Qt version

        print(f"   Qt version: {QT_VERSION_STR}")
        print(f"   PyQt version: {PYQT_VERSION_STR}")

    except ImportError as e:
        print(f"   ✗ PyQt6 import failed: {e}")
        print("\n   Possible solutions:")
        print("   1. Install Visual C++ Redistributables (link above)")
        print("   2. Try Python 3.10 or 3.11 (not 3.12)")
        print("   3. Use Anaconda/Miniconda and install from conda-forge:")
        print("      conda install -c conda-forge pyqt")
        return False

    print("\n" + "=" * 60)
    print("PyQt6 fix completed!")
    return True


def setup_environment_vars() -> None:
    """Setup environment variables for PyQt6."""
    print("\n6. Setting up environment variables...")

    # Find PyQt6 installation
    site_packages = Path(sys.executable).parent / "Lib" / "site-packages"
    pyqt6_path = site_packages / "PyQt6"

    if pyqt6_path.exists():
        qt_plugin_path = pyqt6_path / "Qt6" / "plugins"
        qt_bin_path = pyqt6_path / "Qt6" / "bin"

        if qt_plugin_path.exists():
            os.environ["QT_PLUGIN_PATH"] = str(qt_plugin_path)
            print(f"   Set QT_PLUGIN_PATH: {qt_plugin_path}")

        if qt_bin_path.exists():
            os.environ["PATH"] = str(qt_bin_path) + os.pathsep + os.environ.get("PATH", "")
            print(f"   Added to PATH: {qt_bin_path}")


def main() -> None:
    """Entry point for fixing PyQt6 DLL issues."""
    print("PyQt6 DLL Fix Script")
    print("=" * 60)

    # Check Python version
    print(f"Python version: {sys.version}")
    if sys.version_info >= (3, 12):
        print("WARNING: Python 3.12+ may have compatibility issues with PyQt6")
        print("Recommended: Python 3.10 or 3.11")

    # Run fix
    if fix_pyqt6_windows():
        setup_environment_vars()
        print("\n✓ PyQt6 should now work correctly!")
        print("\nYou can now run: python main.py")
    else:
        print("\n✗ PyQt6 fix failed. Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

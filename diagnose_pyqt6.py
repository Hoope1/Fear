# SPDX-FileCopyrightText: 2025 The Despair Authors
# SPDX-License-Identifier: MIT
#!/usr/bin/env python
"""
Diagnose PyQt6 installation issues on Windows
"""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path


def check_system_info() -> None:
    """Check system information."""
    print("System Information")
    print("=" * 60)
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Python: {sys.version}")
    print(f"Python Path: {sys.executable}")

    # Check if running in virtual environment
    in_venv = hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )
    print(f"Virtual Environment: {'Yes' if in_venv else 'No'}")

    if in_venv:
        print(f"Virtual Environment Path: {sys.prefix}")


def check_vcredist() -> None:
    """Check if Visual C++ Redistributables are installed."""
    print("\n\nVisual C++ Redistributables Check")
    print("=" * 60)

    # Common VC++ DLLs
    dlls_to_check = [
        "msvcp140.dll",
        "vcruntime140.dll",
        "vcruntime140_1.dll",
        "msvcp140_1.dll",
        "msvcp140_2.dll",
    ]

    system32 = Path(os.environ["SystemRoot"]) / "System32"
    missing_dlls = []

    for dll in dlls_to_check:
        dll_path = system32 / dll
        if dll_path.exists():
            print(f"✓ {dll} found")
        else:
            print(f"✗ {dll} NOT FOUND")
            missing_dlls.append(dll)

    if missing_dlls:
        print("\n⚠️  Missing Visual C++ Redistributables!")
        print("Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe")
    else:
        print("\n✓ All Visual C++ Redistributables found")


def main() -> None:
    """Run diagnostics for PyQt6 installation."""
    check_system_info()
    check_vcredist()


if __name__ == "__main__":
    main()

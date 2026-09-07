from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from skeleton_annotation.main_gui import main
else:
    from .main_gui import main

__all__ = ["main"]

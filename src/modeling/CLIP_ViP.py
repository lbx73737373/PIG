"""
Compatibility shim for legacy CLIP-ViP import path.

The refactored codebase keeps the canonical implementation in `src/modeling/PIG.py`
while preserving the original module name expected by baseline CLIP-ViP scripts.
"""

from src.modeling.PIG import *  # noqa: F401,F403


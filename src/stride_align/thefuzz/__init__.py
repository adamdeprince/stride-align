"""TheFuzz 0.22.1-compatible facade.

Replace ``from thefuzz import fuzz, process`` with
``from stride_align.thefuzz import fuzz, process``. Scoring work is
dispatched to stride-align's native kernels; TheFuzz's integer rounding,
legacy preprocessing, extraction tuple shapes, and aliases are retained.
"""

from __future__ import annotations

from stride_align.thefuzz import fuzz, process, utils

# This describes the upstream API/behaviour target, not the stride-align
# distribution version. TheFuzz exposes the same value at package level.
__version__ = "0.22.1"

__all__ = ["fuzz", "process", "utils"]

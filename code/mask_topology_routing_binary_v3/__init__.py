"""Binary-mask topology routing v3.

The package deliberately wraps the frozen v2 geometry implementation instead
of modifying it.  V3 adds rulebook-derived post-clip repair, trunk refinement,
and an explicit topology acceptance gate.
"""

from .config import BinaryV3Config, load_binary_v3_config
from .refinement import refine_clipped_groups_v3
from .quality import audit_binary_topology_v3

__all__ = [
    "BinaryV3Config",
    "load_binary_v3_config",
    "refine_clipped_groups_v3",
    "audit_binary_topology_v3",
]

"""
QOR Mamba — Backward Compatibility Wrapper
============================================
This module has been renamed to qor.cortex (CORTEX architecture).
All imports are forwarded for backward compatibility.

Use qor.cortex directly for new code:
    from qor.cortex import CortexBrain, S4Block
"""

# Re-export everything from cortex for backward compatibility
from qor.cortex import (  # noqa: F401
    S4Block,
    CortexBrain,
    CortexBrain as MambaCfCHybrid,
    _HAS_CORTEX as _HAS_CFC,
)

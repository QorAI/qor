"""
QOR — The Qore Mind
====================
A learning AI architecture with multi-speed memory,
self-modifying neurons, and surprise-driven learning.

Built on the Nested Learning paradigm (NeurIPS 2025).
"""

from .config import QORConfig
from .model import QORModel
from .tokenizer import QORTokenizer

__version__ = "1.0.0"
__all__ = ["QORConfig", "QORModel", "QORTokenizer"]

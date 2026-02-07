"""
QOR — Qora Neuran AI Project
==============================
A learning AI architecture with multi-speed memory,
self-modifying neurons, and surprise-driven learning.

By Ravikash Gupta | QOR Research in (AGI)

Quick Start:
    from qor.setup import create_agent
    agent = create_agent("checkpoints/best_model.pt")
    result = agent.answer("What's the price of Bitcoin?")
"""

__version__ = "1.0.0"

# Lazy imports — torch-dependent modules only load when needed
def __getattr__(name):
    if name == "QORConfig":
        from .config import QORConfig
        return QORConfig
    if name == "QORModel":
        from .model import QORModel
        return QORModel
    if name == "QORTokenizer":
        from .tokenizer import QORTokenizer
        return QORTokenizer
    if name == "create_agent":
        from .setup import create_agent
        return create_agent
    if name == "add_knowledge":
        from .setup import add_knowledge
        return add_knowledge
    if name == "add_tool":
        from .setup import add_tool
        return add_tool
    if name == "agent_stats":
        from .setup import agent_stats
        return agent_stats
    raise AttributeError(f"module 'qor' has no attribute {name}")

__all__ = [
    "QORConfig", "QORModel", "QORTokenizer",
    "create_agent", "add_knowledge", "add_tool", "agent_stats",
]

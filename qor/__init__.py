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
        from .plugins import add_tool
        return add_tool
    if name == "add_api_tool":
        from .plugins import add_api_tool
        return add_api_tool
    if name == "remove_tool":
        from .plugins import remove_tool
        return remove_tool
    if name == "agent_stats":
        from .setup import agent_stats
        return agent_stats
    if name == "QORRuntime":
        from .runtime import QORRuntime
        return QORRuntime
    if name == "RuntimeConfig":
        from .config import RuntimeConfig
        return RuntimeConfig
    if name == "ExchangeKeys":
        from .config import ExchangeKeys
        return ExchangeKeys
    if name == "EXCHANGE_DEFAULTS":
        from .config import EXCHANGE_DEFAULTS
        return EXCHANGE_DEFAULTS
    if name == "QORGraph":
        from .graph import QORGraph
        return QORGraph
    if name == "NodeType":
        from .graph import NodeType
        return NodeType
    if name == "NodeFlags":
        from .graph import NodeFlags
        return NodeFlags
    if name == "resolve_node_type":
        from .graph import resolve_node_type
        return resolve_node_type
    if name == "VisionEncoder":
        from .vision import VisionEncoder
        return VisionEncoder
    if name == "AudioEncoder":
        from .audio import AudioEncoder
        return AudioEncoder
    if name == "VisionConfig":
        from .config import VisionConfig
        return VisionConfig
    if name == "AudioConfig":
        from .config import AudioConfig
        return AudioConfig
    if name == "HashChain":
        from .integrity import HashChain
        return HashChain
    if name == "CacheStore":
        from .cache import CacheStore
        return CacheStore
    if name == "ChatStore":
        from .chat import ChatStore
        return ChatStore
    if name == "QORCrypto":
        from .crypto import QORCrypto
        return QORCrypto
    if name == "SkillLoader":
        from .skills import SkillLoader
        return SkillLoader
    if name == "BrowserEngine":
        from .browser import BrowserEngine
        return BrowserEngine
    if name == "MarketHMM":
        from .quant import MarketHMM
        return MarketHMM
    if name == "QuantMetrics":
        from .quant import QuantMetrics
        return QuantMetrics
    if name == "S4Block":
        from .cortex import S4Block
        return S4Block
    if name == "CortexBrain":
        from .cortex import CortexBrain
        return CortexBrain
    if name == "MambaCfCHybrid":  # backward compat
        from .cortex import CortexBrain
        return CortexBrain
    if name == "KnowledgeTree":
        from . import knowledge_tree
        return knowledge_tree
    if name == "FeedbackDetector":
        from .knowledge_tree import FeedbackDetector
        return FeedbackDetector
    if name == "AnswerFilter":
        from .knowledge_tree import AnswerFilter
        return AnswerFilter
    if name == "TradeLearner":
        from .knowledge_tree import TradeLearner
        return TradeLearner
    if name == "tree_search":
        from .knowledge_tree import tree_search
        return tree_search
    if name == "QSearchEngine":
        from .qsearch import QSearchEngine
        return QSearchEngine
    if name == "QuantumState":
        from .qsearch import QuantumState
        return QuantumState
    if name == "QuantumRegister":
        from .qsearch import QuantumRegister
        return QuantumRegister
    if name == "knowledge_search":
        from .qsearch import knowledge_search
        return knowledge_search
    if name == "NGREBrain":
        from .ngre import NGREBrain
        return NGREBrain
    if name == "MambaTemporalModule":
        from .ngre import MambaTemporalModule
        return MambaTemporalModule
    if name == "InterferenceSearch":
        from .ngre import InterferenceSearch
        return InterferenceSearch
    if name == "OracleNetwork":
        from .ngre import OracleNetwork
        return OracleNetwork
    if name == "create_ngre_brain":
        from .ngre import create_ngre_brain
        return create_ngre_brain
    if name == "NGREConfig":
        from .config import NGREConfig
        return NGREConfig
    if name == "TreeGate":
        from .ngre import TreeGate
        return TreeGate
    if name == "ComplexityGate":
        from .ngre import ComplexityGate
        return ComplexityGate
    if name == "ReasoningLayer":
        from .ngre import ReasoningLayer
        return ReasoningLayer
    if name == "UpstoxClient":
        from .upstox import UpstoxClient
        return UpstoxClient
    if name == "MARKET_CATEGORIES":
        from .config import MARKET_CATEGORIES
        return MARKET_CATEGORIES
    raise AttributeError(f"module 'qor' has no attribute {name}")

__all__ = [
    "QORConfig", "QORModel", "QORTokenizer",
    "create_agent", "add_knowledge", "add_tool", "add_api_tool",
    "remove_tool", "agent_stats", "QORGraph", "QORRuntime", "RuntimeConfig",
    "VisionEncoder", "AudioEncoder", "VisionConfig", "AudioConfig",
    "HashChain", "CacheStore", "ChatStore", "QORCrypto", "SkillLoader",
    "BrowserEngine", "MarketHMM", "QuantMetrics", "S4Block", "CortexBrain", "MambaCfCHybrid",
    "KnowledgeTree", "FeedbackDetector", "AnswerFilter", "TradeLearner",
    "tree_search",
    "QSearchEngine", "QuantumState", "QuantumRegister", "knowledge_search",
    "NGREBrain", "MambaTemporalModule", "InterferenceSearch", "OracleNetwork",
    "create_ngre_brain", "NGREConfig", "TreeGate", "ComplexityGate",
    "ReasoningLayer", "UpstoxClient", "MARKET_CATEGORIES",
]

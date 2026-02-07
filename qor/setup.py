"""
QOR Setup — Wire Everything Together
======================================
This is the ONE file that connects:
  - Knowledge Base (psychology + your documents)
  - 30+ Tools (crypto, weather, news, code, etc.)
  - RAG (document retrieval)
  - Confidence Gate (zero-hallucination routing)

Usage:
    from qor.setup import create_agent

    agent = create_agent(
        model_path="checkpoints/best_model.pt",
        knowledge_dir="knowledge/",      # your .txt/.md files
        memory_path="memory.json",        # persistent memory
    )

    # Chat with zero hallucination
    result = agent.answer("What's the price of Bitcoin?")
    print(result["answer"])  # → fresh price from CoinGecko
    print(result["source"])  # → "tool:price"

    # Or interactive
    agent.interactive_chat()
"""

import os
import torch
from typing import Optional

from qor.config import QORConfig
from qor.model import QORModel
from qor.tokenizer import QORTokenizer
from qor.confidence import ConfidenceGate
from qor.rag import QORRag
from qor.knowledge import KnowledgeBase
from qor.tools import ToolExecutor


def create_agent(
    model_path: str = "checkpoints/best_model.pt",
    tokenizer_path: str = "tokenizer.json",
    knowledge_dir: str = "knowledge",
    memory_path: str = "memory.json",
    config_size: str = "small",
    device: str = "auto",
    verbose: bool = True,
) -> ConfidenceGate:
    """
    Create a fully-wired QOR agent with:
      - Trained model (or fresh one if no checkpoint)
      - Knowledge base (psychology + your documents)
      - 30+ external tools (crypto, weather, news, etc.)
      - RAG retrieval over your documents
      - Confidence gate (zero-hallucination routing)
      - Persistent memory

    Args:
        model_path:     Path to trained model checkpoint
        tokenizer_path: Path to tokenizer.json
        knowledge_dir:  Folder with .txt/.md knowledge files
        memory_path:    Where to save persistent memory
        config_size:    "small" (5M), "medium" (30M), or "large" (100M)
        device:         "auto", "cpu", or "cuda"
        verbose:        Print setup progress

    Returns:
        ConfidenceGate ready to answer questions
    """
    if verbose:
        print("=" * 60)
        print("  QOR Agent Setup — Qora Neuran AI Project")
        print("=" * 60)

    # ------------------------------------------------------------------
    # 1. DEVICE
    # ------------------------------------------------------------------
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        print(f"\n  Device: {device}")

    # ------------------------------------------------------------------
    # 2. CONFIG
    # ------------------------------------------------------------------
    config_map = {"small": QORConfig.small, "medium": QORConfig.medium,
                  "large": QORConfig.large}
    config = config_map.get(config_size, QORConfig.small)()
    if verbose:
        print(f"  Config: {config_size} ({config.model.d_model}d, "
              f"{config.model.n_layers}L)")

    # ------------------------------------------------------------------
    # 3. TOKENIZER
    # ------------------------------------------------------------------
    tokenizer = QORTokenizer()
    if os.path.exists(tokenizer_path):
        tokenizer.load(tokenizer_path)
        if verbose:
            print(f"  Tokenizer: loaded ({tokenizer.vocab_size} tokens)")
    else:
        if verbose:
            print(f"  Tokenizer: using default (train one with qor.prepare_data)")
    config.model.vocab_size = tokenizer.vocab_size

    # ------------------------------------------------------------------
    # 4. MODEL
    # ------------------------------------------------------------------
    model = QORModel(config.model)

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device,
                                weights_only=False)
        if "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
        else:
            model.load_state_dict(checkpoint)
        if verbose:
            params = sum(p.numel() for p in model.parameters())
            print(f"  Model: loaded from {model_path} ({params:,} params)")
    else:
        if verbose:
            params = sum(p.numel() for p in model.parameters())
            print(f"  Model: fresh ({params:,} params) — train with qor.train")

    model = model.to(device)
    model.eval()

    # ------------------------------------------------------------------
    # 5. CONFIDENCE GATE (the brain)
    # ------------------------------------------------------------------
    gate = ConfidenceGate(model, tokenizer)
    gate.memory.path = memory_path
    if verbose:
        print(f"  Confidence Gate: initialized")
        print(f"    Thresholds: high={gate.high_confidence}, "
              f"low={gate.confidence_threshold}, "
              f"hallucination={gate.hallucination_threshold}")

    # ------------------------------------------------------------------
    # 6. KNOWLEDGE BASE (psychology + your documents)
    # ------------------------------------------------------------------
    kb = KnowledgeBase(knowledge_dir)
    kb.load()

    # Connect KB to gate — enhanced context for every query
    _connect_knowledge(gate, kb)

    if verbose:
        print(f"  Knowledge: {len(kb.nodes)} nodes loaded")

    # ------------------------------------------------------------------
    # 7. RAG (document retrieval)
    # ------------------------------------------------------------------
    rag = QORRag()
    if os.path.isdir(knowledge_dir):
        rag.add_folder(knowledge_dir)
        gate.set_rag(rag)
        if verbose:
            print(f"  RAG: connected ({len(rag.store.chunks)} chunks)")
    else:
        if verbose:
            print(f"  RAG: no documents (add .txt files to {knowledge_dir}/)")

    # ------------------------------------------------------------------
    # 8. TOOLS (30+ free APIs)
    # ------------------------------------------------------------------
    tool_executor = ToolExecutor()
    tool_executor.register_all(gate)

    # Store executor on gate for direct access
    gate._tool_executor = tool_executor
    gate._knowledge_base = kb

    if verbose:
        print(f"  Tools: {len(tool_executor.tools)} registered")
        print(f"\n  Memory: {len(gate.memory.entries)} entries "
              f"(from {memory_path})")

    # ------------------------------------------------------------------
    # 9. ENHANCED ANSWER (knowledge-augmented)
    # ------------------------------------------------------------------
    _enhance_answer(gate, kb, tool_executor)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  ✓ Agent ready! Use gate.answer() or gate.interactive_chat()")
        print(f"{'=' * 60}\n")

    return gate


def _connect_knowledge(gate: ConfidenceGate, kb: KnowledgeBase):
    """Connect knowledge base to the confidence gate's memory."""
    # Pre-load psychology knowledge into memory for fast access
    for topic, content in kb.nodes.items():
        if topic.startswith("builtin:"):
            # Don't store full text — just index it
            short = content.content[:200]
            gate.memory.store(
                key=f"kb:{topic}",
                content=short,
                source="knowledge_base",
                category="static",
                confidence=0.9,
            )


def _enhance_answer(gate: ConfidenceGate, kb: KnowledgeBase,
                    tools: ToolExecutor):
    """
    Wrap gate.answer() to add knowledge + smart tool routing.

    Before the model answers, we:
    1. Search knowledge base for relevant context
    2. Check if a specific tool should handle this
    3. Get psychology guidance if relevant
    4. Feed everything to the confidence gate
    """
    original_answer = gate.answer

    def enhanced_answer(question: str, max_new_tokens: int = 200,
                        temperature: float = 0.7, verbose: bool = True) -> dict:
        # A) Smart tool detection — check if a specific tool should answer
        detected_tool = tools.detect_intent(question)
        if detected_tool and _is_tool_query(question):
            if verbose:
                print(f"  ├─ Tool detected: {detected_tool}")
            try:
                tool_result = tools.call(detected_tool, question)
                if tool_result and not tool_result.startswith("[") and \
                   "not found" not in tool_result.lower():
                    # Build answer using tool result
                    prompt = (f"Based on this data:\n{tool_result}\n\n"
                              f"Question: {question}\nAnswer:")
                    answer_text = gate._generate(prompt, max_new_tokens,
                                                 temperature)

                    # Cache in memory
                    gate.memory.store(
                        key=f"tool:{detected_tool}:{question[:40]}",
                        content=tool_result[:500],
                        source=f"tool:{detected_tool}",
                        category="live",
                        confidence=0.95,
                    )

                    return {
                        "question": question,
                        "answer": answer_text,
                        "confidence": 0.95,
                        "source": f"tool:{detected_tool}",
                        "reasoning": f"Answered using {detected_tool} tool",
                        "learned": True,
                        "tool_data": tool_result,
                    }
            except Exception as e:
                if verbose:
                    print(f"  ├─ Tool error: {e}, falling back...")

        # B) Knowledge context — search for relevant docs
        kb_context = kb.get_context(question, max_chars=1000)
        if kb_context:
            # Inject knowledge into memory so gate can find it
            gate.memory.store(
                key=f"context:{question[:40]}",
                content=kb_context[:500],
                source="knowledge_base",
                category="static",
                confidence=0.85,
            )

        # C) Psychology context (for sensitive queries)
        psych = kb.get_psychology_context(question)
        core_node = kb.nodes.get("builtin:core_principles")
        core_len = len(core_node.content) if core_node else 0
        if psych and len(psych) > core_len:
            gate.memory.store(
                key=f"psych:{question[:40]}",
                content=psych[:300],
                source="psychology_knowledge",
                category="static",
                confidence=0.9,
            )

        # D) Fall through to normal confidence gate routing
        return original_answer(question, max_new_tokens=max_new_tokens,
                               temperature=temperature, verbose=verbose)

    gate.answer = enhanced_answer


def _is_tool_query(question: str) -> bool:
    """Check if this question is asking for live/external data."""
    q = question.lower()
    tool_indicators = [
        "price", "weather", "news", "search", "github", "reddit",
        "joke", "trivia", "recipe", "define", "calculate", "time",
        "what time", "convert", "how much", "stock", "crypto",
        "bitcoin", "btc", "eth", "forecast", "hacker news",
        "arxiv", "pypi", "npm", "book", "nasa", "country",
        "ip address", "run code", "execute",
    ]
    return any(indicator in q for indicator in tool_indicators)


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def add_knowledge(gate: ConfidenceGate, text: str, title: str = "custom"):
    """Add knowledge to a running agent."""
    if hasattr(gate, '_knowledge_base'):
        gate._knowledge_base.add_text(text, title=title)
        print(f"  ✓ Added knowledge: {title}")
    if gate.rag:
        gate.rag.add_text(text, source=title)
        print(f"  ✓ Added to RAG: {title}")


def add_knowledge_file(gate: ConfidenceGate, path: str):
    """Add a file to a running agent's knowledge."""
    if hasattr(gate, '_knowledge_base'):
        gate._knowledge_base.add_file(path)
    if gate.rag:
        gate.rag.add_file(path)
    print(f"  ✓ Added file: {path}")


def add_knowledge_folder(gate: ConfidenceGate, folder: str):
    """Add all files from a folder to a running agent."""
    if hasattr(gate, '_knowledge_base'):
        gate._knowledge_base.add_folder(folder)
    if gate.rag:
        gate.rag.add_folder(folder)
    print(f"  ✓ Added folder: {folder}")


def add_tool(gate: ConfidenceGate, name: str, description: str,
             handler, categories: list = None):
    """Add a custom tool to a running agent."""
    gate.tools.register(name, description, handler, categories or ["general"])
    if hasattr(gate, '_tool_executor'):
        gate._tool_executor.tools[name] = (handler, description,
                                            categories or ["general"])
    print(f"  ✓ Added tool: {name}")


def agent_stats(gate: ConfidenceGate):
    """Print agent statistics."""
    print("\n  QOR Agent Stats:")
    print(f"  ─────────────────────────────")
    print(f"    Memory entries:  {len(gate.memory.entries)}")
    print(f"    Tools available: {len(gate.tools.tools)}")
    print(f"    RAG connected:   {gate.rag is not None}")
    if hasattr(gate, '_knowledge_base'):
        print(f"    Knowledge nodes: {len(gate._knowledge_base.nodes)}")
    if hasattr(gate, '_tool_executor'):
        print(f"    Tool executor:   {len(gate._tool_executor.tools)} tools")
    print()


# ==============================================================================
# QUICK START
# ==============================================================================

if __name__ == "__main__":
    """
    Quick start — run this file directly:
        python -m qor.setup
    """
    agent = create_agent()

    print("Commands: quit, memory, tools, stats, knowledge")
    print()

    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not q:
            continue
        if q == "quit":
            break
        if q == "stats":
            agent_stats(agent)
            continue
        if q == "memory":
            agent.memory.stats()
            continue
        if q == "tools":
            for t in agent.tools.list_tools():
                print(f"  {t['name']}: {t['description']}")
            continue
        if q == "knowledge":
            if hasattr(agent, '_knowledge_base'):
                agent._knowledge_base.stats()
            continue

        result = agent.answer(q)
        print(f"\nQOR: {result['answer']}")
        bar = "█" * int(result['confidence'] * 10) + \
              "░" * (10 - int(result['confidence'] * 10))
        print(f"  [{bar}] {result['source']}\n")

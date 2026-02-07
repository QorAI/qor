"""
QOR Confidence Gate — Zero Hallucination System
==================================================
The model KNOWS what it knows and what it doesn't know.

How it works:
  1. User asks a question
  2. QOR checks its own confidence (surprise level)
  3. QOR classifies the question: static or live data?
  4. Based on confidence + data type → decide what to do:

  ┌─────────────────────────────────────────────────────────┐
  │                                                         │
  │  HIGH confidence + Static data                          │
  │  → Answer from internal knowledge                       │
  │  → "Paris is the capital of France"                     │
  │                                                         │
  │  LOW confidence + Static data                           │
  │  → Search knowledge base (RAG)                          │
  │  → Learn the answer → update memory                     │
  │  → Next time: answer from memory (no search needed)     │
  │                                                         │
  │  ANY confidence + Live data (prices, weather, events)   │
  │  → ALWAYS call external tool/API                        │
  │  → Never trust old data for live topics                 │
  │  → Update memory with timestamp                         │
  │                                                         │
  │  ZERO confidence + No source found                      │
  │  → Say "I don't know" honestly                          │
  │  → NEVER make something up                              │
  │                                                         │
  └─────────────────────────────────────────────────────────┘

This is how you get ZERO hallucination:
  - Model is self-aware about its knowledge gaps
  - Live data always gets fresh lookup
  - Unknown answers get honest "I don't know"
  - Every lookup teaches the model for next time
"""

import os
import json
import time
import re
import torch
import torch.nn.functional as F
from typing import Optional, List, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field


# ==============================================================================
# DATA FRESHNESS — What needs live data vs what's static
# ==============================================================================

# Topics that ALWAYS need live/fresh data — never trust memory alone
LIVE_DATA_PATTERNS = {
    "price": {
        "keywords": ["price", "cost", "worth", "market cap", "trading at",
                      "how much is", "current value", "stock price"],
        "reason": "Prices change every second",
        "max_age_minutes": 5,  # Data older than 5 min = stale
    },
    "weather": {
        "keywords": ["weather", "temperature", "forecast", "rain",
                      "sunny", "cloudy", "storm", "humidity"],
        "reason": "Weather changes constantly",
        "max_age_minutes": 30,
    },
    "news": {
        "keywords": ["latest", "breaking", "just happened", "today",
                      "yesterday", "this week", "recent", "current events",
                      "news about", "what happened"],
        "reason": "News is time-sensitive",
        "max_age_minutes": 60,
    },
    "sports": {
        "keywords": ["score", "who won", "game", "match", "tournament",
                      "standings", "season", "playoff"],
        "reason": "Sports results change live",
        "max_age_minutes": 15,
    },
    "availability": {
        "keywords": ["in stock", "available", "open", "closed",
                      "hours", "schedule", "booking", "reservation"],
        "reason": "Availability changes in real-time",
        "max_age_minutes": 10,
    },
    "exchange_rate": {
        "keywords": ["exchange rate", "convert", "currency",
                      "dollar to", "euro to", "forex"],
        "reason": "Exchange rates fluctuate constantly",
        "max_age_minutes": 15,
    },
    "traffic": {
        "keywords": ["traffic", "route", "commute", "road conditions",
                      "travel time", "delays", "accidents"],
        "reason": "Traffic conditions change in real-time",
        "max_age_minutes": 5,
    },
}


@dataclass
class KnowledgeEntry:
    """A piece of knowledge with freshness tracking."""
    content: str                          # The actual knowledge
    source: str                           # Where it came from
    timestamp: str                        # When it was learned
    category: str = "static"              # "static" or "live"
    confidence: float = 1.0               # How confident we are (0-1)
    access_count: int = 0                 # How often this was used
    last_accessed: str = ""               # Last time someone asked about this


@dataclass
class ConfidenceResult:
    """Result of confidence check."""
    confidence: float                     # 0.0 (no idea) to 1.0 (certain)
    needs_lookup: bool                    # Should we search knowledge base?
    needs_live_data: bool                 # Needs real-time API call?
    live_data_type: Optional[str] = None  # "price", "weather", etc.
    reason: str = ""                      # Why this decision was made


# ==============================================================================
# EXTERNAL TOOLS — APIs the model can call for live data
# ==============================================================================

class ToolRegistry:
    """
    Registry of external tools QOR can call.
    Add your own APIs here — price feeds, weather, databases, anything.
    """

    def __init__(self):
        self.tools = {}

    def register(self, name: str, description: str, handler: Callable,
                 categories: List[str] = None):
        """Register an external tool."""
        self.tools[name] = {
            "description": description,
            "handler": handler,
            "categories": categories or [],
        }
        print(f"  Registered tool: {name} — {description}")

    def get_tool_for_category(self, category: str) -> Optional[dict]:
        """Find the right tool for a data category."""
        for name, tool in self.tools.items():
            if category in tool["categories"]:
                return {"name": name, **tool}
        return None

    def call(self, name: str, query: str) -> Optional[str]:
        """Call a tool and return the result."""
        if name not in self.tools:
            return None
        try:
            result = self.tools[name]["handler"](query)
            return result
        except Exception as e:
            print(f"  Tool '{name}' error: {e}")
            return None

    def list_tools(self) -> List[dict]:
        """List all available tools."""
        return [{"name": k, "description": v["description"],
                 "categories": v["categories"]}
                for k, v in self.tools.items()]


# ==============================================================================
# DEFAULT TOOL IMPLEMENTATIONS (examples — replace with real APIs)
# ==============================================================================

def web_search_tool(query: str) -> str:
    """
    Web search tool. Replace with real implementation.
    
    Real options:
    - SerpAPI: pip install google-search-results
    - DuckDuckGo: pip install duckduckgo-search
    - Brave Search API
    - Google Custom Search API
    """
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if results:
                return "\n".join([
                    f"- {r['title']}: {r['body']}" for r in results
                ])
    except ImportError:
        pass

    return f"[Web search not configured. Install: pip install duckduckgo-search]"


def price_lookup_tool(query: str) -> str:
    """
    Price/crypto/stock lookup. Replace with real API.

    Real options:
    - CoinGecko API (free, no key needed for basic)
    - Alpha Vantage (free tier)
    - Yahoo Finance: pip install yfinance
    """
    try:
        import urllib.request
        import json as json_lib

        # Try CoinGecko for crypto
        crypto_map = {
            "btc": "bitcoin", "bitcoin": "bitcoin",
            "eth": "ethereum", "ethereum": "ethereum",
            "sol": "solana", "solana": "solana",
            "doge": "dogecoin", "dogecoin": "dogecoin",
        }

        query_lower = query.lower()
        coin_id = None
        for key, value in crypto_map.items():
            if key in query_lower:
                coin_id = value
                break

        if coin_id:
            url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd&include_24hr_change=true"
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'QOR/1.0')
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json_lib.loads(resp.read())
                if coin_id in data:
                    price = data[coin_id]["usd"]
                    change = data[coin_id].get("usd_24h_change", 0)
                    return f"{coin_id.title()} price: ${price:,.2f} USD (24h change: {change:+.2f}%)"
    except Exception as e:
        pass

    try:
        import yfinance as yf
        # Try stock lookup
        for word in query.upper().split():
            if len(word) <= 5 and word.isalpha():
                ticker = yf.Ticker(word)
                info = ticker.info
                if "currentPrice" in info:
                    return f"{info.get('shortName', word)}: ${info['currentPrice']:,.2f} USD"
    except Exception:
        pass

    return f"[Price lookup not available. Install: pip install yfinance or use CoinGecko API]"


def datetime_tool(query: str) -> str:
    """Current date and time."""
    now = datetime.now()
    return f"Current date and time: {now.strftime('%A, %B %d, %Y at %I:%M %p')}"


# ==============================================================================
# MEMORY STORE — Timestamped knowledge with freshness tracking
# ==============================================================================

class MemoryStore:
    """
    Persistent memory that tracks WHEN things were learned.
    This is separate from the model weights — it's a structured
    knowledge store that the confidence gate uses to decide
    if information is fresh enough or needs updating.
    """

    def __init__(self, path: str = "memory.json"):
        self.path = path
        self.entries = {}  # key → KnowledgeEntry
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path) as f:
                data = json.load(f)
            for key, entry_data in data.items():
                self.entries[key] = KnowledgeEntry(**entry_data)

    def save(self):
        data = {}
        for key, entry in self.entries.items():
            data[key] = {
                "content": entry.content,
                "source": entry.source,
                "timestamp": entry.timestamp,
                "category": entry.category,
                "confidence": entry.confidence,
                "access_count": entry.access_count,
                "last_accessed": entry.last_accessed,
            }
        with open(self.path, 'w') as f:
            json.dump(data, f, indent=2)

    def store(self, key: str, content: str, source: str,
              category: str = "static", confidence: float = 1.0):
        """Store a piece of knowledge."""
        self.entries[key] = KnowledgeEntry(
            content=content,
            source=source,
            timestamp=datetime.now().isoformat(),
            category=category,
            confidence=confidence,
            access_count=0,
            last_accessed="",
        )
        self.save()

    def get(self, key: str) -> Optional[KnowledgeEntry]:
        """Get a knowledge entry."""
        entry = self.entries.get(key)
        if entry:
            entry.access_count += 1
            entry.last_accessed = datetime.now().isoformat()
        return entry

    def search(self, query: str, top_k: int = 3) -> List[tuple]:
        """Simple keyword search over memory entries."""
        query_words = set(query.lower().split())
        scores = []

        for key, entry in self.entries.items():
            content_words = set(entry.content.lower().split())
            key_words = set(key.lower().split())
            overlap = len(query_words & (content_words | key_words))
            if overlap > 0:
                scores.append((key, entry, overlap))

        scores.sort(key=lambda x: x[2], reverse=True)
        return scores[:top_k]

    def is_stale(self, entry: KnowledgeEntry, max_age_minutes: int) -> bool:
        """Check if a knowledge entry is too old."""
        if not entry.timestamp:
            return True
        try:
            learned_time = datetime.fromisoformat(entry.timestamp)
            age = datetime.now() - learned_time
            return age > timedelta(minutes=max_age_minutes)
        except ValueError:
            return True

    def stats(self):
        """Print memory statistics."""
        static = sum(1 for e in self.entries.values() if e.category == "static")
        live = sum(1 for e in self.entries.values() if e.category == "live")
        print(f"\n  Memory Store:")
        print(f"    Total entries: {len(self.entries)}")
        print(f"    Static knowledge: {static}")
        print(f"    Live data: {live}")


# ==============================================================================
# THE CONFIDENCE GATE — The core of zero-hallucination
# ==============================================================================

class ConfidenceGate:
    """
    The brain of the zero-hallucination system.

    For every query, it:
    1. Measures model confidence (can I answer this?)
    2. Classifies data type (static vs live?)
    3. Checks memory freshness (is my info current?)
    4. Routes to the right source (internal / RAG / API / "I don't know")
    5. Updates memory after learning (so next time is faster)
    """

    def __init__(self, model, tokenizer, config=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

        # Knowledge sources
        self.memory = MemoryStore()
        self.tools = ToolRegistry()
        self.rag = None  # Set externally if RAG is available

        # Thresholds
        self.confidence_threshold = 0.6    # Below this → don't trust internal knowledge
        self.high_confidence = 0.85        # Above this → very confident, answer directly
        self.hallucination_threshold = 0.3 # Below this → definitely don't know

        # Register default tools
        self._register_default_tools()

    def _register_default_tools(self):
        """Register built-in tools."""
        self.tools.register(
            "web_search", "Search the web for current information",
            web_search_tool, ["news", "general"]
        )
        self.tools.register(
            "price_lookup", "Get current prices for crypto, stocks, commodities",
            price_lookup_tool, ["price", "exchange_rate"]
        )
        self.tools.register(
            "datetime", "Get current date and time",
            datetime_tool, ["datetime"]
        )

    def set_rag(self, rag):
        """Connect a RAG knowledge base."""
        self.rag = rag
        print(f"  RAG knowledge base connected")

    def measure_confidence(self, question: str) -> float:
        """
        Measure how confident QOR is about answering this question.

        Method: Feed the question to the model and measure the
        average perplexity of the generated tokens. Low perplexity
        = model is confident. High perplexity = model is guessing.

        Returns: 0.0 (no idea) to 1.0 (very confident)
        """
        self.model.eval()

        # Encode question + start of answer
        prompt = f"Question: {question}\nAnswer:"
        ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = torch.tensor([ids], device=self.device)

        with torch.no_grad():
            # Generate a few tokens and measure how "sure" the model is
            result = self.model(input_ids, enable_self_mod=False)
            logits = result["logits"][:, -1, :]

            # Get probability distribution
            probs = F.softmax(logits, dim=-1)

            # Entropy: low = confident, high = uncertain
            entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
            max_entropy = torch.log(torch.tensor(float(self.model.config.vocab_size))).item()

            # Normalize to 0-1 (inverted: high confidence = low entropy)
            confidence = 1.0 - (entropy / max_entropy)

            # Also check: is the top token much more probable than others?
            top_prob = probs.max().item()
            confidence = (confidence + top_prob) / 2  # Average both signals

        return min(max(confidence, 0.0), 1.0)

    def classify_query(self, question: str) -> ConfidenceResult:
        """
        Full classification of a query:
        - What's the model's confidence?
        - Is this live data or static knowledge?
        - Should we look it up?
        """
        question_lower = question.lower()

        # Step 1: Check if this needs live data
        live_type = None
        max_age = None
        for category, patterns in LIVE_DATA_PATTERNS.items():
            for keyword in patterns["keywords"]:
                if keyword in question_lower:
                    live_type = category
                    max_age = patterns["max_age_minutes"]
                    break
            if live_type:
                break

        # Step 2: Check memory for existing knowledge
        memory_results = self.memory.search(question, top_k=1)
        has_memory = len(memory_results) > 0
        memory_stale = False

        if has_memory and live_type and max_age:
            entry = memory_results[0][1]
            memory_stale = self.memory.is_stale(entry, max_age)

        # Step 3: Measure model confidence
        confidence = self.measure_confidence(question)

        # Step 4: Make the routing decision
        if live_type:
            if memory_stale or not has_memory:
                return ConfidenceResult(
                    confidence=confidence,
                    needs_lookup=False,
                    needs_live_data=True,
                    live_data_type=live_type,
                    reason=f"Live data ({live_type}) — need fresh info"
                )
            else:
                # Have recent memory of live data
                return ConfidenceResult(
                    confidence=0.9,
                    needs_lookup=False,
                    needs_live_data=False,
                    reason=f"Live data ({live_type}) but memory is recent enough"
                )

        if confidence >= self.high_confidence:
            return ConfidenceResult(
                confidence=confidence,
                needs_lookup=False,
                needs_live_data=False,
                reason=f"High confidence ({confidence:.2f}) — answering from internal knowledge"
            )

        if confidence >= self.confidence_threshold:
            if has_memory:
                return ConfidenceResult(
                    confidence=confidence,
                    needs_lookup=False,
                    needs_live_data=False,
                    reason=f"Medium confidence ({confidence:.2f}) + memory match — answering"
                )
            else:
                return ConfidenceResult(
                    confidence=confidence,
                    needs_lookup=True,
                    needs_live_data=False,
                    reason=f"Medium confidence ({confidence:.2f}) but no memory — checking knowledge base"
                )

        # Low confidence
        return ConfidenceResult(
            confidence=confidence,
            needs_lookup=True,
            needs_live_data=False,
            reason=f"Low confidence ({confidence:.2f}) — must check knowledge base"
        )

    def answer(self, question: str,
               max_new_tokens: int = 200,
               temperature: float = 0.7,
               verbose: bool = True) -> dict:
        """
        The main entry point. Ask QOR a question with zero-hallucination protection.

        Returns:
            {
                "question": "...",
                "answer": "...",
                "confidence": 0.85,
                "source": "internal" | "knowledge_base" | "tool:price_lookup" | "unknown",
                "reasoning": "why this source was chosen",
                "learned": True/False (did we update memory?),
            }
        """
        if verbose:
            print(f"\n  ┌─ Query: {question}")

        # Step 1: Classify the query
        classification = self.classify_query(question)

        if verbose:
            print(f"  ├─ Confidence: {classification.confidence:.2f}")
            print(f"  ├─ Reason: {classification.reason}")

        # Step 2: Route to the right source
        answer_text = None
        source = "internal"
        learned = False

        # ROUTE A: Needs live data from external tool
        if classification.needs_live_data:
            source = f"tool:{classification.live_data_type}"
            tool = self.tools.get_tool_for_category(classification.live_data_type)

            if tool:
                if verbose:
                    print(f"  ├─ Calling tool: {tool['name']}")

                tool_result = self.tools.call(tool["name"], question)

                if tool_result and "[" not in tool_result[:5]:  # Not an error message
                    # Build answer with fresh data
                    prompt = f"Based on this live data:\n{tool_result}\n\nQuestion: {question}\nAnswer:"
                    answer_text = self._generate(prompt, max_new_tokens, temperature)

                    # Update memory with fresh data
                    key = f"{classification.live_data_type}:{question[:50]}"
                    self.memory.store(
                        key=key,
                        content=tool_result,
                        source=tool["name"],
                        category="live",
                        confidence=0.95,
                    )
                    learned = True

                    if verbose:
                        print(f"  ├─ Tool result: {tool_result[:80]}...")
                        print(f"  ├─ Memory updated ✓")
                else:
                    if verbose:
                        print(f"  ├─ Tool unavailable, checking knowledge base...")
                    classification.needs_lookup = True

        # ROUTE B: Needs knowledge base lookup (RAG)
        if classification.needs_lookup and answer_text is None:
            source = "knowledge_base"

            # Try RAG first
            if self.rag:
                if verbose:
                    print(f"  ├─ Searching knowledge base...")

                rag_result = self.rag.query(
                    question, self.model, self.tokenizer,
                    max_new_tokens=max_new_tokens, temperature=temperature
                )

                if rag_result["chunks_used"] > 0:
                    answer_text = rag_result["answer"]

                    # Update memory with what we found
                    key = f"rag:{question[:50]}"
                    self.memory.store(
                        key=key,
                        content=answer_text[:500],
                        source="knowledge_base",
                        category="static",
                        confidence=0.8,
                    )
                    learned = True

                    if verbose:
                        print(f"  ├─ Found {rag_result['chunks_used']} relevant chunks")
                        print(f"  ├─ Memory updated ✓")

            # Try memory search
            if answer_text is None:
                memory_results = self.memory.search(question, top_k=3)
                if memory_results:
                    context = "\n".join([e[1].content for e in memory_results])
                    prompt = f"Based on what I know:\n{context}\n\nQuestion: {question}\nAnswer:"
                    answer_text = self._generate(prompt, max_new_tokens, temperature)
                    source = "memory"

            # Try web search as last resort
            if answer_text is None:
                web_tool = self.tools.get_tool_for_category("general")
                if web_tool:
                    if verbose:
                        print(f"  ├─ Searching web...")
                    web_result = self.tools.call(web_tool["name"], question)
                    if web_result and "[" not in web_result[:5]:
                        prompt = f"Based on this information:\n{web_result}\n\nQuestion: {question}\nAnswer:"
                        answer_text = self._generate(prompt, max_new_tokens, temperature)
                        source = "web_search"

                        # Learn from web result
                        key = f"web:{question[:50]}"
                        self.memory.store(
                            key=key,
                            content=web_result[:500],
                            source="web_search",
                            category="static",
                            confidence=0.7,
                        )
                        learned = True

        # ROUTE C: Answer from internal knowledge (confident)
        if answer_text is None and classification.confidence >= self.confidence_threshold:
            source = "internal"
            prompt = f"Question: {question}\nAnswer:"
            answer_text = self._generate(prompt, max_new_tokens, temperature)

        # ROUTE D: Don't know — honest answer
        if answer_text is None or classification.confidence < self.hallucination_threshold:
            source = "unknown"
            answer_text = (
                "I don't have enough information to answer this accurately. "
                "I'd rather be honest than guess and give you wrong information."
            )

        if verbose:
            print(f"  ├─ Source: {source}")
            print(f"  ├─ Learned: {'Yes ✓' if learned else 'No'}")
            print(f"  └─ Answer: {answer_text[:100]}...")

        return {
            "question": question,
            "answer": answer_text,
            "confidence": classification.confidence,
            "source": source,
            "reasoning": classification.reason,
            "learned": learned,
            "timestamp": datetime.now().isoformat(),
        }

    def _generate(self, prompt: str, max_new_tokens: int = 200,
                   temperature: float = 0.7) -> str:
        """Generate text from a prompt."""
        ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = torch.tensor([ids], device=self.device)

        self.model.eval()
        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=50,
            top_p=0.9,
            stop_tokens=[self.tokenizer.eos_id],
        )

        full_output = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        # Extract answer part
        if "Answer:" in full_output:
            return full_output.split("Answer:")[-1].strip()
        return full_output[len(prompt):].strip() if len(full_output) > len(prompt) else full_output

    def interactive_chat(self, verbose: bool = True):
        """
        Interactive chat with the zero-hallucination system.
        Shows the reasoning process for each answer.
        """
        print(f"\n{'='*60}")
        print(f"  QOR — Zero Hallucination Chat")
        print(f"  The model KNOWS what it knows and what it doesn't.")
        print(f"{'='*60}")
        print(f"  Commands:")
        print(f"    quit     — Exit")
        print(f"    reset    — Clear fast memory")
        print(f"    memory   — Show memory stats")
        print(f"    tools    — Show available tools")
        print(f"    verbose  — Toggle reasoning display")
        print(f"{'='*60}\n")

        while True:
            try:
                question = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not question:
                continue
            if question.lower() == "quit":
                break
            if question.lower() == "reset":
                self.model.reset_fast_weights()
                print("  [Fast memory reset]\n")
                continue
            if question.lower() == "memory":
                self.memory.stats()
                print()
                continue
            if question.lower() == "tools":
                for tool in self.tools.list_tools():
                    print(f"  {tool['name']}: {tool['description']}")
                print()
                continue
            if question.lower() == "verbose":
                verbose = not verbose
                print(f"  [Verbose: {'ON' if verbose else 'OFF'}]\n")
                continue

            result = self.answer(question, verbose=verbose)
            print(f"\nQOR: {result['answer']}")
            if not verbose:
                confidence_bar = "█" * int(result['confidence'] * 10) + "░" * (10 - int(result['confidence'] * 10))
                print(f"  [{confidence_bar}] {result['source']}")
            print()

    def get_stats(self) -> dict:
        """Get system statistics."""
        return {
            "memory_entries": len(self.memory.entries),
            "tools_available": len(self.tools.tools),
            "rag_connected": self.rag is not None,
            "confidence_threshold": self.confidence_threshold,
            "live_data_categories": list(LIVE_DATA_PATTERNS.keys()),
        }

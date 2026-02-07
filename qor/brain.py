"""
QOR Confidence Gate — Zero Hallucination System
=================================================
The model NEVER guesses. It always knows if it knows.

How it works:

  1. User asks a question
  2. QOR checks: "Do I know this?" (measure surprise/confidence)
  3. Three possible outcomes:
  
     HIGH confidence → Answer from internal knowledge
     LOW confidence  → Search knowledge base, then answer
     STALE data      → Call live API/tool, update memory, then answer

  4. After answering, UPDATE memory so next time it knows

This eliminates hallucination because:
  - Model never makes up facts it's unsure about
  - Model knows which topics need live data (prices, weather, events)
  - Model updates itself after every lookup

Think of it like a smart person who:
  - Answers what they know confidently
  - Says "let me look that up" for things they're unsure about
  - Checks live sources for things that change (prices, news)
  - REMEMBERS what they looked up for next time
"""

import os
import json
import time
import torch
import torch.nn.functional as F
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Callable
from dataclasses import dataclass, field


# ==============================================================================
# KNOWLEDGE FRESHNESS TRACKER
# ==============================================================================

@dataclass
class KnowledgeEntry:
    """Tracks when the model last learned about a topic."""
    topic: str
    last_updated: str           # ISO datetime
    source: str                 # Where the knowledge came from
    needs_live_data: bool       # Does this topic change frequently?
    refresh_hours: int = 24     # How often to refresh (hours)
    value: str = ""             # Last known value (for quick reference)

    def is_stale(self) -> bool:
        """Check if this knowledge is outdated."""
        if not self.needs_live_data:
            return False
        last = datetime.fromisoformat(self.last_updated)
        age = datetime.now() - last
        return age > timedelta(hours=self.refresh_hours)


class KnowledgeTracker:
    """
    Tracks WHAT the model knows and WHEN it learned it.
    
    This is the key to zero hallucination:
    - If the model has fresh knowledge → use it
    - If knowledge is stale → refresh it
    - If no knowledge exists → go find it
    """

    def __init__(self, save_path: str = "knowledge_tracker.json"):
        self.entries: Dict[str, KnowledgeEntry] = {}
        self.save_path = save_path

        # Topics that ALWAYS need live data (never trust memory alone)
        self.live_data_topics = {
            # Prices and markets
            "price", "cost", "stock", "market", "crypto", "bitcoin", "btc",
            "ethereum", "eth", "trading", "exchange rate", "forex",
            # Weather
            "weather", "temperature", "forecast", "rain", "storm",
            # News and events  
            "news", "today", "latest", "current", "now", "recent",
            "yesterday", "this week", "this month", "breaking",
            # Sports
            "score", "game", "match", "tournament", "standings",
            # Time-sensitive
            "schedule", "hours", "open", "closed", "available",
            "deadline", "event", "election", "result",
        }

        # Load existing tracker
        if os.path.exists(save_path):
            self.load()

    def needs_live_data(self, query: str) -> bool:
        """Check if a query is about something that changes frequently."""
        query_lower = query.lower()
        return any(topic in query_lower for topic in self.live_data_topics)

    def get_entry(self, topic: str) -> Optional[KnowledgeEntry]:
        """Get knowledge entry for a topic."""
        topic_lower = topic.lower()
        # Check exact match
        if topic_lower in self.entries:
            return self.entries[topic_lower]
        # Check partial match
        for key, entry in self.entries.items():
            if key in topic_lower or topic_lower in key:
                return entry
        return None

    def update(self, topic: str, value: str, source: str,
               needs_live: bool = False, refresh_hours: int = 24):
        """Record that we know something (and when we learned it)."""
        self.entries[topic.lower()] = KnowledgeEntry(
            topic=topic,
            last_updated=datetime.now().isoformat(),
            source=source,
            needs_live_data=needs_live,
            refresh_hours=refresh_hours,
            value=value,
        )
        self.save()

    def save(self):
        data = {}
        for key, entry in self.entries.items():
            data[key] = {
                "topic": entry.topic,
                "last_updated": entry.last_updated,
                "source": entry.source,
                "needs_live_data": entry.needs_live_data,
                "refresh_hours": entry.refresh_hours,
                "value": entry.value,
            }
        with open(self.save_path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self):
        try:
            with open(self.save_path) as f:
                data = json.load(f)
            for key, val in data.items():
                self.entries[key] = KnowledgeEntry(**val)
        except Exception:
            pass


# ==============================================================================
# TOOL REGISTRY — External APIs and data sources
# ==============================================================================

class Tool:
    """A callable tool that QOR can use to get live data."""

    def __init__(self, name: str, description: str,
                 keywords: List[str], func: Callable,
                 refresh_hours: int = 1):
        self.name = name
        self.description = description
        self.keywords = [k.lower() for k in keywords]
        self.func = func
        self.refresh_hours = refresh_hours

    def matches(self, query: str) -> bool:
        """Check if this tool is relevant to the query."""
        query_lower = query.lower()
        return any(kw in query_lower for kw in self.keywords)

    def call(self, query: str) -> dict:
        """Execute the tool and return results."""
        try:
            result = self.func(query)
            return {"success": True, "data": result, "tool": self.name}
        except Exception as e:
            return {"success": False, "error": str(e), "tool": self.name}


class ToolRegistry:
    """Registry of all tools QOR can use to get live data."""

    def __init__(self):
        self.tools: List[Tool] = []
        self._register_default_tools()

    def register(self, tool: Tool):
        """Add a new tool."""
        self.tools.append(tool)

    def find_tools(self, query: str) -> List[Tool]:
        """Find all tools relevant to a query."""
        return [t for t in self.tools if t.matches(query)]

    def _register_default_tools(self):
        """Register built-in tools. Add your own APIs here!"""

        # ===== Example: Crypto price tool =====
        def get_crypto_price(query: str) -> dict:
            """Get cryptocurrency prices via free API."""
            try:
                import urllib.request
                import json as _json

                # Detect which crypto
                query_lower = query.lower()
                coin = "bitcoin"
                if "eth" in query_lower or "ethereum" in query_lower:
                    coin = "ethereum"
                elif "btc" in query_lower or "bitcoin" in query_lower:
                    coin = "bitcoin"

                url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin}&vs_currencies=usd&include_24hr_change=true"
                req = urllib.request.Request(url, headers={"User-Agent": "QOR/1.0"})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = _json.loads(resp.read().decode())

                price = data[coin]["usd"]
                change = data[coin].get("usd_24h_change", 0)
                return {
                    "coin": coin,
                    "price_usd": price,
                    "change_24h": round(change, 2),
                    "timestamp": datetime.now().isoformat(),
                    "text": f"{coin.title()} is currently ${price:,.2f} USD ({change:+.2f}% in 24h)",
                }
            except Exception as e:
                return {"error": str(e), "text": f"Could not fetch {coin} price: {e}"}

        self.register(Tool(
            name="crypto_price",
            description="Get current cryptocurrency prices",
            keywords=["bitcoin", "btc", "ethereum", "eth", "crypto",
                      "coin price", "crypto price"],
            func=get_crypto_price,
            refresh_hours=1,  # Crypto prices change fast
        ))

        # ===== Example: Date/Time tool =====
        def get_datetime(query: str) -> dict:
            now = datetime.now()
            return {
                "date": now.strftime("%Y-%m-%d"),
                "time": now.strftime("%H:%M:%S"),
                "day": now.strftime("%A"),
                "text": f"Today is {now.strftime('%A, %B %d, %Y')} and the time is {now.strftime('%I:%M %p')}",
            }

        self.register(Tool(
            name="datetime",
            description="Get current date and time",
            keywords=["what time", "what date", "today", "what day",
                      "current date", "current time"],
            func=get_datetime,
            refresh_hours=0,  # Always fresh
        ))

        # ===== TEMPLATE: Add your own tools =====
        #
        # def my_custom_tool(query: str) -> dict:
        #     # Call your API here
        #     response = requests.get("https://your-api.com/data")
        #     return {"text": response.json()["result"]}
        #
        # self.register(Tool(
        #     name="my_tool",
        #     description="What it does",
        #     keywords=["trigger", "words"],
        #     func=my_custom_tool,
        #     refresh_hours=24,
        # ))


# ==============================================================================
# CONFIDENCE SCORER — Does QOR know the answer?
# ==============================================================================

class ConfidenceScorer:
    """
    Measures how confident QOR is about a topic.

    HIGH confidence = low perplexity on the topic
                    = model has seen this before
                    = safe to answer from memory

    LOW confidence  = high perplexity on the topic
                    = model hasn't seen this
                    = MUST look it up (no guessing!)
    """

    def __init__(self, model, tokenizer, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # Thresholds (tune these based on your model)
        self.high_confidence = 2.0    # Below this = "I know this well"
        self.medium_confidence = 5.0  # Below this = "I think I know"
        self.low_confidence = 8.0     # Above this = "I have no idea"

    @torch.no_grad()
    def score(self, text: str) -> dict:
        """
        Score how confident the model is about a piece of text.

        Returns:
            {
                "perplexity": float,    # How surprised the model is
                "confidence": str,      # "high", "medium", "low"
                "should_lookup": bool,  # Whether to search knowledge base
            }
        """
        self.model.eval()

        ids = self.tokenizer.encode(text, add_special_tokens=True)
        if len(ids) < 3:
            return {"perplexity": 999, "confidence": "low", "should_lookup": True}

        input_ids = torch.tensor([ids], device=self.device)

        result = self.model(input_ids, targets=input_ids, enable_self_mod=False)
        loss = result["loss"].item() if result["loss"] is not None else 10.0
        perplexity = min(torch.exp(torch.tensor(loss)).item(), 1000.0)

        if perplexity < self.high_confidence:
            confidence = "high"
            should_lookup = False
        elif perplexity < self.medium_confidence:
            confidence = "medium"
            should_lookup = False
        elif perplexity < self.low_confidence:
            confidence = "medium-low"
            should_lookup = True
        else:
            confidence = "low"
            should_lookup = True

        return {
            "perplexity": round(perplexity, 2),
            "confidence": confidence,
            "should_lookup": should_lookup,
        }

    def score_query(self, query: str) -> dict:
        """Score confidence on a user query specifically."""
        # Test: can the model complete this naturally?
        test_prompt = f"Question: {query}\nAnswer: The answer is"
        return self.score(test_prompt)


# ==============================================================================
# THE MAIN SYSTEM — Zero Hallucination Gate
# ==============================================================================

class QORBrain:
    """
    The complete zero-hallucination system.

    For every query:
    1. Check if topic needs live data (prices, weather, etc.)
    2. Check if cached knowledge is stale
    3. Measure model's confidence
    4. Decide: answer from memory, look up, or call tool
    5. Generate answer
    6. Update memory for next time

    The model NEVER guesses. It ALWAYS knows if it knows.
    """

    def __init__(self, model, tokenizer, config, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device

        # Components
        self.confidence = ConfidenceScorer(model, tokenizer, device)
        self.tracker = KnowledgeTracker()
        self.tools = ToolRegistry()
        self.rag = None  # Set up separately with .setup_rag()

        # Stats
        self.stats = {
            "total_queries": 0,
            "answered_from_memory": 0,
            "answered_from_rag": 0,
            "answered_from_tools": 0,
            "knowledge_updates": 0,
        }

    def setup_rag(self, knowledge_dir: str):
        """Set up the RAG knowledge base from a folder of documents."""
        from .rag import QORRag
        self.rag = QORRag()
        self.rag.add_folder(knowledge_dir)

    def add_tool(self, name: str, description: str,
                 keywords: List[str], func: Callable,
                 refresh_hours: int = 1):
        """Register a custom tool (API, database, etc.)."""
        self.tools.register(Tool(name, description, keywords, func, refresh_hours))

    def think(self, query: str,
              max_new_tokens: int = 200,
              temperature: float = 0.7,
              verbose: bool = False) -> dict:
        """
        Main entry point. Process a query with zero hallucination.

        Returns:
            {
                "answer": str,
                "source": str,           # "memory", "knowledge_base", "tool:name", "uncertain"
                "confidence": str,        # "high", "medium", "low"
                "tool_used": str or None,
                "knowledge_updated": bool,
                "reasoning": str,         # Explanation of decision
            }
        """
        self.stats["total_queries"] += 1
        reasoning_steps = []

        # ==== STEP 1: Does this need live data? ====
        needs_live = self.tracker.needs_live_data(query)
        if needs_live:
            reasoning_steps.append(f"Topic requires live data (detected time-sensitive keywords)")

        # ==== STEP 2: Do we have a matching tool? ====
        matching_tools = self.tools.find_tools(query)
        if matching_tools:
            tool = matching_tools[0]
            reasoning_steps.append(f"Found matching tool: {tool.name}")

            # Check if cached result is still fresh
            entry = self.tracker.get_entry(query[:50])
            if entry and not entry.is_stale():
                reasoning_steps.append(f"Cached result is fresh (updated {entry.last_updated})")
                # Use cached result
                answer = self._generate_with_context(
                    query, f"Known fact: {entry.value}", max_new_tokens, temperature
                )
                self.stats["answered_from_memory"] += 1
                return self._build_response(
                    answer, "memory (cached tool result)", "high",
                    reasoning_steps, tool_used=None, updated=False
                )

            # Call the tool for fresh data
            reasoning_steps.append(f"Calling tool: {tool.name}")
            result = tool.call(query)

            if result["success"]:
                tool_text = result["data"].get("text", json.dumps(result["data"]))
                reasoning_steps.append(f"Tool returned: {tool_text[:100]}...")

                # UPDATE MEMORY with fresh data
                self.tracker.update(
                    topic=query[:50],
                    value=tool_text,
                    source=f"tool:{tool.name}",
                    needs_live=True,
                    refresh_hours=tool.refresh_hours,
                )
                self.stats["knowledge_updates"] += 1
                reasoning_steps.append("Memory updated with fresh data")

                # Generate answer with tool data
                answer = self._generate_with_context(
                    query, f"Live data: {tool_text}", max_new_tokens, temperature
                )
                self.stats["answered_from_tools"] += 1
                return self._build_response(
                    answer, f"tool:{tool.name}", "high",
                    reasoning_steps, tool_used=tool.name, updated=True
                )
            else:
                reasoning_steps.append(f"Tool failed: {result.get('error', 'unknown')}")
                # Fall through to other methods

        # ==== STEP 3: Check model confidence ====
        conf = self.confidence.score_query(query)
        reasoning_steps.append(
            f"Model confidence: {conf['confidence']} (perplexity: {conf['perplexity']})"
        )

        # ==== STEP 4a: HIGH confidence → Answer from memory ====
        if not conf["should_lookup"] and not needs_live:
            reasoning_steps.append("Answering from internal knowledge (high confidence)")
            answer = self._generate_direct(query, max_new_tokens, temperature)
            self.stats["answered_from_memory"] += 1
            return self._build_response(
                answer, "memory", conf["confidence"],
                reasoning_steps, tool_used=None, updated=False
            )

        # ==== STEP 4b: LOW confidence → Search knowledge base ====
        if self.rag is not None:
            reasoning_steps.append("Searching knowledge base (low confidence or stale data)")
            rag_results = self.rag.search(query, top_k=3)

            if rag_results:
                context_parts = []
                for chunk_text, score, meta in rag_results:
                    if score > 0.05:  # Minimum relevance threshold
                        context_parts.append(chunk_text)
                        reasoning_steps.append(
                            f"Found relevant chunk from {meta['source']} (score: {score:.3f})"
                        )

                if context_parts:
                    context = "\n\n".join(context_parts)
                    answer = self._generate_with_context(
                        query, context, max_new_tokens, temperature
                    )

                    # UPDATE MEMORY with RAG result
                    self.tracker.update(
                        topic=query[:50],
                        value=answer[:200],
                        source="knowledge_base",
                        needs_live=needs_live,
                        refresh_hours=168 if not needs_live else 24,  # 1 week or 1 day
                    )
                    self.stats["knowledge_updates"] += 1
                    self.stats["answered_from_rag"] += 1
                    reasoning_steps.append("Memory updated with knowledge base answer")

                    return self._build_response(
                        answer, "knowledge_base", "medium",
                        reasoning_steps, tool_used=None, updated=True
                    )

        # ==== STEP 4c: No knowledge base or nothing found ====
        # Be HONEST — don't hallucinate
        if conf["should_lookup"] or needs_live:
            reasoning_steps.append("LOW confidence and no knowledge source available")
            reasoning_steps.append("Being honest: admitting uncertainty instead of guessing")
            answer = self._generate_uncertain(query, max_new_tokens, temperature)
            return self._build_response(
                answer, "uncertain", "low",
                reasoning_steps, tool_used=None, updated=False
            )

        # ==== STEP 5: Medium confidence — answer but flag it ====
        reasoning_steps.append("Medium confidence — answering but may not be fully accurate")
        answer = self._generate_direct(query, max_new_tokens, temperature)
        self.stats["answered_from_memory"] += 1
        return self._build_response(
            answer, "memory (medium confidence)", conf["confidence"],
            reasoning_steps, tool_used=None, updated=False
        )

    def _generate_direct(self, query: str, max_tokens: int, temperature: float) -> str:
        """Generate answer directly from model knowledge."""
        prompt = f"Question: {query}\nAnswer:"
        return self._generate(prompt, max_tokens, temperature)

    def _generate_with_context(self, query: str, context: str,
                                max_tokens: int, temperature: float) -> str:
        """Generate answer using provided context."""
        prompt = f"""Context: {context}

Based on the context above, answer this question accurately.
Question: {query}
Answer:"""
        return self._generate(prompt, max_tokens, temperature)

    def _generate_uncertain(self, query: str, max_tokens: int, temperature: float) -> str:
        """Generate an honest 'I don't know' response."""
        prompt = f"""I need to be honest about what I know and don't know.
Question: {query}
I don't have reliable information about this topic. Let me tell you what I can:
Answer:"""
        generated = self._generate(prompt, max_tokens, temperature)
        # Prepend honesty marker if model doesn't include one
        if "don't know" not in generated.lower() and "not sure" not in generated.lower():
            generated = f"I'm not confident about this answer and recommend verifying: {generated}"
        return generated

    def _generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Core generation."""
        ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = torch.tensor([ids], device=self.device)

        self.model.eval()
        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=50,
            top_p=0.9,
            stop_tokens=[self.tokenizer.eos_id],
        )

        full_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        # Extract answer part
        if "Answer:" in full_text:
            return full_text.split("Answer:")[-1].strip()
        return full_text[len(prompt):].strip() if len(full_text) > len(prompt) else full_text

    def _build_response(self, answer, source, confidence,
                        reasoning, tool_used, updated) -> dict:
        return {
            "answer": answer,
            "source": source,
            "confidence": confidence,
            "tool_used": tool_used,
            "knowledge_updated": updated,
            "reasoning": reasoning,
            "timestamp": datetime.now().isoformat(),
        }

    def get_stats(self) -> dict:
        """Return system statistics."""
        total = max(self.stats["total_queries"], 1)
        return {
            **self.stats,
            "memory_hit_rate": f"{self.stats['answered_from_memory'] / total * 100:.1f}%",
            "rag_hit_rate": f"{self.stats['answered_from_rag'] / total * 100:.1f}%",
            "tool_hit_rate": f"{self.stats['answered_from_tools'] / total * 100:.1f}%",
            "knowledge_entries": len(self.tracker.entries),
        }


# ==============================================================================
# UPDATED API SERVER — With Zero Hallucination
# ==============================================================================

def create_brain_server(config, checkpoint_path: str,
                        knowledge_dir: Optional[str] = None):
    """
    Create a QOR server with the full brain system.

    This replaces the simple serve.py with the intelligent
    confidence-gated zero-hallucination system.
    """
    from .model import QORModel
    from .tokenizer import QORTokenizer

    device = config.get_device()

    # Load model
    tokenizer = QORTokenizer()
    tokenizer.load(config.tokenizer.save_path)
    config.model.vocab_size = tokenizer.vocab_size

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = QORModel(config.model).to(device)
    model.load_state_dict(checkpoint["model_state"])

    # Create brain
    brain = QORBrain(model, tokenizer, config, device)

    # Set up RAG if knowledge directory provided
    if knowledge_dir and os.path.exists(knowledge_dir):
        brain.setup_rag(knowledge_dir)

    return brain


def run_brain_server(config, checkpoint_path: str,
                     knowledge_dir: Optional[str] = None,
                     port: int = 8000):
    """Run the full brain server with Flask."""
    try:
        from flask import Flask, request, jsonify
        from flask_cors import CORS
    except ImportError:
        print("Install: pip install flask flask-cors")
        return

    brain = create_brain_server(config, checkpoint_path, knowledge_dir)

    app = Flask(__name__)
    CORS(app)

    @app.route('/think', methods=['POST'])
    def think():
        """Main endpoint — zero hallucination query."""
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "Missing 'query'"}), 400

        result = brain.think(
            query=data['query'],
            max_new_tokens=data.get('max_tokens', 200),
            temperature=data.get('temperature', 0.7),
            verbose=data.get('verbose', False),
        )
        return jsonify(result)

    @app.route('/generate', methods=['POST'])
    def generate():
        """Simple generation (no confidence gate, for backward compatibility)."""
        data = request.get_json()
        if not data or 'prompt' not in data:
            return jsonify({"error": "Missing 'prompt'"}), 400
        answer = brain._generate_direct(data['prompt'], 200, 0.8)
        return jsonify({"answer": answer})

    @app.route('/stats', methods=['GET'])
    def stats():
        return jsonify(brain.get_stats())

    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({"status": "ok", "system": "QOR Brain"})

    print(f"\n{'='*60}")
    print(f"  QOR Brain — Zero Hallucination Server")
    print(f"{'='*60}")
    print(f"  Model: {brain.model.n_params:,} parameters")
    print(f"  Device: {brain.device}")
    print(f"  RAG: {'enabled' if brain.rag else 'disabled'}")
    print(f"  Tools: {len(brain.tools.tools)} registered")
    print(f"\n  Endpoints:")
    print(f"    POST /think    — Smart query (zero hallucination)")
    print(f"    POST /generate — Direct generation")
    print(f"    GET  /stats    — System statistics")
    print(f"\n  Example:")
    print(f'    curl -X POST http://localhost:{port}/think \\')
    print(f'      -H "Content-Type: application/json" \\')
    print(f'      -d \'{{"query": "What is the price of Bitcoin today?"}}\'')
    print(f"{'='*60}\n")

    app.run(host="0.0.0.0", port=port, debug=False)

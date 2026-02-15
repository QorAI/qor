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
import math
import json
import time
import re
import threading
import torch
import torch.nn.functional as F
from typing import Optional, List, Callable, Tuple
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field

import pyarrow as pa
import pyarrow.parquet as pq


# ==============================================================================
# MEDIA DETECTION — Find image/audio file paths in user questions
# ==============================================================================

_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff', '.tif'}
_AUDIO_EXTS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma'}

# Match file paths: Windows (C:\...\file.ext) or Unix (/path/file.ext) or relative (dir/file.ext)
_MEDIA_PATH_RE = re.compile(
    r'[A-Z]:[\\\/][^\s"<>|]+?\.(?:jpg|jpeg|png|bmp|gif|webp|tiff|tif|wav|mp3|flac|ogg|m4a|aac|wma)'
    r'|\/[^\s"<>|]+?\.(?:jpg|jpeg|png|bmp|gif|webp|tiff|tif|wav|mp3|flac|ogg|m4a|aac|wma)'
    r'|[\w][\w.\\\/\-]*?\.(?:jpg|jpeg|png|bmp|gif|webp|tiff|tif|wav|mp3|flac|ogg|m4a|aac|wma)',
    re.IGNORECASE,
)


# ==============================================================================
# DATA FRESHNESS — What needs live data vs what's static
# ==============================================================================

# Topics that ALWAYS need live/fresh data — never trust memory alone
LIVE_DATA_PATTERNS = {
    "price": {
        "keywords": ["price", "cost", "worth", "market cap", "trading at",
                      "how much is", "current value", "stock price",
                      "market sentiment", "market overview", "market condition",
                      "market update", "how is the market", "market today",
                      "market doing", "fear and greed", "market indices",
                      "rsi", "macd", "technical analysis", "funding rate",
                      "open interest", "crypto", "bitcoin", "btc", "eth",
                      "ethereum", "solana", "xrp", "gold price", "silver price",
                      "commodities", "dow jones", "s&p 500", "nasdaq",
                      "stock market", "bull", "bear", "rally", "crash",
                      "correction", "ath", "all-time high", "all time high"],
        "reason": "Prices change every second",
        "max_age_minutes": 5,
    },
    "weather": {
        "keywords": ["weather", "temperature", "forecast", "rain",
                      "sunny", "cloudy", "storm", "humidity", "wind",
                      "snow", "heatwave", "cold front", "uv index",
                      "air quality", "pollen"],
        "reason": "Weather changes constantly",
        "max_age_minutes": 30,
    },
    "news": {
        "keywords": ["latest", "breaking", "just happened", "today",
                      "yesterday", "this week", "recent", "current events",
                      "news about", "what happened", "announced", "released",
                      "launched", "update on", "situation in", "crisis",
                      "war in", "conflict", "sanctions", "election",
                      "vote", "protest", "earthquake", "hurricane",
                      "this month", "this year", "right now", "going on"],
        "reason": "News is time-sensitive",
        "max_age_minutes": 60,
    },
    "sports": {
        "keywords": ["score", "who won", "game", "match", "tournament",
                      "standings", "season", "playoff", "championship",
                      "world cup", "premier league", "nba", "nfl", "mlb",
                      "champions league", "super bowl", "olympics",
                      "formula 1", "f1", "ufc", "boxing"],
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
                      "dollar to", "euro to", "forex", "fx rate",
                      "eur/usd", "gbp/usd", "usd/jpy"],
        "reason": "Exchange rates fluctuate constantly",
        "max_age_minutes": 15,
    },
    "traffic": {
        "keywords": ["traffic", "route", "commute", "road conditions",
                      "travel time", "delays", "accidents"],
        "reason": "Traffic conditions change in real-time",
        "max_age_minutes": 5,
    },
    "facts": {
        "keywords": [
            # Factual questions people ask — should use tools, not hallucinate
            # People / Who questions
            "who is", "who was", "who are", "who founded", "who invented",
            "who discovered", "who created", "who leads", "who owns",
            "who built", "who runs", "who holds", "who won", "who lost",
            "who killed", "who married", "who wrote", "who directed",
            "who designed", "who said",
            # What questions
            "what is the", "what are the", "what was the", "what were the",
            "what does", "what did", "what caused", "what happened to",
            "what year", "what date", "what country", "what city",
            "what language", "what religion", "what percent", "what percentage",
            # How questions
            "how many", "how much does", "how much is the", "how old",
            "how tall", "how far", "how long", "how big", "how fast",
            "how deep", "how high", "how wide", "how heavy",
            "how much did", "how much was",
            # Where questions
            "where is", "where was", "where are", "where did",
            "where does", "where do", "where can",
            # When questions
            "when did", "when was", "when is", "when will", "when does",
            # Which questions
            "which country", "which city", "which company", "which president",
            "which government", "which team", "which year",
            # Quantitative / stats
            "how many people", "population of", "population in",
            "gdp of", "gdp per capita", "area of", "size of",
            "height of", "weight of", "distance from", "distance between",
            "number of", "total number", "count of",
            # Specific factual topics
            "reserves", "holdings", "net worth", "salary of",
            "revenue of", "market share", "total value",
            "life expectancy", "literacy rate", "unemployment",
            "inflation rate", "debt of", "budget of",
            # Comparisons
            "largest", "smallest", "biggest", "tallest", "shortest",
            "richest", "poorest", "fastest", "slowest", "oldest",
            "youngest", "most popular", "most expensive", "cheapest",
            "highest", "lowest", "best", "worst", "top 10", "top 5",
            "compared to", "difference between", "vs ",
            # Historical facts
            "founded in", "established in", "born in", "died in",
            "invented in", "discovered in", "built in", "started in",
            "ended in", "happened in", "battle of", "war of",
            "treaty of", "history of",
            # Science & tech
            "speed of light", "boiling point", "melting point",
            "chemical formula", "atomic number", "molecular weight",
            "scientific name", "discovery of", "theory of",
            "law of", "element ", "planet ", "star ",
            # Geography
            "capital of", "capital city", "continent of",
            "located in", "borders", "timezone", "coordinates of",
            "latitude", "longitude", "elevation of",
            "mountain", "river", "ocean", "lake", "island",
            # Government & politics
            "president of", "prime minister", "king of", "queen of",
            "chancellor of", "governor of", "senator", "congress",
            "parliament", "government of", "political party",
            "constitution", "amendment", "law in",
            # Economy & business
            "ceo of", "founder of", "company", "corporation",
            "stock of", "ipo", "valuation", "acquisition",
            "merger", "bankruptcy", "headquarter",
            # Culture & entertainment
            "directed by", "written by", "author of", "singer of",
            "actor in", "movie ", "film ", "song ", "album ",
            "tv show", "series ", "award", "grammy", "oscar",
            "emmy", "nobel prize",
            # Food & health
            "calories in", "nutrition", "vitamin", "symptoms of",
            "treatment for", "cure for", "disease", "side effects",
            "ingredient", "recipe for",
        ],
        "reason": "Factual queries need verification — use tools not training data",
        "max_age_minutes": 1440,  # 24 hours — facts change slowly
    },
}


@dataclass
class KnowledgeEntry:
    """A piece of knowledge with freshness tracking."""
    content: str                          # The actual knowledge
    source: str                           # Where it came from
    timestamp: str                        # When it was learned
    category: str = "static"              # "static", "live", or "historical"
    confidence: float = 1.0               # How confident we are (0-1)
    access_count: int = 0                 # How often this was used
    last_accessed: str = ""               # Last time someone asked about this
    data_hash: str = ""                   # SHA-256 hash of content
    prev_hash: str = ""                   # Previous entry's hash (chain)


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

MEMORY_SCHEMA = pa.schema([
    ("key", pa.string()),
    ("content", pa.string()),
    ("source", pa.string()),
    ("category", pa.string()),
    ("confidence", pa.float32()),
    ("timestamp", pa.timestamp("us", tz="UTC")),
    ("access_count", pa.int32()),
    ("last_accessed", pa.timestamp("us", tz="UTC")),
    ("data_hash", pa.string()),
    ("prev_hash", pa.string()),
])


class MemoryStore:
    """
    Persistent memory that tracks WHEN things were learned.
    Uses Arrow/Parquet for storage — same format as batch files.

    In-memory dict for fast access, flushes to Parquet on save.
    SHA-256 hash chain for tamper detection on every entry.
    """

    def __init__(self, path: str = "memory.parquet"):
        self.path = path
        self.entries = {}  # key → KnowledgeEntry
        self._dirty = False
        self._dirty_count = 0
        self._flush_every = 50  # auto-save after this many writes
        self._chain_head = ""    # Track hash chain head for new entries
        self._load()

    def _load(self):
        """Load entries from Parquet file."""
        # Try parquet first
        if os.path.exists(self.path):
            try:
                # Try new schema first, fall back to reading without schema
                # for backward compat with old files missing hash columns
                try:
                    table = pq.read_table(self.path, schema=MEMORY_SCHEMA)
                except Exception:
                    table = pq.read_table(self.path)
                self._load_from_table(table)
                return
            except Exception:
                pass

        # Fallback: try legacy memory.json in same directory
        json_path = os.path.join(os.path.dirname(self.path) or ".", "memory.json")
        if os.path.exists(json_path):
            try:
                with open(json_path) as f:
                    data = json.load(f)
                for key, entry_data in data.items():
                    self.entries[key] = KnowledgeEntry(**entry_data)
                # Migrate: save as parquet immediately
                self.save()
            except Exception:
                pass

    def _load_from_table(self, table: pa.Table):
        """Populate entries dict from an Arrow table."""
        keys = table.column("key")
        contents = table.column("content")
        sources = table.column("source")
        categories = table.column("category")
        confidences = table.column("confidence")
        timestamps = table.column("timestamp").cast(pa.int64())
        access_counts = table.column("access_count")
        last_accessed_col = table.column("last_accessed").cast(pa.int64())

        # Hash columns (may not exist in old parquet files)
        has_hash = "data_hash" in table.schema.names
        data_hashes = table.column("data_hash") if has_hash else None
        prev_hashes = table.column("prev_hash") if has_hash else None

        for i in range(table.num_rows):
            key = keys[i].as_py()
            ts_us = timestamps[i].as_py()
            la_us = last_accessed_col[i].as_py()

            ts_iso = ""
            if ts_us and ts_us > 0:
                ts_iso = datetime.fromtimestamp(
                    ts_us / 1_000_000, tz=timezone.utc
                ).isoformat()

            la_iso = ""
            if la_us and la_us > 0:
                la_iso = datetime.fromtimestamp(
                    la_us / 1_000_000, tz=timezone.utc
                ).isoformat()

            dh = data_hashes[i].as_py() or "" if data_hashes else ""
            ph = prev_hashes[i].as_py() or "" if prev_hashes else ""

            self.entries[key] = KnowledgeEntry(
                content=contents[i].as_py() or "",
                source=sources[i].as_py() or "",
                timestamp=ts_iso,
                category=categories[i].as_py() or "static",
                confidence=confidences[i].as_py() or 1.0,
                access_count=access_counts[i].as_py() or 0,
                last_accessed=la_iso,
                data_hash=dh,
                prev_hash=ph,
            )
            # Restore chain head from last entry's hash
            if dh:
                self._chain_head = dh

    def save(self):
        """Flush all entries to Parquet."""
        if not self.entries:
            return

        keys = []
        contents = []
        sources = []
        categories = []
        confidences = []
        timestamps = []
        access_counts = []
        last_accessed_list = []

        data_hashes = []
        prev_hashes = []

        for key, entry in self.entries.items():
            keys.append(key)
            contents.append(entry.content)
            sources.append(entry.source)
            categories.append(entry.category)
            confidences.append(entry.confidence)
            timestamps.append(self._iso_to_us(entry.timestamp))
            access_counts.append(entry.access_count)
            last_accessed_list.append(self._iso_to_us(entry.last_accessed))
            data_hashes.append(entry.data_hash)
            prev_hashes.append(entry.prev_hash)

        batch = pa.RecordBatch.from_pydict(
            {
                "key": keys,
                "content": contents,
                "source": sources,
                "category": categories,
                "confidence": confidences,
                "timestamp": timestamps,
                "access_count": access_counts,
                "last_accessed": last_accessed_list,
                "data_hash": data_hashes,
                "prev_hash": prev_hashes,
            },
            schema=MEMORY_SCHEMA,
        )
        table = pa.Table.from_batches([batch])
        pq.write_table(table, self.path)
        self._dirty = False
        self._dirty_count = 0

    @staticmethod
    def _iso_to_us(iso_str: str) -> int:
        """Convert ISO timestamp string to microseconds since epoch."""
        if not iso_str:
            return 0
        try:
            dt = datetime.fromisoformat(iso_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1_000_000)
        except (ValueError, TypeError):
            return 0

    def _maybe_flush(self):
        """Auto-save after enough writes."""
        self._dirty_count += 1
        self._dirty = True
        if self._dirty_count >= self._flush_every:
            self.save()

    def store(self, key: str, content: str, source: str,
              category: str = "static", confidence: float = 1.0):
        """Store a piece of knowledge with SHA-256 hash chain."""
        from .crypto import QORCrypto
        prev_hash = self._chain_head
        data_hash = QORCrypto.hash_sha256(content)
        self._chain_head = data_hash

        self.entries[key] = KnowledgeEntry(
            content=content,
            source=source,
            timestamp=datetime.now(timezone.utc).isoformat(),
            category=category,
            confidence=confidence,
            access_count=0,
            last_accessed="",
            data_hash=data_hash,
            prev_hash=prev_hash,
        )
        self._maybe_flush()

    def get(self, key: str) -> Optional[KnowledgeEntry]:
        """Get a knowledge entry."""
        entry = self.entries.get(key)
        if entry:
            entry.access_count += 1
            entry.last_accessed = datetime.now(timezone.utc).isoformat()
        return entry

    # Stop words that should never count as search matches
    _STOP_WORDS = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
        "that", "this", "these", "those", "it", "its", "i", "me", "my", "we",
        "us", "our", "you", "your", "he", "him", "his", "she", "her", "they",
        "them", "their", "and", "or", "but", "not", "no", "nor", "so", "yet",
        "if", "of", "in", "on", "at", "to", "for", "with", "by", "from",
        "about", "into", "through", "during", "before", "after", "above",
        "below", "between", "under", "again", "then", "once", "here", "there",
        "all", "each", "every", "both", "few", "more", "most", "some", "any",
        "other", "such", "only", "own", "same", "than", "too", "very",
        "just", "also", "now", "today", "tell", "give", "get", "got", "know",
    }

    @staticmethod
    def _clean_words(text: str) -> set:
        """Split text into clean lowercase words, stripping punctuation."""
        # Replace common separators with spaces, then strip remaining punctuation
        cleaned = text.lower().replace(":", " ").replace("_", " ").replace("/", " ")
        cleaned = cleaned.replace("-", " ").replace(",", " ").replace(".", " ")
        cleaned = cleaned.replace("(", " ").replace(")", " ").replace("|", " ")
        cleaned = cleaned.replace("$", " ").replace("%", " ").replace("'", " ")
        return set(cleaned.split())

    def search(self, query: str, top_k: int = 3) -> List[tuple]:
        """Keyword search over memory entries (stop words filtered)."""
        query_words = self._clean_words(query) - self._STOP_WORDS
        if not query_words:
            return []

        scores = []
        for key, entry in self.entries.items():
            content_words = self._clean_words(entry.content) - self._STOP_WORDS
            key_words = self._clean_words(key) - self._STOP_WORDS
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
            if learned_time.tzinfo is None:
                learned_time = learned_time.replace(tzinfo=timezone.utc)
            age = datetime.now(timezone.utc) - learned_time
            return age > timedelta(minutes=max_age_minutes)
        except ValueError:
            return True

    def cleanup_live(self, max_age_days: int = 7) -> int:
        """Remove live entries older than max_age_days. Historical and static entries are NEVER touched."""
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=max_age_days)
        to_delete = []
        for key, entry in self.entries.items():
            if entry.category != "live":
                continue
            if not entry.timestamp:
                to_delete.append(key)
                continue
            try:
                ts = datetime.fromisoformat(entry.timestamp)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts < cutoff:
                    to_delete.append(key)
            except (ValueError, TypeError):
                to_delete.append(key)
        for key in to_delete:
            del self.entries[key]
        if to_delete:
            self.save()
        return len(to_delete)

    def stats(self):
        """Print memory statistics."""
        static = sum(1 for e in self.entries.values() if e.category == "static")
        live = sum(1 for e in self.entries.values() if e.category == "live")
        historical = sum(1 for e in self.entries.values() if e.category == "historical")
        print(f"\n  Memory Store (Parquet):")
        print(f"    Total entries: {len(self.entries)}")
        print(f"    Static knowledge: {static}")
        print(f"    Live data: {live}")
        print(f"    Historical: {historical}")
        if os.path.exists(self.path):
            size_kb = os.path.getsize(self.path) / 1024
            print(f"    File: {self.path} ({size_kb:.1f} KB)")


# ==============================================================================
# ENTITY EXTRACTION — Simple NLP for graph ingestion
# ==============================================================================

# Common verbs that indicate relationships between entities
_RELATION_VERBS = [
    "manages", "leads", "owns", "created", "built", "designed", "wrote",
    "founded", "invented", "discovered", "developed", "maintains",
    "reports to", "works for", "works at", "works with", "belongs to",
    "is part of", "is a", "is an", "contains", "includes", "uses",
    "depends on", "requires", "produces", "causes", "supports",
    "located in", "based in", "lives in",
]


def _extract_entities_and_edges(text: str):
    """
    Extract potential entities and relationships from text.

    Entities: capitalized multi-word phrases (e.g. "ML Group", "Ravi").
    Edges: "Entity verb Entity" patterns.

    Returns:
        List of (subject, predicate, object) tuples.
    """
    edges = []
    if not text:
        return edges

    # Find capitalized words/phrases as entity candidates
    # Matches sequences like "Ravi", "ML Group", "AI Team"
    entity_pattern = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)\b')
    entities = entity_pattern.findall(text)

    # Try to find "Entity verb Entity" patterns
    for verb in _RELATION_VERBS:
        # Pattern: "Subject verb Object" where Subject/Object are capitalized
        pattern = re.compile(
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)\s+'
            + re.escape(verb) + r'\s+'
            + r'([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)\b'
        )
        for match in pattern.finditer(text):
            subj = match.group(1).strip()
            obj = match.group(2).strip()
            if subj and obj and subj != obj:
                edges.append((subj, verb.replace(" ", "_"), obj))

    return edges


def _format_graph_facts(edges):
    """
    Format graph edges as readable facts for prompt injection.

    Example output:
        "Known facts: Ravi manages ML Group. ML Group is part of AI Team."
    """
    if not edges:
        return ""

    facts = []
    for e in edges:
        subj = e.get("subject", "").replace("_", " ")
        pred = e.get("predicate", "").replace("_", " ")
        obj = e.get("object", "").replace("_", " ")
        facts.append(f"{subj} {pred} {obj}")

    return "Known facts: " + ". ".join(facts) + "."


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
        self.memory = MemoryStore("memory.parquet")
        self.tools = ToolRegistry()
        self.rag = None        # Set externally if RAG is available
        self.graph = None      # Set externally if knowledge graph is available
        self.historical = None # Set externally if HistoricalStore is available
        self.cache = None      # Set externally if CacheStore is available

        # Data directory (for loading identity.txt etc.)
        self._data_dir = "qor-data"
        if config and hasattr(config, 'runtime') and hasattr(config.runtime, 'data_dir'):
            self._data_dir = config.runtime.data_dir

        # User ID for knowledge tree (set by cmd_run with session_id)
        self._user_id = "user:default"

        # Thread safety — model inference and store writes serialized via locks
        self._model_lock = threading.Lock()   # GPU/CPU model not thread-safe
        self._store_lock = threading.Lock()   # MemoryStore writes (Parquet mutation + flush)

        # Thresholds
        self.confidence_threshold = 0.6    # Below this → don't trust internal knowledge
        self.high_confidence = 0.85        # Above this → very confident, answer directly
        self.hallucination_threshold = 0.3 # Below this → definitely don't know

        # NGRE Brain + TreeGate (set externally if NGRE is available)
        self._ngre_brain = None  # NGREBrain instance
        self._treegate = None    # TreeGate instance (created when brain is set)

        # Routing statistics — track ComplexityGate decisions
        self._routing_stats = {
            "TEMPLATE": 0, "LLM_FAST": 0, "LLM_THINK": 0, "CLOUD": 0,
            "total": 0,
        }

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

    def set_graph(self, graph):
        """Connect a knowledge graph."""
        self.graph = graph
        print(f"  Knowledge graph connected")

    def set_historical(self, historical):
        """Connect a historical store for permanent archive search."""
        self.historical = historical
        print(f"  Historical store connected")

    def set_cache(self, cache):
        """Connect a CacheStore for tool/API result caching."""
        self.cache = cache
        print(f"  Cache store connected")

    def set_ngre_brain(self, brain):
        """Connect NGRE Brain + create TreeGate for answer routing.

        When set, the answer() method routes through TreeGate
        (tree-first, template-fast, strict LLM prompt, post-verification).
        """
        self._ngre_brain = brain
        try:
            from .ngre import TreeGate
            self._treegate = TreeGate(graph=self.graph, user_id=self._user_id)
            print(f"  NGRE brain connected (TreeGate active)")
        except Exception as e:
            self._treegate = None
            print(f"  NGRE brain connected (TreeGate unavailable: {e})")

    def get_routing_stats(self) -> dict:
        """Return routing statistics from ComplexityGate decisions."""
        stats = dict(self._routing_stats)
        total = stats.get("total", 0)
        if total > 0:
            for key in ("TEMPLATE", "LLM_FAST", "LLM_THINK", "CLOUD"):
                stats[f"{key}_pct"] = round(100 * stats[key] / total, 1)
        return stats

    def measure_confidence(self, question: str) -> float:
        """
        Measure how confident QOR is about answering this question.

        Method: Feed the question to the model and measure the
        average perplexity of the generated tokens. Low perplexity
        = model is confident. High perplexity = model is guessing.

        Returns: 0.0 (no idea) to 1.0 (very confident)
        """
        # Encode question + start of answer
        prompt = f"Question: {question}\nAnswer:"
        ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = torch.tensor([ids], device=self.device)

        with self._model_lock:
            self.model.eval()
            with torch.no_grad():
                # Generate a few tokens and measure how "sure" the model is
                result = self.model(input_ids, enable_self_mod=False)
                logits = result["logits"][:, -1, :]

            # Get probability distribution (float32 to avoid fp16 overflow → NaN)
            probs = F.softmax(logits.float(), dim=-1)

            # Signal 1: Entropy (inverted, weight 0.3)
            entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
            max_entropy = torch.log(torch.tensor(float(self.model.config.vocab_size))).item()
            entropy_signal = 1.0 - (entropy / max(max_entropy, 1e-10))

            # Signal 2: Top-1 probability (weight 0.3)
            sorted_probs, _ = probs.sort(descending=True)
            top1_signal = sorted_probs[0, 0].item()

            # Signal 3: Top-5 probability mass (weight 0.2)
            top5_mass = sorted_probs[0, :5].sum().item()
            top5_signal = top5_mass

            # Signal 4: Top-1/Top-2 ratio "decisiveness" (weight 0.2)
            top2 = sorted_probs[0, 1].item()
            decisiveness_signal = (top1_signal / max(top2, 1e-10))
            decisiveness_signal = min(decisiveness_signal / 10.0, 1.0)  # normalize

            # Weighted combination
            confidence = (
                0.3 * entropy_signal +
                0.3 * top1_signal +
                0.2 * top5_signal +
                0.2 * decisiveness_signal
            )

        # NaN guard — if model produces NaN logits, default to medium confidence
        import math
        if math.isnan(confidence):
            confidence = 0.5
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

    def _resolve_followup(self, question: str, chat_context: list) -> str:
        """Resolve follow-up questions by extracting the topic from recent chat.

        If the question contains vague references like "it", "that", "about it",
        "about them", "about this", we look at the last user message to find
        the actual topic, then expand the current question.

        Examples:
            prev: "what is the bitcoin sentiment today"
            curr: "what people are talking about it"
            resolved: "what people are talking about bitcoin"
        """
        q_lower = question.lower().strip()

        # Quick check: does this question have vague references?
        followup_signals = [
            " it", " them", " that", " this", " its ",
            "about it", "about them", "about that", "about this",
            "more about", "tell me more", "what about",
            "and what", "how about", "what else",
        ]
        is_followup = any(sig in f" {q_lower} " or q_lower.startswith(sig.strip())
                          for sig in followup_signals)

        # Also detect very short questions that are clearly follow-ups
        # IMPORTANT: exempt asset names and analysis keywords so
        # "gold analysis", "eth price", "stock market" aren't treated as follow-ups
        _not_followup_kw = {
            # crypto
            "bitcoin", "btc", "eth", "ethereum", "solana", "sol", "xrp",
            "bnb", "doge", "ada", "cardano", "litecoin", "ltc", "avax",
            "matic", "polygon", "link", "chainlink", "crypto",
            # commodities
            "gold", "silver", "platinum", "oil", "copper", "palladium",
            # stock / indices
            "stock", "nasdaq", "dow", "s&p", "nifty", "sensex",
            "aapl", "msft", "tsla", "googl", "amzn", "nvda", "meta",
            # forex
            "forex", "eur", "gbp", "jpy", "usd", "inr",
            # general data keywords
            "price", "analysis", "market", "weather", "news",
            "sentiment", "forecast", "trading", "futures",
            # greetings / commands
            "hello", "hi", "hey", "help", "status", "quit",
        }
        if len(q_lower.split()) <= 4 and not any(
            kw in q_lower for kw in _not_followup_kw
        ):
            is_followup = True

        if not is_followup:
            return question

        # Extract topic from the last user message in chat context
        topic = ""
        for msg in reversed(chat_context):
            if msg["role"] == "user":
                topic = self._extract_topic(msg["content"])
                break

        if not topic:
            return question

        # Replace vague references with the actual topic
        resolved = question
        replacements = [
            ("about it", f"about {topic}"),
            ("about them", f"about {topic}"),
            ("about that", f"about {topic}"),
            ("about this", f"about {topic}"),
        ]
        replaced = False
        for old, new in replacements:
            if old in resolved.lower():
                # Case-insensitive replace
                idx = resolved.lower().index(old)
                resolved = resolved[:idx] + new + resolved[idx + len(old):]
                replaced = True
                break

        if not replaced:
            # Didn't find a specific pattern to replace — prepend topic as context
            resolved = f"{question} (regarding {topic})"

        return resolved

    @staticmethod
    def _extract_topic(text: str) -> str:
        """Extract the main topic/subject from a user message.

        Looks for known asset names, nouns after key prepositions, etc.
        """
        t = text.lower()

        # Known assets / common topics (extend as needed)
        known_topics = {
            "bitcoin": "Bitcoin", "btc": "Bitcoin",
            "ethereum": "Ethereum", "eth": "Ethereum",
            "solana": "Solana", "sol": "Solana",
            "cardano": "Cardano", "ada": "Cardano",
            "dogecoin": "Dogecoin", "doge": "Dogecoin",
            "xrp": "XRP", "ripple": "XRP",
            "bnb": "BNB", "binance coin": "BNB",
            "polkadot": "Polkadot", "dot": "Polkadot",
            "avalanche": "Avalanche", "avax": "Avalanche",
            "polygon": "Polygon", "matic": "Polygon",
            "litecoin": "Litecoin", "ltc": "Litecoin",
            "chainlink": "Chainlink", "link": "Chainlink",
            "gold": "gold", "silver": "silver", "oil": "oil",
            "s&p": "S&P 500", "nasdaq": "NASDAQ", "dow": "Dow Jones",
        }

        for keyword, label in known_topics.items():
            if keyword in t:
                return label

        # Fallback: extract key noun phrases after common patterns
        import re
        # "what is the X", "how is X", "tell me about X"
        patterns = [
            r'(?:what|how)\s+(?:is|are)\s+(?:the\s+)?(.+?)(?:\s+today|\s+now|\?|$)',
            r'tell\s+me\s+about\s+(.+?)(?:\s+today|\s+now|\?|$)',
            r'(?:price|sentiment|analysis)\s+(?:of|for)\s+(.+?)(?:\s+today|\?|$)',
        ]
        for pat in patterns:
            m = re.search(pat, t)
            if m:
                topic = m.group(1).strip()
                # Clean up
                topic = topic.rstrip('?.! ')
                if len(topic) > 2 and len(topic) < 50:
                    return topic

        # Last resort: take the longest meaningful word (>3 chars, not a stop word)
        stop = {"what", "that", "this", "with", "from", "about", "have", "been",
                "your", "their", "they", "will", "would", "could", "should",
                "today", "going", "does", "like", "think", "know"}
        words = t.split()
        candidates = [w.strip('?.!,') for w in words
                       if len(w) > 3 and w.strip('?.!,') not in stop]
        if candidates:
            return candidates[0]

        return ""

    def answer(self, question: str,
               max_new_tokens: int = 200,
               temperature: float = 0.7,
               verbose: bool = True,
               chat_context: list = None,
               force_tool: bool = False) -> dict:
        """
        The main entry point. Ask QOR a question with zero-hallucination protection.

        Args:
            question: The user's current question.
            chat_context: Recent conversation turns as list of
                          {"role": "user"|"assistant", "content": "..."} dicts.
                          Used to resolve follow-up references ("it", "that", etc.)
                          and to include prior turns in the model prompt.
            force_tool: If True, skip tree freshness check and always call
                        tools. Used after user corrections to force re-fetch.

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
        # Resolve follow-up questions using chat context.
        # If the question contains pronouns/references ("it", "that", "this",
        # "about it", "about them") and we have recent context, expand the
        # question by prepending the topic from the last exchange.
        resolved_question = question
        if chat_context:
            resolved_question = self._resolve_followup(question, chat_context)
            if resolved_question != question and verbose:
                print(f"\n  ┌─ Query: {question}")
                print(f"  ├─ Resolved: {resolved_question}")
            else:
                if verbose:
                    print(f"\n  ┌─ Query: {question}")
        else:
            if verbose:
                print(f"\n  ┌─ Query: {question}")

        # Fast path: greetings/casual chat — skip EVERYTHING (no classify, no DB, no tools).
        # Uses /no_think so model responds directly without extended reasoning.
        # Identity personality guide loaded from knowledge/identity.txt for human-like responses.
        _q_stripped = resolved_question.strip().lower().rstrip("!?.,")
        _GREETINGS = {"hi", "hello", "hey", "howdy", "hola", "yo", "sup",
                       "good morning", "good afternoon", "good evening",
                       "good night", "gm", "whats up", "what's up",
                       "how are you", "how r u", "how are u",
                       "how's it going", "hows it going", "hey there",
                       "hi there", "hello there", "thanks", "thank you",
                       "bye", "goodbye", "see you", "ok", "okay",
                       "nice", "cool", "great", "awesome", "good", "fine",
                       "not bad", "alright", "yep", "nope", "lol", "haha",
                       "what's your name", "whats your name", "who are you",
                       "what are you", "what can you do", "help",
                       "tell me about yourself", "introduce yourself"}
        _is_casual = (_q_stripped in _GREETINGS
                      or (len(_q_stripped) < 15 and any(_q_stripped.startswith(g)
                          for g in {"hi ", "hey ", "hello ", "yo ", "thanks ",
                                    "good ", "bye ", "who are", "what's your",
                                    "whats your", "what are", "tell me about"})))
        if _is_casual:
            # Build identity: system prompt (name + interests) + personality from identity.txt
            identity = getattr(self, 'system_prompt',
                               "You are QOR (Qora Neuran AI), a helpful AI assistant.")

            # Load personality guide from identity.txt (cached after first load)
            if not hasattr(self, '_identity_text'):
                self._identity_text = ""
                try:
                    _id_path = os.path.join(
                        getattr(self, '_data_dir', 'qor-data'),
                        "knowledge", "identity.txt")
                    if os.path.exists(_id_path):
                        with open(_id_path, "r", encoding="utf-8") as f:
                            self._identity_text = f.read()
                except Exception:
                    pass
            if self._identity_text:
                identity += "\n\n" + self._identity_text

            identity += "\n/no_think"

            # Include chat history so model can reference past conversations
            chat_turns = []
            if chat_context:
                for msg in chat_context[-4:]:
                    chat_turns.append({"role": msg["role"], "content": msg["content"]})
            messages = [{"role": "system", "content": identity}]
            messages.extend(chat_turns)
            messages.append({"role": "user", "content": question})
            greeting_reply = self._generate(messages, max_new_tokens=80, temperature=0.8)
            if verbose:
                print(f"  ├─ Fast path: casual /no_think + identity.txt")
                print(f"  └─ Answer: {(greeting_reply or '')[:80]}...")
            return {
                "question": question,
                "answer": greeting_reply or "Hey! What's on your mind?",
                "confidence": 0.95,
                "source": "internal",
                "sources": [],
                "sources_used": ["greeting_fast_path"],
                "reasoning": "casual chat — /no_think with personality guide",
                "timestamp": datetime.now().isoformat(),
                "tool_context": [],
            }

        # Step 1: Classify the query (use resolved question for better classification)
        classification = self.classify_query(resolved_question)

        if verbose:
            print(f"  ├─ Confidence: {classification.confidence:.2f}")
            print(f"  ├─ Reason: {classification.reason}")

        # Step 1b: Knowledge tree adjustments (blocked content, lessons, corrections)
        adjustments = None
        if self.graph is not None:
            try:
                from .knowledge_tree import AnswerFilter
                adjustments = AnswerFilter.get_adjustments(
                    resolved_question, self.graph, self._user_id)
            except Exception:
                pass

        # Step 2: Collect context — DB FIRST, tools ONLY if stale/missing
        # DB is the source of truth for chat. API is only called when DB data
        # is older than the staleness threshold (5min for prices, 30min weather, etc.)
        # This way 100k users hit DB, only 1 API call per 5 min.
        answer_text = None
        source = "internal"
        rag_sources = []
        context_parts = []
        tool_name = None
        need_tool = False  # Only true when tree has no fresh data

        # --- Determine staleness threshold for this question ---
        # Use resolved_question so follow-ups like "what about it" match the real topic
        question_lower = resolved_question.lower()

        # Temporal queries ("yesterday", "last week") should NOT trigger
        # tool calls — the user wants historical data from the tree, not
        # fresh data from APIs. Mark as never-stale so tree data is used.
        _temporal_words = {"yesterday", "last week", "last month", "ago",
                           "previous", "earlier", "history", "historical"}
        is_temporal_query = any(tw in question_lower for tw in _temporal_words)

        max_age = None  # None = static data, never stale
        _market_closed = False  # True when the relevant market is closed
        if not is_temporal_query:
            for category, patterns in LIVE_DATA_PATTERNS.items():
                for keyword in patterns["keywords"]:
                    if keyword in question_lower:
                        max_age = patterns["max_age_minutes"]
                        break
                if max_age is not None:
                    break

        # Check if the asset's market is closed (weekend/holiday)
        # When closed, use historical session data from the tree instead of
        # calling tools that would return stale data or fail.
        if max_age is not None:
            try:
                from .asset_context import detect_asset
                from .quant import is_market_open
                _det_asset = detect_asset(resolved_question)
                if _det_asset and _det_asset.asset_type not in (
                        "crypto", "market_overview"):
                    mkt_status = is_market_open(_det_asset.asset_type)
                    if not mkt_status["open"]:
                        _market_closed = True
                        # Don't enforce staleness — tree data from last close
                        # IS the correct answer when market is closed
                        max_age = None
                        if verbose:
                            print(f"  ├─ Market closed: {mkt_status['reason']}, "
                                  f"using last session close data")
            except Exception:
                pass

        # --- Knowledge Tree (THE single database) ---
        # No cache.parquet check. No memory.parquet check.
        # The tree IS the database. Tool results are stored as knowledge
        # nodes with timestamps. Freshness is checked via node timestamps.

        graph_has_data = False
        has_fresh_tree_data = False  # For tool decision (replaces cache_hit + data_entries)

        from .knowledge_tree import tree_search
        tree_results = tree_search(
            query=resolved_question,
            graph=self.graph,
            user_id=self._user_id,
            verbose=verbose,
            ngre_brain=self._ngre_brain,
        )

        for r in tree_results:
            content = r.get("content", "")
            if content:
                context_parts.append(content)
            src = r.get("source", "")
            if src == "graph" or src.startswith("graph:"):
                graph_has_data = True
            # Check freshness via node timestamp (replaces memory.is_stale)
            # Only count high-relevance results (score >= 0.5) as "fresh data"
            # to avoid a BTC node with generic "price" match blocking gold tools.
            ts = r.get("timestamp", "")
            score = r.get("score", 0.0)
            if (ts and max_age is not None and not has_fresh_tree_data
                    and score >= 0.5):
                try:
                    entry_time = datetime.fromisoformat(ts)
                    age_min = (datetime.now() - entry_time).total_seconds() / 60
                    if age_min <= max_age:
                        has_fresh_tree_data = True
                except Exception:
                    pass

        # --- Decide: do we need to call a tool? ---
        # AGGRESSIVE tool usage — prefer real data over hallucination:
        #   1. force_tool=True (user correction) → ALWAYS call tool
        #   2. Live data question AND tree has no fresh data → MUST call tool
        #   3. No tree context at all + tool exists → call tool
        #   4. Confidence < 80% + no fresh data → call tool (tree may have noise)
        # Only skip tools when tree has fresh data AND confidence >= 80%.
        TOOL_CONFIDENCE_THRESHOLD = 0.80
        executor = getattr(self, '_tool_executor', None)

        if force_tool:
            # User correction — bypass all freshness checks, force re-fetch
            need_tool = True
            has_fresh_tree_data = False  # Treat all tree data as stale
            context_parts.clear()  # Drop stale context
            if verbose:
                print(f"  ├─ Tool forced: re-fetching after user correction")
        elif max_age is not None and not has_fresh_tree_data:
            # Live data question with stale/missing tree data
            need_tool = True
            if verbose:
                print(f"  ├─ Tool needed: live data (max_age={max_age}min), "
                      f"tree has no fresh data")
        elif not context_parts:
            # Tree returned nothing — check if a tool can help
            if executor and executor.detect_intent(resolved_question):
                need_tool = True
                if verbose:
                    print(f"  ├─ Tool needed: tree empty, tool available")
        elif (classification.confidence < TOOL_CONFIDENCE_THRESHOLD
              and not has_fresh_tree_data):
            # Below 80% confidence + no fresh data — verify with tools
            if executor and executor.detect_intent(resolved_question):
                need_tool = True
                if verbose:
                    print(f"  ├─ Tool needed: confidence {classification.confidence:.0%} "
                          f"< {TOOL_CONFIDENCE_THRESHOLD:.0%}, verifying with tool")

        if need_tool:
            # When tool is needed, clear stale tree context — tool data is fresher.
            # Keep graph edge facts (structured, always valid).
            pre_tool_context = list(context_parts)  # backup in case tool fails
            if classification.confidence < TOOL_CONFIDENCE_THRESHOLD:
                # Preserve graph edge facts — drop stale knowledge nodes
                graph_context = []
                if graph_has_data and tree_results:
                    for r in tree_results:
                        src = r.get("source", "")
                        if src == "graph" and r.get("content"):
                            graph_context.append(r["content"])
                            break  # Keep top edge-based result
                context_parts.clear()
                context_parts.extend(graph_context)
                rag_sources.clear()

            # Check if this is an asset query — if so, gather ALL related data in parallel
            from .asset_context import detect_asset, gather_asset_context
            asset = detect_asset(resolved_question)

            # Inject market status note (weekend/holiday awareness)
            if asset and asset.asset_type != "market_overview":
                try:
                    from .quant import market_status_note
                    _mkt_note = market_status_note(
                        asset.asset_type, asset.name)
                    if _mkt_note:
                        context_parts.insert(0, _mkt_note)
                        if verbose:
                            print(f"  ├─ Market: {_mkt_note[:60]}...")
                except Exception:
                    pass

            if asset and executor:
                # Parallel: run ALL related tools for this asset type
                asset_results = gather_asset_context(
                    asset, executor,
                    verbose=verbose,
                )
                for t_name, t_result in asset_results:
                    context_parts.append(t_result)
                if asset_results:
                    tool_name = asset_results[0][0]
                    # Save to tree ONLY (no cache, no memory)
                    for t_name, t_result in asset_results:
                        self._save_to_knowledge(t_name, t_result,
                                                resolved_question, verbose)
            else:
                # Single tool (non-asset queries like weather, jokes, etc.)
                try:
                    tool_result = self._try_tool(resolved_question)
                    if tool_result:
                        tool_name, tool_data = tool_result
                        context_parts.append(tool_data)
                        # Save to tree ONLY (no cache, no memory)
                        self._save_to_knowledge(tool_name, tool_data,
                                                resolved_question, verbose)
                        if verbose:
                            print(f"  ├─ Tool: {tool_name} → tree")
                except Exception as e:
                    if verbose:
                        print(f"  ├─ Tool error: {e}")

            # If tool returned nothing, restore pre-tool context as fallback
            if not context_parts and pre_tool_context:
                context_parts.extend(pre_tool_context)
                if verbose:
                    print(f"  ├─ Tool returned nothing, restoring tree context")
        elif max_age is not None and has_fresh_tree_data:
            if verbose:
                print(f"  ├─ Tool: skipped (tree has fresh data)")
            # Still inject market status note even when using tree data
            try:
                from .asset_context import detect_asset
                from .quant import market_status_note
                _asset = detect_asset(resolved_question)
                if _asset and _asset.asset_type != "market_overview":
                    _mkt_note = market_status_note(
                        _asset.asset_type, _asset.name)
                    if _mkt_note:
                        context_parts.insert(0, _mkt_note)
            except Exception:
                pass

        # Inject market-closed note when using historical tree data
        if _market_closed and context_parts:
            try:
                from .asset_context import detect_asset
                from .quant import market_status_note
                _det = detect_asset(resolved_question)
                if _det:
                    _mkt_note = market_status_note(
                        _det.asset_type, _det.name)
                    if _mkt_note and _mkt_note not in context_parts:
                        context_parts.insert(0, _mkt_note)
            except Exception:
                pass

        # Step 2a-filter: Apply knowledge tree adjustments
        if adjustments:
            # Filter out blocked content from context
            if adjustments.blocked_patterns and context_parts:
                filtered = []
                for part in context_parts:
                    part_lower = part.lower() if isinstance(part, str) else ""
                    blocked = any(bp.lower() in part_lower
                                  for bp in adjustments.blocked_patterns
                                  if bp)
                    if not blocked:
                        filtered.append(part)
                context_parts = filtered
            # Inject trade lessons as context
            if adjustments.trade_lessons:
                lessons_text = "Past experience: " + " | ".join(
                    adjustments.trade_lessons[:3])
                context_parts.append(lessons_text)
            # Inject corrections as context
            if adjustments.corrections:
                corr_text = "User corrections: " + " | ".join(
                    adjustments.corrections[:3])
                context_parts.append(corr_text)

        # Step 2b: Match skill → inject instructions into system prompt
        skill_instructions = ""
        skill_loader = getattr(self, '_skill_loader', None)
        if skill_loader:
            matched_skill = skill_loader.match(resolved_question)
            if matched_skill:
                skill_instructions = matched_skill.instructions
                if verbose:
                    print(f"  ├─ Skill: {matched_skill.name}")

        # Step 2c: Detect media (images/audio) in question + context
        # If the model has vision/audio encoders, scan for file paths and load tensors.
        # These get passed to _generate() which splices them into the token sequence.
        images_tensor = None
        mel_specs_tensor = None
        if (getattr(self.model, 'vision_encoder', None) is not None or
                getattr(self.model, 'audio_encoder', None) is not None):
            media = self._detect_media_paths(question, *context_parts)
            if media["images"] and getattr(self.model, 'vision_encoder', None) is not None:
                images_tensor = self._load_image_tensor(media["images"][0])
                if images_tensor is not None and verbose:
                    print(f"  |- Vision: loaded {os.path.basename(media['images'][0])}")
            if media["audio"] and getattr(self.model, 'audio_encoder', None) is not None:
                mel_specs_tensor = self._load_audio_tensor(media["audio"][0])
                if mel_specs_tensor is not None and verbose:
                    print(f"  |- Audio: loaded {os.path.basename(media['audio'][0])}")

        # Step 2d: Generate answer — route through NGRE TreeGate if available,
        # otherwise use the standard frozen-model path.
        #
        # NGRE TreeGate flow:
        #   1. TreeGate.evaluate() → tier + routing + strict prompt
        #   2. TEMPLATE route → try_template_response() (< 2ms, no LLM)
        #   3. LLM_FAST → /no_think, LLM_THINK → /think, CLOUD → external
        #   4. post_verify() → check claims against context
        #
        # Standard flow (no NGRE):
        #   Model generates with context → "Using only the context above, answer"

        identity = getattr(self, 'system_prompt',
                           "You are QOR (Qora Neuran AI), a helpful AI assistant.")
        # Load personality guide (cached after first load in greeting path)
        if not hasattr(self, '_identity_text'):
            self._identity_text = ""
            try:
                _id_path = os.path.join(self._data_dir, "knowledge", "identity.txt")
                if os.path.exists(_id_path):
                    with open(_id_path, "r", encoding="utf-8") as f:
                        self._identity_text = f.read()
            except Exception:
                pass
        if self._identity_text:
            identity += "\n\n" + self._identity_text
        if skill_instructions:
            identity += f"\n\n## Active Skill Instructions\n{skill_instructions}"

        # Build conversation messages — include recent chat history for continuity
        chat_turns = []
        if chat_context:
            recent = chat_context[-6:]
            for msg in recent:
                chat_turns.append({"role": msg["role"], "content": msg["content"]})

        # --- NGRE TreeGate path (when brain is available) ---
        _ngre_used = False
        routing_level = None

        if self._treegate is not None and context_parts:
            try:
                from .ngre import (TreeGate, ComplexityGate, STRICT_PROMPT_TEMPLATE,
                                   TIER_NONE, ROUTE_TEMPLATE, ROUTE_LLM_FAST,
                                   ROUTE_LLM_THINK, ROUTE_CLOUD)

                # Determine tool_result for TreeGate
                _tool_result = None
                if tool_name and context_parts:
                    _tool_result = (tool_name, context_parts[-1])

                gate_result = self._treegate.evaluate(
                    question=resolved_question,
                    tree_results=tree_results,
                    adjustments=adjustments,
                    tool_result=_tool_result,
                )

                routing_level = gate_result.routing.level

                # Track routing stats
                self._routing_stats["total"] += 1
                self._routing_stats[routing_level] = (
                    self._routing_stats.get(routing_level, 0) + 1)

                if verbose:
                    print(f"  ├─ TreeGate: tier={gate_result.tier}, "
                          f"route={routing_level} "
                          f"({gate_result.routing.reason})")

                # TIER_NONE → zero hallucination
                if gate_result.tier == TIER_NONE and not context_parts:
                    source = "unknown"
                    answer_text = (
                        "I don't have enough information to answer this accurately. "
                        "I'd rather be honest than guess and give you wrong information.")
                    _ngre_used = True

                # TEMPLATE route: try ultra-fast response (< 2ms, no LLM)
                elif routing_level == ROUTE_TEMPLATE:
                    template_answer = self._treegate.try_template_response(
                        resolved_question, gate_result.context_parts or context_parts)
                    if template_answer:
                        answer_text = template_answer
                        source = "template"
                        _ngre_used = True
                        if verbose:
                            print(f"  ├─ Template: {answer_text[:60]}...")

                # LLM routes: use STRICT_PROMPT_TEMPLATE
                if not _ngre_used and routing_level in (
                        ROUTE_LLM_FAST, ROUTE_LLM_THINK, ROUTE_CLOUD):
                    # Use the strict prompt from TreeGate if available
                    strict_prompt = gate_result.prompt
                    if not strict_prompt and context_parts:
                        max_ctx = 12000
                        ctx = ""
                        for part in context_parts:
                            if len(ctx) + len(part) + 1 > max_ctx:
                                break
                            ctx += ("\n" if ctx else "") + part
                        strict_prompt = STRICT_PROMPT_TEMPLATE.format(
                            context=ctx, question=question)

                    if strict_prompt:
                        # /no_think for FAST, /think for THINK/CLOUD
                        think_directive = "/no_think"
                        if routing_level in (ROUTE_LLM_THINK, ROUTE_CLOUD):
                            think_directive = "/think"

                        _sys = identity + f"\n{think_directive}"
                        messages = [{"role": "system", "content": _sys}]
                        messages.extend(chat_turns)
                        messages.append({"role": "user", "content": strict_prompt})

                        answer_text = self._generate(
                            messages, max_new_tokens, temperature,
                            images=images_tensor, mel_specs=mel_specs_tensor)

                        # Post-verify: check claims against context
                        if answer_text:
                            verify = TreeGate.post_verify(
                                answer_text,
                                gate_result.context_parts or context_parts)
                            if not verify["verified"] and verbose:
                                print(f"  ├─ Post-verify: "
                                      f"{len(verify['flags'])} ungrounded claims")

                        # Source attribution
                        if tool_name:
                            source = f"tool:{tool_name}"
                        elif graph_has_data:
                            source = "knowledge_tree"
                        else:
                            source = "knowledge_tree"
                        _ngre_used = True
            except Exception as e:
                # TreeGate failed — fall through to standard path
                if verbose:
                    print(f"  ├─ TreeGate error: {e}")
                _ngre_used = False

        # --- Standard path (no NGRE, or NGRE fallthrough) ---
        if not _ngre_used:
            identity += "\n/no_think"
            if context_parts:
                max_context_chars = 12000
                context = ""
                for part in context_parts:
                    if len(context) + len(part) + 1 > max_context_chars:
                        break
                    if context:
                        context += "\n"
                    context += part
                messages = [{"role": "system", "content": identity}]
                messages.extend(chat_turns)
                messages.append(
                    {"role": "user",
                     "content": f"The following is current live data from "
                                f"our database and tools (not user-provided):"
                                f"\n\n{context}\n\n"
                                f"Based on this real-time data, answer "
                                f"concisely: {question}"},
                )
                answer_text = self._generate(
                    messages, max_new_tokens, temperature,
                    images=images_tensor, mel_specs=mel_specs_tensor)
                if tool_name:
                    source = f"tool:{tool_name}"
                elif graph_has_data:
                    source = "knowledge_tree"
                elif rag_sources:
                    source = "knowledge_base"
                else:
                    source = "knowledge_tree"
            else:
                messages = [{"role": "system", "content": identity}]
                messages.extend(chat_turns)
                messages.append({"role": "user", "content": question})
                answer_text = self._generate(
                    messages, max_new_tokens, temperature,
                    images=images_tensor, mel_specs=mel_specs_tensor)
                source = "internal"

        # Zero hallucination guard — only when NO context and low confidence
        if answer_text is None or (not context_parts and classification.confidence < self.hallucination_threshold):
            source = "unknown"
            answer_text = (
                "I don't have enough information to answer this accurately. "
                "I'd rather be honest than guess and give you wrong information."
            )

        # No learning during answer path — model is frozen.
        # Learning happens in consolidation (runtime.py).
        # But Hebbian edge updates DO fire here — reinforce connections
        # between nodes that contributed to a successful answer.
        if (self.graph is not None and tree_results
                and source != "unknown"
                and hasattr(self.graph, 'hebbian_update')):
            try:
                # Get node IDs that contributed to this answer
                contributing_ids = []
                for r in tree_results:
                    nid = r.get("node_id", "") or r.get("id", "")
                    if nid:
                        contributing_ids.append(nid)
                # Pairwise positive Hebbian on contributing nodes
                for i in range(len(contributing_ids)):
                    for j in range(i + 1, min(i + 3, len(contributing_ids))):
                        self.graph.hebbian_update(
                            contributing_ids[i], contributing_ids[j],
                            reward=0.5)  # Moderate positive reward
            except Exception:
                pass  # Hebbian is best-effort, never block answers

        if verbose:
            print(f"  ├─ Source: {source}")
            print(f"  └─ Answer: {answer_text[:100]}...")

        # Track which sources contributed data
        sources_used = []
        if tree_results:
            tree_srcs = set(r.get("source", "") for r in tree_results)
            sources_used.append(f"tree:{','.join(tree_srcs)}")
        if self.graph is not None and any("→" in p for p in context_parts[:10] if isinstance(p, str)):
            sources_used.append("graph")
        if rag_sources:
            sources_used.append(f"rag:{len(rag_sources)}")
        if tool_name:
            sources_used.append(f"tool:{tool_name}")

        result = {
            "question": question,
            "answer": answer_text,
            "confidence": classification.confidence,
            "source": source,
            "sources": rag_sources,
            "sources_used": sources_used,
            "reasoning": classification.reason,
            "timestamp": datetime.now().isoformat(),
            "tool_context": context_parts[:5] if context_parts else [],
        }
        if routing_level is not None:
            result["routing"] = routing_level
        return result

    # ==================================================================
    # MULTIMODAL — Detect, load, and pass images/audio to the model
    # ==================================================================

    def _detect_media_paths(self, *texts) -> dict:
        """Detect image/audio file paths in one or more text strings.

        Scans the user question AND context (tool results, memory) for
        file paths that point to existing image or audio files.

        Returns:
            {"images": [path, ...], "audio": [path, ...]}
        """
        media = {"images": [], "audio": []}
        seen = set()
        for text in texts:
            if not text:
                continue
            for match in _MEDIA_PATH_RE.finditer(str(text)):
                path = match.group().strip()
                if path in seen:
                    continue
                ext = os.path.splitext(path)[1].lower()
                if not os.path.isfile(path):
                    continue
                seen.add(path)
                if ext in _IMAGE_EXTS:
                    media["images"].append(path)
                elif ext in _AUDIO_EXTS:
                    media["audio"].append(path)
        return media

    def _load_image_tensor(self, image_path: str) -> Optional[torch.Tensor]:
        """Load and preprocess an image for the vision encoder.

        Handles both pretrained (SigLIP — 384x384, normalized) and
        custom (any size, optional grayscale) vision configs.

        Returns:
            (1, C, H, W) tensor on self.device, or None if loading fails.
        """
        vision_config = getattr(self.model, 'vision_config', None)
        if vision_config is None or self.model.vision_encoder is None:
            return None

        try:
            from PIL import Image
        except ImportError:
            return None

        try:
            img = Image.open(image_path).convert("RGB")
        except Exception:
            return None

        if vision_config.use_pretrained:
            size = vision_config.pretrained_image_size
            channels = 3
        else:
            size = vision_config.image_size
            channels = vision_config.in_channels

        try:
            from torchvision import transforms as T
            tlist = [T.Resize((size, size))]
            if channels == 1:
                tlist.append(T.Grayscale(num_output_channels=1))
            tlist.append(T.ToTensor())
            if vision_config.use_pretrained:
                tlist.append(T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
            transform = T.Compose(tlist)
            tensor = transform(img).unsqueeze(0)
        except ImportError:
            # Fallback without torchvision
            import numpy as np
            img = img.resize((size, size))
            if channels == 1:
                img = img.convert("L")
            arr = np.array(img, dtype=np.float32) / 255.0
            if vision_config.use_pretrained:
                arr = (arr - 0.5) / 0.5
            if arr.ndim == 2:
                arr = arr[..., np.newaxis]
            tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

        return tensor.to(self.device)

    def _load_audio_tensor(self, audio_path: str) -> Optional[torch.Tensor]:
        """Load audio file and compute mel spectrogram for the audio encoder.

        Handles both pretrained (Whisper — feature extractor) and
        custom (torchaudio mel spectrogram) audio configs.

        Returns:
            (1, n_mels, n_frames) tensor on self.device, or None if loading fails.
        """
        audio_config = getattr(self.model, 'audio_config', None)
        if audio_config is None or self.model.audio_encoder is None:
            return None

        if audio_config.use_pretrained:
            try:
                from .audio import PretrainedAudioEncoder
            except ImportError:
                return None
            try:
                # Load audio — try librosa first, then soundfile
                waveform = None
                try:
                    import librosa
                    waveform, _ = librosa.load(audio_path, sr=16000, mono=True)
                except ImportError:
                    import soundfile as sf
                    import numpy as np_sf
                    waveform, sr = sf.read(audio_path)
                    if waveform.ndim > 1:
                        waveform = waveform.mean(axis=1)
                    if sr != 16000:
                        # Simple linear resampling
                        ratio = 16000 / sr
                        indices = np_sf.arange(0, len(waveform) * ratio) / ratio
                        indices = np_sf.clip(indices, 0, len(waveform) - 1).astype(int)
                        waveform = waveform[indices]
                if waveform is None:
                    return None
                mel = PretrainedAudioEncoder.compute_mel_spectrogram(waveform, audio_config)
                return mel.to(self.device)  # (1, 80, 3000)
            except Exception:
                return None
        else:
            # Custom audio encoder
            try:
                import torchaudio
            except ImportError:
                return None
            try:
                waveform, sr = torchaudio.load(audio_path)
                if sr != audio_config.sample_rate:
                    waveform = torchaudio.functional.resample(
                        waveform, sr, audio_config.sample_rate)
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                from .audio import AudioEncoder
                mel = AudioEncoder.compute_mel_spectrogram(waveform, audio_config)
                return mel.to(self.device)
            except Exception:
                return None

    # Patterns that signal the model has started generating filler after the real answer
    _STOP_PATTERNS = [
        "\nNote ",
        "\nNote:",
        "\nFor example",
        "\nQuestion:",
        "\nQ:",
        "\nContext:",
    ]

    def _generate(self, prompt, max_new_tokens: int = 200,
                   temperature: float = 0.7,
                   images: Optional[torch.Tensor] = None,
                   mel_specs: Optional[torch.Tensor] = None) -> str:
        """Generate text from a prompt with optional multimodal inputs.

        Args:
            prompt: Either a list of message dicts (chat format) or a plain string.
            images: (1, C, H, W) image tensor — vision encoder processes this.
            mel_specs: (1, n_mels, n_frames) mel spectrogram — audio encoder processes this.

        When images or mel_specs are provided, placeholder tokens are spliced
        into the token sequence. model.forward() replaces those placeholders
        with encoder embeddings during the initial prefill pass.
        """
        if isinstance(prompt, list):
            # Chat messages — use tokenizer's chat template
            ids = self.tokenizer.format_chat(prompt, add_generation_prompt=True)
        else:
            ids = self.tokenizer.encode(prompt, add_special_tokens=True)

        image_positions = None
        audio_positions = None

        # Insert image placeholder tokens into the token sequence
        if images is not None and self.model.vision_encoder is not None:
            patch_id = self.tokenizer.image_patch_id
            start_id = self.tokenizer.image_start_id
            end_id = self.tokenizer.image_end_id

            if patch_id is not None and start_id is not None:
                n_patches = self.model.vision_config.n_patches
                img_tokens = [start_id] + [patch_id] * n_patches + [end_id]
                # Insert after BOS (position 0) — image comes first, then text
                ids = [ids[0]] + img_tokens + ids[1:]
                # Compute positions of all patch placeholder tokens
                positions = [i for i, t in enumerate(ids) if t == patch_id]
                image_positions = torch.tensor([positions], device=self.device)

        # Insert audio placeholder tokens into the token sequence
        if mel_specs is not None and self.model.audio_encoder is not None:
            frame_id = self.tokenizer.audio_frame_id
            start_id = self.tokenizer.audio_start_id
            end_id = self.tokenizer.audio_end_id

            if frame_id is not None and start_id is not None:
                # Compute how many tokens the audio encoder will produce
                audio_config = self.model.audio_config
                if audio_config.use_pretrained:
                    # Whisper: 3000 mel frames -> 1500 encoder outputs -> truncate
                    n_audio_tokens = min(1500, audio_config.max_audio_tokens)
                else:
                    n_frames = mel_specs.shape[2]
                    n_audio_tokens = min(
                        math.ceil(n_frames / audio_config.frame_stride),
                        audio_config.max_audio_tokens,
                    )
                audio_tokens = [start_id] + [frame_id] * n_audio_tokens + [end_id]
                # Insert after BOS (or after image tokens if present)
                ids = [ids[0]] + audio_tokens + ids[1:]
                # Compute positions of all audio frame placeholder tokens
                positions = [i for i, t in enumerate(ids) if t == frame_id]
                audio_positions = torch.tensor([positions], device=self.device)

        # Truncate prompt if it exceeds max_seq_len (leave room for generation)
        max_seq = getattr(self.model.config, 'max_seq_len', 8192)
        max_prompt = max_seq - max_new_tokens - 16  # 16 tokens safety margin
        if len(ids) > max_prompt:
            # Keep the END of the prompt (question + recent context), trim older context
            ids = ids[-max_prompt:]

        prompt_len = len(ids)
        input_ids = torch.tensor([ids], device=self.device)

        with self._model_lock:
            self.model.eval()
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=50,
                top_p=0.9,
                stop_tokens=[self.tokenizer.eos_id],
                images=images,
                image_positions=image_positions,
                mel_specs=mel_specs,
                audio_positions=audio_positions,
            )

        # Only decode the NEWLY generated tokens (skip the prompt)
        new_ids = output_ids[prompt_len:]
        answer = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        # Strip <think>...</think> reasoning tags from model output
        answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
        # Also strip orphaned opening/closing tags
        answer = answer.replace('</think>', '').replace('<think>', '').strip()

        # Truncate at first stop pattern (filler after the real answer)
        for pattern in self._STOP_PATTERNS:
            pos = answer.find(pattern)
            if pos > 0:  # Must have SOME answer before truncating
                answer = answer[:pos].strip()
                break

        return answer

    def _save_to_knowledge(self, tool_name: str, tool_data: str,
                           question: str, verbose: bool = False):
        """Save tool result INTO the knowledge tree (graph).

        Every tool result becomes a graph node with full content + entity edges.
        The tree is the SINGLE source of truth — the AI reads only from here.

        Args:
            tool_name: Name of the tool that produced the data.
            tool_data: The tool result text.
            question: The user's question (for keying).
            verbose: Print debug output.
        """
        if (self.graph is None or not hasattr(self.graph, 'is_open')
                or not self.graph.is_open):
            return

        import hashlib as _hl

        # 1. Store full result as knowledge node (tree_search finds these)
        # Deterministic ID from tool+question — same question OVERWRITES
        # the old node with fresh data + fresh timestamp. No stale duplicates.
        try:
            know_hash = _hl.sha256(
                (tool_name + ":" + question[:80]).encode()).hexdigest()[:8]
            know_id = f"know:{know_hash}"

            # Compute embedding via NGRE brain (Mamba 24-layer → 768-dim)
            embedding = None
            if self._ngre_brain is not None:
                try:
                    embedding = self._ngre_brain.compute_embedding(
                        tool_data[:2000])
                except Exception:
                    pass

            self.graph.add_node(know_id, node_type="knowledge", properties={
                "content": tool_data[:500],
                "source": tool_name,
                "question": question[:100],
                "timestamp": datetime.now().isoformat(),
            },
                embedding=embedding,
                text_summary=tool_data[:200],
            )
            # Discovery edge: user → learned → knowledge node
            self.graph.add_edge(self._user_id, "learned", know_id,
                                confidence=0.85, source=f"tool:{tool_name}")
            if verbose:
                print(f"  ├─ Tree: knowledge node {know_id} stored")
        except Exception:
            pass

        # 2. Extract entities → graph edges (for semantic_query traversal)
        try:
            edges = _extract_entities_and_edges(tool_data)
            for subj, pred, obj in edges:
                self.graph.add_edge(subj, pred, obj,
                                    confidence=0.85, source=f"tool:{tool_name}")
            if edges and verbose:
                print(f"  ├─ Tree: {len(edges)} entity edges extracted")
        except Exception:
            pass

        # 3. Classify historical → store as historical_event node in tree
        try:
            from .runtime import classify_historical
            if classify_historical(tool_data, tool_name):
                evt_hash = _hl.sha256(
                    tool_data[:200].encode()).hexdigest()[:8]
                evt_id = f"hist:{evt_hash}"
                # Reuse embedding computed above (same tool_data)
                hist_emb = embedding  # from step 1
                self.graph.add_node(evt_id, node_type="historical_event",
                                    properties={
                                        "content": tool_data[:200],
                                        "source": tool_name,
                                        "timestamp": datetime.now().isoformat(),
                                    },
                                    embedding=hist_emb,
                                    text_summary=tool_data[:200],
                                    )
                if verbose:
                    print(f"  ├─ Tree: historical event {evt_id}")
        except Exception:
            pass

    def _try_tool(self, question: str) -> Optional[Tuple[str, str]]:
        """Try to find and call a tool for the question.

        Returns (tool_name, tool_data) or None.
        Uses ToolExecutor (with cache + rate limit) first, then PluginManager.
        """
        # Try ToolExecutor (31 built-in tools, cached, rate-limited)
        executor = getattr(self, '_tool_executor', None)
        if executor:
            tool_name = executor.detect_intent(question)
            if tool_name:
                result = executor.call(tool_name, question)
                if result and not result.startswith(("[Tool", "Tool error", "Rate limit")):
                    return (tool_name, result)

        # Try PluginManager (plugins + config tools)
        pm = getattr(self, '_plugin_manager', None)
        if pm:
            pm_name = pm.detect_intent(question)
            if pm_name:
                result = pm.call(pm_name, question)
                if result and not result.startswith(("[Tool", "Tool error")):
                    return (pm_name, result)

        return None

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

"""
QOR Runtime — Continuous Background Reading + Direct DB Storage
================================================================
The model reads from APIs/tools in the background and stores data
DIRECTLY into the database (MemoryStore + Graph + RAG). No batching.

Periodic cleanup prunes old data, rotates checkpoints, decays CMS, compacts graph.

Architecture:
  READ LOOP (background threads) → Direct to MemoryStore + Graph + RAG
  CLEANUP (periodic timer) → 4 steps: live memory prune, checkpoint rotation,
                              CMS slow decay, graph compaction
"""

import os
import glob
import json
import time
import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq

from .config import QORConfig
from .tools import ToolExecutor

logger = logging.getLogger(__name__)


def _get_memory_percent() -> float:
    """Get system memory usage percentage. Uses psutil if available, else /proc."""
    try:
        import psutil
        return psutil.virtual_memory().percent
    except ImportError:
        pass
    # Fallback: read /proc/meminfo (Linux)
    try:
        with open("/proc/meminfo", "r") as f:
            info = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    info[parts[0].rstrip(":")] = int(parts[1])
            total = info.get("MemTotal", 1)
            avail = info.get("MemAvailable", total)
            return (1.0 - avail / total) * 100.0
    except Exception:
        pass
    # Windows fallback: ctypes
    try:
        import ctypes
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(stat)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return float(stat.dwMemoryLoad)
    except Exception:
        return 0.0  # Unknown — assume no pressure


# ==============================================================================
# Arrow schema for batch files
# ==============================================================================

BATCH_SCHEMA = pa.schema([
    ("text", pa.string()),
    ("source", pa.string()),
    ("priority", pa.int8()),
    ("timestamp", pa.timestamp("us", tz="UTC")),
])


# ==============================================================================
# HistoricalStore — Permanent archive (never auto-deleted)
# ==============================================================================

class HistoricalStore:
    """
    Parquet-based permanent archive for historically significant events.
    Data here is NEVER auto-deleted — it's the long-term institutional memory.
    SHA-256 hash chain for tamper detection on every entry.
    """

    def __init__(self, historical_dir: str = "historical"):
        self.dir = historical_dir
        self.path = os.path.join(historical_dir, "historical.parquet")
        os.makedirs(self.dir, exist_ok=True)
        self._entries = []  # list of dicts
        self._dirty = False
        self._chain_head = ""   # Track hash chain head for new entries
        self._load()

    def _load(self):
        """Load existing historical entries from Parquet."""
        if os.path.exists(self.path):
            try:
                from .confidence import MEMORY_SCHEMA
                # Try new schema first, fall back for backward compat
                try:
                    table = pq.read_table(self.path, schema=MEMORY_SCHEMA)
                except Exception:
                    table = pq.read_table(self.path)
                ts_col = table.column("timestamp").cast(pa.int64())
                la_col = table.column("last_accessed").cast(pa.int64())

                # Hash columns (may not exist in old parquet files)
                has_hash = "data_hash" in table.schema.names
                dh_col = table.column("data_hash") if has_hash else None
                ph_col = table.column("prev_hash") if has_hash else None

                for i in range(table.num_rows):
                    ts_us = ts_col[i].as_py()
                    la_us = la_col[i].as_py()
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

                    dh = dh_col[i].as_py() or "" if dh_col else ""
                    ph = ph_col[i].as_py() or "" if ph_col else ""

                    self._entries.append({
                        "key": table.column("key")[i].as_py() or "",
                        "content": table.column("content")[i].as_py() or "",
                        "source": table.column("source")[i].as_py() or "",
                        "category": table.column("category")[i].as_py() or "historical",
                        "confidence": table.column("confidence")[i].as_py() or 1.0,
                        "timestamp": ts_iso,
                        "access_count": table.column("access_count")[i].as_py() or 0,
                        "last_accessed": la_iso,
                        "data_hash": dh,
                        "prev_hash": ph,
                    })
                    # Restore chain head from last entry's hash
                    if dh:
                        self._chain_head = dh
            except Exception as e:
                logger.warning(f"[HistoricalStore] Failed to load {self.path}: {e}")

    def store(self, key: str, content: str, source: str, confidence: float = 1.0):
        """Store a historically significant event with SHA-256 hash."""
        from .crypto import QORCrypto
        prev_hash = self._chain_head
        data_hash = QORCrypto.hash_sha256(content)
        self._chain_head = data_hash

        self._entries.append({
            "key": key,
            "content": content,
            "source": source,
            "category": "historical",
            "confidence": confidence,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "access_count": 0,
            "last_accessed": "",
            "data_hash": data_hash,
            "prev_hash": prev_hash,
        })
        self._dirty = True

    def save(self):
        """Flush all entries to Parquet."""
        if not self._entries:
            return
        from .confidence import MEMORY_SCHEMA, MemoryStore

        keys = [e["key"] for e in self._entries]
        contents = [e["content"] for e in self._entries]
        sources = [e["source"] for e in self._entries]
        categories = [e["category"] for e in self._entries]
        confidences = [e["confidence"] for e in self._entries]
        timestamps = [MemoryStore._iso_to_us(e["timestamp"]) for e in self._entries]
        access_counts = [e["access_count"] for e in self._entries]
        last_accessed = [MemoryStore._iso_to_us(e["last_accessed"]) for e in self._entries]
        data_hashes = [e.get("data_hash", "") for e in self._entries]
        prev_hashes = [e.get("prev_hash", "") for e in self._entries]

        batch = pa.RecordBatch.from_pydict(
            {
                "key": keys,
                "content": contents,
                "source": sources,
                "category": categories,
                "confidence": confidences,
                "timestamp": timestamps,
                "access_count": access_counts,
                "last_accessed": last_accessed,
                "data_hash": data_hashes,
                "prev_hash": prev_hashes,
            },
            schema=MEMORY_SCHEMA,
        )
        table = pa.Table.from_batches([batch])
        pq.write_table(table, self.path)
        self._dirty = False

    def count(self) -> int:
        """Number of historical entries."""
        return len(self._entries)

    # Same stop words as MemoryStore — prevents "is", "what", "today" matching everything
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

    def search(self, query: str, top_k: int = 3) -> list:
        """Keyword search over historical entries (stop words filtered)."""
        query_words = set(query.lower().split()) - self._STOP_WORDS
        if not query_words:
            return []
        scores = []
        for entry in self._entries:
            content_words = set(entry["content"].lower().split()) - self._STOP_WORDS
            key_words = set(entry["key"].lower().split()) - self._STOP_WORDS
            overlap = len(query_words & (content_words | key_words))
            if overlap > 0:
                scores.append((entry["key"], entry, overlap))
        scores.sort(key=lambda x: x[2], reverse=True)
        return scores[:top_k]


# ==============================================================================
# Historical classifier — determines if data is historically significant
# ==============================================================================

# Sources that are ALWAYS live regardless of keyword matches
ALWAYS_LIVE_SOURCES = {"weather", "traffic", "availability"}

# Keywords grouped by category that signal historical significance
_HISTORICAL_KEYWORDS = {
    # ── Markets & Finance ──
    "market_milestones": [
        "all-time high", "all time high", "record high", "record low",
        "market crash", "flash crash", "trillion", "billion milestone",
        "highest ever", "lowest ever", "market cap record",
    ],
    "financial": [
        "ipo", "acquisition", "merger", "halving", "etf approved",
        "bankruptcy", "bailout", "default", "devaluation",
        "rate cut", "rate hike", "interest rate", "fed decision",
        "stock split", "delisted", "circuit breaker", "black monday",
        "black swan", "recession", "depression", "hyperinflation",
        "unicorn", "valued at", "billion valuation", "funding round",
        "series a", "series b", "series c", "series d",
        "went public", "listing", "largest funding",
    ],
    "budget_economy": [
        "budget announced", "union budget", "federal budget",
        "fiscal year", "budget deficit", "budget surplus",
        "gdp growth", "gdp falls", "gdp shrinks",
        "inflation rate", "unemployment rate", "trade deficit",
        "trade surplus", "national debt", "stimulus package",
        "economic reform", "tax reform", "tax cut", "tax hike",
        "austerity", "debt ceiling", "fiscal cliff",
        "central bank", "monetary policy", "quantitative easing",
    ],

    # ── Politics & Governance ──
    "political": [
        "elected", "inaugurated", "impeached", "treaty", "legislation",
        "sanctions", "executive order", "supreme court ruled",
        "constitutional", "declaration", "assassination", "assassinated",
        "coup", "regime change", "war declared", "ceasefire",
        "peace agreement", "resigned", "abdicated", "overthrown",
        "new president", "new prime minister", "sworn in",
    ],
    "elections": [
        "election result", "election results", "wins election",
        "won election", "loses election", "lost election",
        "landslide victory", "majority government", "hung parliament",
        "runoff election", "referendum result", "voter turnout",
        "electoral college", "concedes", "concession speech",
        "recount", "disputed election", "coalition government",
        "prime minister elected", "president elected",
        "governor elected", "mayor elected",
    ],

    # ── Military & Defense ──
    "military": [
        "missile launch", "missile test", "nuclear test",
        "nuclear weapon", "weapons test", "arms deal",
        "military strike", "airstrike", "air strike",
        "invasion", "troops deployed", "military operation",
        "drone strike", "naval blockade", "arms embargo",
        "chemical weapons", "biological weapon",
        "military coup", "martial law", "conscription",
        "defense budget", "arms race", "hypersonic",
        "icbm", "ballistic missile", "warhead",
    ],

    # ── Space & Satellites ──
    "space": [
        "satellite launch", "satellite launched", "rocket launch",
        "space mission", "moon mission", "mars mission",
        "space station", "spacewalk", "orbit",
        "rocket landing", "reusable rocket", "space debris",
        "asteroid", "comet", "exoplanet", "black hole",
        "james webb", "hubble", "telescope discovers",
        "starship", "falcon", "crew dragon", "soyuz",
        "lunar landing", "rover landing", "sample return",
        "space tourism", "commercial spaceflight",
    ],

    # ── Science & Technology ──
    "technology": [
        "launched", "launches", "discovered", "discovers", "invented",
        "breakthrough", "nobel prize", "pulitzer", "first flight",
        "moon landing", "fusion energy", "quantum supremacy",
        "first manned", "first human", "maiden voyage",
        "new invention", "patent granted", "patent filed",
        "scientific discovery", "gene therapy", "crispr",
        "quantum computing", "quantum computer",
        "nuclear fusion", "superconductor",
        "room temperature", "particle discovered",
    ],
    "ai_computing": [
        "artificial intelligence", "ai breakthrough", "ai model",
        "large language model", "gpt", "chatgpt", "claude",
        "generative ai", "ai regulation", "ai safety",
        "agi", "artificial general intelligence",
        "machine learning breakthrough", "deepmind",
        "ai chip", "neural network", "transformer model",
        "ai beats", "ai surpasses", "ai outperforms",
        "autonomous vehicle", "self-driving",
        "robotics breakthrough", "humanoid robot",
    ],

    # ── Disasters & Climate ──
    "disasters": [
        "earthquake", "tsunami", "hurricane", "pandemic", "eruption",
        "wildfire", "famine", "refugee crisis", "mass casualty",
        "cyclone", "typhoon", "tornado", "flooding", "flood",
        "landslide", "avalanche", "drought", "heat wave",
        "climate record", "hottest year", "coldest",
        "species extinct", "oil spill", "nuclear accident",
        "plane crash", "train derailment", "shipwreck",
        "building collapse", "mine collapse", "dam burst",
    ],

    # ── First-time / Historic Events ──
    "first_time_events": [
        "first ever", "first time", "unprecedented", "historic",
        "landmark", "groundbreaking", "never before", "makes history",
        "history-making", "world record", "guinness record",
        "record-breaking", "record breaking", "youngest ever",
        "oldest ever", "largest ever", "smallest ever",
    ],

    # ── Sports ──
    "olympics": [
        "olympic", "olympics", "gold medal", "silver medal",
        "bronze medal", "olympic record", "olympic champion",
        "paralympic", "winter olympics", "summer olympics",
        "opening ceremony", "closing ceremony",
        "olympic qualifier", "olympic debut",
        "world record", "personal best",
    ],
    "cricket": [
        "cricket world cup", "icc world cup", "ashes",
        "test match", "odi", "t20 world cup", "ipl",
        "century scored", "hat-trick", "five-wicket",
        "cricket final", "won the series", "lost the series",
        "test series", "cricket championship",
        "highest score", "fastest century", "double century",
        "all out", "won by", "beats", "defeated",
    ],
    "football_soccer": [
        "world cup", "fifa", "champions league", "premier league",
        "la liga", "bundesliga", "serie a", "ligue 1",
        "euro cup", "copa america", "african cup",
        "goal scored", "hat trick", "penalty shootout",
        "red card", "transfer fee", "transfer record",
        "ballon d'or", "golden boot", "golden ball",
        "promoted", "relegated", "won the league",
        "won the cup", "cup final", "league title",
    ],
    "american_sports": [
        "super bowl", "nfl", "nba finals", "nba championship",
        "world series", "mlb", "stanley cup", "nhl",
        "march madness", "ncaa", "mls cup",
        "touchdown", "home run", "slam dunk", "triple-double",
        "mvp", "rookie of the year", "hall of fame",
        "draft pick", "first overall pick", "trade deadline",
        "playoff", "playoffs", "championship game",
        "won the title", "clinched", "swept",
    ],
    "other_sports": [
        "grand slam", "wimbledon", "us open", "french open",
        "australian open", "formula 1", "f1 race", "grand prix",
        "boxing match", "ufc", "mma", "knockout",
        "tour de france", "marathon", "triathlon",
        "golf major", "masters tournament", "ryder cup",
        "swimming record", "athletics record", "track and field",
        "rugby world cup", "six nations", "world championship",
        "heavyweight champion", "title fight", "title bout",
    ],
}

# Flatten for fast lookup
_HISTORICAL_SIGNALS = []
for _group in _HISTORICAL_KEYWORDS.values():
    _HISTORICAL_SIGNALS.extend(_group)


def classify_historical(text: str, source: str) -> bool:
    """
    Determine if a piece of data is historically significant.
    Conservative: only archives items with clear historical markers.
    Weather/traffic/availability sources are ALWAYS live.
    """
    if source.lower() in ALWAYS_LIVE_SOURCES:
        return False

    text_lower = text.lower()
    for signal in _HISTORICAL_SIGNALS:
        if signal in text_lower:
            return True

    return False


# ==============================================================================
# CheckpointRotator — Daily/weekly/monthly/yearly snapshot management
# ==============================================================================

class CheckpointRotator:
    """Manages checkpoint rotation: daily/weekly/monthly/yearly snapshots."""

    def __init__(self, checkpoint_dir: str):
        self.dir = checkpoint_dir
        os.makedirs(self.dir, exist_ok=True)

    def save_snapshot(self, learner):
        """Save a dated snapshot if one doesn't exist for today."""
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        daily_path = os.path.join(self.dir, f"snapshot_daily_{today}.pt")
        if os.path.exists(daily_path):
            return None  # Already have today's snapshot

        import torch
        torch.save({
            "model_state": learner.model.state_dict(),
            "config": {"model": vars(learner.config.model),
                       "tokenizer": vars(learner.config.tokenizer)},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "snapshot_type": "daily",
        }, daily_path)
        return daily_path

    def rotate(self):
        """Promote daily->weekly->monthly->yearly, prune excess."""
        now = datetime.now(timezone.utc)

        # Promote old dailies to weekly
        dailies = sorted(glob.glob(os.path.join(self.dir, "snapshot_daily_*.pt")))
        for path in dailies:
            dt = self._parse_date(path)
            if dt and (now - dt).days >= 7:
                week_str = dt.strftime("%Y_W%W")
                weekly_path = os.path.join(self.dir, f"snapshot_weekly_{week_str}.pt")
                if not os.path.exists(weekly_path):
                    os.rename(path, weekly_path)
                else:
                    os.remove(path)

        # Promote old weeklies to monthly
        for path in sorted(glob.glob(os.path.join(self.dir, "snapshot_weekly_*.pt"))):
            dt = self._parse_date(path)
            if dt and (now - dt).days >= 30:
                month_str = dt.strftime("%Y%m")
                monthly_path = os.path.join(self.dir, f"snapshot_monthly_{month_str}.pt")
                if not os.path.exists(monthly_path):
                    os.rename(path, monthly_path)
                else:
                    os.remove(path)

        # Promote old monthlies to yearly
        for path in sorted(glob.glob(os.path.join(self.dir, "snapshot_monthly_*.pt"))):
            dt = self._parse_date(path)
            if dt and (now - dt).days >= 365:
                year_str = dt.strftime("%Y")
                yearly_path = os.path.join(self.dir, f"snapshot_yearly_{year_str}.pt")
                if not os.path.exists(yearly_path):
                    os.rename(path, yearly_path)
                else:
                    os.remove(path)

        # Prune: keep at most 7 dailies, 4 weeklies, 12 monthlies, unlimited yearlies
        self._prune("snapshot_daily_*.pt", keep=7)
        self._prune("snapshot_weekly_*.pt", keep=4)
        self._prune("snapshot_monthly_*.pt", keep=12)

    def _prune(self, pattern, keep):
        files = sorted(glob.glob(os.path.join(self.dir, pattern)))
        while len(files) > keep:
            os.remove(files.pop(0))

    def _parse_date(self, path):
        """Extract date from snapshot filename."""
        fname = os.path.basename(path)
        # Try daily: snapshot_daily_YYYYMMDD.pt
        if "daily_" in fname:
            try:
                date_str = fname.split("daily_")[1].replace(".pt", "")
                return datetime.strptime(date_str, "%Y%m%d").replace(tzinfo=timezone.utc)
            except ValueError:
                pass
        # Try weekly: snapshot_weekly_YYYY_WWW.pt
        if "weekly_" in fname:
            try:
                date_str = fname.split("weekly_")[1].replace(".pt", "")
                return datetime.strptime(date_str + "_1", "%Y_W%W_%w").replace(tzinfo=timezone.utc)
            except ValueError:
                pass
        # Try monthly: snapshot_monthly_YYYYMM.pt
        if "monthly_" in fname:
            try:
                date_str = fname.split("monthly_")[1].replace(".pt", "")
                return datetime.strptime(date_str, "%Y%m").replace(tzinfo=timezone.utc)
            except ValueError:
                pass
        return None

    def stats(self):
        """Return counts of each snapshot tier."""
        return {
            "daily": len(glob.glob(os.path.join(self.dir, "snapshot_daily_*.pt"))),
            "weekly": len(glob.glob(os.path.join(self.dir, "snapshot_weekly_*.pt"))),
            "monthly": len(glob.glob(os.path.join(self.dir, "snapshot_monthly_*.pt"))),
            "yearly": len(glob.glob(os.path.join(self.dir, "snapshot_yearly_*.pt"))),
        }


# ==============================================================================
# BatchQueue — Manages batch Parquet files
# ==============================================================================

class BatchQueue:
    """
    Accumulates read-loop results into an in-memory Arrow table,
    flushing to Parquet periodically or when a batch is taken for
    consolidation.

    Storage: batches/batch_YYYYMMDD_HHMMSS.parquet
    """

    def __init__(self, batch_dir: str = "batches", flush_every: int = 100):
        self.batch_dir = batch_dir
        self.flush_every = flush_every
        self._lock = threading.Lock()
        self._texts = []
        self._sources = []
        self._priorities = []
        self._timestamps = []
        self._current_file: Optional[str] = None
        self._total_items = 0  # items written to current parquet + in-memory
        os.makedirs(self.batch_dir, exist_ok=True)
        self._ensure_current_file()

    def _ensure_current_file(self):
        """Create a filename for the current batch if none exists."""
        if self._current_file is None:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            self._current_file = os.path.join(
                self.batch_dir, f"batch_{ts}.parquet"
            )

    def add(self, text: str, source: str, priority: int = 1):
        """Append one item to the current batch (thread-safe)."""
        with self._lock:
            self._texts.append(text)
            self._sources.append(source)
            self._priorities.append(priority)
            self._timestamps.append(datetime.now(timezone.utc))
            self._total_items += 1

            if len(self._texts) >= self.flush_every:
                self._flush_unlocked()

    def _flush_unlocked(self):
        """Write in-memory rows to the current Parquet file (caller holds lock)."""
        if not self._texts:
            return
        # Build RecordBatch from Python lists (avoids pandas shim issues)
        batch = pa.RecordBatch.from_pydict(
            {
                "text": self._texts,
                "source": self._sources,
                "priority": self._priorities,
                "timestamp": [int(ts.timestamp() * 1_000_000) for ts in self._timestamps],
            },
            schema=BATCH_SCHEMA,
        )
        table = pa.Table.from_batches([batch])
        if os.path.exists(self._current_file):
            existing = pq.read_table(self._current_file)
            table = pa.concat_tables([existing, table])
        pq.write_table(table, self._current_file)
        self._texts.clear()
        self._sources.clear()
        self._priorities.clear()
        self._timestamps.clear()

    def take_batch(self) -> list:
        """
        Return all items in the current batch, start a new empty batch.
        Returns list of dicts: [{"text": ..., "source": ..., "priority": ..., "timestamp": ...}]
        """
        with self._lock:
            self._flush_unlocked()
            items = []
            if self._current_file and os.path.exists(self._current_file):
                table = pq.read_table(self._current_file)
                # Cast timestamp to int64 (microseconds since epoch) to avoid pytz
                ts_col = table.column("timestamp").cast(pa.int64())
                for i in range(table.num_rows):
                    us = ts_col[i].as_py()
                    ts_iso = datetime.fromtimestamp(us / 1_000_000, tz=timezone.utc).isoformat()
                    items.append({
                        "text": table.column("text")[i].as_py(),
                        "source": table.column("source")[i].as_py(),
                        "priority": table.column("priority")[i].as_py(),
                        "timestamp": ts_iso,
                    })
            # Start fresh batch
            self._current_file = None
            self._total_items = 0
            self._ensure_current_file()
            return items

    def current_size(self) -> int:
        """How many items in the current batch (in-memory + flushed)."""
        with self._lock:
            return self._total_items

    def flush(self):
        """Force flush in-memory rows to disk."""
        with self._lock:
            self._flush_unlocked()


# ==============================================================================
# ReadLoop — Background thread that fetches from tools
# ==============================================================================

class ReadLoop:
    """
    Background read system — each source runs in its OWN parallel thread.

    crypto_price:bitcoin  → Thread 1 (every 60s)
    crypto_price:ethereum → Thread 2 (every 60s)
    hacker_news           → Thread 3 (every 300s)
    news:technology       → Thread 4 (every 600s)
    weather               → Thread 5 (every 1800s)

    All fetch in parallel, store directly into database (no batching).
    Deduplicates by skipping results identical to the last fetch.
    """

    # Priority → confidence mapping
    _PRIORITY_CONFIDENCE = {0: 1.0, 1: 0.95, 2: 0.85, 3: 0.7, 4: 0.5}

    def __init__(self, config: QORConfig, gate=None, graph=None, rag=None):
        self.config = config
        self._gate = gate      # ConfidenceGate (has .memory)
        self._graph = graph    # QORGraph (RocksDB)
        self._rag = rag        # QORRag (vector store)
        self.executor = ToolExecutor()
        self._stop_event = threading.Event()
        self._threads: list = []
        self._lock = threading.Lock()  # protects _stats, _last_results
        self._last_results: dict = {}  # "tool:query" → last result text
        self._stats = {"fetches": 0, "new_items": 0, "errors": 0, "dupes": 0}

    def start(self):
        """Start one background thread per read source (all parallel)."""
        if self._threads:
            return
        self._stop_event.clear()

        for src in self.config.runtime.read_sources:
            tool = src["tool"]
            query = src.get("query", "")
            name = f"qor-read-{tool}:{query}" if query else f"qor-read-{tool}"
            t = threading.Thread(
                target=self._source_loop, args=(src,),
                daemon=True, name=name,
            )
            t.start()
            self._threads.append(t)
            logger.info(f"[ReadLoop] Started thread: {name}")

        logger.info(f"[ReadLoop] {len(self._threads)} parallel read threads running")

    def stop(self):
        """Signal all read threads to stop."""
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=10)
        self._threads.clear()
        logger.info("[ReadLoop] All read threads stopped")

    @property
    def stats(self) -> dict:
        with self._lock:
            return dict(self._stats)

    def _source_loop(self, source: dict):
        """One thread per source — fetches on its own interval, independently."""
        tool = source["tool"]
        query = source.get("query", "")
        interval = source.get("interval", 60)
        priority = source.get("priority", 1)
        key = f"{tool}:{query}"

        while not self._stop_event.is_set():
            self._fetch_source(key, tool, query, priority)
            self._stop_event.wait(timeout=interval)

    def _fetch_source(self, key: str, tool: str, query: str, priority: int):
        """Fetch a single source and store directly into database."""
        with self._lock:
            self._stats["fetches"] += 1

        try:
            result = self.executor.call(tool, query)
        except Exception as e:
            with self._lock:
                self._stats["errors"] += 1
            logger.warning(f"[ReadLoop] Error fetching {key}: {e}")
            return

        # Deduplicate
        with self._lock:
            if result == self._last_results.get(key):
                self._stats["dupes"] += 1
                return
            self._last_results[key] = result
            self._stats["new_items"] += 1

        # Store directly into database — no batching
        confidence = self._PRIORITY_CONFIDENCE.get(priority, 0.7)
        is_historical = classify_historical(result, tool)
        category = "historical" if is_historical else "live"

        # 1. MemoryStore
        if self._gate is not None:
            self._gate.memory.store(
                key=key,
                content=result[:500],
                source=tool,
                category=category,
                confidence=confidence,
            )

        # 2. Knowledge Graph — extract entities
        if self._graph is not None:
            try:
                if self._graph.is_open:
                    from qor.confidence import _extract_entities_and_edges
                    edges = _extract_entities_and_edges(result)
                    for subj, pred, obj in edges:
                        self._graph.add_edge(
                            subj, pred, obj,
                            confidence=confidence, source=tool,
                        )
            except Exception as e:
                logger.debug(f"[ReadLoop] Graph extract error: {e}")

        # 3. RAG — index text
        if self._rag is not None:
            try:
                self._rag.add_text(result, source=f"{tool}:{query}")
            except Exception:
                pass

        logger.debug(f"[ReadLoop] Stored {key} → memory+graph+rag ({len(result)} chars)")


# ==============================================================================
# QORRuntime — Orchestrator (model is FROZEN, no training)
# ==============================================================================

class QORRuntime:
    """
    Orchestrates background reading + periodic cleanup.

    Model is FROZEN — no training, no weight updates.
    Read loop stores data directly into memory/graph/RAG.
    Periodic cleanup removes old live data, compacts graph, rotates checkpoints.

    Usage:
        runtime = QORRuntime(config)
        runtime.start(learner, gate, graph, rag)
        runtime.status()
        runtime.stop()
    """

    def __init__(self, config: QORConfig):
        self.config = config
        self.read_loop = None
        self._learner = None
        self._gate = None
        self._graph = None
        self._rag = None
        self._historical_store: Optional[HistoricalStore] = None
        self._checkpoint_rotator: Optional[CheckpointRotator] = None
        self._cache_store = None     # CacheStore for tool/API results
        self._chat_store = None      # ChatStore for conversation history
        self._cleanup_timer: Optional[threading.Timer] = None
        self._stop_event = threading.Event()
        self._last_cleanup: Optional[str] = None
        self._cleanup_count: int = 0

        # Trading engines (optional, wired in start() if enabled)
        self._trading_engine = None       # SpotEngine
        self._futures_engine = None       # FuturesEngine
        self._hmm = None                  # MarketHMM (shared by trading engines)
        self._shared_cortex = None        # CortexAnalyzer (shared by trading engines)

        # NGRE modules (optional, wired in start() if available)
        self._health_monitor = None       # GraphHealthMonitor
        self._source_reliability = None   # SourceReliabilityTracker
        self._fact_checker = None         # FactChecker
        self._compressor = None           # SnapshotCompressor
        self._cold_tier = None            # ColdTier
        self._ingestion_daemon = None     # IngestionDaemon

    def start(self, learner, gate=None, graph=None, rag=None,
              cache_store=None, chat_store=None, tool_executor=None):
        """
        Start the runtime: read loop + cleanup scheduler + ingestion daemon.

        Args:
            learner: ContinualLearner (holds model + tokenizer, model is FROZEN)
            gate: ConfidenceGate (has .memory MemoryStore)
            graph: QORGraph (RocksDB knowledge graph)
            rag: QORRag (vector store for retrieval)
            cache_store: CacheStore for tool/API result caching
            chat_store: ChatStore for conversation history
            tool_executor: ToolExecutor for ingestion daemon tool calls
        """
        self._tool_executor = tool_executor
        self._learner = learner
        self._gate = gate
        self._graph = graph
        self._rag = rag
        self._cache_store = cache_store
        self._chat_store = chat_store
        self._stop_event.clear()

        # Historical store for permanent archive
        self._historical_store = HistoricalStore(self.config.runtime.historical_dir)

        # Wire historical store into gate for answer-path search
        if gate is not None and hasattr(gate, 'set_historical'):
            gate.set_historical(self._historical_store)

        # Checkpoint rotator
        if self.config.runtime.checkpoint_rotation:
            self._checkpoint_rotator = CheckpointRotator(self.config.train.checkpoint_dir)

        # Read loop — DISABLED by default (wastes API calls).
        # Tools are called on-demand when user asks + data is stale.
        # To enable background reading, set config.runtime.enable_read_loop = True
        self.read_loop = ReadLoop(self.config, gate=gate, graph=graph, rag=rag)
        if getattr(self.config.runtime, 'enable_read_loop', False):
            self.read_loop.start()
            logger.info("[Runtime] Read loop ENABLED (background API calls active)")
        else:
            logger.info("[Runtime] Read loop disabled — tools called on-demand only")

        # CORTEX training buffer — accumulates (features, target) from closed trades
        self._cortex_train_buf = []

        # Trading engine — disabled by default, must opt in via config
        # Trade learning callback — saves closed trades to AI memory + graph
        # so the AI can learn from wins/losses and give better suggestions
        def _on_trade_close(trade: dict, engine_type: str):
            """Called when any trade closes. Saves summary to memory + graph."""
            from qor.trading import format_trade_summary
            # Skip invalid trades (never filled)
            if trade.get("quantity", 0) <= 0 or trade.get("entry_price", 0) <= 0:
                return
            summary = format_trade_summary(trade, engine_type)
            symbol = trade.get("symbol", "?")
            direction = trade.get("direction", trade.get("side", "?"))
            pnl = trade.get("pnl", 0)
            tag = "win" if pnl > 0 else "loss"
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%d")

            # Save to MemoryStore — AI can search "BTC trades", "losses", etc.
            if gate is not None and hasattr(gate, 'memory'):
                mem_key = f"trade:{engine_type}:{symbol}:{direction}:{ts}:{trade.get('trade_id', '')[:8]}"
                gate.memory.store(mem_key, summary, f"trade:{engine_type}",
                                  category="static", confidence=0.9)

            # Save to KnowledgeGraph — AI can reason about patterns
            if graph is not None and hasattr(graph, 'is_open') and graph.is_open:
                try:
                    strategy = trade.get("strategy", "unknown")
                    pnl_pct = trade.get("pnl_pct", 0)
                    # Edge: "BTC SHORT" → "resulted_in" → "loss -2.5%"
                    graph.add_edge(
                        f"{symbol} {direction}", "resulted_in",
                        f"{tag} {pnl_pct:+.1f}%",
                        confidence=0.85,
                    )
                    # Edge: strategy → "produced" → "win/loss for SYMBOL"
                    graph.add_edge(
                        strategy, "produced",
                        f"{tag} on {symbol} {direction} ({pnl_pct:+.1f}%)",
                        confidence=0.8,
                    )
                except Exception:
                    pass

            logger.info(f"[Runtime] Trade recorded to AI memory: {symbol} {direction} {tag}")

            # Trigger trade learning every 5th closed trade
            try:
                store = None
                if engine_type == "spot" and self._trading_engine:
                    store = self._trading_engine.store
                elif engine_type == "futures" and self._futures_engine:
                    store = self._futures_engine.store
                if store is not None:
                    closed_count = sum(1 for t in store.trades.values()
                                       if t.get("status") != "open")
                    if closed_count > 0 and closed_count % 5 == 0 and graph is not None:
                        from .knowledge_tree import TradeLearner
                        user_id = getattr(gate, '_user_id', 'user:default') if gate else 'user:default'
                        lessons = TradeLearner.analyze_trades(store, graph, user_id)
                        if lessons:
                            logger.info(f"[Runtime] Trade learning: {len(lessons)} new lessons")
            except Exception:
                pass

            # --- CORTEX continuous training from closed trades ---
            try:
                if self._shared_cortex is not None:
                    feat = self._shared_cortex.get_last_features(symbol)
                    if feat is not None:
                        direction = trade.get("direction", trade.get("side", "LONG"))
                        if direction in ("LONG", "BUY"):
                            target = 1.0 if pnl > 0 else -1.0
                        else:  # SHORT / SELL
                            target = -1.0 if pnl > 0 else 1.0
                        self._cortex_train_buf.append((feat.clone(), target))

                        if len(self._cortex_train_buf) >= 10:
                            import torch as _torch
                            features = [f for f, _ in self._cortex_train_buf]
                            targets = [t for _, t in self._cortex_train_buf]
                            result = self._shared_cortex.train_batch(
                                features, targets, epochs=5, lr=0.001)
                            if result.get("trained"):
                                cortex_path = self.config.get_data_path("cortex_shared.pt")
                                self._shared_cortex.save(cortex_path)
                                logger.info("[Runtime] CORTEX trained on %d samples "
                                            "(loss=%.4f), saved",
                                            len(features), result["final_loss"])
                            # Keep last 5 for overlap with next batch
                            self._cortex_train_buf = self._cortex_train_buf[-5:]
            except Exception as e:
                logger.debug(f"[Runtime] CORTEX training error: {e}")

        # HMM regime detection — shared by both spot and futures engines
        self._hmm = None
        try:
            from qor.quant import MarketHMM
            data_dir = getattr(self.config, 'runtime', None)
            data_dir = getattr(data_dir, 'data_dir', 'qor-data') if data_dir else 'qor-data'
            self._hmm = MarketHMM(data_dir=data_dir)
            self._hmm.load()
            # Initial training from graph snapshot data if not yet trained
            if graph and not self._hmm.is_trained:
                self._hmm.train_from_graph(graph)
                self._hmm.save()
            logger.info("[Runtime] HMM regime detector initialized "
                        f"(trained={self._hmm.is_trained})")
        except Exception as e:
            logger.debug(f"[Runtime] HMM not available: {e}")

        # Shared CORTEX analyzer — one instance for both spot and futures
        # (same symbol shares history/state, training data is pooled)
        self._shared_cortex = None
        try:
            from qor.trading import CortexAnalyzer
            self._shared_cortex = CortexAnalyzer()
            # Load saved weights if available
            data_dir = getattr(self.config, 'runtime', None)
            data_dir = getattr(data_dir, 'data_dir', 'qor-data') if data_dir else 'qor-data'
            cortex_path = os.path.join(data_dir, "cortex_shared.pt")
            if os.path.exists(cortex_path):
                self._shared_cortex.load(cortex_path)
            logger.info("[Runtime] Shared CORTEX analyzer loaded "
                        f"(trained={self._shared_cortex._brain._trained})")
            # Wire to tools.py for multi_tf_analysis CORTEX output
            try:
                from qor.tools import set_shared_cortex
                set_shared_cortex(self._shared_cortex)
            except Exception:
                pass
        except Exception as e:
            logger.debug(f"[Runtime] CORTEX not available: {e}")

        self._trading_engine = None
        if getattr(self.config, 'trading', None) and self.config.trading.enabled:
            if self.config.trading.api_key and self.config.trading.api_secret:
                try:
                    from qor.trading import TradingEngine
                    # Get tool_executor from gate if available
                    tex = getattr(gate, '_tool_executor', None) if gate else None
                    self._trading_engine = TradingEngine(
                        self.config, tool_executor=tex, hmm=self._hmm,
                        cortex=self._shared_cortex)
                    self._trading_engine.store.on_close = _on_trade_close
                    self._trading_engine.start()
                    mode = "DEMO" if self.config.trading.testnet else "PRODUCTION"
                    logger.info(f"[Runtime] Trading engine ENABLED ({mode})")
                except Exception as e:
                    logger.warning(f"[Runtime] Trading engine failed to start: {e}")
            else:
                logger.warning("[Runtime] Trading enabled but API keys not set "
                               "(set QOR_BINANCE_KEY and QOR_BINANCE_SECRET)")

        # Futures engine — disabled by default, independent of spot
        self._futures_engine = None
        if getattr(self.config, 'futures', None) and self.config.futures.enabled:
            if self.config.futures.api_key and self.config.futures.api_secret:
                try:
                    from qor.futures import FuturesEngine
                    tex = getattr(gate, '_tool_executor', None) if gate else None
                    self._futures_engine = FuturesEngine(
                        self.config, tool_executor=tex, hmm=self._hmm,
                        cortex=self._shared_cortex)
                    self._futures_engine.store.on_close = _on_trade_close
                    self._futures_engine.start()
                    mode = "TESTNET" if self.config.futures.testnet else "PRODUCTION"
                    lev = self.config.futures.leverage
                    logger.info(f"[Runtime] Futures engine ENABLED ({mode}, {lev}x)")
                except Exception as e:
                    logger.warning(f"[Runtime] Futures engine failed to start: {e}")
            else:
                logger.warning("[Runtime] Futures enabled but API keys not set")

        # --- Multi-exchange engines (from config.exchanges) ---
        self._exchange_engines = {}  # name → TradingEngine
        for ex in getattr(self.config, 'exchanges', []):
            if not getattr(ex, 'enabled', False):
                continue
            if not ex.api_key or not ex.api_secret:
                continue
            if not ex.symbols:
                continue
            try:
                from qor.trading import create_exchange_client, TradingEngine
                from qor.config import TradingConfig
                client = create_exchange_client(
                    ex.name, ex.api_key, ex.api_secret,
                    passphrase=ex.passphrase, testnet=ex.testnet,
                    base_url=ex.base_url,
                    access_token=getattr(ex, 'access_token', ''),
                )
                # Build a TradingConfig-like object for this exchange
                ex_trading = TradingConfig(
                    enabled=True, testnet=ex.testnet,
                    api_key=ex.api_key, api_secret=ex.api_secret,
                    symbols=list(ex.symbols),
                    check_interval_seconds=ex.check_interval_seconds,
                    data_dir=os.path.join(
                        self.config.runtime.data_dir, "exchanges", ex.name),
                )
                os.makedirs(ex_trading.data_dir, exist_ok=True)
                # Patch the config so TradingEngine can read it
                _ex_cfg = type('_ExCfg', (), {'trading': ex_trading})()
                tex = getattr(gate, '_tool_executor', None) if gate else None
                engine = TradingEngine(
                    _ex_cfg, tool_executor=tex, hmm=self._hmm, client=client)
                engine.start()
                self._exchange_engines[ex.name] = engine
                mode = "DEMO" if ex.testnet else "PRODUCTION"
                logger.info(f"[Runtime] Exchange engine '{ex.name}' ENABLED "
                            f"({mode}, symbols={ex.symbols})")
            except Exception as e:
                logger.warning(f"[Runtime] Exchange engine '{ex.name}' failed: {e}")

        # --- CORTEX auto-training (after all engines are created) ---
        if self._shared_cortex and not self._shared_cortex._brain._trained:
            self._bootstrap_cortex_training()
            if self._shared_cortex._brain._trained:
                logger.info("[Runtime] CORTEX is now TRAINED and producing signals")

        # --- NGRE modules (all optional, fail gracefully) ---

        # Graph health monitor + source reliability + fact checker
        try:
            from .health import GraphHealthMonitor, SourceReliabilityTracker, FactChecker
            if graph is not None:
                self._health_monitor = GraphHealthMonitor(graph)
                self._source_reliability = SourceReliabilityTracker(graph)
                executor = getattr(gate, '_tool_executor', None) if gate else None
                self._fact_checker = FactChecker(graph, tool_executor=executor)
                logger.info("[Runtime] NGRE health monitor + fact checker ENABLED")
        except Exception as e:
            logger.debug(f"[Runtime] NGRE health modules not available: {e}")

        # Snapshot compressor (needs ngre_brain for embeddings on aggregated nodes)
        try:
            from .compression import SnapshotCompressor
            data_dir = self.config.runtime.data_dir
            ngre_brain_ref = None
            if gate is not None and hasattr(gate, '_ngre_brain'):
                ngre_brain_ref = gate._ngre_brain
            self._compressor = SnapshotCompressor(
                graph=graph, data_dir=data_dir, ngre_brain=ngre_brain_ref)
            logger.info("[Runtime] NGRE snapshot compressor ENABLED")
        except Exception as e:
            logger.debug(f"[Runtime] NGRE compressor not available: {e}")

        # Cold tier storage
        try:
            from .cold_tier import ColdTier, ColdTierConfig
            data_dir = self.config.runtime.data_dir
            cold_cfg = ColdTierConfig(
                archive_dir=os.path.join(data_dir, "cold_archive"),
            )
            self._cold_tier = ColdTier(cold_cfg)
            logger.info("[Runtime] NGRE cold tier storage ENABLED")
        except Exception as e:
            logger.debug(f"[Runtime] NGRE cold tier not available: {e}")

        # Ingestion daemon — PRD Section 22: 24/7 knowledge ingestion
        # Default sources: crypto 5min, stocks 5min, weather 5min,
        #                  forex 5min, news 15min, historical daily
        try:
            from .ingestion import (IngestionDaemon, IngestionConfig as IngCfg,
                                    IngestionSource)
            enable_ingestion = getattr(
                self.config.runtime, 'enable_ingestion', False)
            if enable_ingestion and graph is not None:
                ing_cfg = IngCfg(
                    enabled=True,
                    check_interval_seconds=30,
                    compression_interval_hours=1.0,
                    fact_check_interval_hours=6.0,
                )
                # Get NGRE brain from gate for embedding computation
                ngre_brain = None
                if gate is not None and hasattr(gate, '_ngre_brain'):
                    ngre_brain = gate._ngre_brain

                # Build dynamic asset list from config
                # Merge: watch_assets + trading symbols + futures symbols
                _all_assets = list(getattr(
                    self.config.runtime, 'watch_assets', None) or [])
                if getattr(self.config, 'trading', None):
                    for s in self.config.trading.symbols:
                        if s not in _all_assets:
                            _all_assets.append(s)
                if getattr(self.config, 'futures', None):
                    for s in self.config.futures.symbols:
                        if s not in _all_assets:
                            _all_assets.append(s)
                # Default assets if none configured
                if not _all_assets:
                    _all_assets = ["BTC", "ETH"]

                self._ingestion_daemon = IngestionDaemon(
                    config=ing_cfg, graph=graph,
                    tool_executor=tool_executor,
                    compressor=self._compressor,
                    fact_checker=self._fact_checker,
                    ngre_brain=ngre_brain,
                    assets=_all_assets,
                    hmm=self._hmm,
                    trading_engine=self._trading_engine,
                    futures_engine=self._futures_engine,
                )

                # Auto-create price + TA sources per asset
                try:
                    from .quant import build_ingestion_sources
                    _asset_sources = build_ingestion_sources(_all_assets)
                    for src_info in _asset_sources:
                        self._ingestion_daemon.add_source(IngestionSource(
                            src_info["name"], src_info["tool"],
                            src_info["query"],
                            interval_minutes=src_info["interval"],
                            priority=src_info["priority"],
                            asset_type=src_info.get("asset_type", ""),
                        ))
                    logger.info("[Runtime] Ingestion: %d asset sources for %s",
                                len(_asset_sources),
                                ", ".join(_all_assets))
                except Exception as e:
                    logger.warning("[Runtime] Dynamic asset sources failed: %s", e)

                # Fixed sources — only things NOT covered by per-asset ingestion.
                # Per-asset sources (price, TA, news, fear_greed, funding,
                # polymarket, economic_calendar, etc.) are handled above
                # by build_ingestion_sources().
                _fixed_sources = [
                    # General (not tied to any specific asset)
                    IngestionSource("weather_local", "weather", "New York",
                                    interval_minutes=30, priority=4),
                    IngestionSource("hacker_news", "hacker_news", "",
                                    interval_minutes=15, priority=2),
                    IngestionSource("crypto_trending", "crypto_trending", "",
                                    interval_minutes=15, priority=3),
                    IngestionSource("market_indices", "market_indices", "",
                                    interval_minutes=60, priority=2),
                    # Daily (background enrichment)
                    IngestionSource("on_this_day", "on_this_day", "",
                                    interval_minutes=1440, priority=4),
                    IngestionSource("arxiv_ai", "arxiv", "machine learning",
                                    interval_minutes=1440, priority=4),
                ]
                for src in _fixed_sources:
                    self._ingestion_daemon.add_source(src)

                self._ingestion_daemon.start()
                logger.info("[Runtime] NGRE ingestion daemon ENABLED "
                            "(%d sources, assets: %s)",
                            len(self._ingestion_daemon._sources),
                            ", ".join(_all_assets))
        except Exception as e:
            logger.debug(f"[Runtime] NGRE ingestion daemon not available: {e}")

        # Periodic cleanup scheduler
        self._schedule_cleanup()

        # Periodic CORTEX kline retraining (keeps ALL symbols fresh)
        self._cortex_retrain_timer = None
        retrain_hours = getattr(self.config.runtime, 'cortex_retrain_hours', 6.0)
        if self._shared_cortex and retrain_hours > 0:
            self._schedule_cortex_retrain()
            logger.info(f"[Runtime] CORTEX periodic retrain every {retrain_hours}h")

        hours = self.config.runtime.cleanup_every_hours
        logger.info(f"[Runtime] Started — read every {self.config.runtime.read_interval}s, "
                     f"cleanup every {hours}h")

    def stop(self):
        """Clean shutdown."""
        self._stop_event.set()
        if self._ingestion_daemon:
            try:
                self._ingestion_daemon.stop()
            except Exception:
                pass
        if self._trading_engine:
            self._trading_engine.stop()
        if self._futures_engine:
            self._futures_engine.stop()
        for name, eng in getattr(self, '_exchange_engines', {}).items():
            try:
                eng.stop()
            except Exception:
                pass
        if self.read_loop:
            self.read_loop.stop()
        if self._cleanup_timer:
            self._cleanup_timer.cancel()
        if getattr(self, '_cortex_retrain_timer', None):
            self._cortex_retrain_timer.cancel()
        if self._gate and self._gate.memory._dirty:
            self._gate.memory.save()
        if self._cache_store and self._cache_store._dirty:
            self._cache_store.save()
        if self._chat_store and self._chat_store._dirty:
            self._chat_store.save()
        # Save shared CORTEX state
        if self._shared_cortex:
            try:
                data_dir = getattr(self.config, 'runtime', None)
                data_dir = getattr(data_dir, 'data_dir', 'qor-data') if data_dir else 'qor-data'
                cortex_path = os.path.join(data_dir, "cortex_shared.pt")
                self._shared_cortex.save(cortex_path)
                logger.info(f"[Runtime] Shared CORTEX saved to {cortex_path}")
            except Exception as e:
                logger.warning(f"[Runtime] CORTEX save failed: {e}")
        logger.info("[Runtime] Stopped")

    def cleanup_now(self) -> dict:
        """Manual cleanup trigger."""
        result = self._run_cleanup()
        return result

    def status(self) -> dict:
        """Return runtime status."""
        s = {
            "last_cleanup": self._last_cleanup,
            "cleanup_count": self._cleanup_count,
            "read_stats": self.read_loop.stats if self.read_loop else {},
            "read_sources": len(self.config.runtime.read_sources),
        }
        if self._gate is not None:
            s["memory_entries"] = len(self._gate.memory.entries)
        if self._graph is not None:
            try:
                if self._graph.is_open:
                    gs = self._graph.stats()
                    s["graph_nodes"] = gs.get("node_count", 0)
                    s["graph_edges"] = gs.get("edge_count", 0)
            except Exception:
                pass
        if self._rag is not None:
            try:
                s["rag_chunks"] = len(self._rag.store.chunks)
            except Exception:
                pass
        if self._historical_store is not None:
            s["historical_entries"] = self._historical_store.count()
        if self._checkpoint_rotator is not None:
            s["checkpoint_snapshots"] = self._checkpoint_rotator.stats()
        if self._cache_store is not None:
            s["cache_entries"] = self._cache_store.count()
        if self._chat_store is not None:
            s["chat_messages"] = self._chat_store.count()
            s["chat_sessions"] = self._chat_store.session_count()
        if self._trading_engine is not None:
            s["trading"] = self._trading_engine.status()
        if self._futures_engine is not None:
            s["futures"] = self._futures_engine.status()
        # Multi-exchange engine stats
        for name, eng in getattr(self, '_exchange_engines', {}).items():
            try:
                s[f"exchange_{name}"] = eng.status()
            except Exception:
                pass
        # NGRE module stats
        if self._health_monitor is not None:
            try:
                s["graph_health"] = self._health_monitor.check_all()
            except Exception:
                pass
        if self._source_reliability is not None:
            try:
                s["source_reliability"] = self._source_reliability.get_all_stats()
            except Exception:
                pass
        if self._compressor is not None:
            try:
                s["compressor"] = self._compressor.stats()
            except Exception:
                pass
        if self._cold_tier is not None:
            try:
                s["cold_tier"] = self._cold_tier.stats()
            except Exception:
                pass
        if self._ingestion_daemon is not None:
            try:
                s["ingestion"] = self._ingestion_daemon.status()
            except Exception:
                pass
        # CORTEX brain status
        if self._shared_cortex is not None:
            try:
                cs = self._shared_cortex.status()
                cs["train_buffer"] = len(getattr(self, '_cortex_train_buf', []))
                s["cortex"] = cs
            except Exception:
                pass
        if self._hmm is not None and self._hmm.is_available:
            s["hmm"] = self._hmm.status()
        if self._gate is not None and hasattr(self._gate, '_routing_stats'):
            s["routing_stats"] = self._gate.get_routing_stats()
        # Knowledge tree stats
        if self._graph is not None and self._graph.is_open:
            try:
                kt = {}
                kt["corrections"] = len(self._graph.list_nodes(node_type="correction"))
                kt["blocked_facts"] = len(self._graph.list_nodes(node_type="blocked_fact"))
                kt["lessons"] = len(self._graph.list_nodes(node_type="lesson"))
                kt["preferences"] = len(self._graph.list_nodes(node_type="preference"))
                kt["trade_patterns"] = len(self._graph.list_nodes(node_type="trade_pattern"))
                s["knowledge_tree"] = kt
            except Exception:
                pass
        return s

    def _bootstrap_cortex_training(self):
        """Auto-train CORTEX from historical kline data via KlineRouter.

        Uses KlineRouter to fetch 90 days of 1h candles from the correct
        exchange for each configured symbol (crypto, stocks, forex, etc.),
        computes all 20 TA indicators, labels with future returns, and
        trains CORTEX brain. Falls back to trade history if kline fetch fails.
        """
        if self._shared_cortex is None:
            return

        # --- Phase 1: Try KlineRouter + CortexTrainer (real TA data) ---
        try:
            from .kline_router import KlineRouter
            from .train_cortex import CortexTrainer

            # Build KlineRouter with all available clients
            kline_client = None
            upstox_kline = None

            # Get Binance client (spot or futures)
            if self._trading_engine and hasattr(self._trading_engine, 'client'):
                kline_client = self._trading_engine.client
            elif self._futures_engine and hasattr(self._futures_engine, 'client'):
                kline_client = self._futures_engine.client

            # Get Upstox client from exchange engines
            for eng in getattr(self, '_exchange_engines', {}).values():
                if hasattr(eng, 'client') and hasattr(eng.client, '_resolve_instrument_key'):
                    upstox_kline = eng.client
                    break

            # Build universal router
            router = KlineRouter(
                config=self.config,
                binance_client=kline_client,
                upstox_client=upstox_kline,
            )

            status = router.status()
            active = [k for k, v in status.items() if v]
            logger.info(f"[Runtime] KlineRouter active sources: {', '.join(active)}")

            trainer = CortexTrainer(router, self._shared_cortex)

            # Gather training symbols from ALL configured engines
            symbols = []

            # Crypto symbols
            if getattr(self.config, 'trading', None) and hasattr(self.config.trading, 'symbols'):
                symbols.extend(self.config.trading.symbols[:3])
            if getattr(self.config, 'futures', None) and hasattr(self.config.futures, 'symbols'):
                for s in self.config.futures.symbols[:2]:
                    if s not in symbols:
                        symbols.append(s)

            # Exchange engine symbols (Upstox, etc.)
            for ex in getattr(self.config, 'exchanges', []):
                if hasattr(ex, 'symbols'):
                    for s in ex.symbols[:3]:
                        if s not in symbols:
                            symbols.append(s)

            # Alpaca / OANDA symbols from config
            if getattr(self.config, 'alpaca', None) and hasattr(self.config.alpaca, 'symbols'):
                for s in self.config.alpaca.symbols[:3]:
                    if s not in symbols:
                        symbols.append(s)
            if getattr(self.config, 'oanda', None) and hasattr(self.config.oanda, 'symbols'):
                for s in self.config.oanda.symbols[:3]:
                    if s not in symbols:
                        symbols.append(s)

            # Fallback defaults
            if not symbols:
                symbols = ["BTC", "ETH"]

            logger.info(f"[Runtime] CORTEX auto-training on: {symbols}")

            result = trainer.train_all(symbols, days=90, epochs=20)

            if result.get("trained"):
                cortex_path = self.config.get_data_path("cortex_shared.pt")
                self._shared_cortex.save(cortex_path)
                logger.info(
                    "[Runtime] CORTEX trained: %d samples from %d symbols (%s)",
                    result.get("total_samples", 0), len(symbols),
                    ", ".join(symbols))
                return  # Success — skip trade fallback
            else:
                logger.info("[Runtime] CORTEX kline training skipped: %s",
                            result.get("reason", "insufficient data"))

        except Exception as e:
            logger.info(f"[Runtime] CORTEX kline training not available: {e}")

        # --- Phase 2: Fallback — bootstrap from historical closed trades ---
        import torch
        features, targets = [], []

        stores = []
        if self._trading_engine and hasattr(self._trading_engine, 'store'):
            stores.append(self._trading_engine.store)
        if self._futures_engine and hasattr(self._futures_engine, 'store'):
            stores.append(self._futures_engine.store)
        for eng in getattr(self, '_exchange_engines', {}).values():
            if hasattr(eng, 'store'):
                stores.append(eng.store)

        for store in stores:
            for trade in store.trades.values():
                if trade.get("status") == "open":
                    continue
                entry = trade.get("entry_price", 0)
                pnl = trade.get("pnl", 0)
                if entry <= 0:
                    continue
                analysis = {
                    "current": entry,
                    "rsi": 50.0, "rsi6": 50.0,
                    "ema21": entry, "ema50": entry, "ema200": entry,
                    "atr": entry * 0.02,
                    "bullish_tfs": 3, "total_tfs": 6,
                }
                symbol = trade.get("symbol", "BTCUSDT")
                try:
                    feat = self._shared_cortex._build_features(analysis, symbol)
                    direction = trade.get("direction", trade.get("side", "LONG"))
                    if direction in ("LONG", "BUY"):
                        target = 1.0 if pnl > 0 else -1.0
                    else:
                        target = -1.0 if pnl > 0 else 1.0
                    features.append(feat.clone())
                    targets.append(target)
                except Exception:
                    continue

        if len(features) >= 5:
            result = self._shared_cortex.train_batch(
                features, targets, epochs=10, lr=0.001)
            if result.get("trained"):
                cortex_path = self.config.get_data_path("cortex_shared.pt")
                self._shared_cortex.save(cortex_path)
                logger.info("[Runtime] CORTEX bootstrapped from %d historical trades "
                            "(loss=%.4f)", len(features), result["final_loss"])
        elif features:
            logger.info("[Runtime] CORTEX: only %d historical trades "
                        "(need 5+ to bootstrap)", len(features))

    def _run_cleanup(self) -> dict:
        """Periodic cleanup — 21 maintenance steps:
        1. Live memory cleanup (remove stale category="live" entries)
        2. Checkpoint rotation (save daily snapshot + promote/prune)
        3. CMS slow memory decay (slow layers only)
        4. Knowledge graph compaction (decay + GC + compact + importance)
        5. Cache expiry (remove entries past TTL)
        6. Chat session cleanup (remove old sessions)
        7. Trade pattern learning (analyze closed trades)
        8. Session knowledge extraction (entities from chat)
        9. Learn folder -> tree (drop .txt files)
        10. Clear temp cache folders
        11. Cold tier demotion (archive old nodes)
        12. Snapshot compression (hourly rollup)
        13. Background fact-checking (6h cycle)
        14. Graph health check
        15. Emergency RAM demotion (if memory > 90%)
        16. News dedup (merge near-duplicate embeddings)
        17. Batch importance recalculation
        18. Topic clustering (group related nodes)
        19. Embedding precision downgrade (f32 -> f16 for old nodes)
        20. Source reliability report (weekly)
        21. HMM regime model retraining (periodic)
        """
        retention_days = self.config.runtime.live_retention_days
        result = {}

        # 1. Live memory cleanup — only removes category="live" entries
        if self._gate is not None:
            removed = self._gate.memory.cleanup_live(max_age_days=retention_days)
            result["memory_removed"] = removed
            if removed > 0:
                logger.info(f"[Cleanup] Removed {removed} stale live memory entries")

        # 2. Checkpoint rotation — save daily snapshot + promote/prune
        if self._checkpoint_rotator is not None and self._learner is not None:
            saved = self._checkpoint_rotator.save_snapshot(self._learner)
            if saved:
                logger.info(f"[Cleanup] Saved daily snapshot: {os.path.basename(saved)}")
            self._checkpoint_rotator.rotate()

        # 3. CMS slow memory decay
        if self._learner is not None:
            decay_rate = self.config.runtime.cms_slow_decay_rate
            if decay_rate > 0:
                for block in self._learner.model.blocks:
                    if hasattr(block, 'cms'):
                        block.cms.decay_slow(decay_rate)
                logger.info(f"[Cleanup] CMS slow decay applied (rate={decay_rate})")

        # 4. Knowledge graph compaction + importance recompute
        if self._graph is not None:
            try:
                if self._graph.is_open:
                    gc_conf = self.config.runtime.graph_gc_min_confidence
                    gc_result = self._graph.full_cleanup(min_confidence=gc_conf)
                    result["graph_removed"] = gc_result.get("removed", 0)
                    result["importance_recomputed"] = gc_result.get("importance_recomputed", 0)
                    if gc_result.get("removed", 0) > 0:
                        logger.info(f"[Cleanup] Graph: removed {gc_result['removed']} stale edges")
                    recomputed = gc_result.get("importance_recomputed", 0)
                    if recomputed > 0:
                        logger.info(f"[Cleanup] Graph: recomputed importance for {recomputed} nodes")
            except Exception as e:
                logger.warning(f"[Cleanup] Graph cleanup error: {e}")

        # 5. Cache expiry — remove entries past their TTL
        if self._cache_store is not None:
            cache_removed = self._cache_store.cleanup_expired()
            result["cache_removed"] = cache_removed
            if cache_removed > 0:
                logger.info(f"[Cleanup] Removed {cache_removed} expired cache entries")

        # 6. Chat expiry — remove sessions older than retention period
        if self._chat_store is not None:
            retention = getattr(self.config.runtime, 'chat_retention_days', 90)
            chat_removed = self._chat_store.cleanup_old(max_age_days=retention)
            result["chat_removed"] = chat_removed
            if chat_removed > 0:
                logger.info(f"[Cleanup] Removed {chat_removed} old chat messages")

        # 7. Trade pattern learning — analyze closed trades for lessons
        if self._graph is not None and self._graph.is_open:
            try:
                from .knowledge_tree import TradeLearner
                user_id = getattr(self._gate, '_user_id', 'user:default') if self._gate else 'user:default'
                for engine in [self._trading_engine, self._futures_engine]:
                    if engine is not None and hasattr(engine, 'store'):
                        lessons = TradeLearner.analyze_trades(
                            engine.store, self._graph, user_id)
                        if lessons:
                            result["new_lessons"] = len(lessons)
                            logger.info(f"[Cleanup] Trade learning: {len(lessons)} new lessons")
            except Exception as e:
                logger.debug(f"[Cleanup] Trade learning error: {e}")

        # 8. Session knowledge extraction — extract entities from recent chat
        if (self._chat_store is not None and self._graph is not None
                and self._graph.is_open):
            try:
                from .confidence import _extract_entities_and_edges
                sessions = self._chat_store.list_sessions()
                # list_sessions returns list of dicts with session_id
                session_ids = [s.get("session_id", "") for s in sessions[-5:]]
                extracted = 0
                for sid in session_ids:
                    msgs = self._chat_store.get_history(sid, last_n=10)
                    text = " ".join(getattr(m, 'content', '') for m in msgs
                                    if getattr(m, 'role', '') == "assistant")
                    if not text:
                        continue
                    triples = _extract_entities_and_edges(text)
                    for subj, pred, obj in triples[:10]:
                        self._graph.add_edge(subj, pred, obj,
                                             confidence=0.6, source="chat_extraction")
                        extracted += 1
                if extracted > 0:
                    result["chat_entities_extracted"] = extracted
                    logger.info(f"[Cleanup] Extracted {extracted} entities from chat")
            except Exception as e:
                logger.debug(f"[Cleanup] Session extraction error: {e}")

        # 9. Learn folder → tree (drop .txt files, content becomes knowledge nodes)
        learn_dir = getattr(self.config.continual, 'learn_dir', None)
        if learn_dir and self._graph is not None and self._graph.is_open:
            try:
                import hashlib as _hl
                from .confidence import _extract_entities_and_edges
                user_id = (getattr(self._gate, '_user_id', 'user:default')
                           if self._gate else 'user:default')
                files_learned = 0
                for fname in os.listdir(learn_dir):
                    if not fname.endswith('.txt'):
                        continue
                    fpath = os.path.join(learn_dir, fname)
                    if not os.path.isfile(fpath):
                        continue
                    try:
                        with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read().strip()
                        if not content:
                            os.remove(fpath)
                            continue
                        # Save as knowledge node
                        know_hash = _hl.sha256(
                            (fname + ":" + content[:200]).encode()
                        ).hexdigest()[:8]
                        know_id = f"know:{know_hash}"
                        self._graph.add_node(know_id, node_type="knowledge",
                                             properties={
                                                 "content": content[:2000],
                                                 "source": f"learn:{fname}",
                                                 "question": fname.replace('.txt', ''),
                                                 "timestamp": datetime.now(
                                                     timezone.utc).isoformat(),
                                             })
                        self._graph.add_edge(user_id, "learned", know_id,
                                             confidence=0.9, source="learn_file")
                        # Extract entities
                        triples = _extract_entities_and_edges(content)
                        for subj, pred, obj in triples[:15]:
                            self._graph.add_edge(subj, pred, obj,
                                                 confidence=0.7,
                                                 source="learn_file")
                        # Delete file after processing
                        os.remove(fpath)
                        files_learned += 1
                    except Exception:
                        pass
                if files_learned > 0:
                    result["files_learned"] = files_learned
                    logger.info(f"[Cleanup] Learned {files_learned} files → tree")
            except Exception as e:
                logger.debug(f"[Cleanup] Learn folder error: {e}")

        # 10. Clear temp cache folders (vision_stream, tts_cache, screenshots)
        data_dir = self.config.runtime.data_dir
        _temp_dirs = ["vision_stream", "tts_cache", "screenshots", "uploads", "browser-profile"]
        temp_cleared = 0
        for dirname in _temp_dirs:
            dirpath = os.path.join(data_dir, dirname)
            if os.path.isdir(dirpath):
                try:
                    for fname in os.listdir(dirpath):
                        fpath = os.path.join(dirpath, fname)
                        if os.path.isfile(fpath):
                            os.remove(fpath)
                            temp_cleared += 1
                except Exception:
                    pass
        if temp_cleared > 0:
            result["temp_files_cleared"] = temp_cleared
            logger.info(f"[Cleanup] Cleared {temp_cleared} temp files "
                        f"(vision_stream, tts_cache, screenshots)")

        # 11. NGRE: Cold tier demotion — archive old nodes
        if self._cold_tier is not None and self._graph is not None:
            try:
                demoted = self._cold_tier.find_demotable_nodes(self._graph)
                archived = 0
                for node_id in demoted[:50]:  # batch limit
                    node = self._graph.get_node(node_id)
                    if node:
                        self._cold_tier.demote_node(node_id, node)
                        archived += 1
                if archived > 0:
                    result["cold_archived"] = archived
                    logger.info(f"[Cleanup] Archived {archived} nodes to cold tier")
            except Exception as e:
                logger.debug(f"[Cleanup] Cold tier error: {e}")

        # 12. NGRE: Snapshot compression (hourly rollup)
        if self._compressor is not None:
            try:
                comp_result = self._compressor.run_hourly_rollup()
                if comp_result.get("output_count", 0) > 0:
                    result["compression"] = comp_result
                    logger.info(f"[Cleanup] Compression: "
                                f"{comp_result.get('input_count', 0)} → "
                                f"{comp_result.get('output_count', 0)} snapshots")
            except Exception as e:
                logger.debug(f"[Cleanup] Compression error: {e}")

        # 13. NGRE: Background fact-checking (6-hour scheduled cycle)
        if self._fact_checker is not None:
            try:
                if time.time() - self._fact_checker._last_run > 21600:  # 6 hours
                    executor = getattr(self._gate, '_tool_executor', None) if self._gate else None
                    fc_result = self._fact_checker.schedule_6h_cycle(tool_executor=executor)
                    checked = fc_result.get("checked", 0)
                    corrected = fc_result.get("corrected", 0)
                    if checked > 0:
                        result["fact_checked"] = checked
                        result["fact_corrected"] = corrected
                        logger.info(f"[Cleanup] 6h fact-check: {checked} checked, "
                                    f"{corrected} corrected")
                else:
                    elapsed = time.time() - self._fact_checker._last_run
                    remaining = 21600 - elapsed
                    logger.debug(f"[Cleanup] Fact-check: {remaining:.0f}s until next 6h cycle")
            except Exception as e:
                logger.debug(f"[Cleanup] Fact-check error: {e}")

        # 14. NGRE: Graph health check
        if self._health_monitor is not None:
            try:
                health = self._health_monitor.check_all()
                result["graph_health"] = health
            except Exception as e:
                logger.debug(f"[Cleanup] Health check error: {e}")

        # 15. NGRE: Emergency RAM demotion when memory pressure is high
        if self._cold_tier is not None and self._graph is not None:
            try:
                mem_pct = _get_memory_percent()
                if mem_pct > 90.0:
                    # Emergency: aggressively archive to free RAM
                    demotable = self._cold_tier.find_demotable_nodes(self._graph)
                    evict_count = max(int(len(demotable) * 0.25), 20)
                    emergency_archived = 0
                    for nid in demotable[:evict_count]:
                        node = self._graph.get_node(nid)
                        if node:
                            self._cold_tier.demote_node(nid, node)
                            emergency_archived += 1
                    # Also evict hot tier entries
                    if hasattr(self._graph, '_hot_tier') and self._graph._hot_tier:
                        evicted = self._graph._hot_tier.evict(
                            count=self._graph._hot_tier._max_size // 4)
                        result["hot_tier_evicted"] = evicted
                    if emergency_archived > 0:
                        result["emergency_archived"] = emergency_archived
                        logger.warning(
                            f"[Cleanup] Emergency RAM demotion: memory at "
                            f"{mem_pct:.0f}%%, archived {emergency_archived} nodes, "
                            f"target < 75%%")
                elif mem_pct > 80.0:
                    result["memory_pressure"] = f"{mem_pct:.0f}%"
                    logger.info(f"[Cleanup] Memory pressure: {mem_pct:.0f}%% "
                                f"(warning threshold: 90%%)")
            except Exception as e:
                logger.debug(f"[Cleanup] Memory check error: {e}")

        # 16. News dedup — find and merge near-duplicate embeddings
        if self._graph is not None and self._graph.is_open:
            try:
                if hasattr(self._graph, '_embedding_index') and self._graph._embedding_index:
                    dupes = self._graph._embedding_index.find_duplicates(threshold=0.92)
                    dedup_count = 0
                    for id_a, id_b, sim in dupes[:50]:  # limit per cycle
                        # Keep the one with higher importance
                        node_a = self._graph.get_node(id_a, track_access=False)
                        node_b = self._graph.get_node(id_b, track_access=False)
                        if node_a and node_b:
                            imp_a = node_a.get("importance", 0.5)
                            imp_b = node_b.get("importance", 0.5)
                            victim = id_b if imp_a >= imp_b else id_a
                            self._graph.delete_node(victim)
                            dedup_count += 1
                    if dedup_count > 0:
                        result["dedup_merged"] = dedup_count
                        logger.info(f"[Cleanup] Dedup: merged {dedup_count} near-duplicates (sim>0.92)")
            except Exception as e:
                logger.debug(f"[Cleanup] Dedup error: {e}")

        # 17. Batch importance recalculation (runs every cleanup cycle)
        if self._graph is not None and self._graph.is_open:
            try:
                recomputed = self._graph.batch_recompute_importance(limit=500)
                if recomputed > 0:
                    result["importance_recalc"] = recomputed
            except Exception as e:
                logger.debug(f"[Cleanup] Importance recalc error: {e}")

        # 18. Topic clustering — group related nodes
        if self._graph is not None and self._graph.is_open:
            try:
                from .knowledge_tree import cluster_topics
                clustered = cluster_topics(self._graph)
                if clustered > 0:
                    result["topics_clustered"] = clustered
                    logger.info(f"[Cleanup] Clustered {clustered} topic groups")
            except Exception as e:
                logger.debug(f"[Cleanup] Topic clustering error: {e}")

        # 19. Embedding precision downgrade — compress old, low-importance embeddings to f16
        if self._graph is not None and self._graph.is_open:
            try:
                if hasattr(self._graph, '_embedding_index') and self._graph._embedding_index:
                    from .graph import NodeFlags
                    downgraded = 0
                    all_nodes = self._graph.list_nodes()
                    for nid, data in all_nodes:
                        if downgraded >= 100:  # batch limit
                            break
                        importance = data.get("importance", 0.5)
                        flags = data.get("flags", 0)
                        # Only downgrade low-importance nodes not already compressed
                        if importance < 0.3 and not NodeFlags.has(flags, NodeFlags.COMPRESSED):
                            if self._graph._embedding_index.quantize(nid, level="f16"):
                                # Mark as compressed
                                data["flags"] = NodeFlags.set(flags, NodeFlags.COMPRESSED)
                                store_data = {k: v for k, v in data.items() if k != "id"}
                                self._graph._put(f"node:{nid}", store_data)
                                downgraded += 1
                    if downgraded > 0:
                        result["embeddings_downgraded"] = downgraded
                        logger.info(f"[Cleanup] Downgraded {downgraded} embeddings f32->f16")
            except Exception as e:
                logger.debug(f"[Cleanup] Embedding downgrade error: {e}")

        # 20. Source reliability weekly report
        if self._source_reliability is not None and self._cleanup_count % 24 == 0:
            try:
                self._source_reliability.save_to_graph()
                report = self._source_reliability.get_all_stats()
                if report:
                    result["source_reliability_report"] = report
                    logger.info(f"[Cleanup] Source reliability report: {len(report)} sources tracked")
            except Exception as e:
                logger.debug(f"[Cleanup] Source reliability report error: {e}")

        # 21. HMM regime model retraining (periodic, from graph snapshot data)
        if self._hmm is not None and self._hmm.is_available:
            try:
                if self._hmm.should_retrain():
                    hmm_result = self._hmm.train_from_graph(self._graph)
                    if hmm_result.get("trained"):
                        self._hmm.save()
                        result["hmm_retrained"] = True
                        result["hmm_observations"] = hmm_result.get("observations", 0)
                        logger.info(f"[Cleanup] HMM retrained on "
                                    f"{hmm_result.get('observations', 0)} observations")
            except Exception as e:
                logger.debug(f"[Cleanup] HMM retrain error: {e}")

        # Flush memory
        if self._gate and self._gate.memory._dirty:
            self._gate.memory.save()

        self._last_cleanup = datetime.now(timezone.utc).isoformat()
        self._cleanup_count += 1
        result["status"] = "done"
        result["timestamp"] = self._last_cleanup
        return result

    def _schedule_cleanup(self):
        """Schedule the next periodic cleanup."""
        if self._stop_event.is_set():
            return
        hours = self.config.runtime.cleanup_every_hours
        seconds = hours * 3600

        def _run():
            if self._stop_event.is_set():
                return
            logger.info("[Runtime] Scheduled cleanup")
            self._run_cleanup()
            self._schedule_cleanup()

        self._cleanup_timer = threading.Timer(seconds, _run)
        self._cleanup_timer.daemon = True
        self._cleanup_timer.start()

    def _schedule_cortex_retrain(self):
        """Schedule periodic CORTEX retraining from fresh kline data."""
        if self._stop_event.is_set():
            return
        hours = getattr(self.config.runtime, 'cortex_retrain_hours', 6.0)
        if hours <= 0:
            return

        def _run():
            if self._stop_event.is_set():
                return
            self._run_cortex_retrain()
            self._schedule_cortex_retrain()

        self._cortex_retrain_timer = threading.Timer(hours * 3600, _run)
        self._cortex_retrain_timer.daemon = True
        self._cortex_retrain_timer.start()

    def _run_cortex_retrain(self):
        """Periodic CORTEX retraining on fresh kline data for ALL symbols.

        Unlike the startup bootstrap (90 days, 20 epochs), this is lighter:
        - Fetches recent N days (default 7) of candles
        - Trains for 5 epochs (incremental, not from scratch)
        - Covers ALL configured symbols so new assets get fresh data
        """
        if self._shared_cortex is None:
            return
        try:
            from .kline_router import KlineRouter
            from .train_cortex import CortexTrainer

            # Build router with available clients
            kline_client = None
            upstox_kline = None
            if self._trading_engine and hasattr(self._trading_engine, 'client'):
                kline_client = self._trading_engine.client
            elif self._futures_engine and hasattr(self._futures_engine, 'client'):
                kline_client = self._futures_engine.client
            for eng in getattr(self, '_exchange_engines', {}).values():
                if hasattr(eng, 'client') and hasattr(eng.client, '_resolve_instrument_key'):
                    upstox_kline = eng.client
                    break

            router = KlineRouter(
                config=self.config,
                binance_client=kline_client,
                upstox_client=upstox_kline,
            )

            trainer = CortexTrainer(router, self._shared_cortex)

            # Gather ALL configured symbols
            symbols = []
            if getattr(self.config, 'trading', None) and hasattr(self.config.trading, 'symbols'):
                symbols.extend(self.config.trading.symbols)
            if getattr(self.config, 'futures', None) and hasattr(self.config.futures, 'symbols'):
                for s in self.config.futures.symbols:
                    if s not in symbols:
                        symbols.append(s)
            for ex in getattr(self.config, 'exchanges', []):
                if hasattr(ex, 'symbols'):
                    for s in ex.symbols:
                        if s not in symbols:
                            symbols.append(s)
            if getattr(self.config, 'alpaca', None) and hasattr(self.config.alpaca, 'symbols'):
                for s in self.config.alpaca.symbols:
                    if s not in symbols:
                        symbols.append(s)
            if getattr(self.config, 'oanda', None) and hasattr(self.config.oanda, 'symbols'):
                for s in self.config.oanda.symbols:
                    if s not in symbols:
                        symbols.append(s)
            if not symbols:
                symbols = ["BTC", "ETH"]

            days = getattr(self.config.runtime, 'cortex_retrain_days', 7)
            logger.info(f"[Runtime] CORTEX periodic retrain: {symbols} "
                        f"({days}d candles)")

            result = trainer.train_all(
                symbols, days=days, interval="1h", epochs=5)

            if result.get("trained"):
                cortex_path = self.config.get_data_path("cortex_shared.pt")
                self._shared_cortex.save(cortex_path)
                logger.info("[Runtime] CORTEX retrained: %d samples from %s "
                            "(loss=%.4f)",
                            result.get("total_samples", 0),
                            ", ".join(symbols),
                            result.get("final_loss", 0))
            else:
                logger.info("[Runtime] CORTEX retrain skipped: %s",
                            result.get("reason", "no data"))

        except Exception as e:
            logger.warning(f"[Runtime] CORTEX periodic retrain failed: {e}")

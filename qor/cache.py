"""
QOR CacheStore — Tool/API Result Cache
========================================
Separate cache for ephemeral tool/API results with per-entry TTL.
Isolates live/ephemeral data from knowledge entries in MemoryStore.

Location: qor-data/cache.parquet

Each entry has a TTL (time-to-live) matching LIVE_DATA_PATTERNS:
  prices: 5 min | weather: 30 min | news: 60 min
  sports: 15 min | exchange rates: 15 min

Hash chain (integrity.py) provides tamper detection on every record.
"""

import os
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import Optional, List

import pyarrow as pa
import pyarrow.parquet as pq

from .integrity import HashChain


# ==============================================================================
# Schema
# ==============================================================================

CACHE_SCHEMA = pa.schema([
    ("key", pa.string()),
    ("content", pa.string()),
    ("source", pa.string()),
    ("tool_name", pa.string()),
    ("confidence", pa.float32()),
    ("timestamp", pa.timestamp("us", tz="UTC")),
    ("ttl_minutes", pa.int32()),
    ("data_hash", pa.string()),
    ("prev_hash", pa.string()),
])


@dataclass
class CacheEntry:
    """A cached tool/API result."""
    key: str
    content: str
    source: str
    tool_name: str
    confidence: float
    timestamp: str       # ISO format
    ttl_minutes: int
    data_hash: str
    prev_hash: str


class CacheStore:
    """
    Persistent cache for tool/API results with TTL-based expiry.

    Uses Arrow/Parquet for storage (same pattern as MemoryStore).
    Hash chain via integrity.py for tamper detection.
    """

    # Default TTLs matching LIVE_DATA_PATTERNS in confidence.py
    DEFAULT_TTLS = {
        "crypto_price": 5,
        "binance_price": 5,
        "crypto_market": 5,
        "crypto_history": 30,
        "convert_currency": 15,
        "weather": 30,
        "news": 60,
        "news_search": 60,
        "hacker_news": 60,
        "web_search": 60,
        "duckduckgo": 60,
        "reddit": 60,
        "sports": 15,
    }

    # Reuse MemoryStore stop words for search
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

    def __init__(self, path: str = "cache.parquet", secret: str = ""):
        self.path = path
        self.entries = {}  # key -> CacheEntry
        self._chain = HashChain("cache", secret=secret)
        self._dirty = False
        self._dirty_count = 0
        self._flush_every = 50
        self._load()

    def _load(self):
        """Load entries from Parquet file."""
        if not os.path.exists(self.path):
            return
        try:
            table = pq.read_table(self.path, schema=CACHE_SCHEMA)
            keys = table.column("key")
            contents = table.column("content")
            sources = table.column("source")
            tool_names = table.column("tool_name")
            confidences = table.column("confidence")
            timestamps = table.column("timestamp").cast(pa.int64())
            ttls = table.column("ttl_minutes")
            data_hashes = table.column("data_hash")
            prev_hashes = table.column("prev_hash")

            last_hash = HashChain.GENESIS
            for i in range(table.num_rows):
                ts_us = timestamps[i].as_py()
                ts_iso = ""
                if ts_us and ts_us > 0:
                    ts_iso = datetime.fromtimestamp(
                        ts_us / 1_000_000, tz=timezone.utc
                    ).isoformat()

                dh = data_hashes[i].as_py() or ""
                ph = prev_hashes[i].as_py() or ""

                entry = CacheEntry(
                    key=keys[i].as_py() or "",
                    content=contents[i].as_py() or "",
                    source=sources[i].as_py() or "",
                    tool_name=tool_names[i].as_py() or "",
                    confidence=confidences[i].as_py() or 0.0,
                    timestamp=ts_iso,
                    ttl_minutes=ttls[i].as_py() or 60,
                    data_hash=dh,
                    prev_hash=ph,
                )
                self.entries[entry.key] = entry
                if dh:
                    last_hash = dh

            # Restore chain head
            self._chain.set_head(last_hash)
        except Exception:
            pass

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

    @staticmethod
    def _clean_words(text: str) -> set:
        """Split text into clean lowercase words."""
        cleaned = text.lower().replace(":", " ").replace("_", " ").replace("/", " ")
        cleaned = cleaned.replace("-", " ").replace(",", " ").replace(".", " ")
        cleaned = cleaned.replace("(", " ").replace(")", " ").replace("|", " ")
        cleaned = cleaned.replace("$", " ").replace("%", " ").replace("'", " ")
        return set(cleaned.split())

    def store(self, key: str, content: str, source: str, tool_name: str,
              confidence: float = 0.85, ttl_minutes: int = None):
        """
        Store a tool/API result with hash chain.

        Args:
            key: Cache key (e.g. "tool:crypto_price:bitcoin price")
            content: Tool result text
            source: Source identifier (e.g. "tool:crypto_price")
            tool_name: Tool name for TTL lookup
            confidence: Confidence score (0-1)
            ttl_minutes: Override TTL (default: lookup from DEFAULT_TTLS)
        """
        if ttl_minutes is None:
            ttl_minutes = self.DEFAULT_TTLS.get(tool_name, 60)

        prev = self._chain.head
        data_hash = self._chain.compute_hash(content)

        self.entries[key] = CacheEntry(
            key=key,
            content=content,
            source=source,
            tool_name=tool_name,
            confidence=confidence,
            timestamp=datetime.now(timezone.utc).isoformat(),
            ttl_minutes=ttl_minutes,
            data_hash=data_hash,
            prev_hash=prev,
        )
        self._dirty = True
        self._dirty_count += 1
        if self._dirty_count >= self._flush_every:
            self.save()

    def get_fresh(self, key: str, max_age_minutes: int = None) -> Optional[CacheEntry]:
        """
        Get a cache entry if it exists and is not stale.

        Args:
            key: Cache key to look up
            max_age_minutes: Override staleness check (default: use entry's TTL)

        Returns:
            CacheEntry if fresh, None if stale or missing.
        """
        entry = self.entries.get(key)
        if entry is None:
            return None

        ttl = max_age_minutes if max_age_minutes is not None else entry.ttl_minutes

        if not entry.timestamp:
            return None
        try:
            ts = datetime.fromisoformat(entry.timestamp)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            age = datetime.now(timezone.utc) - ts
            if age > timedelta(minutes=ttl):
                return None
        except (ValueError, TypeError):
            return None

        return entry

    def search(self, query: str, top_k: int = 3) -> List[CacheEntry]:
        """Keyword search over cached entries (stop words filtered)."""
        query_words = self._clean_words(query) - self._STOP_WORDS
        if not query_words:
            return []

        scores = []
        for key, entry in self.entries.items():
            content_words = self._clean_words(entry.content) - self._STOP_WORDS
            key_words = self._clean_words(key) - self._STOP_WORDS
            overlap = len(query_words & (content_words | key_words))
            if overlap > 0:
                scores.append((entry, overlap))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [e for e, _ in scores[:top_k]]

    def invalidate(self, key: str) -> bool:
        """Remove a specific cache entry by exact key.

        Returns True if the entry was found and removed.
        """
        if key in self.entries:
            del self.entries[key]
            self._rebuild_chain()
            self.save()
            return True
        return False

    def invalidate_matching(self, query: str, tool_name: str = None) -> int:
        """Remove cache entries whose key matches query keywords.

        Used after user corrections to force re-fetch on next question.

        Args:
            query: Keywords to match against cache keys/content.
            tool_name: If provided, only invalidate entries from this tool.

        Returns:
            Number of entries removed.
        """
        query_words = self._clean_words(query) - self._STOP_WORDS
        if not query_words:
            return 0

        to_delete = []
        for key, entry in self.entries.items():
            if tool_name and entry.tool_name != tool_name:
                continue
            key_words = self._clean_words(key) - self._STOP_WORDS
            content_words = self._clean_words(entry.content) - self._STOP_WORDS
            overlap = len(query_words & (key_words | content_words))
            if overlap >= max(1, len(query_words) // 2):
                to_delete.append(key)

        for key in to_delete:
            del self.entries[key]
        if to_delete:
            self._rebuild_chain()
            self.save()
        return len(to_delete)

    def cleanup_expired(self) -> int:
        """Remove entries past their TTL. Returns count removed."""
        now = datetime.now(timezone.utc)
        to_delete = []

        for key, entry in self.entries.items():
            if not entry.timestamp:
                to_delete.append(key)
                continue
            try:
                ts = datetime.fromisoformat(entry.timestamp)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if (now - ts) > timedelta(minutes=entry.ttl_minutes):
                    to_delete.append(key)
            except (ValueError, TypeError):
                to_delete.append(key)

        for key in to_delete:
            del self.entries[key]
        if to_delete:
            # Rebuild chain from remaining entries
            self._rebuild_chain()
            self.save()
        return len(to_delete)

    def verify_chain(self) -> dict:
        """Check hash chain integrity across all entries."""
        # Build ordered records by insertion order (dict preserves order in Python 3.7+)
        records = []
        for entry in self.entries.values():
            records.append({
                "content": entry.content,
                "data_hash": entry.data_hash,
                "prev_hash": entry.prev_hash,
            })
        return self._chain.verify_chain(records)

    def save(self):
        """Flush all entries to Parquet."""
        if not self.entries:
            # Write empty file so path exists
            table = pa.table({
                "key": pa.array([], type=pa.string()),
                "content": pa.array([], type=pa.string()),
                "source": pa.array([], type=pa.string()),
                "tool_name": pa.array([], type=pa.string()),
                "confidence": pa.array([], type=pa.float32()),
                "timestamp": pa.array([], type=pa.timestamp("us", tz="UTC")),
                "ttl_minutes": pa.array([], type=pa.int32()),
                "data_hash": pa.array([], type=pa.string()),
                "prev_hash": pa.array([], type=pa.string()),
            })
            pq.write_table(table, self.path)
            self._dirty = False
            self._dirty_count = 0
            return

        keys = []
        contents = []
        sources = []
        tool_names = []
        confidences = []
        timestamps = []
        ttls = []
        data_hashes = []
        prev_hashes = []

        for entry in self.entries.values():
            keys.append(entry.key)
            contents.append(entry.content)
            sources.append(entry.source)
            tool_names.append(entry.tool_name)
            confidences.append(entry.confidence)
            timestamps.append(self._iso_to_us(entry.timestamp))
            ttls.append(entry.ttl_minutes)
            data_hashes.append(entry.data_hash)
            prev_hashes.append(entry.prev_hash)

        batch = pa.RecordBatch.from_pydict(
            {
                "key": keys,
                "content": contents,
                "source": sources,
                "tool_name": tool_names,
                "confidence": confidences,
                "timestamp": timestamps,
                "ttl_minutes": ttls,
                "data_hash": data_hashes,
                "prev_hash": prev_hashes,
            },
            schema=CACHE_SCHEMA,
        )
        table = pa.Table.from_batches([batch])
        pq.write_table(table, self.path)
        self._dirty = False
        self._dirty_count = 0

    def _rebuild_chain(self):
        """Rebuild hash chain after deletion (re-hash remaining entries in order)."""
        self._chain.reset()
        for entry in self.entries.values():
            prev = self._chain.head
            data_hash = self._chain.compute_hash(entry.content)
            entry.prev_hash = prev
            entry.data_hash = data_hash

    def count(self) -> int:
        """Number of cached entries."""
        return len(self.entries)

    def stats(self):
        """Print cache statistics."""
        now = datetime.now(timezone.utc)
        fresh = 0
        stale = 0
        for entry in self.entries.values():
            try:
                ts = datetime.fromisoformat(entry.timestamp)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if (now - ts) <= timedelta(minutes=entry.ttl_minutes):
                    fresh += 1
                else:
                    stale += 1
            except (ValueError, TypeError):
                stale += 1

        print(f"\n  Cache Store (Parquet):")
        print(f"    Total entries: {len(self.entries)}")
        print(f"    Fresh: {fresh}")
        print(f"    Stale: {stale}")
        if os.path.exists(self.path):
            size_kb = os.path.getsize(self.path) / 1024
            print(f"    File: {self.path} ({size_kb:.1f} KB)")

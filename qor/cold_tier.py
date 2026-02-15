"""
QOR Cold Tier Storage
======================
zstd-compressed archive for old graph nodes (90+ days).

PRD Section 12: Three-tier storage:
  Hot:  RAM (HotTierCache, < 1us)
  Warm: RocksDB (existing graph, < 1ms)
  Cold: zstd compressed archive (this module, 50-200ms)

Monthly shards: cold_archive/YYYY-MM.zst
Format: JSON Lines inside zstd (searchable, append-friendly)
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Try to import zstandard
try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False
    logger.debug("zstandard not installed; cold tier uses gzip fallback")

# Gzip fallback
import gzip


@dataclass
class ColdTierConfig:
    """Configuration for cold tier storage."""
    archive_dir: str = "qor-data/cold_archive"
    warm_to_cold_days: int = 90
    compression_level: int = 3   # zstd level (1-22, 3=fast default)
    max_archive_gb: float = 100.0
    emergency_evict_pct: float = 0.25  # Evict 25% of hot tier on overflow


@dataclass
class ColdNodeEntry:
    """A node stored in cold archive."""
    node_id: str
    node_type: str
    properties: Dict[str, Any]
    confidence: float = 0.5
    archived_at: str = ""
    original_timestamp: str = ""


class ColdTier:
    """
    zstd-compressed archive for old graph nodes.

    Usage:
        cold = ColdTier(config)
        cold.demote_nodes(graph, node_ids)  # Warm -> Cold
        node = cold.promote_node("entity:123")  # Cold -> Warm
        results = cold.search("Bitcoin price")  # Search cold archive
    """

    def __init__(self, config: Optional[ColdTierConfig] = None):
        self.config = config or ColdTierConfig()
        os.makedirs(self.config.archive_dir, exist_ok=True)
        self._index = {}  # node_id -> shard filename (in-memory index)
        self._load_index()

    def _shard_name(self, timestamp: str = "") -> str:
        """Get shard filename for a given timestamp."""
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                return f"{dt.strftime('%Y-%m')}.jsonl"
            except (ValueError, AttributeError):
                pass
        return f"{datetime.now().strftime('%Y-%m')}.jsonl"

    def _shard_path(self, shard_name: str, compressed: bool = True) -> str:
        """Get full path to a shard file."""
        ext = ".zst" if (HAS_ZSTD and compressed) else ".gz"
        return os.path.join(self.config.archive_dir, shard_name + ext)

    def _load_index(self):
        """Load cold tier index from disk."""
        index_path = os.path.join(self.config.archive_dir, "index.json")
        if os.path.exists(index_path):
            try:
                with open(index_path, "r") as f:
                    self._index = json.load(f)
            except Exception:
                self._index = {}

    def _save_index(self):
        """Persist cold tier index to disk."""
        index_path = os.path.join(self.config.archive_dir, "index.json")
        try:
            with open(index_path, "w") as f:
                json.dump(self._index, f)
        except Exception as e:
            logger.warning("Failed to save cold index: %s", e)

    def _compress_data(self, data: bytes) -> bytes:
        """Compress data with zstd or gzip fallback."""
        if HAS_ZSTD:
            cctx = zstd.ZstdCompressor(level=self.config.compression_level)
            return cctx.compress(data)
        return gzip.compress(data, compresslevel=6)

    def _decompress_data(self, data: bytes) -> bytes:
        """Decompress data with zstd or gzip fallback."""
        if HAS_ZSTD:
            dctx = zstd.ZstdDecompressor()
            return dctx.decompress(data)
        return gzip.decompress(data)

    def demote_nodes(self, graph, node_ids: List[str]) -> int:
        """
        Demote nodes from warm (RocksDB) to cold (zstd archive).

        Args:
            graph: QORGraph instance
            node_ids: list of entity IDs to demote

        Returns:
            Number of nodes successfully demoted
        """
        if not graph or not node_ids:
            return 0

        # Group nodes by target shard
        shard_entries = {}
        demoted = 0

        for nid in node_ids:
            data = graph.get_node(nid)
            if not data:
                continue

            props = data.get("properties", {})
            ts = props.get("timestamp", "")
            shard = self._shard_name(ts)

            entry = ColdNodeEntry(
                node_id=nid,
                node_type=props.get("type", "knowledge"),
                properties=props,
                confidence=props.get("confidence", 0.5),
                archived_at=datetime.now().isoformat(),
                original_timestamp=ts,
            )

            if shard not in shard_entries:
                shard_entries[shard] = []
            shard_entries[shard].append(entry)

        # Write to shards
        for shard_name, entries in shard_entries.items():
            try:
                self._append_to_shard(shard_name, entries)
                for entry in entries:
                    self._index[entry.node_id] = shard_name
                    # Delete from warm tier
                    try:
                        graph.delete_node(entry.node_id)
                        demoted += 1
                    except Exception:
                        pass
            except Exception as e:
                logger.warning("Failed to write shard %s: %s", shard_name, e)

        self._save_index()
        logger.info("Demoted %d nodes to cold tier", demoted)
        return demoted

    def _append_to_shard(self, shard_name: str,
                         entries: List[ColdNodeEntry]):
        """Append entries to a compressed shard file."""
        # Read existing data
        shard_path = self._shard_path(shard_name)
        existing_lines = []

        if os.path.exists(shard_path):
            try:
                with open(shard_path, "rb") as f:
                    raw = self._decompress_data(f.read())
                    existing_lines = raw.decode("utf-8").strip().split("\n")
                    existing_lines = [l for l in existing_lines if l.strip()]
            except Exception:
                existing_lines = []

        # Add new entries
        for entry in entries:
            line = json.dumps({
                "node_id": entry.node_id,
                "node_type": entry.node_type,
                "properties": entry.properties,
                "confidence": entry.confidence,
                "archived_at": entry.archived_at,
                "original_timestamp": entry.original_timestamp,
            })
            existing_lines.append(line)

        # Write compressed
        data = ("\n".join(existing_lines) + "\n").encode("utf-8")
        compressed = self._compress_data(data)
        with open(shard_path, "wb") as f:
            f.write(compressed)

    def promote_node(self, node_id: str, graph=None) -> Optional[Dict]:
        """
        Promote a node from cold tier back to warm (RocksDB).

        Args:
            node_id: entity ID to promote
            graph: QORGraph to write back to (optional)

        Returns:
            Node data dict, or None if not found
        """
        shard_name = self._index.get(node_id)
        if not shard_name:
            return None

        shard_path = self._shard_path(shard_name)
        if not os.path.exists(shard_path):
            return None

        # Read and find the node
        try:
            with open(shard_path, "rb") as f:
                raw = self._decompress_data(f.read())
            lines = raw.decode("utf-8").strip().split("\n")

            for line in lines:
                if not line.strip():
                    continue
                entry = json.loads(line)
                if entry.get("node_id") == node_id:
                    # Write back to graph if provided
                    if graph and hasattr(graph, 'add_node'):
                        graph.add_node(
                            node_id,
                            node_type=entry.get("node_type", "knowledge"),
                            properties=entry.get("properties", {}),
                            confidence=entry.get("confidence", 0.5),
                        )
                        # Remove from index
                        self._index.pop(node_id, None)
                        self._save_index()
                    return entry
        except Exception as e:
            logger.warning("Failed to promote %s: %s", node_id, e)

        return None

    def search(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        Search cold archive by keyword matching.

        Args:
            query: search query string
            max_results: maximum results to return

        Returns:
            List of matching node dicts
        """
        results = []
        query_lower = query.lower()
        query_words = set(query_lower.split())

        # Scan all shards
        for fname in os.listdir(self.config.archive_dir):
            if not (fname.endswith(".zst") or fname.endswith(".gz")):
                continue

            fpath = os.path.join(self.config.archive_dir, fname)
            try:
                with open(fpath, "rb") as f:
                    raw = self._decompress_data(f.read())
                lines = raw.decode("utf-8").strip().split("\n")

                for line in lines:
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    content = json.dumps(
                        entry.get("properties", {})).lower()
                    # Simple keyword match
                    matches = sum(
                        1 for w in query_words if w in content)
                    if matches >= max(1, len(query_words) // 2):
                        entry["_score"] = matches
                        results.append(entry)

                    if len(results) >= max_results * 3:
                        break  # Enough candidates
            except Exception:
                continue

        # Sort by score and return top results
        results.sort(key=lambda x: x.get("_score", 0), reverse=True)
        return results[:max_results]

    def find_demotable_nodes(self, graph,
                             max_age_days: int = 0,
                             limit: int = 1000) -> List[str]:
        """
        Find warm-tier nodes eligible for cold demotion.

        Criteria (PRD Section 12.1):
          - Age > warm_to_cold_days (default 90)
          - access_count < 10
          - Not summary or entity type
        """
        if max_age_days <= 0:
            max_age_days = self.config.warm_to_cold_days

        if not graph or not graph.is_open:
            return []

        cutoff = (datetime.now() - timedelta(
            days=max_age_days)).isoformat()
        candidates = []

        try:
            # Scan nodes
            all_nodes = graph.list_nodes(limit=limit * 5)
            for nid in all_nodes:
                if len(candidates) >= limit:
                    break

                data = graph.get_node(nid, track_access=False)
                if not data:
                    continue

                props = data.get("properties", {})
                ts = props.get("timestamp", "")
                ntype = props.get("type", "").lower()
                access_count = props.get("access_count", 0)

                # Skip protected types
                if ntype in ("user", "entity", "summary", "preference"):
                    continue

                # Skip anchored/pinned nodes
                flags = props.get("flags", 0)
                if isinstance(flags, int) and (flags & 0x0A):  # ANCHORED|PINNED
                    continue

                # Check age
                if ts and ts < cutoff and access_count < 10:
                    candidates.append(nid)
        except Exception as e:
            logger.warning("Error finding demotable nodes: %s", e)

        return candidates

    def stats(self) -> Dict[str, Any]:
        """Return cold tier statistics."""
        total_size = 0
        shard_count = 0
        node_count = len(self._index)

        for fname in os.listdir(self.config.archive_dir):
            if fname.endswith(".zst") or fname.endswith(".gz"):
                fpath = os.path.join(self.config.archive_dir, fname)
                total_size += os.path.getsize(fpath)
                shard_count += 1

        return {
            "node_count": node_count,
            "shard_count": shard_count,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "archive_dir": self.config.archive_dir,
            "has_zstd": HAS_ZSTD,
        }

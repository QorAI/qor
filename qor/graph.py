"""
QOR Knowledge Graph — RocksDB-Backed Graph Database
=====================================================
A persistent, queryable knowledge graph that replaces flat JSON memory
with structured entity-relationship storage.

Key Schema:
    node:{entity_id}                            -> {type, properties, created_at, updated_at}
    edge:{subject}:{predicate}:{object}         -> {confidence, timestamp, source, weight}
    reverse:{object}:{predicate_inv}:{subject}  -> {forward_key}
    syn:{predicate}                             -> {canonical, aliases: [...]}
    idx:sub:{entity_id}                         -> [list of edge keys from this entity]
    idx:obj:{entity_id}                         -> [list of edge keys to this entity]
    idx:pred:{predicate}                        -> [list of edge keys with this predicate]

Backend:
    rocksdict (Rust-based RocksDB binding, works on Windows)
    REQUIRED — no fallback, the graph must persist to disk

Usage:
    from qor.graph import QORGraph, GraphConfig

    config = GraphConfig(db_path="my_graph")
    with QORGraph(config) as g:
        g.add_node("python", node_type="language", properties={"paradigm": "multi"})
        g.add_node("guido", node_type="person")
        g.add_edge("guido", "created", "python", confidence=1.0)
        edges = g.get_edges("guido", direction="out")
        path = g.find_path("guido", "python")
"""

import json
import logging
import math
import os
import re
import shutil
import time
from collections import deque, OrderedDict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ==============================================================================
# SERIALIZATION — msgpack preferred, json fallback
# ==============================================================================

try:
    import msgpack
except ImportError:
    raise ImportError(
        "msgpack is REQUIRED for graph serialization.\n"
        "Install it: pip install msgpack"
    )


def _serialize(obj: Any) -> bytes:
    return msgpack.packb(obj, use_bin_type=True)


def _deserialize(data: bytes) -> Any:
    return msgpack.unpackb(data, raw=False)


_SERIALIZER = "msgpack"


# ==============================================================================
# NODE TYPES — Typed graph nodes for NGRE architecture
# ==============================================================================

class NodeType(IntEnum):
    """Typed node categories for the knowledge graph.

    Every node has a type_id (int) stored alongside the legacy string type.
    New code uses the enum; old code keeps passing strings — resolve_node_type()
    bridges the two worlds.
    """
    ENTITY = 0
    USER = 1
    KNOWLEDGE = 2
    CORRECTION = 3
    BLOCKED_FACT = 4
    PREFERENCE = 5
    TRADE_PATTERN = 6
    LESSON = 7
    HISTORICAL = 8
    TOPIC = 9
    SNAPSHOT = 10
    EVENT = 11
    RELATIONSHIP = 12
    SUMMARY = 13
    CATEGORY = 14
    TEMPORAL = 15


# Legacy string → NodeType mapping (backward compat bridge)
NODE_TYPE_MAP: Dict[str, NodeType] = {
    "entity": NodeType.ENTITY,
    "user": NodeType.USER,
    "knowledge": NodeType.KNOWLEDGE,
    "correction": NodeType.CORRECTION,
    "blocked_fact": NodeType.BLOCKED_FACT,
    "preference": NodeType.PREFERENCE,
    "trade_pattern": NodeType.TRADE_PATTERN,
    "lesson": NodeType.LESSON,
    "historical_event": NodeType.HISTORICAL,
    "historical": NodeType.HISTORICAL,
    "topic": NodeType.TOPIC,
    "session_topic": NodeType.TOPIC,
    "snapshot": NodeType.SNAPSHOT,
    "event": NodeType.EVENT,
    "relationship": NodeType.RELATIONSHIP,
    "summary": NodeType.SUMMARY,
    "category": NodeType.CATEGORY,
    "temporal": NodeType.TEMPORAL,
    # Common legacy strings from tests/examples
    "person": NodeType.ENTITY,
    "language": NodeType.ENTITY,
    "cryptocurrency": NodeType.ENTITY,
    "market": NodeType.ENTITY,
    "consensus": NodeType.ENTITY,
    "concept": NodeType.ENTITY,
}

# Reverse mapping: NodeType → canonical string for storage
NODE_TYPE_NAMES: Dict[NodeType, str] = {
    NodeType.ENTITY: "entity",
    NodeType.USER: "user",
    NodeType.KNOWLEDGE: "knowledge",
    NodeType.CORRECTION: "correction",
    NodeType.BLOCKED_FACT: "blocked_fact",
    NodeType.PREFERENCE: "preference",
    NodeType.TRADE_PATTERN: "trade_pattern",
    NodeType.LESSON: "lesson",
    NodeType.HISTORICAL: "historical_event",
    NodeType.TOPIC: "topic",
    NodeType.SNAPSHOT: "snapshot",
    NodeType.EVENT: "event",
    NodeType.RELATIONSHIP: "relationship",
    NodeType.SUMMARY: "summary",
    NodeType.CATEGORY: "category",
    NodeType.TEMPORAL: "temporal",
}


def resolve_node_type(raw) -> NodeType:
    """Convert any representation to a NodeType enum value.

    Accepts: string ("entity"), int (0), or NodeType enum.
    Unknown strings default to ENTITY for backward compatibility.
    """
    if isinstance(raw, NodeType):
        return raw
    if isinstance(raw, int):
        try:
            return NodeType(raw)
        except ValueError:
            return NodeType.ENTITY
    if isinstance(raw, str):
        return NODE_TYPE_MAP.get(raw.lower().strip(), NodeType.ENTITY)
    return NodeType.ENTITY


# ==============================================================================
# NODE FLAGS — Bitfield flags for node metadata
# ==============================================================================

class NodeFlags:
    """Bitfield flags for node importance overrides and lifecycle state."""
    LANDMARK = 0x01      # Permanently important node (importance=1.0)
    ANCHORED = 0x02      # Minimum importance floor (importance >= 0.8)
    QUARANTINED = 0x04   # Untrusted/disputed (importance=0.0 for scoring)
    PINNED = 0x08        # Never auto-deleted by cleanup
    STALE = 0x10         # Marked for refresh on next access
    STAGED = 0x20        # Node staged for promotion to higher tier
    PROMOTED = 0x40      # Node has been promoted (tier upgrade complete)
    COMPRESSED = 0x80    # Embedding has been quantized/compressed
    NEEDS_RECHECK = 0x100  # Node flagged for fact-check verification

    @staticmethod
    def has(flags: int, flag: int) -> bool:
        """Check if a flag is set."""
        return bool(flags & flag)

    @staticmethod
    def set(flags: int, flag: int) -> int:
        """Set a flag."""
        return flags | flag

    @staticmethod
    def clear(flags: int, flag: int) -> int:
        """Clear a flag."""
        return flags & ~flag


# ==============================================================================
# TYPE BONUS — Per-type importance score bonus
# ==============================================================================

_TYPE_BONUS: Dict[NodeType, float] = {
    NodeType.ENTITY: 0.0,
    NodeType.USER: 0.3,
    NodeType.KNOWLEDGE: 0.1,
    NodeType.CORRECTION: 0.2,
    NodeType.BLOCKED_FACT: 0.15,
    NodeType.PREFERENCE: 0.2,
    NodeType.TRADE_PATTERN: 0.15,
    NodeType.LESSON: 0.25,
    NodeType.HISTORICAL: 0.2,
    NodeType.TOPIC: 0.05,
    NodeType.SNAPSHOT: 0.1,
    NodeType.EVENT: 0.15,
    NodeType.RELATIONSHIP: 0.05,
    NodeType.SUMMARY: 0.1,
    NodeType.CATEGORY: 0.05,
    NodeType.TEMPORAL: 0.1,
}


# ==============================================================================
# BACKEND — rocksdict (REQUIRED)
# ==============================================================================

try:
    from rocksdict import Rdict, Options, WriteBatch, BlockBasedOptions
except ImportError:
    raise ImportError(
        "rocksdict is REQUIRED for the QOR knowledge graph.\n"
        "Install it: pip install rocksdict\n"
        "This is NOT optional — without it, the graph cannot persist to disk."
    )


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class GraphConfig:
    """Configuration for the QOR Knowledge Graph."""
    db_path: str = "graph_db"               # Directory for RocksDB storage
    cache_size_mb: int = 64                 # Block cache size in MB
    enable_bloom_filters: bool = True       # Enable bloom filters for faster lookups
    bloom_bits_per_key: int = 10            # Bits per key for bloom filter
    max_open_files: int = 256               # Max open file descriptors
    serializer: str = "msgpack"             # "msgpack" or "json" (auto-detected)

    # NGRE fields
    enable_embeddings: bool = True          # Enable HNSW embedding index
    embedding_dim: int = 768                # Embedding vector dimension
    hnsw_max_elements: int = 100_000        # Max elements in HNSW index
    hnsw_ef_construction: int = 200         # HNSW construction parameter
    hnsw_M: int = 16                        # HNSW connections per element
    hot_tier_max_nodes: int = 10_000        # Max nodes in hot tier cache
    hot_tier_promote_threshold: int = 3     # Access count to promote to hot tier
    hebbian_lr: float = 0.001               # Hebbian learning rate
    importance_recency_decay: float = 0.01  # Recency decay for importance scoring


# ==============================================================================
# HOT TIER CACHE — In-memory LRU for frequently accessed nodes
# ==============================================================================

class HotTierCache:
    """In-memory LRU cache for frequently accessed graph nodes.

    Write-through to RocksDB — the cache is a performance layer, not a
    separate store. Eviction is LRU (least recently used).
    """

    def __init__(self, max_nodes: int = 10_000, promote_threshold: int = 3):
        self._max = max_nodes
        self._promote_threshold = promote_threshold
        self._cache: OrderedDict = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, nid: str) -> Optional[Dict]:
        """Get a node from the hot tier. Returns None on miss."""
        if nid in self._cache:
            self._cache.move_to_end(nid)
            self._hits += 1
            return self._cache[nid]
        self._misses += 1
        return None

    def put(self, nid: str, data: Dict):
        """Add or update a node in the hot tier."""
        if nid in self._cache:
            self._cache.move_to_end(nid)
            self._cache[nid] = data
        else:
            self._cache[nid] = data
            if len(self._cache) > self._max:
                self._cache.popitem(last=False)

    @property
    def _max_size(self) -> int:
        """Maximum cache capacity."""
        return self._max

    def should_promote(self, access_count: int) -> bool:
        """Check if a node's access count qualifies for hot tier promotion."""
        return access_count >= self._promote_threshold

    def evict(self, nid: str = None, count: int = 0) -> int:
        """Remove node(s) from the hot tier.

        If nid is given, remove that specific node.
        If count > 0, evict `count` LRU entries and return number evicted.
        """
        if nid is not None:
            self._cache.pop(nid, None)
            return 1
        evicted = 0
        for _ in range(min(count, len(self._cache))):
            self._cache.popitem(last=False)
            evicted += 1
        return evicted

    def clear(self):
        """Clear all cached nodes."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def stats(self) -> Dict:
        """Return hit/miss statistics."""
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max": self._max,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 4) if total > 0 else 0.0,
        }


# ==============================================================================
# EMBEDDING INDEX — HNSW approximate nearest neighbor search
# ==============================================================================

class EmbeddingIndex:
    """HNSW index for node embedding vectors.

    Uses hnswlib for fast ANN search. Falls back to brute-force numpy
    if hnswlib is not installed. Persists to .npy + .hnsw files alongside
    the RocksDB directory.
    """

    def __init__(self, dim: int = 768, max_elements: int = 100_000,
                 ef_construction: int = 200, M: int = 16):
        self._dim = dim
        self._max_elements = max_elements
        self._ef_construction = ef_construction
        self._M = M
        self._use_hnsw = False
        self._index = None
        self._id_to_label: Dict[str, int] = {}
        self._label_to_id: Dict[int, str] = {}
        self._next_label = 0
        # Brute-force fallback storage
        self._np_vectors: Optional[Any] = None  # numpy array
        self._np_ids: List[str] = []

        try:
            import hnswlib
            self._hnsw_lib = hnswlib
            self._use_hnsw = True
        except ImportError:
            self._hnsw_lib = None
            self._use_hnsw = False

        try:
            import numpy as np
            self._np = np
        except ImportError:
            self._np = None

    def init_index(self):
        """Initialize the HNSW index (or numpy fallback)."""
        if self._use_hnsw and self._hnsw_lib is not None:
            self._index = self._hnsw_lib.Index(space='cosine', dim=self._dim)
            self._index.init_index(
                max_elements=self._max_elements,
                ef_construction=self._ef_construction,
                M=self._M,
            )
            self._index.set_ef(50)
        # Always initialize numpy mirror (used by quantize, cosine_similarity, etc.)
        if self._np is not None:
            self._np_vectors = self._np.empty((0, self._dim), dtype='float32')
            self._np_ids = []

    def add(self, node_id: str, embedding) -> bool:
        """Add a vector for a node. Returns True on success."""
        if self._np is not None:
            embedding = self._np.asarray(embedding, dtype='float32').ravel()
            if embedding.shape[0] != self._dim:
                return False

        if self._use_hnsw and self._index is not None:
            if node_id in self._id_to_label:
                # Update: HNSW doesn't support in-place update, skip
                return True
            label = self._next_label
            self._next_label += 1
            try:
                self._index.add_items(embedding.reshape(1, -1), [label])
                self._id_to_label[node_id] = label
                self._label_to_id[label] = node_id
            except Exception:
                return False
            # Also store in numpy mirror for quantize/cosine_similarity/find_duplicates
            if self._np is not None and self._np_vectors is not None:
                if node_id in self._np_ids:
                    idx = self._np_ids.index(node_id)
                    self._np_vectors[idx] = embedding
                else:
                    self._np_vectors = self._np.vstack(
                        [self._np_vectors, embedding.reshape(1, -1)]
                    )
                    self._np_ids.append(node_id)
            return True
        elif self._np is not None and self._np_vectors is not None:
            if node_id in self._np_ids:
                idx = self._np_ids.index(node_id)
                self._np_vectors[idx] = embedding
            else:
                self._np_vectors = self._np.vstack(
                    [self._np_vectors, embedding.reshape(1, -1)]
                )
                self._np_ids.append(node_id)
            return True
        return False

    def search(self, embedding, k: int = 10,
               filter_ids: Optional[set] = None) -> List[Tuple[str, float]]:
        """Search for k nearest neighbors. Returns list of (node_id, distance)."""
        if self._np is not None:
            embedding = self._np.asarray(embedding, dtype='float32').ravel()

        if self._use_hnsw and self._index is not None:
            if self._index.get_current_count() == 0:
                return []
            actual_k = min(k, self._index.get_current_count())
            try:
                labels, distances = self._index.knn_query(
                    embedding.reshape(1, -1), k=actual_k
                )
                results = []
                for label, dist in zip(labels[0], distances[0]):
                    nid = self._label_to_id.get(int(label))
                    if nid is not None:
                        if filter_ids is None or nid in filter_ids:
                            results.append((nid, float(dist)))
                return results
            except Exception:
                return []
        elif self._np is not None and self._np_vectors is not None:
            if len(self._np_ids) == 0:
                return []
            # Brute-force cosine distance
            norms = self._np.linalg.norm(self._np_vectors, axis=1, keepdims=True)
            norms = self._np.clip(norms, 1e-8, None)
            normed = self._np_vectors / norms
            q_norm = embedding / max(self._np.linalg.norm(embedding), 1e-8)
            sims = normed @ q_norm
            actual_k = min(k, len(self._np_ids))
            top_idx = self._np.argsort(sims)[::-1][:actual_k]
            results = []
            for idx in top_idx:
                nid = self._np_ids[idx]
                dist = 1.0 - float(sims[idx])  # cosine distance
                if filter_ids is None or nid in filter_ids:
                    results.append((nid, dist))
            return results
        return []

    def remove(self, node_id: str):
        """Remove a node's embedding. HNSW mark_deleted, numpy removes row."""
        if self._use_hnsw and self._index is not None:
            label = self._id_to_label.pop(node_id, None)
            if label is not None:
                try:
                    self._index.mark_deleted(label)
                except Exception:
                    pass
                self._label_to_id.pop(label, None)
            # Also remove from numpy mirror
            if self._np is not None and self._np_vectors is not None:
                if node_id in self._np_ids:
                    idx = self._np_ids.index(node_id)
                    self._np_vectors = self._np.delete(self._np_vectors, idx, axis=0)
                    self._np_ids.pop(idx)
        elif self._np is not None and self._np_vectors is not None:
            if node_id in self._np_ids:
                idx = self._np_ids.index(node_id)
                self._np_vectors = self._np.delete(self._np_vectors, idx, axis=0)
                self._np_ids.pop(idx)

    def save(self, dir_path: str):
        """Save index to disk alongside the RocksDB directory."""
        os.makedirs(dir_path, exist_ok=True)
        if self._use_hnsw and self._index is not None:
            idx_path = os.path.join(dir_path, "embeddings.hnsw")
            try:
                self._index.save_index(idx_path)
            except Exception as e:
                logger.warning("Failed to save HNSW index: %s", e)
            # Save ID mappings
            map_path = os.path.join(dir_path, "embedding_ids.json")
            try:
                with open(map_path, "w") as f:
                    json.dump({
                        "id_to_label": self._id_to_label,
                        "next_label": self._next_label,
                    }, f)
            except Exception:
                pass
        elif self._np is not None and self._np_vectors is not None:
            npy_path = os.path.join(dir_path, "embeddings.npy")
            ids_path = os.path.join(dir_path, "embedding_ids.json")
            try:
                self._np.save(npy_path, self._np_vectors)
                with open(ids_path, "w") as f:
                    json.dump({"np_ids": self._np_ids}, f)
            except Exception as e:
                logger.warning("Failed to save numpy embeddings: %s", e)

    def load(self, dir_path: str) -> bool:
        """Load index from disk. Returns True if loaded successfully."""
        if self._use_hnsw and self._hnsw_lib is not None:
            idx_path = os.path.join(dir_path, "embeddings.hnsw")
            map_path = os.path.join(dir_path, "embedding_ids.json")
            if os.path.exists(idx_path) and os.path.exists(map_path):
                try:
                    self._index = self._hnsw_lib.Index(
                        space='cosine', dim=self._dim
                    )
                    self._index.load_index(
                        idx_path, max_elements=self._max_elements
                    )
                    self._index.set_ef(50)
                    with open(map_path) as f:
                        data = json.load(f)
                    self._id_to_label = {
                        k: int(v) for k, v in data["id_to_label"].items()
                    }
                    self._label_to_id = {
                        v: k for k, v in self._id_to_label.items()
                    }
                    self._next_label = data.get("next_label", 0)
                    return True
                except Exception as e:
                    logger.warning("Failed to load HNSW index: %s", e)
                    self.init_index()
                    return False
        elif self._np is not None:
            npy_path = os.path.join(dir_path, "embeddings.npy")
            ids_path = os.path.join(dir_path, "embedding_ids.json")
            if os.path.exists(npy_path) and os.path.exists(ids_path):
                try:
                    self._np_vectors = self._np.load(npy_path)
                    with open(ids_path) as f:
                        data = json.load(f)
                    self._np_ids = data.get("np_ids", [])
                    return True
                except Exception as e:
                    logger.warning("Failed to load numpy embeddings: %s", e)
                    self.init_index()
                    return False
        return False

    def count(self) -> int:
        """Number of embeddings in the index."""
        if self._use_hnsw and self._index is not None:
            return self._index.get_current_count()
        if self._np_vectors is not None:
            return len(self._np_ids)
        return 0

    # --- 4-level embedding precision (PRD §6.2) ---

    def quantize(self, node_id: str, level: str = "f16") -> bool:
        """Quantize a node's embedding to lower precision.

        Levels: f32 (default), f16 (half), int8 (byte), lsh (binary hash).
        Only affects numpy fallback storage. HNSW always uses f32 internally.
        Returns True if quantization applied.
        """
        if self._np is None or self._np_vectors is None:
            return False
        if node_id not in self._np_ids:
            return False
        idx = self._np_ids.index(node_id)
        vec = self._np_vectors[idx]

        if level == "f16":
            self._np_vectors[idx] = vec.astype('float16').astype('float32')
        elif level == "int8":
            # Scale to [-127, 127] and back
            scale = max(abs(vec.max()), abs(vec.min()), 1e-8)
            quantized = self._np.clip(vec / scale * 127, -127, 127).astype('int8')
            self._np_vectors[idx] = (quantized.astype('float32') / 127) * scale
        elif level == "lsh":
            # Binary LSH: sign of each dimension (1-bit per dim)
            self._np_vectors[idx] = self._np.sign(vec).astype('float32')
        return True

    def cosine_similarity(self, id_a: str, id_b: str) -> float:
        """Compute cosine similarity between two node embeddings."""
        if self._np is None or self._np_vectors is None:
            return 0.0
        if id_a not in self._np_ids or id_b not in self._np_ids:
            return 0.0
        idx_a = self._np_ids.index(id_a)
        idx_b = self._np_ids.index(id_b)
        va = self._np_vectors[idx_a]
        vb = self._np_vectors[idx_b]
        norm_a = self._np.linalg.norm(va)
        norm_b = self._np.linalg.norm(vb)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(self._np.dot(va, vb) / (norm_a * norm_b))

    def find_duplicates(self, threshold: float = 0.92) -> List[Tuple[str, str, float]]:
        """Find near-duplicate embeddings above cosine similarity threshold.

        Returns list of (id_a, id_b, similarity) pairs.
        """
        if self._np is None or self._np_vectors is None:
            return []
        n = len(self._np_ids)
        if n < 2:
            return []
        # Normalize all vectors
        norms = self._np.linalg.norm(self._np_vectors, axis=1, keepdims=True)
        norms = self._np.clip(norms, 1e-8, None)
        normed = self._np_vectors / norms
        # Compute pairwise cosine similarities (upper triangle only)
        dupes = []
        # Batch for efficiency: process in chunks of 500
        batch = min(n, 500)
        for i in range(0, n, batch):
            end_i = min(i + batch, n)
            sims = normed[i:end_i] @ normed.T
            for row_idx in range(end_i - i):
                for col_idx in range(i + row_idx + 1, n):
                    sim = float(sims[row_idx, col_idx])
                    if sim >= threshold:
                        dupes.append((
                            self._np_ids[i + row_idx],
                            self._np_ids[col_idx],
                            sim,
                        ))
        return dupes


# ==============================================================================
# DEFAULT SYNONYMS — Built-in predicate synonyms for natural language queries
# ==============================================================================

DEFAULT_SYNONYMS = {
    "created": {
        "canonical": "created",
        "aliases": ["made", "built", "developed", "authored", "wrote",
                     "invented", "designed", "constructed", "founded"],
    },
    "is_a": {
        "canonical": "is_a",
        "aliases": ["is", "is a", "is an", "type of", "kind of",
                     "instance of", "belongs to", "classified as"],
    },
    "part_of": {
        "canonical": "part_of",
        "aliases": ["part of", "belongs to", "component of",
                     "member of", "included in", "within"],
    },
    "has": {
        "canonical": "has",
        "aliases": ["has", "contains", "includes", "possesses",
                     "owns", "holds", "features"],
    },
    "related_to": {
        "canonical": "related_to",
        "aliases": ["related to", "associated with", "connected to",
                     "linked to", "similar to", "relevant to"],
    },
    "located_in": {
        "canonical": "located_in",
        "aliases": ["located in", "found in", "lives in", "based in",
                     "situated in", "resides in", "in"],
    },
    "causes": {
        "canonical": "causes",
        "aliases": ["causes", "leads to", "results in", "produces",
                     "triggers", "induces", "brings about"],
    },
    "used_for": {
        "canonical": "used_for",
        "aliases": ["used for", "used in", "useful for", "helps with",
                     "applied to", "employed for"],
    },
    "depends_on": {
        "canonical": "depends_on",
        "aliases": ["depends on", "requires", "needs", "relies on",
                     "built on", "based on"],
    },
    "knows": {
        "canonical": "knows",
        "aliases": ["knows", "understands", "learned", "studied",
                     "familiar with", "experienced in"],
    },
    "prefers": {
        "canonical": "prefers",
        "aliases": ["prefers", "likes", "enjoys", "favors", "wants more of"],
    },
    "dislikes": {
        "canonical": "dislikes",
        "aliases": ["dislikes", "hates", "avoids", "not interested in",
                     "wants less of"],
    },
    "corrected": {
        "canonical": "corrected",
        "aliases": ["corrected", "fixed", "amended", "revised", "disputed"],
    },
    "blocked": {
        "canonical": "blocked",
        "aliases": ["blocked", "banned", "rejected", "excluded", "filtered out"],
    },
    "learned_from": {
        "canonical": "learned_from",
        "aliases": ["learned from", "derived from", "extracted from",
                     "concluded from"],
    },
    "resulted_in_lesson": {
        "canonical": "resulted_in_lesson",
        "aliases": ["resulted in lesson", "taught", "demonstrated",
                     "showed pattern"],
    },
    "interests_in": {
        "canonical": "interests_in",
        "aliases": ["interests in", "interested in", "curious about",
                     "follows", "tracks"],
    },
    "learned": {
        "canonical": "learned",
        "aliases": ["learned", "acquired", "discovered",
                     "found out", "gathered"],
    },
}


# ==============================================================================
# QOR KNOWLEDGE GRAPH
# ==============================================================================

class QORGraph:
    """
    RocksDB-backed knowledge graph with structured entity-relationship storage.

    Uses rocksdict (Rust-based RocksDB binding) for persistent storage.
    rocksdict is REQUIRED — there is no fallback.

    Key prefixes simulate column families:
        node:     Entity nodes
        edge:     Forward edges (subject -> object)
        reverse:  Reverse edges (object -> subject)
        syn:      Predicate synonyms
        idx:sub:  Subject index (entity -> outgoing edge keys)
        idx:obj:  Object index (entity -> incoming edge keys)
        idx:pred: Predicate index (predicate -> edge keys)
    """

    def __init__(self, config: Optional[GraphConfig] = None):
        self.config = config or GraphConfig()
        self._db = None
        self._is_open = False
        self._backend_type = "rocksdict"
        self._hot_tier: Optional[HotTierCache] = None
        self._embedding_index: Optional[EmbeddingIndex] = None
        logger.info("QORGraph initialized (backend=%s, serializer=%s)",
                     self._backend_type, _SERIALIZER)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self):
        """Open or create the database."""
        if self._is_open:
            return

        opts = Options()
        opts.set_max_open_files(self.config.max_open_files)
        opts.create_if_missing(True)
        cache_bytes = self.config.cache_size_mb * 1024 * 1024
        opts.set_write_buffer_size(cache_bytes)
        if self.config.enable_bloom_filters:
            table_opts = BlockBasedOptions()
            table_opts.set_bloom_filter(self.config.bloom_bits_per_key, False)
            opts.set_block_based_table_factory(table_opts)
        self._db = Rdict(self.config.db_path, options=opts)
        logger.info("RocksDB opened at %s", self.config.db_path)

        self._is_open = True

        # Initialize hot tier cache
        self._hot_tier = HotTierCache(
            max_nodes=self.config.hot_tier_max_nodes,
            promote_threshold=self.config.hot_tier_promote_threshold,
        )

        # Initialize embedding index
        if self.config.enable_embeddings:
            self._embedding_index = EmbeddingIndex(
                dim=self.config.embedding_dim,
                max_elements=self.config.hnsw_max_elements,
                ef_construction=self.config.hnsw_ef_construction,
                M=self.config.hnsw_M,
            )
            # Try loading from disk first
            emb_dir = os.path.join(self.config.db_path, ".embeddings")
            if not self._embedding_index.load(emb_dir):
                self._embedding_index.init_index()
            logger.info("Embedding index initialized (dim=%d)", self.config.embedding_dim)

        # Load default synonyms if the synonym store is empty
        syn_keys = self._prefix_scan("syn:")
        if not syn_keys:
            self.load_default_synonyms()

    def close(self):
        """Close the database."""
        if self._db is not None and self._is_open:
            # Save embedding index to disk
            if self._embedding_index is not None:
                emb_dir = os.path.join(self.config.db_path, ".embeddings")
                self._embedding_index.save(emb_dir)
            self._db.close()
            self._is_open = False
            if self._hot_tier is not None:
                self._hot_tier.clear()
            logger.info("QORGraph closed")

    def destroy(self):
        """Close and delete the database entirely."""
        self.close()
        if os.path.exists(self.config.db_path):
            shutil.rmtree(self.config.db_path, ignore_errors=True)
            logger.info("Database destroyed at %s", self.config.db_path)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    @property
    def is_open(self) -> bool:
        """Whether the database is currently open."""
        return self._is_open

    # ------------------------------------------------------------------
    # Low-level get / put / delete / scan
    # ------------------------------------------------------------------

    def _put(self, key: str, value: Any):
        """Serialize and store a value."""
        self._db[key] = _serialize(value)

    def _get(self, key: str) -> Optional[Any]:
        """Retrieve and deserialize a value, or return None."""
        try:
            raw = self._db[key]
            if raw is None:
                return None
            return _deserialize(raw)
        except (KeyError, TypeError):
            return None

    def _delete(self, key: str):
        """Delete a key from the database."""
        try:
            del self._db[key]
        except (KeyError, TypeError):
            pass

    def _prefix_scan(self, prefix: str) -> List[Tuple[str, Any]]:
        """Return all (key, deserialized_value) pairs matching a key prefix."""
        results = []
        for k, v in self._db.items():
            if isinstance(k, str) and k.startswith(prefix):
                results.append((k, _deserialize(v)))
            elif isinstance(k, bytes) and k.decode("utf-8").startswith(prefix):
                results.append((k.decode("utf-8"), _deserialize(v)))
        return results

    def _batch_write(self, ops: List[Tuple[str, str, Any]]):
        """
        Execute a batch of operations atomically.
        Each op is (op_type, key, value) where op_type is "put" or "delete".
        """
        wb = WriteBatch()
        for op_type, key, value in ops:
            if op_type == "put":
                wb.put(key, _serialize(value))
            elif op_type == "delete":
                wb.delete(key)
        self._db.write(wb)

    # ------------------------------------------------------------------
    # Entity ID normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_id(text: str) -> str:
        """Normalize an entity ID: lowercase, strip, spaces to underscores."""
        return re.sub(r"\s+", "_", text.strip().lower())

    # ------------------------------------------------------------------
    # Node CRUD
    # ------------------------------------------------------------------

    def add_node(self, entity_id: str, node_type: str = "entity",
                 properties: Optional[Dict] = None,
                 confidence: Optional[float] = None,
                 source: Optional[str] = None,
                 embedding=None,
                 text_summary: Optional[str] = None,
                 numeric_value: Optional[float] = None,
                 ttl: Optional[int] = None,
                 question: Optional[str] = None,
                 flags: int = 0) -> str:
        """
        Add a node to the graph.

        Args:
            entity_id: Human-readable entity identifier.
            node_type: Category of the node (e.g. "person", "language").
            properties: Arbitrary key-value properties.
            confidence: Node confidence score (0.0-1.0).
            source: Origin of this node's data.
            embedding: Optional vector embedding (list or numpy array).
            text_summary: Short text summary for display.
            numeric_value: Optional numeric value (e.g. price, score).
            ttl: Time-to-live in minutes (None = no expiry).
            question: The question/query this node answers.
            flags: NodeFlags bitfield.

        Returns:
            The normalized entity ID.
        """
        nid = self._normalize_id(entity_id)
        now = datetime.now().isoformat()
        resolved_type = resolve_node_type(node_type)
        type_id = int(resolved_type)

        existing = self._get(f"node:{nid}")
        if existing:
            # Merge properties into existing node
            if properties:
                existing.setdefault("properties", {}).update(properties)
            existing["updated_at"] = now
            # Update type_id if not present
            if "type_id" not in existing:
                existing["type_id"] = type_id
            # Update optional NGRE fields if provided
            if confidence is not None:
                existing["confidence"] = confidence
            if source is not None:
                existing["source"] = source
            if text_summary is not None:
                existing["text_summary"] = text_summary
            if numeric_value is not None:
                existing["numeric_value"] = numeric_value
            if ttl is not None:
                existing["ttl"] = ttl
            if question is not None:
                existing["question"] = question
            if flags != 0:
                existing["flags"] = existing.get("flags", 0) | flags
            self._put(f"node:{nid}", existing)
            logger.debug("Node updated: %s", nid)
        else:
            node_data = {
                "type": node_type,
                "type_id": type_id,
                "properties": properties or {},
                "created_at": now,
                "updated_at": now,
                "access_count": 0,
                "importance": 0.5,
                "flags": flags,
            }
            if confidence is not None:
                node_data["confidence"] = confidence
            if source is not None:
                node_data["source"] = source
            if text_summary is not None:
                node_data["text_summary"] = text_summary
            if numeric_value is not None:
                node_data["numeric_value"] = numeric_value
            if ttl is not None:
                node_data["ttl"] = ttl
            if question is not None:
                node_data["question"] = question
            self._put(f"node:{nid}", node_data)
            logger.debug("Node created: %s (%s, type_id=%d)", nid, node_type, type_id)

        # Update type index: idx:type:{type_id} → list of node IDs
        type_idx_key = f"idx:type:{type_id}"
        type_idx = self._get(type_idx_key) or []
        if nid not in type_idx:
            type_idx.append(nid)
            self._put(type_idx_key, type_idx)

        # Add to HNSW embedding index if embedding provided
        if embedding is not None and self._embedding_index is not None:
            self._embedding_index.add(nid, embedding)

        # Add to hot tier cache
        if self._hot_tier is not None:
            data = self._get(f"node:{nid}")
            if data:
                data["id"] = nid
                self._hot_tier.put(nid, data)

        return nid

    def get_node(self, entity_id: str, track_access: bool = True) -> Optional[Dict]:
        """Get a node by entity ID, or None if not found.

        Args:
            entity_id: Entity identifier.
            track_access: If True, increment access_count and update last_accessed.
        """
        nid = self._normalize_id(entity_id)

        # Check hot tier first
        if self._hot_tier is not None:
            cached = self._hot_tier.get(nid)
            if cached is not None:
                if track_access:
                    cached["access_count"] = cached.get("access_count", 0) + 1
                    cached["last_accessed"] = datetime.now().isoformat()
                    # Write through to RocksDB
                    store_data = {k: v for k, v in cached.items() if k != "id"}
                    self._put(f"node:{nid}", store_data)
                    self._hot_tier.put(nid, cached)
                return cached

        data = self._get(f"node:{nid}")
        if data:
            data["id"] = nid
            # Normalize old nodes missing NGRE fields
            self._normalize_node_data(data)

            if track_access:
                data["access_count"] = data.get("access_count", 0) + 1
                data["last_accessed"] = datetime.now().isoformat()
                # Write through to RocksDB
                store_data = {k: v for k, v in data.items() if k != "id"}
                self._put(f"node:{nid}", store_data)

            # Promote to hot tier if threshold met
            if (self._hot_tier is not None
                    and self._hot_tier.should_promote(data.get("access_count", 0))):
                self._hot_tier.put(nid, data)

        return data

    def update_node(self, entity_id: str, properties: Dict):
        """Merge properties into an existing node."""
        nid = self._normalize_id(entity_id)
        existing = self._get(f"node:{nid}")
        if existing is None:
            raise KeyError(f"Node not found: {nid}")
        existing.setdefault("properties", {}).update(properties)
        existing["updated_at"] = datetime.now().isoformat()
        self._put(f"node:{nid}", existing)
        logger.debug("Node properties updated: %s", nid)

    def delete_node(self, entity_id: str):
        """
        Delete a node and cascade: remove all edges referencing it
        and clean up all indexes (including type index, HNSW, hot tier).
        """
        nid = self._normalize_id(entity_id)

        # Read node data before deletion for type index cleanup
        node_data = self._get(f"node:{nid}")

        # Gather all edge keys involving this node
        edge_keys_to_delete = set()

        # Outgoing edges
        sub_idx = self._get(f"idx:sub:{nid}")
        if sub_idx:
            edge_keys_to_delete.update(sub_idx)

        # Incoming edges
        obj_idx = self._get(f"idx:obj:{nid}")
        if obj_idx:
            edge_keys_to_delete.update(obj_idx)

        # Delete each edge properly (forward, reverse, indexes)
        for edge_key in edge_keys_to_delete:
            parts = edge_key.split(":")
            if len(parts) >= 4:
                # edge:subject:predicate:object
                subj, pred, obj = parts[1], parts[2], ":".join(parts[3:])
                self._delete_edge_internal(subj, pred, obj)

        # Remove from type index
        if node_data:
            type_id = node_data.get("type_id")
            if type_id is None:
                resolved = resolve_node_type(node_data.get("type", "entity"))
                type_id = int(resolved)
            type_idx_key = f"idx:type:{type_id}"
            type_idx = self._get(type_idx_key) or []
            if nid in type_idx:
                type_idx.remove(nid)
                self._put(type_idx_key, type_idx)

        # Remove from embedding index
        if self._embedding_index is not None:
            self._embedding_index.remove(nid)

        # Remove from hot tier
        if self._hot_tier is not None:
            self._hot_tier.evict(nid)

        # Delete the node itself
        self._delete(f"node:{nid}")
        logger.debug("Node deleted (cascading): %s", nid)

    def list_nodes(self, node_type: Optional[str] = None,
                   limit: int = 0) -> List[Tuple[str, Dict]]:
        """
        List all nodes, optionally filtered by type.

        Uses idx:type:{N} index for fast lookup when filtering by type.
        Falls back to prefix scan if the index is missing.

        Args:
            node_type: Filter by node type (e.g. "snapshot", "entity").
            limit: Max nodes to return (0 = unlimited).

        Returns:
            List of (entity_id, node_data) tuples.
        """
        if node_type is not None:
            # Fast path: use type index
            resolved = resolve_node_type(node_type)
            type_id = int(resolved)
            type_idx = self._get(f"idx:type:{type_id}")
            if type_idx is not None:
                results = []
                for nid in type_idx:
                    if limit and len(results) >= limit:
                        break
                    data = self._get(f"node:{nid}")
                    if data:
                        data["id"] = nid
                        results.append((nid, data))
                return results
            # Fall back to prefix scan (old data without type index)
            results = []
            for key, data in self._prefix_scan("node:"):
                if limit and len(results) >= limit:
                    break
                nid = key[5:]  # strip "node:"
                if data.get("type") == node_type:
                    data["id"] = nid
                    results.append((nid, data))
                elif data.get("type_id") == type_id:
                    data["id"] = nid
                    results.append((nid, data))
            return results

        # No filter — return all nodes
        results = []
        for key, data in self._prefix_scan("node:"):
            if limit and len(results) >= limit:
                break
            nid = key[5:]  # strip "node:"
            data["id"] = nid
            results.append((nid, data))
        return results

    # ------------------------------------------------------------------
    # Edge CRUD
    # ------------------------------------------------------------------

    def _edge_key(self, subject: str, predicate: str, obj: str) -> str:
        """Build forward edge key."""
        return f"edge:{subject}:{predicate}:{obj}"

    def _reverse_key(self, subject: str, predicate: str, obj: str) -> str:
        """Build reverse edge key."""
        return f"reverse:{obj}:{predicate}:{subject}"

    def add_edge(self, subject: str, predicate: str, obj: str,
                 confidence: float = 1.0, source: str = "manual",
                 weight: float = 1.0):
        """
        Add a directed edge between two entities.

        Creates forward edge, reverse edge, and updates all indexes
        atomically via batch write.

        Args:
            subject: Source entity ID.
            predicate: Relationship name.
            obj: Target entity ID.
            confidence: Confidence score 0.0-1.0.
            source: Origin of this knowledge.
            weight: Edge weight for traversal scoring.
        """
        s = self._normalize_id(subject)
        p = self._normalize_id(predicate)
        o = self._normalize_id(obj)
        now = datetime.now().isoformat()

        # Auto-create nodes if they don't exist
        if self._get(f"node:{s}") is None:
            self.add_node(s)
        if self._get(f"node:{o}") is None:
            self.add_node(o)

        # SHA-256 hash for edge integrity
        from .crypto import QORCrypto
        data_hash = QORCrypto.hash_sha256(f"{s}:{p}:{o}:{confidence}")

        edge_data = {
            "subject": s,
            "predicate": p,
            "object": o,
            "confidence": confidence,
            "timestamp": now,
            "source": source,
            "weight": weight,
            "data_hash": data_hash,
        }

        fwd_key = self._edge_key(s, p, o)
        rev_key = self._reverse_key(s, p, o)

        # Build batch operations
        ops = [
            ("put", fwd_key, edge_data),
            ("put", rev_key, {"forward_key": fwd_key}),
        ]

        # Update subject index (outgoing edges from s)
        sub_idx = self._get(f"idx:sub:{s}") or []
        if fwd_key not in sub_idx:
            sub_idx.append(fwd_key)
        ops.append(("put", f"idx:sub:{s}", sub_idx))

        # Update object index (incoming edges to o)
        obj_idx = self._get(f"idx:obj:{o}") or []
        if fwd_key not in obj_idx:
            obj_idx.append(fwd_key)
        ops.append(("put", f"idx:obj:{o}", obj_idx))

        # Update predicate index
        pred_idx = self._get(f"idx:pred:{p}") or []
        if fwd_key not in pred_idx:
            pred_idx.append(fwd_key)
        ops.append(("put", f"idx:pred:{p}", pred_idx))

        self._batch_write(ops)
        logger.debug("Edge added: %s -[%s]-> %s (conf=%.2f)", s, p, o, confidence)

        # Enforce edge budget for both endpoints (only when count exceeds 200)
        sub_count = len(self._get(f"idx:sub:{s}") or []) + len(self._get(f"idx:obj:{s}") or [])
        if sub_count > 200:
            self.enforce_edge_budget(s)
        obj_count = len(self._get(f"idx:sub:{o}") or []) + len(self._get(f"idx:obj:{o}") or [])
        if obj_count > 200:
            self.enforce_edge_budget(o)

    def get_edge(self, subject: str, predicate: str, obj: str) -> Optional[Dict]:
        """Get a specific edge by subject, predicate, object."""
        s = self._normalize_id(subject)
        p = self._normalize_id(predicate)
        o = self._normalize_id(obj)
        return self._get(self._edge_key(s, p, o))

    def get_edges(self, entity_id: str, direction: str = "out") -> List[Dict]:
        """
        Get all edges for an entity.

        Args:
            entity_id: The entity to query.
            direction: "out" for outgoing, "in" for incoming, "both" for all.

        Returns:
            List of edge data dicts.
        """
        nid = self._normalize_id(entity_id)
        edges = []

        if direction in ("out", "both"):
            sub_idx = self._get(f"idx:sub:{nid}") or []
            for edge_key in sub_idx:
                edge_data = self._get(edge_key)
                if edge_data:
                    edges.append(edge_data)

        if direction in ("in", "both"):
            obj_idx = self._get(f"idx:obj:{nid}") or []
            for edge_key in obj_idx:
                edge_data = self._get(edge_key)
                if edge_data:
                    edges.append(edge_data)

        return edges

    def delete_edge(self, subject: str, predicate: str, obj: str):
        """Delete an edge and clean up its reverse entry and all indexes."""
        s = self._normalize_id(subject)
        p = self._normalize_id(predicate)
        o = self._normalize_id(obj)
        self._delete_edge_internal(s, p, o)

    def _delete_edge_internal(self, s: str, p: str, o: str):
        """Internal edge deletion on already-normalized IDs."""
        fwd_key = self._edge_key(s, p, o)
        rev_key = self._reverse_key(s, p, o)

        ops = [
            ("delete", fwd_key, None),
            ("delete", rev_key, None),
        ]

        # Clean subject index
        sub_idx = self._get(f"idx:sub:{s}") or []
        if fwd_key in sub_idx:
            sub_idx.remove(fwd_key)
            ops.append(("put", f"idx:sub:{s}", sub_idx))

        # Clean object index
        obj_idx = self._get(f"idx:obj:{o}") or []
        if fwd_key in obj_idx:
            obj_idx.remove(fwd_key)
            ops.append(("put", f"idx:obj:{o}", obj_idx))

        # Clean predicate index
        pred_idx = self._get(f"idx:pred:{p}") or []
        if fwd_key in pred_idx:
            pred_idx.remove(fwd_key)
            ops.append(("put", f"idx:pred:{p}", pred_idx))

        self._batch_write(ops)
        logger.debug("Edge deleted: %s -[%s]-> %s", s, p, o)

    def update_edge_confidence(self, subject: str, predicate: str, obj: str,
                               new_confidence: float):
        """Update the confidence score of an existing edge."""
        s = self._normalize_id(subject)
        p = self._normalize_id(predicate)
        o = self._normalize_id(obj)
        fwd_key = self._edge_key(s, p, o)

        edge_data = self._get(fwd_key)
        if edge_data is None:
            raise KeyError(f"Edge not found: {s} -[{p}]-> {o}")

        edge_data["confidence"] = max(0.0, min(1.0, new_confidence))
        edge_data["timestamp"] = datetime.now().isoformat()
        self._put(fwd_key, edge_data)
        logger.debug("Edge confidence updated: %s -[%s]-> %s = %.2f",
                      s, p, o, new_confidence)

    def enforce_edge_budget(self, entity_id: str, max_edges: int = 200) -> int:
        """Enforce maximum edge count per entity with type-balanced pruning.

        If the total number of edges (in + out) for the entity exceeds max_edges,
        prunes the lowest-confidence edges until at max_edges. Tries to keep at
        least 1 edge per predicate type (type-balanced pruning).

        Args:
            entity_id: The entity to enforce the budget on.
            max_edges: Maximum number of edges allowed (default 200).

        Returns:
            Number of pruned edges.
        """
        nid = self._normalize_id(entity_id)

        # Collect all edges (in + out) with their data and keys
        all_edges = []  # list of (edge_key, edge_data, direction)

        sub_idx = self._get(f"idx:sub:{nid}") or []
        for edge_key in sub_idx:
            edge_data = self._get(edge_key)
            if edge_data:
                all_edges.append((edge_key, edge_data, "out"))

        obj_idx = self._get(f"idx:obj:{nid}") or []
        for edge_key in obj_idx:
            edge_data = self._get(edge_key)
            if edge_data:
                all_edges.append((edge_key, edge_data, "in"))

        total = len(all_edges)
        if total <= max_edges:
            return 0

        excess = total - max_edges

        # Group edges by predicate type
        by_predicate: Dict[str, List[Tuple[str, Dict, str]]] = {}
        for edge_key, edge_data, direction in all_edges:
            pred = edge_data.get("predicate", "unknown")
            by_predicate.setdefault(pred, []).append((edge_key, edge_data, direction))

        # Build set of protected edges: the highest-confidence edge per predicate
        protected_keys = set()
        for pred, edges in by_predicate.items():
            best = max(edges, key=lambda e: e[1].get("confidence", 0.0))
            protected_keys.add(best[0])

        # Sort all edges by confidence ascending (lowest first) for pruning
        all_edges.sort(key=lambda e: e[1].get("confidence", 0.0))

        pruned = 0
        for edge_key, edge_data, direction in all_edges:
            if pruned >= excess:
                break
            # Skip protected edges (keep at least 1 per predicate type)
            if edge_key in protected_keys:
                continue
            # Delete this edge
            s = edge_data.get("subject", "")
            p = edge_data.get("predicate", "")
            o = edge_data.get("object", "")
            if s and p and o:
                self._delete_edge_internal(s, p, o)
                pruned += 1

        if pruned > 0:
            logger.info("Edge budget enforced for '%s': pruned %d edges (was %d, max %d)",
                        nid, pruned, total, max_edges)
        return pruned

    # ------------------------------------------------------------------
    # Synonym Management
    # ------------------------------------------------------------------

    def add_synonym(self, canonical: str, alias: str):
        """
        Register a synonym for a predicate.

        Args:
            canonical: The canonical predicate name.
            alias: An alternative name that maps to it.
        """
        canonical = self._normalize_id(canonical)
        alias_norm = self._normalize_id(alias)

        syn_data = self._get(f"syn:{canonical}")
        if syn_data is None:
            syn_data = {"canonical": canonical, "aliases": []}

        if alias_norm not in syn_data["aliases"]:
            syn_data["aliases"].append(alias_norm)
            self._put(f"syn:{canonical}", syn_data)
            logger.debug("Synonym added: '%s' -> '%s'", alias_norm, canonical)

    def resolve_predicate(self, text: str) -> str:
        """
        Resolve a predicate to its canonical form using synonyms.

        Args:
            text: Predicate text (may be an alias).

        Returns:
            Canonical predicate name, or the normalized input if no synonym found.
        """
        norm = self._normalize_id(text)

        # Check if it's already a canonical form
        if self._get(f"syn:{norm}") is not None:
            return norm

        # Search through all synonym entries
        for key, data in self._prefix_scan("syn:"):
            if norm in data.get("aliases", []):
                return data["canonical"]

        return norm

    def get_synonyms(self, predicate: str) -> List[str]:
        """Get all aliases for a predicate."""
        canonical = self._normalize_id(predicate)
        syn_data = self._get(f"syn:{canonical}")
        if syn_data:
            return syn_data.get("aliases", [])
        return []

    def load_default_synonyms(self):
        """Load built-in synonym sets into the graph."""
        for canonical, data in DEFAULT_SYNONYMS.items():
            syn_entry = {
                "canonical": self._normalize_id(canonical),
                "aliases": [self._normalize_id(a) for a in data["aliases"]],
            }
            self._put(f"syn:{self._normalize_id(canonical)}", syn_entry)
        logger.info("Loaded %d default synonym sets", len(DEFAULT_SYNONYMS))

    # ------------------------------------------------------------------
    # Graph Traversal
    # ------------------------------------------------------------------

    def traverse(self, start_entity: str, max_depth: int = 3,
                 direction: str = "out") -> Dict:
        """
        Breadth-first traversal from a starting entity.

        Args:
            start_entity: Entity to start from.
            max_depth: Maximum hop distance.
            direction: "out", "in", or "both".

        Returns:
            Dict with "nodes" (visited node data) and "edges" (traversed edges).
        """
        start = self._normalize_id(start_entity)
        visited_nodes = {}
        collected_edges = []
        queue = deque([(start, 0)])
        visited = {start}

        while queue:
            current, depth = queue.popleft()

            # Fetch node data
            node_data = self._get(f"node:{current}")
            if node_data:
                node_data["id"] = current
                visited_nodes[current] = node_data

            if depth >= max_depth:
                continue

            # Get edges in the requested direction
            edges = self.get_edges(current, direction=direction)
            for edge in edges:
                collected_edges.append(edge)

                # Determine the neighbor
                if direction == "in":
                    neighbor = edge.get("subject", "")
                else:
                    neighbor = edge.get("object", "")

                if direction == "both":
                    # For both, check which side is not the current node
                    if edge.get("subject") == current:
                        neighbor = edge.get("object", "")
                    else:
                        neighbor = edge.get("subject", "")

                if neighbor and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))

        return {
            "start": start,
            "nodes": visited_nodes,
            "edges": collected_edges,
            "depth": max_depth,
        }

    def find_path(self, source: str, target: str,
                  max_depth: int = 5) -> Optional[List[Dict]]:
        """
        Find a path between two entities using BFS.

        Args:
            source: Starting entity.
            target: Target entity.
            max_depth: Maximum search depth.

        Returns:
            List of edge dicts forming the path, or None if no path found.
        """
        src = self._normalize_id(source)
        tgt = self._normalize_id(target)

        if src == tgt:
            return []

        # BFS with parent tracking
        queue = deque([(src, [])])
        visited = {src}

        while queue:
            current, path = queue.popleft()

            if len(path) >= max_depth:
                continue

            edges = self.get_edges(current, direction="both")
            for edge in edges:
                # Determine neighbor
                if edge.get("subject") == current:
                    neighbor = edge.get("object", "")
                else:
                    neighbor = edge.get("subject", "")

                if not neighbor or neighbor in visited:
                    continue

                new_path = path + [edge]

                if neighbor == tgt:
                    return new_path

                visited.add(neighbor)
                queue.append((neighbor, new_path))

        return None

    def query_pattern(self, subject: Optional[str] = None,
                      predicate: Optional[str] = None,
                      obj: Optional[str] = None) -> List[Dict]:
        """
        Query edges matching a pattern (any field can be None for wildcard).

        Examples:
            query_pattern(subject="guido")              -> all edges from guido
            query_pattern(predicate="created")           -> all "created" edges
            query_pattern(subject="guido", obj="python") -> guido->python edges
        """
        results = []

        # Choose the most selective index to scan
        if subject is not None:
            s = self._normalize_id(subject)
            sub_idx = self._get(f"idx:sub:{s}") or []
            for edge_key in sub_idx:
                edge_data = self._get(edge_key)
                if edge_data and self._matches_pattern(edge_data, subject, predicate, obj):
                    results.append(edge_data)

        elif obj is not None:
            o = self._normalize_id(obj)
            obj_idx = self._get(f"idx:obj:{o}") or []
            for edge_key in obj_idx:
                edge_data = self._get(edge_key)
                if edge_data and self._matches_pattern(edge_data, subject, predicate, obj):
                    results.append(edge_data)

        elif predicate is not None:
            p = self._normalize_id(predicate)
            # Also resolve synonym
            canonical = self.resolve_predicate(p)
            pred_idx = self._get(f"idx:pred:{canonical}") or []
            for edge_key in pred_idx:
                edge_data = self._get(edge_key)
                if edge_data:
                    results.append(edge_data)
            # If canonical differs from p, also check original
            if canonical != p:
                pred_idx2 = self._get(f"idx:pred:{p}") or []
                for edge_key in pred_idx2:
                    edge_data = self._get(edge_key)
                    if edge_data and edge_data not in results:
                        results.append(edge_data)
        else:
            # Full scan — no filter specified
            for key, data in self._prefix_scan("edge:"):
                results.append(data)

        return results

    def _matches_pattern(self, edge_data: Dict, subject: Optional[str],
                         predicate: Optional[str], obj: Optional[str]) -> bool:
        """Check if an edge matches a query pattern."""
        if subject is not None:
            if edge_data.get("subject") != self._normalize_id(subject):
                return False
        if predicate is not None:
            p = self._normalize_id(predicate)
            canonical = self.resolve_predicate(p)
            edge_pred = edge_data.get("predicate", "")
            if edge_pred != p and edge_pred != canonical:
                return False
        if obj is not None:
            if edge_data.get("object") != self._normalize_id(obj):
                return False
        return True

    # ------------------------------------------------------------------
    # NGRE Methods — Node normalization, importance, Hebbian, embeddings
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_node_data(data: Dict):
        """Fill missing NGRE fields with defaults for backward compatibility.

        Called on nodes read from RocksDB that were created before NGRE.
        Mutates data in-place.
        """
        data.setdefault("type_id", int(resolve_node_type(data.get("type", "entity"))))
        data.setdefault("access_count", 0)
        data.setdefault("importance", 0.5)
        data.setdefault("flags", 0)
        data.setdefault("last_accessed", "")
        data.setdefault("confidence", 1.0)
        data.setdefault("source", "")

    def compute_importance(self, entity_id: str) -> float:
        """Compute importance score for a node using 5 signals.

        Signals (weighted):
            - access_freq (0.25): normalized access count
            - recency (0.25): exponential decay from last access
            - connectivity (0.15): number of edges (in+out)
            - edge_strength (0.15): average confidence of connected edges
            - type_bonus (0.20): per-NodeType bonus from _TYPE_BONUS

        Flag overrides:
            - LANDMARK → 1.0
            - ANCHORED → min 0.8
            - QUARANTINED → 0.0

        Returns:
            Importance score in [0.0, 1.0].
        """
        nid = self._normalize_id(entity_id)
        data = self._get(f"node:{nid}")
        if data is None:
            return 0.0

        self._normalize_node_data(data)
        flags = data.get("flags", 0)

        # Flag overrides
        if NodeFlags.has(flags, NodeFlags.QUARANTINED):
            return 0.0
        if NodeFlags.has(flags, NodeFlags.LANDMARK):
            return 1.0

        # 1. Access frequency (0.25) — log-scaled, max at ~100 accesses
        access_count = data.get("access_count", 0)
        access_score = min(1.0, math.log1p(access_count) / math.log1p(100))

        # 2. Recency (0.25) — exponential decay from last access
        recency_score = 0.5  # default if no last_accessed
        last_accessed = data.get("last_accessed", "")
        if last_accessed:
            try:
                la_dt = datetime.fromisoformat(last_accessed)
                age_hours = (datetime.now() - la_dt).total_seconds() / 3600.0
                decay = self.config.importance_recency_decay
                recency_score = math.exp(-decay * age_hours)
            except (ValueError, TypeError):
                pass

        # 3. Connectivity (0.15) — number of edges, capped at 50
        out_edges = self._get(f"idx:sub:{nid}") or []
        in_edges = self._get(f"idx:obj:{nid}") or []
        total_edges = len(out_edges) + len(in_edges)
        connectivity_score = min(1.0, total_edges / 50.0)

        # 4. Edge strength (0.15) — average confidence of connected edges
        edge_confs = []
        for edge_key in (out_edges + in_edges)[:20]:  # sample up to 20
            edge_data = self._get(edge_key)
            if edge_data:
                edge_confs.append(edge_data.get("confidence", 0.5))
        edge_strength = sum(edge_confs) / len(edge_confs) if edge_confs else 0.5

        # 5. Type bonus (0.20)
        type_id = data.get("type_id", 0)
        try:
            node_type = NodeType(type_id)
        except ValueError:
            node_type = NodeType.ENTITY
        type_bonus = _TYPE_BONUS.get(node_type, 0.0)

        # Weighted combination
        importance = (
            0.25 * access_score
            + 0.25 * recency_score
            + 0.15 * connectivity_score
            + 0.15 * edge_strength
            + 0.20 * type_bonus
        )
        importance = max(0.0, min(1.0, importance))

        # ANCHORED floor
        if NodeFlags.has(flags, NodeFlags.ANCHORED):
            importance = max(0.8, importance)

        # Store updated importance
        data["importance"] = importance
        store_data = {k: v for k, v in data.items() if k != "id"}
        self._put(f"node:{nid}", store_data)

        return importance

    def batch_recompute_importance(self, limit: int = 1000) -> int:
        """Recompute importance scores for up to `limit` nodes.

        Called during cleanup to keep importance values fresh.

        Returns:
            Number of nodes recomputed.
        """
        count = 0
        for key, data in self._prefix_scan("node:"):
            if count >= limit:
                break
            nid = key[5:]  # strip "node:"
            self.compute_importance(nid)
            count += 1
        logger.info("Recomputed importance for %d nodes", count)
        return count

    def hebbian_update(self, source_id: str, target_id: str,
                       reward: float, lr: Optional[float] = None):
        """Hebbian edge weight update: w += lr * reward * importance[i] * importance[j].

        Strengthens connections between co-activated important nodes.
        Positive reward = reinforce, negative = weaken.

        Args:
            source_id: Source node entity ID.
            target_id: Target node entity ID.
            reward: Reward signal (-1.0 to 1.0).
            lr: Learning rate (defaults to config.hebbian_lr).
        """
        if lr is None:
            lr = self.config.hebbian_lr

        src = self._normalize_id(source_id)
        tgt = self._normalize_id(target_id)

        # Get importance scores
        src_data = self._get(f"node:{src}")
        tgt_data = self._get(f"node:{tgt}")
        if src_data is None or tgt_data is None:
            return

        self._normalize_node_data(src_data)
        self._normalize_node_data(tgt_data)
        imp_i = src_data.get("importance", 0.5)
        imp_j = tgt_data.get("importance", 0.5)

        # Find and update edges between source and target
        sub_idx = self._get(f"idx:sub:{src}") or []
        for edge_key in sub_idx:
            edge_data = self._get(edge_key)
            if edge_data and edge_data.get("object") == tgt:
                old_w = edge_data.get("weight", 1.0)
                delta = lr * reward * imp_i * imp_j
                new_w = max(0.0, min(2.0, old_w + delta))
                edge_data["weight"] = new_w
                edge_data["timestamp"] = datetime.now().isoformat()
                self._put(edge_key, edge_data)
                logger.debug("Hebbian update: %s→%s weight %.4f→%.4f (Δ=%.6f)",
                             src, tgt, old_w, new_w, delta)

    def search_by_embedding(self, embedding, k: int = 10,
                            node_type: Optional[str] = None) -> List[Tuple[str, float]]:
        """Search for nodes by embedding similarity (ANN via HNSW index).

        Args:
            embedding: Query vector (list or numpy array).
            k: Number of nearest neighbors to return.
            node_type: Optional filter — only return nodes of this type.

        Returns:
            List of (entity_id, distance) tuples, sorted by distance.
        """
        if self._embedding_index is None:
            return []

        filter_ids = None
        if node_type is not None:
            resolved = resolve_node_type(node_type)
            type_id = int(resolved)
            type_idx = self._get(f"idx:type:{type_id}")
            if type_idx is not None:
                filter_ids = set(type_idx)

        return self._embedding_index.search(embedding, k=k, filter_ids=filter_ids)

    def update_embedding_ema(self, entity_id: str, new_embedding,
                             alpha: float = 0.05) -> bool:
        """Update a node's embedding using Exponential Moving Average (EMA).

        Computes: new_vec = alpha * new_embedding + (1 - alpha) * old_embedding

        This allows embeddings to drift gradually as the entity's context evolves,
        rather than replacing the entire vector on each update.

        Args:
            entity_id: The entity whose embedding to update.
            new_embedding: The new embedding vector (list or numpy array).
            alpha: EMA blending factor. 0.05 for normal updates, 0.2 for major events.
                   Higher alpha = more weight on the new embedding.

        Returns:
            True if the update was applied, False if entity not found or no index.
        """
        if self._embedding_index is None:
            return False

        nid = self._normalize_id(entity_id)
        idx = self._embedding_index

        # Get the existing embedding
        old_vec = None
        if idx._np is not None and idx._np_vectors is not None and nid in idx._np_ids:
            pos = idx._np_ids.index(nid)
            old_vec = idx._np_vectors[pos].copy()
        elif idx._use_hnsw and idx._index is not None:
            label = idx._id_to_label.get(nid)
            if label is not None:
                try:
                    raw = idx._index.get_items([label])[0]
                    if idx._np is not None:
                        old_vec = idx._np.asarray(raw, dtype='float32')
                except Exception:
                    pass

        if old_vec is None:
            # No existing embedding — treat as a fresh add
            return idx.add(nid, new_embedding)

        # Compute EMA
        if idx._np is not None:
            new_arr = idx._np.asarray(new_embedding, dtype='float32').ravel()
            ema_vec = alpha * new_arr + (1.0 - alpha) * old_vec
        else:
            # Pure Python fallback (no numpy)
            if hasattr(new_embedding, '__iter__'):
                new_list = list(new_embedding)
            else:
                return False
            ema_vec = [
                alpha * n + (1.0 - alpha) * o
                for n, o in zip(new_list, old_vec)
            ]

        # Update in numpy mirror
        if idx._np is not None and idx._np_vectors is not None and nid in idx._np_ids:
            pos = idx._np_ids.index(nid)
            if idx._np is not None:
                ema_vec = idx._np.asarray(ema_vec, dtype='float32')
            idx._np_vectors[pos] = ema_vec

        # For HNSW: remove old and re-add (HNSW doesn't support in-place update)
        if idx._use_hnsw and idx._index is not None:
            old_label = idx._id_to_label.pop(nid, None)
            if old_label is not None:
                try:
                    idx._index.mark_deleted(old_label)
                except Exception:
                    pass
                idx._label_to_id.pop(old_label, None)
            new_label = idx._next_label
            idx._next_label += 1
            if idx._np is not None:
                ema_arr = idx._np.asarray(ema_vec, dtype='float32').reshape(1, -1)
            else:
                ema_arr = ema_vec
            try:
                idx._index.add_items(ema_arr, [new_label])
                idx._id_to_label[nid] = new_label
                idx._label_to_id[new_label] = nid
            except Exception:
                return False

        logger.debug("Embedding EMA updated for '%s' (alpha=%.3f)", nid, alpha)
        return True

    def get_search_tensors(self, max_nodes: int = 0) -> Dict:
        """Build tensor views of the graph for interference search.

        Returns dict suitable for InterferenceSearch.search():
            - node_ids:    list of entity_id strings (defines ordering)
            - embeddings:  (N, dim) float32 tensor
            - adj_indices: (2, E) long tensor (sparse COO format)
            - adj_weights: (E,) float32 tensor
            - confidence:  (N,) float32 tensor

        Priority: hot tier nodes first, then any remaining embedded nodes.
        If max_nodes > 0, limits to that many nodes.
        """
        try:
            import torch
        except ImportError:
            return {"node_ids": [], "embeddings": None,
                    "adj_indices": None, "adj_weights": None,
                    "confidence": None}

        if self._embedding_index is None or not self._is_open:
            return {"node_ids": [], "embeddings": torch.tensor([]),
                    "adj_indices": torch.zeros(2, 0, dtype=torch.long),
                    "adj_weights": torch.tensor([]),
                    "confidence": torch.tensor([])}

        # Collect node IDs with embeddings
        node_ids = []
        if self._embedding_index._use_hnsw and self._embedding_index._index:
            node_ids = list(self._embedding_index._id_to_label.keys())
        elif self._embedding_index._np_ids:
            node_ids = list(self._embedding_index._np_ids)

        if not node_ids:
            return {"node_ids": [], "embeddings": torch.tensor([]),
                    "adj_indices": torch.zeros(2, 0, dtype=torch.long),
                    "adj_weights": torch.tensor([]),
                    "confidence": torch.tensor([])}

        if max_nodes > 0 and len(node_ids) > max_nodes:
            node_ids = node_ids[:max_nodes]

        N = len(node_ids)
        id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

        # Build embeddings tensor
        emb_list = []
        for nid in node_ids:
            vec = self._get_node_embedding(nid)
            if vec is not None:
                emb_list.append(vec)
            else:
                emb_list.append([0.0] * self.config.embedding_dim)
        embeddings = torch.tensor(emb_list, dtype=torch.float32)

        # Build sparse adjacency from edges between these nodes
        src_list, dst_list, weight_list = [], [], []
        for nid in node_ids:
            idx_i = id_to_idx[nid]
            sub_idx = self._get(f"idx:sub:{nid}") or []
            for edge_key in sub_idx:
                edge_data = self._get(edge_key)
                if edge_data:
                    target = edge_data.get("object", "")
                    if target in id_to_idx:
                        idx_j = id_to_idx[target]
                        w = float(edge_data.get("weight", 1.0))
                        src_list.append(idx_i)
                        dst_list.append(idx_j)
                        weight_list.append(w)

        if src_list:
            adj_indices = torch.tensor([src_list, dst_list], dtype=torch.long)
            adj_weights = torch.tensor(weight_list, dtype=torch.float32)
        else:
            adj_indices = torch.zeros(2, 0, dtype=torch.long)
            adj_weights = torch.tensor([], dtype=torch.float32)

        # Build confidence vector
        conf_list = []
        for nid in node_ids:
            data = self._get(f"node:{nid}")
            if data:
                props = data.get("properties", {})
                c = float(props.get("confidence", 0.5))
                # QUARANTINED nodes get zero confidence in search
                flags = int(props.get("flags", 0))
                if flags & 0x04:  # NodeFlags.QUARANTINED
                    c = 0.0
                conf_list.append(c)
            else:
                conf_list.append(0.5)
        confidence = torch.tensor(conf_list, dtype=torch.float32)

        return {
            "node_ids": node_ids,
            "embeddings": embeddings,
            "adj_indices": adj_indices,
            "adj_weights": adj_weights,
            "confidence": confidence,
        }

    def _get_node_embedding(self, entity_id: str):
        """Retrieve raw embedding vector for a node (from HNSW or numpy)."""
        if self._embedding_index is None:
            return None
        idx = self._embedding_index
        if idx._use_hnsw and idx._index:
            label = idx._id_to_label.get(entity_id)
            if label is not None:
                try:
                    vec = idx._index.get_items([label])[0]
                    return list(vec)
                except Exception:
                    return None
        elif idx._np is not None and idx._np_ids:
            if entity_id in idx._np_ids:
                i = idx._np_ids.index(entity_id)
                if idx._np_vectors is not None and i < len(idx._np_vectors):
                    return idx._np_vectors[i].tolist()
        return None

    def _is_complex_question(self, question: str) -> bool:
        """
        Confidence gate: is this a simple or complex question?

        Complex signals: multiple hops, constraints, comparisons, aggregations.
        Simple: single entity lookup, one-hop relation.
        """
        q = question.lower()
        # Multi-hop signals
        multi_hop = any(s in q for s in ["of the", "of a", "'s", "that", "which", "whose"])
        # Constraint signals
        constrained = any(s in q for s in [
            "before", "after", "more than", "less than", "greater than",
            "above", "below", "between", "where", "when",
        ])
        # Aggregation signals
        aggregation = any(s in q for s in ["how many", "how much", "count", "total", "average"])
        # Comparison signals
        comparison = any(s in q for s in ["compare", "versus", "vs", "difference between"])
        # Chain signals
        chain = any(s in q for s in ["trace", "track", "chain", "path", "follow"])

        return multi_hop or constrained or aggregation or comparison or chain

    def semantic_query(self, question: str) -> Dict:
        """
        Answer a natural language question using the knowledge graph.

        Routes complex questions through the Hereditary Question Decomposer
        for multi-hop traversal. Simple questions use direct lookup.

        Args:
            question: Natural language question.

        Returns:
            Dict with "answer", "confidence", "path", "edges".
        """
        # Confidence gate: route complex questions through hereditary decomposer
        if self._is_complex_question(question):
            try:
                from .hereditary import hereditary_query
                result = hereditary_query(self, question)
                # Normalize output format for compatibility
                return {
                    "answer": result.get("answer", "No answer found."),
                    "confidence": result.get("confidence", 0.0),
                    "path": result.get("results", []),
                    "entities_found": result.get("entities_found", []),
                    "predicate_found": result.get("relations_found", [None])[0] if result.get("relations_found") else None,
                    "edge_count": result.get("edge_count", 0),
                    "template": result.get("template", None),
                    "hops_completed": result.get("hops_completed", 0),
                }
            except Exception:
                pass  # Fall through to simple query

        # Simple query: direct entity/predicate lookup
        question_lower = question.lower().strip()
        words = re.findall(r"[a-z_]+", question_lower)

        # Collect all known entity IDs for matching
        all_nodes = self._prefix_scan("node:")
        entity_ids = {key[5:] for key, _ in all_nodes}  # strip "node:"

        # Find entities mentioned in the question
        found_entities = []
        for nid in entity_ids:
            # Match multi-word entity IDs (underscores -> spaces)
            nid_words = nid.replace("_", " ")
            if nid in question_lower or nid_words in question_lower:
                found_entities.append(nid)

        # Find predicates mentioned in the question
        found_predicate = None
        for word in words:
            resolved = self.resolve_predicate(word)
            if resolved != word or self._get(f"syn:{word}") is not None:
                found_predicate = resolved
                break

        # Also check multi-word phrases
        if found_predicate is None:
            for syn_key, syn_data in self._prefix_scan("syn:"):
                for alias in syn_data.get("aliases", []):
                    alias_spaced = alias.replace("_", " ")
                    if alias_spaced in question_lower or alias in question_lower:
                        found_predicate = syn_data["canonical"]
                        break
                if found_predicate:
                    break

        # Query the graph
        matching_edges = []
        if found_entities and found_predicate:
            for eid in found_entities:
                matching_edges.extend(
                    self.query_pattern(subject=eid, predicate=found_predicate)
                )
                matching_edges.extend(
                    self.query_pattern(obj=eid, predicate=found_predicate)
                )
        elif found_entities:
            for eid in found_entities:
                matching_edges.extend(self.get_edges(eid, direction="both"))
        elif found_predicate:
            matching_edges.extend(self.query_pattern(predicate=found_predicate))

        # Deduplicate edges
        seen = set()
        unique_edges = []
        for e in matching_edges:
            key = (e.get("subject"), e.get("predicate"), e.get("object"))
            if key not in seen:
                seen.add(key)
                unique_edges.append(e)

        # Build answer
        if unique_edges:
            # Sort by confidence
            unique_edges.sort(key=lambda e: e.get("confidence", 0), reverse=True)
            best = unique_edges[0]
            avg_conf = sum(e.get("confidence", 0) for e in unique_edges) / len(unique_edges)

            answer_parts = []
            for e in unique_edges:
                answer_parts.append(
                    f"{e['subject']} -[{e['predicate']}]-> {e['object']} "
                    f"(confidence: {e.get('confidence', 0):.2f})"
                )

            return {
                "answer": "; ".join(answer_parts),
                "confidence": avg_conf,
                "path": unique_edges,
                "entities_found": found_entities,
                "predicate_found": found_predicate,
                "edge_count": len(unique_edges),
            }

        return {
            "answer": "No relevant knowledge found in the graph.",
            "confidence": 0.0,
            "path": [],
            "entities_found": found_entities,
            "predicate_found": found_predicate,
            "edge_count": 0,
        }

    # ------------------------------------------------------------------
    # Confidence Decay
    # ------------------------------------------------------------------

    def get_effective_confidence(self, edge_data: Dict,
                                 decay_rate: float = 0.01) -> float:
        """
        Get confidence adjusted for age-based decay.

        Confidence decays exponentially: c_eff = c * exp(-decay_rate * age_hours)

        Args:
            edge_data: Edge data dict with "confidence" and "timestamp".
            decay_rate: Decay rate per hour (default 0.01).

        Returns:
            Effective confidence (0.0-1.0).
        """
        base_confidence = edge_data.get("confidence", 1.0)
        timestamp_str = edge_data.get("timestamp", "")

        if not timestamp_str:
            return base_confidence

        try:
            created = datetime.fromisoformat(timestamp_str)
            age_hours = (datetime.now() - created).total_seconds() / 3600.0
            import math
            effective = base_confidence * math.exp(-decay_rate * age_hours)
            return max(0.0, min(1.0, effective))
        except (ValueError, TypeError):
            return base_confidence

    def decay_all(self, category: Optional[str] = None,
                  decay_rate: float = 0.01):
        """
        Apply time-based confidence decay to all edges.

        Args:
            category: If set, only decay edges from this source category.
            decay_rate: Decay rate per hour.
        """
        count = 0
        for key, data in self._prefix_scan("edge:"):
            if category and data.get("source") != category:
                continue

            effective = self.get_effective_confidence(data, decay_rate)
            if effective != data.get("confidence"):
                data["confidence"] = effective
                data["timestamp"] = datetime.now().isoformat()
                self._put(key, data)
                count += 1

        logger.info("Decayed %d edges (rate=%.4f)", count, decay_rate)

    def gc_stale_edges(self, min_confidence: float = 0.1) -> int:
        """
        Garbage-collect edges below a confidence threshold.

        Args:
            min_confidence: Edges below this confidence are removed.

        Returns:
            Number of edges removed.
        """
        removed = 0
        edges_to_delete = []

        for key, data in self._prefix_scan("edge:"):
            effective = self.get_effective_confidence(data)
            if effective < min_confidence:
                edges_to_delete.append(
                    (data.get("subject"), data.get("predicate"), data.get("object"))
                )

        for s, p, o in edges_to_delete:
            if s and p and o:
                self._delete_edge_internal(s, p, o)
                removed += 1

        logger.info("GC removed %d stale edges (threshold=%.2f)",
                     removed, min_confidence)
        return removed

    def compact(self):
        """Trigger RocksDB compaction to reclaim disk space."""
        if hasattr(self._db, 'compact_range'):
            try:
                self._db.compact_range(b"", b"\xff\xff\xff\xff")
            except TypeError:
                # Fallback if signature differs
                try:
                    self._db.compact_range()
                except Exception:
                    pass
            logger.info("RocksDB compaction completed")

    def per_step_maintenance(self, accessed_nodes: List[str] = None):
        """Lightweight maintenance to run per inference step.

        - Bump access_count for accessed nodes
        - Micro-decay confidence on a random sample of edges (0.0001 per step)
        - Much lighter than full cleanup -- designed for per-query execution

        Args:
            accessed_nodes: List of node IDs that were accessed in this step.
        """
        if not self._is_open:
            return

        # Bump access counts for nodes used in this step
        if accessed_nodes:
            now_iso = datetime.now().isoformat()
            for nid in accessed_nodes:
                nid_norm = self._normalize_id(nid)
                data = self._get(f"node:{nid_norm}")
                if data:
                    data["access_count"] = data.get("access_count", 0) + 1
                    data["last_accessed"] = now_iso
                    self._put(f"node:{nid_norm}", data)

        # Micro-decay: sample 10 random edges, decay confidence by 0.0001
        import random
        edges = self._prefix_scan("edge:")
        if edges:
            sample_size = min(10, len(edges))
            sample = random.sample(edges, sample_size)
            for key, data in sample:
                conf = data.get("confidence", 1.0)
                new_conf = max(0.0, conf - 0.0001)
                if new_conf != conf:
                    data["confidence"] = new_conf
                    self._put(key, data)

    def full_cleanup(self, decay_rate: float = 0.01,
                     min_confidence: float = 0.1):
        """Full cleanup cycle: decay -> GC -> compact -> session_topic cleanup -> importance.
        Called during runtime consolidation cleanup."""
        if not self._is_open:
            return {"decayed": 0, "removed": 0, "compacted": False}

        self.decay_all(decay_rate=decay_rate)
        removed = self.gc_stale_edges(min_confidence=min_confidence)
        self.compact()

        # Cleanup old session_topic nodes (older than 90 days)
        session_removed = 0
        try:
            session_nodes = self.list_nodes(node_type="session_topic")
            cutoff = (datetime.now() - timedelta(days=90)).isoformat()
            for nid, data in session_nodes:
                created = data.get("properties", {}).get("created_at", "")
                if created and created < cutoff:
                    self.delete_node(nid)
                    session_removed += 1
            if session_removed > 0:
                logger.info("Cleaned up %d old session_topic nodes", session_removed)
        except Exception:
            pass

        # Recompute importance scores for nodes
        recomputed = self.batch_recompute_importance()

        return {"decayed": True, "removed": removed, "compacted": True,
                "session_topics_removed": session_removed,
                "importance_recomputed": recomputed}

    def is_stale(self, edge_data: Dict, max_age_minutes: float = 60) -> bool:
        """
        Check if an edge's data is too old.

        Args:
            edge_data: Edge data dict with "timestamp".
            max_age_minutes: Maximum age in minutes before considered stale.

        Returns:
            True if the edge is stale.
        """
        timestamp_str = edge_data.get("timestamp", "")
        if not timestamp_str:
            return True
        try:
            created = datetime.fromisoformat(timestamp_str)
            age = datetime.now() - created
            return age > timedelta(minutes=max_age_minutes)
        except (ValueError, TypeError):
            return True

    # ------------------------------------------------------------------
    # Stats & Export
    # ------------------------------------------------------------------

    def stats(self) -> Dict:
        """
        Get graph statistics.

        Returns:
            Dict with node_count, edge_count, synonym_count, predicate_count,
            backend, serializer, db_size_bytes, hot_tier, embedding_count,
            and type_distribution.
        """
        node_count = len(self._prefix_scan("node:"))
        edge_count = len(self._prefix_scan("edge:"))
        synonym_count = len(self._prefix_scan("syn:"))
        pred_keys = self._prefix_scan("idx:pred:")
        predicate_count = len(pred_keys)

        db_size = 0
        if os.path.exists(self.config.db_path) and os.path.isdir(self.config.db_path):
            for dirpath, _, filenames in os.walk(self.config.db_path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    db_size += os.path.getsize(fp)

        result = {
            "node_count": node_count,
            "edge_count": edge_count,
            "synonym_count": synonym_count,
            "predicate_count": predicate_count,
            "backend": self._backend_type,
            "serializer": _SERIALIZER,
            "db_size_bytes": db_size,
            "is_open": self._is_open,
        }

        # Hot tier stats
        if self._hot_tier is not None:
            result["hot_tier"] = self._hot_tier.stats()

        # Embedding count
        if self._embedding_index is not None:
            result["embedding_count"] = self._embedding_index.count()

        # Type distribution from type indexes
        type_dist = {}
        for nt in NodeType:
            type_idx = self._get(f"idx:type:{int(nt)}")
            if type_idx:
                type_dist[NODE_TYPE_NAMES.get(nt, str(nt))] = len(type_idx)
        if type_dist:
            result["type_distribution"] = type_dist

        return result

    def export_json(self, path: str):
        """
        Export the entire graph to a JSON file.

        Args:
            path: Output file path.
        """
        data = {
            "nodes": {},
            "edges": [],
            "synonyms": {},
            "exported_at": datetime.now().isoformat(),
        }

        for key, value in self._prefix_scan("node:"):
            nid = key[5:]
            data["nodes"][nid] = value

        for key, value in self._prefix_scan("edge:"):
            data["edges"].append(value)

        for key, value in self._prefix_scan("syn:"):
            pred = key[4:]
            data["synonyms"][pred] = value

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info("Graph exported to %s (%d nodes, %d edges)",
                     path, len(data["nodes"]), len(data["edges"]))

    def import_json(self, path: str):
        """
        Import a graph from a JSON file (merges into current database).

        Args:
            path: Input file path.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Import nodes
        for nid, node_data in data.get("nodes", {}).items():
            self._put(f"node:{nid}", node_data)

        # Import edges
        for edge_data in data.get("edges", []):
            s = edge_data.get("subject", "")
            p = edge_data.get("predicate", "")
            o = edge_data.get("object", "")
            if s and p and o:
                self.add_edge(
                    s, p, o,
                    confidence=edge_data.get("confidence", 1.0),
                    source=edge_data.get("source", "import"),
                    weight=edge_data.get("weight", 1.0),
                )

        # Import synonyms
        for pred, syn_data in data.get("synonyms", {}).items():
            self._put(f"syn:{pred}", syn_data)

        logger.info("Graph imported from %s", path)

    # ------------------------------------------------------------------
    # Binary Serialization (NGRE format)
    # ------------------------------------------------------------------

    _BINARY_MAGIC = b'NGRE'
    _BINARY_VERSION = 1

    def export_binary(self, path: str):
        """Export graph to compact binary format (ngre_graph.bin).

        Binary layout:
            - Magic bytes: b'NGRE' (4 bytes)
            - Version: uint32 (1)
            - Node count: uint32
            - Edge count: uint32
            - For each node: len(id):uint16, id:bytes, len(data):uint32, data:msgpack_bytes
            - For each edge: len(key):uint16, key:bytes, len(data):uint32, data:msgpack_bytes

        Args:
            path: Output file path.
        """
        import struct

        nodes = self._prefix_scan("node:")
        edges = self._prefix_scan("edge:")

        with open(path, "wb") as f:
            # Header
            f.write(self._BINARY_MAGIC)
            f.write(struct.pack("<I", self._BINARY_VERSION))
            f.write(struct.pack("<I", len(nodes)))
            f.write(struct.pack("<I", len(edges)))

            # Nodes
            for key, data in nodes:
                nid = key[5:]  # strip "node:"
                id_bytes = nid.encode("utf-8")
                data_bytes = msgpack.packb(data, use_bin_type=True)
                f.write(struct.pack("<H", len(id_bytes)))
                f.write(id_bytes)
                f.write(struct.pack("<I", len(data_bytes)))
                f.write(data_bytes)

            # Edges
            for key, data in edges:
                key_bytes = key.encode("utf-8")
                data_bytes = msgpack.packb(data, use_bin_type=True)
                f.write(struct.pack("<H", len(key_bytes)))
                f.write(key_bytes)
                f.write(struct.pack("<I", len(data_bytes)))
                f.write(data_bytes)

        logger.info("Graph exported to binary %s (%d nodes, %d edges)",
                     path, len(nodes), len(edges))

    def import_binary(self, path: str):
        """Import graph from binary format (ngre_graph.bin).

        Validates magic bytes and version before importing. Merges into
        the current database (existing data is preserved, conflicts are
        overwritten by the imported data).

        Args:
            path: Input file path.

        Raises:
            ValueError: If magic bytes or version are invalid.
        """
        import struct

        with open(path, "rb") as f:
            # Validate header
            magic = f.read(4)
            if magic != self._BINARY_MAGIC:
                raise ValueError(
                    f"Invalid binary graph file: expected magic b'NGRE', "
                    f"got {magic!r}"
                )

            version = struct.unpack("<I", f.read(4))[0]
            if version != self._BINARY_VERSION:
                raise ValueError(
                    f"Unsupported binary graph version: {version} "
                    f"(expected {self._BINARY_VERSION})"
                )

            node_count = struct.unpack("<I", f.read(4))[0]
            edge_count = struct.unpack("<I", f.read(4))[0]

            # Import nodes
            imported_nodes = 0
            for _ in range(node_count):
                id_len = struct.unpack("<H", f.read(2))[0]
                nid = f.read(id_len).decode("utf-8")
                data_len = struct.unpack("<I", f.read(4))[0]
                data_bytes = f.read(data_len)
                data = msgpack.unpackb(data_bytes, raw=False)
                self._put(f"node:{nid}", data)
                imported_nodes += 1

            # Import edges
            imported_edges = 0
            for _ in range(edge_count):
                key_len = struct.unpack("<H", f.read(2))[0]
                key = f.read(key_len).decode("utf-8")
                data_len = struct.unpack("<I", f.read(4))[0]
                data_bytes = f.read(data_len)
                data = msgpack.unpackb(data_bytes, raw=False)

                # Re-add via add_edge to rebuild all indexes and reverse edges
                s = data.get("subject", "")
                p = data.get("predicate", "")
                o = data.get("object", "")
                if s and p and o:
                    self.add_edge(
                        s, p, o,
                        confidence=data.get("confidence", 1.0),
                        source=data.get("source", "binary_import"),
                        weight=data.get("weight", 1.0),
                    )
                    imported_edges += 1

        logger.info("Graph imported from binary %s (%d nodes, %d edges)",
                     path, imported_nodes, imported_edges)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        if not self._is_open:
            return f"QORGraph(path='{self.config.db_path}', closed)"
        s = self.stats()
        return (
            f"QORGraph(path='{self.config.db_path}', "
            f"nodes={s['node_count']}, edges={s['edge_count']}, "
            f"backend={s['backend']}, serializer={s['serializer']})"
        )

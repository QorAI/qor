"""
QOR Snapshot Compression Pipeline
===================================
Hierarchical rollup aggregation for graph snapshot nodes.

PRD Section 14:
  Stage 1: Hourly — 12 five-min snapshots -> 1 hourly (12:1)
  Stage 2: Daily  — 24 hourly -> 1 daily (24:1)
  Stage 3: Weekly/Monthly — 7 daily -> 1 weekly, 4 weekly -> 1 monthly

Target: 1 year of data per entity -> ~30 nodes (from ~105K raw). 3,500x reduction.
"""

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class AggregatedSnapshot:
    """Result of aggregating multiple snapshots."""
    entity: str              # "AAPL", "BTC", "weather:NYC"
    period: str              # "hourly", "daily", "weekly", "monthly"
    start_time: str          # ISO timestamp
    end_time: str            # ISO timestamp
    count: int               # Number of source snapshots
    summary: str             # Aggregated text
    open_val: Optional[float] = None
    close_val: Optional[float] = None
    high_val: Optional[float] = None
    low_val: Optional[float] = None
    mean_val: Optional[float] = None
    trend: str = ""          # "up", "down", "flat"
    source_ids: List[str] = field(default_factory=list)


def _extract_numeric(text: str) -> Optional[float]:
    """Extract a primary numeric value from text."""
    patterns = [
        r'\$?([\d,]+\.?\d*)',
        r'([\d,]+\.?\d*)\s*%',
        r'price[:\s]+([\d,]+\.?\d*)',
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            try:
                return float(m.group(1).replace(",", ""))
            except (ValueError, IndexError):
                continue
    return None


def _detect_trend(values: List[float]) -> str:
    """Detect trend from a list of values."""
    if len(values) < 2:
        return "flat"
    first_half = sum(values[:len(values) // 2]) / max(len(values) // 2, 1)
    second_half = sum(values[len(values) // 2:]) / max(
        len(values) - len(values) // 2, 1)
    pct_change = (second_half - first_half) / max(abs(first_half), 0.01)
    if pct_change > 0.01:
        return "up"
    elif pct_change < -0.01:
        return "down"
    return "flat"


def _dedup_texts(texts: List[str], threshold: float = 0.5) -> List[str]:
    """Deduplication using word overlap (fast) + optional cosine similarity."""
    if not texts:
        return []
    result = [texts[0]]
    for text in texts[1:]:
        words_new = set(text.lower().split())
        is_dup = False
        for existing in result:
            words_existing = set(existing.lower().split())
            if not words_new or not words_existing:
                continue
            overlap = len(words_new & words_existing) / max(
                min(len(words_new), len(words_existing)), 1)
            if overlap > threshold:
                is_dup = True
                break
        if not is_dup:
            result.append(text)
    return result


def cosine_dedup_texts(texts: List[str], threshold: float = 0.92) -> List[str]:
    """Cosine similarity dedup using TF-IDF vectors.

    Better than word overlap for detecting paraphrased duplicates.
    Falls back to _dedup_texts if numpy not available.
    """
    if not texts or len(texts) < 2:
        return texts
    try:
        import numpy as np
    except ImportError:
        return _dedup_texts(texts, threshold=0.5)

    # Build simple TF-IDF vectors
    # Vocabulary: all unique words across texts
    vocab: Dict[str, int] = {}
    doc_words = []
    for text in texts:
        words = set(text.lower().split())
        doc_words.append(words)
        for w in words:
            if w not in vocab:
                vocab[w] = len(vocab)

    if not vocab:
        return texts

    # Build TF-IDF matrix (N x V)
    n = len(texts)
    v = len(vocab)
    matrix = np.zeros((n, v), dtype='float32')
    for i, words in enumerate(doc_words):
        for w in words:
            matrix[i, vocab[w]] = 1.0

    # IDF: log(N / df) per term
    df = np.sum(matrix > 0, axis=0) + 1  # +1 smoothing
    idf = np.log(n / df)
    matrix *= idf

    # Normalize rows
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    normed = matrix / norms

    # Greedy dedup: keep text if not similar to any already-kept text
    keep = [0]
    for i in range(1, n):
        is_dup = False
        for j in keep:
            sim = float(np.dot(normed[i], normed[j]))
            if sim >= threshold:
                is_dup = True
                break
        if not is_dup:
            keep.append(i)

    return [texts[i] for i in keep]


def cluster_topics(texts: List[str], n_clusters: int = 5,
                   ) -> List[Dict]:
    """Cluster texts into topics using simple k-means on TF-IDF vectors.

    Returns list of {cluster_id, centroid_text, members, size}.
    Falls back to single cluster if numpy not available.
    """
    if not texts:
        return []
    try:
        import numpy as np
    except ImportError:
        return [{"cluster_id": 0, "centroid_text": texts[0],
                 "members": texts, "size": len(texts)}]

    # Build TF-IDF matrix (same as cosine_dedup_texts)
    vocab: Dict[str, int] = {}
    doc_words = []
    for text in texts:
        words = set(text.lower().split())
        doc_words.append(words)
        for w in words:
            if w not in vocab:
                vocab[w] = len(vocab)

    if not vocab or len(texts) <= n_clusters:
        return [{"cluster_id": i, "centroid_text": t,
                 "members": [t], "size": 1} for i, t in enumerate(texts)]

    n = len(texts)
    v = len(vocab)
    matrix = np.zeros((n, v), dtype='float32')
    for i, words in enumerate(doc_words):
        for w in words:
            matrix[i, vocab[w]] = 1.0
    df = np.sum(matrix > 0, axis=0) + 1
    idf = np.log(n / df)
    matrix *= idf

    # Normalize
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    matrix /= norms

    # Simple k-means (10 iterations)
    n_clusters = min(n_clusters, n)
    # Initialize centroids randomly
    rng = np.random.RandomState(42)
    centroid_idx = rng.choice(n, n_clusters, replace=False)
    centroids = matrix[centroid_idx].copy()

    assignments = np.zeros(n, dtype='int32')
    for _ in range(10):
        # Assign each point to nearest centroid (cosine)
        sims = matrix @ centroids.T
        assignments = np.argmax(sims, axis=1)
        # Update centroids
        for c in range(n_clusters):
            mask = assignments == c
            if mask.any():
                centroids[c] = matrix[mask].mean(axis=0)
                norm = np.linalg.norm(centroids[c])
                if norm > 1e-8:
                    centroids[c] /= norm

    # Build cluster results
    clusters = []
    for c in range(n_clusters):
        members = [texts[i] for i in range(n) if assignments[i] == c]
        if not members:
            continue
        # Find the text closest to centroid
        member_idx = [i for i in range(n) if assignments[i] == c]
        sims = matrix[member_idx] @ centroids[c]
        best = member_idx[int(np.argmax(sims))]
        clusters.append({
            "cluster_id": c,
            "centroid_text": texts[best][:200],
            "members": members,
            "size": len(members),
        })

    return sorted(clusters, key=lambda x: -x["size"])


class SnapshotCompressor:
    """
    Hierarchical snapshot compression for graph nodes.

    Usage:
        compressor = SnapshotCompressor(graph)
        compressor.run_hourly_rollup()   # Every hour
        compressor.run_daily_rollup()    # Every midnight
        compressor.run_weekly_rollup()   # Every Sunday
        compressor.run_monthly_rollup()  # 1st of month
    """

    def __init__(self, graph=None, data_dir: str = "qor-data",
                 ngre_brain=None):
        self.graph = graph
        self._data_dir = data_dir
        self.ngre_brain = ngre_brain

    def aggregate_snapshots(self, nodes: List[Dict],
                            period: str = "hourly",
                            entity: str = "") -> Optional[AggregatedSnapshot]:
        """
        Aggregate multiple snapshot nodes into a single summary.

        Args:
            nodes: list of node dicts with {content, timestamp, ...}
            period: "hourly", "daily", "weekly", "monthly"
            entity: entity key (e.g., "AAPL", "BTC")

        Returns:
            AggregatedSnapshot or None
        """
        if not nodes:
            return None

        values = []
        texts = []
        timestamps = []
        source_ids = []

        for n in nodes:
            props = n.get("properties", n)
            content = props.get("content", "")
            texts.append(content)
            ts = props.get("timestamp", "")
            timestamps.append(ts)

            nid = n.get("id", "") or props.get("id", "")
            if nid:
                source_ids.append(nid)

            val = _extract_numeric(content)
            if val is not None:
                values.append(val)

        # Dedup texts
        unique_texts = _dedup_texts(texts)

        # Build summary
        if values:
            summary_parts = [
                f"{entity} {period} summary ({len(nodes)} samples):",
                f"  Open: {values[0]:.2f}, Close: {values[-1]:.2f}",
                f"  High: {max(values):.2f}, Low: {min(values):.2f}",
                f"  Mean: {sum(values)/len(values):.2f}",
            ]
            trend = _detect_trend(values)
        else:
            # Text-only (news, events)
            summary_parts = [
                f"{entity} {period} summary ({len(nodes)} items):"
            ]
            for t in unique_texts[:5]:
                summary_parts.append(f"  - {t[:200]}")
            trend = ""

        return AggregatedSnapshot(
            entity=entity,
            period=period,
            start_time=min(timestamps) if timestamps else "",
            end_time=max(timestamps) if timestamps else "",
            count=len(nodes),
            summary="\n".join(summary_parts),
            open_val=values[0] if values else None,
            close_val=values[-1] if values else None,
            high_val=max(values) if values else None,
            low_val=min(values) if values else None,
            mean_val=sum(values) / len(values) if values else None,
            trend=trend,
            source_ids=source_ids,
        )

    def run_hourly_rollup(self) -> Dict[str, int]:
        """
        Roll up snapshot nodes older than 1 hour into hourly summaries.
        12 five-minute snapshots -> 1 hourly node.

        Returns:
            {"rolled_up": N, "removed": N, "created": N}
        """
        if not self.graph or not self.graph.is_open:
            return {"rolled_up": 0, "removed": 0, "created": 0}

        return self._rollup(
            max_age_hours=1,
            period="hourly",
            source_period="snapshot",
            min_nodes=3,
        )

    def run_daily_rollup(self) -> Dict[str, int]:
        """Roll hourly summaries older than 24h into daily summaries."""
        if not self.graph or not self.graph.is_open:
            return {"rolled_up": 0, "removed": 0, "created": 0}

        return self._rollup(
            max_age_hours=24,
            period="daily",
            source_period="hourly",
            min_nodes=2,
        )

    def run_weekly_rollup(self) -> Dict[str, int]:
        """Roll daily summaries older than 7 days into weekly summaries."""
        if not self.graph or not self.graph.is_open:
            return {"rolled_up": 0, "removed": 0, "created": 0}

        return self._rollup(
            max_age_hours=7 * 24,
            period="weekly",
            source_period="daily",
            min_nodes=2,
        )

    def run_monthly_rollup(self) -> Dict[str, int]:
        """Roll weekly summaries older than 30 days into monthly summaries."""
        if not self.graph or not self.graph.is_open:
            return {"rolled_up": 0, "removed": 0, "created": 0}

        return self._rollup(
            max_age_hours=30 * 24,
            period="monthly",
            source_period="weekly",
            min_nodes=2,
        )

    def _rollup(self, max_age_hours: int, period: str,
                source_period: str, min_nodes: int) -> Dict[str, int]:
        """Generic rollup: find old nodes of source_period, aggregate."""
        now = datetime.now()
        cutoff = now - timedelta(hours=max_age_hours)
        cutoff_iso = cutoff.isoformat()

        # Find snapshot-type nodes
        rolled_up = 0
        removed = 0
        created = 0

        try:
            # Scan for snapshot nodes grouped by entity
            entity_nodes = {}
            nodes = self.graph.list_nodes(
                node_type="snapshot")
            for item in nodes:
                # list_nodes returns [(nid, data), ...]
                if isinstance(item, tuple):
                    nid, data = item
                else:
                    nid = item
                    data = self.graph.get_node(nid)
                if not data:
                    continue
                props = data.get("properties", {})
                ts = props.get("timestamp", "")
                node_period = props.get("period", "snapshot")

                if node_period != source_period:
                    continue
                if ts and ts > cutoff_iso:
                    continue  # Too recent

                entity = props.get("entity", "unknown")
                if entity not in entity_nodes:
                    entity_nodes[entity] = []
                entity_nodes[entity].append({"id": nid, **data})

            # Aggregate per entity
            for entity, old_nodes in entity_nodes.items():
                if len(old_nodes) < min_nodes:
                    continue

                agg = self.aggregate_snapshots(old_nodes, period, entity)
                if not agg:
                    continue

                # Create aggregated node
                agg_hash = hashlib.sha256(
                    f"{entity}:{period}:{agg.start_time}".encode()
                ).hexdigest()[:10]
                agg_id = f"snap:{period}:{agg_hash}"

                # Compute embedding for aggregated summary
                embedding = None
                if self.ngre_brain is not None:
                    try:
                        embedding = self.ngre_brain.compute_embedding(
                            agg.summary[:2000])
                    except Exception:
                        pass

                self.graph.add_node(agg_id, node_type="snapshot",
                                    properties={
                    "content": agg.summary,
                    "entity": entity,
                    "period": period,
                    "start_time": agg.start_time,
                    "end_time": agg.end_time,
                    "count": agg.count,
                    "trend": agg.trend,
                    "timestamp": now.isoformat(),
                },
                    embedding=embedding,
                    text_summary=agg.summary[:200],
                )
                created += 1

                # Delete source nodes
                for node in old_nodes:
                    try:
                        self.graph.delete_node(node["id"])
                        removed += 1
                    except Exception:
                        pass

                rolled_up += len(old_nodes)

        except Exception as e:
            logger.warning("Rollup error: %s", e)

        logger.info("%s rollup: %d rolled up, %d removed, %d created",
                    period, rolled_up, removed, created)
        return {"rolled_up": rolled_up, "removed": removed, "created": created}

    def get_compression_stats(self) -> Dict[str, int]:
        """Return counts of nodes at each compression level."""
        stats = {"snapshot": 0, "hourly": 0, "daily": 0,
                 "weekly": 0, "monthly": 0}
        if not self.graph or not self.graph.is_open:
            return stats

        try:
            nodes = self.graph.list_nodes(node_type="snapshot")
            for item in nodes:
                if isinstance(item, tuple):
                    nid, data = item
                else:
                    nid = item
                    data = self.graph.get_node(nid)
                if data:
                    period = data.get("properties", {}).get("period",
                                                            "snapshot")
                    if period in stats:
                        stats[period] += 1
        except Exception:
            pass

        return stats

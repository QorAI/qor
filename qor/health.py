"""
QOR Graph Health Monitoring
=============================
8 health metrics with thresholds (PRD Section 27).

Metrics:
  1. Node churn rate (new-deleted / total per day)
  2. Edge density (avg edges per node)
  3. Confidence distribution (% edges with conf > 0.7)
  4. Graph size (total nodes vs limit)
  5. Index staleness (time since last reindex)
  6. Correction backlog (unprocessed corrections)
  7. Cold tier size (archive bytes)
  8. Orphan nodes (nodes with no edges)
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class HealthMetric:
    """A single health metric measurement."""
    name: str
    value: float
    threshold_min: float = 0.0
    threshold_max: float = float("inf")
    unit: str = ""
    healthy: bool = True
    action: str = ""


@dataclass
class HealthReport:
    """Complete graph health report."""
    timestamp: str = ""
    metrics: List[HealthMetric] = field(default_factory=list)
    overall_healthy: bool = True
    actions_needed: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Format health report for display."""
        lines = ["Graph Health Report:"]
        for m in self.metrics:
            status = "OK" if m.healthy else "WARNING"
            val_str = f"{m.value:.1f}" if isinstance(m.value, float) else str(m.value)
            lines.append(f"  {m.name}: {val_str}{m.unit} [{status}]")
        if self.actions_needed:
            lines.append("\nActions needed:")
            for a in self.actions_needed:
                lines.append(f"  - {a}")
        return "\n".join(lines)


class GraphHealthMonitor:
    """
    Monitors 8 health metrics for the knowledge graph.

    Usage:
        monitor = GraphHealthMonitor(graph, cold_tier)
        report = monitor.check_health()
        print(report.summary())
    """

    # PRD Section 27 thresholds
    THRESHOLDS = {
        "node_churn_rate": {"max": 5.0, "unit": "%/day"},
        "edge_density": {"min": 3.0, "max": 50.0, "unit": " edges/node"},
        "confidence_high_pct": {"min": 40.0, "unit": "%"},
        "graph_size": {"max": 1_000_000, "unit": " nodes"},
        "index_staleness": {"max": 3600, "unit": "s"},
        "correction_backlog": {"max": 100, "unit": ""},
        "cold_tier_size": {"max": 100_000, "unit": " MB"},
        "orphan_pct": {"max": 2.0, "unit": "%"},
    }

    def __init__(self, graph=None, cold_tier=None):
        self.graph = graph
        self.cold_tier = cold_tier
        self._last_node_count = None
        self._last_check_time = None

    def check_health(self) -> HealthReport:
        """Run all 8 health checks and return report."""
        report = HealthReport(timestamp=datetime.now().isoformat())

        checks = [
            self._check_node_churn,
            self._check_edge_density,
            self._check_confidence_distribution,
            self._check_graph_size,
            self._check_index_staleness,
            self._check_correction_backlog,
            self._check_cold_tier_size,
            self._check_orphan_nodes,
        ]

        for check_fn in checks:
            try:
                metric = check_fn()
                report.metrics.append(metric)
                if not metric.healthy:
                    report.overall_healthy = False
                    if metric.action:
                        report.actions_needed.append(metric.action)
            except Exception as e:
                report.metrics.append(HealthMetric(
                    name=check_fn.__name__.replace("_check_", ""),
                    value=-1, healthy=True,
                    action=f"Check failed: {e}",
                ))

        return report

    def _get_stats(self) -> Dict:
        """Get graph stats safely."""
        if not self.graph or not self.graph.is_open:
            return {}
        try:
            return self.graph.stats()
        except Exception:
            return {}

    def _check_node_churn(self) -> HealthMetric:
        """Metric 1: Node churn rate (% change per day)."""
        stats = self._get_stats()
        node_count = stats.get("node_count", 0)
        churn_pct = 0.0

        if self._last_node_count is not None and self._last_check_time:
            elapsed_hours = max(
                (time.time() - self._last_check_time) / 3600, 0.01)
            delta = abs(node_count - self._last_node_count)
            churn_pct = (delta / max(node_count, 1)) * (24 / elapsed_hours) * 100

        self._last_node_count = node_count
        self._last_check_time = time.time()

        t = self.THRESHOLDS["node_churn_rate"]
        healthy = churn_pct <= t["max"]
        return HealthMetric(
            name="Node churn rate", value=churn_pct,
            threshold_max=t["max"], unit=t["unit"], healthy=healthy,
            action="Consolidate old data into cold tier" if not healthy else "",
        )

    def _check_edge_density(self) -> HealthMetric:
        """Metric 2: Average edges per node."""
        stats = self._get_stats()
        nodes = max(stats.get("node_count", 1), 1)
        edges = stats.get("edge_count", 0)
        density = edges / nodes

        t = self.THRESHOLDS["edge_density"]
        healthy = t["min"] <= density <= t["max"]
        action = ""
        if density > t["max"]:
            action = "Prune low-confidence edges"
        elif density < t["min"]:
            action = "Graph underpopulated, run bootstrap"
        return HealthMetric(
            name="Edge density", value=density,
            threshold_min=t["min"], threshold_max=t["max"],
            unit=t["unit"], healthy=healthy, action=action,
        )

    def _check_confidence_distribution(self) -> HealthMetric:
        """Metric 3: % of edges with confidence > 0.7."""
        stats = self._get_stats()
        total = max(stats.get("edge_count", 1), 1)
        high_conf = 0

        # Sample edges to estimate distribution
        if self.graph and self.graph.is_open:
            try:
                # Sample up to 200 nodes, check their edges
                sample_nodes = self.graph.list_nodes(limit=200)
                checked = 0
                for nid, _data in sample_nodes:
                    edges = self.graph.get_edges(nid)
                    for e in edges:
                        checked += 1
                        if e.get("confidence", 0) > 0.7:
                            high_conf += 1
                    if checked > 500:
                        break
                if checked > 0:
                    pct = (high_conf / checked) * 100
                else:
                    pct = 50.0  # Default
            except Exception:
                pct = 50.0
        else:
            pct = 50.0

        t = self.THRESHOLDS["confidence_high_pct"]
        healthy = pct >= t["min"]
        return HealthMetric(
            name="High-confidence edges", value=pct,
            threshold_min=t["min"], unit=t["unit"], healthy=healthy,
            action="Re-verify stale edges" if not healthy else "",
        )

    def _check_graph_size(self) -> HealthMetric:
        """Metric 4: Total node count vs limit."""
        stats = self._get_stats()
        count = stats.get("node_count", 0)

        t = self.THRESHOLDS["graph_size"]
        healthy = count <= t["max"]
        return HealthMetric(
            name="Graph size", value=count,
            threshold_max=t["max"], unit=t["unit"], healthy=healthy,
            action="Promote old nodes to cold tier" if not healthy else "",
        )

    def _check_index_staleness(self) -> HealthMetric:
        """Metric 5: Time since last HNSW reindex."""
        staleness = 0
        if self.graph and self.graph.is_open:
            try:
                idx = getattr(self.graph, '_embedding_index', None)
                if idx and hasattr(idx, '_last_save_time'):
                    staleness = time.time() - idx._last_save_time
                else:
                    staleness = 0  # No index = OK
            except Exception:
                staleness = 0

        t = self.THRESHOLDS["index_staleness"]
        healthy = staleness <= t["max"]
        return HealthMetric(
            name="Index staleness", value=staleness,
            threshold_max=t["max"], unit=t["unit"], healthy=healthy,
            action="Rebuild embedding index" if not healthy else "",
        )

    def _check_correction_backlog(self) -> HealthMetric:
        """Metric 6: Unprocessed correction nodes."""
        count = 0
        if self.graph and self.graph.is_open:
            try:
                corrs = self.graph.list_nodes(node_type="correction",
                                              limit=200)
                count = len(corrs)
            except Exception:
                pass

        t = self.THRESHOLDS["correction_backlog"]
        healthy = count <= t["max"]
        return HealthMetric(
            name="Correction backlog", value=count,
            threshold_max=t["max"], unit=t["unit"], healthy=healthy,
            action="Trigger cascade engine" if not healthy else "",
        )

    def _check_cold_tier_size(self) -> HealthMetric:
        """Metric 7: Cold tier archive size in MB."""
        size_mb = 0.0
        if self.cold_tier:
            try:
                stats = self.cold_tier.stats()
                size_mb = stats.get("total_size_mb", 0.0)
            except Exception:
                pass

        t = self.THRESHOLDS["cold_tier_size"]
        healthy = size_mb <= t["max"]
        return HealthMetric(
            name="Cold tier size", value=size_mb,
            threshold_max=t["max"], unit=t["unit"], healthy=healthy,
            action="Delete >1 year old snapshots" if not healthy else "",
        )

    def _check_orphan_nodes(self) -> HealthMetric:
        """Metric 8: % of nodes with no edges."""
        orphan_pct = 0.0
        if self.graph and self.graph.is_open:
            try:
                sample = self.graph.list_nodes(limit=500)
                orphans = 0
                for nid, _data in sample:
                    edges = self.graph.get_edges(nid)
                    if not edges:
                        orphans += 1
                if sample:
                    orphan_pct = (orphans / len(sample)) * 100
            except Exception:
                pass

        t = self.THRESHOLDS["orphan_pct"]
        healthy = orphan_pct <= t["max"]
        return HealthMetric(
            name="Orphan nodes", value=orphan_pct,
            threshold_max=t["max"], unit=t["unit"], healthy=healthy,
            action="Mark orphans for deletion or reindex" if not healthy else "",
        )


# ==============================================================================
# Source Reliability Tracking (Phase E.22)
# ==============================================================================

class SourceReliabilityTracker:
    """
    Track corrections per source, adjust default confidence.

    PRD Section 21.3: Sources with many corrections get lower
    default confidence for new nodes.
    """

    def __init__(self, graph=None):
        self.graph = graph
        self._stats = {}  # source_name -> {queries: int, corrections: int}

    def record_query(self, source: str):
        """Record a successful query from a source."""
        if source not in self._stats:
            self._stats[source] = {"queries": 0, "corrections": 0,
                                   "degraded": False}
        self._stats[source]["queries"] += 1

    def record_correction(self, source: str):
        """Record a correction for a source.

        Also checks if corrections/queries ratio > 0.3 and sets "degraded" flag.
        """
        if source not in self._stats:
            self._stats[source] = {"queries": 0, "corrections": 0,
                                   "degraded": False}
        self._stats[source]["corrections"] += 1
        # Check degraded threshold
        queries = self._stats[source]["queries"]
        corrections = self._stats[source]["corrections"]
        if queries > 0 and (corrections / queries) > 0.3:
            self._stats[source]["degraded"] = True

    def get_reliability(self, source: str) -> float:
        """
        Get reliability score for a source.
        reliability = 1 - (corrections / (queries + 1))
        """
        stats = self._stats.get(source, {"queries": 0, "corrections": 0})
        return 1.0 - (stats["corrections"] / (stats["queries"] + 1))

    def get_penalty(self, source: str) -> float:
        """
        Get penalty multiplier for a source based on correction ratio.

        Returns:
            1.0 — 0 corrections (no penalty)
            0.9 — ratio 0.1-0.2
            0.8 — ratio 0.2-0.3
            0.7 — ratio > 0.3 (degraded)
        """
        stats = self._stats.get(source)
        if not stats:
            return 1.0
        queries = stats.get("queries", 0)
        corrections = stats.get("corrections", 0)
        if queries == 0 or corrections == 0:
            return 1.0
        ratio = corrections / queries
        if ratio > 0.3:
            return 0.7
        elif ratio > 0.2:
            return 0.8
        elif ratio > 0.1:
            return 0.9
        return 1.0

    def adjust_confidence(self, confidence: float, source: str) -> float:
        """Multiply confidence by source reliability AND penalty."""
        return confidence * self.get_reliability(source) * self.get_penalty(source)

    def get_all_stats(self) -> Dict[str, Dict]:
        """Return reliability stats for all tracked sources."""
        result = {}
        for source, stats in self._stats.items():
            result[source] = {
                **stats,
                "reliability": self.get_reliability(source),
                "penalty": self.get_penalty(source),
                "degraded": stats.get("degraded", False),
            }
        return result

    def get_weekly_report(self) -> str:
        """
        Format all source stats as a human-readable text report.

        Returns:
            Multi-line string with source reliability summary.
        """
        if not self._stats:
            return "Source Reliability Report: No sources tracked yet."

        lines = [
            "Source Reliability Report",
            "=" * 60,
            f"{'Source':<25} {'Queries':>8} {'Corrections':>12} "
            f"{'Ratio':>7} {'Penalty':>8} {'Status':>10}",
            "-" * 60,
        ]

        # Sort by correction ratio descending (worst first)
        sorted_sources = sorted(
            self._stats.items(),
            key=lambda x: (
                x[1]["corrections"] / max(x[1]["queries"], 1)
            ),
            reverse=True,
        )

        total_queries = 0
        total_corrections = 0
        degraded_count = 0

        for source, stats in sorted_sources:
            queries = stats.get("queries", 0)
            corrections = stats.get("corrections", 0)
            degraded = stats.get("degraded", False)
            ratio = corrections / max(queries, 1)
            penalty = self.get_penalty(source)
            status = "DEGRADED" if degraded else "OK"

            total_queries += queries
            total_corrections += corrections
            if degraded:
                degraded_count += 1

            lines.append(
                f"{source:<25} {queries:>8} {corrections:>12} "
                f"{ratio:>7.2f} {penalty:>8.1f} {status:>10}"
            )

        lines.append("-" * 60)
        lines.append(
            f"{'TOTAL':<25} {total_queries:>8} {total_corrections:>12} "
            f"{'':>7} {'':>8} "
            f"{degraded_count} degraded"
        )
        lines.append(f"\nSources tracked: {len(self._stats)}")
        if degraded_count > 0:
            lines.append(
                f"WARNING: {degraded_count} source(s) degraded "
                f"(correction ratio > 0.3)"
            )

        return "\n".join(lines)

    def save_to_graph(self):
        """Persist source stats to graph."""
        if not self.graph or not self.graph.is_open:
            return
        for source, stats in self._stats.items():
            nid = f"source_stats:{source}"
            self.graph.add_node(nid, node_type="knowledge", properties={
                "source": source,
                "queries": stats["queries"],
                "corrections": stats["corrections"],
                "reliability": self.get_reliability(source),
                "timestamp": datetime.now().isoformat(),
            })

    def load_from_graph(self):
        """Load source stats from graph."""
        if not self.graph or not self.graph.is_open:
            return
        try:
            nodes = self.graph.list_nodes(limit=1000)
            for nid, data in nodes:
                if not nid.startswith("source_stats:"):
                    continue
                if not data:
                    continue
                props = data.get("properties", {})
                source = props.get("source", "")
                if source:
                    self._stats[source] = {
                        "queries": props.get("queries", 0),
                        "corrections": props.get("corrections", 0),
                    }
        except Exception:
            pass


# ==============================================================================
# Automatic Fact-Checking (Phase E.23)
# ==============================================================================

class FactChecker:
    """
    Background fact-checker: verifies stale high-importance nodes.

    PRD Section 21.1: Every 6 hours, check 50 stale nodes.
    If value changed > 5%, create correction + cascade.
    """

    VERIFIABLE_TYPES = {"knowledge", "snapshot", "trade_pattern"}

    # 6 hours in seconds
    _6H_INTERVAL = 21600

    def __init__(self, graph=None, source_tracker=None, tool_executor=None):
        self.graph = graph
        self.source_tracker = source_tracker
        self._tool_executor = tool_executor
        self._last_run = 0.0

    @property
    def time_since_last_run(self) -> float:
        """Seconds since the last fact-check run. 0 if never run."""
        if self._last_run <= 0:
            return 0.0
        return time.time() - self._last_run

    def find_stale_nodes(self, max_age_hours: float = 168,
                         min_importance: float = 0.5,
                         limit: int = 50) -> List[str]:
        """
        Find high-importance nodes that need re-verification.

        Criteria:
          - importance > min_importance
          - type in VERIFIABLE_TYPES
          - age > max_age_hours
          - confidence > 0.5
        """
        if not self.graph or not self.graph.is_open:
            return []

        cutoff = (datetime.now() - __import__("datetime").timedelta(
            hours=max_age_hours)).isoformat()
        candidates = []

        try:
            all_nodes = self.graph.list_nodes(limit=limit * 10)
            for nid, data in all_nodes:
                if len(candidates) >= limit:
                    break

                if not data:
                    continue

                props = data.get("properties", {})
                ntype = props.get("type", "").lower()
                if ntype not in self.VERIFIABLE_TYPES:
                    continue

                importance = props.get("importance", 0.0)
                if importance < min_importance:
                    continue

                confidence = props.get("confidence", 0.5)
                if confidence < 0.5:
                    continue

                ts = props.get("timestamp", "")
                if ts and ts < cutoff:
                    candidates.append(nid)
        except Exception as e:
            logger.warning("Error finding stale nodes: %s", e)

        return candidates

    def verify_node(self, node_id: str,
                    fetch_fn=None,
                    threshold_pct: float = 5.0) -> Optional[Dict]:
        """
        Verify a single node by re-fetching its data.

        Args:
            node_id: node to verify
            fetch_fn: callable(question) -> str (tool/skill call)
            threshold_pct: % difference to trigger correction

        Returns:
            {"verified": bool, "changed": bool, "old": str, "new": str}
            or None if can't verify
        """
        if not self.graph or not fetch_fn:
            return None

        data = self.graph.get_node(node_id)
        if not data:
            return None

        props = data.get("properties", {})
        old_content = props.get("content", "")
        question = props.get("question", "")
        if not question:
            # Try to infer question from content
            question = old_content[:100]

        try:
            new_content = fetch_fn(question)
        except Exception:
            return None

        if not new_content:
            return {"verified": True, "changed": False,
                    "old": old_content, "new": ""}

        # Compare values
        import re
        old_nums = re.findall(r'[\d,]+\.?\d*', old_content)
        new_nums = re.findall(r'[\d,]+\.?\d*', new_content)

        changed = False
        if old_nums and new_nums:
            try:
                old_val = float(old_nums[0].replace(",", ""))
                new_val = float(new_nums[0].replace(",", ""))
                pct_diff = abs(new_val - old_val) / max(abs(old_val), 0.01) * 100
                changed = pct_diff > threshold_pct
            except (ValueError, IndexError):
                # Text comparison fallback
                changed = old_content.strip() != new_content.strip()
        else:
            changed = old_content.strip() != new_content.strip()

        if changed:
            # Update node with fresh data
            props["content"] = new_content[:2000]
            props["last_verified"] = datetime.now().isoformat()
            props["confidence"] = min(
                props.get("confidence", 0.5) * 1.1, 1.0)
            self.graph.add_node(node_id, node_type=props.get("type", "knowledge"),
                                properties=props)
        else:
            # Boost confidence of verified node
            props["last_verified"] = datetime.now().isoformat()
            props["confidence"] = min(
                props.get("confidence", 0.5) * 1.1, 1.0)
            self.graph.add_node(node_id, node_type=props.get("type", "knowledge"),
                                properties=props)

        return {
            "verified": True,
            "changed": changed,
            "old": old_content[:200],
            "new": new_content[:200],
        }

    def run_batch(self, fetch_fn=None, limit: int = 50) -> Dict[str, Any]:
        """
        Run a batch of fact-checks.

        Returns:
            {"checked": int, "changed": int, "verified": int}
        """
        stale = self.find_stale_nodes(limit=limit)
        checked = 0
        changed_count = 0
        verified_count = 0

        for nid in stale:
            result = self.verify_node(nid, fetch_fn)
            if result:
                checked += 1
                if result.get("changed"):
                    changed_count += 1
                if result.get("verified"):
                    verified_count += 1

        self._last_run = time.time()
        logger.info("Fact-check: %d checked, %d changed, %d verified",
                    checked, changed_count, verified_count)

        return {
            "checked": checked,
            "changed": changed_count,
            "verified": verified_count,
        }

    def schedule_6h_cycle(self, tool_executor=None) -> Dict[str, Any]:
        """
        Run a 6-hour background verification cycle (PRD Section 21.1).

        Finds 50 stale high-importance nodes via find_stale_nodes(),
        verifies each with a fetch function wrapping tool_executor,
        and returns a summary.

        Args:
            tool_executor: ToolExecutor instance with .call(tool, query)
                           method. Falls back to self._tool_executor.

        Returns:
            {"checked": int, "changed": int, "corrected": int,
             "skipped_interval": bool}
        """
        executor = tool_executor or self._tool_executor

        # Build a fetch function from the tool executor
        fetch_fn = None
        if executor is not None:
            def _fetch_via_tools(question: str) -> Optional[str]:
                """Try to answer a question using the tool executor."""
                try:
                    # Try web_search first as general-purpose verifier
                    result = executor.call("web_search", question)
                    if result:
                        return result
                except Exception:
                    pass
                try:
                    # Fallback: try duckduckgo instant answer
                    result = executor.call("duckduckgo", question)
                    if result:
                        return result
                except Exception:
                    pass
                return None
            fetch_fn = _fetch_via_tools

        # Find stale nodes (50 batch)
        stale_nodes = self.find_stale_nodes(limit=50)
        checked = 0
        changed_count = 0
        corrected_count = 0

        for nid in stale_nodes:
            result = self.verify_node(nid, fetch_fn=fetch_fn)
            if result is None:
                continue
            checked += 1
            if result.get("changed"):
                changed_count += 1
                # If content changed, this counts as a correction
                corrected_count += 1
                # Trigger cascade if graph supports it
                try:
                    from .knowledge_tree import cascade_correction_full
                    old_content = result.get("old", "")
                    new_content = result.get("new", "")
                    if old_content and new_content:
                        cascade_correction_full(
                            self.graph, nid,
                            old_content, new_content,
                            max_depth=2,
                        )
                except Exception:
                    pass

        self._last_run = time.time()
        logger.info(
            "6h fact-check cycle: %d checked, %d changed, %d corrected",
            checked, changed_count, corrected_count,
        )

        return {
            "checked": checked,
            "changed": changed_count,
            "corrected": corrected_count,
            "skipped_interval": False,
        }

"""
QOR Ingestion Daemon
=====================
Background data ingestion for knowledge graph enrichment.

PRD Section 24: 24/7 ingestion pipeline:
  - Polls configured sources on schedule
  - Extracts entities/relations from new data
  - Writes to knowledge graph with source tracking
  - Triggers snapshot compression on schedule
  - Runs fact-checking on stale nodes

This is a DAEMON — runs in a background thread alongside the main runtime.
"""

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class IngestionSource:
    """A configured data source for ingestion."""
    name: str
    tool_name: str                    # Tool to call (from ToolExecutor)
    query: str                        # Query/topic to ingest
    interval_minutes: float = 60.0    # How often to poll
    priority: int = 3                 # P0-P4 (lower = more important)
    enabled: bool = True
    last_run: float = 0.0
    run_count: int = 0
    error_count: int = 0
    asset_type: str = ""              # "crypto", "stock", "commodity", "forex" (for market hours)


@dataclass
class IngestionConfig:
    """Configuration for the ingestion daemon."""
    enabled: bool = False              # Must opt-in
    check_interval_seconds: float = 30.0
    max_concurrent: int = 3
    compression_interval_hours: float = 1.0
    fact_check_interval_hours: float = 6.0
    max_errors_before_disable: int = 10


class IngestionDaemon:
    """
    Background data ingestion daemon.

    Usage:
        daemon = IngestionDaemon(config, graph, tool_executor)
        daemon.add_source(IngestionSource(
            name="crypto_prices",
            tool_name="crypto_price",
            query="BTC",
            interval_minutes=5,
        ))
        daemon.start()
        ...
        daemon.stop()
    """

    def __init__(self, config: Optional[IngestionConfig] = None,
                 graph=None, tool_executor=None,
                 compressor=None, fact_checker=None,
                 ngre_brain=None,
                 assets: Optional[List[str]] = None,
                 hmm=None, trading_engine=None, futures_engine=None):
        self.config = config or IngestionConfig()
        self.graph = graph
        self.tool_executor = tool_executor
        self.compressor = compressor
        self.fact_checker = fact_checker
        self.ngre_brain = ngre_brain  # NGREBrain for computing embeddings
        self.hmm = hmm                # MarketHMM for regime data
        self.trading_engine = trading_engine  # SpotEngine for position data
        self.futures_engine = futures_engine  # FuturesEngine for position data
        self._assets = assets or []    # Tracked asset symbols

        # Market session tracker — saves daily open/close prices as
        # permanent historical_event nodes (Asia, London, US sessions)
        self._session_tracker = None
        try:
            from .sessions import SessionTracker
            self._session_tracker = SessionTracker(
                graph=graph,
                tool_executor=tool_executor,
                ngre_brain=ngre_brain,
                assets=assets,
            )
        except Exception:
            pass

        self._sources: List[IngestionSource] = []
        self._last_analysis_save = 0.0  # Throttle analysis snapshots
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._stats = {
            "ingested": 0,
            "errors": 0,
            "compressions": 0,
            "fact_checks": 0,
        }
        self._last_compression = 0.0
        self._last_fact_check = 0.0

    def add_source(self, source: IngestionSource):
        """Add an ingestion source."""
        with self._lock:
            self._sources.append(source)
        logger.info("Added ingestion source: %s (every %.0f min)",
                    source.name, source.interval_minutes)

    def remove_source(self, name: str):
        """Remove an ingestion source by name."""
        with self._lock:
            self._sources = [s for s in self._sources if s.name != name]

    def start(self):
        """Start the ingestion daemon."""
        if not self.config.enabled:
            logger.info("Ingestion daemon disabled (config.enabled=False)")
            return

        if self._thread and self._thread.is_alive():
            logger.warning("Ingestion daemon already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="ingestion-daemon")
        self._thread.start()
        logger.info("Ingestion daemon started (%d sources)",
                    len(self._sources))

    def stop(self):
        """Stop the ingestion daemon."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
            self._thread = None
        logger.info("Ingestion daemon stopped")

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def status(self) -> Dict[str, Any]:
        """Return daemon status."""
        with self._lock:
            sources = [{
                "name": s.name,
                "tool": s.tool_name,
                "interval_min": s.interval_minutes,
                "enabled": s.enabled,
                "runs": s.run_count,
                "errors": s.error_count,
            } for s in self._sources]

        result = {
            "running": self.is_running,
            "sources": sources,
            "stats": dict(self._stats),
        }
        if self._session_tracker is not None:
            result["sessions"] = self._session_tracker.status()
        return result

    def _run_loop(self):
        """Main daemon loop."""
        logger.info("Ingestion loop started")
        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception as e:
                logger.warning("Ingestion tick error: %s", e)
                self._stats["errors"] += 1

            self._stop_event.wait(self.config.check_interval_seconds)

    def _tick(self):
        """One ingestion cycle."""
        now = time.time()

        # Market session tracker — check if a session just opened/closed
        if self._session_tracker is not None:
            try:
                self._session_tracker.tick()
            except Exception as e:
                logger.debug("Session tracker tick error: %s", e)

        # Check sources due for polling
        due_sources = []
        with self._lock:
            for source in self._sources:
                if not source.enabled:
                    continue
                if source.error_count >= self.config.max_errors_before_disable:
                    source.enabled = False
                    logger.warning("Disabled source %s after %d errors",
                                   source.name, source.error_count)
                    continue
                elapsed = now - source.last_run
                if elapsed >= source.interval_minutes * 60:
                    due_sources.append(source)

        # Sort by priority (lower = higher priority)
        due_sources.sort(key=lambda s: s.priority)

        # Process up to max_concurrent sources
        for source in due_sources[:self.config.max_concurrent]:
            self._ingest_source(source)

        # Periodic compression
        if (self.compressor and
                now - self._last_compression >=
                self.config.compression_interval_hours * 3600):
            self._run_compression()
            self._last_compression = now

        # Periodic fact-checking
        if (self.fact_checker and
                now - self._last_fact_check >=
                self.config.fact_check_interval_hours * 3600):
            self._run_fact_check()
            self._last_fact_check = now

        # Save combined analysis snapshots (HMM + positions + regime)
        # Throttled internally to every 5 min
        try:
            self._save_analysis_snapshots()
        except Exception as e:
            logger.debug("Analysis snapshot error: %s", e)

    def _ingest_source(self, source: IngestionSource):
        """Ingest data from a single source."""
        if not self.tool_executor:
            return

        # Skip polling for closed markets (weekends/holidays)
        if source.asset_type and source.asset_type != "crypto":
            try:
                from .quant import is_market_open
                mkt = is_market_open(source.asset_type)
                if not mkt["open"]:
                    source.last_run = time.time()  # Don't retry immediately
                    return
            except Exception:
                pass

        try:
            # Call the tool (ToolExecutor.call() handles cache + rate limit)
            result = self.tool_executor.call(
                source.tool_name, source.query)
            if not result:
                source.last_run = time.time()
                source.run_count += 1
                return

            # Store in knowledge graph
            # PRD §10: time-series data (prices, weather) = snapshot nodes
            #           news/articles = event nodes
            #           everything else = knowledge nodes
            if self.graph and self.graph.is_open:
                ts_now = int(time.time())
                node_id = f"ingest:{source.name}:{ts_now}"

                # Determine node_type per PRD §10
                _snapshot_tools = {
                    "crypto_price", "binance_price", "stock_quote",
                    "commodities", "forex_rates", "weather",
                    "technical_analysis", "fear_greed",
                    "market_indices", "crypto_market",
                }
                _event_tools = {
                    "news_search", "hacker_news", "news",
                    "on_this_day", "gdelt",
                }
                if source.tool_name in _snapshot_tools:
                    node_type = "snapshot"
                elif source.tool_name in _event_tools:
                    node_type = "event"
                else:
                    node_type = "knowledge"

                # Check if this data is historically significant
                # (president elected, market crash, earthquake, etc.)
                # If so, ALSO save as permanent historical_event node
                # that never gets deleted or compressed.
                content_str_check = str(result)[:2000]
                try:
                    from .runtime import classify_historical
                    if classify_historical(content_str_check,
                                           source.tool_name):
                        import hashlib as _hl
                        hist_hash = _hl.sha256(
                            content_str_check[:200].encode()
                        ).hexdigest()[:10]
                        hist_id = f"hist:{hist_hash}"

                        hist_emb = None
                        if self.ngre_brain is not None:
                            try:
                                hist_emb = self.ngre_brain.compute_embedding(
                                    content_str_check)
                            except Exception:
                                pass

                        self.graph.add_node(
                            hist_id,
                            node_type="historical_event",
                            properties={
                                "content": content_str_check[:500],
                                "source": source.tool_name,
                                "query": source.query,
                                "date": datetime.now().strftime("%Y-%m-%d"),
                                "timestamp": datetime.now().isoformat(),
                                "confidence": 1.0,
                                "ingested": True,
                            },
                            embedding=hist_emb,
                            text_summary=content_str_check[:200],
                            confidence=1.0,
                            source=source.tool_name,
                        )
                        logger.info(
                            "Historical event detected from %s: %s",
                            source.name, content_str_check[:80])
                except Exception:
                    pass

                # Compute embedding via NGRE brain (Mamba 24-layer)
                embedding = None
                content_str = str(result)[:2000]
                if self.ngre_brain is not None:
                    try:
                        embedding = self.ngre_brain.compute_embedding(
                            content_str)
                    except Exception:
                        pass

                self.graph.add_node(node_id, node_type=node_type,
                                    properties={
                    "content": content_str,
                    "source": source.tool_name,
                    "query": source.query,
                    "entity": source.query or source.name,
                    "period": "snapshot",
                    "timestamp": datetime.now().isoformat(),
                    "confidence": 0.85,
                    "ingested": True,
                },
                    embedding=embedding,
                    text_summary=content_str[:200],
                )

                # Extract entities and add edges
                self._extract_and_link(node_id, str(result), source.name)

            source.last_run = time.time()
            source.run_count += 1
            self._stats["ingested"] += 1

        except Exception as e:
            source.error_count += 1
            self._stats["errors"] += 1
            logger.warning("Ingestion error for %s: %s", source.name, e)

    def _extract_and_link(self, node_id: str, text: str, source: str):
        """Extract entities from ingested text and create graph edges."""
        if not self.graph or not self.graph.is_open:
            return

        import re
        # Simple entity extraction: capitalized words, ticker symbols
        entities = set()

        # Capitalized multi-word names
        for m in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', text):
            entities.add(m.group(1))

        # Ticker symbols (all caps 2-5 chars)
        for m in re.finditer(r'\b([A-Z]{2,5})\b', text):
            word = m.group(1)
            # Filter common English words that look like tickers
            if word not in {"THE", "AND", "FOR", "FROM", "WITH", "THIS",
                            "THAT", "NOT", "ARE", "WAS", "BUT", "HAS"}:
                entities.add(word)

        # Link entities to the ingested node
        for entity in list(entities)[:10]:  # Limit to prevent explosion
            entity_id = f"entity:{entity.lower().replace(' ', '_')}"
            try:
                # Ensure entity node exists
                if not self.graph.get_node(entity_id):
                    # Compute embedding for entity name
                    ent_emb = None
                    if self.ngre_brain is not None:
                        try:
                            ent_emb = self.ngre_brain.compute_embedding(
                                entity)
                        except Exception:
                            pass
                    self.graph.add_node(entity_id, node_type="entity",
                                        properties={
                        "name": entity,
                        "source": source,
                        "timestamp": datetime.now().isoformat(),
                    },
                        embedding=ent_emb,
                        text_summary=entity,
                    )
                # Link ingested data to entity
                self.graph.add_edge(
                    node_id, "mentions", entity_id,
                    confidence=0.7, source=source)
            except Exception:
                pass

    def _run_compression(self):
        """Run snapshot compression."""
        if not self.compressor:
            return
        try:
            result = self.compressor.run_hourly_rollup()
            self._stats["compressions"] += 1
            if result.get("created", 0) > 0:
                logger.info("Compression: rolled up %d -> %d",
                            result.get("rolled_up", 0),
                            result.get("created", 0))
        except Exception as e:
            logger.warning("Compression error: %s", e)

    def _run_fact_check(self):
        """Run background fact-checking."""
        if not self.fact_checker:
            return
        try:
            # Use tool_executor as the fetch function
            fetch_fn = None
            if self.tool_executor:
                def _fetch(q):
                    r = self.tool_executor.execute("web_search", q)
                    return str(r) if r else ""
                fetch_fn = _fetch

            result = self.fact_checker.run_batch(fetch_fn=fetch_fn, limit=50)
            self._stats["fact_checks"] += 1
            logger.info("Fact-check: %d checked, %d changed",
                        result.get("checked", 0), result.get("changed", 0))
        except Exception as e:
            logger.warning("Fact-check error: %s", e)

    def _save_analysis_snapshots(self):
        """Save a combined analysis snapshot per tracked asset.

        Compiles price + TA + HMM regime + trading positions into ONE node
        so tree_search finds comprehensive data for any asset query.
        Runs every 5 minutes (throttled).
        """
        if not self.graph or not getattr(self.graph, 'is_open', False):
            return
        if not self._assets:
            return

        now = time.time()
        if now - self._last_analysis_save < 300:  # 5 min throttle
            return
        self._last_analysis_save = now

        for symbol in self._assets:
            try:
                self._save_one_analysis(symbol)
            except Exception as e:
                logger.debug("Analysis snapshot %s: %s", symbol, e)

    def _save_one_analysis(self, symbol: str):
        """Build and save one combined analysis node for a symbol.

        Saves HMM regime intelligence alongside price data. Trading positions
        are NOT duplicated here — they live in TradeStore/FuturesTradeStore
        and are accessed directly by the trading engines.
        """
        from .quant import classify_asset, is_market_open

        at = classify_asset(symbol)

        # Skip closed markets
        mkt = is_market_open(at.asset_type)
        if not mkt["open"]:
            return

        lines = [f"=== {at.display} Analysis ({datetime.now().strftime('%Y-%m-%d %H:%M UTC')}) ==="]

        # HMM regime — the key intelligence not stored by other sources
        if self.hmm is not None:
            try:
                status = self.hmm.status()
                sym_state = status.get("symbols", {}).get(at.display, {})
                if sym_state:
                    state_name = sym_state.get("state", "UNKNOWN")
                    conf = sym_state.get("confidence", 0)
                    signal = sym_state.get("signal", "HOLD")
                    strength = sym_state.get("signal_strength", 0)
                    regime_age = sym_state.get("regime_age", 0)
                    lines.append(
                        f"HMM Regime: {state_name} (confidence: {conf:.1%}, "
                        f"age: {regime_age} ticks)")
                    lines.append(
                        f"HMM Signal: {signal} (strength: {strength:+.2f})")
                    if sym_state.get("transition"):
                        lines.append(
                            f"Regime Transition: {sym_state.get('prev_state', '?')} "
                            f"→ {state_name}")
            except Exception:
                pass

        # Only save if we have more than the header
        if len(lines) <= 1:
            return

        content = "\n".join(lines)

        # Deterministic node ID — overwrites previous snapshot for same asset
        node_id = f"analysis:{at.display.lower().replace(' ', '_')}"

        # Compute embedding
        embedding = None
        if self.ngre_brain is not None:
            try:
                embedding = self.ngre_brain.compute_embedding(content)
            except Exception:
                pass

        self.graph.add_node(node_id, node_type="snapshot",
                            properties={
            "content": content,
            "source": "analysis_snapshot",
            "query": at.display,
            "entity": at.display,
            "name": f"{at.display} Analysis",
            "timestamp": datetime.now().isoformat(),
            "confidence": 0.9,
            "asset_type": at.asset_type,
        },
            embedding=embedding,
            text_summary=content[:200],
        )

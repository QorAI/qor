"""
QOR Market Sessions — Daily Open/Close Historical Snapshots
=============================================================
Tracks 3 global market sessions and saves open/close prices
as permanent historical_event nodes in the knowledge graph.

Sessions (UTC):
  Asia   (Tokyo):    00:00 – 08:00 UTC
  London:            08:00 – 16:00 UTC
  US     (New York): 13:00 – 21:00 UTC

Assets tracked are DYNAMIC — auto-detected from config:
  - TradingConfig.symbols (spot trading pairs)
  - FuturesConfig.symbols (futures trading pairs)
  - RuntimeConfig.watch_assets (non-traded assets to track)

Works for crypto, stocks, commodities, forex — any asset type.

At each session OPEN and CLOSE boundary, fetches live prices
for ALL tracked assets and writes a historical_event node that is
NEVER auto-deleted, NEVER compressed. This is the permanent
daily record.

Example node content (with BTC, AAPL, gold, EUR/USD):
  Market Session Close — US (2026-02-14)
    BTC:     Open $69,500.00 → Close $69,628.00 (+0.18%)
    AAPL:    Open $189.50    → Close $190.20    (+0.37%)
    Gold:    Open $5,100.00  → Close $5,120.00  (+0.39%)
    EUR/USD: Open $1.0852    → Close $1.0860    (+0.07%)

Usage:
    tracker = SessionTracker(graph, tool_executor, assets=["BTC", "AAPL", "gold"])
    tracker.tick()  # Call every ingestion cycle (30s)
"""

import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Session definitions ──────────────────────────────────────────────

@dataclass
class MarketSession:
    """A named market session with UTC open/close hours."""
    name: str          # "Asia", "London", "US"
    open_hour: int     # UTC hour session opens
    close_hour: int    # UTC hour session closes
    region: str = ""   # "Tokyo", "London", "New York"


SESSIONS = [
    MarketSession("Asia",   open_hour=0,  close_hour=8,  region="Tokyo"),
    MarketSession("London", open_hour=8,  close_hour=16, region="London"),
    MarketSession("US",     open_hour=13, close_hour=21, region="New York"),
]


# ── Price extraction ─────────────────────────────────────────────────

def _extract_price(tool_result: str) -> Optional[float]:
    """Extract the primary price number from a tool result string."""
    import re
    patterns = [
        r'\$\s*([\d,]+\.?\d*)',          # $69,628.00
        r'price[:\s]+([\d,]+\.?\d*)',    # price: 69628
        r'([\d,]+\.\d{2})\s',           # 69628.00
        r'EUR/USD[:\s]+([\d.]+)',        # EUR/USD: 1.1862
        r'GBP/USD[:\s]+([\d.]+)',        # GBP/USD: 1.2650
        r'USD/JPY[:\s]+([\d.]+)',        # USD/JPY: 149.50
        r'([\d.]+)\s*USD',              # 1.0860 USD
    ]
    for pat in patterns:
        m = re.search(pat, tool_result, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1).replace(",", ""))
            except (ValueError, IndexError):
                continue
    return None


# ── Session Tracker ──────────────────────────────────────────────────

class SessionTracker:
    """
    Tracks market session boundaries and saves open/close prices
    as permanent historical_event nodes.

    Assets are dynamic — passed at init from config, not hardcoded.
    Call tick() from the ingestion daemon loop (every 30s).
    """

    def __init__(self, graph=None, tool_executor=None,
                 ngre_brain=None, assets: Optional[List[str]] = None):
        self.graph = graph
        self.tool_executor = tool_executor
        self.ngre_brain = ngre_brain

        # Build asset routing from dynamic list
        self._assets: List[Tuple[str, str, str, bool]] = []
        if assets:
            try:
                from .quant import build_session_assets
                self._assets = build_session_assets(assets)
            except Exception:
                pass
        # Fallback if nothing configured
        if not self._assets:
            self._assets = [
                ("crypto_price", "bitcoin", "BTC", True),
                ("crypto_price", "ethereum", "ETH", True),
            ]

        # Track which session events we've already recorded today
        # Key: "Asia:open:2026-02-14", Value: True
        self._recorded: Dict[str, bool] = {}

        # Store session open prices so we can compute change% at close
        # Key: "Asia:2026-02-14:BTC", Value: 69500.0
        self._open_prices: Dict[str, float] = {}

        logger.info("[Sessions] Tracking %d assets: %s",
                    len(self._assets),
                    ", ".join(a[2] for a in self._assets))

    def tick(self):
        """Check if any session boundary just passed. Call every ~30s."""
        if not self.graph or not getattr(self.graph, 'is_open', False):
            return
        if not self.tool_executor:
            return

        now_utc = datetime.now(timezone.utc)
        today_str = now_utc.strftime("%Y-%m-%d")
        current_hour = now_utc.hour
        current_min = now_utc.minute

        for session in SESSIONS:
            # Check OPEN boundary: hour == open_hour, within first 5 min
            open_key = f"{session.name}:open:{today_str}"
            if (current_hour == session.open_hour
                    and current_min < 6
                    and open_key not in self._recorded):
                self._record_session_event(session, "open", now_utc,
                                           today_str)
                self._recorded[open_key] = True

            # Check CLOSE boundary: hour == close_hour, within first 5 min
            close_key = f"{session.name}:close:{today_str}"
            if (current_hour == session.close_hour
                    and current_min < 6
                    and close_key not in self._recorded):
                self._record_session_event(session, "close", now_utc,
                                           today_str)
                self._recorded[close_key] = True

        # Cleanup old recorded keys (keep only today + yesterday)
        yesterday_str = (now_utc - timedelta(days=1)).strftime("%Y-%m-%d")
        self._recorded = {
            k: v for k, v in self._recorded.items()
            if today_str in k or yesterday_str in k
        }

    def _record_session_event(self, session: MarketSession,
                              event_type: str,
                              now_utc: datetime,
                              today_str: str):
        """Fetch prices and save historical node."""
        prices = self._fetch_prices()
        if not prices:
            logger.warning("Session %s %s: no prices fetched",
                           session.name, event_type)
            return

        # Store open prices for change% calculation at close
        if event_type == "open":
            for asset_name, price in prices.items():
                key = f"{session.name}:{today_str}:{asset_name}"
                self._open_prices[key] = price

        # Build content text
        lines = [
            f"Market Session {event_type.upper()} — "
            f"{session.name} / {session.region} ({today_str})"
        ]

        for asset_name, price in prices.items():
            if event_type == "open":
                lines.append(f"  {asset_name}: Open ${price:,.2f}")
            else:
                # Close: show open → close with change%
                open_key = f"{session.name}:{today_str}:{asset_name}"
                open_price = self._open_prices.get(open_key)
                if open_price and open_price > 0:
                    change_pct = ((price - open_price) / open_price) * 100
                    sign = "+" if change_pct >= 0 else ""
                    lines.append(
                        f"  {asset_name}: Open ${open_price:,.2f} → "
                        f"Close ${price:,.2f} ({sign}{change_pct:.2f}%)"
                    )
                else:
                    lines.append(f"  {asset_name}: Close ${price:,.2f}")

        content = "\n".join(lines)

        # Create permanent historical_event node
        node_id = f"session:{session.name.lower()}:{event_type}:{today_str}"

        # Compute embedding
        embedding = None
        if self.ngre_brain is not None:
            try:
                embedding = self.ngre_brain.compute_embedding(content)
            except Exception:
                pass

        self.graph.add_node(node_id, node_type="historical_event",
                            properties={
            "content": content,
            "source": "session_tracker",
            "session": session.name,
            "region": session.region,
            "event_type": event_type,
            "date": today_str,
            "timestamp": now_utc.isoformat(),
            "prices": {name: price for name, price in prices.items()},
            "confidence": 1.0,
        },
            embedding=embedding,
            text_summary=content[:200],
            confidence=1.0,
            source="session_tracker",
        )

        logger.info("[Sessions] %s %s recorded — %d assets, node=%s",
                    session.name, event_type.upper(), len(prices), node_id)

    def _fetch_prices(self) -> Dict[str, float]:
        """Fetch current prices for all tracked assets.

        Skips assets whose markets are closed (weekends/holidays).
        Crypto (24/7) is always fetched.
        """
        prices = {}
        for tool_name, query, display_name, is_crypto in self._assets:
            # Skip non-crypto assets when their market is closed
            if not is_crypto:
                try:
                    from .quant import classify_asset, is_market_open
                    at = classify_asset(display_name)
                    mkt = is_market_open(at.asset_type)
                    if not mkt["open"]:
                        continue
                except Exception:
                    pass
            try:
                result = self.tool_executor.call(tool_name, query)
                if result:
                    price = _extract_price(str(result))
                    if price is not None:
                        prices[display_name] = price
            except Exception as e:
                logger.debug("Session price fetch %s: %s", display_name, e)
        return prices

    def status(self) -> Dict[str, Any]:
        """Return tracker status."""
        now_utc = datetime.now(timezone.utc)
        current_hour = now_utc.hour

        active_sessions = []
        for s in SESSIONS:
            if s.open_hour <= current_hour < s.close_hour:
                active_sessions.append(s.name)

        return {
            "active_sessions": active_sessions,
            "recorded_today": len([k for k in self._recorded
                                   if now_utc.strftime("%Y-%m-%d") in k]),
            "open_prices_cached": len(self._open_prices),
            "tracked_assets": [a[2] for a in self._assets],
        }

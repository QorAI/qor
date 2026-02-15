"""
QOR Trading Engine — AI Automated Trading (Binance Spot Demo Mode)
====================================================================
Autonomous trading bot that runs every 5 minutes and:
- Analyzes all symbols via multi-timeframe technical analysis
- Opens positions when confluence is strong (>= 3/6 TFs bullish)
- DCA (Dollar Cost Average) into losing positions at support levels
- Partial take profit: sells portions at TP1, rest at TP2
- Trailing stop loss: ratchets SL up using ATR + support levels
- Break-even protection: moves SL to entry after TP1 hit
- Closes on full trend reversal (all TFs flip bearish)
- Learns from history: reduces size on losing symbols, skips bad patterns

Architecture:
  BinanceClient      — Authenticated Binance API (HMAC-SHA256)
  TradeStore          — Parquet database of all trades (hash-chained)
  PositionManager     — Full AI decision engine (every 5 min)
  TradingEngine       — Background thread loop orchestrator

Every 5-minute tick per symbol:
  1. Fetch multi-TF analysis (weekly → 5min)
  2. If NO position:
     - Check entry conditions (confluence, R:R, cooldown, history)
     - If valid → BUY
  3. If HAS position:
     a. Price >= TP1 and haven't sold partial → PARTIAL_TP (sell %)
     b. Price >= TP2 → SELL (close all)
     c. Price <= SL → SELL (stop loss hit)
     d. All TFs flipped bearish → SELL (trend reversal)
     e. Price dropped N% + at support + DCA budget left → DCA (add to position)
     f. Trailing stop: if price above entry, ratchet SL up
     g. After partial TP1: move SL to breakeven
     h. Otherwise → HOLD
"""

import hashlib
import hmac
import json
import logging
import os
import re
import socket
import threading
import time
import urllib.parse
import urllib.request
import urllib.error
import uuid
from collections import deque
from datetime import datetime, timezone, timedelta

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# ==============================================================================
# CortexAnalyzer — CORTEX Brain for Trading
# ==============================================================================
# Uses CortexBrain from qor.cortex (brain-inspired sequential intelligence).
#
# CORTEX = CfC + Mamba + Fusion
#   C — Continuous-time reflex    O — Observation of deep history
#   R — Reasoning layer           T — Temporal memory
#   E — Execution decision        X — eXtended architecture
#
# Architecture (every 5-min tick):
#   Observation (S4, 500+ candles, cached) → context vector
#   Reflex (every tick, hidden state per symbol) → signal
#   Reasoning → BUY/SELL/HOLD decision

try:
    import torch
    from qor.cortex import CortexBrain
    _HAS_CORTEX = True
except ImportError:
    _HAS_CORTEX = False

# HMM regime detection (from qor.quant)
try:
    from qor.quant import MarketHMM, HMMSignal
    _HAS_HMM = True
except ImportError:
    _HAS_HMM = False


class CortexAnalyzer:
    """Trading-specific wrapper around CortexBrain (from qor.cortex).

    Converts TA indicators into a 22-dim feature vector, feeds through
    CORTEX brain, outputs signal/label for trading decisions.

    Feature vector (22 inputs):
         0: RSI(14)       1: RSI(6)        2: RSI momentum   3: EMA21 dev
         4: EMA50 dev     5: EMA200 dev    6: MACD hist/ATR  7: BB %B
         8: ATR % price   9: Keltner pos  10: Relative vol  11: OBV direction
        12: VWAP dev     13: ADX norm     14: TF confluence  15: Funding rate
        16: OI change    17: Body ratio   18: Upper wick     19: Lower wick
        20: Poly sentiment (up_prob)       21: Fear & Greed (0-1)
        22: VP POC position (0-1)          23: VP density (0-1)
    """

    INPUT_SIZE = 24
    OBS_DIM = 32
    REFLEX_NEURONS = 24
    HISTORY_LEN = 500
    OBS_INTERVAL = 360   # 6 min

    def __init__(self):
        if not _HAS_CORTEX:
            raise ImportError("CORTEX analyzer requires: pip install ncps torch")

        self._brain = CortexBrain(
            input_size=self.INPUT_SIZE, output_size=1,
            observation_dim=self.OBS_DIM, reflex_neurons=self.REFLEX_NEURONS,
            reflex_output=8, history_len=self.HISTORY_LEN,
            observation_interval=self.OBS_INTERVAL,
        )
        self._prev = {}  # Per-symbol previous tick for momentum
        self._last_results = {}  # Per-symbol last analyze() result
        self._last_features = {}  # Per-symbol last feature tensor for CORTEX training

    def _build_features(self, analysis: dict, symbol: str) -> torch.Tensor:
        """Convert parsed TA analysis into normalized 20-dim feature vector.

        Matches Trading.md specification for CORTEX brain input.
        """
        current = analysis.get("current", 0.0)
        rsi = analysis.get("rsi", 50.0)
        rsi6 = analysis.get("rsi6", 50.0)
        ema21 = analysis.get("ema21", current)
        ema50 = analysis.get("ema50", current)
        ema200 = analysis.get("ema200", current)
        atr = analysis.get("atr_daily", analysis.get("atr", 0.0))
        bullish = analysis.get("bullish_tfs", 0)
        total = analysis.get("total_tfs", 1) or 1
        bb_upper = analysis.get("bb_upper", current * 1.02)
        bb_lower = analysis.get("bb_lower", current * 0.98)
        macd_hist = analysis.get("macd_hist", 0.0)
        adx = analysis.get("adx", 25.0)
        keltner_pos = analysis.get("keltner_pos", 0.0)
        rel_vol = analysis.get("rel_vol", 0.0)
        obv_dir = analysis.get("obv_dir", 0.0)
        vwap = analysis.get("vwap", current)
        funding_rate = analysis.get("funding_rate", 0.0)
        oi_change = analysis.get("oi_change", 0.0)
        body_ratio = analysis.get("body_ratio", 0.5)
        upper_wick = analysis.get("upper_wick", 0.25)
        lower_wick = analysis.get("lower_wick", 0.25)

        # --- Normalize all features to ~(-1, 1) or (0, 1) range ---
        # 0: RSI(14) normalized (0-1)
        f_rsi14 = rsi / 100.0
        # 1: RSI(6) normalized (0-1)
        f_rsi6 = rsi6 / 100.0
        # 2: RSI momentum (delta from previous tick)
        prev = self._prev.get(symbol, {"rsi": rsi, "price": current})
        f_rsi_mom = max(-1.0, min(1.0, (rsi - prev["rsi"]) / 100.0))
        # 3: Price vs EMA21 % deviation
        f_ema21 = max(-0.1, min(0.1, (current - ema21) / max(current, 1)))
        # 4: Price vs EMA50 % deviation
        f_ema50 = max(-0.1, min(0.1, (current - ema50) / max(current, 1)))
        # 5: Price vs EMA200 % deviation (wider range)
        f_ema200 = max(-0.2, min(0.2, (current - ema200) / max(current, 1)))
        # 6: MACD histogram / ATR (normalized momentum)
        f_macd = max(-1.0, min(1.0, macd_hist / max(atr, 1)))
        # 7: Bollinger Band %B position (-1 to 1)
        bb_range = max(bb_upper - bb_lower, 0.01)
        f_bb = max(-1.0, min(1.0, (current - bb_lower) / bb_range * 2 - 1))
        # 8: ATR % of price (volatility, 0-1)
        f_atr = min((atr / max(current, 1)) * 100 / 5.0, 1.0)
        # 9: Keltner channel position (-1 to 1)
        f_keltner = max(-1.0, min(1.0, keltner_pos))
        # 10: Relative volume (0-3, clamped)
        f_rel_vol = min(rel_vol / 3.0, 1.0)
        # 11: OBV direction (-1 to 1)
        f_obv = max(-1.0, min(1.0, obv_dir))
        # 12: Price vs VWAP % deviation — scaled to (-1, 1) range
        f_vwap = max(-1.0, min(1.0, (current - vwap) / max(current, 1) * 20))
        # 13: ADX normalized (0-1, 100 max)
        f_adx = min(adx / 100.0, 1.0)
        # 14: TF confluence ratio (0-1)
        f_tf = bullish / total
        # 15: Funding rate — scaled to (-1, 1) range
        f_funding = max(-1.0, min(1.0, funding_rate * 100))
        # 16: OI change — scaled to (-1, 1) range
        f_oi = max(-1.0, min(1.0, oi_change * 10))
        # 17: Candlestick body/range ratio (0-1)
        f_body = max(0.0, min(1.0, body_ratio))
        # 18: Upper wick ratio (0-1)
        f_uwk = max(0.0, min(1.0, upper_wick))
        # 19: Lower wick ratio (0-1)
        f_lwk = max(0.0, min(1.0, lower_wick))
        # 20: Polymarket sentiment — up probability (0-1, 0.5 = neutral)
        f_poly_sentiment = max(0.0, min(1.0,
            analysis.get("poly_up_prob", 0.5)))
        # 21: Fear & Greed index (0-1, 0.5 = neutral)
        f_fear_greed = max(0.0, min(1.0,
            analysis.get("fear_greed_value", 50) / 100.0))
        # 22: VP POC position — where price sits in value area (0=VAL, 0.5≈POC, 1=VAH)
        vp_vah = analysis.get("vp_vah", 0)
        vp_val = analysis.get("vp_val", 0)
        va_range = vp_vah - vp_val
        if analysis.get("vp_available") and va_range > 0:
            f_vp_pos = max(0.0, min(1.0, (current - vp_val) / va_range))
        else:
            f_vp_pos = 0.5  # neutral default
        # 23: VP density — 1.0 near HVN, 0.0 near LVN, 0.5 neutral
        hvn = analysis.get("vp_hvn", [])
        lvn = analysis.get("vp_lvn", [])
        if analysis.get("vp_available") and (hvn or lvn):
            near_hvn = any(abs(current - h) / max(current, 1) < 0.005 for h in hvn)
            near_lvn = any(abs(current - l) / max(current, 1) < 0.005 for l in lvn)
            if near_hvn and not near_lvn:
                f_vp_dens = 1.0
            elif near_lvn and not near_hvn:
                f_vp_dens = 0.0
            else:
                f_vp_dens = 0.5
        else:
            f_vp_dens = 0.5  # neutral default

        self._prev[symbol] = {"rsi": rsi, "price": current}

        features_tensor = torch.tensor([[
            f_rsi14, f_rsi6, f_rsi_mom, f_ema21, f_ema50, f_ema200,
            f_macd, f_bb, f_atr, f_keltner, f_rel_vol, f_obv,
            f_vwap, f_adx, f_tf, f_funding, f_oi, f_body, f_uwk, f_lwk,
            f_poly_sentiment, f_fear_greed, f_vp_pos, f_vp_dens,
        ]], dtype=torch.float32)
        self._last_features[symbol] = features_tensor
        return features_tensor

    def get_last_features(self, symbol: str):
        """Return the last computed feature tensor for a symbol (for training)."""
        return self._last_features.get(symbol)

    @torch.no_grad()
    def analyze(self, analysis: dict, symbol: str) -> dict:
        """Run CORTEX analysis. Returns signal/confidence/label."""
        x = self._build_features(analysis, symbol)
        raw = self._brain(x, instance_id=symbol)

        # Untrained model → return neutral (no influence on trades)
        if not self._brain._trained:
            result = {
                "signal": 0.0, "confidence": 0.0, "label": "NEUTRAL",
                "history_candles": len(self._brain._history.get(symbol, [])),
                "trained": False,
            }
            self._last_results[symbol] = result
            return result

        signal = torch.tanh(raw).item()
        # Calibrated confidence: sqrt stretches weak signals into meaningful
        # range (0.18 signal → 42% conf instead of 18%), caps at 95%
        confidence = min(0.95, abs(signal) ** 0.5)

        if signal > 0.5:       label = "STRONG_BUY"
        elif signal > 0.15:    label = "BUY"
        elif signal < -0.5:    label = "STRONG_SELL"
        elif signal < -0.15:   label = "SELL"
        else:                  label = "NEUTRAL"

        result = {
            "signal": round(signal, 4),
            "confidence": round(confidence, 4),
            "label": label,
            "history_candles": len(self._brain._history.get(symbol, [])),
            "trained": True,
        }
        self._last_results[symbol] = result
        return result

    def reset_symbol(self, symbol: str):
        """Clear all state for a symbol."""
        self._brain.reset_instance(symbol)
        self._prev.pop(symbol, None)

    def train_batch(self, features: list, targets: list,
                    epochs: int = 10, lr: float = 1e-3) -> dict:
        """Train CORTEX on historical (feature_vector, target_signal) pairs."""
        return self._brain.train_batch(features, targets, epochs=epochs, lr=lr)

    def save(self, path: str):
        self._brain.save(path)

    def load(self, path: str) -> bool:
        return self._brain.load(path)

    def status(self) -> dict:
        return self._brain.status()


# ==============================================================================
# ExchangeClient — Abstract base for all exchange/broker clients
# ==============================================================================

class ExchangeClient:
    """Abstract exchange client interface.

    Any exchange/broker that implements these methods can plug into
    PositionManager, TradingEngine, and FuturesEngine.  All methods use
    the same signatures regardless of exchange.

    Core methods (required for spot):
        get_price, get_balance, get_account, place_order, round_qty, get_lot_size

    Futures methods (optional — implement for futures support):
        get_positions, get_position, get_mark_price, get_open_orders,
        set_leverage, set_margin_type, set_hedge_mode, set_multi_asset_mode

    Implementations: BinanceClient, BinanceFuturesClient, UpstoxClient, etc.
    """

    name: str = "base"          # Exchange identifier
    quote: str = "USD"          # Default quote currency
    supports_futures: bool = False  # Set True if futures methods are implemented

    def __init__(self):
        """Base init for exchange clients. Subclasses should call super().__init__()."""
        pass

    # ── Core methods (required) ────────────────────────────────────────

    def get_price(self, symbol: str) -> float:
        """Get current price for a trading pair (e.g. 'BTCUSDT', 'AAPL')."""
        raise NotImplementedError

    def get_balance(self, asset: str) -> float:
        """Get available (free) balance for an asset."""
        raise NotImplementedError

    def get_account(self) -> dict:
        """Get full account info including all balances.
        Must return dict with 'balances' key → list of {asset, free, locked}.
        """
        raise NotImplementedError

    def place_order(self, symbol: str, side: str, quantity: float,
                    order_type: str = "MARKET", price: float = None) -> dict:
        """Place an order. side = 'BUY' or 'SELL'."""
        raise NotImplementedError

    def round_qty(self, symbol: str, quantity: float) -> float:
        """Round quantity to exchange's lot size rules."""
        raise NotImplementedError

    def get_lot_size(self, symbol: str) -> dict:
        """Get lot size info: {min_qty, max_qty, step_size}."""
        raise NotImplementedError

    def cancel_order(self, symbol: str, order_id) -> dict:
        """Cancel an open order."""
        raise NotImplementedError

    def format_pair(self, symbol: str) -> str:
        """Convert base symbol to exchange pair format.
        e.g. Binance: 'BTC' → 'BTCUSDT', Alpaca: 'AAPL' → 'AAPL'
        """
        return symbol

    # ── Futures methods (optional — override for futures support) ──────

    def get_positions(self) -> list:
        """Get all open futures positions.
        Returns list of dicts with: symbol, positionAmt, entryPrice,
        markPrice, unRealizedProfit, leverage, marginType.
        """
        return []

    def get_position(self, symbol: str) -> dict:
        """Get position for a specific symbol."""
        for p in self.get_positions():
            if p.get("symbol") == symbol:
                return p
        return {}

    def get_mark_price(self, symbol: str) -> dict:
        """Get mark price and funding rate.
        Returns: {markPrice, lastFundingRate, nextFundingTime}.
        """
        return {"markPrice": self.get_price(symbol),
                "lastFundingRate": "0", "nextFundingTime": 0}

    def get_open_orders(self, symbol: str = None) -> list:
        """Get open orders, optionally filtered by symbol."""
        return []

    def set_leverage(self, symbol: str, leverage: int) -> dict:
        """Set leverage for a symbol. No-op if exchange doesn't support it."""
        return {}

    def set_margin_type(self, symbol: str, margin_type: str) -> dict:
        """Set margin type (ISOLATED/CROSSED). No-op if not supported."""
        return {}

    def set_hedge_mode(self, enabled: bool) -> dict:
        """Enable/disable hedge mode. No-op if not supported."""
        return {}

    def set_multi_asset_mode(self, enabled: bool) -> dict:
        """Enable/disable multi-asset margin. No-op if not supported."""
        return {}


def create_exchange_client(exchange_name: str, api_key: str, api_secret: str,
                           passphrase: str = "", testnet: bool = True,
                           base_url: str = "", **kwargs) -> 'ExchangeClient':
    """Factory: create the right exchange client by name.

    Args:
        exchange_name: "binance", "alpaca", "oanda", "okx", "bybit", etc.
        api_key, api_secret: Credentials
        passphrase: Required by some exchanges (OKX, Coinbase)
        testnet: Use sandbox/demo mode
        base_url: Override default URL

    Returns:
        ExchangeClient instance ready to trade
    """
    name = exchange_name.lower().replace("-", "_").replace(" ", "_")

    if name == "binance":
        return BinanceClient(api_key, api_secret, testnet=testnet)

    if name == "upstox":
        from qor.upstox import UpstoxClient
        return UpstoxClient(api_key, api_secret, testnet=testnet,
                            access_token=kwargs.get("access_token", ""))

    # Future implementations — uncomment as they're built:
    # if name == "alpaca":
    #     return AlpacaClient(api_key, api_secret, testnet=testnet, base_url=base_url)
    # if name in ("oanda",):
    #     return OandaClient(api_key, api_secret, testnet=testnet, base_url=base_url)
    # if name == "okx":
    #     return OKXClient(api_key, api_secret, passphrase=passphrase, testnet=testnet)
    # if name == "bybit":
    #     return BybitClient(api_key, api_secret, testnet=testnet)
    # if name == "coinbase":
    #     return CoinbaseClient(api_key, api_secret, passphrase=passphrase, testnet=testnet)

    raise ValueError(f"Unsupported exchange: {exchange_name}. "
                     f"Supported: binance, upstox. Coming soon: alpaca, oanda, okx, bybit, coinbase")


# ==============================================================================
# BinanceClient — Authenticated API with HMAC-SHA256
# ==============================================================================

class BinanceClient(ExchangeClient):
    """Binance Spot API client with HMAC-SHA256 signing.

    Supports Demo Mode (demo-api.binance.com) and production (api.binance.com).
    Demo mode uses real market data but simulated balances — no real money.
    """

    DEMO_URL = "https://demo-api.binance.com"
    PRODUCTION_URL = "https://api.binance.com"

    name = "binance"
    quote = "USDT"

    # Spot API weight limit: 1200/min
    _WEIGHT_LIMIT = 1200

    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = self.DEMO_URL if testnet else self.PRODUCTION_URL
        self._lot_cache = {}  # symbol -> {minQty, maxQty, stepSize}
        self._time_offset = 0  # ms offset: server_time - local_time
        self._time_synced = False
        # API rate limit tracking
        self._api_weight = 0
        self._weight_window_start = time.time()

    def format_pair(self, symbol: str) -> str:
        """BTC → BTCUSDT, ETH → ETHUSDT."""
        s = symbol.upper()
        if s.endswith(self.quote):
            return s
        return f"{s}{self.quote}"

    def _track_weight(self, weight: int = 1):
        """Track API weight usage and sleep if approaching limit."""
        now = time.time()
        if now - self._weight_window_start > 60:
            self._api_weight = 0
            self._weight_window_start = now
        self._api_weight += weight
        if self._api_weight > self._WEIGHT_LIMIT - 200:  # leave 200 headroom
            sleep_time = 60 - (now - self._weight_window_start)
            if sleep_time > 0:
                logger.warning(f"[Binance] Rate limit approaching "
                               f"({self._api_weight}/{self._WEIGHT_LIMIT}), "
                               f"sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
                self._api_weight = 0
                self._weight_window_start = time.time()

    def _sync_time(self):
        """Sync local clock with Binance server to avoid recvWindow errors."""
        for attempt in range(3):
            try:
                url = f"{self.base_url}/api/v3/time"
                req = urllib.request.Request(url, headers={"User-Agent": "QOR-Trading/1.0"})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    server_time = json.loads(resp.read().decode())["serverTime"]
                local_time = int(time.time() * 1000)
                self._time_offset = server_time - local_time
                self._time_synced = True
                if abs(self._time_offset) > 1000:
                    logger.info(f"[Binance] Clock offset: {self._time_offset}ms")
                return
            except Exception as e:
                if attempt < 2:
                    time.sleep(1)
                else:
                    logger.warning(f"[Binance] Time sync failed after 3 attempts: {e}")

    def _sign(self, params: dict) -> str:
        """HMAC-SHA256 sign query string."""
        if not self._time_synced:
            self._sync_time()
        params["timestamp"] = int(time.time() * 1000) + self._time_offset
        params["recvWindow"] = 10000
        qs = urllib.parse.urlencode(params)
        sig = hmac.new(
            self.api_secret.encode(), qs.encode(), hashlib.sha256
        ).hexdigest()
        return f"{qs}&signature={sig}"

    def _request(self, method: str, endpoint: str, params: dict = None,
                 signed: bool = True, _retry: bool = False) -> dict:
        """Make API request with auto-retry on timeout and timestamp errors."""
        params = params or {}
        max_attempts = 1 if _retry else 2  # retry once on timeout

        # Track API weight before request (most endpoints = weight 1-10)
        weight = 10 if "account" in endpoint else 1
        self._track_weight(weight)

        for attempt in range(max_attempts):
            req_params = dict(params)  # fresh copy each attempt
            url = f"{self.base_url}{endpoint}"

            if signed:
                qs = self._sign(req_params)
            else:
                qs = urllib.parse.urlencode(req_params) if req_params else ""

            if method == "GET":
                url = f"{url}?{qs}" if qs else url
                req = urllib.request.Request(url, method="GET")
            elif method == "DELETE":
                url = f"{url}?{qs}" if qs else url
                req = urllib.request.Request(url, method="DELETE")
            else:
                req = urllib.request.Request(
                    url, data=qs.encode(), method=method,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )

            req.add_header("X-MBX-APIKEY", self.api_key)
            req.add_header("User-Agent", "QOR-Trading/1.0")

            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    # Track actual weight from response headers
                    used = resp.getheader("X-MBX-USED-WEIGHT-1M")
                    if used:
                        self._api_weight = int(used)
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as e:
                body = e.read().decode() if e.fp else ""
                # Auto-retry on timestamp error: re-sync clock and try once more
                if e.code == 400 and "-1021" in body and signed and not _retry:
                    logger.warning("[Binance] Timestamp error, re-syncing clock...")
                    self._sync_time()
                    return self._request(method, endpoint, params, signed=True, _retry=True)
                logger.error(f"[Binance] {method} {endpoint}: HTTP {e.code} — {body}")
                raise RuntimeError(f"Binance API error {e.code}: {body}")
            except (socket.timeout, urllib.error.URLError, OSError) as e:
                is_timeout = "timed out" in str(e).lower() or isinstance(e, socket.timeout)
                if is_timeout and attempt < max_attempts - 1:
                    logger.warning(f"[Binance] {method} {endpoint}: timeout, retrying ({attempt+1}/{max_attempts})...")
                    self._sync_time()  # re-sync clock after timeout
                    time.sleep(2)
                    continue
                logger.error(f"[Binance] {method} {endpoint}: {e}")
                raise
            except Exception as e:
                logger.error(f"[Binance] {method} {endpoint}: {e}")
                raise

    def get_account(self) -> dict:
        """Get account info including balances."""
        return self._request("GET", "/api/v3/account")

    def get_balance(self, asset: str) -> float:
        """Get free balance for a specific asset."""
        account = self.get_account()
        for b in account.get("balances", []):
            if b["asset"] == asset.upper():
                return float(b["free"])
        return 0.0

    def get_price(self, symbol: str) -> float:
        """Get current price for a trading pair (e.g. BTCUSDT)."""
        data = self._request(
            "GET", "/api/v3/ticker/price",
            params={"symbol": symbol}, signed=False,
        )
        return float(data.get("price", 0))

    def get_klines(self, symbol: str, interval: str = "1h",
                   limit: int = 1500, start_time: int = None,
                   end_time: int = None) -> list:
        """Fetch historical klines (candlesticks) from Binance Spot.

        Returns list of klines in Binance format:
        [open_time, open, high, low, close, volume, close_time,
         quote_volume, trades, taker_buy_base, taker_buy_quote, ignore]
        """
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 1500),
        }
        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time

        weight = 1 if limit <= 100 else 2 if limit <= 500 else 5 if limit <= 1000 else 10
        self._track_weight(weight)
        return self._request("GET", "/api/v3/klines", params, signed=False)

    def place_order(self, symbol: str, side: str, quantity: float,
                    order_type: str = "MARKET", price: float = None) -> dict:
        """Place a spot order."""
        params = {
            "symbol": symbol,
            "side": side.upper(),
            "type": order_type,
            "quantity": f"{quantity:.8f}".rstrip("0").rstrip("."),
        }
        if order_type == "MARKET":
            params["newOrderRespType"] = "RESULT"
        if order_type == "LIMIT" and price is not None:
            params["timeInForce"] = "GTC"
            params["price"] = f"{price:.8f}".rstrip("0").rstrip(".")

        return self._request("POST", "/api/v3/order", params)

    def cancel_order(self, symbol: str, order_id: int) -> dict:
        """Cancel an open order."""
        return self._request(
            "DELETE", "/api/v3/order",
            params={"symbol": symbol, "orderId": order_id},
        )

    def get_open_orders(self, symbol: str = None) -> list:
        """Get all open orders."""
        params = {}
        if symbol:
            params["symbol"] = symbol
        return self._request("GET", "/api/v3/openOrders", params)

    def get_exchange_info(self, symbol: str) -> dict:
        """Get exchange info for a symbol (lot size, tick size, etc.)."""
        data = self._request(
            "GET", "/api/v3/exchangeInfo",
            params={"symbol": symbol}, signed=False,
        )
        for s in data.get("symbols", []):
            if s["symbol"] == symbol:
                return s
        return {}

    def get_lot_size(self, symbol: str) -> dict:
        """Get LOT_SIZE filter for a symbol (min/max qty, step). Cached."""
        if symbol in self._lot_cache:
            return self._lot_cache[symbol]
        info = self.get_exchange_info(symbol)
        for f in info.get("filters", []):
            if f["filterType"] == "LOT_SIZE":
                lot = {
                    "minQty": float(f["minQty"]),
                    "maxQty": float(f["maxQty"]),
                    "stepSize": float(f["stepSize"]),
                }
                self._lot_cache[symbol] = lot
                return lot
        return {"minQty": 0.00001, "maxQty": 9999999, "stepSize": 0.00001}

    def round_qty(self, symbol: str, quantity: float) -> float:
        """Round quantity to lot step size."""
        try:
            lot = self.get_lot_size(symbol)
            step = lot["stepSize"]
            if step > 0:
                quantity = int(quantity / step) * step
        except Exception:
            pass
        return quantity


# ==============================================================================
# TradeStore — Parquet Trade History Database (hash-chained)
# ==============================================================================

TRADE_SCHEMA = pa.schema([
    ("trade_id",      pa.string()),
    ("symbol",        pa.string()),
    ("side",          pa.string()),
    ("entry_price",   pa.float64()),   # Average entry (recalculated on DCA)
    ("exit_price",    pa.float64()),
    ("quantity",      pa.float64()),   # Total quantity (increases on DCA)
    ("stop_loss",     pa.float64()),
    ("take_profit",   pa.float64()),   # TP1
    ("take_profit2",  pa.float64()),   # TP2
    ("status",        pa.string()),    # open, closed_tp, closed_sl, closed_manual, closed_reversal
    ("pnl",           pa.float64()),
    ("pnl_pct",       pa.float64()),
    ("strategy",      pa.string()),
    ("entry_reason",  pa.string()),
    ("exit_reason",   pa.string()),
    ("entry_time",    pa.int64()),
    ("exit_time",     pa.int64()),
    ("dca_count",     pa.int64()),     # How many DCA adds have been done
    ("tp1_hit",       pa.bool_()),     # Whether TP1 partial has been taken
    ("original_qty",  pa.float64()),   # Quantity before any partial sells
    ("cost_basis",    pa.float64()),   # Total USDT spent (for avg price calc)
    ("data_hash",     pa.string()),
    ("prev_hash",     pa.string()),
])


def format_trade_summary(trade: dict, engine_type: str = "spot") -> str:
    """Format a closed trade as a human-readable summary for AI learning.

    This is saved to MemoryStore so the AI can learn from wins/losses,
    answer questions about trading history, and improve future decisions.
    """
    symbol = trade.get("symbol", "?")
    direction = trade.get("direction", trade.get("side", "?"))
    entry = trade.get("entry_price", 0)
    exit_p = trade.get("exit_price", 0)
    pnl = trade.get("pnl", 0)
    pnl_pct = trade.get("pnl_pct", 0)
    status = trade.get("status", "?")
    entry_reason = trade.get("entry_reason", "")
    exit_reason = trade.get("exit_reason", "")
    strategy = trade.get("strategy", "")
    dca_count = trade.get("dca_count", 0)
    leverage = trade.get("leverage", 1)
    qty = trade.get("quantity", 0)
    tp1_hit = trade.get("tp1_hit", False)
    tag = "WIN" if pnl > 0 else "LOSS" if pnl < 0 else "BREAKEVEN"

    # Calculate hold duration
    duration = ""
    entry_t = trade.get("entry_time", 0)
    exit_t = trade.get("exit_time", 0)
    if entry_t and exit_t and exit_t > entry_t:
        dur_min = (exit_t - entry_t) / (1000000 * 60)  # microseconds to minutes
        if dur_min < 60:
            duration = f"{dur_min:.0f}min"
        elif dur_min < 1440:
            duration = f"{dur_min / 60:.1f}h"
        else:
            duration = f"{dur_min / 1440:.1f}d"

    lines = [
        f"{engine_type.upper()} Trade {tag}: {symbol} {direction}",
        f"  Entry: ${entry:,.2f} -> Exit: ${exit_p:,.2f}",
        f"  P&L: ${pnl:+,.2f} ({pnl_pct:+.2f}%)",
    ]
    if leverage > 1:
        lines.append(f"  Leverage: {leverage}x | Qty: {qty}")
    if duration:
        lines.append(f"  Duration: {duration}")
    lines.append(f"  Status: {status}")
    if entry_reason:
        lines.append(f"  Entry reason: {entry_reason}")
    if exit_reason:
        lines.append(f"  Exit reason: {exit_reason}")
    if strategy:
        lines.append(f"  Strategy: {strategy}")
    if dca_count > 0:
        lines.append(f"  DCA adds: {dca_count}")
    if tp1_hit:
        lines.append(f"  Partial TP1: taken")
    return "\n".join(lines)


class TradeStore:
    """Parquet-based trade history with hash chain for integrity."""

    def __init__(self, path: str, engine_type: str = "spot"):
        self.path = path
        self.engine_type = engine_type  # "spot", "upstox", "alpaca", etc.
        self.trades: dict = {}  # trade_id -> dict
        self._chain_head = ""
        self._dirty = False
        self._lock = threading.Lock()  # Protects self.trades across threads
        self.on_close = None  # Callback: on_close(trade_dict, engine_type)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._load()

    def _load(self):
        """Load trades from Parquet."""
        if not os.path.exists(self.path):
            return
        try:
            table = pq.read_table(self.path)
            for i in range(table.num_rows):
                trade = {}
                for col_name in TRADE_SCHEMA.names:
                    if col_name in table.schema.names:
                        val = table.column(col_name)[i].as_py()
                        trade[col_name] = val if val is not None else self._default(col_name)
                    else:
                        trade[col_name] = self._default(col_name)
                self.trades[trade["trade_id"]] = trade
                if trade.get("data_hash"):
                    self._chain_head = trade["data_hash"]
            logger.info(f"[TradeStore] Loaded {len(self.trades)} trades from {self.path}")
        except Exception as e:
            logger.warning(f"[TradeStore] Failed to load {self.path}: {e}")

    def _default(self, col_name):
        """Default value for a column."""
        field = TRADE_SCHEMA.field(col_name)
        if pa.types.is_float64(field.type):
            return 0.0
        if pa.types.is_int64(field.type):
            return 0
        if pa.types.is_boolean(field.type):
            return False
        return ""

    def _hash(self, content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()

    def _now_us(self) -> int:
        return int(datetime.now(timezone.utc).timestamp() * 1_000_000)

    def open_trade(self, symbol: str, side: str, entry_price: float,
                   quantity: float, stop_loss: float, take_profit: float,
                   reason: str, strategy: str = "multi_tf_trend",
                   take_profit2: float = 0.0) -> str:
        """Open a new trade. Returns trade_id."""
        trade_id = str(uuid.uuid4())[:12]
        prev_hash = self._chain_head
        content = f"{trade_id}:{symbol}:{side}:{entry_price}:{quantity}"
        data_hash = self._hash(content)
        self._chain_head = data_hash
        cost_basis = entry_price * quantity

        trade = {
            "trade_id": trade_id,
            "symbol": symbol,
            "side": side,
            "entry_price": entry_price,
            "exit_price": 0.0,
            "quantity": quantity,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "take_profit2": take_profit2,
            "status": "open",
            "pnl": 0.0,
            "pnl_pct": 0.0,
            "strategy": strategy,
            "entry_reason": reason,
            "exit_reason": "",
            "entry_time": self._now_us(),
            "exit_time": 0,
            "dca_count": 0,
            "tp1_hit": False,
            "original_qty": quantity,
            "cost_basis": cost_basis,
            "data_hash": data_hash,
            "prev_hash": prev_hash,
        }
        self.trades[trade_id] = trade
        self._dirty = True
        logger.info(f"[Trade] OPEN {side} {symbol}: qty={quantity:.6f} "
                     f"entry=${entry_price:,.2f} SL=${stop_loss:,.2f} "
                     f"TP1=${take_profit:,.2f} TP2=${take_profit2:,.2f}")
        return trade_id

    def add_dca(self, trade_id: str, add_price: float, add_qty: float):
        """DCA: add to position, recalculate average entry."""
        trade = self.trades.get(trade_id)
        if not trade or trade["status"] != "open":
            return
        old_cost = trade["cost_basis"]
        new_cost = add_price * add_qty
        total_cost = old_cost + new_cost
        total_qty = trade["quantity"] + add_qty
        trade["entry_price"] = total_cost / total_qty  # new average
        trade["quantity"] = total_qty
        trade["cost_basis"] = total_cost
        trade["dca_count"] = trade.get("dca_count", 0) + 1
        trade["entry_reason"] += f" | DCA#{trade['dca_count']} @${add_price:,.2f}"
        self._dirty = True
        logger.info(f"[Trade] DCA #{trade['dca_count']} {trade['symbol']}: "
                     f"+{add_qty:.6f} @${add_price:,.2f} → "
                     f"avg=${trade['entry_price']:,.2f} total={total_qty:.6f}")

    def partial_close(self, trade_id: str, sell_qty: float, sell_price: float,
                      reason: str):
        """Sell a portion of the position (partial TP)."""
        trade = self.trades.get(trade_id)
        if not trade or trade["status"] != "open":
            return
        trade["quantity"] -= sell_qty
        trade["tp1_hit"] = True
        # Realize partial P&L into the reason log
        partial_pnl = (sell_price - trade["entry_price"]) * sell_qty
        trade["entry_reason"] += (f" | Partial @${sell_price:,.2f} "
                                  f"qty={sell_qty:.6f} pnl=${partial_pnl:+,.2f}")
        self._dirty = True
        logger.info(f"[Trade] PARTIAL {trade['symbol']}: sold {sell_qty:.6f} "
                     f"@${sell_price:,.2f} pnl=${partial_pnl:+,.2f} — {reason}")

    def close_trade(self, trade_id: str, exit_price: float,
                    status: str, reason: str):
        """Close a trade and calculate P&L."""
        trade = self.trades.get(trade_id)
        if not trade or trade["status"] != "open":
            return

        trade["exit_price"] = exit_price
        trade["status"] = status
        trade["exit_reason"] = reason
        trade["exit_time"] = self._now_us()

        if trade["side"] == "BUY":
            trade["pnl"] = (exit_price - trade["entry_price"]) * trade["quantity"]
            if trade["entry_price"] > 0:
                trade["pnl_pct"] = ((exit_price / trade["entry_price"]) - 1) * 100
        else:
            trade["pnl"] = (trade["entry_price"] - exit_price) * trade["quantity"]
            if exit_price > 0:
                trade["pnl_pct"] = ((trade["entry_price"] / exit_price) - 1) * 100

        self._dirty = True
        tag = "WIN" if trade["pnl"] > 0 else "LOSS"
        logger.info(f"[Trade] CLOSE {trade['symbol']} [{tag}]: "
                     f"${trade['entry_price']:,.2f} -> ${exit_price:,.2f} "
                     f"P&L: ${trade['pnl']:+,.2f} ({trade['pnl_pct']:+.2f}%) — {reason}")

        # Notify AI learning system
        if self.on_close and trade.get("pnl") is not None:
            try:
                self.on_close(trade, self.engine_type)
            except Exception as e:
                logger.debug(f"[TradeStore] on_close callback error: {e}")

    def update_sl_tp(self, trade_id: str, new_sl: float = None,
                     new_tp: float = None):
        """Update stop loss and/or take profit for an open trade."""
        trade = self.trades.get(trade_id)
        if not trade or trade["status"] != "open":
            return
        if new_sl is not None:
            trade["stop_loss"] = new_sl
        if new_tp is not None:
            trade["take_profit"] = new_tp
        self._dirty = True

    def get_open_trades(self) -> list:
        with self._lock:
            return [dict(t) for t in self.trades.values() if t["status"] == "open"]

    def get_symbol_open(self, symbol: str) -> dict:
        with self._lock:
            for t in self.trades.values():
                if t["symbol"] == symbol and t["status"] == "open":
                    return t
            return None

    def get_stats(self) -> dict:
        with self._lock:
            closed = [t for t in self.trades.values() if t["status"] != "open"]
        if not closed:
            return {
                "total_trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
                "avg_win_pct": 0.0, "avg_loss_pct": 0.0, "total_pnl_usdt": 0.0,
                "best_trade_pct": 0.0, "worst_trade_pct": 0.0,
                "avg_hold_hours": 0.0, "profit_factor": 0.0, "by_symbol": {},
            }

        wins = [t for t in closed if t["pnl"] > 0]
        losses = [t for t in closed if t["pnl"] <= 0]
        gross_profit = sum(t["pnl"] for t in wins) if wins else 0.0
        gross_loss = abs(sum(t["pnl"] for t in losses)) if losses else 0.0

        hold_hours = []
        for t in closed:
            if t["entry_time"] and t["exit_time"]:
                h = (t["exit_time"] - t["entry_time"]) / 3_600_000_000
                hold_hours.append(h)

        by_symbol = {}
        for t in closed:
            sym = t["symbol"]
            if sym not in by_symbol:
                by_symbol[sym] = {"trades": 0, "wins": 0, "pnl": 0.0}
            by_symbol[sym]["trades"] += 1
            if t["pnl"] > 0:
                by_symbol[sym]["wins"] += 1
            by_symbol[sym]["pnl"] += t["pnl"]
        for s in by_symbol.values():
            s["win_rate"] = (s["wins"] / s["trades"] * 100) if s["trades"] else 0.0

        stats = {
            "total_trades": len(closed),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": (len(wins) / len(closed) * 100) if closed else 0.0,
            "avg_win_pct": (sum(t["pnl_pct"] for t in wins) / len(wins)) if wins else 0.0,
            "avg_loss_pct": (sum(t["pnl_pct"] for t in losses) / len(losses)) if losses else 0.0,
            "total_pnl_usdt": sum(t["pnl"] for t in closed),
            "best_trade_pct": max((t["pnl_pct"] for t in closed), default=0.0),
            "worst_trade_pct": min((t["pnl_pct"] for t in closed), default=0.0),
            "avg_hold_hours": (sum(hold_hours) / len(hold_hours)) if hold_hours else 0.0,
            "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0.0,
            "by_symbol": by_symbol,
        }

        # Quant metrics (from qor.quant — all 12 formulas)
        if len(closed) >= 5:
            try:
                from qor.quant import QuantMetrics
                qm = QuantMetrics()
                stats["quant"] = qm.full_report(closed)
            except Exception:
                pass

        return stats

    def get_recent_symbol_stats(self, symbol: str, n: int = 20) -> dict:
        closed = [t for t in self.trades.values()
                  if t["symbol"] == symbol and t["status"] != "open"]
        recent = closed[-n:]
        if not recent:
            return {"trades": 0, "win_rate": 0.0, "avg_pnl_pct": 0.0}
        wins = sum(1 for t in recent if t["pnl"] > 0)
        return {
            "trades": len(recent),
            "win_rate": (wins / len(recent) * 100),
            "avg_pnl_pct": sum(t["pnl_pct"] for t in recent) / len(recent),
        }

    def save(self):
        with self._lock:
            if not self.trades:
                return
            data = {col: [] for col in TRADE_SCHEMA.names}
            for t in self.trades.values():
                for col in TRADE_SCHEMA.names:
                    data[col].append(t.get(col, self._default(col)))

        batch = pa.RecordBatch.from_pydict(data, schema=TRADE_SCHEMA)
        table = pa.Table.from_batches([batch])
        pq.write_table(table, self.path)
        self._dirty = False


# ==============================================================================
# Shared Analysis Parser (used by both Spot + Futures PositionManagers)
# ==============================================================================

def parse_analysis(text: str) -> dict:
    """Parse multi-TF analysis text into structured data.

    Shared between PositionManager (spot) and FuturesPositionManager.
    Extracts bias, entry/SL/TP levels, per-TF ATR/RSI, supports/resistances,
    divergence signals, Fibonacci/Pivot levels, and AI_FEATURES.
    """
    result = {
        "bias": "NEUTRAL", "entry": 0.0, "stop_loss": 0.0,
        "tp1": 0.0, "tp2": 0.0, "risk_reward": 0.0, "current": 0.0,
        "atr": 0.0, "atr_daily": 0.0, "bullish_tfs": 0, "bearish_tfs": 0,
        "total_tfs": 0, "rsi": 50.0,
        "supports": [], "resistances": [],
        "ema21": 0.0, "ema50": 0.0,
        "macd_hist": 0.0, "bb_upper": 0.0, "bb_lower": 0.0,
        # Per-TF ATR for mode-based SL/TP
        "atr_5m": 0.0, "atr_15m": 0.0, "atr_30m": 0.0,
        "atr_1h": 0.0, "atr_4h": 0.0, "atr_weekly": 0.0,
        # Per-TF RSI for momentum early exit
        "rsi_5m": 50.0, "rsi_15m": 50.0, "rsi_30m": 50.0,
        "rsi_1h": 50.0, "rsi_4h": 50.0,
        # Extended features for 20-dim CORTEX vector
        "rsi6": 50.0, "ema200": 0.0, "adx": 25.0,
        "keltner_pos": 0.0, "rel_vol": 0.0, "obv_dir": 0.0,
        "vwap": 0.0, "body_ratio": 0.5, "upper_wick": 0.25,
        "lower_wick": 0.25, "funding_rate": 0.0, "oi_change": 0.0,
        # FVG + Fibonacci + confluence score
        "fvg_bias": 0.0, "fib_bias": 0.0,
        "fib_sup": 0.0, "fib_res": 0.0,
        # Pivot Points
        "pivot_bias": 0.0, "pp": 0.0,
        "pivot_s1": 0.0, "pivot_r1": 0.0,
        "pivot_s2": 0.0, "pivot_r2": 0.0,
        # Divergence
        "div_score": 0.0,
        "confluence_score": 0.0,
    }
    if not text:
        return result

    m = re.search(r'\$([0-9,]+\.?\d*)\)', text)
    if m:
        result["current"] = float(m.group(1).replace(",", ""))

    bias_m = re.search(r'Bias:\s*(\w+)', text)
    if bias_m:
        result["bias"] = bias_m.group(1).upper()

    for key, pat in [
        ("entry", r'Entry:\s*\$([0-9,]+\.?\d*)'),
        ("stop_loss", r'Stop Loss:\s*\$([0-9,]+\.?\d*)'),
        ("tp1", r'Take Profit 1:\s*\$([0-9,]+\.?\d*)'),
        ("tp2", r'Take Profit 2:\s*\$([0-9,]+\.?\d*)'),
    ]:
        m = re.search(pat, text)
        if m:
            result[key] = float(m.group(1).replace(",", ""))

    rr_m = re.search(r'R:R\s+([0-9.]+)', text)
    if rr_m:
        result["risk_reward"] = float(rr_m.group(1))

    # TF confluence — parse both bullish and bearish counts
    tf_m = re.search(r'(\d+)/(\d+)\s+TFs\s+(bullish|bearish)', text)
    if tf_m:
        count = int(tf_m.group(1))
        total = int(tf_m.group(2))
        if tf_m.group(3) == "bullish":
            result["bullish_tfs"] = count
            result["bearish_tfs"] = total - count
        else:
            result["bearish_tfs"] = count
            result["bullish_tfs"] = total - count
        result["total_tfs"] = total

    # ATR from ALL timeframes for mode-based SL/TP
    for label, atr_key in [
        ("5-MIN", "atr_5m"), ("15-MIN", "atr_15m"),
        ("30-MIN", "atr_30m"),
        ("1-HOUR", "atr_1h"), ("4-HOUR", "atr_4h"),
        ("DAILY", "atr_daily"), ("WEEKLY", "atr_weekly"),
    ]:
        atr_m = re.search(
            rf'{label}.*?ATR:\s*\$([0-9,]+\.?\d*)', text, re.DOTALL)
        if atr_m:
            result[atr_key] = float(atr_m.group(1).replace(",", ""))
    # Legacy "atr" field — weekly or fallback
    result["atr"] = result["atr_weekly"] or result["atr_daily"]
    if result["atr"] == 0:
        atr_m = re.search(r'ATR:\s*\$([0-9,]+\.?\d*)', text)
        if atr_m:
            result["atr"] = float(atr_m.group(1).replace(",", ""))
    if result["atr_daily"] == 0:
        result["atr_daily"] = result["atr"]

    # RSI from ALL timeframes for momentum early exit
    for label, rsi_key in [
        ("5-MIN", "rsi_5m"), ("15-MIN", "rsi_15m"),
        ("30-MIN", "rsi_30m"),
        ("1-HOUR", "rsi_1h"), ("4-HOUR", "rsi_4h"),
    ]:
        rsi_m = re.search(
            rf'{label}.*?RSI:\s*([0-9.]+)', text, re.DOTALL)
        if rsi_m:
            result[rsi_key] = float(rsi_m.group(1))

    # Parse per-TF bias from "Score: +15 (BULLISH)" lines
    for label, bias_key in [
        ("5-MIN", "bias_5m"), ("15-MIN", "bias_15m"),
        ("30-MIN", "bias_30m"),
        ("1-HOUR", "bias_1h"), ("4-HOUR", "bias_4h"),
        ("DAILY", "bias_daily"), ("WEEKLY", "bias_weekly"),
    ]:
        bias_m = re.search(
            rf'{label}.*?Score:.*?\((\w+)\)', text, re.DOTALL)
        if bias_m:
            result[bias_key] = bias_m.group(1).upper()

    # Parse per-TF divergence from "Divergence: RSI bullish + OBV bullish"
    for label, div_prefix in [
        ("5-MIN", "div_5m"), ("15-MIN", "div_15m"),
        ("30-MIN", "div_30m"),
        ("1-HOUR", "div_1h"), ("4-HOUR", "div_4h"),
    ]:
        div_m = re.search(
            rf'{label}.*?Divergence:\s*([^\n|]+?)(?:\s*\|)', text, re.DOTALL)
        if div_m:
            div_text = div_m.group(1).strip().lower()
            if div_text != "none":
                rsi_d = "bullish" if "rsi bullish" in div_text else (
                    "bearish" if "rsi bearish" in div_text else "none")
                macd_d = "bullish" if "macd bullish" in div_text else (
                    "bearish" if "macd bearish" in div_text else "none")
                obv_d = "bullish" if "obv bullish" in div_text else (
                    "bearish" if "obv bearish" in div_text else "none")
                result[f"{div_prefix}_rsi"] = rsi_d
                result[f"{div_prefix}_macd"] = macd_d
                result[f"{div_prefix}_obv"] = obv_d

    # Daily RSI (primary)
    rsi_m = re.search(r'(?:DAILY|Swing).*?RSI:\s*([0-9.]+)', text, re.DOTALL)
    if rsi_m:
        result["rsi"] = float(rsi_m.group(1))

    # EMA from daily
    ema21_m = re.search(r'(?:DAILY|Swing).*?EMA21:\s*\$([0-9,]+\.?\d*)', text, re.DOTALL)
    if ema21_m:
        result["ema21"] = float(ema21_m.group(1).replace(",", ""))
    ema50_m = re.search(r'(?:DAILY|Swing).*?EMA50:\s*\$([0-9,]+\.?\d*)', text, re.DOTALL)
    if ema50_m:
        result["ema50"] = float(ema50_m.group(1).replace(",", ""))

    # Support / Resistance
    sup_m = re.search(r'Support:\s*([\$0-9,. |]+)', text)
    if sup_m and "none" not in sup_m.group(1).lower():
        result["supports"] = [
            float(s.strip().replace("$", "").replace(",", ""))
            for s in sup_m.group(1).split("|") if s.strip()
        ]
    res_m = re.search(r'Resistance:\s*([\$0-9,. |]+)', text)
    if res_m and "none" not in res_m.group(1).lower():
        result["resistances"] = [
            float(r.strip().replace("$", "").replace(",", ""))
            for r in res_m.group(1).split("|") if r.strip()
        ]

    # MACD histogram from daily timeframe
    macd_m = re.search(r'(?:DAILY|Swing).*?Hist:\s*([-0-9,.]+)', text, re.DOTALL)
    if macd_m:
        result["macd_hist"] = float(macd_m.group(1).replace(",", ""))

    # Bollinger Bands
    bb_upper_m = re.search(r'Upper\s*\$([0-9,]+\.?\d*)', text)
    if bb_upper_m:
        result["bb_upper"] = float(bb_upper_m.group(1).replace(",", ""))
    bb_lower_m = re.search(r'Lower\s*\$([0-9,]+\.?\d*)', text)
    if bb_lower_m:
        result["bb_lower"] = float(bb_lower_m.group(1).replace(",", ""))

    # AI_FEATURES structured line — extended indicators for 20-dim vector
    ai_m = re.search(r'AI_FEATURES:\s*(.+)', text)
    if ai_m:
        for pair in ai_m.group(1).split("|"):
            pair = pair.strip()
            if "=" in pair:
                key, val = pair.split("=", 1)
                key = key.strip()
                try:
                    result[key] = float(val.strip())
                except ValueError:
                    pass

    return result


# ==============================================================================
# PositionManager — Full AI Decision Engine
# ==============================================================================

class PositionManager:
    """Every 5 minutes: analyze, manage positions, enter/exit/DCA/adjust.

    Decision priority for EXISTING positions (checked in order):
      1. SL hit          → SELL all (stop loss)
      2. TP2 hit         → SELL all (take profit 2)
      3. TP1 hit (first) → PARTIAL_TP (sell portion) + move SL to breakeven
      4. All TFs bearish  → SELL all (trend reversal)
      5. DCA conditions   → DCA (add to position at support)
      6. Trailing stop    → ADJUST_SL (ratchet up)
      7. None of above    → HOLD

    Decision for NO position:
      - Confluence >= 3/6 TFs bullish + R:R >= min + cooldown passed → BUY
    """

    def __init__(self, config, client: ExchangeClient, store: TradeStore,
                 tool_executor=None, hmm: 'MarketHMM | None' = None,
                 cortex: 'CortexAnalyzer | None' = None):
        self.config = config
        self.client = client
        self.store = store
        self.tool_executor = tool_executor
        self.hmm = hmm

        # CORTEX analyzer — shared instance from runtime, or create own
        self.cortex = cortex
        if self.cortex is None and _HAS_CORTEX:
            try:
                self.cortex = CortexAnalyzer()
                logger.info("[PM] CORTEX analyzer initialized (local instance) "
                            f"(obs d={CortexAnalyzer.OBS_DIM}, "
                            f"reflex {CortexAnalyzer.REFLEX_NEURONS} neurons, "
                            f"history {CortexAnalyzer.HISTORY_LEN} candles)")
            except Exception as e:
                logger.warning(f"[PM] CORTEX analyzer init failed: {e}")
        elif self.cortex is not None:
            logger.info("[PM] Using shared CORTEX analyzer")

    def _get_trade_atr(self, analysis: dict) -> float:
        """Return ATR appropriate for the current trade_mode.

        - scalp:  5m → 30m  (tight SL/TP for fast trades)
        - stable: 30m → 4h  (medium swings)
        - secure: 4h → 1w   (wide SL/TP, safe)

        All modes still use ALL timeframes for trend direction (bias).
        """
        mode = getattr(self.config, "trade_mode", "scalp")
        if mode == "scalp":
            atr = (analysis.get("atr_15m") or analysis.get("atr_30m")
                   or analysis.get("atr_5m", 0))
        elif mode == "stable":
            atr = (analysis.get("atr_1h") or analysis.get("atr_4h")
                   or analysis.get("atr_30m", 0))
        else:  # secure
            atr = (analysis.get("atr_daily") or analysis.get("atr_weekly")
                   or analysis.get("atr_4h", 0))
        if atr <= 0:
            atr = (analysis.get("atr_daily") or analysis.get("atr")
                   or analysis.get("atr_1h") or analysis.get("atr_15m", 0))
        return atr

    def _get_reversal_threshold(self, total_tfs: int) -> int:
        """Mode-aware reversal threshold.

        - scalp:  2 TFs (5m-30m flip fast)
        - stable: 3 TFs (30m-4h)
        - secure: 4 TFs (4h-1w, high conviction)
        """
        mode = getattr(self.config, "trade_mode", "scalp")
        if mode == "scalp":
            return min(2, total_tfs)
        elif mode == "stable":
            return min(3, total_tfs)
        else:
            return max(4, int(total_tfs * 2 / 3))

    def _check_scalp_short_tf_reversal(self, analysis: dict) -> tuple:
        """Check if 2+ of 5m/15m/30m TFs flipped bearish (against LONG).

        NOTE: Spot is LONG-only, so this only checks for bearish reversals.
        The futures version (_check_scalp_short_tf_reversal in FuturesPositionManager)
        accepts a direction parameter and handles both LONG and SHORT.
        Returns (flipped: bool, count: int, details: str).
        """
        mode = getattr(self.config, "trade_mode", "scalp")
        if mode != "scalp":
            return False, 0, ""

        flipped = 0
        tfs_checked = []
        for tf_name in ("5m", "15m", "30m"):
            bias = analysis.get(f"bias_{tf_name}", "").upper()
            if not bias:
                continue
            if bias == "BEARISH":
                flipped += 1
                tfs_checked.append(tf_name)

        if flipped >= 2:
            return True, flipped, "+".join(tfs_checked)
        return False, flipped, ""

    def _check_divergence_close(self, analysis: dict) -> tuple:
        """Check if multi-TF divergence signals warrant closing LONG position.

        Scalp:  checks 5m, 15m, 30m — needs 2+ TFs with 2+ div types.
        Stable: checks 30m, 1h, 4h.
        Secure: checks 1h, 4h.

        Returns (should_close: bool, detail: str).
        """
        mode = getattr(self.config, "trade_mode", "scalp")

        # For LONG: bearish divergence = momentum fading → close
        close_div = "bearish"

        if mode == "scalp":
            tf_checks = ["5m", "15m", "30m"]
            min_tfs_matching = 2
        elif mode == "stable":
            tf_checks = ["30m", "1h", "4h"]
            min_tfs_matching = 2
        else:
            tf_checks = ["1h", "4h"]
            min_tfs_matching = 2

        tfs_with_div = 0
        tfs_with_extreme_rsi = 0
        details = []

        for tf in tf_checks:
            rsi_d = analysis.get(f"div_{tf}_rsi", "none")
            macd_d = analysis.get(f"div_{tf}_macd", "none")
            obv_d = analysis.get(f"div_{tf}_obv", "none")
            tf_rsi = analysis.get(f"rsi_{tf}", 50)

            matching_types = sum(1 for d in [rsi_d, macd_d, obv_d]
                                 if d == close_div)
            if matching_types >= 2:
                tfs_with_div += 1
                details.append(tf)

            if tf_rsi < 30:
                tfs_with_extreme_rsi += 1

        if tfs_with_div >= min_tfs_matching and tfs_with_extreme_rsi >= min_tfs_matching:
            detail = (f"{'+'.join(details)} {close_div} div "
                      f"({tfs_with_div}/{len(tf_checks)} TFs, "
                      f"RSI extreme on {tfs_with_extreme_rsi})")
            return True, detail

        return False, ""

    def _get_analysis(self, symbol: str) -> str:
        if self.tool_executor:
            try:
                return self.tool_executor.call("multi_tf_analysis", symbol)
            except Exception as e:
                logger.warning(f"[PM] multi_tf_analysis via executor failed: {e}")
        try:
            from qor.tools import get_multi_timeframe_analysis
            return get_multi_timeframe_analysis(symbol)
        except Exception as e:
            logger.error(f"[PM] Analysis failed for {symbol}: {e}")
            return ""

    def _parse_analysis(self, text: str) -> dict:
        return parse_analysis(text)

    # ------------------------------------------------------------------
    # Main entry: evaluate one symbol
    # ------------------------------------------------------------------
    def evaluate_symbol(self, symbol: str) -> list:
        """Analyze symbol, return list of actions (can be multiple per tick).

        Returns list of decision dicts. E.g.:
          [{"action": "PARTIAL_TP", ...}, {"action": "ADJUST_SL", ...}]
        or just [{"action": "HOLD", ...}]
        """
        pair = self.client.format_pair(symbol)

        # Run multi-TF analysis
        analysis_text = self._get_analysis(symbol)
        if not analysis_text or "not available" in analysis_text.lower():
            return [{"action": "HOLD", "symbol": symbol,
                     "reason": "analysis not available"}]

        analysis = self._parse_analysis(analysis_text)

        # Get current price from exchange
        try:
            current = self.client.get_price(pair)
        except Exception:
            current = analysis["current"]
        if current <= 0:
            return [{"action": "HOLD", "symbol": symbol,
                     "reason": "no price data"}]

        analysis["current"] = current

        # Sentiment signals FIRST — needed by CORTEX features 20-21
        try:
            from qor.tools import (get_polymarket_sentiment,
                                   get_fear_greed_value,
                                   get_polymarket_calendar)
            poly_data = get_polymarket_sentiment(symbol)
            analysis["poly_up_prob"] = poly_data["up_prob"]
            analysis["poly_available"] = poly_data["available"]
            analysis["fear_greed_value"] = get_fear_greed_value()
            analysis["calendar_events"] = get_polymarket_calendar()
        except Exception as e:
            logger.debug(f"[PM] Sentiment fetch failed for {symbol}: {e}")
        analysis.setdefault("poly_up_prob", 0.5)
        analysis.setdefault("poly_available", False)
        analysis.setdefault("fear_greed_value", 50)
        analysis.setdefault("calendar_events", [])

        # Volume Profile — needed by CORTEX features 22-23
        try:
            from qor.tools import get_volume_profile
            vp_data = get_volume_profile(symbol)
            analysis["vp_poc"] = vp_data["poc"]
            analysis["vp_vah"] = vp_data["vah"]
            analysis["vp_val"] = vp_data["val"]
            analysis["vp_hvn"] = vp_data.get("hvn_zones", [])
            analysis["vp_lvn"] = vp_data.get("lvn_zones", [])
            analysis["vp_available"] = vp_data["available"]
        except Exception as e:
            logger.debug(f"[PM] Volume profile fetch failed for {symbol}: {e}")
        analysis.setdefault("vp_poc", 0)
        analysis.setdefault("vp_vah", 0)
        analysis.setdefault("vp_val", 0)
        analysis.setdefault("vp_hvn", [])
        analysis.setdefault("vp_lvn", [])
        analysis.setdefault("vp_available", False)

        # Quant signals — Hurst, Z-Score, asset volatility from live prices
        try:
            from qor.tools import get_quant_signals
            qs = get_quant_signals(symbol)
            analysis["hurst"] = qs["hurst"]
            analysis["hurst_regime"] = qs["hurst_regime"]
            analysis["z_score"] = qs["z_score"]
            analysis["z_label"] = qs["z_label"]
            analysis["asset_vol"] = qs["asset_vol"]
            analysis["quant_signals_available"] = qs["available"]
            analysis["price_returns"] = qs.get("price_returns", [])
        except Exception as e:
            logger.debug(f"[PM] Quant signals fetch failed for {symbol}: {e}")
        analysis.setdefault("hurst", 0.5)
        analysis.setdefault("hurst_regime", "random")
        analysis.setdefault("z_score", 0.0)
        analysis.setdefault("z_label", "neutral")
        analysis.setdefault("asset_vol", 0.0)
        analysis.setdefault("price_returns", [])
        analysis.setdefault("quant_signals_available", False)

        # CORTEX analysis — uses sentiment features 20-21 + VP features 22-23
        cortex_result = None
        if self.cortex:
            try:
                cortex_result = self.cortex.analyze(analysis, symbol)
                analysis["cortex_signal"] = cortex_result["signal"]
                analysis["cortex_label"] = cortex_result["label"]
                analysis["cortex_confidence"] = cortex_result["confidence"]
                candles = cortex_result.get("history_candles", 0)
                logger.info(
                    f"[PM] CORTEX {symbol}: {cortex_result['label']} "
                    f"(signal={cortex_result['signal']:.3f}, "
                    f"conf={cortex_result['confidence']:.3f}, "
                    f"history={candles} candles)")
            except Exception as e:
                logger.warning(f"[PM] CORTEX analysis failed for {symbol}: {e}")

        # HMM regime detection — runs every tick alongside CORTEX
        hmm_signal = None
        if self.hmm and self.hmm.is_available:
            try:
                hmm_signal = self.hmm.get_signal(analysis, symbol)
                analysis["hmm_state"] = hmm_signal.state_name
                analysis["hmm_confidence"] = hmm_signal.confidence
                analysis["hmm_signal"] = hmm_signal.signal
                analysis["hmm_strength"] = hmm_signal.signal_strength
                if hmm_signal.transition:
                    logger.info(
                        f"[PM] HMM {symbol}: REGIME CHANGE "
                        f"{hmm_signal.prev_state_name} → "
                        f"{hmm_signal.state_name} "
                        f"(conf={hmm_signal.confidence:.3f}, "
                        f"signal={hmm_signal.signal})")
            except Exception as e:
                logger.warning(f"[PM] HMM analysis failed for {symbol}: {e}")

        existing = self.store.get_symbol_open(symbol)

        if existing:
            decisions = self._manage_position(existing, analysis, symbol)
        else:
            decisions = self._evaluate_entry(analysis, symbol, current)

        # Annotate all decisions with CORTEX + HMM data
        if cortex_result:
            for d in decisions:
                d["cortex_signal"] = cortex_result["signal"]
                d["cortex_label"] = cortex_result["label"]
        if hmm_signal:
            for d in decisions:
                d["hmm_state"] = hmm_signal.state_name
                d["hmm_signal"] = hmm_signal.signal
                d["hmm_confidence"] = hmm_signal.confidence

        return decisions

    # ------------------------------------------------------------------
    # Position management (has position)
    # ------------------------------------------------------------------
    def _manage_position(self, trade: dict, analysis: dict, symbol: str) -> list:
        """Full position management: SL, TP, partial, DCA, trailing, reversal."""
        actions = []
        current = analysis["current"]
        entry = trade["entry_price"]
        sl = trade["stop_loss"]
        tp1 = trade["take_profit"]
        tp2 = trade.get("take_profit2", 0) or tp1 * 1.5
        tid = trade["trade_id"]
        tp1_hit = trade.get("tp1_hit", False)

        # --- 1. SL hit → close all ---
        if current <= sl:
            actions.append({
                "action": "SELL", "symbol": symbol, "trade_id": tid,
                "exit_status": "closed_sl",
                "reason": f"SL hit at ${sl:,.2f} (entry ${entry:,.2f})",
            })
            return actions

        # --- 2. TP2 hit → close all remaining ---
        if tp2 > 0 and current >= tp2:
            actions.append({
                "action": "SELL", "symbol": symbol, "trade_id": tid,
                "exit_status": "closed_tp",
                "reason": f"TP2 hit at ${tp2:,.2f}",
            })
            return actions

        # --- 3. TP1 hit (first time) → partial take profit ---
        if not tp1_hit and tp1 > 0 and current >= tp1:
            if self.config.partial_tp_enabled:
                sell_pct = self.config.partial_tp1_pct / 100.0
                sell_qty = trade["quantity"] * sell_pct
                actions.append({
                    "action": "PARTIAL_TP", "symbol": symbol, "trade_id": tid,
                    "sell_qty": sell_qty, "sell_price": current,
                    "reason": f"TP1 hit ${tp1:,.2f} — selling {self.config.partial_tp1_pct:.0f}%",
                })
                # After partial TP1 → move SL to breakeven
                if self.config.move_sl_to_be and entry > sl:
                    actions.append({
                        "action": "ADJUST_SL", "symbol": symbol, "trade_id": tid,
                        "new_sl": entry,
                        "reason": f"SL to breakeven ${entry:,.2f} after TP1",
                    })
                return actions
            else:
                # No partial — close all at TP1
                actions.append({
                    "action": "SELL", "symbol": symbol, "trade_id": tid,
                    "exit_status": "closed_tp",
                    "reason": f"TP1 hit at ${tp1:,.2f}",
                })
                return actions

        # --- 4. Trend reversal → close (mode-aware threshold) ---
        cortex_label = analysis.get("cortex_label", "")
        cortex_signal = analysis.get("cortex_signal", 0)
        cortex_reversal = cortex_label == "STRONG_SELL" and analysis["bearish_tfs"] >= analysis["total_tfs"] * 0.5
        total = analysis["total_tfs"]
        reversal_thresh = self._get_reversal_threshold(total)
        majority_bearish = total > 0 and analysis["bearish_tfs"] >= reversal_thresh

        # Scalp mode: also check if 2+ of 5m/15m/30m flipped bearish
        scalp_rev, scalp_cnt, scalp_detail = self._check_scalp_short_tf_reversal(analysis)

        if majority_bearish or cortex_reversal or scalp_rev:
            reason_parts = []
            if majority_bearish:
                reason_parts.append(f"{analysis['bearish_tfs']}/{total} TFs bearish (thresh={reversal_thresh})")
            if scalp_rev:
                reason_parts.append(f"scalp {scalp_cnt}/3 short TFs bearish ({scalp_detail})")
            if cortex_reversal:
                reason_parts.append(f"CORTEX STRONG_SELL ({cortex_signal:.3f})")
            actions.append({
                "action": "SELL", "symbol": symbol, "trade_id": tid,
                "exit_status": "closed_reversal",
                "reason": f"trend reversal — {' + '.join(reason_parts)}",
            })
            return actions

        # --- 4a. Divergence-based close ---
        # Scalp: 5m+15m+30m, Stable: 30m+1h+4h, Secure: 1h+4h
        # Requires 2+ div types (RSI+MACD or RSI+OBV) + RSI extreme
        div_close, div_detail = self._check_divergence_close(analysis)
        pnl_pct = ((current / entry) - 1) * 100 if entry > 0 else 0
        if div_close:
            actions.append({
                "action": "SELL", "symbol": symbol, "trade_id": tid,
                "exit_status": "closed_divergence",
                "reason": f"divergence close: {div_detail}, P&L {pnl_pct:+.1f}%",
            })
            return actions

        # --- 4b. Adverse momentum early exit ---
        # Checks 5m + 15m + 30m RSI (all scalp TFs)
        rsi_5m = analysis.get("rsi_5m", 50)
        rsi_15m = analysis.get("rsi_15m", 50)
        rsi_30m = analysis.get("rsi_30m", 50)
        # Close if losing >1% AND any 2 of 3 short TFs show extreme bearish RSI
        bearish_rsi_count = sum(1 for r in [rsi_5m, rsi_15m, rsi_30m]
                                if r < 30)
        if pnl_pct < -1.0 and bearish_rsi_count >= 2:
            actions.append({
                "action": "SELL", "symbol": symbol, "trade_id": tid,
                "exit_status": "closed_momentum",
                "reason": (f"adverse momentum: RSI 5m={rsi_5m:.0f} "
                           f"15m={rsi_15m:.0f} 30m={rsi_30m:.0f}, "
                           f"P&L {pnl_pct:+.1f}%"),
            })
            return actions

        # --- 4c. Tighten SL when half TFs turn against + losing ---
        half_bearish = total > 0 and analysis["bearish_tfs"] >= total // 2
        if half_bearish and pnl_pct < 0:
            atr = self._get_trade_atr(analysis)
            if atr > 0:
                tight_sl = current - atr  # 1x ATR instead of 2x
                if tight_sl > sl:
                    actions.append({
                        "action": "ADJUST_SL", "symbol": symbol, "trade_id": tid,
                        "new_sl": tight_sl,
                        "reason": (f"tighten SL: {analysis['bearish_tfs']}/{total} "
                                   f"TFs against, ${sl:,.2f} -> ${tight_sl:,.2f}"),
                    })
                    return actions

        # --- 5. DCA: price dropped, at support, budget left ---
        dca_action = self._check_dca(trade, analysis, symbol, current)
        if dca_action:
            actions.append(dca_action)

        # --- 6. Trailing stop / dynamic SL adjustment ---
        sl_action = self._check_trailing_sl(trade, analysis, current)
        if sl_action:
            actions.append(sl_action)

        # --- 7. Dynamic TP adjustment (if resistance moved) ---
        if analysis["resistances"] and not tp1_hit:
            nearest_res = min(r for r in analysis["resistances"] if r > current) \
                if any(r > current for r in analysis["resistances"]) else 0
            if nearest_res > 0 and nearest_res != tp1 and nearest_res > entry:
                actions.append({
                    "action": "ADJUST_TP", "symbol": symbol, "trade_id": tid,
                    "new_tp": nearest_res,
                    "reason": f"TP adjusted to resistance ${nearest_res:,.2f}",
                })

        if not actions:
            actions.append({
                "action": "HOLD", "symbol": symbol,
                "reason": f"holding ({pnl_pct:+.1f}% from entry)",
            })
        return actions

    def _check_dca(self, trade: dict, analysis: dict, symbol: str,
                   current: float) -> dict:
        """Check if we should DCA into this position."""
        if not self.config.dca_enabled:
            return None

        dca_count = trade.get("dca_count", 0)
        if dca_count >= self.config.dca_max_adds:
            return None

        entry = trade["entry_price"]
        drop_pct = ((entry - current) / entry) * 100 if entry > 0 else 0

        # Need price to have dropped at least dca_drop_pct from avg entry
        required_drop = self.config.dca_drop_pct * (dca_count + 1)  # Deeper for each DCA
        if drop_pct < required_drop:
            return None

        # Check if near a support level (within 1 ATR)
        atr = self._get_trade_atr(analysis)
        near_support = False
        if analysis["supports"] and atr > 0:
            for sup in analysis["supports"]:
                if abs(current - sup) <= atr:
                    near_support = True
                    break

        # Also DCA if RSI is oversold (< 30) even without support
        rsi_oversold = analysis["rsi"] < 35

        if not near_support and not rsi_oversold:
            return None

        # Don't DCA if all TFs are bearish (falling knife)
        if analysis["bearish_tfs"] >= analysis["total_tfs"] and analysis["total_tfs"] > 0:
            return None

        # CORTEX veto: don't DCA into a STRONG_SELL — liquid neurons see reversal
        if analysis.get("cortex_label") == "STRONG_SELL":
            return None

        # Need at least 1 bullish TF to confirm some buying pressure
        if analysis["bullish_tfs"] == 0:
            return None

        support_str = "near support" if near_support else ""
        rsi_str = f"RSI={analysis['rsi']:.0f}" if rsi_oversold else ""
        reason_parts = [f"drop {drop_pct:.1f}%"]
        if support_str:
            reason_parts.append(support_str)
        if rsi_str:
            reason_parts.append(rsi_str)

        return {
            "action": "DCA", "symbol": symbol,
            "trade_id": trade["trade_id"],
            "dca_number": dca_count + 1,
            "atr": atr,  # Pass real ATR for SL recalculation
            "reason": f"DCA #{dca_count + 1}: {', '.join(reason_parts)}",
        }

    def _check_trailing_sl(self, trade: dict, analysis: dict,
                           current: float) -> dict:
        """Check if SL should be moved up (trailing stop)."""
        if not self.config.trailing_stop:
            return None

        entry = trade["entry_price"]
        sl = trade["stop_loss"]
        tid = trade["trade_id"]
        atr = self._get_trade_atr(analysis)

        if atr <= 0:
            return None

        # Calculate new SL: current price minus N * ATR
        new_sl = current - self.config.stop_loss_atr_mult * atr

        # Also consider nearest support below current price
        for s in sorted(analysis["supports"], reverse=True):
            if s < current:
                new_sl = max(new_sl, s * 0.995)  # just below support
                break

        # After TP1 hit, SL should be at least at breakeven
        if trade.get("tp1_hit") and self.config.move_sl_to_be:
            new_sl = max(new_sl, entry)

        # Only move SL UP (never down) — this is a trailing stop
        if new_sl > sl:
            return {
                "action": "ADJUST_SL", "symbol": trade["symbol"],
                "trade_id": tid, "new_sl": new_sl,
                "reason": f"trailing SL: ${sl:,.2f} -> ${new_sl:,.2f}",
            }
        return None

    # ------------------------------------------------------------------
    # Entry evaluation (no position)
    # ------------------------------------------------------------------
    def _evaluate_entry(self, analysis: dict, symbol: str,
                        current: float) -> list:
        """Evaluate whether to open a new position."""
        hold = {"action": "HOLD", "symbol": symbol}

        block_reason = self._should_enter(analysis, symbol)
        if block_reason:
            hold["reason"] = block_reason
            return [hold]

        # Spot only — can only go LONG
        if analysis["bias"] not in ("LONG", "BULLISH"):
            hold["reason"] = f"bias is {analysis['bias']} — spot only"
            return [hold]

        # Always use live price + mode-appropriate ATR for SL/TP
        entry_price = current
        atr = self._get_trade_atr(analysis)
        mode = getattr(self.config, "trade_mode", "scalp")

        # ATR floor: minimum 0.3% of price to prevent unreasonably tight SL/TP
        if current > 0 and atr > 0:
            min_atr = current * 0.003
            if atr < min_atr:
                logger.debug(f"[PM] ATR floor: {atr:.4f} -> {min_atr:.4f} "
                             f"(0.3% of ${current:,.2f})")
                atr = min_atr

        # Confluence score strength check — weak signals should not enter
        cscore = analysis.get("confluence_score", 0)
        total_tfs = analysis.get("total_tfs", 7)
        min_score = total_tfs * 15
        if cscore < min_score:  # spot = LONG only, need positive score
            hold["reason"] = (f"weak confluence score: {cscore:+.0f} "
                              f"(need +{min_score:.0f})")
            return [hold]

        # Fibonacci + Pivot levels as SL/TP confluence
        fib_sup = analysis.get("fib_sup", 0)
        fib_res = analysis.get("fib_res", 0)
        pivot_s1 = analysis.get("pivot_s1", 0)
        pivot_s2 = analysis.get("pivot_s2", 0)
        pivot_r1 = analysis.get("pivot_r1", 0)
        pivot_r2 = analysis.get("pivot_r2", 0)

        # Compute SL/TP from mode-appropriate ATR, refined with Fib + Pivots
        sl_mult = self.config.stop_loss_atr_mult
        tp_mult = self.config.take_profit_atr_mult
        sl_price = current - sl_mult * atr if atr > 0 else 0
        tp1_price = current + tp_mult * atr if atr > 0 else 0
        tp2_price = current + tp_mult * atr * 2 if atr > 0 else 0

        # Refine SL with Fib support / Pivot S1/S2 — tighten if closer
        for level in [fib_sup, pivot_s1, pivot_s2]:
            if level > 0 and level < current and atr > 0:
                candidate = level - 0.2 * atr  # just below the level
                if candidate > sl_price and (current - candidate) >= 0.5 * atr:
                    sl_price = candidate
        # Refine TP with Fib resistance / Pivot R1/R2 — use if better R:R
        for level in [fib_res, pivot_r1, pivot_r2]:
            if level > current and atr > 0 and sl_price > 0:
                level_rr = (level - current) / max(current - sl_price, 0.01)
                curr_rr = (tp1_price - current) / max(current - sl_price, 0.01)
                if level_rr > curr_rr:
                    tp1_price = level

        # Volume Profile SL/TP refinement (LONG only for spot)
        vp_val = analysis.get("vp_val", 0)
        vp_vah = analysis.get("vp_vah", 0)
        if analysis.get("vp_available") and atr > 0:
            # SL: If VAL is between SL and entry → tighten SL to just below VAL
            if vp_val > 0 and vp_val < current:
                vp_sl = vp_val - 0.2 * atr
                if vp_sl > sl_price and (current - vp_sl) >= 0.5 * atr:
                    sl_price = vp_sl
            # TP: If VAH is above entry and offers better R:R → use as TP
            if vp_vah > current and sl_price > 0:
                vp_rr = (vp_vah - current) / max(current - sl_price, 0.01)
                curr_rr = (tp1_price - current) / max(current - sl_price, 0.01)
                if vp_rr > curr_rr:
                    tp1_price = vp_vah

        # Hurst TP adjustment — trending markets: widen TP, mean-reverting: tighten
        hurst = analysis.get("hurst", 0.5)
        if analysis.get("quant_signals_available") and atr > 0:
            if hurst > 0.6:
                # Trending: let profits run — widen TP by 20%
                tp1_price = current + (tp1_price - current) * 1.2
                tp2_price = current + (tp2_price - current) * 1.2
            elif hurst < 0.4:
                # Mean-reverting: expect pullback — tighten TP by 20%
                tp1_price = current + (tp1_price - current) * 0.8
                tp2_price = current + (tp2_price - current) * 0.8

        # Validate R:R
        risk = current - sl_price
        reward = tp1_price - current
        if risk <= 0 or reward <= 0:
            hold["reason"] = "invalid SL/TP levels"
            return [hold]

        rr = reward / risk
        min_rr = self.config.min_risk_reward

        # Historical adjustment: increase min R:R for losing symbols
        sym_stats = self.store.get_recent_symbol_stats(symbol, n=20)
        if sym_stats["trades"] >= 5 and sym_stats["win_rate"] < 50:
            min_rr = self.config.min_risk_reward * 1.5  # Tighter entry

        if rr < min_rr:
            hold["reason"] = f"R:R {rr:.1f} < min {min_rr:.1f}"
            return [hold]

        # ============================================================
        # Signal multiplier system — all signals INFLUENCE position
        # size instead of blocking.  Each factor adjusts a multiplier.
        # Combined with Kelly + vol cap to produce the final size.
        # ============================================================
        size_mult = 1.0
        mult_notes = []

        # Sharpe — negative edge reduces size
        sharpe = analysis.get("quant_sharpe")
        if sharpe is not None:
            if sharpe < -0.5:
                size_mult *= 0.5
                mult_notes.append(f"Sharpe({sharpe:.2f})×0.5")
            elif sharpe < 0:
                size_mult *= 0.7
                mult_notes.append(f"Sharpe({sharpe:.2f})×0.7")

        # Sortino — downside risk specifically (more relevant than Sharpe for trading)
        sortino = analysis.get("quant_sortino")
        if sortino is not None:
            if sortino < -1.0:
                size_mult *= 0.7
                mult_notes.append(f"Sortino({sortino:.2f})×0.7")
            elif sortino < -0.5:
                size_mult *= 0.85
                mult_notes.append(f"Sortino({sortino:.2f})×0.85")

        # CAPM Alpha — outperformance vs buy-and-hold
        alpha = analysis.get("quant_alpha")
        if alpha is not None:
            if alpha < -0.05:
                # Underperforming buy-and-hold — reduce size
                size_mult *= 0.7
                mult_notes.append(f"α({alpha:.3f})×0.7")
            elif alpha > 0.05:
                # Outperforming — boost size
                size_mult *= 1.15
                mult_notes.append(f"α({alpha:.3f})×1.15")

        # Information Ratio — consistency of alpha
        ir = analysis.get("quant_ir")
        if ir is not None:
            if ir < -0.5:
                # Inconsistently underperforming — reduce
                size_mult *= 0.8
                mult_notes.append(f"IR({ir:.2f})×0.8")
            elif ir > 0.5:
                # Consistently outperforming — boost
                size_mult *= 1.1
                mult_notes.append(f"IR({ir:.2f})×1.1")

        # Z-Score — overbought reduces size for LONG
        z_score = analysis.get("z_score", 0.0)
        if analysis.get("quant_signals_available"):
            if z_score > 2.0:
                size_mult *= 0.5
                mult_notes.append(f"Z({z_score:.1f})×0.5")
            elif z_score > 1.5:
                size_mult *= 0.7
                mult_notes.append(f"Z({z_score:.1f})×0.7")

        # Hurst — mean-reverting reduces trend-trade size, trending boosts
        hurst = analysis.get("hurst", 0.5)
        if analysis.get("quant_signals_available"):
            if hurst < 0.4:
                size_mult *= 0.8
                mult_notes.append(f"H({hurst:.2f})×0.8")
            elif hurst > 0.65:
                size_mult *= 1.1
                mult_notes.append(f"H({hurst:.2f})×1.1")

        # Polymarket contra — crowd predicts down, reduce LONG size
        poly_up = analysis.get("poly_up_prob", 0.5)
        poly_thresh = getattr(self.config, "poly_block_threshold", 0.35)
        if analysis.get("poly_available") and poly_up < poly_thresh:
            size_mult *= 0.7
            mult_notes.append(f"POLY({poly_up*100:.0f}%)×0.7")

        # Fear & Greed — extreme greed reduces LONG size
        fg = analysis.get("fear_greed_value", 50)
        fg_greed = getattr(self.config, "fg_extreme_greed", 85)
        if fg > fg_greed:
            size_mult *= 0.7
            mult_notes.append(f"F&G({fg})×0.7")

        # Divergence — bearish divergence reduces LONG size
        div_score = analysis.get("div_score", 0)
        if div_score <= -15:
            size_mult *= 0.7
            mult_notes.append(f"div({div_score})×0.7")

        # CORTEX — reduces on contra, boosts on strong agreement
        cortex_label = analysis.get("cortex_label", "")
        cortex_signal = analysis.get("cortex_signal", 0)
        if cortex_label == "STRONG_SELL":
            size_mult *= 0.3
            mult_notes.append(f"CORTEX(SS)×0.3")
        elif cortex_label == "SELL":
            size_mult *= 0.5
            mult_notes.append(f"CORTEX(S)×0.5")
        elif cortex_label == "STRONG_BUY":
            size_mult *= 1.2
            mult_notes.append(f"CORTEX(SB)×1.2")

        # HMM regime — contra regime reduces size, aligned boosts
        hmm_state = analysis.get("hmm_state", "")
        hmm_conf = analysis.get("hmm_confidence", 0)
        if hmm_state in ("BEAR", "STRONG_BEAR") and hmm_conf >= 0.6:
            size_mult *= 0.5
            mult_notes.append(f"HMM({hmm_state})×0.5")
        elif hmm_state in ("BULL", "STRONG_BULL") and hmm_conf >= 0.7:
            size_mult *= 1.15
            mult_notes.append(f"HMM({hmm_state})×1.15")
        elif hmm_state == "CHOPPY" and hmm_conf >= 0.7:
            size_mult *= 0.7
            mult_notes.append(f"HMM(CHOPPY)×0.7")

        # ============================================================
        # Compute Kelly position size WITH all signal multipliers
        # This IS the decision — size is known at decision time
        # ============================================================
        position_pct = self._kelly_pct(symbol, analysis=analysis)
        if position_pct <= 0:
            hold["reason"] = "Kelly: no edge"
            return [hold]

        # Apply signal multipliers
        position_pct *= size_mult

        # Floor check — if too small after adjustments, skip
        min_viable = self.config.kelly_min_pct
        if position_pct < min_viable:
            hold["reason"] = (f"size too small after signals: "
                              f"{position_pct:.1f}% (mult={size_mult:.2f}, "
                              f"{', '.join(mult_notes)})")
            return [hold]

        # Clamp to max
        position_pct = min(position_pct, self.config.max_position_pct)

        # ============================================================
        # Build entry reason with all signal annotations
        # ============================================================
        tf_str = f"{analysis['bullish_tfs']}/{analysis['total_tfs']}"
        cortex_str = f", CORTEX={cortex_label}({cortex_signal:.2f})" if cortex_label else ""
        hmm_str = f", HMM={hmm_state}({hmm_conf:.2f})" if hmm_state else ""
        poly_str = f", POLY={poly_up*100:.0f}%Up" if analysis.get("poly_available") else ""
        fg_str = f", F&G={fg}"
        vp_poc = analysis.get("vp_poc", 0)
        vp_str = f", VP:POC=${vp_poc:,.2f}" if analysis.get("vp_available") and vp_poc > 0 else ""
        q_exp = analysis.get("quant_expectancy")
        quant_str = f", E={q_exp:.2f}R" if q_exp is not None else ""
        sharpe_str = f", Sharpe={sharpe:.2f}" if sharpe is not None else ""
        sortino_str = f", Sortino={sortino:.2f}" if sortino is not None else ""
        alpha_str = f", α={alpha:.3f}" if alpha is not None else ""
        ir_str = f", IR={ir:.2f}" if ir is not None else ""
        qs_str = ""
        if analysis.get("quant_signals_available"):
            qs_str = f", H={hurst:.2f}({analysis.get('hurst_regime', '?')}), Z={z_score:.2f}"
        size_str = f", Kelly={position_pct:.1f}%"
        if mult_notes:
            size_str += f" ({' '.join(mult_notes)})"
        entry_reason = (
            f"{analysis['bias']} bias, {tf_str} TFs bullish, "
            f"RSI {analysis['rsi']:.0f}, R:R {rr:.1f}:1, "
            f"ATR ${atr:,.2f} ({mode}){cortex_str}{hmm_str}"
            f"{poly_str}{fg_str}{vp_str}{quant_str}{sharpe_str}"
            f"{sortino_str}{alpha_str}{ir_str}{qs_str}{size_str}"
        )

        return [{
            "action": "BUY", "symbol": symbol,
            "entry_price": entry_price,
            "stop_loss": sl_price,
            "take_profit": tp1_price,
            "take_profit2": tp2_price,
            "risk_reward": rr,
            "reason": entry_reason,
            "position_pct": position_pct,
        }]

    def _should_enter(self, analysis: dict, symbol: str) -> str:
        """Check all entry gate conditions.

        Returns empty string if OK, or a reason string if entry is blocked.
        """
        # Cooldown check
        recent_closed = [
            t for t in self.store.trades.values()
            if t["symbol"] == symbol and t["status"] != "open"
        ]
        if recent_closed:
            last = max(recent_closed, key=lambda t: t["exit_time"])
            if last["exit_time"] > 0:
                exit_dt = datetime.fromtimestamp(
                    last["exit_time"] / 1_000_000, tz=timezone.utc)
                cooldown = timedelta(minutes=self.config.cooldown_minutes)
                remaining = cooldown - (datetime.now(timezone.utc) - exit_dt)
                if remaining.total_seconds() > 0:
                    mins = int(remaining.total_seconds() / 60)
                    return f"cooldown: {mins}min remaining"

        # Max open positions
        open_count = len(self.store.get_open_trades())
        if open_count >= self.config.max_open_positions:
            return f"max positions reached ({open_count}/{self.config.max_open_positions})"

        # Multi-TF confluence: need >= 3 TFs bullish (or >50%)
        if analysis["total_tfs"] > 0:
            min_bullish = max(3, analysis["total_tfs"] // 2 + 1)
            if analysis["bullish_tfs"] < min_bullish:
                return f"weak confluence: {analysis['bullish_tfs']}/{analysis['total_tfs']} bullish (need {min_bullish})"

        # Quant edge validation — replaces simple win-rate check
        # Uses Expectancy + Risk of Ruin from qor.quant
        sym_closed = [t for t in self.store.trades.values()
                      if t["symbol"] == symbol and t["status"] != "open"]
        sym_recent = sym_closed[-30:]
        if len(sym_recent) >= 5:
            try:
                from qor.quant import QuantMetrics
                qm = QuantMetrics()
                wins = [t for t in sym_recent if t["pnl"] > 0]
                losses = [t for t in sym_recent if t["pnl"] <= 0]
                if wins and losses:
                    wr = len(wins) / len(sym_recent)
                    avg_w = sum(t["pnl_pct"] for t in wins) / len(wins)
                    avg_l = abs(sum(t["pnl_pct"] for t in losses) / len(losses))
                    edge = qm.quick_edge_check(wr, avg_w, avg_l)
                    if not edge["has_edge"]:
                        return (f"quant: {edge['verdict']} "
                                f"(E={edge['expectancy']:.2f}, "
                                f"RoR={edge['risk_of_ruin']:.1%})")
                    # Store for logging
                    analysis["quant_expectancy"] = edge["expectancy"]
                    analysis["quant_ror"] = edge["risk_of_ruin"]
                    analysis["quant_optimal_pct"] = edge["half_kelly_pct"]

                    # Max drawdown circuit breaker — block if MDD > 30%
                    equity = [10000.0]
                    for t in sym_closed:
                        equity.append(equity[-1] + t.get("pnl", 0))
                    mdd = qm.max_drawdown(equity)
                    if mdd["mdd_pct"] > 30:
                        return (f"quant: max drawdown {mdd['mdd_pct']:.1f}% "
                                f"(limit 30%)")
            except Exception:
                pass
        elif len(sym_recent) >= 5:
            # Fallback: simple win rate check
            sym_stats = self.store.get_recent_symbol_stats(symbol, n=20)
            if sym_stats["win_rate"] < 40:
                return f"poor history: {sym_stats['win_rate']:.0f}% win rate"

        # Compute Sharpe/Sortino/Alpha/IR for sizing multipliers (not blocking)
        if len(sym_recent) >= 20:
            try:
                from qor.quant import QuantMetrics
                qm = QuantMetrics()
                returns = [t.get("pnl_pct", 0.0) for t in sym_recent]
                analysis["quant_sharpe"] = qm.sharpe_ratio(returns)
                analysis["quant_sortino"] = qm.sortino_ratio(returns)
                # CAPM Alpha + Information Ratio vs buy-and-hold
                bench = analysis.get("price_returns", [])
                if bench and len(returns) >= 10:
                    n = min(len(returns), len(bench))
                    analysis["quant_alpha"] = qm.capm_alpha(returns[:n], bench[:n])
                    analysis["quant_ir"] = qm.information_ratio(returns[:n], bench[:n])
            except Exception:
                pass

        # Calendar event gate: skip entry if high-impact event imminent
        cal_minutes = getattr(self.config, "calendar_block_minutes", 60)
        for ev in analysis.get("calendar_events", []):
            if (ev.get("impact_level") == "high"
                    and 0 < ev.get("minutes_until_end", -1) <= cal_minutes):
                return (f"calendar event imminent: {ev['name']} "
                        f"({ev['minutes_until_end']:.0f}min)")

        return ""

    # ------------------------------------------------------------------
    # Kelly Criterion position sizing
    # ------------------------------------------------------------------

    def _kelly_pct(self, symbol: str, analysis: dict = None) -> float:
        """Kelly Criterion position size as % of available capital.

        Enhanced with:
        - Edge Decay: reduce sizing if expectancy declining over time
        - Volatility Sizing cap: cap position using target vol / asset vol

        Returns 0.0 if no edge (caller should skip entry).
        Returns max_position_pct if Kelly disabled or not enough history.
        """
        if not self.config.kelly_enabled:
            return self.config.max_position_pct

        closed = [t for t in self.store.trades.values()
                  if t["symbol"] == symbol and t["status"] != "open"]
        recent = closed[-30:]

        if len(recent) < self.config.kelly_min_trades:
            return self.config.max_position_pct  # Not enough data

        wins = [t for t in recent if t["pnl_pct"] > 0]
        losses = [t for t in recent if t["pnl_pct"] <= 0]
        if not wins or not losses:
            return self.config.max_position_pct

        p = len(wins) / len(recent)
        avg_win = sum(t["pnl_pct"] for t in wins) / len(wins)
        avg_loss = abs(sum(t["pnl_pct"] for t in losses) / len(losses))
        if avg_loss <= 0:
            return self.config.max_position_pct

        b = avg_win / avg_loss  # Payoff ratio
        q = 1.0 - p

        # Kelly: f* = (b*p - q) / b
        kelly_full = (b * p - q) / b

        if kelly_full <= 0:
            logger.info(
                f"[Kelly] {symbol}: NO EDGE (f*={kelly_full:.3f}, "
                f"WR={p*100:.0f}%, b={b:.2f}, {len(recent)} trades)")
            return 0.0

        kelly_adj = kelly_full * self.config.kelly_fraction
        kelly_pct = kelly_adj * 100.0

        # Edge Decay — compare recent 10 vs older 10 expectancy
        # If edge is decaying, reduce Kelly by 30%
        if len(recent) >= 20:
            try:
                from qor.quant import QuantMetrics
                qm = QuantMetrics()
                older = recent[:len(recent)//2]
                newer = recent[len(recent)//2:]
                w_old = [t for t in older if t["pnl"] > 0]
                l_old = [t for t in older if t["pnl"] <= 0]
                w_new = [t for t in newer if t["pnl"] > 0]
                l_new = [t for t in newer if t["pnl"] <= 0]
                if w_old and l_old and w_new and l_new:
                    e_old = qm.expectancy(
                        len(w_old), len(l_old),
                        sum(t["pnl_pct"] for t in w_old) / len(w_old),
                        abs(sum(t["pnl_pct"] for t in l_old) / len(l_old)))
                    e_new = qm.expectancy(
                        len(w_new), len(l_new),
                        sum(t["pnl_pct"] for t in w_new) / len(w_new),
                        abs(sum(t["pnl_pct"] for t in l_new) / len(l_new)))
                    if e_old > 0 and e_new < e_old * 0.5:
                        # Edge decaying by >50% — reduce Kelly
                        kelly_pct *= 0.7
                        logger.info(
                            f"[Kelly] {symbol}: Edge Decay — "
                            f"E_old={e_old:.3f} → E_new={e_new:.3f}, "
                            f"reducing Kelly 30%")
            except Exception:
                pass

        # Volatility Sizing cap — cap position by target vol / asset vol
        # Prevents over-allocation to high-volatility assets
        if analysis and analysis.get("asset_vol", 0) > 0:
            target_vol = 0.10  # 10% target portfolio volatility
            asset_vol = analysis["asset_vol"]
            vol_cap_frac = target_vol / asset_vol
            vol_cap_pct = vol_cap_frac * 100.0
            if vol_cap_pct < kelly_pct:
                logger.info(
                    f"[Kelly] {symbol}: Vol cap {vol_cap_pct:.1f}% "
                    f"(target_vol=10%, asset_vol={asset_vol:.1%}) "
                    f"< Kelly {kelly_pct:.1f}%")
                kelly_pct = vol_cap_pct

        # Clamp: floor at kelly_min_pct, ceiling at max_position_pct
        kelly_pct = max(kelly_pct, self.config.kelly_min_pct)
        kelly_pct = min(kelly_pct, self.config.max_position_pct)

        logger.info(
            f"[Kelly] {symbol}: f*={kelly_full:.3f} "
            f"x{self.config.kelly_fraction} = {kelly_pct:.1f}% "
            f"(WR={p*100:.0f}%, b={b:.2f}, {len(recent)} trades)")
        return kelly_pct


# ==============================================================================
# TradingEngine — Main 5-Minute Loop
# ==============================================================================

class TradingEngine:
    """AI automated trading with Binance Spot Demo Mode.

    Every 5 minutes per symbol:
      analyze → decide (list of actions) → execute each → save
    """

    def __init__(self, config, tool_executor=None, hmm=None, client=None,
                 cortex=None):
        self.config = config.trading
        # Accept pre-built client or create BinanceClient from config
        if client is not None:
            self.client = client
        else:
            self.client = BinanceClient(
                self.config.api_key, self.config.api_secret,
                testnet=self.config.testnet,
            )
        trades_path = os.path.join(self.config.data_dir, "trades.parquet")
        self.store = TradeStore(path=trades_path)

        # HMM regime detection — shared instance passed from runtime
        self.hmm = hmm
        if not hmm and _HAS_HMM:
            try:
                self.hmm = MarketHMM(data_dir=self.config.data_dir)
                self.hmm.load()
                logger.warning("[Trading] HMM created locally (not shared with futures)")
            except Exception:
                pass

        self.manager = PositionManager(
            self.config, self.client, self.store,
            tool_executor=tool_executor,
            hmm=self.hmm,
            cortex=cortex,
        )
        self._stop_event = threading.Event()
        self._thread = None
        self._tick_count = 0
        self._last_tick = None
        self._last_error = None
        self._activity_log = deque(maxlen=100)  # Thread-safe rolling log

    def _log_activity(self, symbol: str, action: str, detail: str = ""):
        """Add entry to the rolling activity log."""
        entry = {
            "time": datetime.now(timezone.utc).isoformat(),
            "tick": self._tick_count,
            "symbol": symbol,
            "action": action,
            "detail": detail,
        }
        self._activity_log.append(entry)

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="qor-trading")
        self._thread.start()
        trade_mode = getattr(self.config, 'trade_mode', 'scalp').upper()
        logger.info(f"[Trading] Engine started — symbols: {self.config.symbols}, "
                     f"interval: {self.config.check_interval_seconds}s, "
                     f"trade_mode: {trade_mode}, "
                     f"mode: {'DEMO' if self.config.testnet else 'PRODUCTION'}")

    def _loop(self):
        while not self._stop_event.is_set():
            try:
                self._tick()
                self._last_error = None
            except Exception as e:
                self._last_error = str(e)
                logger.error(f"[Trading] Tick error: {e}")
            self._stop_event.wait(timeout=self.config.check_interval_seconds)

    def _reconcile_positions(self, exchange_balances: dict = None):
        """Sync store with Binance — close trades whose asset balance is gone.

        Args:
            exchange_balances: Pre-fetched {symbol: balance} from _get_exchange_balances().
                If None, fetches account data (one API call for all symbols).
        """
        open_trades = self.store.get_open_trades()
        if not open_trades:
            return
        DUST_USD_THRESHOLD = 1.0
        # Fetch all balances in one API call if not provided
        if exchange_balances is None:
            try:
                account = self.client.get_account()
                exchange_balances = {}
                for b in account.get("balances", []):
                    total = float(b.get("free", 0)) + float(b.get("locked", 0))
                    if total > 0:
                        exchange_balances[b["asset"]] = total
            except Exception:
                return
        for trade in open_trades:
            sym = trade["symbol"]
            balance = exchange_balances.get(sym, 0)
            pair = self.client.format_pair(sym)
            try:
                price = self.client.get_price(pair)
            except Exception:
                price = trade.get("entry_price", 0)
            usd_value = balance * price if price > 0 else 0
            if balance <= 0 or usd_value < DUST_USD_THRESHOLD:
                reason = ("Asset balance is zero on exchange" if balance <= 0
                          else f"Dust balance: {balance:.8f} (${usd_value:.2f})")
                self.store.close_trade(
                    trade_id=trade["trade_id"],
                    exit_price=price,
                    status="closed_reconciled",
                    reason=reason,
                )
                logger.info(f"[Trading] Reconciled: {trade['trade_id']} {sym} "
                            f"closed ({reason})")
                self._log_activity(sym, "CLOSE",
                                   f"Reconciled — {reason}")

    def _get_exchange_balances(self) -> dict:
        """Get non-dust asset balances from Binance for configured symbols.

        Returns {symbol: total_balance} where total = free + locked.
        Only returns symbols we're configured to trade.
        Filters out dust (balance worth < $1 USD).
        """
        DUST_USD_THRESHOLD = 1.0  # Ignore balances worth less than $1
        result = {}
        try:
            account = self.client.get_account()
            symbols_set = set(self.config.symbols)
            for b in account.get("balances", []):
                asset = b["asset"]
                if asset not in symbols_set:
                    continue
                total = float(b.get("free", 0)) + float(b.get("locked", 0))
                if total > 0:
                    # Check USD value to filter dust
                    try:
                        price = self.client.get_price(self.client.format_pair(asset))
                        usd_value = total * price
                        if usd_value < DUST_USD_THRESHOLD:
                            logger.debug(f"[Trading] Ignoring {asset} dust: "
                                         f"{total:.8f} (${usd_value:.2f})")
                            continue
                    except Exception:
                        pass  # If can't get price, include it
                    result[asset] = total
        except Exception as e:
            logger.warning(f"[Trading] Cannot fetch exchange balances: {e}")
        return result

    def _tick(self):
        """One 5-minute cycle: check Binance first, then analyze and act.

        Flow per symbol:
          1. Fetch REAL balances from Binance (source of truth)
          2. Reconcile ghost trades (local store says open but exchange has 0)
          3. For each symbol with balance → manage position
          4. For symbols without balance → check limits, then evaluate entry
        """
        self._tick_count += 1
        self._last_tick = datetime.now(timezone.utc).isoformat()
        logger.info(f"[Trading] === Tick #{self._tick_count} ===")
        self._log_activity("ALL", "SCAN", f"Tick #{self._tick_count} started")

        # Step 1: Check what's ACTUALLY held on Binance (one API call)
        exchange_balances = self._get_exchange_balances()

        # Sync store with exchange — reuse balances (avoids N+1 API calls)
        self._reconcile_positions(exchange_balances)
        open_count = len(exchange_balances)
        if exchange_balances:
            logger.info(f"[Trading] Holdings on Binance: "
                        f"{', '.join(f'{s}={b:.6f}' for s, b in exchange_balances.items())}")

        # Step 2: Check allocated fund
        allocated = self.config.allocated_fund
        used = 0.0
        if allocated > 0:
            used = sum(t.get("cost_basis", 0) for t in self.store.get_open_trades())
            logger.info(f"[Trading] Fund: ${used:,.2f} / ${allocated:,.2f} used")

        for symbol in self.config.symbols:
            try:
                has_balance = exchange_balances.get(symbol, 0) > 0
                existing = self.store.get_symbol_open(symbol) if has_balance else None

                if has_balance and existing:
                    # Position EXISTS on Binance + local store — manage it
                    # Always sync quantity with exchange (Binance is source of truth)
                    exchange_qty = exchange_balances[symbol]
                    if exchange_qty != existing["quantity"]:
                        existing["quantity"] = exchange_qty
                    decisions = self.manager.evaluate_symbol(symbol)
                elif has_balance and not existing:
                    # Balance on exchange but no local trade — skip (manual buy)
                    self._log_activity(symbol, "HOLD",
                        f"Balance {exchange_balances[symbol]:.6f} but no local trade")
                    continue
                else:
                    # No position — check limits before evaluating entry
                    if open_count >= self.config.max_open_positions:
                        self._log_activity(symbol, "HOLD",
                            f"Max positions ({open_count}/{self.config.max_open_positions})")
                        continue
                    if allocated > 0 and used >= allocated:
                        self._log_activity(symbol, "HOLD",
                            f"Fund limit (${used:,.0f}/${allocated:,.0f})")
                        continue
                    decisions = self.manager.evaluate_symbol(symbol)

                for decision in decisions:
                    action = decision["action"]
                    reason = decision.get('reason', '')
                    cortex_info = ""
                    if decision.get("cortex_label"):
                        cortex_info = (f" [CORTEX:{decision['cortex_label']} "
                                       f"{decision.get('cortex_signal', 0):.2f}]")
                    logger.info(f"[Trading] {symbol}: {action} — {reason}{cortex_info}")
                    self._log_activity(symbol, action, f"{reason}{cortex_info}")

                    if action == "BUY":
                        self._execute_buy(symbol, decision)
                    elif action == "SELL":
                        self._execute_sell(symbol, decision)
                    elif action == "PARTIAL_TP":
                        self._execute_partial_tp(symbol, decision)
                    elif action == "DCA":
                        self._execute_dca(symbol, decision)
                    elif action == "ADJUST_SL":
                        self._execute_adjust_sl(decision)
                    elif action == "ADJUST_TP":
                        self._execute_adjust_tp(decision)
                    # HOLD → no action
                if not decisions:
                    self._log_activity(symbol, "HOLD", "No action needed")
            except Exception as e:
                logger.error(f"[Trading] Error processing {symbol}: {e}")
                self._log_activity(symbol, "ERROR", str(e))

        self.store.save()

    # ------------------------------------------------------------------
    # Execution methods
    # ------------------------------------------------------------------

    def _execute_buy(self, symbol: str, decision: dict):
        """Place initial buy order + record in TradeStore."""
        pair = self.client.format_pair(symbol)
        quote = getattr(self.client, 'quote', 'USDT')

        try:
            quote_balance = self.client.get_balance(quote)
        except Exception as e:
            logger.error(f"[Trading] Cannot get balance: {e}")
            return

        # Allocated fund cap
        allocated = self.config.allocated_fund
        if allocated > 0:
            used = sum(t.get("cost_basis", 0) for t in self.store.get_open_trades())
            available = min(quote_balance, allocated - used)
        else:
            available = quote_balance

        # Position size computed in _evaluate_entry (Kelly + signal multipliers)
        position_pct = decision.get("position_pct", self.config.max_position_pct)
        position_usdt = available * (position_pct / 100.0)
        if position_usdt < 10:
            logger.info(f"[Trading] Insufficient fund: ${available:.2f} "
                         f"(balance=${quote_balance:.2f}, allocated=${allocated:.2f})")
            return

        try:
            price = self.client.get_price(pair)
        except Exception as e:
            logger.error(f"[Trading] Cannot get price for {pair}: {e}")
            return
        if price <= 0:
            return

        quantity = self.client.round_qty(pair, position_usdt / price)
        try:
            lot = self.client.get_lot_size(pair)
            if quantity < lot["minQty"]:
                return
        except Exception:
            pass

        try:
            order = self.client.place_order(pair, "BUY", quantity)
            fills = order.get("fills", [])
            # Weighted average fill price across all fill levels
            if fills:
                total_fill_qty = sum(float(f["qty"]) for f in fills)
                fill_price = (sum(float(f["price"]) * float(f["qty"]) for f in fills)
                              / total_fill_qty) if total_fill_qty > 0 else price
            else:
                fill_price = price
            fill_qty = float(order.get("executedQty", quantity))
            # Use actual exchange balance as truth for received quantity.
            # Commission may be in BNB (if fee discount enabled) or in the
            # asset itself — querying actual balance handles both cases.
            try:
                actual_received = self.client.get_balance(symbol)
                if actual_received > 0 and actual_received < fill_qty:
                    logger.info(f"[Trading] BUY {pair}: order qty={fill_qty:.8f}, "
                                f"actual balance={actual_received:.8f} "
                                f"(commission deducted)")
                    fill_qty = actual_received
            except Exception:
                # Fallback: try per-fill commission deduction
                total_commission = 0.0
                for fill in fills:
                    if fill.get("commissionAsset", "").upper() == symbol.upper():
                        total_commission += float(fill.get("commission", 0))
                if total_commission > 0:
                    fill_qty -= total_commission
        except Exception as e:
            logger.error(f"[Trading] BUY failed {pair}: {e}")
            return

        if fill_qty <= 0:
            logger.warning(f"[Trading] BUY {pair}: zero fill, skipping")
            return

        # Reason already includes Kelly + signal multiplier info from _evaluate_entry
        reason = decision["reason"]

        self.store.open_trade(
            symbol=symbol, side="BUY",
            entry_price=fill_price, quantity=fill_qty,
            stop_loss=decision["stop_loss"],
            take_profit=decision["take_profit"],
            take_profit2=decision.get("take_profit2", 0),
            reason=reason,
        )

    def _execute_sell(self, symbol: str, decision: dict):
        """Close entire position."""
        pair = self.client.format_pair(symbol)
        trade_id = decision.get("trade_id", "")
        trade = self.store.trades.get(trade_id)
        if not trade:
            return

        # Use ACTUAL exchange balance (not store qty) to avoid fee mismatch.
        # Binance deducts ~0.1% commission from received asset on BUY,
        # so store qty is slightly higher than real balance.
        try:
            actual_balance = self.client.get_balance(symbol)
        except Exception:
            actual_balance = trade["quantity"]
        quantity = self.client.round_qty(pair, min(trade["quantity"], actual_balance))
        if quantity <= 0:
            return

        try:
            order = self.client.place_order(pair, "SELL", quantity)
            fills = order.get("fills", [])
            if fills:
                total_qty = sum(float(f["qty"]) for f in fills)
                fill_price = (sum(float(f["price"]) * float(f["qty"]) for f in fills)
                              / total_qty) if total_qty > 0 else 0
            else:
                fill_price = 0
            if fill_price <= 0:
                fill_price = self.client.get_price(pair)
        except Exception as e:
            logger.error(f"[Trading] SELL failed {pair}: {e}")
            return

        self.store.close_trade(
            trade_id=trade_id, exit_price=fill_price,
            status=decision.get("exit_status", "closed_manual"),
            reason=decision["reason"],
        )

    def _execute_partial_tp(self, symbol: str, decision: dict):
        """Sell a portion of position (partial take profit)."""
        pair = self.client.format_pair(symbol)
        trade_id = decision["trade_id"]
        trade = self.store.trades.get(trade_id)
        if not trade:
            return

        # Cap sell_qty to actual exchange balance (fee mismatch protection)
        try:
            actual_balance = self.client.get_balance(symbol)
        except Exception:
            actual_balance = trade["quantity"]
        sell_qty = self.client.round_qty(pair, min(decision["sell_qty"], actual_balance))
        if sell_qty <= 0:
            return

        try:
            order = self.client.place_order(pair, "SELL", sell_qty)
            fills = order.get("fills", [])
            if fills:
                total_qty = sum(float(f["qty"]) for f in fills)
                fill_price = (sum(float(f["price"]) * float(f["qty"]) for f in fills)
                              / total_qty) if total_qty > 0 else 0
            else:
                fill_price = 0
            if fill_price <= 0:
                fill_price = self.client.get_price(pair)
        except Exception as e:
            logger.error(f"[Trading] PARTIAL_TP failed {pair}: {e}")
            return

        self.store.partial_close(
            trade_id=trade_id, sell_qty=sell_qty,
            sell_price=fill_price, reason=decision["reason"],
        )

    def _execute_dca(self, symbol: str, decision: dict):
        """DCA: buy more of the asset to lower average entry."""
        pair = self.client.format_pair(symbol)
        quote = getattr(self.client, 'quote', 'USDT')
        trade_id = decision["trade_id"]
        trade = self.store.trades.get(trade_id)
        if not trade:
            return

        try:
            quote_balance = self.client.get_balance(quote)
        except Exception as e:
            logger.error(f"[Trading] Cannot get balance for DCA: {e}")
            return

        # DCA size = original position size × multiplier^dca_count
        dca_num = decision.get("dca_number", 1)
        base_usdt = trade.get("cost_basis", 0) / max(trade.get("dca_count", 0) + 1, 1)
        dca_usdt = base_usdt * (self.config.dca_multiplier ** (dca_num - 1))
        dca_usdt = min(dca_usdt, quote_balance * 0.5)  # Never use more than 50% remaining

        if dca_usdt < 10:
            logger.info(f"[Trading] DCA skipped: insufficient balance ${quote_balance:.2f}")
            return

        try:
            price = self.client.get_price(pair)
        except Exception:
            return
        if price <= 0:
            return

        quantity = self.client.round_qty(pair, dca_usdt / price)

        try:
            order = self.client.place_order(pair, "BUY", quantity)
            fills = order.get("fills", [])
            if fills:
                total_fill_qty = sum(float(f["qty"]) for f in fills)
                fill_price = (sum(float(f["price"]) * float(f["qty"]) for f in fills)
                              / total_fill_qty) if total_fill_qty > 0 else price
            else:
                fill_price = price
            fill_qty = float(order.get("executedQty", quantity))
            # Use actual balance to determine net received qty (handles BNB fees)
            try:
                actual_balance = self.client.get_balance(symbol)
                prev_qty = trade.get("quantity", 0)
                if actual_balance > prev_qty:
                    fill_qty = actual_balance - prev_qty
            except Exception:
                # Fallback: try per-fill commission deduction
                for fill in fills:
                    if fill.get("commissionAsset", "").upper() == symbol.upper():
                        fill_qty -= float(fill.get("commission", 0))
        except Exception as e:
            logger.error(f"[Trading] DCA BUY failed {pair}: {e}")
            return

        self.store.add_dca(trade_id=trade_id, add_price=fill_price, add_qty=fill_qty)

        # Recalculate SL based on new average entry using actual ATR from decision
        new_entry = trade["entry_price"]  # Already updated by add_dca
        atr_value = decision.get("atr", 0)
        if atr_value > 0:
            new_sl = new_entry - atr_value * self.config.stop_loss_atr_mult
        else:
            # Fallback: 2% below new average entry
            new_sl = new_entry * (1 - 0.02 * self.config.stop_loss_atr_mult)
        if new_sl < trade["stop_loss"]:
            self.store.update_sl_tp(trade_id, new_sl=new_sl)

    def _execute_adjust_sl(self, decision: dict):
        """Adjust stop loss."""
        trade_id = decision.get("trade_id", "")
        new_sl = decision.get("new_sl", 0)
        if trade_id and new_sl > 0:
            self.store.update_sl_tp(trade_id, new_sl=new_sl)

    def _execute_adjust_tp(self, decision: dict):
        """Adjust take profit."""
        trade_id = decision.get("trade_id", "")
        new_tp = decision.get("new_tp", 0)
        if trade_id and new_tp > 0:
            self.store.update_sl_tp(trade_id, new_tp=new_tp)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
        if self.store._dirty:
            self.store.save()
        # Persist CORTEX and HMM state on shutdown
        if self.manager.cortex:
            try:
                cortex_path = os.path.join(self.config.data_dir, "cortex_shared.pt")
                self.manager.cortex.save(cortex_path)
                logger.info(f"[Trading] CORTEX saved to {cortex_path}")
            except Exception as e:
                logger.warning(f"[Trading] CORTEX save failed: {e}")
        if self.hmm:
            try:
                self.hmm.save()
                logger.info("[Trading] HMM saved")
            except Exception as e:
                logger.warning(f"[Trading] HMM save failed: {e}")
        logger.info("[Trading] Engine stopped")

    def status(self) -> dict:
        stats = self.store.get_stats()
        trade_mode = getattr(self.config, 'trade_mode', 'scalp')
        result = {
            "enabled": True,
            "mode": "DEMO" if self.config.testnet else "PRODUCTION",
            "trade_mode": trade_mode.upper(),
            "symbols": self.config.symbols,
            "open_positions": len(self.store.get_open_trades()),
            "total_trades": stats.get("total_trades", 0),
            "tick_count": self._tick_count,
            "last_tick": self._last_tick,
            "last_error": self._last_error,
            "stats": stats,
            "activity": list(self._activity_log)[-20:],
        }
        # CORTEX analyzer status
        if self.manager.cortex:
            result["cortex"] = self.manager.cortex.status()
        # HMM regime detection status
        if self.hmm and self.hmm.is_available:
            result["hmm"] = self.hmm.status()
        return result


# Backward compatibility alias
CfCMarketAnalyzer = CortexAnalyzer

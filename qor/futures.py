"""
QOR Futures Engine — Multi-Exchange Futures Trading
=====================================================
Autonomous futures bot that runs alongside the spot engine:
- Analyzes all symbols via multi-timeframe technical analysis
- Opens LONG or SHORT positions based on bias
- Leverage-aware sizing (1x-10x, default 5x, isolated margin)
- Funding rate awareness — skips extreme funding against direction
- Liquidation tracking + warnings
- DCA, partial TP, trailing stop, break-even protection
- Direction-agnostic SL/TP (inverted for SHORT positions)

Architecture:
  ExchangeClient        — Generic interface (futures methods optional)
  BinanceFuturesClient  — Binance USDT-M Futures API (/fapi/)
  create_futures_client — Factory: creates the right client by exchange name
  FuturesTradeStore     — Parquet database with futures-specific columns
  FuturesPositionManager — LONG + SHORT decision engine
  FuturesEngine         — Background thread loop orchestrator (any exchange)
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

# Import CORTEX analyzer from trading module (shared implementation)
try:
    from qor.trading import CortexAnalyzer, parse_analysis
    _HAS_CORTEX = True
except ImportError:
    _HAS_CORTEX = False

# HMM regime detection (from qor.quant)
try:
    from qor.quant import MarketHMM, HMMSignal
    _HAS_HMM = True
except ImportError:
    _HAS_HMM = False


# ==============================================================================
# BinanceFuturesClient — USDT-M Futures API with HMAC-SHA256
# ==============================================================================

class BinanceFuturesClient:
    """Binance USDT-M Futures API client with HMAC-SHA256 signing.

    Supports Testnet (testnet.binancefuture.com) and production (fapi.binance.com).
    Uses /fapi/ endpoints for USDT-M perpetual futures.

    Implements the ExchangeClient futures interface so FuturesEngine
    can work with any exchange — not just Binance.
    """

    TESTNET_URL = "https://testnet.binancefuture.com"
    PRODUCTION_URL = "https://fapi.binance.com"

    name = "binance_futures"
    quote = "USDT"
    supports_futures = True

    # Futures API weight limit: 2400/min
    _WEIGHT_LIMIT = 2400

    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = self.TESTNET_URL if testnet else self.PRODUCTION_URL
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

    def _sync_time(self):
        """Sync local clock with Binance server to avoid recvWindow errors."""
        for attempt in range(3):
            try:
                url = f"{self.base_url}/fapi/v1/time"
                req = urllib.request.Request(url, headers={"User-Agent": "QOR-Futures/1.0"})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    server_time = json.loads(resp.read().decode())["serverTime"]
                local_time = int(time.time() * 1000)
                self._time_offset = server_time - local_time
                self._time_synced = True
                if abs(self._time_offset) > 1000:
                    logger.info(f"[Futures] Clock offset: {self._time_offset}ms")
                return
            except Exception as e:
                if attempt < 2:
                    time.sleep(1)
                else:
                    logger.warning(f"[Futures] Time sync failed after 3 attempts: {e}")

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

    def _track_weight(self, weight: int = 1):
        """Track API weight usage and sleep if approaching limit."""
        now = time.time()
        if now - self._weight_window_start > 60:
            self._api_weight = 0
            self._weight_window_start = now
        self._api_weight += weight
        if self._api_weight > self._WEIGHT_LIMIT - 400:  # leave 400 headroom
            sleep_time = 60 - (now - self._weight_window_start)
            if sleep_time > 0:
                logger.warning(f"[Futures] Rate limit approaching "
                               f"({self._api_weight}/{self._WEIGHT_LIMIT}), "
                               f"sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
                self._api_weight = 0
                self._weight_window_start = time.time()

    def _request(self, method: str, endpoint: str, params: dict = None,
                 signed: bool = True, _retry: bool = False) -> dict:
        """Make API request with auto-retry on timeout and timestamp errors."""
        params = params or {}
        max_attempts = 1 if _retry else 2  # retry once on timeout

        # Track API weight before request
        weight = 10 if "account" in endpoint or "positionRisk" in endpoint else 1
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
            req.add_header("User-Agent", "QOR-Futures/1.0")

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
                    logger.warning("[Futures] Timestamp error, re-syncing clock...")
                    self._sync_time()
                    return self._request(method, endpoint, params, signed=True, _retry=True)
                logger.error(f"[Futures] {method} {endpoint}: HTTP {e.code} — {body}")
                raise RuntimeError(f"Binance Futures API error {e.code}: {body}")
            except (socket.timeout, urllib.error.URLError, OSError) as e:
                is_timeout = "timed out" in str(e).lower() or isinstance(e, socket.timeout)
                if is_timeout and attempt < max_attempts - 1:
                    logger.warning(f"[Futures] {method} {endpoint}: timeout, retrying ({attempt+1}/{max_attempts})...")
                    self._sync_time()  # re-sync clock after timeout
                    time.sleep(2)
                    continue
                logger.error(f"[Futures] {method} {endpoint}: {e}")
                raise
            except Exception as e:
                logger.error(f"[Futures] {method} {endpoint}: {e}")
                raise

    # --- Historical Klines ---

    def get_klines(self, symbol: str, interval: str = "1h",
                   limit: int = 1500, start_time: int = None,
                   end_time: int = None) -> list:
        """Fetch historical klines (candlesticks) from Binance Futures.

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
        return self._request("GET", "/fapi/v1/klines", params, signed=False)

    # --- Account & Position ---

    def get_account(self) -> dict:
        """Get futures account info (balances + positions)."""
        return self._request("GET", "/fapi/v2/account")

    def get_balance(self, asset: str = "USDT") -> float:
        """Get available balance for an asset."""
        account = self.get_account()
        for b in account.get("assets", []):
            if b["asset"] == asset.upper():
                return float(b["availableBalance"])
        return 0.0

    def get_positions(self) -> list:
        """Get all position info."""
        return self._request("GET", "/fapi/v2/positionRisk")

    def get_position(self, symbol: str) -> dict:
        """Get position for a specific symbol."""
        positions = self._request(
            "GET", "/fapi/v2/positionRisk",
            params={"symbol": symbol},
        )
        for p in positions:
            if p["symbol"] == symbol:
                return p
        return {}

    # --- Market Data ---

    def get_mark_price(self, symbol: str) -> dict:
        """Get mark price + funding rate for a symbol."""
        data = self._request(
            "GET", "/fapi/v1/premiumIndex",
            params={"symbol": symbol}, signed=False,
        )
        return {
            "mark_price": float(data.get("markPrice", 0)),
            "funding_rate": float(data.get("lastFundingRate", 0)),
            "next_funding_time": int(data.get("nextFundingTime", 0)),
        }

    def get_price(self, symbol: str) -> float:
        """Get last price for a trading pair."""
        data = self._request(
            "GET", "/fapi/v1/ticker/price",
            params={"symbol": symbol}, signed=False,
        )
        return float(data.get("price", 0))

    # --- Leverage & Margin ---

    def set_leverage(self, symbol: str, leverage: int) -> dict:
        """Set leverage for a symbol (1-125)."""
        try:
            return self._request(
                "POST", "/fapi/v1/leverage",
                params={"symbol": symbol, "leverage": leverage},
            )
        except RuntimeError as e:
            # Already set — not an error
            if "No need to change" in str(e):
                return {"leverage": leverage, "symbol": symbol}
            raise

    def set_margin_type(self, symbol: str, margin_type: str) -> dict:
        """Set margin type: ISOLATED or CROSSED."""
        try:
            return self._request(
                "POST", "/fapi/v1/marginType",
                params={"symbol": symbol, "marginType": margin_type},
            )
        except RuntimeError as e:
            # Already set — not an error
            if "No need to change" in str(e):
                return {"marginType": margin_type, "symbol": symbol}
            raise

    def set_hedge_mode(self, hedge: bool = True) -> dict:
        """Enable/disable hedge mode (dual position side).
        Hedge mode allows simultaneous LONG and SHORT on the same symbol.
        """
        try:
            return self._request(
                "POST", "/fapi/v1/positionSide/dual",
                params={"dualSidePosition": str(hedge).lower()},
            )
        except RuntimeError as e:
            if "No need to change" in str(e):
                return {"dualSidePosition": hedge}
            raise

    def set_multi_asset_mode(self, multi: bool = True) -> dict:
        """Enable/disable multi-asset margin mode.
        Multi-asset mode allows using multiple currencies as margin.
        """
        try:
            return self._request(
                "POST", "/fapi/v1/multiAssetsMargin",
                params={"multiAssetsMargin": str(multi).lower()},
            )
        except RuntimeError as e:
            if "No need to change" in str(e):
                return {"multiAssetsMargin": multi}
            raise

    # --- Orders ---

    def place_order(self, symbol: str, side: str, quantity: float,
                    order_type: str = "MARKET", price: float = None,
                    reduce_only: bool = False,
                    position_side: str = None) -> dict:
        """Place a futures order.

        In hedge mode, position_side must be "LONG" or "SHORT".
        - Open LONG:  side=BUY,  position_side=LONG
        - Close LONG: side=SELL, position_side=LONG
        - Open SHORT: side=SELL, position_side=SHORT
        - Close SHORT: side=BUY, position_side=SHORT
        """
        params = {
            "symbol": symbol,
            "side": side.upper(),
            "type": order_type,
            "quantity": f"{quantity:.8f}".rstrip("0").rstrip("."),
        }
        if position_side:
            params["positionSide"] = position_side.upper()
        elif reduce_only:
            params["reduceOnly"] = "true"
        if order_type == "MARKET":
            params["newOrderRespType"] = "RESULT"
        if order_type == "LIMIT" and price is not None:
            params["timeInForce"] = "GTC"
            params["price"] = f"{price:.8f}".rstrip("0").rstrip(".")
        if order_type in ("STOP_MARKET", "TAKE_PROFIT_MARKET"):
            if price is not None:
                params["stopPrice"] = f"{price:.8f}".rstrip("0").rstrip(".")
            params["closePosition"] = "true"
            params.pop("quantity", None)

        return self._request("POST", "/fapi/v1/order", params)

    def cancel_order(self, symbol: str, order_id: int) -> dict:
        """Cancel an open order."""
        return self._request(
            "DELETE", "/fapi/v1/order",
            params={"symbol": symbol, "orderId": order_id},
        )

    def get_open_orders(self, symbol: str = None) -> list:
        """Get all open orders."""
        params = {}
        if symbol:
            params["symbol"] = symbol
        return self._request("GET", "/fapi/v1/openOrders", params)

    def get_all_orders(self, symbol: str, limit: int = 50) -> list:
        """Get recent orders (open + closed)."""
        return self._request("GET", "/fapi/v1/allOrders",
                             params={"symbol": symbol, "limit": limit})

    def get_trade_history(self, symbol: str, limit: int = 50) -> list:
        """Get filled trades for a symbol."""
        return self._request("GET", "/fapi/v1/userTrades",
                             params={"symbol": symbol, "limit": limit})

    def get_income(self, income_type: str = None, limit: int = 100) -> list:
        """Get income history (REALIZED_PNL, FUNDING_FEE, COMMISSION, etc)."""
        params = {"limit": limit}
        if income_type:
            params["incomeType"] = income_type
        return self._request("GET", "/fapi/v1/income", params)

    # --- Exchange Info ---

    def get_lot_size(self, symbol: str) -> dict:
        """Get LOT_SIZE filter for a symbol (min/max qty, step)."""
        if symbol in self._lot_cache:
            return self._lot_cache[symbol]
        data = self._request(
            "GET", "/fapi/v1/exchangeInfo", signed=False,
        )
        for s in data.get("symbols", []):
            if s["symbol"] == symbol:
                for f in s.get("filters", []):
                    if f["filterType"] == "LOT_SIZE":
                        lot = {
                            "minQty": float(f["minQty"]),
                            "maxQty": float(f["maxQty"]),
                            "stepSize": float(f["stepSize"]),
                        }
                        self._lot_cache[symbol] = lot
                        return lot
        return {"minQty": 0.001, "maxQty": 9999999, "stepSize": 0.001}

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
# FuturesTradeStore — Parquet Trade History with futures-specific columns
# ==============================================================================

FUTURES_TRADE_SCHEMA = pa.schema([
    ("trade_id",          pa.string()),
    ("symbol",            pa.string()),
    ("side",              pa.string()),     # BUY or SELL (order side)
    ("entry_price",       pa.float64()),
    ("exit_price",        pa.float64()),
    ("quantity",          pa.float64()),
    ("stop_loss",         pa.float64()),
    ("take_profit",       pa.float64()),    # TP1
    ("take_profit2",      pa.float64()),    # TP2
    ("status",            pa.string()),     # open, closed_tp, closed_sl, closed_manual, closed_reversal
    ("pnl",              pa.float64()),
    ("pnl_pct",          pa.float64()),
    ("strategy",          pa.string()),
    ("entry_reason",      pa.string()),
    ("exit_reason",       pa.string()),
    ("entry_time",        pa.int64()),
    ("exit_time",         pa.int64()),
    ("dca_count",         pa.int64()),
    ("tp1_hit",           pa.bool_()),
    ("original_qty",      pa.float64()),
    ("cost_basis",        pa.float64()),
    ("data_hash",         pa.string()),
    ("prev_hash",         pa.string()),
    # Futures-specific
    ("leverage",          pa.int64()),
    ("margin_type",       pa.string()),     # ISOLATED/CROSSED
    ("liquidation_price", pa.float64()),
    ("direction",         pa.string()),     # LONG/SHORT
    ("funding_paid",      pa.float64()),
    ("margin_used",       pa.float64()),    # USDT margin allocated
])


class FuturesTradeStore:
    """Parquet-based futures trade history with hash chain."""

    def __init__(self, path: str):
        self.path = path
        self.trades: dict = {}  # trade_id -> dict
        self._chain_head = ""
        self._dirty = False
        self._lock = threading.Lock()  # Protects self.trades across threads
        self.on_close = None  # Callback: on_close(trade_dict, engine_type)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            return
        try:
            table = pq.read_table(self.path)
            for i in range(table.num_rows):
                trade = {}
                for col_name in FUTURES_TRADE_SCHEMA.names:
                    if col_name in table.schema.names:
                        val = table.column(col_name)[i].as_py()
                        trade[col_name] = val if val is not None else self._default(col_name)
                    else:
                        trade[col_name] = self._default(col_name)
                self.trades[trade["trade_id"]] = trade
                if trade.get("data_hash"):
                    self._chain_head = trade["data_hash"]
            logger.info(f"[FuturesStore] Loaded {len(self.trades)} trades from {self.path}")
        except Exception as e:
            logger.warning(f"[FuturesStore] Failed to load {self.path}: {e}")

    def _default(self, col_name):
        field = FUTURES_TRADE_SCHEMA.field(col_name)
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
                   reason: str, direction: str = "LONG",
                   leverage: int = 1, margin_type: str = "ISOLATED",
                   liquidation_price: float = 0.0,
                   margin_used: float = 0.0,
                   strategy: str = "multi_tf_trend",
                   take_profit2: float = 0.0) -> str:
        """Open a new futures trade. Returns trade_id."""
        trade_id = str(uuid.uuid4())[:12]
        prev_hash = self._chain_head
        content = f"{trade_id}:{symbol}:{side}:{direction}:{entry_price}:{quantity}"
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
            # Futures-specific
            "leverage": leverage,
            "margin_type": margin_type,
            "liquidation_price": liquidation_price,
            "direction": direction,
            "funding_paid": 0.0,
            "margin_used": margin_used,
        }
        self.trades[trade_id] = trade
        self._dirty = True
        logger.info(f"[Futures] OPEN {direction} {symbol} {leverage}x: qty={quantity:.6f} "
                     f"entry=${entry_price:,.2f} SL=${stop_loss:,.2f} "
                     f"TP1=${take_profit:,.2f} TP2=${take_profit2:,.2f} "
                     f"liq=${liquidation_price:,.2f}")
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
        trade["entry_price"] = total_cost / total_qty
        trade["quantity"] = total_qty
        trade["cost_basis"] = total_cost
        trade["dca_count"] = trade.get("dca_count", 0) + 1
        trade["entry_reason"] += f" | DCA#{trade['dca_count']} @${add_price:,.2f}"
        self._dirty = True
        logger.info(f"[Futures] DCA #{trade['dca_count']} {trade['symbol']} "
                     f"{trade['direction']}: +{add_qty:.6f} @${add_price:,.2f} -> "
                     f"avg=${trade['entry_price']:,.2f} total={total_qty:.6f}")

    def partial_close(self, trade_id: str, sell_qty: float, sell_price: float,
                      reason: str):
        """Close a portion of the position (partial TP)."""
        trade = self.trades.get(trade_id)
        if not trade or trade["status"] != "open":
            return
        trade["quantity"] -= sell_qty
        trade["tp1_hit"] = True
        direction = trade["direction"]
        if direction == "LONG":
            partial_pnl = (sell_price - trade["entry_price"]) * sell_qty
        else:
            partial_pnl = (trade["entry_price"] - sell_price) * sell_qty
        trade["entry_reason"] += (f" | Partial @${sell_price:,.2f} "
                                  f"qty={sell_qty:.6f} pnl=${partial_pnl:+,.2f}")
        self._dirty = True
        logger.info(f"[Futures] PARTIAL {trade['symbol']} {direction}: sold {sell_qty:.6f} "
                     f"@${sell_price:,.2f} pnl=${partial_pnl:+,.2f} -- {reason}")

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

        direction = trade["direction"]
        if direction == "LONG":
            trade["pnl"] = (exit_price - trade["entry_price"]) * trade["quantity"]
            if trade["entry_price"] > 0:
                trade["pnl_pct"] = ((exit_price / trade["entry_price"]) - 1) * 100
        else:  # SHORT
            trade["pnl"] = (trade["entry_price"] - exit_price) * trade["quantity"]
            if exit_price > 0:
                trade["pnl_pct"] = ((trade["entry_price"] / exit_price) - 1) * 100

        self._dirty = True
        tag = "WIN" if trade["pnl"] > 0 else "LOSS"
        logger.info(f"[Futures] CLOSE {trade['symbol']} {direction} [{tag}]: "
                     f"${trade['entry_price']:,.2f} -> ${exit_price:,.2f} "
                     f"P&L: ${trade['pnl']:+,.2f} ({trade['pnl_pct']:+.2f}%) -- {reason}")

        # Notify AI learning system
        if self.on_close and trade.get("pnl") is not None:
            try:
                self.on_close(trade, "futures")
            except Exception as e:
                logger.debug(f"[FuturesStore] on_close callback error: {e}")

    def update_sl_tp(self, trade_id: str, new_sl: float = None,
                     new_tp: float = None):
        trade = self.trades.get(trade_id)
        if not trade or trade["status"] != "open":
            return
        if new_sl is not None:
            trade["stop_loss"] = new_sl
        if new_tp is not None:
            trade["take_profit"] = new_tp
        self._dirty = True

    def update_funding(self, trade_id: str, funding_amount: float):
        """Accumulate funding payment for a trade."""
        trade = self.trades.get(trade_id)
        if not trade or trade["status"] != "open":
            return
        trade["funding_paid"] = trade.get("funding_paid", 0.0) + funding_amount
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
                "avg_hold_hours": 0.0, "profit_factor": 0.0,
                "total_funding_paid": 0.0, "by_symbol": {},
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
            "total_funding_paid": sum(t.get("funding_paid", 0) for t in closed),
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
            data = {col: [] for col in FUTURES_TRADE_SCHEMA.names}
            for t in self.trades.values():
                for col in FUTURES_TRADE_SCHEMA.names:
                    data[col].append(t.get(col, self._default(col)))

        batch = pa.RecordBatch.from_pydict(data, schema=FUTURES_TRADE_SCHEMA)
        table = pa.Table.from_batches([batch])
        pq.write_table(table, self.path)
        self._dirty = False


# ==============================================================================
# FuturesPositionManager — LONG + SHORT Decision Engine
# ==============================================================================

class FuturesPositionManager:
    """Full AI decision engine for futures: can go LONG or SHORT.

    Key differences from spot PositionManager:
    1. Can go SHORT (when bias is BEARISH)
    2. Leverage-aware position sizing
    3. Liquidation tracking
    4. Funding rate awareness
    5. Direction-agnostic SL/TP (inverted for SHORT)
    """

    def __init__(self, config, client: BinanceFuturesClient,
                 store: FuturesTradeStore, tool_executor=None,
                 hmm: 'MarketHMM | None' = None,
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
                logger.info("[FuturesPM] CORTEX analyzer initialized (local instance) "
                            f"(obs d={CortexAnalyzer.OBS_DIM}, "
                            f"reflex {CortexAnalyzer.REFLEX_NEURONS} neurons, "
                            f"history {CortexAnalyzer.HISTORY_LEN} candles)")
            except Exception as e:
                logger.warning(f"[FuturesPM] CORTEX analyzer init failed: {e}")
        elif self.cortex is not None:
            logger.info("[FuturesPM] Using shared CORTEX analyzer")

    def _get_trade_atr(self, analysis: dict) -> float:
        """Return the ATR appropriate for the current trade_mode.

        - scalp:  5m → 30m  (tight SL/TP for fast trades)
        - stable: 30m → 4h  (medium swings)
        - secure: 4h → 1w   (wide SL/TP, safe)

        All modes still use ALL timeframes for trend direction (bias).
        This only affects SL/TP/trailing stop calculations.
        """
        mode = getattr(self.config, "trade_mode", "scalp")
        if mode == "scalp":
            # 5m to 30m range
            atr = (analysis.get("atr_15m") or analysis.get("atr_30m")
                   or analysis.get("atr_5m", 0))
        elif mode == "stable":
            # 30m to 4h range
            atr = (analysis.get("atr_1h") or analysis.get("atr_4h")
                   or analysis.get("atr_30m", 0))
        else:  # secure
            # 4h to 1w range
            atr = (analysis.get("atr_daily") or analysis.get("atr_weekly")
                   or analysis.get("atr_4h", 0))
        # Fallback: if mode-specific ATR is 0, use any available
        if atr <= 0:
            atr = (analysis.get("atr_daily") or analysis.get("atr")
                   or analysis.get("atr_1h") or analysis.get("atr_15m", 0))
        return atr

    def _get_reversal_threshold(self, total_tfs: int) -> int:
        """Mode-aware reversal threshold.

        - scalp:  2 TFs (5m-30m flip fast, don't wait for daily/weekly)
        - stable: 3 TFs (30m-4h, moderate patience)
        - secure: 4 TFs (4h-1w, high conviction needed)
        """
        mode = getattr(self.config, "trade_mode", "scalp")
        if mode == "scalp":
            return min(2, total_tfs)
        elif mode == "stable":
            return min(3, total_tfs)
        else:  # secure
            return max(4, int(total_tfs * 2 / 3))

    def _check_scalp_short_tf_reversal(self, analysis: dict,
                                        against: str) -> tuple:
        """Check if 2+ of 5m/15m/30m TFs flipped against position.

        against: "LONG" or "SHORT" — the direction we're checking against.
        For LONG: checks if short TFs turned bearish.
        For SHORT: checks if short TFs turned bullish.

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
            if against == "LONG" and bias == "BEARISH":
                flipped += 1
                tfs_checked.append(tf_name)
            elif against == "SHORT" and bias == "BULLISH":
                flipped += 1
                tfs_checked.append(tf_name)

        if flipped >= 2:
            detail = "+".join(tfs_checked)
            return True, flipped, detail
        return False, flipped, ""

    def _check_divergence_close(self, analysis: dict,
                                 direction: str) -> tuple:
        """Check if multi-TF divergence signals warrant closing a position.

        Scalp mode:  checks 5m, 15m, 30m — if all show divergence against
                     position AND RSI is extreme AND at least 2 div types
                     match (RSI+MACD or RSI+OBV), close for quick recovery.
        Stable mode: checks 30m, 1h, 4h — same 2-type match rule.
        Secure mode: checks 1h, 4h — same 2-type match rule.

        Returns (should_close: bool, detail: str).
        """
        mode = getattr(self.config, "trade_mode", "scalp")

        # Which direction of divergence closes the position?
        # LONG position: bullish divergence = recovery (keep), bearish div = close
        # Actually: if we're LONG and TFs show BEARISH divergence → momentum fading
        # But the user wants: if SHORT and all TFs show BULLISH div → recovery coming
        # So: close_div = opposite of position direction
        close_div = "bullish" if direction == "SHORT" else "bearish"

        if mode == "scalp":
            tf_checks = ["5m", "15m", "30m"]
            min_tfs_matching = 2  # at least 2 of 3 TFs
        elif mode == "stable":
            tf_checks = ["30m", "1h", "4h"]
            min_tfs_matching = 2
        else:  # secure
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

            # Count how many div types match the close direction
            matching_types = sum(1 for d in [rsi_d, macd_d, obv_d]
                                 if d == close_div)

            # Need at least 2 divergence types matching (RSI+MACD or RSI+OBV)
            if matching_types >= 2:
                tfs_with_div += 1
                details.append(tf)

            # Check RSI extreme (oversold for LONG close, overbought for SHORT close)
            if direction == "LONG" and tf_rsi < 30:
                tfs_with_extreme_rsi += 1
            elif direction == "SHORT" and tf_rsi > 70:
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
                logger.warning(f"[FuturesPM] multi_tf_analysis via executor failed: {e}")
        try:
            from qor.tools import get_multi_timeframe_analysis
            return get_multi_timeframe_analysis(symbol)
        except Exception as e:
            logger.error(f"[FuturesPM] Analysis failed for {symbol}: {e}")
            return ""

    def _parse_analysis(self, text: str) -> dict:
        return parse_analysis(text)

    # ------------------------------------------------------------------
    # Main entry: evaluate one symbol
    # ------------------------------------------------------------------
    def evaluate_symbol(self, symbol: str) -> list:
        """Analyze symbol, return list of actions."""
        pair = self.client.format_pair(symbol)

        analysis_text = self._get_analysis(symbol)
        if not analysis_text or "not available" in analysis_text.lower():
            return [{"action": "HOLD", "symbol": symbol,
                     "reason": "analysis not available"}]

        analysis = self._parse_analysis(analysis_text)

        try:
            current = self.client.get_price(pair)
        except Exception:
            current = analysis["current"]
        if current <= 0:
            return [{"action": "HOLD", "symbol": symbol,
                     "reason": "no price data"}]

        analysis["current"] = current

        # Get funding rate for the symbol
        try:
            mark_data = self.client.get_mark_price(pair)
            analysis["funding_rate"] = mark_data["funding_rate"]
        except Exception:
            analysis["funding_rate"] = 0.0

        # Sentiment signals: Polymarket + Fear & Greed + Calendar (cached, fast)
        # Must run BEFORE CORTEX so features 20-21 are populated
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
            logger.debug(f"[FuturesPM] Sentiment fetch failed for {symbol}: {e}")
        # Always ensure defaults exist (even if fetch succeeded partially)
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
            logger.debug(f"[FuturesPM] Volume profile fetch failed for {symbol}: {e}")
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
            logger.debug(f"[FuturesPM] Quant signals fetch failed for {symbol}: {e}")
        analysis.setdefault("hurst", 0.5)
        analysis.setdefault("hurst_regime", "random")
        analysis.setdefault("z_score", 0.0)
        analysis.setdefault("z_label", "neutral")
        analysis.setdefault("asset_vol", 0.0)
        analysis.setdefault("price_returns", [])
        analysis.setdefault("quant_signals_available", False)

        # CORTEX analysis — observation scans deep history, reflex reacts to now
        # Runs AFTER sentiment so features 20-21 (poly, F&G) and 22-23 (VP)
        # are available in the analysis dict for the 24-dim feature vector
        cortex_result = None
        if self.cortex:
            try:
                cortex_result = self.cortex.analyze(analysis, symbol)
                analysis["cortex_signal"] = cortex_result["signal"]
                analysis["cortex_label"] = cortex_result["label"]
                analysis["cortex_confidence"] = cortex_result["confidence"]
                candles = cortex_result.get("history_candles", 0)
                logger.info(
                    f"[FuturesPM] CORTEX {symbol}: {cortex_result['label']} "
                    f"(signal={cortex_result['signal']:.3f}, "
                    f"conf={cortex_result['confidence']:.3f}, "
                    f"history={candles} candles)")
            except Exception as e:
                logger.warning(f"[FuturesPM] CORTEX analysis failed for {symbol}: {e}")

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
                        f"[FuturesPM] HMM {symbol}: REGIME CHANGE "
                        f"{hmm_signal.prev_state_name} → "
                        f"{hmm_signal.state_name} "
                        f"(conf={hmm_signal.confidence:.3f}, "
                        f"signal={hmm_signal.signal})")
            except Exception as e:
                logger.warning(f"[FuturesPM] HMM analysis failed for {symbol}: {e}")

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
    # Exchange position management (Binance-first approach)
    # ------------------------------------------------------------------
    def evaluate_symbol_with_position(self, symbol: str, ex_pos: dict) -> list:
        """Manage an existing position using live Binance data.

        Called by _tick() when Binance shows an open position for this symbol.
        Finds/creates a matching local trade, fetches analysis, then manages.

        Args:
            symbol: Asset symbol (e.g. "BTC")
            ex_pos: Exchange position dict with keys:
                direction, quantity, entry_price, mark_price, pnl, leverage, position_side
        """
        pair = self.client.format_pair(symbol)
        direction = ex_pos["direction"]

        # Find or create a local trade to track this position
        trade = self._find_local_trade(symbol, direction)
        if trade:
            # Sync local trade with live exchange data
            trade["quantity"] = ex_pos["quantity"]
            if ex_pos["entry_price"] > 0:
                trade["entry_price"] = ex_pos["entry_price"]
        else:
            # Position exists on Binance but not in local store — import it
            sl, tp, tp2 = self._get_sl_tp_from_orders(pair, direction)
            lev = ex_pos.get("leverage", self.config.leverage)
            margin = (ex_pos["entry_price"] * ex_pos["quantity"]) / max(lev, 1)
            trade_id = self.store.open_trade(
                symbol=symbol,
                side="BUY" if direction == "LONG" else "SELL",
                entry_price=ex_pos["entry_price"],
                quantity=ex_pos["quantity"],
                stop_loss=sl, take_profit=tp, take_profit2=tp2,
                reason=f"Imported from exchange ({direction})",
                direction=direction,
                leverage=lev,
                margin_type=self.config.margin_type,
                liquidation_price=0.0,
                margin_used=margin,
            )
            trade = self.store.trades[trade_id]
            logger.info(f"[FuturesPM] Imported {direction} {symbol} from exchange: "
                        f"qty={ex_pos['quantity']:.6f} entry=${ex_pos['entry_price']:,.2f} "
                        f"SL=${sl:,.2f} TP=${tp:,.2f}")

        # Get market analysis
        analysis_text = self._get_analysis(symbol)
        if not analysis_text or "not available" in analysis_text.lower():
            return [{"action": "HOLD", "symbol": symbol,
                     "reason": "analysis not available"}]

        analysis = self._parse_analysis(analysis_text)

        # Use exchange mark price (more accurate than ticker)
        current = ex_pos["mark_price"]
        if current <= 0:
            try:
                current = self.client.get_price(pair)
            except Exception:
                current = analysis["current"]
        analysis["current"] = current

        # Funding rate
        try:
            mark_data = self.client.get_mark_price(pair)
            analysis["funding_rate"] = mark_data["funding_rate"]
        except Exception:
            analysis["funding_rate"] = 0.0

        # CORTEX analysis
        cortex_result = None
        if self.cortex:
            try:
                cortex_result = self.cortex.analyze(analysis, symbol)
                analysis["cortex_signal"] = cortex_result["signal"]
                analysis["cortex_label"] = cortex_result["label"]
                analysis["cortex_confidence"] = cortex_result["confidence"]
                candles = cortex_result.get("history_candles", 0)
                logger.info(
                    f"[FuturesPM] CORTEX {symbol} ({direction}): "
                    f"{cortex_result['label']} "
                    f"(signal={cortex_result['signal']:.3f}, "
                    f"conf={cortex_result['confidence']:.3f}, "
                    f"history={candles} candles)")
            except Exception as e:
                logger.warning(f"[FuturesPM] CORTEX analysis failed for {symbol}: {e}")

        decisions = self._manage_position(trade, analysis, symbol)

        if cortex_result:
            for d in decisions:
                d["cortex_signal"] = cortex_result["signal"]
                d["cortex_label"] = cortex_result["label"]

        return decisions

    def _find_local_trade(self, symbol: str, direction: str) -> dict:
        """Find matching open trade in local store by symbol + direction."""
        for t in self.store.trades.values():
            if (t["symbol"] == symbol and t["status"] == "open"
                    and t.get("direction", "LONG") == direction):
                return t
        return None

    def _get_sl_tp_from_orders(self, pair: str, direction: str) -> tuple:
        """Get SL/TP prices from open orders on Binance for a specific direction.

        In hedge mode, filters by positionSide to match the correct direction.
        Returns (stop_loss, take_profit, take_profit2).
        """
        sl, tp, tp2 = 0.0, 0.0, 0.0
        try:
            orders = self.client.get_open_orders(pair)
            for o in orders:
                # In hedge mode, match by positionSide
                pos_side = o.get("positionSide", "BOTH")
                if pos_side != "BOTH" and pos_side != direction:
                    continue
                otype = o.get("type", "")
                stop_price = float(o.get("stopPrice", 0))
                if stop_price <= 0:
                    continue
                if otype == "STOP_MARKET":
                    sl = stop_price
                elif otype == "TAKE_PROFIT_MARKET":
                    tp2 = stop_price
                    if tp <= 0:
                        tp = stop_price
        except Exception as e:
            logger.warning(f"[FuturesPM] Cannot read open orders for {pair}: {e}")
        return sl, tp, tp2

    # ------------------------------------------------------------------
    # Position management (has position)
    # ------------------------------------------------------------------
    def _manage_position(self, trade: dict, analysis: dict, symbol: str) -> list:
        """Full position management for LONG or SHORT positions."""
        actions = []
        current = analysis["current"]
        entry = trade["entry_price"]
        sl = trade["stop_loss"]
        tp1 = trade["take_profit"]
        tp2 = trade.get("take_profit2", 0) or tp1 * 1.5
        tid = trade["trade_id"]
        tp1_hit = trade.get("tp1_hit", False)
        direction = trade.get("direction", "LONG")

        # --- Check funding cost ---
        funding_paid = trade.get("funding_paid", 0.0)
        margin_used = trade.get("margin_used", 0.0)
        if margin_used > 0 and abs(funding_paid) > margin_used * (self.config.max_funding_cost_pct / 100):
            close_side = "SELL" if direction == "LONG" else "BUY"
            actions.append({
                "action": "CLOSE", "symbol": symbol, "trade_id": tid,
                "close_side": close_side,
                "exit_status": "closed_funding",
                "reason": f"funding cost ${funding_paid:,.4f} exceeded {self.config.max_funding_cost_pct}% of margin",
            })
            return actions

        if direction == "LONG":
            return self._manage_long(trade, analysis, symbol, actions,
                                     current, entry, sl, tp1, tp2, tid, tp1_hit)
        else:
            return self._manage_short(trade, analysis, symbol, actions,
                                      current, entry, sl, tp1, tp2, tid, tp1_hit)

    def _manage_long(self, trade, analysis, symbol, actions,
                     current, entry, sl, tp1, tp2, tid, tp1_hit):
        """Manage a LONG position (same logic as spot)."""
        # SL hit
        if current <= sl:
            actions.append({
                "action": "CLOSE", "symbol": symbol, "trade_id": tid,
                "close_side": "SELL", "exit_status": "closed_sl",
                "reason": f"SL hit at ${sl:,.2f} (entry ${entry:,.2f})",
            })
            return actions

        # TP2 hit
        if tp2 > 0 and current >= tp2:
            actions.append({
                "action": "CLOSE", "symbol": symbol, "trade_id": tid,
                "close_side": "SELL", "exit_status": "closed_tp",
                "reason": f"TP2 hit at ${tp2:,.2f}",
            })
            return actions

        # TP1 hit (first time) -> partial
        if not tp1_hit and tp1 > 0 and current >= tp1:
            if self.config.partial_tp_enabled:
                sell_pct = self.config.partial_tp1_pct / 100.0
                sell_qty = trade["quantity"] * sell_pct
                actions.append({
                    "action": "PARTIAL_TP", "symbol": symbol, "trade_id": tid,
                    "close_side": "SELL", "sell_qty": sell_qty, "sell_price": current,
                    "reason": f"TP1 hit ${tp1:,.2f} -- selling {self.config.partial_tp1_pct:.0f}%",
                })
                if self.config.move_sl_to_be and entry > sl:
                    actions.append({
                        "action": "ADJUST_SL", "symbol": symbol, "trade_id": tid,
                        "new_sl": entry,
                        "reason": f"SL to breakeven ${entry:,.2f} after TP1",
                    })
                return actions
            else:
                actions.append({
                    "action": "CLOSE", "symbol": symbol, "trade_id": tid,
                    "close_side": "SELL", "exit_status": "closed_tp",
                    "reason": f"TP1 hit at ${tp1:,.2f}",
                })
                return actions

        # Trend reversal — mode-aware threshold
        cortex_label = analysis.get("cortex_label", "")
        cortex_signal = analysis.get("cortex_signal", 0)
        cortex_reversal = cortex_label == "STRONG_SELL" and analysis["bearish_tfs"] >= analysis["total_tfs"] * 0.5
        total = analysis["total_tfs"]
        reversal_thresh = self._get_reversal_threshold(total)
        majority_bearish = total > 0 and analysis["bearish_tfs"] >= reversal_thresh

        # Scalp mode: also check if 2+ of 5m/15m/30m flipped bearish
        scalp_rev, scalp_cnt, scalp_detail = self._check_scalp_short_tf_reversal(
            analysis, against="LONG")

        if majority_bearish or cortex_reversal or scalp_rev:
            reason_parts = []
            if majority_bearish:
                reason_parts.append(f"{analysis['bearish_tfs']}/{total} TFs bearish (thresh={reversal_thresh})")
            if scalp_rev:
                reason_parts.append(f"scalp {scalp_cnt}/3 short TFs bearish ({scalp_detail})")
            if cortex_reversal:
                reason_parts.append(f"CORTEX STRONG_SELL ({cortex_signal:.3f})")
            actions.append({
                "action": "CLOSE", "symbol": symbol, "trade_id": tid,
                "close_side": "SELL", "exit_status": "closed_reversal",
                "reason": f"trend reversal -- {' + '.join(reason_parts)}",
            })
            return actions

        # Divergence-based close — multi-TF divergence against LONG position
        # Scalp: 5m+15m+30m, Stable: 30m+1h+4h, Secure: 1h+4h
        # Requires 2+ div types (RSI+MACD or RSI+OBV) + RSI extreme
        div_close, div_detail = self._check_divergence_close(analysis, "LONG")
        pnl_pct = ((current / entry) - 1) * 100 if entry > 0 else 0
        if div_close:
            actions.append({
                "action": "CLOSE", "symbol": symbol, "trade_id": tid,
                "close_side": "SELL", "exit_status": "closed_divergence",
                "reason": f"divergence close: {div_detail}, P&L {pnl_pct:+.1f}%",
            })
            return actions

        # Adverse momentum early exit — short TF RSI extreme against LONG
        # Checks 5m + 15m + 30m RSI (all scalp TFs)
        rsi_5m = analysis.get("rsi_5m", 50)
        rsi_15m = analysis.get("rsi_15m", 50)
        rsi_30m = analysis.get("rsi_30m", 50)
        # Close if losing >1% AND any 2 of 3 short TFs show extreme bearish RSI
        bearish_rsi_count = sum(1 for r in [rsi_5m, rsi_15m, rsi_30m]
                                if r < 30)
        if pnl_pct < -1.0 and bearish_rsi_count >= 2:
            actions.append({
                "action": "CLOSE", "symbol": symbol, "trade_id": tid,
                "close_side": "SELL", "exit_status": "closed_momentum",
                "reason": (f"adverse momentum: RSI 5m={rsi_5m:.0f} "
                           f"15m={rsi_15m:.0f} 30m={rsi_30m:.0f}, "
                           f"P&L {pnl_pct:+.1f}%"),
            })
            return actions

        # Tighten SL aggressively when momentum turns against
        # Half the TFs bearish + losing → move SL to 1 ATR below current
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

        # DCA
        dca_action = self._check_dca_long(trade, analysis, symbol, current)
        if dca_action:
            actions.append(dca_action)

        # Trailing stop (move SL UP)
        sl_action = self._check_trailing_sl_long(trade, analysis, current)
        if sl_action:
            actions.append(sl_action)

        if not actions:
            actions.append({
                "action": "HOLD", "symbol": symbol,
                "reason": f"LONG holding ({pnl_pct:+.1f}% from entry)",
            })
        return actions

    def _manage_short(self, trade, analysis, symbol, actions,
                      current, entry, sl, tp1, tp2, tid, tp1_hit):
        """Manage a SHORT position (inverted SL/TP logic)."""
        # SL hit (price went UP against short)
        if current >= sl:
            actions.append({
                "action": "CLOSE", "symbol": symbol, "trade_id": tid,
                "close_side": "BUY", "exit_status": "closed_sl",
                "reason": f"SL hit at ${sl:,.2f} (entry ${entry:,.2f})",
            })
            return actions

        # TP2 hit (price went DOWN in favor of short)
        if tp2 > 0 and current <= tp2:
            actions.append({
                "action": "CLOSE", "symbol": symbol, "trade_id": tid,
                "close_side": "BUY", "exit_status": "closed_tp",
                "reason": f"TP2 hit at ${tp2:,.2f}",
            })
            return actions

        # TP1 hit (price dropped to TP1)
        if not tp1_hit and tp1 > 0 and current <= tp1:
            if self.config.partial_tp_enabled:
                sell_pct = self.config.partial_tp1_pct / 100.0
                sell_qty = trade["quantity"] * sell_pct
                actions.append({
                    "action": "PARTIAL_TP", "symbol": symbol, "trade_id": tid,
                    "close_side": "BUY", "sell_qty": sell_qty, "sell_price": current,
                    "reason": f"TP1 hit ${tp1:,.2f} -- closing {self.config.partial_tp1_pct:.0f}%",
                })
                if self.config.move_sl_to_be and sl > entry:
                    actions.append({
                        "action": "ADJUST_SL", "symbol": symbol, "trade_id": tid,
                        "new_sl": entry,
                        "reason": f"SL to breakeven ${entry:,.2f} after TP1",
                    })
                return actions
            else:
                actions.append({
                    "action": "CLOSE", "symbol": symbol, "trade_id": tid,
                    "close_side": "BUY", "exit_status": "closed_tp",
                    "reason": f"TP1 hit at ${tp1:,.2f}",
                })
                return actions

        # Trend reversal — mode-aware threshold
        cortex_label = analysis.get("cortex_label", "")
        cortex_signal = analysis.get("cortex_signal", 0)
        cortex_reversal = cortex_label == "STRONG_BUY" and analysis["bullish_tfs"] >= analysis["total_tfs"] * 0.5
        total = analysis["total_tfs"]
        reversal_thresh = self._get_reversal_threshold(total)
        majority_bullish = total > 0 and analysis["bullish_tfs"] >= reversal_thresh

        # Scalp mode: also check if 2+ of 5m/15m/30m flipped bullish
        scalp_rev, scalp_cnt, scalp_detail = self._check_scalp_short_tf_reversal(
            analysis, against="SHORT")

        if majority_bullish or cortex_reversal or scalp_rev:
            reason_parts = []
            if majority_bullish:
                reason_parts.append(f"{analysis['bullish_tfs']}/{total} TFs bullish (thresh={reversal_thresh})")
            if scalp_rev:
                reason_parts.append(f"scalp {scalp_cnt}/3 short TFs bullish ({scalp_detail})")
            if cortex_reversal:
                reason_parts.append(f"CORTEX STRONG_BUY ({cortex_signal:.3f})")
            actions.append({
                "action": "CLOSE", "symbol": symbol, "trade_id": tid,
                "close_side": "BUY", "exit_status": "closed_reversal",
                "reason": f"trend reversal -- {' + '.join(reason_parts)}",
            })
            return actions

        # Divergence-based close — multi-TF divergence against SHORT position
        # Scalp: 5m+15m+30m, Stable: 30m+1h+4h, Secure: 1h+4h
        # Requires 2+ div types (RSI+MACD or RSI+OBV) + RSI extreme
        div_close, div_detail = self._check_divergence_close(analysis, "SHORT")
        pnl_pct = ((entry / current) - 1) * 100 if current > 0 else 0
        if div_close:
            actions.append({
                "action": "CLOSE", "symbol": symbol, "trade_id": tid,
                "close_side": "BUY", "exit_status": "closed_divergence",
                "reason": f"divergence close: {div_detail}, P&L {pnl_pct:+.1f}%",
            })
            return actions

        # Adverse momentum early exit — short TF RSI extreme against SHORT
        # Checks 5m + 15m + 30m RSI (all scalp TFs)
        rsi_5m = analysis.get("rsi_5m", 50)
        rsi_15m = analysis.get("rsi_15m", 50)
        rsi_30m = analysis.get("rsi_30m", 50)
        # Close if losing >1% AND any 2 of 3 short TFs show extreme bullish RSI
        bullish_rsi_count = sum(1 for r in [rsi_5m, rsi_15m, rsi_30m]
                                if r > 70)
        if pnl_pct < -1.0 and bullish_rsi_count >= 2:
            actions.append({
                "action": "CLOSE", "symbol": symbol, "trade_id": tid,
                "close_side": "BUY", "exit_status": "closed_momentum",
                "reason": (f"adverse momentum: RSI 5m={rsi_5m:.0f} "
                           f"15m={rsi_15m:.0f} 30m={rsi_30m:.0f}, "
                           f"P&L {pnl_pct:+.1f}%"),
            })
            return actions

        # Tighten SL aggressively when momentum turns against
        # Half the TFs bullish + losing → move SL to 1 ATR above current
        half_bullish = total > 0 and analysis["bullish_tfs"] >= total // 2
        if half_bullish and pnl_pct < 0:
            atr = self._get_trade_atr(analysis)
            if atr > 0:
                tight_sl = current + atr  # 1x ATR instead of 2x
                if tight_sl < sl:
                    actions.append({
                        "action": "ADJUST_SL", "symbol": symbol, "trade_id": tid,
                        "new_sl": tight_sl,
                        "reason": (f"tighten SL: {analysis['bullish_tfs']}/{total} "
                                   f"TFs against, ${sl:,.2f} -> ${tight_sl:,.2f}"),
                    })
                    return actions

        # DCA (add to short when price rises)
        dca_action = self._check_dca_short(trade, analysis, symbol, current)
        if dca_action:
            actions.append(dca_action)

        # Trailing stop (move SL DOWN for short)
        sl_action = self._check_trailing_sl_short(trade, analysis, current)
        if sl_action:
            actions.append(sl_action)

        if not actions:
            actions.append({
                "action": "HOLD", "symbol": symbol,
                "reason": f"SHORT holding ({pnl_pct:+.1f}% from entry)",
            })
        return actions

    # ------------------------------------------------------------------
    # DCA helpers
    # ------------------------------------------------------------------
    def _check_dca_long(self, trade, analysis, symbol, current):
        """Check DCA for LONG position (price dropped)."""
        if not self.config.dca_enabled:
            return None
        dca_count = trade.get("dca_count", 0)
        if dca_count >= self.config.dca_max_adds:
            return None
        entry = trade["entry_price"]
        drop_pct = ((entry - current) / entry) * 100 if entry > 0 else 0
        required_drop = self.config.dca_drop_pct * (dca_count + 1)
        if drop_pct < required_drop:
            return None
        atr = self._get_trade_atr(analysis)
        near_support = False
        if analysis["supports"] and atr > 0:
            for sup in analysis["supports"]:
                if abs(current - sup) <= atr:
                    near_support = True
                    break
        rsi_oversold = analysis["rsi"] < 35
        if not near_support and not rsi_oversold:
            return None
        if analysis["bearish_tfs"] >= analysis["total_tfs"] and analysis["total_tfs"] > 0:
            return None
        # CORTEX veto: don't DCA into a STRONG_SELL
        if analysis.get("cortex_label") == "STRONG_SELL":
            return None
        if analysis["bullish_tfs"] == 0:
            return None
        rsi_val = analysis["rsi"]
        rsi_str = f" RSI={rsi_val:.0f}" if rsi_oversold else ""
        sup_str = " near support" if near_support else ""
        return {
            "action": "DCA", "symbol": symbol, "trade_id": trade["trade_id"],
            "dca_side": "BUY", "dca_number": dca_count + 1,
            "reason": f"DCA #{dca_count + 1}: drop {drop_pct:.1f}%{sup_str}{rsi_str}",
        }

    def _check_dca_short(self, trade, analysis, symbol, current):
        """Check DCA for SHORT position (price rose against short)."""
        if not self.config.dca_enabled:
            return None
        dca_count = trade.get("dca_count", 0)
        if dca_count >= self.config.dca_max_adds:
            return None
        entry = trade["entry_price"]
        rise_pct = ((current - entry) / entry) * 100 if entry > 0 else 0
        required_rise = self.config.dca_drop_pct * (dca_count + 1)
        if rise_pct < required_rise:
            return None
        atr = self._get_trade_atr(analysis)
        near_resistance = False
        if analysis["resistances"] and atr > 0:
            for res in analysis["resistances"]:
                if abs(current - res) <= atr:
                    near_resistance = True
                    break
        rsi_overbought = analysis["rsi"] > 65
        if not near_resistance and not rsi_overbought:
            return None
        if analysis["bullish_tfs"] >= analysis["total_tfs"] and analysis["total_tfs"] > 0:
            return None
        # CORTEX veto: don't DCA into a STRONG_BUY (against short direction)
        if analysis.get("cortex_label") == "STRONG_BUY":
            return None
        if analysis["bearish_tfs"] == 0:
            return None
        rsi_val = analysis["rsi"]
        rsi_str = f" RSI={rsi_val:.0f}" if rsi_overbought else ""
        res_str = " near resistance" if near_resistance else ""
        return {
            "action": "DCA", "symbol": symbol, "trade_id": trade["trade_id"],
            "dca_side": "SELL", "dca_number": dca_count + 1,
            "reason": f"DCA #{dca_count + 1}: rise {rise_pct:.1f}%{res_str}{rsi_str}",
        }

    # ------------------------------------------------------------------
    # Trailing stop helpers
    # ------------------------------------------------------------------
    def _check_trailing_sl_long(self, trade, analysis, current):
        """Trailing SL for LONG: move SL UP as price rises."""
        if not self.config.trailing_stop:
            return None
        entry = trade["entry_price"]
        sl = trade["stop_loss"]
        atr = self._get_trade_atr(analysis)
        if atr <= 0:
            return None
        new_sl = current - self.config.stop_loss_atr_mult * atr
        for s in sorted(analysis["supports"], reverse=True):
            if s < current:
                new_sl = max(new_sl, s * 0.995)
                break
        if trade.get("tp1_hit") and self.config.move_sl_to_be:
            new_sl = max(new_sl, entry)
        if new_sl > sl:
            return {
                "action": "ADJUST_SL", "symbol": trade["symbol"],
                "trade_id": trade["trade_id"], "new_sl": new_sl,
                "reason": f"trailing SL: ${sl:,.2f} -> ${new_sl:,.2f}",
            }
        return None

    def _check_trailing_sl_short(self, trade, analysis, current):
        """Trailing SL for SHORT: move SL DOWN as price drops."""
        if not self.config.trailing_stop:
            return None
        entry = trade["entry_price"]
        sl = trade["stop_loss"]
        atr = self._get_trade_atr(analysis)
        if atr <= 0:
            return None
        new_sl = current + self.config.stop_loss_atr_mult * atr
        for r in sorted(analysis["resistances"]):
            if r > current:
                new_sl = min(new_sl, r * 1.005)
                break
        if trade.get("tp1_hit") and self.config.move_sl_to_be:
            new_sl = min(new_sl, entry)
        if new_sl < sl:
            return {
                "action": "ADJUST_SL", "symbol": trade["symbol"],
                "trade_id": trade["trade_id"], "new_sl": new_sl,
                "reason": f"trailing SL: ${sl:,.2f} -> ${new_sl:,.2f}",
            }
        return None

    # ------------------------------------------------------------------
    # Entry evaluation (no position)
    # ------------------------------------------------------------------
    def _evaluate_entry(self, analysis: dict, symbol: str,
                        current: float) -> list:
        """Evaluate entry for LONG or SHORT based on bias."""
        hold = {"action": "HOLD", "symbol": symbol}

        block_reason = self._should_enter(analysis, symbol)
        if block_reason:
            hold["reason"] = block_reason
            return [hold]

        # Use mode-appropriate ATR for SL/TP (scalp=short TF, secure=daily)
        atr = self._get_trade_atr(analysis)

        # ATR floor: minimum 0.3% of price to prevent unreasonably tight SL/TP
        if current > 0 and atr > 0:
            min_atr = current * 0.003
            if atr < min_atr:
                logger.debug(f"[FuturesPM] ATR floor: {atr:.4f} -> {min_atr:.4f} "
                             f"(0.3% of ${current:,.2f})")
                atr = min_atr

        # Confluence score strength check — weak signals should not enter
        # Score is sum of all TF scores (-100..+100 each)
        cscore = analysis.get("confluence_score", 0)
        total_tfs = analysis.get("total_tfs", 7)
        # Minimum absolute score threshold: at least ~25 per TF on average
        min_score = total_tfs * 15
        if abs(cscore) < min_score:
            hold["reason"] = (f"weak confluence score: {cscore:+.0f} "
                              f"(need ±{min_score:.0f})")
            return [hold]

        # Determine direction from TF majority (not just analysis bias)
        # Futures can go LONG or SHORT — pick the stronger direction
        bullish_tfs = analysis.get("bullish_tfs", 0)
        bearish_tfs = analysis.get("bearish_tfs", 0)
        bias = analysis["bias"]

        if bearish_tfs > bullish_tfs:
            direction = "SHORT"
        elif bullish_tfs > bearish_tfs:
            direction = "LONG"
        elif bias in ("SHORT", "BEARISH"):
            direction = "SHORT"
        elif bias in ("LONG", "BULLISH"):
            direction = "LONG"
        else:
            hold["reason"] = f"neutral: {bullish_tfs} bullish = {bearish_tfs} bearish"
            return [hold]

        entry_price = current  # Always use live price, not text-parsed daily-based entry
        mode = getattr(self.config, "trade_mode", "scalp")

        # Fibonacci + Pivot levels as SL/TP confluence (from analysis text)
        fib_sup = analysis.get("fib_sup", 0)
        fib_res = analysis.get("fib_res", 0)
        pivot_s1 = analysis.get("pivot_s1", 0)
        pivot_s2 = analysis.get("pivot_s2", 0)
        pivot_r1 = analysis.get("pivot_r1", 0)
        pivot_r2 = analysis.get("pivot_r2", 0)

        # ALWAYS recalculate SL/TP from mode-appropriate ATR
        # Refine with Fibonacci levels when they fall within reasonable range
        sl_mult = self.config.stop_loss_atr_mult
        tp_mult = self.config.take_profit_atr_mult
        if direction == "LONG":
            sl_price = current - sl_mult * atr if atr > 0 else 0
            tp1_price = current + tp_mult * atr if atr > 0 else 0
            tp2_price = current + tp_mult * atr * 2 if atr > 0 else 0
            # Refine SL with Fib support / Pivot S1 — tighten if closer
            # but not too close (at least 0.5x ATR away)
            best_sup = sl_price
            for level in [fib_sup, pivot_s1, pivot_s2]:
                if level > 0 and level < current and atr > 0:
                    candidate = level - 0.2 * atr  # just below the level
                    if candidate > best_sup and (current - candidate) >= 0.5 * atr:
                        best_sup = candidate
            sl_price = best_sup
            # Refine TP with Fib resistance / Pivot R1 — use if better R:R
            for level in [fib_res, pivot_r1, pivot_r2]:
                if level > current and atr > 0 and sl_price > 0:
                    level_rr = (level - current) / max(current - sl_price, 0.01)
                    curr_rr = (tp1_price - current) / max(current - sl_price, 0.01)
                    if level_rr > curr_rr:
                        tp1_price = level
            # Volume Profile refinement (LONG) — VAL as support, VAH as target
            vp_val = analysis.get("vp_val", 0)
            vp_vah = analysis.get("vp_vah", 0)
            if analysis.get("vp_available") and atr > 0:
                if vp_val > 0 and vp_val < current:
                    vp_sl = vp_val - 0.2 * atr
                    if vp_sl > sl_price and (current - vp_sl) >= 0.5 * atr:
                        sl_price = vp_sl
                if vp_vah > current and sl_price > 0:
                    vp_rr = (vp_vah - current) / max(current - sl_price, 0.01)
                    curr_rr = (tp1_price - current) / max(current - sl_price, 0.01)
                    if vp_rr > curr_rr:
                        tp1_price = vp_vah
            # Hurst TP adjustment (LONG) — trending: widen, mean-reverting: tighten
            hurst = analysis.get("hurst", 0.5)
            if analysis.get("quant_signals_available") and atr > 0:
                if hurst > 0.6:
                    tp1_price = current + (tp1_price - current) * 1.2
                    tp2_price = current + (tp2_price - current) * 1.2
                elif hurst < 0.4:
                    tp1_price = current + (tp1_price - current) * 0.8
                    tp2_price = current + (tp2_price - current) * 0.8
            risk = current - sl_price
            reward = tp1_price - current
        else:  # SHORT
            sl_price = current + sl_mult * atr if atr > 0 else 0
            tp1_price = current - tp_mult * atr if atr > 0 else 0
            tp2_price = current - tp_mult * atr * 2 if atr > 0 else 0
            # Refine SL with Fib resistance / Pivot R1 — tighten if closer
            best_res = sl_price
            for level in [fib_res, pivot_r1, pivot_r2]:
                if level > 0 and level > current and atr > 0:
                    candidate = level + 0.2 * atr  # just above the level
                    if candidate < best_res and (candidate - current) >= 0.5 * atr:
                        best_res = candidate
            sl_price = best_res
            # Refine TP with Fib support / Pivot S1 — use if better R:R
            for level in [fib_sup, pivot_s1, pivot_s2]:
                if level > 0 and level < current and atr > 0 and sl_price > 0:
                    level_rr = (current - level) / max(sl_price - current, 0.01)
                    curr_rr = (current - tp1_price) / max(sl_price - current, 0.01)
                    if level_rr > curr_rr:
                        tp1_price = level
            # Volume Profile refinement (SHORT) — VAH as resistance, VAL as target
            vp_val = analysis.get("vp_val", 0)
            vp_vah = analysis.get("vp_vah", 0)
            if analysis.get("vp_available") and atr > 0:
                # SL: If VAH is between entry and SL → tighten SL to just above VAH
                if vp_vah > 0 and vp_vah > current:
                    vp_sl = vp_vah + 0.2 * atr
                    if vp_sl < sl_price and (vp_sl - current) >= 0.5 * atr:
                        sl_price = vp_sl
                # TP: If VAL is below entry and offers better R:R → use as TP
                if vp_val > 0 and vp_val < current and sl_price > 0:
                    vp_rr = (current - vp_val) / max(sl_price - current, 0.01)
                    curr_rr = (current - tp1_price) / max(sl_price - current, 0.01)
                    if vp_rr > curr_rr:
                        tp1_price = vp_val
            # Hurst TP adjustment (SHORT) — trending: widen, mean-reverting: tighten
            hurst = analysis.get("hurst", 0.5)
            if analysis.get("quant_signals_available") and atr > 0:
                if hurst > 0.6:
                    tp1_price = current - (current - tp1_price) * 1.2
                    tp2_price = current - (current - tp2_price) * 1.2
                elif hurst < 0.4:
                    tp1_price = current - (current - tp1_price) * 0.8
                    tp2_price = current - (current - tp2_price) * 0.8
            risk = sl_price - current
            reward = current - tp1_price

        # Funding rate — keep as hard block (real financial cost)
        funding_rate = analysis.get("funding_rate", 0.0)
        if direction == "LONG" and funding_rate > self.config.funding_rate_threshold:
            hold["reason"] = f"funding rate {funding_rate:.4f} too high for LONG"
            return [hold]
        if direction == "SHORT" and funding_rate < -self.config.funding_rate_threshold:
            hold["reason"] = f"funding rate {funding_rate:.4f} too negative for SHORT"
            return [hold]

        if risk <= 0 or reward <= 0:
            hold["reason"] = "invalid SL/TP levels"
            return [hold]

        rr = reward / risk
        min_rr = self.config.min_risk_reward

        sym_stats = self.store.get_recent_symbol_stats(symbol, n=20)
        if sym_stats["trades"] >= 5 and sym_stats["win_rate"] < 50:
            min_rr = self.config.min_risk_reward * 1.5

        if rr < min_rr:
            hold["reason"] = f"R:R {rr:.1f} < min {min_rr:.1f}"
            return [hold]

        # ============================================================
        # Signal multiplier system — all signals INFLUENCE position
        # size instead of blocking.  Direction-aware for futures.
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

        # Sortino — downside risk specifically
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
                size_mult *= 0.7
                mult_notes.append(f"α({alpha:.3f})×0.7")
            elif alpha > 0.05:
                size_mult *= 1.15
                mult_notes.append(f"α({alpha:.3f})×1.15")

        # Information Ratio — consistency of alpha
        ir = analysis.get("quant_ir")
        if ir is not None:
            if ir < -0.5:
                size_mult *= 0.8
                mult_notes.append(f"IR({ir:.2f})×0.8")
            elif ir > 0.5:
                size_mult *= 1.1
                mult_notes.append(f"IR({ir:.2f})×1.1")

        # Z-Score — direction-aware: overbought bad for LONG, oversold bad for SHORT
        z_score = analysis.get("z_score", 0.0)
        if analysis.get("quant_signals_available"):
            if direction == "LONG":
                if z_score > 2.0:
                    size_mult *= 0.5
                    mult_notes.append(f"Z({z_score:.1f})×0.5")
                elif z_score > 1.5:
                    size_mult *= 0.7
                    mult_notes.append(f"Z({z_score:.1f})×0.7")
            else:  # SHORT
                if z_score < -2.0:
                    size_mult *= 0.5
                    mult_notes.append(f"Z({z_score:.1f})×0.5")
                elif z_score < -1.5:
                    size_mult *= 0.7
                    mult_notes.append(f"Z({z_score:.1f})×0.7")

        # Hurst — mean-reverting reduces size, trending boosts
        hurst = analysis.get("hurst", 0.5)
        if analysis.get("quant_signals_available"):
            if hurst < 0.4:
                size_mult *= 0.8
                mult_notes.append(f"H({hurst:.2f})×0.8")
            elif hurst > 0.65:
                size_mult *= 1.1
                mult_notes.append(f"H({hurst:.2f})×1.1")

        # Polymarket — direction-aware contra signal reduces size
        poly_up = analysis.get("poly_up_prob", 0.5)
        poly_thresh = getattr(self.config, "poly_block_threshold", 0.35)
        if analysis.get("poly_available"):
            if direction == "LONG" and poly_up < poly_thresh:
                size_mult *= 0.7
                mult_notes.append(f"POLY({poly_up*100:.0f}%)×0.7")
            elif direction == "SHORT" and poly_up > (1 - poly_thresh):
                size_mult *= 0.7
                mult_notes.append(f"POLY({poly_up*100:.0f}%)×0.7")

        # Fear & Greed — direction-aware extremes reduce size
        fg = analysis.get("fear_greed_value", 50)
        fg_greed = getattr(self.config, "fg_extreme_greed", 85)
        fg_fear = getattr(self.config, "fg_extreme_fear", 15)
        if direction == "LONG" and fg > fg_greed:
            size_mult *= 0.7
            mult_notes.append(f"F&G({fg})×0.7")
        elif direction == "SHORT" and fg < fg_fear:
            size_mult *= 0.7
            mult_notes.append(f"F&G({fg})×0.7")

        # Divergence — contra divergence reduces size
        div_score = analysis.get("div_score", 0)
        if direction == "LONG" and div_score <= -15:
            size_mult *= 0.7
            mult_notes.append(f"div({div_score})×0.7")
        elif direction == "SHORT" and div_score >= 15:
            size_mult *= 0.7
            mult_notes.append(f"div({div_score})×0.7")

        # CORTEX — reduces on contra, boosts on strong agreement
        cortex_label = analysis.get("cortex_label", "")
        cortex_signal = analysis.get("cortex_signal", 0)
        if direction == "LONG":
            if cortex_label == "STRONG_SELL":
                size_mult *= 0.3
                mult_notes.append(f"CORTEX(SS)×0.3")
            elif cortex_label == "SELL":
                size_mult *= 0.5
                mult_notes.append(f"CORTEX(S)×0.5")
            elif cortex_label == "STRONG_BUY":
                size_mult *= 1.2
                mult_notes.append(f"CORTEX(SB)×1.2")
        else:  # SHORT
            if cortex_label == "STRONG_BUY":
                size_mult *= 0.3
                mult_notes.append(f"CORTEX(SB)×0.3")
            elif cortex_label == "BUY":
                size_mult *= 0.5
                mult_notes.append(f"CORTEX(B)×0.5")
            elif cortex_label == "STRONG_SELL":
                size_mult *= 1.2
                mult_notes.append(f"CORTEX(SS)×1.2")

        # HMM regime — contra reduces, aligned boosts
        hmm_state = analysis.get("hmm_state", "")
        hmm_conf = analysis.get("hmm_confidence", 0)
        if hmm_conf >= 0.6:
            if direction == "LONG" and hmm_state in ("BEAR", "STRONG_BEAR"):
                size_mult *= 0.5
                mult_notes.append(f"HMM({hmm_state})×0.5")
            elif direction == "SHORT" and hmm_state in ("BULL", "STRONG_BULL"):
                size_mult *= 0.5
                mult_notes.append(f"HMM({hmm_state})×0.5")
            elif direction == "LONG" and hmm_state in ("BULL", "STRONG_BULL") and hmm_conf >= 0.7:
                size_mult *= 1.15
                mult_notes.append(f"HMM({hmm_state})×1.15")
            elif direction == "SHORT" and hmm_state in ("BEAR", "STRONG_BEAR") and hmm_conf >= 0.7:
                size_mult *= 1.15
                mult_notes.append(f"HMM({hmm_state})×1.15")
        if hmm_state == "CHOPPY" and hmm_conf >= 0.7:
            size_mult *= 0.7
            mult_notes.append(f"HMM(CHOPPY)×0.7")

        # ============================================================
        # Compute Kelly position size WITH all signal multipliers
        # This IS the decision — size is known at decision time
        # ============================================================
        position_pct = self._kelly_pct(symbol, direction, analysis=analysis)
        if position_pct <= 0:
            hold["reason"] = f"Kelly: no edge ({direction})"
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
        tf_count = analysis['bearish_tfs'] if direction == "SHORT" else analysis['bullish_tfs']
        tf_label = "bearish" if direction == "SHORT" else "bullish"
        tf_str = f"{tf_count}/{analysis['total_tfs']}"
        order_side = "BUY" if direction == "LONG" else "SELL"
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
            f"{direction} {tf_str} TFs {tf_label}, "
            f"RSI {analysis['rsi']:.0f}, R:R {rr:.1f}:1, "
            f"ATR ${atr:,.2f} ({mode}), "
            f"funding {funding_rate:.4f}{cortex_str}{hmm_str}"
            f"{poly_str}{fg_str}{vp_str}{quant_str}{sharpe_str}"
            f"{sortino_str}{alpha_str}{ir_str}{qs_str}{size_str}"
        )

        return [{
            "action": "OPEN", "symbol": symbol,
            "direction": direction,
            "order_side": order_side,
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
        # Cooldown
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

        # Multi-TF confluence — futures can go EITHER direction
        if analysis["total_tfs"] > 0:
            min_required = max(3, analysis["total_tfs"] // 2 + 1)
            bullish = analysis["bullish_tfs"]
            bearish = analysis["bearish_tfs"]
            # Allow entry if EITHER direction has enough confluence
            if bullish < min_required and bearish < min_required:
                return (f"weak confluence: {bullish} bullish, {bearish} bearish "
                        f"of {analysis['total_tfs']} (need {min_required})")

        # Quant edge validation — Expectancy + Risk of Ruin + Max Drawdown
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
    # Kelly Criterion position sizing (direction-aware)
    # ------------------------------------------------------------------

    def _kelly_pct(self, symbol: str, direction: str = None,
                   analysis: dict = None) -> float:
        """Kelly Criterion position size as % of available capital.

        Enhanced with:
        - Edge Decay: reduce sizing if expectancy declining over time
        - Volatility Sizing cap: cap position using target vol / asset vol

        For futures, filters by direction (LONG/SHORT) so each side
        has its own edge estimate.  Returns 0.0 if no edge.
        Returns max_position_pct if Kelly disabled or not enough history.
        """
        if not self.config.kelly_enabled:
            return self.config.max_position_pct

        closed = [t for t in self.store.trades.values()
                  if t["symbol"] == symbol and t["status"] != "open"]
        if direction:
            closed = [t for t in closed if t.get("direction") == direction]
        recent = closed[-30:]

        if len(recent) < self.config.kelly_min_trades:
            return self.config.max_position_pct

        wins = [t for t in recent if t["pnl_pct"] > 0]
        losses = [t for t in recent if t["pnl_pct"] <= 0]
        if not wins or not losses:
            return self.config.max_position_pct

        p = len(wins) / len(recent)
        avg_win = sum(t["pnl_pct"] for t in wins) / len(wins)
        avg_loss = abs(sum(t["pnl_pct"] for t in losses) / len(losses))
        if avg_loss <= 0:
            return self.config.max_position_pct

        b = avg_win / avg_loss
        q = 1.0 - p

        kelly_full = (b * p - q) / b

        if kelly_full <= 0:
            dir_tag = f" {direction}" if direction else ""
            logger.info(
                f"[Kelly] {symbol}{dir_tag}: NO EDGE (f*={kelly_full:.3f}, "
                f"WR={p*100:.0f}%, b={b:.2f}, {len(recent)} trades)")
            return 0.0

        kelly_adj = kelly_full * self.config.kelly_fraction
        kelly_pct = kelly_adj * 100.0

        # Edge Decay — compare recent half vs older half expectancy
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
                        kelly_pct *= 0.7
                        dir_tag = f" {direction}" if direction else ""
                        logger.info(
                            f"[Kelly] {symbol}{dir_tag}: Edge Decay — "
                            f"E_old={e_old:.3f} → E_new={e_new:.3f}, "
                            f"reducing Kelly 30%")
            except Exception:
                pass

        # Volatility Sizing cap — cap position by target vol / asset vol
        if analysis and analysis.get("asset_vol", 0) > 0:
            target_vol = 0.10  # 10% target portfolio volatility
            asset_vol = analysis["asset_vol"]
            vol_cap_frac = target_vol / asset_vol
            vol_cap_pct = vol_cap_frac * 100.0
            if vol_cap_pct < kelly_pct:
                dir_tag = f" {direction}" if direction else ""
                logger.info(
                    f"[Kelly] {symbol}{dir_tag}: Vol cap {vol_cap_pct:.1f}% "
                    f"(target_vol=10%, asset_vol={asset_vol:.1%}) "
                    f"< Kelly {kelly_pct:.1f}%")
                kelly_pct = vol_cap_pct

        kelly_pct = max(kelly_pct, self.config.kelly_min_pct)
        kelly_pct = min(kelly_pct, self.config.max_position_pct)

        dir_tag = f" {direction}" if direction else ""
        logger.info(
            f"[Kelly] {symbol}{dir_tag}: f*={kelly_full:.3f} "
            f"x{self.config.kelly_fraction} = {kelly_pct:.1f}% "
            f"(WR={p*100:.0f}%, b={b:.2f}, {len(recent)} trades)")
        return kelly_pct


# ==============================================================================
# FuturesEngine — Main 5-Minute Loop
# ==============================================================================

class FuturesEngine:
    """USDT-M Futures automated trading engine.

    Runs alongside spot TradingEngine. Same API keys, different endpoints.
    Every 5 minutes per symbol:
      analyze -> decide (LONG/SHORT/HOLD) -> execute -> save
    """

    def __init__(self, config, tool_executor=None, hmm=None, client=None,
                 cortex=None, exchange_name: str = "binance"):
        self.config = config.futures
        # Accept pre-built client or create one via factory
        if client is not None:
            self.client = client
        else:
            self.client = create_futures_client(
                exchange_name,
                self.config.api_key, self.config.api_secret,
                testnet=self.config.testnet,
            )
        trades_path = os.path.join(self.config.data_dir, "futures_trades.parquet")
        self.store = FuturesTradeStore(path=trades_path)

        # HMM regime detection — shared instance passed from runtime
        self.hmm = hmm
        if not hmm and _HAS_HMM:
            try:
                self.hmm = MarketHMM(data_dir=self.config.data_dir)
                self.hmm.load()
                logger.warning("[Futures] HMM created locally (not shared with spot)")
            except Exception:
                pass

        self.manager = FuturesPositionManager(
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
        self._leverage_set = False
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

    def _setup_symbols(self):
        """Set hedge mode + multi-asset mode + leverage + margin type."""
        if self._leverage_set:
            return
        # Account-level settings (hedge mode + multi-asset)
        if getattr(self.config, 'hedge_mode', True):
            try:
                self.client.set_hedge_mode(True)
                logger.info("[Futures] Hedge mode enabled (dual position side)")
            except Exception as e:
                logger.warning(f"[Futures] Hedge mode setup: {e}")
        if getattr(self.config, 'multi_asset_mode', True):
            try:
                self.client.set_multi_asset_mode(True)
                logger.info("[Futures] Multi-asset margin mode enabled")
            except Exception as e:
                logger.warning(f"[Futures] Multi-asset mode setup: {e}")
        # Per-symbol settings
        for symbol in self.config.symbols:
            pair = self.client.format_pair(symbol)
            try:
                self.client.set_leverage(pair, self.config.leverage)
                self.client.set_margin_type(pair, self.config.margin_type)
                logger.info(f"[Futures] {pair}: {self.config.leverage}x "
                            f"{self.config.margin_type}")
            except Exception as e:
                logger.warning(f"[Futures] Setup {pair} failed: {e}")
        self._leverage_set = True

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._setup_symbols()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="qor-futures")
        self._thread.start()
        hedge = "HEDGE" if getattr(self.config, 'hedge_mode', True) else "ONE-WAY"
        multi = "+MULTI-ASSET" if getattr(self.config, 'multi_asset_mode', True) else ""
        trade_mode = getattr(self.config, 'trade_mode', 'scalp').upper()
        logger.info(f"[Futures] Engine started -- symbols: {self.config.symbols}, "
                     f"interval: {self.config.check_interval_seconds}s, "
                     f"leverage: {self.config.leverage}x, "
                     f"margin: {self.config.margin_type}, "
                     f"position: {hedge}{multi}, "
                     f"trade_mode: {trade_mode}, "
                     f"mode: {'TESTNET' if self.config.testnet else 'PRODUCTION'}")
        logger.info("[Futures] NOTE: Ensure Hedge Mode and Multi-Asset Mode are "
                     "enabled in your Binance Futures settings (app or web).")

    def _loop(self):
        while not self._stop_event.is_set():
            try:
                self._tick()
                self._last_error = None
            except Exception as e:
                self._last_error = str(e)
                logger.error(f"[Futures] Tick error: {e}")
            self._stop_event.wait(timeout=self.config.check_interval_seconds)

    def _reconcile_positions(self, raw_positions: list = None):
        """Sync store with Binance — close trades that no longer exist on exchange.

        Args:
            raw_positions: Pre-fetched list from client.get_positions().
                If None, fetches from Binance (one API call).
        """
        open_trades = self.store.get_open_trades()
        if not open_trades:
            return
        if raw_positions is None:
            try:
                raw_positions = self.client.get_positions()
            except Exception as e:
                logger.warning(f"[Futures] Reconcile: cannot fetch positions: {e}")
                return
        exchange_positions = raw_positions
        # Build map: symbol -> positionAmt (absolute value) + notional
        DUST_NOTIONAL = 1.0
        exchange_qty = {}
        exchange_notional = {}
        for p in exchange_positions:
            sym = p.get("symbol", "")
            amt = abs(float(p.get("positionAmt", 0)))
            mark = float(p.get("markPrice", 0))
            notional = amt * mark if mark > 0 else abs(float(p.get("notional", 0)))
            # In hedge mode there are separate LONG/SHORT entries
            side = p.get("positionSide", "BOTH")
            key = f"{sym}:{side}"
            exchange_qty[key] = amt
            exchange_notional[key] = notional
            # Also store by symbol-only for non-hedge fallback
            if sym not in exchange_qty or amt > 0:
                exchange_qty[sym] = exchange_qty.get(sym, 0) + amt
                exchange_notional[sym] = exchange_notional.get(sym, 0) + notional

        for trade in open_trades:
            sym = trade["symbol"]
            pair = self.client.format_pair(sym)
            direction = trade.get("direction", "LONG")
            # Check hedge-mode key first, then plain symbol
            hedge_key = f"{pair}:{direction}"
            qty = exchange_qty.get(hedge_key, exchange_qty.get(pair, 0))
            notional = exchange_notional.get(hedge_key, exchange_notional.get(pair, 0))
            if qty <= 0 or notional < DUST_NOTIONAL:
                # Position gone or dust on Binance — close in store
                try:
                    price = self.client.get_price(pair)
                except Exception:
                    price = trade.get("entry_price", 0)
                reason = ("Position no longer exists on exchange" if qty <= 0
                          else f"Dust position: notional ${notional:.2f}")
                self.store.close_trade(
                    trade_id=trade["trade_id"],
                    exit_price=price,
                    status="closed_reconciled",
                    reason=reason,
                )
                # Cancel any remaining open orders (SL/TP) for this pair
                try:
                    open_orders = self.client.get_open_orders(pair)
                    for oo in open_orders:
                        if oo.get("type") in ("STOP_MARKET",
                                              "TAKE_PROFIT_MARKET"):
                            self.client.cancel_order(pair, oo["orderId"])
                            logger.info(
                                f"[Futures] Reconcile: cancelled "
                                f"{oo['type']} order {pair} "
                                f"#{oo['orderId']}")
                except Exception as ce:
                    logger.warning(
                        f"[Futures] Reconcile: cancel orders "
                        f"failed {pair}: {ce}")
                logger.info(f"[Futures] Reconciled: {trade['trade_id']} {sym} "
                            f"{direction} closed ({reason})")
                self._log_activity(sym, "CLOSE",
                                   f"Reconciled — {reason}")

    def _get_exchange_positions(self, raw_positions: list = None) -> dict:
        """Parse real positions from Binance. Returns {symbol: [pos_list]}.

        Args:
            raw_positions: Pre-fetched list from client.get_positions().
                If None, fetches from Binance.

        In hedge mode, a symbol can have both LONG and SHORT positions,
        so each symbol maps to a list of position dicts.
        Filters out dust positions (notional < $1).
        """
        DUST_NOTIONAL = 1.0  # Ignore positions worth less than $1
        result = {}
        try:
            positions = raw_positions if raw_positions is not None else self.client.get_positions()
            for p in positions:
                amt = float(p.get("positionAmt", 0))
                if amt == 0:
                    continue
                mark = float(p.get("markPrice", 0))
                notional = abs(amt) * mark if mark > 0 else abs(float(p.get("notional", 0)))
                if notional < DUST_NOTIONAL:
                    logger.debug(f"[Futures] Ignoring dust position: "
                                 f"{p['symbol']} amt={amt} notional=${notional:.2f}")
                    continue
                quote = getattr(self.client, 'quote', 'USDT')
                sym = p["symbol"].replace(quote, "")
                pos = {
                    "direction": "LONG" if amt > 0 else "SHORT",
                    "quantity": abs(amt),
                    "entry_price": float(p.get("entryPrice", 0)),
                    "mark_price": mark,
                    "pnl": float(p.get("unRealizedProfit", 0)),
                    "leverage": int(p.get("leverage", 1)),
                    "position_side": p.get("positionSide", "BOTH"),
                }
                if sym not in result:
                    result[sym] = []
                result[sym].append(pos)
        except Exception as e:
            logger.warning(f"[Futures] Cannot fetch exchange positions: {e}")
        return result

    def _tick(self):
        """One cycle: check Binance positions first, then analyze and act.

        Flow per symbol:
          1. Fetch REAL positions from Binance (source of truth)
          2. For each existing position → manage (SL, TP, DCA, trailing, close)
          3. If room for new positions → evaluate entry
             - Hedge mode: can open opposite direction if one side is open
             - One-way mode: only open if no position at all
          4. Respect allocated fund limit and max positions
        """
        self._tick_count += 1
        self._last_tick = datetime.now(timezone.utc).isoformat()
        logger.info(f"[Futures] === Tick #{self._tick_count} ===")
        self._log_activity("ALL", "SCAN", f"Tick #{self._tick_count} started")

        # Step 0: Fetch positions once from Binance (reused by reconcile + tick)
        try:
            raw_positions = self.client.get_positions()
        except Exception as e:
            logger.warning(f"[Futures] Cannot fetch positions: {e}")
            raw_positions = []

        # Reconcile — detect positions closed by exchange (SL/TP orders)
        try:
            self._reconcile_positions(raw_positions)
        except Exception as e:
            logger.warning(f"[Futures] Reconcile error: {e}")

        # Step 1: Parse what's ACTUALLY open on Binance (reuses same data)
        exchange_positions = self._get_exchange_positions(raw_positions)
        total_open = sum(len(plist) for plist in exchange_positions.values())
        if exchange_positions:
            summary = ", ".join(
                f"{s} {'/'.join(p['direction'] for p in ps)}"
                for s, ps in exchange_positions.items()
            )
            logger.info(f"[Futures] Open on Binance ({total_open}): {summary}")
            self._log_activity("ALL", "POSITIONS", f"Open ({total_open}): {summary}")

        # Step 2: Check allocated fund (reuse raw_positions to avoid extra API call)
        allocated = self.config.allocated_fund
        margin_used = self._total_margin_used(raw_positions) if allocated > 0 else 0
        if allocated > 0:
            logger.info(f"[Futures] Fund: ${margin_used:,.2f} / ${allocated:,.2f} used")

        hedge_mode = getattr(self.config, 'hedge_mode', True)

        for symbol in self.config.symbols:
            try:
                positions = exchange_positions.get(symbol, [])
                open_directions = {p["direction"] for p in positions}

                # --- Step A: Manage existing positions ---
                for ex_pos in positions:
                    decisions = self.manager.evaluate_symbol_with_position(
                        symbol, ex_pos)
                    self._execute_decisions(symbol, decisions)

                # --- Step B: Evaluate new entry if room ---
                # Check global limits
                if total_open >= self.config.max_open_positions:
                    if not positions:
                        self._log_activity(symbol, "HOLD",
                            f"Max positions ({total_open}/{self.config.max_open_positions})")
                    continue
                if allocated > 0 and margin_used >= allocated:
                    if not positions:
                        self._log_activity(symbol, "HOLD",
                            f"Fund limit (${margin_used:,.0f}/${allocated:,.0f})")
                    continue

                # Determine if we can open a new position
                can_open = False
                if hedge_mode:
                    # Hedge mode: can have LONG + SHORT simultaneously
                    # Can open if not both directions are taken
                    can_open = len(open_directions) < 2
                else:
                    # One-way mode: only open if no position at all
                    can_open = len(positions) == 0

                if can_open:
                    decisions = self.manager.evaluate_symbol(symbol)
                    # In hedge mode, filter out OPEN in a direction we already have
                    if open_directions:
                        decisions = [
                            d for d in decisions
                            if d["action"] != "OPEN"
                            or d.get("direction") not in open_directions
                        ]
                    self._execute_decisions(symbol, decisions)
                elif not positions:
                    self._log_activity(symbol, "HOLD", "No action needed")

            except Exception as e:
                logger.error(f"[Futures] Error processing {symbol}: {e}")
                self._log_activity(symbol, "ERROR", str(e))

        self.store.save()

    def _execute_decisions(self, symbol: str, decisions: list):
        """Execute a list of trading decisions."""
        for decision in decisions:
            action = decision["action"]
            reason = decision.get('reason', '')
            cortex_info = ""
            if decision.get("cortex_label"):
                cortex_info = (f" [CORTEX: {decision['cortex_label']} "
                               f"{decision.get('cortex_signal', 0):.2f}]")
            logger.info(f"[Futures] {symbol}: {action} -- {reason}{cortex_info}")
            self._log_activity(symbol, action, f"{reason}{cortex_info}")

            if action == "OPEN":
                self._execute_open(symbol, decision)
            elif action == "CLOSE":
                self._execute_close(symbol, decision)
            elif action == "PARTIAL_TP":
                self._execute_partial_tp(symbol, decision)
            elif action == "DCA":
                self._execute_dca(symbol, decision)
            elif action == "ADJUST_SL":
                self._execute_adjust_sl(decision)
            # HOLD -> no action

    # ------------------------------------------------------------------
    # Execution methods
    # ------------------------------------------------------------------

    def _total_margin_used(self, raw_positions: list = None) -> float:
        """Total margin currently used by open positions.

        Args:
            raw_positions: Pre-fetched positions list (avoids redundant API call).
                           If None, fetches from Binance.
        """
        try:
            positions = raw_positions if raw_positions is not None else self.client.get_positions()
            total = 0.0
            for p in positions:
                amt = abs(float(p.get("positionAmt", 0)))
                if amt > 0:
                    entry = float(p.get("entryPrice", 0))
                    lev = int(p.get("leverage", 1))
                    total += (amt * entry) / lev if lev > 0 else 0
            return total
        except Exception:
            # Fallback to local store
            return sum(t.get("margin_used", 0) for t in self.store.get_open_trades())

    def _calculate_liquidation_price(self, entry: float, leverage: int,
                                     direction: str) -> float:
        """Estimate liquidation price for isolated margin."""
        if leverage <= 0:
            return 0.0
        margin_ratio = 1.0 / leverage
        # Maintenance margin varies by tier on Binance (default 0.6%)
        maint = getattr(self.config, 'maintenance_margin_pct', 0.006)
        if direction == "LONG":
            return entry * (1 - margin_ratio + maint)
        else:
            return entry * (1 + margin_ratio - maint)

    def _execute_open(self, symbol: str, decision: dict):
        """Open a new LONG or SHORT position."""
        pair = self.client.format_pair(symbol)
        quote = getattr(self.client, 'quote', 'USDT')
        direction = decision["direction"]
        order_side = decision["order_side"]

        try:
            quote_balance = self.client.get_balance(quote)
        except Exception as e:
            logger.error(f"[Futures] Cannot get balance: {e}")
            return

        # Allocated fund cap — never use more than this
        allocated = self.config.allocated_fund
        if allocated > 0:
            available = min(quote_balance, allocated - self._total_margin_used())
        else:
            available = quote_balance

        if available < 5:
            logger.info(f"[Futures] Insufficient available fund: ${available:.2f} "
                         f"(balance=${quote_balance:.2f}, allocated=${allocated:.2f})")
            return

        # Position size computed in _evaluate_entry (Kelly + signal multipliers)
        leverage = self.config.leverage
        position_pct = decision.get("position_pct", self.config.max_position_pct)
        position_usdt = available * (position_pct / 100.0)
        notional = position_usdt * leverage
        if position_usdt < 5:
            logger.info(f"[Futures] Position too small: ${position_usdt:.2f} USDT")
            return

        try:
            price = self.client.get_price(pair)
        except Exception as e:
            logger.error(f"[Futures] Cannot get price for {pair}: {e}")
            return
        if price <= 0:
            return

        quantity = self.client.round_qty(pair, notional / price)
        try:
            lot = self.client.get_lot_size(pair)
            if quantity < lot["minQty"]:
                return
        except Exception:
            pass

        try:
            # In hedge mode, pass positionSide so Binance knows which side
            pos_side = direction if getattr(self.config, 'hedge_mode', True) else None
            order = self.client.place_order(pair, order_side, quantity,
                                            position_side=pos_side)
            fill_price = float(order.get("avgPrice", 0))
            if fill_price <= 0:
                fill_price = price
            fill_qty = float(order.get("executedQty", quantity))
        except Exception as e:
            logger.error(f"[Futures] OPEN {direction} failed {pair}: {e}")
            return

        if fill_qty <= 0:
            logger.warning(f"[Futures] OPEN {direction} {pair}: zero fill, skipping")
            return

        liq_price = self._calculate_liquidation_price(fill_price, leverage, direction)

        sl_price = decision["stop_loss"]
        tp_price = decision["take_profit"]

        # Reason already includes Kelly + signal multiplier info from _evaluate_entry
        reason = decision["reason"]

        self.store.open_trade(
            symbol=symbol, side=order_side,
            entry_price=fill_price, quantity=fill_qty,
            stop_loss=sl_price,
            take_profit=tp_price,
            take_profit2=decision.get("take_profit2", 0),
            reason=reason,
            direction=direction,
            leverage=leverage,
            margin_type=self.config.margin_type,
            liquidation_price=liq_price,
            margin_used=position_usdt,
        )

        # Place SL/TP orders on Binance (TP uses TP2 = full close target)
        pos_side = direction if getattr(self.config, 'hedge_mode', True) else None
        sl_side = "SELL" if direction == "LONG" else "BUY"
        tp2_price = decision.get("take_profit2", 0) or tp_price

        if sl_price > 0:
            try:
                self.client.place_order(
                    pair, sl_side, fill_qty,
                    order_type="STOP_MARKET", price=sl_price,
                    position_side=pos_side)
                logger.info(f"[Futures] SL order placed {pair} @ ${sl_price:,.2f}")
            except Exception as e:
                logger.warning(f"[Futures] SL order failed {pair}: {e}")

        if tp2_price > 0:
            try:
                self.client.place_order(
                    pair, sl_side, fill_qty,
                    order_type="TAKE_PROFIT_MARKET", price=tp2_price,
                    position_side=pos_side)
                logger.info(f"[Futures] TP order placed {pair} @ ${tp2_price:,.2f}")
            except Exception as e:
                logger.warning(f"[Futures] TP order failed {pair}: {e}")

    def _execute_close(self, symbol: str, decision: dict):
        """Close entire position (LONG or SHORT)."""
        pair = self.client.format_pair(symbol)
        trade_id = decision.get("trade_id", "")
        trade = self.store.trades.get(trade_id)
        if not trade:
            return

        close_side = decision.get("close_side", "SELL")
        # Use actual exchange position size to avoid qty mismatch
        direction = trade.get("direction", "LONG")
        actual_qty = trade["quantity"]
        try:
            pos = self.client.get_position(pair)
            exchange_amt = abs(float(pos.get("positionAmt", 0)))
            if exchange_amt > 0:
                actual_qty = min(trade["quantity"], exchange_amt)
        except Exception:
            pass
        quantity = self.client.round_qty(pair, actual_qty)
        if quantity <= 0:
            return

        try:
            # Hedge mode: use positionSide instead of reduceOnly
            direction = trade.get("direction", "LONG")
            pos_side = direction if getattr(self.config, 'hedge_mode', True) else None
            order = self.client.place_order(
                pair, close_side, quantity,
                reduce_only=not pos_side, position_side=pos_side)
            fill_price = float(order.get("avgPrice", 0))
            if fill_price <= 0:
                fill_price = self.client.get_price(pair)
        except Exception as e:
            logger.error(f"[Futures] CLOSE failed {pair}: {e}")
            return

        # Cancel open SL/TP orders for this symbol
        try:
            open_orders = self.client.get_open_orders(pair)
            for oo in open_orders:
                if oo.get("type") in ("STOP_MARKET", "TAKE_PROFIT_MARKET"):
                    self.client.cancel_order(pair, oo["orderId"])
                    logger.info(f"[Futures] Cancelled {oo['type']} order {pair}")
        except Exception as e:
            logger.warning(f"[Futures] Cancel SL/TP orders failed {pair}: {e}")

        self.store.close_trade(
            trade_id=trade_id, exit_price=fill_price,
            status=decision.get("exit_status", "closed_manual"),
            reason=decision["reason"],
        )

    def _execute_partial_tp(self, symbol: str, decision: dict):
        """Partial take profit (reduce position)."""
        pair = self.client.format_pair(symbol)
        trade_id = decision["trade_id"]
        trade = self.store.trades.get(trade_id)
        if not trade:
            return

        close_side = decision.get("close_side", "SELL")
        # Cap to actual exchange position to avoid qty mismatch
        max_qty = decision["sell_qty"]
        try:
            pos = self.client.get_position(pair)
            exchange_amt = abs(float(pos.get("positionAmt", 0)))
            if exchange_amt > 0:
                max_qty = min(max_qty, exchange_amt)
        except Exception:
            pass
        sell_qty = self.client.round_qty(pair, max_qty)
        if sell_qty <= 0:
            return

        try:
            direction = trade.get("direction", "LONG")
            pos_side = direction if getattr(self.config, 'hedge_mode', True) else None
            order = self.client.place_order(
                pair, close_side, sell_qty,
                reduce_only=not pos_side, position_side=pos_side)
            fill_price = float(order.get("avgPrice", 0))
            if fill_price <= 0:
                fill_price = self.client.get_price(pair)
        except Exception as e:
            logger.error(f"[Futures] PARTIAL_TP failed {pair}: {e}")
            return

        self.store.partial_close(
            trade_id=trade_id, sell_qty=sell_qty,
            sell_price=fill_price, reason=decision["reason"],
        )

        # After partial TP1: update SL and TP2 orders on Binance
        direction = trade.get("direction", "LONG")
        pos_side = direction if getattr(self.config, 'hedge_mode', True) else None
        sl_side = "SELL" if direction == "LONG" else "BUY"
        entry = trade.get("entry_price", 0)
        remaining_qty = self.client.round_qty(pair, trade["quantity"])

        # Cancel ALL existing SL/TP orders and replace with correct quantities
        try:
            open_orders = self.client.get_open_orders(pair)
            for oo in open_orders:
                if oo.get("type") in ("STOP_MARKET", "TAKE_PROFIT_MARKET"):
                    self.client.cancel_order(pair, oo["orderId"])
                    logger.info(f"[Futures] Cancelled {oo['type']} order {pair} "
                                f"(replacing after partial TP)")
        except Exception as e:
            logger.warning(f"[Futures] Cancel SL/TP orders failed {pair}: {e}")

        # Place new SL with correct remaining quantity
        if self.config.move_sl_to_be and entry > 0:
            try:
                self.client.place_order(
                    pair, sl_side, remaining_qty,
                    order_type="STOP_MARKET", price=entry,
                    position_side=pos_side)
                logger.info(f"[Futures] SL moved to break-even {pair} @ ${entry:,.2f} "
                            f"qty={remaining_qty}")
            except Exception as e:
                logger.warning(f"[Futures] SL break-even update failed {pair}: {e}")

        # Place new TP2 with remaining quantity (not closePosition)
        tp2_price = trade.get("take_profit2", 0) or trade.get("take_profit", 0)
        if tp2_price > 0 and remaining_qty > 0:
            try:
                self.client.place_order(
                    pair, sl_side, remaining_qty,
                    order_type="TAKE_PROFIT_MARKET", price=tp2_price,
                    position_side=pos_side)
                logger.info(f"[Futures] TP2 replaced {pair} @ ${tp2_price:,.2f} "
                            f"qty={remaining_qty}")
            except Exception as e:
                logger.warning(f"[Futures] TP2 replacement failed {pair}: {e}")

    def _execute_dca(self, symbol: str, decision: dict):
        """DCA: add to position (LONG or SHORT)."""
        pair = self.client.format_pair(symbol)
        quote = getattr(self.client, 'quote', 'USDT')
        trade_id = decision["trade_id"]
        trade = self.store.trades.get(trade_id)
        if not trade:
            return

        try:
            quote_balance = self.client.get_balance(quote)
        except Exception as e:
            logger.error(f"[Futures] Cannot get balance for DCA: {e}")
            return

        dca_num = decision.get("dca_number", 1)
        base_usdt = trade.get("margin_used", 0) / max(trade.get("dca_count", 0) + 1, 1)
        dca_usdt = base_usdt * (self.config.dca_multiplier ** (dca_num - 1))
        dca_usdt = min(dca_usdt, quote_balance * 0.5)

        if dca_usdt < 5:
            logger.info(f"[Futures] DCA skipped: insufficient balance ${quote_balance:.2f}")
            return

        try:
            price = self.client.get_price(pair)
        except Exception:
            return
        if price <= 0:
            return

        leverage = self.config.leverage
        notional = dca_usdt * leverage
        quantity = self.client.round_qty(pair, notional / price)
        dca_side = decision.get("dca_side", "BUY")

        try:
            direction = trade.get("direction", "LONG")
            pos_side = direction if getattr(self.config, 'hedge_mode', True) else None
            order = self.client.place_order(pair, dca_side, quantity,
                                            position_side=pos_side)
            fill_price = float(order.get("avgPrice", 0))
            if fill_price <= 0:
                fill_price = price
            fill_qty = float(order.get("executedQty", quantity))
        except Exception as e:
            logger.error(f"[Futures] DCA failed {pair}: {e}")
            return

        self.store.add_dca(trade_id=trade_id, add_price=fill_price, add_qty=fill_qty)
        trade["margin_used"] = trade.get("margin_used", 0) + dca_usdt

    def _execute_adjust_sl(self, decision: dict):
        trade_id = decision.get("trade_id", "")
        new_sl = decision.get("new_sl", 0)
        if trade_id and new_sl > 0:
            self.store.update_sl_tp(trade_id, new_sl=new_sl)
            # Update SL order on Binance
            trade = self.store.trades.get(trade_id)
            if trade:
                pair = self.client.format_pair(trade['symbol'])
                direction = trade.get("direction", "LONG")
                pos_side = direction if getattr(self.config, 'hedge_mode', True) else None
                sl_side = "SELL" if direction == "LONG" else "BUY"
                try:
                    # Cancel old SL order, place new one
                    for oo in self.client.get_open_orders(pair):
                        if oo.get("type") == "STOP_MARKET":
                            self.client.cancel_order(pair, oo["orderId"])
                    self.client.place_order(
                        pair, sl_side, trade["quantity"],
                        order_type="STOP_MARKET", price=new_sl,
                        position_side=pos_side)
                    logger.info(f"[Futures] SL updated {pair} @ ${new_sl:,.2f}")
                except Exception as e:
                    logger.warning(f"[Futures] SL update failed {pair}: {e}")

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
                logger.info(f"[Futures] CORTEX saved to {cortex_path}")
            except Exception as e:
                logger.warning(f"[Futures] CORTEX save failed: {e}")
        if self.hmm:
            try:
                self.hmm.save()
                logger.info("[Futures] HMM saved")
            except Exception as e:
                logger.warning(f"[Futures] HMM save failed: {e}")
        logger.info("[Futures] Engine stopped")

    def set_leverage(self, leverage: int):
        """Change leverage for all symbols (live)."""
        leverage = max(1, min(leverage, self.config.max_leverage))
        self.config.leverage = leverage
        self._leverage_set = False
        self._setup_symbols()
        logger.info(f"[Futures] Leverage changed to {leverage}x")

    def status(self) -> dict:
        stats = self.store.get_stats()
        trade_mode = getattr(self.config, 'trade_mode', 'scalp')
        result = {
            "enabled": True,
            "mode": "TESTNET" if self.config.testnet else "PRODUCTION",
            "trade_mode": trade_mode.upper(),
            "leverage": self.config.leverage,
            "margin_type": self.config.margin_type,
            "hedge_mode": getattr(self.config, 'hedge_mode', True),
            "multi_asset_mode": getattr(self.config, 'multi_asset_mode', True),
            "symbols": self.config.symbols,
            "open_positions": len(self.store.get_open_trades()),
            "total_trades": stats.get("total_trades", 0),
            "total_funding_paid": stats.get("total_funding_paid", 0.0),
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


def create_futures_client(exchange_name: str, api_key: str, api_secret: str,
                          passphrase: str = "", testnet: bool = True,
                          base_url: str = "", **kwargs):
    """Factory: create the right futures client by exchange name.

    Returns a client with supports_futures=True that FuturesEngine can use.
    For exchanges without a dedicated futures client, falls back to the
    spot ExchangeClient (which has default no-op futures methods).

    Args:
        exchange_name: "binance", "bybit", "okx", etc.
        api_key, api_secret: Credentials
        testnet: Use sandbox/demo mode

    Returns:
        Client instance with futures support
    """
    name = exchange_name.lower().replace("-", "_").replace(" ", "_")

    if name in ("binance", "binance_futures"):
        return BinanceFuturesClient(api_key, api_secret, testnet=testnet)

    # For other exchanges, use the spot client from trading.py
    # (has default no-op futures methods via ExchangeClient base)
    from qor.trading import create_exchange_client
    return create_exchange_client(
        name, api_key, api_secret,
        passphrase=passphrase, testnet=testnet,
        base_url=base_url, **kwargs)

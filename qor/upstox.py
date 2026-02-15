"""
UpstoxClient — Indian Markets (NSE/BSE/MCX) Trading

Implements ExchangeClient for Upstox broker covering:
  - NSE equities (RELIANCE, TCS, INFY, etc.)
  - BSE equities
  - MCX commodities (Gold, Silver, Crude Oil)
  - NSE indices (NIFTY, BANKNIFTY)

Key differences from Binance:
  - OAuth2 Bearer token auth (daily refresh) instead of HMAC signing
  - instrument_key format ("NSE_EQ|INE002A01018") instead of simple pairs
  - Integer quantities (whole shares) instead of fractional
  - Order placement returns order_id, needs polling for fill data
"""

import gzip
import io
import json
import logging
import math
import time
import urllib.request
import urllib.parse

from qor.trading import ExchangeClient

logger = logging.getLogger(__name__)


class UpstoxClient(ExchangeClient):
    """Upstox API client for Indian markets (NSE/BSE/MCX).

    Uses OAuth2 Bearer token authentication. The access_token is obtained
    via OAuth2 authorization code flow and is valid until 3:30 AM IST next day.

    Usage:
        client = UpstoxClient(api_key, api_secret, access_token="...")
        price = client.get_price("RELIANCE")
        client.place_order("RELIANCE", "BUY", 10)
    """

    BASE_URL = "https://api.upstox.com/v2"
    HFT_URL = "https://api-hft.upstox.com/v3"

    name = "upstox"
    quote = "INR"

    # Exchanges to search when resolving symbols
    # NSE_FO/BSE_FO/MCX_FO = futures & options segments
    EXCHANGES = ["NSE", "BSE", "MCX", "NSE_FO", "BSE_FO", "MCX_FO"]
    supports_futures = True

    # Instrument data URLs
    INSTRUMENT_URL = "https://assets.upstox.com/market-quote/instruments/exchange/{exchange}.json.gz"

    def __init__(self, api_key: str, api_secret: str, testnet: bool = False,
                 access_token: str = "", **kwargs):
        self.api_key = api_key          # = client_id
        self.api_secret = api_secret    # = client_secret
        self.access_token = access_token
        self.base_url = self.BASE_URL
        self.hft_url = self.HFT_URL
        self._instruments = {}          # exchange → list of instrument dicts
        self._symbol_cache = {}         # symbol (upper) → instrument_key
        self._lot_cache = {}            # instrument_key → {lot_size, tick_size, freeze_quantity}

    # ── HTTP helper ──────────────────────────────────────────────────────

    def _request(self, method: str, endpoint: str, params: dict = None,
                 body: dict = None, hft: bool = False) -> dict:
        """Make authenticated API request.

        Args:
            method: "GET", "POST", "PUT", "DELETE"
            endpoint: API path (e.g. "/v2/market-quote/ltp")
            params: URL query parameters
            body: JSON request body (for POST/PUT)
            hft: Use HFT endpoint (for order placement)

        Returns:
            Parsed JSON response dict
        """
        base = self.hft_url if hft else self.base_url
        url = f"{base}{endpoint}"

        if params:
            qs = urllib.parse.urlencode(params)
            url = f"{url}?{qs}"

        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Accept": "application/json",
        }

        data = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body).encode("utf-8")

        req = urllib.request.Request(url, data=data, headers=headers, method=method)

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as e:
            error_body = ""
            try:
                error_body = e.read().decode("utf-8")
            except Exception:
                pass
            logger.error(f"[Upstox] HTTP {e.code} {method} {endpoint}: {error_body}")
            raise
        except Exception as e:
            logger.error(f"[Upstox] Request failed {method} {endpoint}: {e}")
            raise

    # ── Instrument resolution ────────────────────────────────────────────

    def _load_instruments(self, exchange: str):
        """Download and cache instrument data for an exchange.

        Upstox publishes instrument files as gzipped JSON at:
        https://assets.upstox.com/market-quote/instruments/exchange/{exchange}.json.gz
        """
        if exchange in self._instruments:
            return

        url = self.INSTRUMENT_URL.format(exchange=exchange)
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=30) as resp:
                compressed = resp.read()
                raw = gzip.decompress(compressed).decode("utf-8")
                instruments = json.loads(raw)
                self._instruments[exchange] = instruments

                # Index by trading_symbol for fast lookup
                for inst in instruments:
                    sym = inst.get("trading_symbol", "").upper()
                    ikey = inst.get("instrument_key", "")
                    if sym and ikey:
                        # Prefer NSE over BSE for equities
                        if sym not in self._symbol_cache or exchange == "NSE":
                            self._symbol_cache[sym] = ikey
                        # Cache lot size info
                        self._lot_cache[ikey] = {
                            "lot_size": inst.get("lot_size", 1),
                            "tick_size": inst.get("tick_size", 0.05),
                            "freeze_quantity": inst.get("freeze_quantity", 0),
                        }

                logger.info(f"[Upstox] Loaded {len(instruments)} instruments for {exchange}")
        except Exception as e:
            logger.warning(f"[Upstox] Failed to load instruments for {exchange}: {e}")
            self._instruments[exchange] = []

    def _ensure_instruments(self):
        """Load all exchange instruments if not already cached."""
        for exchange in self.EXCHANGES:
            self._load_instruments(exchange)

    def _resolve_instrument_key(self, symbol: str) -> str:
        """Resolve a human symbol to Upstox instrument_key.

        Examples:
            "RELIANCE" → "NSE_EQ|INE002A01018"
            "TCS"      → "NSE_EQ|INE467B01029"
            "GOLDM"    → "MCX_FO|..."

        Search order:
            1. Exact trading_symbol match (case-insensitive)
            2. Name substring match (first result)

        Returns:
            instrument_key string

        Raises:
            ValueError if symbol cannot be resolved
        """
        self._ensure_instruments()

        sym_upper = symbol.upper().strip()

        # Direct cache hit
        if sym_upper in self._symbol_cache:
            return self._symbol_cache[sym_upper]

        # Search by name across all loaded exchanges
        for exchange in self.EXCHANGES:
            for inst in self._instruments.get(exchange, []):
                name = inst.get("name", "").upper()
                if sym_upper in name:
                    ikey = inst.get("instrument_key", "")
                    if ikey:
                        self._symbol_cache[sym_upper] = ikey
                        return ikey

        raise ValueError(
            f"Cannot resolve symbol '{symbol}' to Upstox instrument_key. "
            f"Searched {sum(len(v) for v in self._instruments.values())} instruments "
            f"across {', '.join(self.EXCHANGES)}."
        )

    # ── ExchangeClient interface (8 methods) ─────────────────────────────

    def format_pair(self, symbol: str) -> str:
        """Convert symbol to Upstox instrument_key.

        "RELIANCE" → "NSE_EQ|INE002A01018"
        """
        return self._resolve_instrument_key(symbol)

    def get_price(self, symbol: str) -> float:
        """Get last traded price for a symbol.

        GET /v2/market-quote/ltp?instrument_key={key}
        """
        ikey = self._resolve_instrument_key(symbol)
        encoded_key = urllib.parse.quote(ikey, safe="")
        resp = self._request("GET", "/v2/market-quote/ltp",
                             params={"instrument_key": encoded_key})

        # Response: {"status": "success", "data": {"NSE_EQ:INE002A01018": {"last_price": 2450.5, ...}}}
        data = resp.get("data", {})
        for key_data in data.values():
            ltp = key_data.get("last_price")
            if ltp is not None:
                return float(ltp)

        raise ValueError(f"No price data for {symbol} ({ikey})")

    def get_balance(self, asset: str = "INR") -> float:
        """Get available margin.

        GET /v2/user/get-funds-and-margin?segment=SEC
        For MCX commodities: segment=COM
        """
        segment = "COM" if asset.upper() == "COM" else "SEC"
        resp = self._request("GET", "/v2/user/get-funds-and-margin",
                             params={"segment": segment})

        data = resp.get("data", {})
        # Available margin field
        available = data.get("available_margin", 0)
        if available:
            return float(available)

        # Fallback: try equity field
        equity = data.get("equity", {})
        if isinstance(equity, dict):
            return float(equity.get("available_margin", 0))

        return 0.0

    def get_account(self) -> dict:
        """Get full account info with balances.

        Returns dict with 'balances' key matching ExchangeClient contract:
        {"balances": [{"asset": "INR", "free": X, "locked": Y}]}
        """
        balances = []

        for segment in ("SEC", "COM"):
            try:
                resp = self._request("GET", "/v2/user/get-funds-and-margin",
                                     params={"segment": segment})
                data = resp.get("data", {})
                available = float(data.get("available_margin", 0))
                used = float(data.get("used_margin", 0))
                label = "INR" if segment == "SEC" else "INR_COM"
                balances.append({
                    "asset": label,
                    "free": available,
                    "locked": used,
                })
            except Exception as e:
                logger.debug(f"[Upstox] get_account segment={segment} failed: {e}")

        return {"balances": balances}

    def place_order(self, symbol: str, side: str, quantity: float,
                    order_type: str = "MARKET", price: float = None) -> dict:
        """Place an order on Upstox.

        POST api-hft.upstox.com/v3/order/place

        Args:
            symbol: Trading symbol (e.g. "RELIANCE")
            side: "BUY" or "SELL"
            quantity: Number of shares (rounded to lot size)
            order_type: "MARKET" or "LIMIT"
            price: Required for LIMIT orders

        Returns:
            Dict with 'fills' and 'executedQty' matching ExchangeClient contract
        """
        ikey = self._resolve_instrument_key(symbol)
        qty = int(self.round_qty(symbol, quantity))

        # F&O instruments use INTRADAY product, equities use DELIVERY
        is_fo = "_FO|" in ikey  # NSE_FO|..., MCX_FO|..., BSE_FO|...
        product = "I" if is_fo else "D"  # I=Intraday/F&O, D=Delivery/CNC

        body = {
            "instrument_token": ikey,
            "quantity": qty,
            "transaction_type": side.upper(),
            "order_type": order_type.upper(),
            "product": product,
            "validity": "DAY",
            "price": 0,
            "trigger_price": 0,
            "disclosed_quantity": 0,
            "is_amo": False,
        }

        if order_type.upper() == "LIMIT" and price is not None:
            body["price"] = float(price)

        # Place order via HFT endpoint
        resp = self._request("POST", "/v3/order/place", body=body, hft=True)

        # Response: {"status": "success", "data": {"order_ids": ["abc123"]}}
        data = resp.get("data", {})
        order_ids = data.get("order_ids", [])

        if not order_ids:
            raise ValueError(f"Order placement failed: {resp}")

        order_id = order_ids[0]

        # Poll for fill data (market orders fill almost instantly)
        fill_price, fill_qty = self._poll_order_fill(order_id, timeout=10)

        return {
            "orderId": order_id,
            "fills": [{"price": fill_price, "qty": fill_qty}] if fill_price else [],
            "executedQty": fill_qty,
            "status": "FILLED" if fill_qty > 0 else "PENDING",
        }

    def _poll_order_fill(self, order_id: str, timeout: int = 10) -> tuple:
        """Poll order details until filled or timeout.

        Returns:
            (fill_price, fill_qty) tuple
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                resp = self._request("GET", "/v2/order/details",
                                     params={"order_id": order_id})
                data = resp.get("data", {})
                status = data.get("status", "").lower()
                if status in ("complete", "filled", "traded"):
                    avg_price = float(data.get("average_price", 0))
                    filled_qty = int(data.get("filled_quantity",
                                              data.get("quantity", 0)))
                    return avg_price, filled_qty
                if status in ("rejected", "cancelled"):
                    logger.warning(f"[Upstox] Order {order_id} {status}: "
                                   f"{data.get('status_message', '')}")
                    return 0, 0
            except Exception as e:
                logger.debug(f"[Upstox] Poll error for {order_id}: {e}")

            time.sleep(0.5)

        # Timeout — return whatever we have
        logger.warning(f"[Upstox] Order {order_id} fill poll timed out after {timeout}s")
        return 0, 0

    def round_qty(self, symbol: str, quantity: float) -> float:
        """Round quantity to lot size (Indian markets use whole shares).

        Returns integer quantity rounded down to nearest lot_size.
        """
        lot_info = self._get_lot_info(symbol)
        lot_size = lot_info.get("lot_size", 1)
        if lot_size <= 0:
            lot_size = 1
        # Round down to nearest lot
        rounded = math.floor(quantity / lot_size) * lot_size
        return float(max(rounded, lot_size))

    def get_lot_size(self, symbol: str) -> dict:
        """Get lot size info for a symbol.

        Returns: {"min_qty": lot_size, "max_qty": freeze_quantity, "step_size": lot_size}
        """
        lot_info = self._get_lot_info(symbol)
        lot = lot_info.get("lot_size", 1)
        freeze = lot_info.get("freeze_quantity", 0)
        return {
            "min_qty": float(lot),
            "max_qty": float(freeze) if freeze > 0 else 100000.0,
            "step_size": float(lot),
        }

    def _get_lot_info(self, symbol: str) -> dict:
        """Get cached lot info for a symbol, loading instruments if needed."""
        ikey = self._resolve_instrument_key(symbol)
        if ikey in self._lot_cache:
            return self._lot_cache[ikey]
        # Fallback defaults for equity
        return {"lot_size": 1, "tick_size": 0.05, "freeze_quantity": 0}

    def cancel_order(self, symbol: str, order_id) -> dict:
        """Cancel an open order.

        DELETE /v3/order/cancel?order_id={id}
        """
        resp = self._request("DELETE", "/v3/order/cancel",
                             params={"order_id": str(order_id)}, hft=True)
        return resp.get("data", resp)

    # ── Futures methods (ExchangeClient interface) ──────────────────────

    def get_positions(self) -> list:
        """Get all open positions (equity + F&O + MCX).

        GET /v2/portfolio/short-term-positions

        Returns list of dicts matching ExchangeClient futures contract:
        [{symbol, positionAmt, entryPrice, markPrice, unRealizedProfit, ...}]
        """
        try:
            resp = self._request("GET", "/v2/portfolio/short-term-positions")
        except Exception as e:
            logger.debug(f"[Upstox] get_positions failed: {e}")
            return []

        data = resp.get("data", [])
        if not isinstance(data, list):
            return []

        positions = []
        for p in data:
            qty = int(p.get("quantity", 0) or 0)
            if qty == 0:
                continue
            entry_price = float(p.get("average_price", 0) or 0)
            ltp = float(p.get("last_price", 0) or 0)
            pnl = float(p.get("pnl", 0) or 0)
            positions.append({
                "symbol": p.get("trading_symbol", p.get("instrument_token", "")),
                "positionAmt": qty,
                "entryPrice": entry_price,
                "markPrice": ltp,
                "unRealizedProfit": pnl,
                "leverage": 1,
                "marginType": "NRML",
                "instrument_key": p.get("instrument_token", ""),
                "exchange": p.get("exchange", ""),
                "product": p.get("product", ""),
            })
        return positions

    def get_position(self, symbol: str) -> dict:
        """Get position for a specific symbol."""
        for p in self.get_positions():
            sym = p.get("symbol", "").upper()
            if symbol.upper() in sym:
                return p
        return {}

    def get_open_orders(self, symbol: str = None) -> list:
        """Get open/pending orders.

        GET /v2/order/retrieve-all
        """
        try:
            resp = self._request("GET", "/v2/order/retrieve-all")
        except Exception as e:
            logger.debug(f"[Upstox] get_open_orders failed: {e}")
            return []

        data = resp.get("data", [])
        if not isinstance(data, list):
            return []

        orders = []
        for o in data:
            status = (o.get("status", "") or "").lower()
            if status not in ("open", "pending", "trigger pending", "after market order req received"):
                continue
            order_sym = o.get("trading_symbol", "")
            if symbol and symbol.upper() not in order_sym.upper():
                continue
            orders.append({
                "orderId": o.get("order_id", ""),
                "symbol": order_sym,
                "side": o.get("transaction_type", ""),
                "type": o.get("order_type", ""),
                "quantity": int(o.get("quantity", 0) or 0),
                "price": float(o.get("price", 0) or 0),
                "stopPrice": float(o.get("trigger_price", 0) or 0),
                "status": status,
            })
        return orders

    def get_mark_price(self, symbol: str) -> dict:
        """Get current price as mark price (no funding rate for Indian markets)."""
        try:
            price = self.get_price(symbol)
        except Exception:
            price = 0
        return {
            "markPrice": str(price),
            "lastFundingRate": "0",
            "nextFundingTime": 0,
        }

    # ── OAuth2 token helper ──────────────────────────────────────────────

    @staticmethod
    def get_access_token(code: str, client_id: str, client_secret: str,
                         redirect_uri: str) -> dict:
        """Exchange OAuth2 authorization code for access token.

        POST https://api.upstox.com/v2/login/authorization/token

        This is called once when the user authenticates via browser.
        The returned access_token is valid until 3:30 AM IST next day.

        Args:
            code: Authorization code from OAuth2 redirect
            client_id: Upstox API key (app client_id)
            client_secret: Upstox API secret (app client_secret)
            redirect_uri: Must match the redirect URI registered in Upstox app

        Returns:
            Dict with access_token, extended_token, user_id, exchanges, etc.
        """
        url = "https://api.upstox.com/v2/login/authorization/token"

        body = urllib.parse.urlencode({
            "code": code,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        }).encode("utf-8")

        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        }

        req = urllib.request.Request(url, data=body, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw)
        except urllib.error.HTTPError as e:
            error_body = ""
            try:
                error_body = e.read().decode("utf-8")
            except Exception:
                pass
            raise ValueError(f"Token exchange failed (HTTP {e.code}): {error_body}")

    # ── Status ───────────────────────────────────────────────────────────

    def status(self) -> dict:
        """Return client status info."""
        return {
            "exchange": self.name,
            "quote": self.quote,
            "has_token": bool(self.access_token),
            "instruments_loaded": sum(len(v) for v in self._instruments.values()),
            "symbols_cached": len(self._symbol_cache),
        }

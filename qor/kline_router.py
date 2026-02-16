"""
QOR Universal Kline Router — Multi-Exchange Historical OHLCV
===============================================================
Routes get_klines() calls to the correct API based on asset type:

  Crypto (BTC, ETH, SOL)         → Binance /api/v3/klines
  Indian stocks (RELIANCE, TCS)  → Upstox V3 /v3/historical-candle/...
  Indian commodities (GOLDM)     → Upstox V3 (MCX segment)
  Indian futures/options          → Upstox V3 (NSE_FO/MCX_FO)
  US stocks (AAPL, TSLA, NVDA)   → Alpaca /v2/stocks/{sym}/bars
  Forex (EUR/USD, USD/JPY)       → OANDA /v3/instruments/{pair}/candles
  Gold, Silver, Oil (XAU, XAG)   → OANDA (XAU_USD, XAG_USD CFDs)
  Everything else                 → Yahoo Finance (free, no API key)

ALL clients return data in unified Binance kline format so
CortexTrainer works identically on any asset type:
  [open_time_ms, open, high, low, close, volume, close_time_ms,
   quote_vol, trades, taker_buy_base, taker_buy_quote, ignore]

Usage:
    from qor.kline_router import KlineRouter

    router = KlineRouter(config)
    klines = router.get_klines("BTC", interval="1h", days=90)    # → Binance
    klines = router.get_klines("RELIANCE", interval="1h", days=90) # → Upstox
    klines = router.get_klines("AAPL", interval="1h", days=90)    # → Alpaca
    klines = router.get_klines("EUR/USD", interval="1h", days=90) # → OANDA
    klines = router.get_klines("gold", interval="1d", days=90)    # → OANDA

    # With CortexTrainer:
    from qor.train_cortex import CortexTrainer
    trainer = CortexTrainer(router, cortex_analyzer)
    result = trainer.train_all(["BTC", "AAPL", "RELIANCE", "gold"])
"""

import json
import logging
import math
import time
import urllib.parse
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta
from typing import List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Unified Kline Format
# =============================================================================
# All clients return: [open_time, O, H, L, C, vol, close_time, qvol, n, tbv, tbq, _]
# This matches Binance format exactly so CortexTrainer.Candles parses it unchanged.


def _to_binance_format(open_time_ms: int, o: float, h: float, l: float,
                       c: float, v: float, close_time_ms: int = 0) -> list:
    """Convert OHLCV to Binance kline list format."""
    if close_time_ms == 0:
        close_time_ms = open_time_ms + 59999
    return [
        open_time_ms, str(o), str(h), str(l), str(c), str(v),
        close_time_ms, "0", 0, "0", "0", "0",
    ]


# =============================================================================
# Alpaca Client — US Stocks (AAPL, TSLA, MSFT, NVDA, SPY, QQQ)
# =============================================================================

class AlpacaKlineClient:
    """Alpaca Market Data API v2 — historical bars for US equities.

    Endpoint: GET https://data.alpaca.markets/v2/stocks/{symbol}/bars
    Auth: APCA-API-KEY-ID + APCA-API-SECRET-KEY headers
    Free tier: IEX data, 200 req/min. Algo Trader Plus: full SIP data.

    Intervals: 1Min, 5Min, 15Min, 30Min, 1Hour, 4Hour, 1Day, 1Week, 1Month
    Max 10000 bars per request, paginated via next_page_token.
    """

    BASE_URL = "https://data.alpaca.markets"
    PAPER_URL = "https://data.sandbox.alpaca.markets"

    # Map our interval names to Alpaca format
    INTERVAL_MAP = {
        "1m": "1Min", "3m": "5Min", "5m": "5Min",
        "15m": "15Min", "30m": "30Min",
        "1h": "1Hour", "2h": "1Hour", "4h": "4Hour",
        "1d": "1Day", "1w": "1Week", "1M": "1Month",
    }

    def __init__(self, api_key: str, api_secret: str, paper: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = self.PAPER_URL if paper else self.BASE_URL

    def get_klines(self, symbol: str, interval: str = "1h",
                   limit: int = 1500, start_time: int = None,
                   end_time: int = None) -> list:
        """Fetch historical bars from Alpaca.

        Returns data in Binance kline format for CortexTrainer compatibility.
        """
        tf = self.INTERVAL_MAP.get(interval, "1Hour")

        params = {
            "timeframe": tf,
            "limit": min(limit, 10000),
            "adjustment": "split",  # Adjust for stock splits
            "feed": "iex",          # Free tier; use "sip" for paid
        }

        if start_time:
            params["start"] = datetime.fromtimestamp(
                start_time / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if end_time:
            params["end"] = datetime.fromtimestamp(
                end_time / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        sym = symbol.upper().replace(" ", "")
        url = f"{self.base_url}/v2/stocks/{sym}/bars"
        qs = urllib.parse.urlencode(params)
        url = f"{url}?{qs}"

        headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
            "Accept": "application/json",
        }

        all_bars = []
        next_token = None

        for _ in range(10):  # Max 10 pages
            page_url = url
            if next_token:
                page_url = f"{url}&page_token={next_token}"

            req = urllib.request.Request(page_url, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = json.loads(resp.read().decode())
            except Exception as e:
                logger.error(f"[Alpaca] Klines failed for {sym}: {e}")
                break

            bars = data.get("bars", [])
            if not bars:
                break

            for bar in bars:
                # Alpaca bar: {t, o, h, l, c, v, n, vw}
                ts = bar.get("t", "")
                try:
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    open_ms = int(dt.timestamp() * 1000)
                except Exception:
                    continue

                all_bars.append(_to_binance_format(
                    open_ms,
                    float(bar.get("o", 0)),
                    float(bar.get("h", 0)),
                    float(bar.get("l", 0)),
                    float(bar.get("c", 0)),
                    float(bar.get("v", 0)),
                ))

            next_token = data.get("next_page_token")
            if not next_token or len(all_bars) >= limit:
                break
            time.sleep(0.2)

        logger.info(f"[Alpaca] Fetched {len(all_bars)} bars for {sym}")
        return all_bars[:limit]

    def format_pair(self, symbol: str) -> str:
        return symbol.upper()


# =============================================================================
# OANDA Client — Forex + Gold/Silver/Oil CFDs
# =============================================================================

class OandaKlineClient:
    """OANDA v20 REST API — historical candles for forex and CFDs.

    Endpoint: GET /v3/instruments/{instrument}/candles
    Auth: Bearer token in Authorization header
    Max 5000 candles per request.

    Supports:
      Forex: EUR_USD, GBP_USD, USD_JPY, USD_INR, etc.
      Metals: XAU_USD (gold), XAG_USD (silver), XPT_USD (platinum)
      Energy: BCO_USD (Brent), WTICO_USD (WTI crude), NATGAS_USD
      Indices: US30_USD (Dow), SPX500_USD, NAS100_USD

    Granularity: S5, S10, S15, S30, M1, M2, M4, M5, M10, M15, M30,
                 H1, H2, H3, H4, H6, H8, H12, D, W, M
    """

    PRACTICE_URL = "https://api-fxpractice.oanda.com"
    LIVE_URL = "https://api-fxtrade.oanda.com"

    INTERVAL_MAP = {
        "1m": "M1", "3m": "M5", "5m": "M5",
        "15m": "M15", "30m": "M30",
        "1h": "H1", "2h": "H2", "4h": "H4",
        "1d": "D", "1w": "W", "1M": "M",
    }

    # Map common names to OANDA instrument codes
    INSTRUMENT_MAP = {
        # Forex
        "eur/usd": "EUR_USD", "eurusd": "EUR_USD",
        "gbp/usd": "GBP_USD", "gbpusd": "GBP_USD",
        "usd/jpy": "USD_JPY", "usdjpy": "USD_JPY",
        "usd/cad": "USD_CAD", "usdcad": "USD_CAD",
        "aud/usd": "AUD_USD", "audusd": "AUD_USD",
        "nzd/usd": "NZD_USD", "nzdusd": "NZD_USD",
        "usd/chf": "USD_CHF", "usdchf": "USD_CHF",
        "eur/gbp": "EUR_GBP", "eurgbp": "EUR_GBP",
        "eur/jpy": "EUR_JPY", "eurjpy": "EUR_JPY",
        "usd/inr": "USD_INR", "usdinr": "USD_INR",
        # Metals
        "gold": "XAU_USD", "xau": "XAU_USD", "xauusd": "XAU_USD",
        "silver": "XAG_USD", "xag": "XAG_USD", "xagusd": "XAG_USD",
        "platinum": "XPT_USD", "xpt": "XPT_USD",
        "palladium": "XPD_USD",
        # Energy
        "oil": "WTICO_USD", "crude": "WTICO_USD", "wti": "WTICO_USD",
        "crude oil": "WTICO_USD", "crudeoil": "WTICO_USD",
        "brent": "BCO_USD", "brent crude": "BCO_USD",
        "natural gas": "NATGAS_USD", "natgas": "NATGAS_USD",
        # Indices (for training, not trading)
        "sp500": "SPX500_USD", "spx": "SPX500_USD",
        "dow": "US30_USD", "us30": "US30_USD",
        "nasdaq": "NAS100_USD", "nas100": "NAS100_USD",
    }

    def __init__(self, api_token: str, account_id: str = "",
                 practice: bool = True):
        self.api_token = api_token
        self.account_id = account_id
        self.base_url = self.PRACTICE_URL if practice else self.LIVE_URL

    def _resolve_instrument(self, symbol: str) -> str:
        """Map symbol to OANDA instrument code."""
        sym = symbol.lower().strip()
        if sym in self.INSTRUMENT_MAP:
            return self.INSTRUMENT_MAP[sym]
        # Try as-is with underscore format
        if "_" in symbol:
            return symbol.upper()
        # Try XXX/YYY → XXX_YYY
        if "/" in symbol:
            return symbol.upper().replace("/", "_")
        raise ValueError(f"Cannot resolve '{symbol}' to OANDA instrument. "
                         f"Known: {', '.join(sorted(set(self.INSTRUMENT_MAP.values())))}")

    def get_klines(self, symbol: str, interval: str = "1h",
                   limit: int = 1500, start_time: int = None,
                   end_time: int = None) -> list:
        """Fetch historical candles from OANDA.

        Returns data in Binance kline format.
        """
        instrument = self._resolve_instrument(symbol)
        granularity = self.INTERVAL_MAP.get(interval, "H1")

        params = {
            "granularity": granularity,
            "price": "M",  # Mid prices (average of bid/ask)
        }

        if start_time and end_time:
            params["from"] = datetime.fromtimestamp(
                start_time / 1000, tz=timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%SZ")
            params["to"] = datetime.fromtimestamp(
                end_time / 1000, tz=timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%SZ")
        elif start_time:
            params["from"] = datetime.fromtimestamp(
                start_time / 1000, tz=timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%SZ")
            params["count"] = min(limit, 5000)
        else:
            params["count"] = min(limit, 5000)

        qs = urllib.parse.urlencode(params)
        url = f"{self.base_url}/v3/instruments/{instrument}/candles?{qs}"

        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Accept": "application/json",
        }

        all_candles = []
        req = urllib.request.Request(url, headers=headers)

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode() if e.fp else ""
            logger.error(f"[OANDA] {instrument}: HTTP {e.code} — {body}")
            return []
        except Exception as e:
            logger.error(f"[OANDA] Klines failed for {instrument}: {e}")
            return []

        candles = data.get("candles", [])
        for c in candles:
            if not c.get("complete", True):
                continue  # Skip incomplete candles

            mid = c.get("mid", {})
            ts_str = c.get("time", "")
            try:
                dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                open_ms = int(dt.timestamp() * 1000)
            except Exception:
                continue

            all_candles.append(_to_binance_format(
                open_ms,
                float(mid.get("o", 0)),
                float(mid.get("h", 0)),
                float(mid.get("l", 0)),
                float(mid.get("c", 0)),
                float(c.get("volume", 0)),
            ))

        logger.info(f"[OANDA] Fetched {len(all_candles)} candles "
                     f"for {instrument} ({granularity})")
        return all_candles[:limit]

    def format_pair(self, symbol: str) -> str:
        return self._resolve_instrument(symbol)


# =============================================================================
# Upstox V3 Kline Client — Indian Stocks, Commodities, Futures
# =============================================================================

class UpstoxKlineClient:
    """Upstox Historical Candle V3 API — Indian markets.

    V3 Endpoint:
      GET /v3/historical-candle/{instrument_key}/{unit}/{interval}/{to_date}/{from_date}

    V2 Endpoint (fallback):
      GET /v2/historical-candle/{instrument_key}/{interval}/{to_date}/{from_date}

    Intraday (today only):
      GET /v2/historical-candle/intraday/{instrument_key}/{interval}

    V3 Units: minutes, hours, days, weeks, months
    V3 Intervals: any integer (e.g. minutes/5, hours/1, days/1)

    Data availability:
      Minutes/Hours: from Jan 2022
      Days/Weeks/Months: from Jan 2000

    Candle format: [timestamp_iso, open, high, low, close, volume, oi]
    Returns newest first — we reverse for chronological order.

    Covers: NSE, BSE equities + MCX commodities + NSE_FO/MCX_FO futures
    """

    BASE_URL = "https://api.upstox.com"

    def __init__(self, api_key: str, api_secret: str, access_token: str = ""):
        self.api_key = api_key
        self.api_secret = api_secret
        self.access_token = access_token
        self._symbol_cache = {}  # symbol → instrument_key
        self._instruments_loaded = False

    def _headers(self):
        h = {"Accept": "application/json"}
        if self.access_token:
            h["Authorization"] = f"Bearer {self.access_token}"
        return h

    def _resolve_instrument_key(self, symbol: str) -> str:
        """Resolve symbol to Upstox instrument_key.

        Lazy-loads instrument files on first call.
        Delegates to UpstoxClient if available, otherwise uses cache.
        """
        sym = symbol.upper().strip()
        if sym in self._symbol_cache:
            return self._symbol_cache[sym]

        # Try importing UpstoxClient for full resolution
        try:
            from qor.upstox import UpstoxClient
            client = UpstoxClient(self.api_key, self.api_secret,
                                  access_token=self.access_token)
            ikey = client._resolve_instrument_key(sym)
            self._symbol_cache[sym] = ikey
            return ikey
        except Exception:
            pass

        raise ValueError(
            f"Cannot resolve '{symbol}' to Upstox instrument_key. "
            f"Ensure UpstoxClient is available or pre-populate cache.")

    def get_klines(self, symbol: str, interval: str = "1h",
                   limit: int = 1500, start_time: int = None,
                   end_time: int = None) -> list:
        """Fetch historical candles from Upstox V3 API.

        Returns data in Binance kline format.
        """
        ikey = self._resolve_instrument_key(symbol)
        encoded_key = urllib.parse.quote(ikey, safe="")

        # Map interval to Upstox V3 unit/interval
        unit, num = self._map_interval(interval)

        # Date range
        if end_time:
            to_date = datetime.fromtimestamp(
                end_time / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        else:
            to_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        if start_time:
            from_date = datetime.fromtimestamp(
                start_time / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        else:
            # Estimate days back based on interval and limit
            interval_mins = self._interval_minutes(interval)
            days_back = max(int((limit * interval_mins) / 1440) + 5, 30)
            from_date = (datetime.now(timezone.utc) -
                         timedelta(days=days_back)).strftime("%Y-%m-%d")

        # V3 endpoint: /v3/historical-candle/{key}/{unit}/{interval}/{to}/{from}
        endpoint = (f"/v3/historical-candle/{encoded_key}/"
                    f"{unit}/{num}/{to_date}/{from_date}")
        url = f"{self.BASE_URL}{endpoint}"

        try:
            req = urllib.request.Request(url, headers=self._headers())
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode() if e.fp else ""
            logger.warning(f"[Upstox V3] {symbol}: HTTP {e.code} — {body}")
            # Fallback to V2 API
            return self._get_klines_v2(encoded_key, interval, to_date,
                                       from_date, limit)
        except Exception as e:
            logger.error(f"[Upstox] Klines failed for {symbol}: {e}")
            return []

        raw_candles = data.get("data", {}).get("candles", [])
        return self._parse_candles(raw_candles, limit)

    def _get_klines_v2(self, encoded_key: str, interval: str,
                       to_date: str, from_date: str, limit: int) -> list:
        """Fallback to V2 historical candle API."""
        v2_interval = self._map_interval_v2(interval)
        endpoint = (f"/v2/historical-candle/{encoded_key}/"
                    f"{v2_interval}/{to_date}/{from_date}")
        url = f"{self.BASE_URL}{endpoint}"

        try:
            req = urllib.request.Request(url, headers=self._headers())
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
        except Exception as e:
            logger.error(f"[Upstox V2] Fallback failed: {e}")
            return []

        raw_candles = data.get("data", {}).get("candles", [])
        return self._parse_candles(raw_candles, limit)

    def _parse_candles(self, raw_candles: list, limit: int) -> list:
        """Parse Upstox candle format to Binance format.

        Upstox: [timestamp_iso, open, high, low, close, volume, oi]
        Returns newest-first → reversed to chronological order.
        """
        result = []
        for c in reversed(raw_candles):  # Reverse: oldest first
            if len(c) < 6:
                continue
            ts = c[0]
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                open_ms = int(dt.timestamp() * 1000)
            except Exception:
                continue
            result.append(_to_binance_format(
                open_ms,
                float(c[1]),  # open
                float(c[2]),  # high
                float(c[3]),  # low
                float(c[4]),  # close
                float(c[5]),  # volume
            ))
        logger.info(f"[Upstox] Parsed {len(result)} candles")
        return result[-limit:]

    def _map_interval(self, interval: str):
        """Map interval string to Upstox V3 (unit, interval_number)."""
        mapping = {
            "1m": ("minutes", "1"), "3m": ("minutes", "3"),
            "5m": ("minutes", "5"), "15m": ("minutes", "15"),
            "30m": ("minutes", "30"),
            "1h": ("hours", "1"), "2h": ("hours", "2"),
            "4h": ("hours", "4"),
            "1d": ("days", "1"), "1w": ("weeks", "1"),
            "1M": ("months", "1"),
        }
        return mapping.get(interval, ("hours", "1"))

    def _map_interval_v2(self, interval: str) -> str:
        """Map interval to Upstox V2 format."""
        mapping = {
            "1m": "1minute", "3m": "1minute", "5m": "1minute",
            "15m": "30minute", "30m": "30minute",
            "1h": "30minute", "2h": "30minute", "4h": "day",
            "1d": "day", "1w": "week", "1M": "month",
        }
        return mapping.get(interval, "day")

    def _interval_minutes(self, interval: str) -> int:
        mins = {
            "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
            "1h": 60, "2h": 120, "4h": 240,
            "1d": 1440, "1w": 10080, "1M": 43200,
        }
        return mins.get(interval, 60)

    def format_pair(self, symbol: str) -> str:
        return self._resolve_instrument_key(symbol)


# =============================================================================
# Yahoo Finance Client — Universal Fallback (no API key needed)
# =============================================================================

class YahooKlineClient:
    """Yahoo Finance — free historical OHLCV data for any asset.

    Uses the public chart API (no key needed):
      GET https://query1.finance.yahoo.com/v8/finance/chart/{symbol}

    Covers: US/global stocks, ETFs, indices, commodities, forex, crypto.
    Limitations: Rate limited, may block heavy usage. Use as fallback only.

    Symbol format:
      Stocks: AAPL, MSFT, TSLA
      Indices: ^GSPC (S&P 500), ^DJI, ^IXIC
      Commodities: GC=F (gold), SI=F (silver), CL=F (oil)
      Forex: EURUSD=X, GBPUSD=X
      Crypto: BTC-USD, ETH-USD
    """

    BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart"

    # Map common names to Yahoo symbols
    SYMBOL_MAP = {
        "gold": "GC=F", "xau": "GC=F", "xauusd": "GC=F",
        "silver": "SI=F", "xag": "SI=F", "xagusd": "SI=F",
        "platinum": "PL=F", "xpt": "PL=F",
        "oil": "CL=F", "crude": "CL=F", "crude oil": "CL=F",
        "wti": "CL=F", "crudeoil": "CL=F",
        "brent": "BZ=F", "brent crude": "BZ=F",
        "natural gas": "NG=F", "natgas": "NG=F",
        "copper": "HG=F",
        # Forex
        "eur/usd": "EURUSD=X", "eurusd": "EURUSD=X",
        "gbp/usd": "GBPUSD=X", "gbpusd": "GBPUSD=X",
        "usd/jpy": "USDJPY=X", "usdjpy": "USDJPY=X",
        "usd/cad": "USDCAD=X", "usdcad": "USDCAD=X",
        "usd/inr": "USDINR=X", "usdinr": "USDINR=X",
        # Crypto
        "btc": "BTC-USD", "bitcoin": "BTC-USD",
        "eth": "ETH-USD", "ethereum": "ETH-USD",
        "sol": "SOL-USD", "solana": "SOL-USD",
        # Indices
        "sp500": "^GSPC", "spx": "^GSPC",
        "dow": "^DJI", "nasdaq": "^IXIC",
        "nifty": "^NSEI", "nifty 50": "^NSEI",
        "sensex": "^BSESN", "banknifty": "^NSEBANK",
    }

    INTERVAL_MAP = {
        "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
        "1h": "1h", "2h": "1h", "4h": "1h",
        "1d": "1d", "1w": "1wk", "1M": "1mo",
    }

    def __init__(self):
        pass  # No API key needed

    def _resolve_symbol(self, symbol: str) -> str:
        sym = symbol.lower().strip()
        if sym in self.SYMBOL_MAP:
            return self.SYMBOL_MAP[sym]
        # If already looks like a ticker, use as-is
        return symbol.upper()

    def get_klines(self, symbol: str, interval: str = "1h",
                   limit: int = 1500, start_time: int = None,
                   end_time: int = None) -> list:
        """Fetch historical candles from Yahoo Finance."""
        yahoo_sym = self._resolve_symbol(symbol)
        yahoo_interval = self.INTERVAL_MAP.get(interval, "1h")

        params = {"interval": yahoo_interval}

        if start_time and end_time:
            params["period1"] = int(start_time / 1000)
            params["period2"] = int(end_time / 1000)
        elif start_time:
            params["period1"] = int(start_time / 1000)
            params["period2"] = int(time.time())
        else:
            # Default ranges based on interval
            range_map = {
                "1m": "7d", "5m": "60d", "15m": "60d", "30m": "60d",
                "1h": "730d", "1d": "10y", "1wk": "10y", "1mo": "10y",
            }
            params["range"] = range_map.get(yahoo_interval, "730d")

        qs = urllib.parse.urlencode(params)
        url = f"{self.BASE_URL}/{urllib.parse.quote(yahoo_sym)}?{qs}"

        headers = {
            "User-Agent": "Mozilla/5.0 QOR-Trading/1.0",
            "Accept": "application/json",
        }

        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
        except Exception as e:
            logger.error(f"[Yahoo] Klines failed for {yahoo_sym}: {e}")
            return []

        result_data = data.get("chart", {}).get("result", [])
        if not result_data:
            logger.warning(f"[Yahoo] No data for {yahoo_sym}")
            return []

        r = result_data[0]
        timestamps = r.get("timestamp", [])
        quotes = r.get("indicators", {}).get("quote", [{}])[0]

        opens = quotes.get("open", [])
        highs = quotes.get("high", [])
        lows = quotes.get("low", [])
        closes = quotes.get("close", [])
        volumes = quotes.get("volume", [])

        result = []
        for i in range(len(timestamps)):
            if i >= len(opens) or opens[i] is None or closes[i] is None:
                continue
            open_ms = int(timestamps[i]) * 1000
            result.append(_to_binance_format(
                open_ms,
                float(opens[i] or 0),
                float(highs[i] or 0),
                float(lows[i] or 0),
                float(closes[i] or 0),
                float(volumes[i] or 0),
            ))

        logger.info(f"[Yahoo] Fetched {len(result)} candles for {yahoo_sym}")
        return result[-limit:]

    def format_pair(self, symbol: str) -> str:
        return self._resolve_symbol(symbol)


# =============================================================================
# KlineRouter — Smart Routing Based on Asset Classification
# =============================================================================

class KlineRouter:
    """Universal kline fetcher that routes to the correct exchange API.

    Routes based on asset classification from qor.quant.classify_asset():
      crypto      → Binance (if client available) or Yahoo
      stock       → Alpaca (US) or Upstox (Indian) or Yahoo
      commodity   → OANDA or Upstox (MCX) or Yahoo
      forex       → OANDA or Yahoo

    Implements the same get_klines() + format_pair() interface as
    BinanceClient, so CortexTrainer works identically.

    Usage:
        router = KlineRouter(config)
        # Automatically routes to the right API:
        klines = router.get_klines("BTC", interval="1h", limit=1000)
        klines = router.get_klines("AAPL", interval="1h", limit=1000)
        klines = router.get_klines("gold", interval="1h", limit=1000)
    """

    def __init__(self, config=None, binance_client=None,
                 upstox_client=None, alpaca_key: str = "",
                 alpaca_secret: str = "", oanda_token: str = "",
                 oanda_account: str = ""):
        """
        Args:
            config: QORConfig (reads API keys from config.alpaca, config.oanda, etc.)
            binance_client: Pre-built BinanceClient (for crypto)
            upstox_client: Pre-built UpstoxClient or UpstoxKlineClient (for Indian)
            alpaca_key/secret: Alpaca API credentials (for US stocks)
            oanda_token/account: OANDA API credentials (for forex/commodities)
        """
        self._binance = binance_client
        self._upstox = upstox_client
        self._alpaca = None
        self._oanda = None
        self._yahoo = YahooKlineClient()  # Always available (no key)

        # Read keys from config if provided
        if config:
            # Alpaca
            ak = alpaca_key or getattr(getattr(config, 'alpaca', None),
                                       'api_key', '')
            als = alpaca_secret or getattr(getattr(config, 'alpaca', None),
                                           'api_secret', '')
            if ak and als:
                self._alpaca = AlpacaKlineClient(ak, als)

            # OANDA
            ot = oanda_token or getattr(getattr(config, 'oanda', None),
                                        'api_token', '')
            oa = oanda_account or getattr(getattr(config, 'oanda', None),
                                          'account_id', '')
            if ot:
                practice = getattr(getattr(config, 'oanda', None),
                                   'practice', True)
                self._oanda = OandaKlineClient(ot, oa, practice=practice)

            # Upstox
            if not self._upstox:
                uk = getattr(getattr(config, 'upstox', None), 'api_key', '')
                us = getattr(getattr(config, 'upstox', None), 'api_secret', '')
                ut = getattr(getattr(config, 'upstox', None),
                             'access_token', '')
                if uk and ut:
                    self._upstox = UpstoxKlineClient(uk, us, ut)
        else:
            # Direct key construction
            if alpaca_key and alpaca_secret:
                self._alpaca = AlpacaKlineClient(alpaca_key, alpaca_secret)
            if oanda_token:
                self._oanda = OandaKlineClient(oanda_token, oanda_account)

    def _classify(self, symbol: str) -> str:
        """Classify asset type. Returns 'crypto', 'stock', 'commodity', 'forex'."""
        try:
            from qor.quant import classify_asset
            at = classify_asset(symbol)
            return at.asset_type
        except Exception:
            pass

        # Inline fallback classification (no external imports)
        sym = symbol.lower().strip()
        _CRYPTO = {
            "btc", "bitcoin", "eth", "ethereum", "sol", "solana", "xrp",
            "ripple", "ada", "cardano", "doge", "dogecoin", "avax", "dot",
            "matic", "polygon", "link", "chainlink", "bnb", "ltc", "uni",
            "atom", "near", "apt", "sui", "arb", "op", "trx", "shib",
        }
        _COMMODITY = {
            "gold", "xau", "xauusd", "silver", "xag", "xagusd", "platinum",
            "xpt", "oil", "crude", "crude oil", "crudeoil", "wti", "brent",
            "natural gas", "natgas", "copper", "palladium",
            "goldm", "goldguinea", "silverm", "silvermic", "crudeoilm",
        }
        _FOREX = {
            "eur/usd", "eurusd", "gbp/usd", "gbpusd", "usd/jpy", "usdjpy",
            "usd/cad", "usdcad", "aud/usd", "audusd", "nzd/usd", "nzdusd",
            "usd/chf", "usdchf", "eur/gbp", "eurgbp", "eur/jpy", "eurjpy",
            "usd/inr", "usdinr",
        }
        if sym in _CRYPTO:
            return "crypto"
        if sym in _COMMODITY:
            return "commodity"
        if sym in _FOREX or "/" in sym:
            return "forex"
        return "stock"

    def _is_indian_stock(self, symbol: str) -> bool:
        """Check if symbol is an Indian market instrument."""
        sym = symbol.lower().strip()
        _INDIAN = {
                "reliance", "tcs", "infy", "hdfcbank", "icicibank", "sbin",
                "bhartiartl", "hindunilvr", "itc", "kotakbank", "wipro",
                "hcltech", "tatamotors", "maruti", "bajfinance", "bajfinsv",
                "axisbank", "sunpharma", "titan", "ultracemco", "ntpc",
                "powergrid", "ongc", "adanient", "adaniports", "techm",
                "tatasteel", "jswsteel", "indusindbk", "asianpaint",
                "nestleind", "drreddy", "divislab", "cipla", "bpcl",
                "coalindia", "grasim", "heromotoco", "eichermot", "lt",
                "m&m", "britannia", "apollohosp",
                "nifty", "nifty 50", "banknifty", "nifty bank", "sensex",
                "finnifty", "nifty fin service",
                # MCX commodities
                "goldm", "goldguinea", "silverm", "silvermic",
                "crudeoilm", "naturalgas", "zinc", "lead",
                "aluminium", "nickel", "cottoncandy", "menthaoil",
        }
        return sym in _INDIAN

    def get_klines(self, symbol: str, interval: str = "1h",
                   limit: int = 1500, start_time: int = None,
                   end_time: int = None) -> list:
        """Route to the correct API and fetch klines.

        Routing priority:
          1. Crypto → Binance (if available) → Yahoo fallback
          2. Indian stock/commodity → Upstox → Yahoo fallback
          3. US stock → Alpaca → Yahoo fallback
          4. Forex/Metals → OANDA → Yahoo fallback
          5. Everything else → Yahoo
        """
        asset_type = self._classify(symbol)
        is_indian = self._is_indian_stock(symbol)

        # 1. Crypto → Binance
        if asset_type == "crypto":
            if self._binance and hasattr(self._binance, 'get_klines'):
                try:
                    pair = self._binance.format_pair(symbol)
                    return self._binance.get_klines(
                        pair, interval=interval, limit=limit,
                        start_time=start_time, end_time=end_time)
                except Exception as e:
                    logger.warning(f"[Router] Binance failed for {symbol}: {e}")
            # Fallback to Yahoo
            logger.warning(f"[Router] FALLBACK: Using Yahoo for {symbol} "
                           f"(crypto) — data may differ from Binance")
            return self._yahoo.get_klines(
                symbol, interval=interval, limit=limit,
                start_time=start_time, end_time=end_time)

        # 2. Indian stocks/commodities → Upstox
        if is_indian:
            if self._upstox:
                try:
                    return self._upstox.get_klines(
                        symbol, interval=interval, limit=limit,
                        start_time=start_time, end_time=end_time)
                except Exception as e:
                    logger.warning(f"[Router] Upstox failed for {symbol}: {e}")
            # Fallback to Yahoo
            logger.warning(f"[Router] FALLBACK: Using Yahoo for {symbol} "
                           f"(Indian market) — volume/OI data may be missing")
            return self._yahoo.get_klines(
                symbol, interval=interval, limit=limit,
                start_time=start_time, end_time=end_time)

        # 3. Forex + Metals/Energy → OANDA
        if asset_type in ("forex", "commodity"):
            if self._oanda:
                try:
                    return self._oanda.get_klines(
                        symbol, interval=interval, limit=limit,
                        start_time=start_time, end_time=end_time)
                except Exception as e:
                    logger.warning(f"[Router] OANDA failed for {symbol}: {e}")
            # Fallback to Yahoo
            logger.warning(f"[Router] FALLBACK: Using Yahoo for {symbol} "
                           f"({asset_type}) — volume=0 for forex, "
                           f"feature 10 (rel_vol) will be meaningless")
            return self._yahoo.get_klines(
                symbol, interval=interval, limit=limit,
                start_time=start_time, end_time=end_time)

        # 4. US/Global stocks → Alpaca
        if asset_type == "stock":
            if self._alpaca:
                try:
                    return self._alpaca.get_klines(
                        symbol, interval=interval, limit=limit,
                        start_time=start_time, end_time=end_time)
                except Exception as e:
                    logger.warning(f"[Router] Alpaca failed for {symbol}: {e}")
            # Fallback to Yahoo
            logger.warning(f"[Router] FALLBACK: Using Yahoo for {symbol} "
                           f"(stock) — prices are split-adjusted, "
                           f"intraday data may have gaps")
            return self._yahoo.get_klines(
                symbol, interval=interval, limit=limit,
                start_time=start_time, end_time=end_time)

        # 5. Fallback: Yahoo
        return self._yahoo.get_klines(
            symbol, interval=interval, limit=limit,
            start_time=start_time, end_time=end_time)

    def format_pair(self, symbol: str) -> str:
        """Format symbol for the appropriate exchange."""
        asset_type = self._classify(symbol)
        if asset_type == "crypto" and self._binance:
            return self._binance.format_pair(symbol)
        return symbol.upper()

    def status(self) -> dict:
        """Return which clients are available."""
        return {
            "binance": self._binance is not None,
            "upstox": self._upstox is not None,
            "alpaca": self._alpaca is not None,
            "oanda": self._oanda is not None,
            "yahoo": True,  # Always available
        }

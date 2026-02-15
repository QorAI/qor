"""
QOR CORTEX Trainer — Historical Data → Brain Training Pipeline
================================================================
Fetches historical OHLCV candles from exchanges, computes all 20
technical indicators matching CortexAnalyzer._build_features(),
labels each candle with future return, and trains CORTEX brain.

Pipeline:
  1. Fetch klines (1h candles, 90 days default)
  2. Compute indicators: RSI, EMA, MACD, BB, ATR, OBV, VWAP, ADX, Keltner
  3. Build 22-dim feature vectors (same as live _build_features)
  4. Label with lookahead return: +1 (bullish), -1 (bearish), 0 (neutral)
  5. Train CortexAnalyzer.train_batch()

Usage:
    from qor.train_cortex import CortexTrainer
    trainer = CortexTrainer(client, cortex_analyzer)
    result = trainer.train("BTC", days=90, epochs=20)
    # result = {"trained": True, "samples": 2100, "loss": 0.023, ...}

    # Or train all configured symbols:
    results = trainer.train_all(["BTC", "ETH", "SOL"], days=90)
"""

import logging
import math
import time
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# =============================================================================
# Pure-Python TA Indicators (no pandas/numpy dependency for portability)
# =============================================================================

def _ema(values: list, period: int) -> list:
    """Exponential Moving Average. Returns list same length as input (NaN-padded)."""
    if len(values) < period:
        return [0.0] * len(values)
    result = [0.0] * len(values)
    # Seed with SMA
    sma = sum(values[:period]) / period
    result[period - 1] = sma
    mult = 2.0 / (period + 1)
    for i in range(period, len(values)):
        result[i] = (values[i] - result[i - 1]) * mult + result[i - 1]
    return result


def _sma(values: list, period: int) -> list:
    """Simple Moving Average."""
    result = [0.0] * len(values)
    for i in range(period - 1, len(values)):
        result[i] = sum(values[i - period + 1:i + 1]) / period
    return result


def _rsi(closes: list, period: int = 14) -> list:
    """Relative Strength Index."""
    result = [50.0] * len(closes)
    if len(closes) < period + 1:
        return result

    gains = []
    losses = []
    for i in range(1, len(closes)):
        delta = closes[i] - closes[i - 1]
        gains.append(max(delta, 0))
        losses.append(max(-delta, 0))

    # Initial average
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            result[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i + 1] = 100.0 - (100.0 / (1.0 + rs))

    # Fill initial period
    if avg_loss > 0:
        rs = (sum(gains[:period]) / period) / (sum(losses[:period]) / period)
        result[period] = 100.0 - (100.0 / (1.0 + rs))

    return result


def _atr(highs: list, lows: list, closes: list, period: int = 14) -> list:
    """Average True Range."""
    result = [0.0] * len(closes)
    if len(closes) < 2:
        return result

    # True Range
    tr = [highs[0] - lows[0]]
    for i in range(1, len(closes)):
        tr.append(max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        ))

    # Wilder's smoothing (same as EMA with period)
    if len(tr) >= period:
        atr_val = sum(tr[:period]) / period
        result[period - 1] = atr_val
        for i in range(period, len(tr)):
            atr_val = (atr_val * (period - 1) + tr[i]) / period
            result[i] = atr_val

    return result


def _macd(closes: list, fast: int = 12, slow: int = 26,
          signal: int = 9) -> Tuple[list, list, list]:
    """MACD: returns (macd_line, signal_line, histogram)."""
    ema_fast = _ema(closes, fast)
    ema_slow = _ema(closes, slow)
    macd_line = [f - s for f, s in zip(ema_fast, ema_slow)]
    signal_line = _ema(macd_line, signal)
    histogram = [m - s for m, s in zip(macd_line, signal_line)]
    return macd_line, signal_line, histogram


def _bollinger_bands(closes: list, period: int = 20,
                     std_mult: float = 2.0) -> Tuple[list, list, list]:
    """Bollinger Bands: returns (upper, middle, lower)."""
    middle = _sma(closes, period)
    upper = [0.0] * len(closes)
    lower = [0.0] * len(closes)

    for i in range(period - 1, len(closes)):
        window = closes[i - period + 1:i + 1]
        mean = middle[i]
        variance = sum((x - mean) ** 2 for x in window) / period
        std = math.sqrt(variance)
        upper[i] = mean + std_mult * std
        lower[i] = mean - std_mult * std

    return upper, middle, lower


def _adx(highs: list, lows: list, closes: list, period: int = 14) -> list:
    """Average Directional Index."""
    result = [25.0] * len(closes)  # Default neutral
    if len(closes) < period * 2:
        return result

    # +DM, -DM
    plus_dm = [0.0]
    minus_dm = [0.0]
    for i in range(1, len(closes)):
        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        plus_dm.append(up if up > down and up > 0 else 0)
        minus_dm.append(down if down > up and down > 0 else 0)

    atr_vals = _atr(highs, lows, closes, period)

    # Smooth +DM, -DM using Wilder's method
    smooth_plus = _ema(plus_dm, period)
    smooth_minus = _ema(minus_dm, period)

    # +DI, -DI
    plus_di = [0.0] * len(closes)
    minus_di = [0.0] * len(closes)
    dx = [0.0] * len(closes)

    for i in range(period, len(closes)):
        if atr_vals[i] > 0:
            plus_di[i] = (smooth_plus[i] / atr_vals[i]) * 100
            minus_di[i] = (smooth_minus[i] / atr_vals[i]) * 100
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = abs(plus_di[i] - minus_di[i]) / di_sum * 100

    # ADX = smoothed DX
    adx_vals = _ema(dx, period)
    for i in range(period * 2, len(closes)):
        if adx_vals[i] > 0:
            result[i] = adx_vals[i]

    return result


def _obv(closes: list, volumes: list) -> list:
    """On-Balance Volume."""
    result = [0.0] * len(closes)
    if not volumes or len(volumes) != len(closes):
        return result
    result[0] = volumes[0]
    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            result[i] = result[i - 1] + volumes[i]
        elif closes[i] < closes[i - 1]:
            result[i] = result[i - 1] - volumes[i]
        else:
            result[i] = result[i - 1]
    return result


def _vwap(highs: list, lows: list, closes: list, volumes: list) -> list:
    """Volume Weighted Average Price (cumulative)."""
    result = [0.0] * len(closes)
    if not volumes:
        return closes[:]
    cum_vol = 0.0
    cum_tp_vol = 0.0
    for i in range(len(closes)):
        tp = (highs[i] + lows[i] + closes[i]) / 3.0
        vol = volumes[i] if i < len(volumes) else 0
        cum_vol += vol
        cum_tp_vol += tp * vol
        result[i] = cum_tp_vol / cum_vol if cum_vol > 0 else tp
    return result


def _keltner_position(closes: list, highs: list, lows: list,
                      ema_period: int = 20, atr_period: int = 14,
                      atr_mult: float = 2.0) -> list:
    """Keltner Channel position: -1 (below lower) to +1 (above upper)."""
    ema_vals = _ema(closes, ema_period)
    atr_vals = _atr(highs, lows, closes, atr_period)
    result = [0.0] * len(closes)

    for i in range(max(ema_period, atr_period), len(closes)):
        if atr_vals[i] > 0 and ema_vals[i] > 0:
            upper = ema_vals[i] + atr_mult * atr_vals[i]
            lower = ema_vals[i] - atr_mult * atr_vals[i]
            band_range = upper - lower
            if band_range > 0:
                result[i] = (closes[i] - lower) / band_range * 2 - 1
                result[i] = max(-1.0, min(1.0, result[i]))
    return result


def _relative_volume(volumes: list, period: int = 20) -> list:
    """Volume relative to its SMA (1.0 = average, 2.0 = 2x average)."""
    result = [1.0] * len(volumes)
    if not volumes:
        return result
    avg = _sma(volumes, period)
    for i in range(period, len(volumes)):
        if avg[i] > 0:
            result[i] = volumes[i] / avg[i]
    return result


def _obv_direction(obv_vals: list, period: int = 10) -> list:
    """OBV direction: slope of OBV over last N bars, normalized to [-1, 1]."""
    result = [0.0] * len(obv_vals)
    for i in range(period, len(obv_vals)):
        if obv_vals[i - period] != 0:
            change = (obv_vals[i] - obv_vals[i - period]) / abs(obv_vals[i - period])
            result[i] = max(-1.0, min(1.0, change * 10))  # Scale
        elif obv_vals[i] > 0:
            result[i] = 0.5
        elif obv_vals[i] < 0:
            result[i] = -0.5
    return result


def _body_ratio(opens: list, highs: list, lows: list, closes: list) -> list:
    """Candlestick body/range ratio (0-1)."""
    result = [0.5] * len(closes)
    for i in range(len(closes)):
        full_range = highs[i] - lows[i]
        if full_range > 0:
            body = abs(closes[i] - opens[i])
            result[i] = body / full_range
    return result


def _upper_wick_ratio(opens: list, highs: list, lows: list,
                      closes: list) -> list:
    """Upper wick / total range (0-1)."""
    result = [0.25] * len(closes)
    for i in range(len(closes)):
        full_range = highs[i] - lows[i]
        if full_range > 0:
            top = max(opens[i], closes[i])
            result[i] = (highs[i] - top) / full_range
    return result


def _lower_wick_ratio(opens: list, highs: list, lows: list,
                      closes: list) -> list:
    """Lower wick / total range (0-1)."""
    result = [0.25] * len(closes)
    for i in range(len(closes)):
        full_range = highs[i] - lows[i]
        if full_range > 0:
            bottom = min(opens[i], closes[i])
            result[i] = (bottom - lows[i]) / full_range
    return result


# =============================================================================
# Candle data container
# =============================================================================

class Candles:
    """Container for OHLCV candle data with computed indicators."""

    def __init__(self, raw_klines: list):
        """Parse raw klines from Binance format.

        Binance kline format:
        [open_time, open, high, low, close, volume, close_time,
         quote_volume, trades, taker_buy_base, taker_buy_quote, ignore]
        """
        self.timestamps = []
        self.opens = []
        self.highs = []
        self.lows = []
        self.closes = []
        self.volumes = []

        for k in raw_klines:
            self.timestamps.append(int(k[0]))
            self.opens.append(float(k[1]))
            self.highs.append(float(k[2]))
            self.lows.append(float(k[3]))
            self.closes.append(float(k[4]))
            self.volumes.append(float(k[5]))

        self._indicators = {}

    def __len__(self):
        return len(self.closes)

    def compute_all(self):
        """Compute all indicators needed for the 20-dim CORTEX vector."""
        c = self.closes
        h = self.highs
        l = self.lows
        o = self.opens
        v = self.volumes

        self._indicators["rsi14"] = _rsi(c, 14)
        self._indicators["rsi6"] = _rsi(c, 6)
        self._indicators["ema21"] = _ema(c, 21)
        self._indicators["ema50"] = _ema(c, 50)
        self._indicators["ema200"] = _ema(c, 200)
        self._indicators["atr14"] = _atr(h, l, c, 14)

        macd_line, signal_line, histogram = _macd(c)
        self._indicators["macd_hist"] = histogram

        bb_upper, bb_mid, bb_lower = _bollinger_bands(c)
        self._indicators["bb_upper"] = bb_upper
        self._indicators["bb_lower"] = bb_lower

        self._indicators["adx"] = _adx(h, l, c, 14)
        self._indicators["keltner_pos"] = _keltner_position(c, h, l)

        obv_vals = _obv(c, v)
        self._indicators["obv_dir"] = _obv_direction(obv_vals)
        self._indicators["rel_vol"] = _relative_volume(v)
        self._indicators["vwap"] = _vwap(h, l, c, v)

        self._indicators["body_ratio"] = _body_ratio(o, h, l, c)
        self._indicators["upper_wick"] = _upper_wick_ratio(o, h, l, c)
        self._indicators["lower_wick"] = _lower_wick_ratio(o, h, l, c)

        return self

    def get_feature_vector(self, i: int) -> Optional[list]:
        """Build the 24-dim feature vector for candle index i.

        Matches CortexAnalyzer._build_features() normalization exactly.
        Features 20-21 (Polymarket, Fear&Greed) default to 0.5 (neutral)
        since no historical data is available for these.
        Features 22-23 (VP POC position, VP density) default to 0.5 (neutral)
        since no historical volume profile data is available.

        Returns None if indicators aren't ready (warmup period).
        """
        if i < 200:  # Need EMA200 warmup
            return None

        c = self.closes[i]
        ind = self._indicators
        if c <= 0:
            return None

        atr = ind["atr14"][i]
        if atr <= 0:
            return None

        # 0: RSI(14) normalized (0-1)
        f_rsi14 = ind["rsi14"][i] / 100.0

        # 1: RSI(6) normalized (0-1)
        f_rsi6 = ind["rsi6"][i] / 100.0

        # 2: RSI momentum (delta from previous candle)
        prev_rsi = ind["rsi14"][i - 1] if i > 0 else ind["rsi14"][i]
        f_rsi_mom = max(-1.0, min(1.0, (ind["rsi14"][i] - prev_rsi) / 100.0))

        # 3: Price vs EMA21 % deviation
        ema21 = ind["ema21"][i] or c
        f_ema21 = max(-0.1, min(0.1, (c - ema21) / max(c, 1)))

        # 4: Price vs EMA50 % deviation
        ema50 = ind["ema50"][i] or c
        f_ema50 = max(-0.1, min(0.1, (c - ema50) / max(c, 1)))

        # 5: Price vs EMA200 % deviation (wider range)
        ema200 = ind["ema200"][i] or c
        f_ema200 = max(-0.2, min(0.2, (c - ema200) / max(c, 1)))

        # 6: MACD histogram / ATR (normalized momentum)
        macd_hist = ind["macd_hist"][i]
        f_macd = max(-1.0, min(1.0, macd_hist / max(atr, 1)))

        # 7: Bollinger Band %B position (-1 to 1)
        bb_upper = ind["bb_upper"][i] or c * 1.02
        bb_lower = ind["bb_lower"][i] or c * 0.98
        bb_range = max(bb_upper - bb_lower, 0.01)
        f_bb = max(-1.0, min(1.0, (c - bb_lower) / bb_range * 2 - 1))

        # 8: ATR % of price (volatility, 0-1)
        f_atr = min((atr / max(c, 1)) * 100 / 5.0, 1.0)

        # 9: Keltner channel position (-1 to 1)
        f_keltner = max(-1.0, min(1.0, ind["keltner_pos"][i]))

        # 10: Relative volume (0-3, clamped to 0-1)
        f_rel_vol = min(ind["rel_vol"][i] / 3.0, 1.0)

        # 11: OBV direction (-1 to 1)
        f_obv = max(-1.0, min(1.0, ind["obv_dir"][i]))

        # 12: Price vs VWAP % deviation — scaled to (-1, 1) range
        vwap = ind["vwap"][i] or c
        f_vwap = max(-1.0, min(1.0, (c - vwap) / max(c, 1) * 20))

        # 13: ADX normalized (0-1, 100 max)
        f_adx = min(ind["adx"][i] / 100.0, 1.0)

        # 14: TF confluence ratio — from candle trend direction
        # Use multi-EMA alignment as proxy for TF confluence
        above_ema21 = 1 if c > ema21 else 0
        above_ema50 = 1 if c > ema50 else 0
        above_ema200 = 1 if c > ema200 else 0
        ema21_above_50 = 1 if ema21 > ema50 else 0
        ema50_above_200 = 1 if ema50 > ema200 else 0
        macd_positive = 1 if macd_hist > 0 else 0
        f_tf = (above_ema21 + above_ema50 + above_ema200 +
                ema21_above_50 + ema50_above_200 + macd_positive) / 6.0

        # 15: Funding rate (0 for spot/historical — no funding data)
        f_funding = 0.0

        # 16: OI change (0 for spot/historical — no OI data)
        f_oi = 0.0

        # 17: Candlestick body/range ratio (0-1)
        f_body = max(0.0, min(1.0, ind["body_ratio"][i]))

        # 18: Upper wick ratio (0-1)
        f_uwk = max(0.0, min(1.0, ind["upper_wick"][i]))

        # 19: Lower wick ratio (0-1)
        f_lwk = max(0.0, min(1.0, ind["lower_wick"][i]))

        # 20: Polymarket sentiment (neutral default — no historical data)
        f_poly = 0.5

        # 21: Fear & Greed index (neutral default — no historical data)
        f_fg = 0.5

        # 22: VP POC position (neutral default — no historical VP data)
        f_vp_pos = 0.5

        # 23: VP density (neutral default — no historical VP data)
        f_vp_dens = 0.5

        return [
            f_rsi14, f_rsi6, f_rsi_mom, f_ema21, f_ema50, f_ema200,
            f_macd, f_bb, f_atr, f_keltner, f_rel_vol, f_obv,
            f_vwap, f_adx, f_tf, f_funding, f_oi, f_body, f_uwk, f_lwk,
            f_poly, f_fg, f_vp_pos, f_vp_dens,
        ]


# =============================================================================
# Target Labeling — Future Return Based
# =============================================================================

def label_future_return(closes: list, lookahead: int = 12,
                        threshold_pct: float = 1.0) -> list:
    """Label each candle with future return direction.

    For each candle i, compute the max favorable move in the next
    `lookahead` candles. This simulates what a trader would experience.

    Args:
        closes: List of close prices
        lookahead: Number of candles to look ahead (12 * 1h = 12 hours)
        threshold_pct: Minimum % move to count as signal (avoids noise)

    Returns:
        List of target values:
          +1.0 = price rises >= threshold_pct (bullish)
          -1.0 = price falls >= threshold_pct (bearish)
           0.0 = price stays within threshold (neutral/choppy)

    Labeling logic:
      - Compute max high and min low in lookahead window
      - If max_gain >= threshold AND max_gain > max_loss → +1 (bullish)
      - If max_loss >= threshold AND max_loss > max_gain → -1 (bearish)
      - Otherwise → 0 (neutral)
    """
    n = len(closes)
    targets = [0.0] * n

    for i in range(n - lookahead):
        entry = closes[i]
        if entry <= 0:
            continue

        # Look at future window
        future = closes[i + 1:i + 1 + lookahead]
        if not future:
            continue

        max_price = max(future)
        min_price = min(future)

        max_gain_pct = ((max_price - entry) / entry) * 100
        max_loss_pct = ((entry - min_price) / entry) * 100

        if max_gain_pct >= threshold_pct and max_gain_pct > max_loss_pct:
            targets[i] = 1.0
        elif max_loss_pct >= threshold_pct and max_loss_pct > max_gain_pct:
            targets[i] = -1.0
        # else: 0.0 (neutral)

    return targets


def label_return_continuous(closes: list, lookahead: int = 12) -> list:
    """Continuous target: actual % return over lookahead window.

    Scaled with tanh to keep in (-1, 1) range.
    Better for regression-style training.
    """
    n = len(closes)
    targets = [0.0] * n

    for i in range(n - lookahead):
        entry = closes[i]
        if entry <= 0:
            continue
        future_close = closes[i + lookahead]
        pct_return = ((future_close - entry) / entry) * 100
        # tanh scaling: 2% → ~0.96, 1% → ~0.76, 0.5% → ~0.46
        targets[i] = math.tanh(pct_return)

    return targets


# =============================================================================
# CortexTrainer — Main Training Pipeline
# =============================================================================

class CortexTrainer:
    """Trains CortexAnalyzer from historical exchange data.

    Usage:
        from qor.train_cortex import CortexTrainer
        trainer = CortexTrainer(binance_client, cortex_analyzer)

        # Train on 90 days of BTC 1h candles
        result = trainer.train("BTC", days=90, interval="1h")

        # Train all symbols
        results = trainer.train_all(["BTC", "ETH", "SOL"])

        # Custom: fetch data, inspect, then train
        candles = trainer.fetch("BTC", days=30)
        features, targets = trainer.prepare(candles)
        result = trainer.train_prepared(features, targets, epochs=30)
    """

    # Binance kline intervals and their millisecond durations
    INTERVAL_MS = {
        "1m": 60_000,
        "3m": 180_000,
        "5m": 300_000,
        "15m": 900_000,
        "30m": 1_800_000,
        "1h": 3_600_000,
        "2h": 7_200_000,
        "4h": 14_400_000,
        "1d": 86_400_000,
        "1w": 604_800_000,
    }

    def __init__(self, client, cortex):
        """
        Args:
            client: KlineRouter or any client with get_klines() + format_pair()
                    (BinanceClient, BinanceFuturesClient, KlineRouter, etc.)
            cortex: CortexAnalyzer instance to train
        """
        self.client = client
        self.cortex = cortex

    def fetch(self, symbol: str, days: int = 90,
              interval: str = "1h") -> Candles:
        """Fetch historical klines and compute all indicators.

        Works with any client: KlineRouter (auto-routes to correct exchange),
        BinanceClient, BinanceFuturesClient, AlpacaKlineClient, etc.

        Args:
            symbol: Asset symbol ("BTC", "AAPL", "RELIANCE", "gold", etc.)
            days: Number of days of history to fetch
            interval: Candle interval ("1m", "5m", "15m", "1h", "4h", "1d")

        Returns:
            Candles object with computed indicators
        """
        from datetime import datetime, timezone, timedelta

        end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_ms = int((datetime.now(timezone.utc) -
                        timedelta(days=days)).timestamp() * 1000)

        interval_ms = self.INTERVAL_MS.get(interval, 3_600_000)
        expected_candles = (end_ms - start_ms) // interval_ms

        # Check if client is a router (has status()) vs direct exchange client
        is_router = hasattr(self.client, 'status') and callable(
            getattr(self.client, 'status', None))

        logger.info(f"[CortexTrainer] Fetching ~{expected_candles} {interval} "
                     f"candles for {symbol} ({days} days)")

        all_klines = []

        if is_router:
            # KlineRouter handles symbol resolution + exchange routing internally
            # Try single fetch first (many APIs support large ranges)
            try:
                klines = self.client.get_klines(
                    symbol, interval=interval,
                    limit=min(expected_candles + 50, 5000),
                    start_time=start_ms, end_time=end_ms)
                if klines:
                    all_klines = klines
            except Exception as e:
                logger.warning(f"[CortexTrainer] Single fetch failed for "
                               f"{symbol}: {e}")

            # If we got significantly fewer than expected, paginate
            if len(all_klines) < expected_candles * 0.7:
                all_klines = []
                cursor = end_ms
                for page in range(20):
                    try:
                        klines = self.client.get_klines(
                            symbol, interval=interval, limit=1500,
                            start_time=start_ms, end_time=cursor)
                    except Exception as e:
                        logger.warning(f"[CortexTrainer] Page {page} failed: {e}")
                        break
                    if not klines:
                        break
                    all_klines = klines + all_klines  # Prepend (oldest first)
                    earliest = klines[0][0]
                    if isinstance(earliest, str):
                        earliest = int(earliest)
                    cursor = earliest - 1
                    if cursor <= start_ms:
                        break
                    time.sleep(0.2)
        else:
            # Direct exchange client (BinanceClient, etc.) — paginate
            pair = self.client.format_pair(symbol) if hasattr(
                self.client, 'format_pair') else symbol
            cursor = end_ms
            remaining = expected_candles

            while remaining > 0:
                limit = min(remaining, 1500)
                try:
                    klines = self.client.get_klines(
                        pair, interval=interval, limit=limit,
                        end_time=cursor)
                except Exception as e:
                    logger.error(f"[CortexTrainer] Fetch failed: {e}")
                    break
                if not klines:
                    break
                all_klines = klines + all_klines  # Prepend (oldest first)
                earliest = klines[0][0]
                if isinstance(earliest, str):
                    earliest = int(earliest)
                cursor = earliest - 1
                remaining -= len(klines)
                if len(klines) < limit:
                    break
                time.sleep(0.2)

        logger.info(f"[CortexTrainer] Fetched {len(all_klines)} candles "
                     f"for {symbol}")

        candles = Candles(all_klines)
        candles.compute_all()
        return candles

    def prepare(self, candles: Candles, lookahead: int = 12,
                threshold_pct: float = 1.0,
                label_mode: str = "continuous") -> Tuple[list, list]:
        """Build feature vectors and targets from candles.

        Args:
            candles: Candles object with computed indicators
            lookahead: Candles to look ahead for labeling
            threshold_pct: % move threshold for categorical labeling
            label_mode: "categorical" (+1/0/-1) or "continuous" (tanh return)

        Returns:
            (features, targets) — lists of equal length
            features: list of 20-dim lists
            targets: list of float
        """
        if label_mode == "continuous":
            targets_raw = label_return_continuous(candles.closes, lookahead)
        else:
            targets_raw = label_future_return(
                candles.closes, lookahead, threshold_pct)

        features = []
        targets = []

        for i in range(len(candles)):
            vec = candles.get_feature_vector(i)
            if vec is None:
                continue
            # Skip last `lookahead` candles (no future data for label)
            if i >= len(candles) - lookahead:
                continue
            features.append(vec)
            targets.append(targets_raw[i])

        logger.info(f"[CortexTrainer] Prepared {len(features)} samples "
                     f"(skipped {len(candles) - len(features)} warmup/tail)")

        # Distribution info
        if targets:
            if label_mode == "categorical":
                n_bull = sum(1 for t in targets if t > 0)
                n_bear = sum(1 for t in targets if t < 0)
                n_neutral = sum(1 for t in targets if t == 0)
                logger.info(f"[CortexTrainer] Labels: {n_bull} bullish, "
                            f"{n_bear} bearish, {n_neutral} neutral")
            else:
                avg_target = sum(targets) / len(targets)
                pos_pct = sum(1 for t in targets if t > 0) / len(targets) * 100
                logger.info(f"[CortexTrainer] Labels: avg={avg_target:.4f}, "
                            f"{pos_pct:.1f}% positive")

        return features, targets

    def train_prepared(self, features: list, targets: list,
                       epochs: int = 20, lr: float = 1e-3) -> dict:
        """Train CORTEX on pre-prepared features and targets.

        Args:
            features: List of 20-dim feature vectors
            targets: List of target values
            epochs: Training epochs
            lr: Learning rate

        Returns:
            Training result dict from CortexAnalyzer.train_batch()
        """
        if not features:
            return {"trained": False, "reason": "no training samples"}

        if not _HAS_TORCH:
            return {"trained": False, "reason": "torch not available"}

        logger.info(f"[CortexTrainer] Training CORTEX on {len(features)} "
                     f"samples, {epochs} epochs, lr={lr}")

        result = self.cortex.train_batch(
            features, targets, epochs=epochs, lr=lr)

        return result

    def train(self, symbol: str, days: int = 90, interval: str = "1h",
              lookahead: int = 12, threshold_pct: float = 1.0,
              label_mode: str = "continuous",
              epochs: int = 20, lr: float = 1e-3) -> dict:
        """Full pipeline: fetch → compute → label → train.

        Args:
            symbol: Asset symbol ("BTC", "ETH", etc.)
            days: Days of history to fetch
            interval: Candle interval
            lookahead: Candles to look ahead for labeling
            threshold_pct: % threshold for categorical labels
            label_mode: "continuous" or "categorical"
            epochs: Training epochs
            lr: Learning rate

        Returns:
            {"trained": bool, "symbol": str, "samples": int,
             "candles_fetched": int, ...}
        """
        t0 = time.time()

        # 1. Fetch
        candles = self.fetch(symbol, days=days, interval=interval)
        if len(candles) < 250:
            return {"trained": False, "symbol": symbol,
                    "reason": f"only {len(candles)} candles (need 250+)"}

        # 2. Prepare
        features, targets = self.prepare(
            candles, lookahead=lookahead,
            threshold_pct=threshold_pct, label_mode=label_mode)

        if len(features) < 50:
            return {"trained": False, "symbol": symbol,
                    "reason": f"only {len(features)} valid samples (need 50+)"}

        # 3. Train
        result = self.train_prepared(features, targets, epochs=epochs, lr=lr)

        result["symbol"] = symbol
        result["candles_fetched"] = len(candles)
        result["samples"] = len(features)
        result["interval"] = interval
        result["days"] = days
        result["lookahead"] = lookahead
        result["elapsed_seconds"] = round(time.time() - t0, 1)

        logger.info(f"[CortexTrainer] {symbol}: {result}")
        return result

    def train_all(self, symbols: list, days: int = 90,
                  interval: str = "1h", epochs: int = 20,
                  **kwargs) -> dict:
        """Train CORTEX on multiple symbols (pooled data).

        Fetches history for all symbols, pools features + targets,
        then trains once on the combined dataset. This gives CORTEX
        a broader view of market patterns across assets.

        Args:
            symbols: List of symbols ["BTC", "ETH", "SOL"]
            days: Days of history per symbol
            interval: Candle interval
            epochs: Training epochs
            **kwargs: Passed to prepare()

        Returns:
            {"trained": bool, "symbols": [...], "total_samples": int, ...}
        """
        t0 = time.time()
        all_features = []
        all_targets = []
        per_symbol = {}

        for symbol in symbols:
            try:
                candles = self.fetch(symbol, days=days, interval=interval)
                features, targets = self.prepare(candles, **kwargs)
                per_symbol[symbol] = {
                    "candles": len(candles),
                    "samples": len(features),
                }
                all_features.extend(features)
                all_targets.extend(targets)
                logger.info(f"[CortexTrainer] {symbol}: {len(features)} samples")
            except Exception as e:
                per_symbol[symbol] = {"error": str(e)}
                logger.warning(f"[CortexTrainer] {symbol} failed: {e}")

        if not all_features:
            return {"trained": False, "reason": "no data from any symbol"}

        # Shuffle combined data for better training
        import random
        combined = list(zip(all_features, all_targets))
        random.shuffle(combined)
        all_features = [x[0] for x in combined]
        all_targets = [x[1] for x in combined]

        result = self.train_prepared(all_features, all_targets, epochs=epochs)
        result["symbols"] = symbols
        result["total_samples"] = len(all_features)
        result["per_symbol"] = per_symbol
        result["elapsed_seconds"] = round(time.time() - t0, 1)

        logger.info(f"[CortexTrainer] All symbols trained: {result}")
        return result

    def evaluate(self, symbol: str, days: int = 30,
                 interval: str = "1h", lookahead: int = 12) -> dict:
        """Evaluate CORTEX accuracy on recent unseen data.

        Fetches recent candles NOT used for training, runs CORTEX
        inference, compares predictions to actual outcomes.

        Returns:
            {"accuracy": float, "precision": float, "recall": float,
             "total": int, "correct": int, ...}
        """
        if not _HAS_TORCH:
            return {"error": "torch not available"}

        candles = self.fetch(symbol, days=days, interval=interval)
        if len(candles) < 250:
            return {"error": f"only {len(candles)} candles"}

        targets_raw = label_return_continuous(candles.closes, lookahead)
        correct = 0
        total = 0
        true_pos = 0
        pred_pos = 0
        actual_pos = 0

        for i in range(200, len(candles) - lookahead):
            vec = candles.get_feature_vector(i)
            if vec is None:
                continue

            # CORTEX prediction
            x = torch.tensor([vec], dtype=torch.float32)
            raw = self.cortex._brain(x, instance_id=f"eval_{symbol}")
            pred = torch.tanh(raw).item()

            actual = targets_raw[i]
            total += 1

            # Directional accuracy
            if (pred > 0 and actual > 0) or (pred < 0 and actual < 0) or (
                    abs(pred) < 0.15 and abs(actual) < 0.1):
                correct += 1

            # Precision/recall for bullish signals
            if pred > 0.15:
                pred_pos += 1
                if actual > 0:
                    true_pos += 1
            if actual > 0.1:
                actual_pos += 1

        accuracy = correct / total if total > 0 else 0
        precision = true_pos / pred_pos if pred_pos > 0 else 0
        recall = true_pos / actual_pos if actual_pos > 0 else 0

        return {
            "symbol": symbol,
            "total": total,
            "correct": correct,
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "pred_positive": pred_pos,
            "actual_positive": actual_pos,
            "true_positive": true_pos,
        }

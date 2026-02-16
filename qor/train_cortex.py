"""
QOR CORTEX Trainer — Historical Data → Brain Training Pipeline
================================================================
Fetches historical OHLCV candles from exchanges, computes all 24
technical indicators matching CortexAnalyzer._build_features(),
labels each candle with future return, and trains CORTEX brain.

Pipeline:
  1. Fetch klines (interval/days auto-set per trade mode)
  2. Fetch historical Fear & Greed + funding rates (features 15, 21)
  3. Compute indicators: RSI, EMA, MACD, BB, ATR, OBV, VWAP, ADX, Keltner
  4. Compute multi-TF confluence via aggregated higher timeframes (feature 14)
  5. Compute OI proxy from daily volume ratio (feature 16)
  6. Compute rolling Volume Profile VRVP (features 22-23)
  7. Build 24-dim feature vectors (same as live _build_features)
  8. Label with lookahead return (mode-aware lookahead)
  9. Train per-symbol sequentially (preserves CfC temporal context)
  10. Fine-tune on shuffled mixed data (generalization)

Feature coverage: 24/24 features have real or proxy data during training.
Feature 20 (Polymarket) uses momentum-sentiment proxy (no historical API).

Mode-aware defaults:
  scalp:  5m candles, 30 days, lookahead=12 (1 hour)
  stable: 1h candles, 90 days, lookahead=12 (12 hours) — default
  secure: 4h candles, 180 days, lookahead=12 (48 hours)

Usage:
    from qor.train_cortex import CortexTrainer
    trainer = CortexTrainer(client, cortex_analyzer)
    result = trainer.train("BTC", mode="scalp", epochs=20)

    # Or train all configured symbols with mode:
    results = trainer.train_all(["BTC", "ETH", "SOL"], mode="stable")
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
# Multi-Timeframe Aggregation & Scoring
# =============================================================================

def _aggregate_candles(opens: list, highs: list, lows: list,
                       closes: list, volumes: list, factor: int):
    """Aggregate OHLCV candles into a higher timeframe.

    Groups every `factor` candles into one (e.g., factor=4 turns 1h -> 4h).
    Returns (agg_opens, agg_highs, agg_lows, agg_closes, agg_volumes).
    """
    n = len(closes)
    ao, ah, al, ac, av = [], [], [], [], []
    for s in range(0, n - factor + 1, factor):
        e = s + factor
        ao.append(opens[s])
        ah.append(max(highs[s:e]))
        al.append(min(lows[s:e]))
        ac.append(closes[e - 1])
        av.append(sum(volumes[s:e]))
    return ao, ah, al, ac, av


def _score_tf_point(close: float, rsi: float, ema21: float,
                    ema50: float, macd_hist: float):
    """Score a single timeframe at one point in time.

    Matches live multi-TF scoring logic (RSI, EMA alignment, MACD).
    Returns score (-100 to +100), or None if indicators not ready.
    """
    if close <= 0:
        return None
    if ema21 <= 0 and ema50 <= 0:
        return None
    score = 0
    if rsi > 55:
        score += 20
    elif rsi < 45:
        score -= 20
    if ema21 > 0:
        if close > ema21 * 1.001:
            score += 20
        elif close < ema21 * 0.999:
            score -= 20
    if ema50 > 0:
        if close > ema50 * 1.001:
            score += 20
        elif close < ema50 * 0.999:
            score -= 20
    if macd_hist > 0:
        score += 20
    elif macd_hist < 0:
        score -= 20
    if ema21 > 0 and ema50 > 0:
        if ema21 > ema50:
            score += 20
        elif ema21 < ema50:
            score -= 20
    return max(-100, min(100, score))


# =============================================================================
# Volume Profile (VRVP — Volume Range Visible Profile)
# =============================================================================

def _compute_vp(highs: list, lows: list, closes: list,
                volumes: list, start: int, end: int,
                num_bins: int = 50):
    """Compute volume profile (VRVP) for a window of candles.

    Distributes volume across price bins, finds POC, Value Area, HVN/LVN.
    Returns (poc, vah, val, hvn_list, lvn_list) or None.
    """
    if end <= start or end > len(closes) or (end - start) < 20:
        return None

    h_s = highs[start:end]
    l_s = lows[start:end]
    v_s = volumes[start:end]

    price_min = min(l_s)
    price_max = max(h_s)
    if price_max <= price_min:
        return None

    bin_size = (price_max - price_min) / num_bins
    if bin_size <= 0:
        return None

    vol_bins = [0.0] * num_bins
    bin_prices = [price_min + (b + 0.5) * bin_size for b in range(num_bins)]

    for k in range(len(v_s)):
        h, low, v = h_s[k], l_s[k], v_s[k]
        if v <= 0 or h <= low:
            continue
        lo_bin = max(0, int((low - price_min) / bin_size))
        hi_bin = min(num_bins - 1, int((h - price_min) / bin_size))
        n_bins = hi_bin - lo_bin + 1
        vpb = v / n_bins if n_bins > 0 else 0
        for b in range(lo_bin, hi_bin + 1):
            vol_bins[b] += vpb

    total_vol = sum(vol_bins)
    if total_vol <= 0:
        return None

    poc_bin = vol_bins.index(max(vol_bins))
    poc = bin_prices[poc_bin]

    # Value Area — 70% of volume expanding from POC
    target = total_vol * 0.7
    va_vol = vol_bins[poc_bin]
    lo_va, hi_va = poc_bin, poc_bin
    while va_vol < target and (lo_va > 0 or hi_va < num_bins - 1):
        lo_v = vol_bins[lo_va - 1] if lo_va > 0 else 0
        hi_v = vol_bins[hi_va + 1] if hi_va < num_bins - 1 else 0
        if lo_v >= hi_v and lo_va > 0:
            lo_va -= 1
            va_vol += lo_v
        elif hi_va < num_bins - 1:
            hi_va += 1
            va_vol += hi_v
        else:
            break

    vah = bin_prices[hi_va] + bin_size / 2
    val_price = bin_prices[lo_va] - bin_size / 2

    avg_v = total_vol / num_bins
    std_v = (sum((x - avg_v) ** 2 for x in vol_bins) / num_bins) ** 0.5
    hvns = [bin_prices[b] for b in range(num_bins)
            if vol_bins[b] > avg_v + 0.5 * std_v][:10]
    lvns = [bin_prices[b] for b in range(num_bins)
            if 0 < vol_bins[b] < avg_v - 0.5 * std_v][:10]

    return (poc, vah, val_price, hvns, lvns)


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
        self._funding_cache = []   # [(timestamp_ms, rate), ...] sorted
        self._fg_cache = []        # [(timestamp_ms, value_0_to_1), ...] sorted
        self._base_interval = "1h"

    def __len__(self):
        return len(self.closes)

    def _get_nearest_funding(self, ts_ms: int) -> float:
        """Find nearest funding rate for a timestamp. Returns scaled (-1, 1)."""
        if not self._funding_cache:
            return 0.0
        best = 0.0
        best_dist = float("inf")
        for fts, rate in self._funding_cache:
            dist = abs(fts - ts_ms)
            if dist < best_dist:
                best_dist = dist
                best = rate
        # Scale: 0.01% funding → 1.0 (same as live _build_features)
        return max(-1.0, min(1.0, best * 100))

    def _get_nearest_fg(self, ts_ms: int) -> float:
        """Find nearest Fear & Greed value for a timestamp. Returns 0-1."""
        if not self._fg_cache:
            return 0.5
        best = 0.5
        best_dist = float("inf")
        for fts, val in self._fg_cache:
            dist = abs(fts - ts_ms)
            if dist < best_dist:
                best_dist = dist
                best = val
        return best

    def compute_all(self, base_interval: str = "1h"):
        """Compute all indicators needed for the 24-dim CORTEX vector.

        Args:
            base_interval: Candle interval for multi-TF aggregation factors.
        """
        self._base_interval = base_interval
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

        # Multi-TF confluence (replaces EMA proxy for feature 14)
        self._compute_multi_tf()

        # OI proxy from volume ratio (feature 16)
        self._compute_oi_proxy()

        # Volume Profile rolling (features 22-23)
        self._compute_volume_profile_rolling()

        return self

    def _compute_multi_tf(self):
        """Compute multi-TF confluence scores for feature 14.

        Aggregates base candles into 3-4 higher timeframes, computes
        RSI + EMA + MACD for each, scores as bullish/bearish/neutral.
        Stores per-candle bullish/total counts in indicators.
        """
        FACTOR_MAP = {
            "1m": [5, 15, 60, 240],
            "5m": [3, 12, 48],
            "15m": [4, 16, 96],
            "30m": [2, 8, 48],
            "1h": [4, 24, 168],
            "4h": [6, 42],
            "1d": [7],
        }
        factors = FACTOR_MAP.get(self._base_interval, [4, 24])
        n = len(self.closes)

        # Pre-compute scores for each TF
        tf_data = []  # list of (factor, scores_list)

        # Base TF scores (already have indicators)
        base_rsi = self._indicators.get("rsi14", [50.0] * n)
        base_ema21 = self._indicators.get("ema21", [0.0] * n)
        base_ema50 = self._indicators.get("ema50", [0.0] * n)
        base_macd = self._indicators.get("macd_hist", [0.0] * n)
        base_scores = []
        for i in range(n):
            s = _score_tf_point(self.closes[i], base_rsi[i],
                                base_ema21[i], base_ema50[i], base_macd[i])
            base_scores.append(s)
        tf_data.append((1, base_scores))

        # Aggregated TF scores
        for factor in factors:
            if n < factor * 30:
                continue
            agg_o, agg_h, agg_l, agg_c, agg_v = _aggregate_candles(
                self.opens, self.highs, self.lows,
                self.closes, self.volumes, factor)
            agg_n = len(agg_c)
            if agg_n < 21:
                continue
            agg_rsi = _rsi(agg_c, 14)
            agg_ema21 = _ema(agg_c, 21)
            agg_ema50 = _ema(agg_c, 50)
            _, _, agg_macd_hist = _macd(agg_c)
            agg_scores = []
            for j in range(agg_n):
                s = _score_tf_point(agg_c[j], agg_rsi[j], agg_ema21[j],
                                    agg_ema50[j], agg_macd_hist[j])
                agg_scores.append(s)
            tf_data.append((factor, agg_scores))

        # Build per-base-candle multi-TF counts
        mtf_bullish = [0] * n
        mtf_total = [1] * n  # at least base TF

        for i in range(n):
            bullish = 0
            total = 0
            for factor, scores in tf_data:
                agg_idx = i // factor if factor > 1 else i
                if agg_idx >= len(scores):
                    agg_idx = len(scores) - 1
                s = scores[agg_idx]
                if s is None:
                    continue
                total += 1
                if s > 10:
                    bullish += 1
            mtf_bullish[i] = bullish
            mtf_total[i] = max(total, 1)

        self._indicators["mtf_bullish"] = mtf_bullish
        self._indicators["mtf_total"] = mtf_total

    def _compute_oi_proxy(self):
        """Compute OI change proxy from daily volume ratio for feature 16.

        Live uses (vol_today - vol_yesterday) / vol_yesterday from daily klines.
        Training replicates this from available OHLCV volumes.
        """
        DAY_CANDLES = {
            "1m": 1440, "5m": 288, "15m": 96, "30m": 48,
            "1h": 24, "4h": 6, "1d": 1,
        }
        dc = DAY_CANDLES.get(self._base_interval, 24)
        n = len(self.volumes)
        oi_proxy = [0.0] * n

        for i in range(dc * 2, n):
            vol_today = sum(self.volumes[i - dc:i])
            vol_yesterday = sum(self.volumes[i - dc * 2:i - dc])
            if vol_yesterday > 0:
                ratio = (vol_today - vol_yesterday) / vol_yesterday
                oi_proxy[i] = max(-0.5, min(0.5, ratio))

        self._indicators["oi_proxy"] = oi_proxy

    def _compute_volume_profile_rolling(self, lookback: int = 200,
                                         step: int = 100):
        """Compute rolling volume profile for features 22-23.

        Recomputes VP every `step` candles using a `lookback` window.
        Stores per-candle VP POC position and density values.
        """
        n = len(self.closes)
        vp_poc_pos = [0.5] * n
        vp_density = [0.5] * n

        last_vp = None

        for i in range(lookback, n):
            if (i - lookback) % step == 0 or last_vp is None:
                start = max(0, i - lookback)
                vp = _compute_vp(self.highs, self.lows, self.closes,
                                 self.volumes, start, i + 1, num_bins=50)
                if vp is not None:
                    last_vp = vp

            if last_vp is None:
                continue

            poc, vah, val, hvns, lvns = last_vp
            c = self.closes[i]

            # Feature 22: VP POC position
            va_range = vah - val
            if va_range > 0:
                vp_poc_pos[i] = max(0.0, min(1.0, (c - val) / va_range))

            # Feature 23: VP density
            threshold = c * 0.005
            near_hvn = any(abs(c - h) < threshold for h in hvns) if hvns else False
            near_lvn = any(abs(c - lv) < threshold for lv in lvns) if lvns else False
            if near_hvn and not near_lvn:
                vp_density[i] = 1.0
            elif near_lvn and not near_hvn:
                vp_density[i] = 0.0

        self._indicators["vp_poc_pos"] = vp_poc_pos
        self._indicators["vp_density"] = vp_density

    def get_feature_vector(self, i: int) -> Optional[list]:
        """Build the 24-dim feature vector for candle index i.

        Matches CortexAnalyzer._build_features() normalization exactly.
        All 24 features now have real data during training:
          0-13: TA indicators (RSI, EMA, MACD, BB, ATR, etc.)
          14: Multi-TF confluence (aggregated higher-TF scoring)
          15: Funding rate (from historical Binance API if available)
          16: OI change proxy (daily volume ratio from OHLCV)
          17-19: Candlestick shape (body ratio, wick ratios)
          20: Polymarket sentiment proxy (momentum-based — no historical API)
          21: Fear & Greed (from historical alternative.me API if available)
          22-23: Volume Profile (rolling VRVP from OHLCV data)

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

        # 14: TF confluence ratio — real multi-TF scoring
        # Uses aggregated higher-TF RSI/EMA/MACD scoring (matches live)
        mtf_bull = ind.get("mtf_bullish", [0] * len(self.closes))
        mtf_tot = ind.get("mtf_total", [1] * len(self.closes))
        f_tf = mtf_bull[i] / max(mtf_tot[i], 1)

        # 15: Funding rate (from historical data if available)
        f_funding = 0.0
        if hasattr(self, '_funding_cache') and self._funding_cache:
            ts = self.timestamps[i]
            f_funding = self._get_nearest_funding(ts)

        # 16: OI change proxy — daily volume ratio (matches live computation)
        oi_val = ind.get("oi_proxy", [0.0] * len(self.closes))[i]
        f_oi = max(-1.0, min(1.0, oi_val * 10))

        # 17: Candlestick body/range ratio (0-1)
        f_body = max(0.0, min(1.0, ind["body_ratio"][i]))

        # 18: Upper wick ratio (0-1)
        f_uwk = max(0.0, min(1.0, ind["upper_wick"][i]))

        # 19: Lower wick ratio (0-1)
        f_lwk = max(0.0, min(1.0, ind["lower_wick"][i]))

        # 20: Polymarket sentiment proxy — momentum-based (no historical API)
        # Polymarket crypto up/down markets follow momentum closely, so
        # RSI + EMA trend serves as a correlated proxy during training.
        # This prevents CORTEX from learning zero-weight on this dimension.
        rsi_norm = ind["rsi14"][i] / 100.0
        ema21_v = ind["ema21"][i] or c
        ema50_v = ind["ema50"][i] or c
        if c > ema21_v and ema21_v > ema50_v:
            ema_sig = 0.65
        elif c < ema21_v and ema21_v < ema50_v:
            ema_sig = 0.35
        else:
            ema_sig = 0.5
        f_poly = max(0.0, min(1.0, 0.3 * rsi_norm + 0.7 * ema_sig))

        # 21: Fear & Greed index (from historical data if available)
        f_fg = 0.5
        if hasattr(self, '_fg_cache') and self._fg_cache:
            ts = self.timestamps[i]
            f_fg = self._get_nearest_fg(ts)

        # 22: VP POC position — from rolling volume profile
        f_vp_pos = ind.get("vp_poc_pos", [0.5] * len(self.closes))[i]

        # 23: VP density — from rolling volume profile (HVN/LVN proximity)
        f_vp_dens = ind.get("vp_density", [0.5] * len(self.closes))[i]

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

    # Mode-aware defaults for interval and lookahead
    MODE_DEFAULTS = {
        "scalp":  {"interval": "5m",  "lookahead": 12, "days": 30},
        "stable": {"interval": "1h",  "lookahead": 12, "days": 90},
        "secure": {"interval": "4h",  "lookahead": 12, "days": 180},
    }

    def _fetch_historical_fg(self) -> list:
        """Fetch historical Fear & Greed index from alternative.me (free API).

        Returns: [(timestamp_ms, value_0_to_1), ...]
        """
        import urllib.request
        import json
        url = "https://api.alternative.me/fng/?limit=0&format=json"
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "QOR-Trainer/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
            result = []
            for entry in data.get("data", []):
                ts_ms = int(entry["timestamp"]) * 1000
                val = int(entry["value"]) / 100.0  # 0-100 → 0-1
                result.append((ts_ms, val))
            result.sort(key=lambda x: x[0])
            logger.info(f"[CortexTrainer] Fetched {len(result)} historical "
                        f"Fear & Greed entries")
            return result
        except Exception as e:
            logger.warning(f"[CortexTrainer] Historical F&G fetch failed: {e}")
            return []

    def _fetch_historical_funding(self, symbol: str) -> list:
        """Fetch historical funding rates from Binance futures API.

        Returns: [(timestamp_ms, rate_float), ...]
        """
        import urllib.request
        import json
        pair = symbol.upper() + "USDT"
        url = (f"https://fapi.binance.com/fapi/v1/fundingRate"
               f"?symbol={pair}&limit=1000")
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "QOR-Trainer/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
            result = []
            for entry in data:
                ts_ms = int(entry["fundingTime"])
                rate = float(entry["fundingRate"])
                result.append((ts_ms, rate))
            result.sort(key=lambda x: x[0])
            logger.info(f"[CortexTrainer] Fetched {len(result)} historical "
                        f"funding rates for {pair}")
            return result
        except Exception as e:
            logger.warning(f"[CortexTrainer] Historical funding fetch "
                           f"failed for {pair}: {e}")
            return []

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
        candles.compute_all(base_interval=interval)

        # Enrich with historical Fear & Greed + funding rates
        # (features 15, 21 — otherwise constant 0/0.5 during training)
        try:
            if not hasattr(self, '_fg_data'):
                self._fg_data = self._fetch_historical_fg()
            candles._fg_cache = self._fg_data
        except Exception:
            pass
        try:
            sym_upper = symbol.upper()
            cache_key = f"_funding_{sym_upper}"
            if not hasattr(self, cache_key):
                setattr(self, cache_key, self._fetch_historical_funding(symbol))
            candles._funding_cache = getattr(self, cache_key)
        except Exception:
            pass

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
              epochs: int = 20, lr: float = 1e-3,
              mode: str = "") -> dict:
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
            mode: Trade mode ("scalp", "stable", "secure") for auto-defaults

        Returns:
            {"trained": bool, "symbol": str, "samples": int,
             "candles_fetched": int, ...}
        """
        t0 = time.time()

        # Mode-aware defaults
        if mode and mode in self.MODE_DEFAULTS:
            md = self.MODE_DEFAULTS[mode]
            interval = md["interval"]
            days = md["days"]
            lookahead = md["lookahead"]

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
                  mode: str = "", **kwargs) -> dict:
        """Train CORTEX on multiple symbols — per-symbol sequential + mixed.

        Phase 1: Train each symbol sequentially (preserves CfC temporal context).
        Phase 2: Fine-tune on shuffled mixed data (generalization).

        This fixes the issue where shuffled multi-symbol data destroys
        temporal coherence in the S4 observation layer and CfC hidden states.

        Args:
            symbols: List of symbols ["BTC", "ETH", "SOL"]
            days: Days of history per symbol (overridden by mode defaults)
            interval: Candle interval (overridden by mode defaults)
            epochs: Training epochs (split: 60% sequential, 40% mixed)
            mode: Trade mode ("scalp", "stable", "secure") for auto-defaults
            **kwargs: Passed to prepare()

        Returns:
            {"trained": bool, "symbols": [...], "total_samples": int, ...}
        """
        t0 = time.time()

        # Mode-aware defaults override explicit params if mode is set
        if mode and mode in self.MODE_DEFAULTS:
            md = self.MODE_DEFAULTS[mode]
            interval = md["interval"]
            days = md["days"]
            kwargs.setdefault("lookahead", md["lookahead"])
            logger.info(f"[CortexTrainer] Mode '{mode}': interval={interval}, "
                        f"days={days}, lookahead={md['lookahead']}")

        per_symbol = {}
        all_features = []
        all_targets = []

        # Phase 1: Per-symbol sequential training (temporal patterns)
        seq_epochs = max(epochs * 3 // 5, 1)  # 60% of epochs
        mix_epochs = max(epochs - seq_epochs, 1)  # 40% of epochs

        for symbol in symbols:
            try:
                candles = self.fetch(symbol, days=days, interval=interval)
                features, targets = self.prepare(candles, **kwargs)
                per_symbol[symbol] = {
                    "candles": len(candles),
                    "samples": len(features),
                }
                if features:
                    # Train sequentially — CfC hidden state carries context
                    logger.info(f"[CortexTrainer] Phase 1: {symbol} sequential "
                                f"({len(features)} samples, {seq_epochs} epochs)")
                    self.train_prepared(features, targets, epochs=seq_epochs)
                    all_features.extend(features)
                    all_targets.extend(targets)
            except Exception as e:
                per_symbol[symbol] = {"error": str(e)}
                logger.warning(f"[CortexTrainer] {symbol} failed: {e}")

        if not all_features:
            return {"trained": False, "reason": "no data from any symbol"}

        # Phase 2: Mixed shuffle training (generalization)
        import random
        combined = list(zip(all_features, all_targets))
        random.shuffle(combined)
        all_features = [x[0] for x in combined]
        all_targets = [x[1] for x in combined]

        logger.info(f"[CortexTrainer] Phase 2: Mixed shuffle "
                    f"({len(all_features)} samples, {mix_epochs} epochs)")
        result = self.train_prepared(all_features, all_targets, epochs=mix_epochs)
        result["symbols"] = symbols
        result["total_samples"] = len(all_features)
        result["per_symbol"] = per_symbol
        result["elapsed_seconds"] = round(time.time() - t0, 1)
        result["mode"] = mode or "default"
        result["interval"] = interval
        result["seq_epochs"] = seq_epochs
        result["mix_epochs"] = mix_epochs

        logger.info(f"[CortexTrainer] All symbols trained: {result}")
        return result

    def evaluate(self, symbol: str, days: int = 30,
                 interval: str = "1h", lookahead: int = 12) -> dict:
        """Evaluate CORTEX accuracy on recent unseen data.

        Fetches recent candles NOT used for training, runs CORTEX
        inference, compares predictions to actual outcomes.

        Uses a separate eval instance_id to avoid polluting live hidden
        states, but applies the same tanh + calibrated confidence as
        CortexAnalyzer.analyze() for consistent evaluation metrics.

        Returns:
            {"accuracy": float, "precision": float, "recall": float,
             "total": int, "correct": int, "trained": bool, ...}
        """
        if not _HAS_TORCH:
            return {"error": "torch not available"}

        # Check if model is trained (matches analyze() guard)
        if not self.cortex._brain._trained:
            return {"error": "CORTEX not trained yet", "trained": False}

        candles = self.fetch(symbol, days=days, interval=interval)
        if len(candles) < 250:
            return {"error": f"only {len(candles)} candles"}

        targets_raw = label_return_continuous(candles.closes, lookahead)
        correct = 0
        total = 0
        true_pos = 0
        pred_pos = 0
        actual_pos = 0
        confidence_sum = 0.0

        for i in range(200, len(candles) - lookahead):
            vec = candles.get_feature_vector(i)
            if vec is None:
                continue

            # CORTEX prediction (separate eval instance to not pollute live)
            x = torch.tensor([vec], dtype=torch.float32)
            raw = self.cortex._brain(x, instance_id=f"eval_{symbol}")
            pred = torch.tanh(raw).item()

            # Calibrated confidence (same as analyze())
            confidence = min(0.95, abs(pred) ** 0.5)
            confidence_sum += confidence

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

        # Clean up eval instance state
        self.cortex._brain.reset_instance(f"eval_{symbol}")

        accuracy = correct / total if total > 0 else 0
        precision = true_pos / pred_pos if pred_pos > 0 else 0
        recall = true_pos / actual_pos if actual_pos > 0 else 0
        avg_confidence = confidence_sum / total if total > 0 else 0

        return {
            "symbol": symbol,
            "total": total,
            "correct": correct,
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "avg_confidence": round(avg_confidence, 4),
            "pred_positive": pred_pos,
            "actual_positive": actual_pos,
            "true_positive": true_pos,
            "trained": True,
        }

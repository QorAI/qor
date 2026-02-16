"""
QOR Tools — 40+ Free API Tools (No Keys Required)
=====================================================
Ported from Go tools.go into native Python.

All tools use FREE public APIs — no API keys needed.

Categories:
  - Crypto & Finance: CoinGecko, Binance, Frankfurter
  - Trading Intelligence: Fear & Greed, Funding Rates, Open Interest,
    Technical Analysis (RSI/EMA/MACD/BB/ATR), Polymarket, Trending
  - Knowledge: Wikipedia, DuckDuckGo, Dictionary
  - Weather: Open-Meteo (no key needed!)
  - News: Free news mirror, Hacker News
  - Code: PyPI, npm, GitHub, arXiv, HuggingFace
  - Entertainment: Jokes, Trivia, Recipes, Reddit, NASA
  - Utility: Calculator, Time, Currency, Country info

Usage:
    from qor.tools import ToolExecutor

    tools = ToolExecutor()
    tools.register_all(gate)  # registers with ConfidenceGate

    # Or call directly:
    result = tools.call("get_crypto_price", "bitcoin")
"""

import logging
import re
import json
import time
import math
import threading
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable

logger = logging.getLogger(__name__)

# Shared CORTEX analyzer — set by runtime for multi_tf_analysis integration
_shared_cortex = None


def set_shared_cortex(cortex):
    """Wire a shared CortexAnalyzer for multi_tf_analysis CORTEX output."""
    global _shared_cortex
    _shared_cortex = cortex


# Browser tool imports (graceful fallback if browser.py or playwright missing)
try:
    from qor.browser import browse_agent as _browse_agent_fn
    from qor.browser import browse_web as _browse_web_simple
    from qor.browser import browse_screenshot as _browse_screenshot

    def _browse_web(query):
        """Use multi-step browse agent for smarter extraction."""
        return _browse_agent_fn(query, verbose=True)

except Exception:
    def _browse_web(query):
        return "Browser not available. Install with: pip install playwright && playwright install chromium"
    def _browse_screenshot(query):
        return "Browser not available. Install with: pip install playwright && playwright install chromium"

# Document & Video tool imports (graceful fallback)
try:
    from qor.documents import read_document as _read_document
except Exception:
    def _read_document(query):
        return "Document reader not available. Check qor/documents.py."

try:
    from qor.video import read_video as _read_video
except Exception:
    def _read_video(query):
        return "Video reader not available. Check qor/video.py."


# ==============================================================================
# HTTP HELPER (no dependencies needed — uses stdlib)
# ==============================================================================

def _http_get(url: str, headers: Dict[str, str] = None,
              timeout: int = 15) -> dict:
    """Simple HTTP GET that returns parsed JSON."""
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "QOR-AI-Agent/1.0")
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.reason}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Connection failed: {e.reason}")


def _http_post(url: str, data: dict, timeout: int = 15) -> dict:
    """Simple HTTP POST with JSON body."""
    body = json.dumps(data).encode()
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", "QOR-AI-Agent/1.0")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.reason}")


# ==============================================================================
# TECHNICAL INDICATOR HELPERS (pure Python math, no numpy/pandas)
# ==============================================================================

def _calc_ema(prices: list, period: int) -> float:
    """Exponential Moving Average."""
    if len(prices) < period:
        return prices[-1] if prices else 0.0
    k = 2.0 / (period + 1)
    ema = sum(prices[:period]) / period  # seed with SMA
    for p in prices[period:]:
        ema = p * k + ema * (1.0 - k)
    return ema


def _calc_ema_series(prices: list, period: int) -> list:
    """EMA series (returns list same length as prices, first period-1 are SMA)."""
    if not prices:
        return []
    result = []
    k = 2.0 / (period + 1)
    ema = sum(prices[:period]) / period if len(prices) >= period else prices[0]
    for i, p in enumerate(prices):
        if i < period:
            sma = sum(prices[:i + 1]) / (i + 1)
            result.append(sma)
            if i == period - 1:
                ema = sma
        else:
            ema = p * k + ema * (1.0 - k)
            result.append(ema)
    return result


def _calc_rsi(closes: list, period: int = 14) -> float:
    """Relative Strength Index (0-100)."""
    if len(closes) < period + 1:
        return 50.0  # neutral default
    gains = []
    losses = []
    for i in range(1, len(closes)):
        delta = closes[i] - closes[i - 1]
        gains.append(max(delta, 0.0))
        losses.append(max(-delta, 0.0))
    # First average
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    # Smoothed (Wilder's method)
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _calc_macd(closes: list) -> tuple:
    """MACD line, signal line, histogram. Returns (macd, signal, hist)."""
    ema12 = _calc_ema_series(closes, 12)
    ema26 = _calc_ema_series(closes, 26)
    if len(closes) < 26:
        return (0.0, 0.0, 0.0)
    macd_line = [ema12[i] - ema26[i] for i in range(len(closes))]
    signal = _calc_ema_series(macd_line[25:], 9)  # EMA(9) of MACD starting from period 26
    macd_val = macd_line[-1]
    signal_val = signal[-1] if signal else 0.0
    hist_val = macd_val - signal_val
    return (macd_val, signal_val, hist_val)


def _calc_bollinger(closes: list, period: int = 20, std_mult: float = 2.0) -> tuple:
    """Bollinger Bands. Returns (middle, upper, lower)."""
    if len(closes) < period:
        mid = sum(closes) / len(closes) if closes else 0.0
        return (mid, mid, mid)
    window = closes[-period:]
    middle = sum(window) / period
    variance = sum((x - middle) ** 2 for x in window) / period
    std = math.sqrt(variance)
    return (middle, middle + std_mult * std, middle - std_mult * std)


def _calc_atr(highs: list, lows: list, closes: list, period: int = 14) -> float:
    """Average True Range — volatility measure."""
    if len(closes) < 2:
        return 0.0
    trs = []
    for i in range(1, len(closes)):
        h = highs[i] if i < len(highs) else closes[i]
        l = lows[i] if i < len(lows) else closes[i]
        prev_c = closes[i - 1]
        tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
        trs.append(tr)
    if not trs:
        return 0.0
    if len(trs) <= period:
        return sum(trs) / len(trs)
    # Wilder's smoothing
    atr = sum(trs[:period]) / period
    for i in range(period, len(trs)):
        atr = (atr * (period - 1) + trs[i]) / period
    return atr


def _find_support_resistance(highs, lows, closes, lookback=50):
    """Find key support/resistance from recent swing highs/lows."""
    recent_h = highs[-lookback:]
    recent_l = lows[-lookback:]
    resistances = []
    supports = []
    for i in range(2, len(recent_h) - 2):
        if (recent_h[i] > recent_h[i-1] and recent_h[i] > recent_h[i-2] and
                recent_h[i] > recent_h[i+1] and recent_h[i] > recent_h[i+2]):
            resistances.append(recent_h[i])
        if (recent_l[i] < recent_l[i-1] and recent_l[i] < recent_l[i-2] and
                recent_l[i] < recent_l[i+1] and recent_l[i] < recent_l[i+2]):
            supports.append(recent_l[i])
    current = closes[-1]
    res = sorted([r for r in resistances if r > current])[:2]
    sup = sorted([s for s in supports if s < current], reverse=True)[:2]
    return sup, res


def _calc_adx(highs: list, lows: list, closes: list, period: int = 14) -> float:
    """Average Directional Index (0-100). Measures trend strength."""
    if len(closes) < period + 2:
        return 25.0  # neutral default
    # True Range + Directional Movement
    plus_dm = []
    minus_dm = []
    trs = []
    for i in range(1, len(closes)):
        h = highs[i] if i < len(highs) else closes[i]
        l = lows[i] if i < len(lows) else closes[i]
        prev_h = highs[i - 1] if (i - 1) < len(highs) else closes[i - 1]
        prev_l = lows[i - 1] if (i - 1) < len(lows) else closes[i - 1]
        prev_c = closes[i - 1]
        tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
        trs.append(tr)
        up = h - prev_h
        down = prev_l - l
        plus_dm.append(up if up > down and up > 0 else 0.0)
        minus_dm.append(down if down > up and down > 0 else 0.0)
    if len(trs) < period:
        return 25.0
    # Wilder's smoothing
    atr_s = sum(trs[:period]) / period
    pdm_s = sum(plus_dm[:period]) / period
    mdm_s = sum(minus_dm[:period]) / period
    dx_vals = []
    for i in range(period, len(trs)):
        atr_s = (atr_s * (period - 1) + trs[i]) / period
        pdm_s = (pdm_s * (period - 1) + plus_dm[i]) / period
        mdm_s = (mdm_s * (period - 1) + minus_dm[i]) / period
        if atr_s == 0:
            continue
        plus_di = 100 * pdm_s / atr_s
        minus_di = 100 * mdm_s / atr_s
        di_sum = plus_di + minus_di
        if di_sum == 0:
            continue
        dx_vals.append(abs(plus_di - minus_di) / di_sum * 100)
    if not dx_vals:
        return 25.0
    # ADX = smoothed DX
    adx = sum(dx_vals[:period]) / min(period, len(dx_vals))
    for i in range(period, len(dx_vals)):
        adx = (adx * (period - 1) + dx_vals[i]) / period
    return adx


def _calc_obv_direction(closes: list, volumes: list) -> float:
    """On-Balance Volume direction indicator (-1 to 1).
    Normalized slope of OBV over last 20 bars."""
    if not volumes or len(closes) < 2:
        return 0.0
    n = min(len(closes), len(volumes))
    obv = 0.0
    obv_series = [0.0]
    for i in range(1, n):
        vol = volumes[i] if i < len(volumes) else 0.0
        if closes[i] > closes[i - 1]:
            obv += vol
        elif closes[i] < closes[i - 1]:
            obv -= vol
        obv_series.append(obv)
    # Direction: compare recent OBV to 20-bar ago OBV
    lookback = min(20, len(obv_series) - 1)
    if lookback <= 0 or obv_series[-1] == obv_series[-lookback - 1] == 0:
        return 0.0
    recent = obv_series[-1]
    past = obv_series[-lookback - 1]
    max_vol = max(abs(v) for v in obv_series[-lookback:]) if lookback > 0 else 1.0
    if max_vol == 0:
        return 0.0
    return max(-1.0, min(1.0, (recent - past) / max_vol))


def _calc_vwap(closes: list, highs: list, lows: list, volumes: list) -> float:
    """Volume-Weighted Average Price over available data."""
    n = min(len(closes), len(highs), len(lows), len(volumes))
    if n == 0:
        return closes[-1] if closes else 0.0
    total_pv = 0.0
    total_v = 0.0
    for i in range(n):
        typical = (highs[i] + lows[i] + closes[i]) / 3.0
        vol = volumes[i] if volumes[i] > 0 else 0.0
        total_pv += typical * vol
        total_v += vol
    return total_pv / total_v if total_v > 0 else closes[-1]


def _find_fvg(highs: list, lows: list, closes: list, opens: list,
              lookback: int = 20) -> dict:
    """Find Fair Value Gaps (FVGs) in recent price action.

    A bullish FVG: candle[i-2] high < candle[i] low (gap up not filled).
    A bearish FVG: candle[i-2] low > candle[i] high (gap down not filled).

    Returns dict with:
      bullish_fvgs: list of {top, bottom, filled} — unfilled bullish gaps
      bearish_fvgs: list of {top, bottom, filled} — unfilled bearish gaps
      nearest_bullish: closest bullish FVG below current price (support)
      nearest_bearish: closest bearish FVG above current price (resistance)
      fvg_bias: +1 if price in/near bullish FVG, -1 if bearish, 0 if none
    """
    n = len(highs)
    if n < 3:
        return {"bullish_fvgs": [], "bearish_fvgs": [],
                "nearest_bullish": None, "nearest_bearish": None, "fvg_bias": 0}

    current = closes[-1]
    start = max(0, n - lookback)
    bullish_fvgs = []
    bearish_fvgs = []

    for i in range(start + 2, n):
        # Bullish FVG: candle[i-2] high < candle[i] low — gap up
        gap_bottom = highs[i - 2]
        gap_top = lows[i]
        if gap_top > gap_bottom:
            # Check if filled by any candle after i
            filled = False
            for j in range(i + 1, n):
                if lows[j] <= gap_bottom:
                    filled = True
                    break
            if not filled:
                bullish_fvgs.append({"top": gap_top, "bottom": gap_bottom})

        # Bearish FVG: candle[i-2] low > candle[i] high — gap down
        gap_top_b = lows[i - 2]
        gap_bottom_b = highs[i]
        if gap_top_b > gap_bottom_b:
            filled = False
            for j in range(i + 1, n):
                if highs[j] >= gap_top_b:
                    filled = True
                    break
            if not filled:
                bearish_fvgs.append({"top": gap_top_b, "bottom": gap_bottom_b})

    # Find nearest FVGs to current price
    nearest_bull = None
    nearest_bear = None
    for fvg in bullish_fvgs:
        if fvg["top"] <= current:  # below price = support
            if nearest_bull is None or fvg["top"] > nearest_bull["top"]:
                nearest_bull = fvg
    for fvg in bearish_fvgs:
        if fvg["bottom"] >= current:  # above price = resistance
            if nearest_bear is None or fvg["bottom"] < nearest_bear["bottom"]:
                nearest_bear = fvg

    # FVG bias: +1 if price is sitting in or near a bullish FVG, -1 for bearish
    fvg_bias = 0
    for fvg in bullish_fvgs:
        if fvg["bottom"] <= current <= fvg["top"] * 1.005:
            fvg_bias = 1
            break
    if fvg_bias == 0:
        for fvg in bearish_fvgs:
            if fvg["bottom"] * 0.995 <= current <= fvg["top"]:
                fvg_bias = -1
                break

    return {
        "bullish_fvgs": bullish_fvgs, "bearish_fvgs": bearish_fvgs,
        "nearest_bullish": nearest_bull, "nearest_bearish": nearest_bear,
        "fvg_bias": fvg_bias,
    }


def _calc_fibonacci_levels(highs: list, lows: list, closes: list,
                           lookback: int = 50) -> dict:
    """Calculate Fibonacci retracement levels from recent swing high/low.

    Finds the major swing high and swing low in the lookback window,
    then computes standard Fibonacci retracement levels (23.6%, 38.2%,
    50%, 61.8%, 78.6%) between them.

    Returns dict with:
      swing_high, swing_low: the detected swing points
      direction: "UP" if latest close > midpoint (measuring retracement of uptrend)
                 "DOWN" if below (measuring retracement of downtrend)
      levels: dict of {0.236: price, 0.382: price, 0.5: price, 0.618: price, 0.786: price}
      nearest_support: closest fib level below current price
      nearest_resistance: closest fib level above current price
      fib_bias: +1 if price bouncing off 38.2-61.8 support, -1 if rejecting, 0 neutral
    """
    n = len(closes)
    if n < 10:
        return {"swing_high": 0, "swing_low": 0, "direction": "NONE",
                "levels": {}, "nearest_support": 0, "nearest_resistance": 0,
                "fib_bias": 0}

    start = max(0, n - lookback)
    window_highs = highs[start:]
    window_lows = lows[start:]
    swing_high = max(window_highs)
    swing_low = min(window_lows)
    current = closes[-1]

    if swing_high <= swing_low:
        return {"swing_high": swing_high, "swing_low": swing_low,
                "direction": "NONE", "levels": {},
                "nearest_support": 0, "nearest_resistance": 0, "fib_bias": 0}

    diff = swing_high - swing_low
    mid = (swing_high + swing_low) / 2

    # Determine trend direction for retracement calculation
    if current > mid:
        # Uptrend — retrace down from high
        direction = "UP"
        levels = {
            0.236: swing_high - 0.236 * diff,
            0.382: swing_high - 0.382 * diff,
            0.500: swing_high - 0.500 * diff,
            0.618: swing_high - 0.618 * diff,
            0.786: swing_high - 0.786 * diff,
        }
    else:
        # Downtrend — retrace up from low
        direction = "DOWN"
        levels = {
            0.236: swing_low + 0.236 * diff,
            0.382: swing_low + 0.382 * diff,
            0.500: swing_low + 0.500 * diff,
            0.618: swing_low + 0.618 * diff,
            0.786: swing_low + 0.786 * diff,
        }

    # Find nearest support/resistance fib levels
    nearest_sup = 0
    nearest_res = 0
    for _, level in sorted(levels.items()):
        if level <= current and level > nearest_sup:
            nearest_sup = level
        if level >= current and (nearest_res == 0 or level < nearest_res):
            nearest_res = level

    # Fib bias: bouncing off golden zone (38.2-61.8%) is significant
    fib_382 = levels.get(0.382, 0)
    fib_618 = levels.get(0.618, 0)
    golden_low = min(fib_382, fib_618)
    golden_high = max(fib_382, fib_618)
    tolerance = diff * 0.02  # 2% of range

    fib_bias = 0
    if golden_low - tolerance <= current <= golden_high + tolerance:
        # Price in golden zone — bias depends on trend direction
        if direction == "UP":
            fib_bias = 1   # retracement to golden zone in uptrend = bullish bounce
        else:
            fib_bias = -1  # retracement to golden zone in downtrend = bearish rejection

    return {
        "swing_high": swing_high, "swing_low": swing_low,
        "direction": direction, "levels": levels,
        "nearest_support": nearest_sup, "nearest_resistance": nearest_res,
        "fib_bias": fib_bias,
    }


def _calc_pivot_points(highs: list, lows: list, closes: list) -> dict:
    """Calculate Classic Pivot Points from previous period's High/Low/Close.

    PP  = (H + L + C) / 3
    S1  = 2×PP - H     R1 = 2×PP - L
    S2  = PP - (H - L)  R2 = PP + (H - L)
    S3  = L - 2×(H-PP)  R3 = H + 2×(PP-L)

    Uses the second-to-last candle (previous completed period) to set levels
    for the current candle.
    """
    if len(closes) < 2:
        return {"pp": 0, "s1": 0, "s2": 0, "s3": 0,
                "r1": 0, "r2": 0, "r3": 0, "pivot_bias": 0}

    h = highs[-2]
    l = lows[-2]
    c = closes[-2]
    current = closes[-1]

    pp = (h + l + c) / 3.0
    s1 = 2 * pp - h
    r1 = 2 * pp - l
    s2 = pp - (h - l)
    r2 = pp + (h - l)
    s3 = l - 2 * (h - pp)
    r3 = h + 2 * (pp - l)

    # Bias: above PP = bullish territory, below = bearish
    pivot_bias = 1 if current > pp else (-1 if current < pp else 0)

    return {
        "pp": pp, "s1": s1, "s2": s2, "s3": s3,
        "r1": r1, "r2": r2, "r3": r3,
        "pivot_bias": pivot_bias,
    }


def _find_swing_points(values: list, order: int = 5) -> list:
    """Find swing highs and swing lows in a series.

    A swing high at index i: values[i] >= all values in [i-order..i+order].
    A swing low  at index i: values[i] <= all values in [i-order..i+order].

    Returns list of (index, value, type) where type is "high" or "low".
    """
    points = []
    n = len(values)
    for i in range(order, n - order):
        window = values[i - order: i + order + 1]
        if values[i] == max(window):
            points.append((i, values[i], "high"))
        elif values[i] == min(window):
            points.append((i, values[i], "low"))
    return points


def _detect_divergence(closes: list, highs: list, lows: list,
                       rsi_series: list, macd_series: list,
                       obv_series: list = None,
                       lookback: int = 40) -> dict:
    """Detect bullish and bearish divergence between price and RSI/MACD/OBV.

    Bullish divergence:  price makes LOWER low,  indicator makes HIGHER low
    Bearish divergence:  price makes HIGHER high, indicator makes LOWER high

    Checks last `lookback` candles for swing points. Divergence on 30m+ TFs
    is a powerful reversal signal.

    OBV (On-Balance Volume) divergence:
      Bullish: price lower low + OBV higher low (accumulation despite selling)
      Bearish: price higher high + OBV lower high (distribution despite buying)

    Returns:
      rsi_div: "bullish" | "bearish" | "none"
      macd_div: "bullish" | "bearish" | "none"
      obv_div: "bullish" | "bearish" | "none"
      div_score: -15 to +15 (net divergence signal for scoring)
    """
    n = min(len(closes), len(lows), len(highs))
    if n < 20:
        return {"rsi_div": "none", "macd_div": "none",
                "obv_div": "none", "div_score": 0}

    start = max(0, n - lookback)
    price_lows = lows[start:]
    price_highs = highs[start:]

    # Build indicator series aligned to the window
    rsi_win = rsi_series[start:] if len(rsi_series) >= n else rsi_series
    macd_win = macd_series[start:] if len(macd_series) >= n else macd_series
    obv_win = []
    if obv_series and len(obv_series) >= n:
        obv_win = obv_series[start:]
    elif obv_series:
        obv_win = obv_series

    rsi_div = "none"
    macd_div = "none"
    obv_div = "none"
    div_score = 0

    order = 3  # swing detection sensitivity (3 candles each side)

    # --- Find swing lows for bullish divergence ---
    price_swing_lows = _find_swing_points(price_lows, order=order)
    price_swing_lows = [(i, v, t) for i, v, t in price_swing_lows if t == "low"]

    if len(price_swing_lows) >= 2:
        # Last two swing lows
        prev_sw = price_swing_lows[-2]
        curr_sw = price_swing_lows[-1]

        # Price made lower low?
        if curr_sw[1] < prev_sw[1]:
            # Check RSI at those points
            if (prev_sw[0] < len(rsi_win) and curr_sw[0] < len(rsi_win)):
                rsi_prev = rsi_win[prev_sw[0]]
                rsi_curr = rsi_win[curr_sw[0]]
                if rsi_curr > rsi_prev:  # RSI higher low = bullish divergence
                    rsi_div = "bullish"
                    div_score += 15

            # Check MACD at those points
            if (prev_sw[0] < len(macd_win) and curr_sw[0] < len(macd_win)):
                macd_prev = macd_win[prev_sw[0]]
                macd_curr = macd_win[curr_sw[0]]
                if macd_curr > macd_prev:  # MACD higher low = bullish divergence
                    if macd_div != "bullish":
                        macd_div = "bullish"
                    if rsi_div == "bullish":
                        div_score += 5  # double divergence bonus

            # Check OBV at those points (accumulation despite price drop)
            if (obv_win and prev_sw[0] < len(obv_win)
                    and curr_sw[0] < len(obv_win)):
                obv_prev = obv_win[prev_sw[0]]
                obv_curr = obv_win[curr_sw[0]]
                if obv_curr > obv_prev:  # OBV higher low = bullish volume div
                    obv_div = "bullish"
                    div_score += 10
                    if rsi_div == "bullish":
                        div_score += 5  # triple divergence bonus (RSI+OBV)

    # --- Find swing highs for bearish divergence ---
    price_swing_highs = _find_swing_points(price_highs, order=order)
    price_swing_highs = [(i, v, t) for i, v, t in price_swing_highs if t == "high"]

    if len(price_swing_highs) >= 2:
        prev_sw = price_swing_highs[-2]
        curr_sw = price_swing_highs[-1]

        # Price made higher high?
        if curr_sw[1] > prev_sw[1]:
            # Check RSI
            if (prev_sw[0] < len(rsi_win) and curr_sw[0] < len(rsi_win)):
                rsi_prev = rsi_win[prev_sw[0]]
                rsi_curr = rsi_win[curr_sw[0]]
                if rsi_curr < rsi_prev:  # RSI lower high = bearish divergence
                    rsi_div = "bearish" if rsi_div == "none" else rsi_div
                    div_score -= 15

            # Check MACD
            if (prev_sw[0] < len(macd_win) and curr_sw[0] < len(macd_win)):
                macd_prev = macd_win[prev_sw[0]]
                macd_curr = macd_win[curr_sw[0]]
                if macd_curr < macd_prev:  # MACD lower high = bearish divergence
                    if macd_div == "none":
                        macd_div = "bearish"
                    if rsi_div == "bearish":
                        div_score -= 5  # double divergence bonus

            # Check OBV (distribution despite price rise)
            if (obv_win and prev_sw[0] < len(obv_win)
                    and curr_sw[0] < len(obv_win)):
                obv_prev = obv_win[prev_sw[0]]
                obv_curr = obv_win[curr_sw[0]]
                if obv_curr < obv_prev:  # OBV lower high = bearish volume div
                    obv_div = "bearish" if obv_div == "none" else obv_div
                    div_score -= 10
                    if rsi_div == "bearish":
                        div_score -= 5  # triple divergence bonus (RSI+OBV)

    return {"rsi_div": rsi_div, "macd_div": macd_div,
            "obv_div": obv_div, "div_score": div_score}


def _score_tf(stats: dict) -> int:
    """Score a single timeframe from -100 (max bearish) to +100 (max bullish).

    Uses ALL computed indicators (additive first, then multiplier):
      1. EMA alignment (±20), 2. RSI momentum (±20), 3. MACD histogram (±15),
      4. Bollinger position (±10), 5. EMA200 context (±10), 6. OBV volume (±5),
      7. FVG bias (±10), 8. Fibonacci bias (±10), 9. Pivot Point (±5),
      10. Divergence (±15/±10/±5),
      11. ADX trend-strength multiplier (0.6x–1.3x) — applied LAST.

    Score > 10 = BULLISH TF,  < -10 = BEARISH TF,  else NEUTRAL.
    """
    score = 0
    price = stats.get("current", 0)
    if price <= 0:
        return 0

    # --- 1. EMA Trend alignment (±20) ---
    ema21 = stats.get("ema21", 0)
    ema50 = stats.get("ema50", 0)
    if ema21 > 0 and ema50 > 0:
        if price > ema21 > ema50:
            score += 20          # full bullish stack
        elif price > ema21:
            score += 10          # above fast EMA only
        elif price < ema21 < ema50:
            score -= 20          # full bearish stack
        elif price < ema21:
            score -= 10          # below fast EMA only

    # --- 2. RSI momentum (±20) ---
    rsi = stats.get("rsi", 50)
    if rsi >= 65:
        score += 20
    elif rsi >= 55:
        score += 10
    elif rsi <= 35:
        score -= 20
    elif rsi <= 45:
        score -= 10
    # Exhaustion adjustment — overbought/oversold fade
    if rsi > 80:
        score -= 5
    elif rsi < 20:
        score += 5

    # --- 3. MACD histogram direction (±15) ---
    hist = stats.get("macd_hist", 0)
    if hist > 0:
        score += 15
    elif hist < 0:
        score -= 15

    # --- 4. Bollinger Band position (±10) ---
    bb_upper = stats.get("bb_upper", 0)
    bb_lower = stats.get("bb_lower", 0)
    if bb_upper > bb_lower > 0:
        bb_mid = (bb_upper + bb_lower) / 2
        if price > bb_mid:
            score += 10          # above middle band
        else:
            score -= 10          # below middle band

    # --- 5. EMA200 long-term context (±10) ---
    ema200 = stats.get("ema200", 0)
    if ema200 > 0:
        if price > ema200:
            score += 10
        else:
            score -= 10

    # --- 6. OBV volume confirmation (±5) ---
    obv = stats.get("obv_dir", 0)
    if obv > 0.2:
        score += 5
    elif obv < -0.2:
        score -= 5

    # --- 7. FVG bias (±10) ---
    fvg_bias = stats.get("fvg_bias", 0)
    score += fvg_bias * 10  # +10 if in bullish FVG, -10 if bearish FVG

    # --- 8. Fibonacci bias (±10) ---
    fib_bias = stats.get("fib_bias", 0)
    score += fib_bias * 10  # +10 if bouncing off golden zone support

    # --- 9. Pivot Point bias (±5) ---
    pivot_bias = stats.get("pivot_bias", 0)
    score += pivot_bias * 5  # above PP = bullish, below = bearish

    # --- 10. Divergence (±15 RSI, ±5 double, ±10 OBV, ±5 triple) ---
    div_score = stats.get("div_score", 0)
    score += div_score  # RSI ±15, MACD +5, OBV ±10, triple ±5

    # --- 11. ADX trend-strength multiplier (applied AFTER all components) ---
    adx = stats.get("adx", 25)
    if adx > 30:
        score = int(score * 1.3)    # strong trend → amplify
    elif adx < 15:
        score = int(score * 0.6)    # choppy → dampen

    return max(-100, min(100, score))


def _calc_trade_levels(current, atr, ema21, ema50, rsi, bb_upper, bb_lower,
                       supports, resistances,
                       fib_data=None, fvg_data=None):
    """Calculate entry, stop loss, take profit based on TA signals.

    Uses support/resistance + Fibonacci levels + FVG zones to place
    optimal SL/TP. Fib 61.8% and FVG edges act as confluence zones.
    """
    # Bullish: price above EMA21, not extremely overbought (RSI < 80)
    bullish = current > ema21 and rsi < 80
    # Bearish: price below EMA21, not extremely oversold (RSI > 15)
    bearish = current < ema21 and rsi > 15

    # Gather all support/resistance confluences from Fib + FVG + S/R
    all_supports = list(supports) if supports else []
    all_resistances = list(resistances) if resistances else []

    if fib_data and fib_data.get("levels"):
        fib_sup = fib_data.get("nearest_support", 0)
        fib_res = fib_data.get("nearest_resistance", 0)
        if fib_sup > 0:
            all_supports.append(fib_sup)
        if fib_res > 0:
            all_resistances.append(fib_res)
        # Add 61.8% (golden ratio) as key level
        fib_618 = fib_data["levels"].get(0.618, 0)
        if fib_618 > 0:
            if fib_618 < current:
                all_supports.append(fib_618)
            else:
                all_resistances.append(fib_618)

    if fvg_data:
        # Bullish FVG bottom = support zone
        nb = fvg_data.get("nearest_bullish")
        if nb and nb["bottom"] < current:
            all_supports.append(nb["bottom"])
        # Bearish FVG top = resistance zone
        nbear = fvg_data.get("nearest_bearish")
        if nbear and nbear["top"] > current:
            all_resistances.append(nbear["top"])

    # Sort and deduplicate
    all_supports = sorted(set(s for s in all_supports if 0 < s < current),
                          reverse=True)   # highest first (nearest to price)
    all_resistances = sorted(set(r for r in all_resistances if r > current))  # lowest first

    if bullish:
        entry = current
        # SL: best support (S/R, Fib, FVG) or ATR fallback
        stop_loss = max(all_supports[0] if all_supports else current - 2 * atr,
                        current - 2 * atr)
        # TP: nearest resistance or ATR fallback
        tp1 = all_resistances[0] if all_resistances else current + 2 * atr
        tp2 = all_resistances[1] if len(all_resistances) > 1 else current + 3 * atr
        bias = "LONG"
        # If support/resistance gives bad R:R, use ATR-based levels
        risk = abs(current - stop_loss)
        reward = abs(tp1 - current)
        if risk > 0 and (reward / risk) < 1.5:
            tp1 = current + 2 * atr
            tp2 = current + 3 * atr
    elif bearish:
        entry = current
        # SHORT: SL at nearest resistance (S/R, Fib, FVG)
        stop_loss = min(all_resistances[0] if all_resistances else current + 2 * atr,
                        current + 2 * atr)
        # TP: nearest support below
        tp1 = all_supports[0] if all_supports else current - 2 * atr
        tp2 = all_supports[1] if len(all_supports) > 1 else current - 3 * atr
        bias = "SHORT"
        risk = abs(stop_loss - current)
        reward = abs(current - tp1)
        if risk > 0 and (reward / risk) < 1.5:
            tp1 = current - 2 * atr
            tp2 = current - 3 * atr
    else:
        entry = current
        stop_loss = current - 1.5 * atr
        tp1 = current + 1.5 * atr
        tp2 = current + 3 * atr
        bias = "NEUTRAL"

    risk = abs(current - stop_loss)
    reward1 = abs(tp1 - current)
    rr1 = reward1 / risk if risk > 0 else 0

    return {"bias": bias, "entry": entry, "stop_loss": stop_loss,
            "tp1": tp1, "tp2": tp2, "risk_reward": rr1}


def _extract_crypto_symbol(query: str) -> str:
    """Extract Binance futures symbol from query (e.g. 'btc' -> 'BTC')."""
    symbols = {
        "bitcoin": "BTC", "btc": "BTC",
        "ethereum": "ETH", "eth": "ETH",
        "solana": "SOL", "sol": "SOL",
        "cardano": "ADA", "ada": "ADA",
        "dogecoin": "DOGE", "doge": "DOGE",
        "xrp": "XRP", "ripple": "XRP",
        "polkadot": "DOT", "dot": "DOT",
        "bnb": "BNB", "binance coin": "BNB",
        "avax": "AVAX", "avalanche": "AVAX",
        "matic": "MATIC", "polygon": "MATIC",
        "link": "LINK", "chainlink": "LINK",
        "atom": "ATOM", "cosmos": "ATOM",
        "near": "NEAR", "sui": "SUI",
        "apt": "APT", "aptos": "APT",
        "arb": "ARB", "arbitrum": "ARB",
        "op": "OP", "optimism": "OP",
        "pepe": "PEPE", "shib": "SHIB",
    }
    q = query.lower()
    for key, sym in symbols.items():
        if key in q:
            return sym
    # Try to extract raw uppercase symbol
    match = re.search(r'\b([A-Z]{2,6})\b', query)
    if match:
        return match.group(1)
    return "BTC"


def _extract_coingecko_id(query: str) -> str:
    """Extract CoinGecko coin ID from query."""
    ids = {
        "bitcoin": "bitcoin", "btc": "bitcoin",
        "ethereum": "ethereum", "eth": "ethereum",
        "solana": "solana", "sol": "solana",
        "cardano": "cardano", "ada": "cardano",
        "dogecoin": "dogecoin", "doge": "dogecoin",
        "xrp": "ripple", "ripple": "ripple",
        "polkadot": "polkadot", "dot": "polkadot",
        "bnb": "binancecoin", "binance coin": "binancecoin",
        "avax": "avalanche-2", "avalanche": "avalanche-2",
        "matic": "matic-network", "polygon": "matic-network",
        "link": "chainlink", "chainlink": "chainlink",
        "atom": "cosmos", "cosmos": "cosmos",
        "near": "near", "sui": "sui",
        "apt": "aptos", "aptos": "aptos",
        "arb": "arbitrum", "arbitrum": "arbitrum",
        "pepe": "pepe", "shib": "shiba-inu",
    }
    q = query.lower()
    for key, coin_id in ids.items():
        if key in q:
            return coin_id
    return "bitcoin"


def _extract_stock_symbol(query: str) -> str:
    """Extract stock ticker symbol from query."""
    # Common company name -> ticker mappings
    companies = {
        "apple": "AAPL", "microsoft": "MSFT", "google": "GOOGL",
        "alphabet": "GOOGL", "amazon": "AMZN", "tesla": "TSLA",
        "nvidia": "NVDA", "meta": "META", "facebook": "META",
        "netflix": "NFLX", "amd": "AMD", "intel": "INTC",
        "disney": "DIS", "boeing": "BA", "walmart": "WMT",
        "jpmorgan": "JPM", "goldman": "GS", "berkshire": "BRK-B",
        "visa": "V", "mastercard": "MA", "paypal": "PYPL",
        "coca cola": "KO", "pepsi": "PEP", "nike": "NKE",
        "ibm": "IBM", "oracle": "ORCL", "salesforce": "CRM",
        "uber": "UBER", "spotify": "SPOT", "zoom": "ZM",
        "snap": "SNAP", "twitter": "X", "coinbase": "COIN",
        "palantir": "PLTR", "snowflake": "SNOW", "shopify": "SHOP",
        "alibaba": "BABA", "samsung": "005930.KS", "toyota": "TM",
        "sony": "SONY", "tsmc": "TSM", "arm": "ARM",
    }
    # Indian stocks (NSE) — mapped to Yahoo Finance .NS suffix
    indian_stocks = {
        "reliance": "RELIANCE.NS", "tcs": "TCS.NS", "infosys": "INFY.NS",
        "infy": "INFY.NS", "hdfcbank": "HDFCBANK.NS", "hdfc bank": "HDFCBANK.NS",
        "icicibank": "ICICIBANK.NS", "icici bank": "ICICIBANK.NS",
        "sbin": "SBIN.NS", "sbi": "SBIN.NS", "state bank": "SBIN.NS",
        "bhartiartl": "BHARTIARTL.NS", "bharti airtel": "BHARTIARTL.NS",
        "airtel": "BHARTIARTL.NS",
        "itc": "ITC.NS", "hindunilvr": "HINDUNILVR.NS",
        "hindustan unilever": "HINDUNILVR.NS", "hul": "HINDUNILVR.NS",
        "kotakbank": "KOTAKBANK.NS", "kotak": "KOTAKBANK.NS",
        "wipro": "WIPRO.NS", "hcltech": "HCLTECH.NS",
        "techm": "TECHM.NS", "tech mahindra": "TECHM.NS",
        "tatamotors": "TATAMOTORS.NS", "tata motors": "TATAMOTORS.NS",
        "tatasteel": "TATASTEEL.NS", "tata steel": "TATASTEEL.NS",
        "maruti": "MARUTI.NS", "bajfinance": "BAJFINANCE.NS",
        "bajaj finance": "BAJFINANCE.NS",
        "sunpharma": "SUNPHARMA.NS", "sun pharma": "SUNPHARMA.NS",
        "adanient": "ADANIENT.NS", "adani": "ADANIENT.NS",
        "lt": "LT.NS", "larsen": "LT.NS",
        "axisbank": "AXISBANK.NS", "axis bank": "AXISBANK.NS",
        "powergrid": "POWERGRID.NS", "ntpc": "NTPC.NS",
        "ongc": "ONGC.NS", "coalindia": "COALINDIA.NS",
        "indusindbk": "INDUSINDBK.NS", "indusind": "INDUSINDBK.NS",
    }
    # NSE index futures — Yahoo Finance symbols
    indian_indices = {
        "nifty": "^NSEI", "nifty 50": "^NSEI", "nifty50": "^NSEI",
        "banknifty": "^NSEBANK", "bank nifty": "^NSEBANK",
        "sensex": "^BSESN",
    }
    q = query.lower()
    # Check Indian stocks first (longer names match better)
    for name, ticker in indian_stocks.items():
        if name in q:
            return ticker
    for name, ticker in indian_indices.items():
        if name in q:
            return ticker
    for name, ticker in companies.items():
        if name in q:
            return ticker
    # Check if query IS an NSE symbol (all uppercase, 2-12 chars)
    raw = query.strip().upper()
    if re.match(r'^[A-Z]{2,12}$', raw):
        # Check if it looks like an Indian stock (not a known crypto)
        _crypto_names = {"BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE",
                         "AVAX", "DOT", "LINK", "ATOM", "NEAR", "SUI",
                         "ARB", "PEPE", "LTC", "MATIC", "UNI", "AAVE"}
        if raw not in _crypto_names:
            # Try with .NS suffix — Yahoo Finance resolves NSE stocks
            return f"{raw}.NS"
    # Try to extract raw ticker (2-5 uppercase letters)
    match = re.search(r'\b([A-Z]{1,5}(?:-[A-Z])?)\b', query)
    if match:
        candidate = match.group(1)
        # Filter out common non-ticker words
        if candidate not in ("I", "A", "THE", "IN", "OF", "FOR", "IS",
                             "IT", "AT", "TO", "ON", "BY", "OR", "AN",
                             "AND", "BUT", "NOT", "GET", "SET", "NEW"):
            return candidate
    return ""


def _extract_forex_base(query: str) -> tuple:
    """Extract forex base and target currencies from query."""
    # Valid ISO 4217 currency codes to match against
    valid_codes = {"USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD",
                   "CNY", "INR", "BRL", "KRW", "RUB", "TRY", "MXN", "ZAR",
                   "SEK", "NOK", "DKK", "SGD", "HKD", "TWD", "THB", "MYR",
                   "IDR", "PHP", "PLN", "CZK", "HUF", "ILS", "ARS", "CLP",
                   "COP", "PEN", "EGP", "NGN", "KES", "SAR", "AED", "QAR"}
    # Try explicit pair patterns: "EUR/USD", "EUR-USD", "EURUSD", "EUR to USD"
    pair_match = re.search(r'\b([A-Z]{3})\s*[/\-]\s*([A-Z]{3})\b', query.upper())
    if pair_match:
        a, b = pair_match.group(1), pair_match.group(2)
        if a in valid_codes and b in valid_codes:
            return a, b
    # Try "X to Y" pattern
    to_match = re.search(r'\b([A-Z]{3})\s+to\s+([A-Z]{3})\b', query.upper())
    if to_match:
        a, b = to_match.group(1), to_match.group(2)
        if a in valid_codes and b in valid_codes:
            return a, b
    # Try 6-letter concatenated pair: "EURUSD"
    concat_match = re.search(r'\b([A-Z]{6})\b', query.upper())
    if concat_match:
        pair = concat_match.group(1)
        a, b = pair[:3], pair[3:]
        if a in valid_codes and b in valid_codes:
            return a, b
    # Try standalone currency codes in the query
    found_codes = []
    for word in query.upper().split():
        clean = word.strip(".,!?/()-")
        if clean in valid_codes and clean not in found_codes:
            found_codes.append(clean)
    if len(found_codes) >= 2:
        return found_codes[0], found_codes[1]
    if len(found_codes) == 1:
        base = found_codes[0]
        return (base, "USD") if base != "USD" else ("EUR", "USD")
    # Try common currency names
    currencies = {
        "dollar": "USD", "euro": "EUR", "pound": "GBP",
        "yen": "JPY", "yuan": "CNY", "renminbi": "CNY",
        "franc": "CHF", "rupee": "INR", "real": "BRL",
        "won": "KRW", "ruble": "RUB", "lira": "TRY",
        "peso": "MXN", "rand": "ZAR", "krona": "SEK",
        "krone": "NOK", "ringgit": "MYR", "baht": "THB",
    }
    found = []
    for name, code in currencies.items():
        if name in query.lower():
            found.append(code)
    if len(found) >= 2:
        return found[0], found[1]
    if len(found) == 1:
        base = found[0]
        return (base, "USD") if base != "USD" else ("EUR", "USD")
    return "USD", ""  # default: show all rates from USD


# ==============================================================================
# CRYPTO & FINANCE
# ==============================================================================

def get_crypto_price(query: str) -> str:
    """Get cryptocurrency price from CoinGecko. FREE, no key needed."""
    coin = _extract_crypto(query)
    currency = "usd"
    url = (f"https://api.coingecko.com/api/v3/simple/price"
           f"?ids={coin}&vs_currencies={currency}"
           f"&include_24hr_change=true&include_market_cap=true")
    data = _http_get(url)
    if coin not in data:
        return f"Coin '{coin}' not found on CoinGecko"
    d = data[coin]
    price = d.get(currency, 0)
    change = d.get(f"{currency}_24h_change", 0)
    mcap = d.get(f"{currency}_market_cap", 0)
    direction = "up" if change > 0 else "down"
    return (f"{coin.title()}: ${price:,.2f} ({direction} {abs(change):.1f}% "
            f"in 24h) | Market cap: ${mcap:,.0f}")


def get_binance_price(query: str) -> str:
    """Get real-time price with 24h stats from Binance."""
    symbol = _extract_trading_symbol(query).replace("-", "")
    if not symbol.endswith("USDT"):
        symbol += "USDT"
    url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
    data = _http_get(url)
    price = float(data.get("lastPrice", 0))
    high = float(data.get("highPrice", 0))
    low = float(data.get("lowPrice", 0))
    change_pct = float(data.get("priceChangePercent", 0))
    volume = float(data.get("quoteVolume", 0))
    direction = "up" if change_pct > 0 else "down"
    return (f"{data.get('symbol', symbol)}: ${price:,.2f} "
            f"({direction} {abs(change_pct):.1f}% in 24h) | "
            f"24h High: ${high:,.2f} | 24h Low: ${low:,.2f} | "
            f"24h Volume: ${volume:,.0f} USDT (Binance)")


def get_crypto_market(query: str) -> str:
    """Get top cryptocurrencies by market cap."""
    limit = 10
    url = (f"https://api.coingecko.com/api/v3/coins/markets"
           f"?vs_currency=usd&order=market_cap_desc"
           f"&per_page={limit}&page=1&sparkline=false")
    data = _http_get(url)
    lines = ["Top Cryptocurrencies by Market Cap:"]
    for c in data[:limit]:
        change = c.get("price_change_percentage_24h", 0) or 0
        lines.append(
            f"  {c['market_cap_rank']}. {c['name']} ({c['symbol'].upper()}): "
            f"${c['current_price']:,.2f} ({change:+.1f}%)"
        )
    return "\n".join(lines)


def convert_currency(query: str) -> str:
    """Convert currency using Frankfurter (free, no key)."""
    # Try to extract "100 USD to EUR" pattern
    match = re.search(r'(\d+\.?\d*)\s*([A-Za-z]{3})\s*(?:to|in)\s*([A-Za-z]{3})',
                      query, re.IGNORECASE)
    if match:
        amount, from_c, to_c = float(match.group(1)), match.group(2).upper(), match.group(3).upper()
    else:
        amount, from_c, to_c = 1.0, "USD", "EUR"
    url = f"https://api.frankfurter.app/latest?from={from_c}&to={to_c}"
    data = _http_get(url)
    rates = data.get("rates", {})
    if to_c in rates:
        converted = rates[to_c] * amount
        return f"{amount} {from_c} = {converted:.2f} {to_c} (rate: {rates[to_c]:.4f})"
    return f"Could not convert {from_c} to {to_c}"


def crypto_history(query: str) -> str:
    """Get historical crypto price for a specific date. FREE, no key."""
    coin = _extract_crypto(query)
    # Extract date from query: "bitcoin price on 2021-11-10" or "btc november 2021"
    date_match = re.search(r'(\d{4})-(\d{1,2})-(\d{1,2})', query)
    if not date_match:
        # Try "DD Month YYYY" or "Month DD YYYY"
        months = {"jan": "01", "feb": "02", "mar": "03", "apr": "04",
                  "may": "05", "jun": "06", "jul": "07", "aug": "08",
                  "sep": "09", "oct": "10", "nov": "11", "dec": "12"}
        for m_name, m_num in months.items():
            m = re.search(rf'{m_name}\w*\s+(\d{{1,2}})\s*,?\s*(\d{{4}})', query.lower())
            if m:
                date_match = type('M', (), {'group': lambda s, i: {1: m.group(2), 2: m_num, 3: m.group(1)}[i]})()
                break
            m = re.search(rf'(\d{{1,2}})\s+{m_name}\w*\s+(\d{{4}})', query.lower())
            if m:
                date_match = type('M', (), {'group': lambda s, i: {1: m.group(2), 2: m_num, 3: m.group(1)}[i]})()
                break
    if not date_match:
        return f"Specify a date. Example: 'bitcoin price on 2021-11-10'"
    dd = f"{int(date_match.group(3)):02d}"
    mm = f"{int(date_match.group(2)):02d}"
    yyyy = date_match.group(1)
    url = (f"https://api.coingecko.com/api/v3/coins/{coin}/history"
           f"?date={dd}-{mm}-{yyyy}")
    data = _http_get(url)
    market = data.get("market_data", {})
    if not market:
        return f"No data for {coin} on {yyyy}-{mm}-{dd}"
    price = market.get("current_price", {}).get("usd", 0)
    mcap = market.get("market_cap", {}).get("usd", 0)
    return (f"{coin.title()} on {yyyy}-{mm}-{dd}: "
            f"${price:,.2f} | Market cap: ${mcap:,.0f}")


# ==============================================================================
# TRADING INTELLIGENCE (FREE APIs, no keys)
# ==============================================================================

def get_fear_greed(query: str) -> str:
    """Crypto Fear & Greed Index from alternative.me. FREE, no key."""
    url = "https://api.alternative.me/fng/?limit=1"
    data = _http_get(url)
    entries = data.get("data", [])
    if not entries:
        return "Fear & Greed Index not available"
    entry = entries[0]
    value = int(entry.get("value", 0))
    classification = entry.get("value_classification", "Unknown")
    timestamp = entry.get("timestamp", "")
    # Add interpretation
    if value <= 20:
        interpretation = "Extreme fear — potential buying opportunity (contrarian signal)"
    elif value <= 40:
        interpretation = "Fear — market is cautious, watch for reversal"
    elif value <= 60:
        interpretation = "Neutral — market is undecided"
    elif value <= 80:
        interpretation = "Greed — market is optimistic, consider taking profits"
    else:
        interpretation = "Extreme greed — potential sell signal (contrarian signal)"
    date_str = ""
    if timestamp:
        try:
            dt = datetime.fromtimestamp(int(timestamp), tz=timezone.utc)
            date_str = f" ({dt.strftime('%Y-%m-%d')})"
        except (ValueError, OSError):
            pass
    return (f"Fear & Greed Index: {value}/100 — {classification}{date_str}\n"
            f"  {interpretation}")


def get_polymarket(query: str) -> str:
    """Prediction market odds from Polymarket. FREE, no key.
    Covers crypto, politics, elections, sports, games, world events."""
    # Extract search terms
    q = _extract_query_text(query).lower()
    stop = {"polymarket", "prediction", "predictions", "market", "odds",
            "probability", "what", "does", "say", "about", "the", "of", "for",
            "is", "are", "a", "an", "in", "on", "to", "how", "do", "my",
            "this", "that", "it", "be", "have", "has", "can", "get", "me",
            "sentiment", "overview", "condition", "looking", "will", "who",
            "win", "chances", "forecast", "bet", "betting", "current"}
    filter_words = [w for w in q.split() if w not in stop and len(w) > 1]
    search_term = " ".join(filter_words[:3]) if filter_words else ""

    lines = ["Polymarket Predictions:"]

    # Detect crypto-specific queries and use tag-based filtering
    crypto_terms = {"bitcoin", "btc", "ethereum", "eth", "solana", "crypto",
                    "dogecoin", "xrp", "cardano", "bnb", "avalanche", "polkadot"}
    is_crypto = any(t in q for t in crypto_terms)

    if is_crypto and search_term:
        # Fetch crypto-tagged markets directly (bypasses politics-dominated volume)
        markets_url = ("https://gamma-api.polymarket.com/markets?"
                       "closed=false&limit=100&order=liquidityNum&ascending=false"
                       "&tag=crypto")
        try:
            markets = _http_get(markets_url, timeout=15) or []
        except RuntimeError:
            markets = []
        if isinstance(markets, list) and search_term:
            keywords = search_term.lower().split()
            scored = []
            for m in markets:
                question = m.get("question", "").lower()
                hits = sum(1 for kw in keywords if kw in question)
                if hits > 0:
                    scored.append((hits, m))
            scored.sort(key=lambda x: (-x[0],
                        -float(x[1].get("liquidityNum", 0) or 0)))
            for _, m in scored[:5]:
                _append_market_line(lines, m)
        # If tag filter got results, return early
        if len(lines) > 1:
            return "\n".join(lines)

    # Generic path: fetch top events by volume (politics, sports, etc.)
    events_url = ("https://gamma-api.polymarket.com/events?"
                  "closed=false&limit=30&order=volume&ascending=false")
    events = []
    try:
        events = _http_get(events_url, timeout=15) or []
    except RuntimeError:
        pass

    if not isinstance(events, list):
        events = []

    # Client-side filtering: match search terms against event titles
    if search_term and events:
        keywords = search_term.lower().split()
        scored = []
        for ev in events:
            title = ev.get("title", "").lower()
            hits = sum(1 for kw in keywords if kw in title)
            if hits > 0:
                scored.append((hits, ev))
        scored.sort(key=lambda x: (-x[0], -float(x[1].get("volume", 0))))
        matched = [ev for _, ev in scored[:5]]
    else:
        matched = events[:5]  # No filter: show top by volume

    for ev in matched:
        title = ev.get("title", "?")
        volume = ev.get("volume", ev.get("volumeNum", 0))
        vol_str = f" | Vol: ${float(volume):,.0f}" if volume else ""
        lines.append(f"  {title}{vol_str}")
        # Show sub-markets with YES/NO odds
        sub_markets = ev.get("markets", [])
        if isinstance(sub_markets, list):
            for sm in sub_markets[:3]:
                _append_market_line(lines, sm)

    # Fallback: fetch active markets directly if events returned nothing
    if len(lines) <= 1:
        try:
            mkts = _http_get(
                "https://gamma-api.polymarket.com/markets?"
                "active=true&closed=false&limit=10"
                "&order=liquidityNum&ascending=false",
                timeout=15) or []
            if isinstance(mkts, list):
                if search_term:
                    keywords = search_term.lower().split()
                    mkts = [m for m in mkts
                            if any(kw in m.get("question", "").lower()
                                   for kw in keywords)]
                for m in mkts[:5]:
                    _append_market_line(lines, m)
        except RuntimeError:
            pass

    return "\n".join(lines) if len(lines) > 1 else "No Polymarket predictions found"


def _append_market_line(lines: list, m: dict):
    """Format a single Polymarket market entry and append to lines."""
    question = m.get("question", m.get("title", "?"))
    prices_raw = m.get("outcomePrices", "")
    yes_pct, no_pct = "?", "?"
    try:
        if isinstance(prices_raw, str) and prices_raw:
            prices = json.loads(prices_raw)
        elif isinstance(prices_raw, list):
            prices = prices_raw
        else:
            prices = []
        if len(prices) >= 2:
            yes_pct = f"{float(prices[0]) * 100:.0f}%"
            no_pct = f"{float(prices[1]) * 100:.0f}%"
        elif len(prices) == 1:
            yes_pct = f"{float(prices[0]) * 100:.0f}%"
    except (json.JSONDecodeError, ValueError, IndexError):
        pass
    volume = m.get("volume", m.get("volumeNum", 0))
    vol_str = f" | Vol: ${float(volume):,.0f}" if volume else ""
    lines.append(f"  {question}")
    lines.append(f"    YES: {yes_pct} / NO: {no_pct}{vol_str}")


# ==============================================================================
# Structured sentiment functions for trading engine (not chat tools)
# ==============================================================================

# Module-level caches for sentiment data (simple TTL via timestamp)
_poly_sentiment_cache: Dict[str, tuple] = {}   # symbol -> (data, timestamp)
_fg_value_cache: tuple = (50, 0.0)             # (value, timestamp)
_calendar_cache: tuple = ([], 0.0)             # (events, timestamp)
_volume_profile_cache: Dict[str, tuple] = {}   # symbol -> (data, timestamp)


def get_polymarket_sentiment(symbol: str) -> dict:
    """Get Polymarket Up/Down probability for a crypto symbol.

    Fetches the next active Up/Down recurring market for the symbol.
    Returns structured dict for trading engine consumption.

    Args:
        symbol: Crypto symbol ("BTC", "ETH", "SOL", "XRP")

    Returns:
        {up_prob: float, down_prob: float, liquidity: float,
         volume_24h: float, timeframe: str, available: bool}
    """
    neutral = {"up_prob": 0.5, "down_prob": 0.5, "liquidity": 0,
               "volume_24h": 0, "timeframe": "", "available": False}

    sym = symbol.upper()
    # Only crypto symbols have Polymarket Up/Down series
    series_map = {
        "BTC": ["btc-updown-5m", "btc-updown-15m", "btc-updown-hourly", "btc-updown-daily"],
        "ETH": ["eth-updown-15m", "eth-updown-hourly"],
        "SOL": ["sol-updown-15m"],
        "XRP": ["xrp-updown-15m"],
    }
    slugs = series_map.get(sym)
    if not slugs:
        return neutral

    # Check cache (5-min TTL)
    cached = _poly_sentiment_cache.get(sym)
    if cached and (time.time() - cached[1]) < 300:
        return cached[0]

    try:
        url = ("https://gamma-api.polymarket.com/events?"
               "tag=crypto-prices&closed=false"
               "&order=startDate&ascending=false&limit=50")
        events = _http_get(url, timeout=10)
        if not isinstance(events, list):
            events = []
    except Exception:
        return neutral

    # Find best match: prefer shortest timeframe (5m > 15m > hourly > daily)
    best = None
    best_slug_idx = len(slugs)  # Lower index = shorter timeframe = preferred

    for ev in events:
        ev_slug = ev.get("slug", "")
        for idx, slug in enumerate(slugs):
            if slug in ev_slug and idx < best_slug_idx:
                markets = ev.get("markets", [])
                if markets:
                    best = ev
                    best_slug_idx = idx
                    break

    if not best:
        _poly_sentiment_cache[sym] = (neutral, time.time())
        return neutral

    # Parse Up/Down from the event's markets
    markets = best.get("markets", [])
    up_prob = 0.5
    down_prob = 0.5
    total_liquidity = 0
    total_volume = 0
    timeframe = ""

    # Extract timeframe from slug
    slug = best.get("slug", "")
    if "5m" in slug:
        timeframe = "5m"
    elif "15m" in slug:
        timeframe = "15m"
    elif "hourly" in slug:
        timeframe = "1h"
    elif "daily" in slug:
        timeframe = "1d"

    for m in markets:
        prices_raw = m.get("outcomePrices", "")
        outcomes_raw = m.get("outcomes", [])
        liq = float(m.get("liquidityNum", m.get("liquidity", 0)) or 0)
        vol = float(m.get("volumeNum", m.get("volume", 0)) or 0)
        total_liquidity += liq
        total_volume += vol

        try:
            if isinstance(prices_raw, str) and prices_raw:
                prices = json.loads(prices_raw)
            elif isinstance(prices_raw, list):
                prices = prices_raw
            else:
                continue
        except (json.JSONDecodeError, ValueError):
            continue

        # Parse outcomes array (may be JSON string or list)
        if isinstance(outcomes_raw, str):
            try:
                outcomes_raw = json.loads(outcomes_raw)
            except (json.JSONDecodeError, ValueError):
                outcomes_raw = []

        # Map prices using outcomes array directly (not question text)
        # Polymarket outcomes = ["Up", "Down"], prices = ["0.55", "0.45"]
        for i, outcome in enumerate(outcomes_raw):
            if i >= len(prices):
                break
            try:
                price = float(prices[i])
            except (ValueError, TypeError):
                continue
            ol = outcome.strip().lower()
            if ol in ("up", "higher", "above"):
                up_prob = price
            elif ol in ("down", "lower", "below"):
                down_prob = price

    # Normalize if both extracted (they should sum to ~1.0)
    total = up_prob + down_prob
    if total > 0 and abs(total - 1.0) > 0.1:
        up_prob = up_prob / total
        down_prob = down_prob / total

    result = {
        "up_prob": round(up_prob, 4),
        "down_prob": round(down_prob, 4),
        "liquidity": round(total_liquidity, 0),
        "volume_24h": round(total_volume, 0),
        "timeframe": timeframe,
        "available": True,
    }

    _poly_sentiment_cache[sym] = (result, time.time())
    logger.info(f"[PM] Polymarket {sym}: Up={up_prob*100:.1f}% ({timeframe})")
    return result


def get_fear_greed_value() -> int:
    """Get Fear & Greed index as raw integer (0-100) for trading engine.

    Structured counterpart of get_fear_greed() which returns human-readable text.
    5-min cache. Returns 50 (neutral) on failure.
    """
    global _fg_value_cache
    cached_val, cached_ts = _fg_value_cache
    if (time.time() - cached_ts) < 300:
        return cached_val

    try:
        data = _http_get("https://api.alternative.me/fng/?limit=1", timeout=10)
        entries = data.get("data", [])
        if entries:
            value = int(entries[0].get("value", 50))
            _fg_value_cache = (value, time.time())
            logger.info(f"[PM] Fear & Greed: {value}/100")
            return value
    except Exception:
        pass

    # Return last known value or neutral
    return cached_val


def get_polymarket_calendar() -> list:
    """Get high-impact upcoming events from Polymarket for risk management.

    Fetches high-volume prediction markets related to macro events
    (Fed/FOMC/CPI/NFP/Bitcoin ETF/SEC/tariffs) and returns them with
    impact classification and time-to-resolution.

    30-min cache. Returns [] on failure.

    Returns:
        [{name, probability, volume, end_date, impact_level, minutes_until_end}]
    """
    global _calendar_cache
    cached_events, cached_ts = _calendar_cache
    if (time.time() - cached_ts) < 1800:
        return cached_events

    high_keywords = {
        "fed", "fomc", "cpi", "inflation", "nfp", "gdp", "bitcoin etf",
        "sec", "tariff", "election", "government shutdown", "rate cut",
        "rate hike", "interest rate", "debt ceiling", "default",
    }
    medium_keywords = {
        "earnings", "housing", "pmi", "halving", "upgrade", "ipo",
        "jobs report", "unemployment", "retail sales",
    }

    try:
        url = ("https://gamma-api.polymarket.com/events?"
               "closed=false&limit=50&order=volume&ascending=false")
        events = _http_get(url, timeout=10)
        if not isinstance(events, list):
            events = []
    except Exception:
        return cached_events if cached_events else []

    results = []
    now = datetime.now(timezone.utc)

    for ev in events:
        title = ev.get("title", ev.get("name", "")).lower()
        end_date_str = ev.get("endDate", ev.get("end_date", ""))
        volume = float(ev.get("volume", ev.get("volumeNum", 0)) or 0)

        # Classify impact
        impact = None
        for kw in high_keywords:
            if kw in title:
                impact = "high"
                break
        if not impact:
            for kw in medium_keywords:
                if kw in title:
                    impact = "medium"
                    break
        if not impact:
            continue  # Skip events that don't match any keyword

        # Parse end date
        minutes_until_end = -1
        if end_date_str:
            try:
                # Handle various date formats
                for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ",
                            "%Y-%m-%d"):
                    try:
                        end_dt = datetime.strptime(end_date_str, fmt).replace(
                            tzinfo=timezone.utc)
                        minutes_until_end = (end_dt - now).total_seconds() / 60
                        break
                    except ValueError:
                        continue
            except Exception:
                pass

        # Get probability from first market
        probability = 0.5
        markets = ev.get("markets", [])
        if markets:
            m = markets[0]
            prices_raw = m.get("outcomePrices", "")
            try:
                if isinstance(prices_raw, str) and prices_raw:
                    prices = json.loads(prices_raw)
                elif isinstance(prices_raw, list):
                    prices = prices_raw
                else:
                    prices = []
                if prices:
                    probability = float(prices[0])
            except (json.JSONDecodeError, ValueError, IndexError):
                pass

        results.append({
            "name": ev.get("title", ev.get("name", "Unknown")),
            "probability": round(probability, 3),
            "volume": round(volume, 0),
            "end_date": end_date_str,
            "impact_level": impact,
            "minutes_until_end": round(minutes_until_end, 0),
        })

    _calendar_cache = (results, time.time())
    if results:
        logger.info(f"[PM] Calendar: {len(results)} events "
                    f"({sum(1 for e in results if e['impact_level'] == 'high')} high-impact)")
    return results


def get_volume_profile(symbol: str, asset_type: str = "crypto") -> dict:
    """Compute Volume Profile (VRVP) from OHLCV klines for any asset.

    Returns POC, VAH, VAL, HVN zones, LVN zones.
    Works for crypto (Binance 1h klines), stocks, commodities, forex (Yahoo Finance).

    Args:
        symbol: Asset symbol (e.g. "BTC", "AAPL", "GC=F", "EURUSD=X")
        asset_type: One of "crypto", "stock", "commodity", "forex"
    """
    global _volume_profile_cache
    neutral = {"poc": 0, "vah": 0, "val": 0, "hvn_zones": [],
               "lvn_zones": [], "total_volume": 0, "available": False}

    # Check cache (5-min TTL)
    cache_key = f"{asset_type}:{symbol}"
    cached = _volume_profile_cache.get(cache_key)
    if cached and (time.time() - cached[1]) < 300:
        return cached[0]

    try:
        highs, lows, closes, volumes = [], [], [], []

        if asset_type == "crypto":
            # Binance 1h klines — 500 candles ≈ 20 days
            pair = symbol.upper().rstrip("USDT") + "USDT"
            url = (f"https://api.binance.com/api/v3/klines"
                   f"?symbol={pair}&interval=1h&limit=500")
            data = _http_get(url, timeout=10)
            if not data or not isinstance(data, list) or len(data) < 50:
                return neutral
            for k in data:
                highs.append(float(k[2]))
                lows.append(float(k[3]))
                closes.append(float(k[4]))
                volumes.append(float(k[5]))
        else:
            # Yahoo Finance — use 1h interval, 1mo range for stocks/commodities/forex
            yahoo_sym = symbol
            # Map common names to Yahoo symbols if needed
            q = symbol.lower()
            for name, ysym in _COMMODITY_YAHOO.items():
                if name == q or q == ysym.lower():
                    yahoo_sym = ysym
                    break
            for pair, ysym in _FOREX_YAHOO.items():
                if pair == q or q == ysym.lower():
                    yahoo_sym = ysym
                    break
            ohlc = _fetch_ohlc_yahoo(yahoo_sym, interval="1h", range_="1mo")
            if not ohlc or len(ohlc.get("closes", [])) < 50:
                # Fallback to daily data with longer range
                ohlc = _fetch_ohlc_yahoo(yahoo_sym, interval="1d", range_="6mo")
            if not ohlc or len(ohlc.get("closes", [])) < 20:
                return neutral
            highs = ohlc["highs"]
            lows = ohlc["lows"]
            closes = ohlc["closes"]
            volumes = ohlc.get("volumes", [])

        if not volumes or len(volumes) < 20:
            return neutral

        # --- Volume Profile computation ---
        price_high = max(highs)
        price_low = min(lows)
        price_range = price_high - price_low
        if price_range <= 0:
            return neutral

        n_bins = 100
        bin_size = price_range / n_bins
        vol_bins = [0.0] * n_bins

        # Distribute each candle's volume across its high-low range bins
        for i in range(len(highs)):
            h, l, v = highs[i], lows[i], volumes[i]
            if v <= 0 or h <= l:
                continue
            low_bin = max(0, int((l - price_low) / bin_size))
            high_bin = min(n_bins - 1, int((h - price_low) / bin_size))
            span = high_bin - low_bin + 1
            per_bin = v / span
            for b in range(low_bin, high_bin + 1):
                vol_bins[b] += per_bin

        total_vol = sum(vol_bins)
        if total_vol <= 0:
            return neutral

        # POC — bin with highest volume
        poc_bin = vol_bins.index(max(vol_bins))
        poc_price = price_low + (poc_bin + 0.5) * bin_size

        # Value Area (70%) — expand outward from POC
        va_vol = vol_bins[poc_bin]
        va_target = total_vol * 0.70
        lo, hi = poc_bin, poc_bin
        while va_vol < va_target and (lo > 0 or hi < n_bins - 1):
            up_vol = vol_bins[hi + 1] if hi + 1 < n_bins else 0
            dn_vol = vol_bins[lo - 1] if lo - 1 >= 0 else 0
            if up_vol >= dn_vol and hi + 1 < n_bins:
                hi += 1
                va_vol += vol_bins[hi]
            elif lo - 1 >= 0:
                lo -= 1
                va_vol += vol_bins[lo]
            else:
                hi = min(hi + 1, n_bins - 1)
                va_vol += vol_bins[hi]

        val_price = price_low + lo * bin_size          # Value Area Low
        vah_price = price_low + (hi + 1) * bin_size    # Value Area High

        # HVN / LVN detection
        avg_vol = total_vol / n_bins
        hvn_threshold = avg_vol * 1.5
        lvn_threshold = avg_vol * 0.3
        hvn_zones = []
        lvn_zones = []
        for b in range(n_bins):
            bp = round(price_low + (b + 0.5) * bin_size, 2)
            if vol_bins[b] > hvn_threshold:
                hvn_zones.append(bp)
            elif vol_bins[b] < lvn_threshold:
                lvn_zones.append(bp)

        result = {
            "poc": round(poc_price, 2),
            "vah": round(vah_price, 2),
            "val": round(val_price, 2),
            "hvn_zones": hvn_zones[:20],  # Cap to avoid huge lists
            "lvn_zones": lvn_zones[:20],
            "total_volume": round(total_vol, 2),
            "available": True,
        }
        _volume_profile_cache[cache_key] = (result, time.time())
        logger.info(f"[VP] {symbol}: POC=${result['poc']:,.2f} "
                    f"VAH=${result['vah']:,.2f} VAL=${result['val']:,.2f} "
                    f"({len(hvn_zones)} HVN, {len(lvn_zones)} LVN)")
        return result

    except Exception as e:
        logger.debug(f"[VP] Volume profile failed for {symbol}: {e}")
        return neutral


# -- Quant Signals cache (Hurst, Z-Score, asset volatility from live prices) --
_quant_signals_cache: Dict[str, tuple] = {}   # key -> (data, timestamp)


def get_quant_signals(symbol: str, asset_type: str = "crypto") -> dict:
    """Compute live quant signals (Hurst, Z-Score, asset volatility) from price data.

    Uses hourly OHLC for crypto (Binance), daily for stocks/commodities/forex (Yahoo).
    Cached with 5-min TTL.

    Returns:
        {"hurst": 0.55, "z_score": -0.8, "asset_vol": 0.45,
         "hurst_regime": "trending", "z_label": "neutral", "available": True}
    """
    import math

    neutral = {
        "hurst": 0.5, "z_score": 0.0, "asset_vol": 0.0,
        "hurst_regime": "random", "z_label": "neutral",
        "price_returns": [], "available": False,
    }

    cache_key = f"{asset_type}:{symbol}"
    cached = _quant_signals_cache.get(cache_key)
    if cached:
        data, ts = cached
        if time.time() - ts < 300:  # 5-min TTL
            return data

    try:
        # Fetch closing prices
        closes = None
        sym_lower = symbol.lower().strip()

        if asset_type == "crypto":
            ohlc = _fetch_ohlc_binance(symbol, interval="1h", limit=200)
            if ohlc:
                closes = ohlc["closes"]
        else:
            # Stocks, commodities, forex via Yahoo
            yahoo_sym = None
            if asset_type == "commodity":
                yahoo_sym = _COMMODITY_YAHOO.get(sym_lower)
            elif asset_type == "forex":
                yahoo_sym = _FOREX_YAHOO.get(sym_lower)
            if not yahoo_sym:
                yahoo_sym = symbol.upper()
            ohlc = _fetch_ohlc_yahoo(yahoo_sym, interval="1d", range_="6mo")
            if ohlc:
                closes = ohlc["closes"]

        if not closes or len(closes) < 30:
            return neutral

        # --- Hurst Exponent (R/S analysis) ---
        from qor.quant import QuantMetrics
        hurst = QuantMetrics.hurst_exponent(closes)

        # --- Z-Score (20-period) ---
        z_score = QuantMetrics.z_score(closes, window=20)

        # --- Asset Volatility (annualized from daily/hourly returns) ---
        log_returns = []
        for i in range(1, len(closes)):
            if closes[i] > 0 and closes[i - 1] > 0:
                log_returns.append(math.log(closes[i] / closes[i - 1]))
        if len(log_returns) >= 10:
            mean_r = sum(log_returns) / len(log_returns)
            var_r = sum((r - mean_r) ** 2 for r in log_returns) / len(log_returns)
            std_r = math.sqrt(var_r)
            # Annualize: sqrt(periods_per_year) * std
            if asset_type == "crypto":
                ann_factor = math.sqrt(365 * 24)  # hourly data, crypto 24/7
            else:
                ann_factor = math.sqrt(252)  # daily data, stocks
            asset_vol = round(std_r * ann_factor, 4)
        else:
            asset_vol = 0.0

        # Labels
        if hurst > 0.6:
            hurst_regime = "trending"
        elif hurst < 0.4:
            hurst_regime = "mean_reverting"
        else:
            hurst_regime = "random"

        if z_score > 2.0:
            z_label = "overbought"
        elif z_score < -2.0:
            z_label = "oversold"
        elif z_score > 1.0:
            z_label = "elevated"
        elif z_score < -1.0:
            z_label = "depressed"
        else:
            z_label = "neutral"

        # Per-period percentage returns (for CAPM Alpha / Information Ratio)
        pct_returns = []
        for i in range(1, len(closes)):
            if closes[i - 1] > 0:
                pct_returns.append((closes[i] - closes[i - 1]) / closes[i - 1] * 100.0)

        result = {
            "hurst": hurst,
            "z_score": z_score,
            "asset_vol": asset_vol,
            "hurst_regime": hurst_regime,
            "z_label": z_label,
            "price_returns": pct_returns[-30:],  # last 30 for benchmark
            "available": True,
        }
        _quant_signals_cache[cache_key] = (result, time.time())
        logger.info(f"[QUANT] {symbol}: H={hurst:.3f} ({hurst_regime}), "
                    f"Z={z_score:.2f} ({z_label}), vol={asset_vol:.2%}")
        return result

    except Exception as e:
        logger.debug(f"[QUANT] Quant signals failed for {symbol}: {e}")
        return neutral


def get_funding_rate(query: str) -> str:
    """Binance perpetual funding rate. FREE, no key."""
    symbol = _extract_crypto_symbol(query)
    url = (f"https://fapi.binance.com/fapi/v1/fundingRate"
           f"?symbol={symbol}USDT&limit=3")
    try:
        data = _http_get(url)
    except RuntimeError as e:
        return f"Funding rate not available for {symbol}: {e}"
    if not data:
        return f"No funding rate data for {symbol}USDT"
    lines = [f"Funding Rate — {symbol}USDT:"]
    for entry in data:
        rate = float(entry.get("fundingRate", 0))
        ts = entry.get("fundingTime", 0)
        rate_pct = rate * 100
        date_str = ""
        if ts:
            try:
                dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
                date_str = dt.strftime("%Y-%m-%d %H:%M UTC")
            except (ValueError, OSError):
                pass
        lines.append(f"  {date_str}: {rate_pct:+.4f}%")
    # Interpret latest
    latest_rate = float(data[-1].get("fundingRate", 0))
    if latest_rate > 0.01:
        interp = "Very high positive — longs overcrowded, potential short squeeze risk"
    elif latest_rate > 0.001:
        interp = "Positive — more longs than shorts, slightly bullish bias"
    elif latest_rate > -0.001:
        interp = "Neutral — balanced market"
    elif latest_rate > -0.01:
        interp = "Negative — more shorts than longs, slightly bearish bias"
    else:
        interp = "Very negative — shorts overcrowded, potential long squeeze"
    lines.append(f"  Interpretation: {interp}")
    return "\n".join(lines)


def get_open_interest(query: str) -> str:
    """Binance futures open interest. FREE, no key."""
    symbol = _extract_crypto_symbol(query)
    url = f"https://fapi.binance.com/fapi/v1/openInterest?symbol={symbol}USDT"
    try:
        data = _http_get(url)
    except RuntimeError as e:
        return f"Open interest not available for {symbol}: {e}"
    oi = float(data.get("openInterest", 0))
    # Also get current price for notional value
    try:
        price_url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}USDT"
        price_data = _http_get(price_url)
        price = float(price_data.get("price", 0))
        notional = oi * price
        price_str = f" (${notional:,.0f} notional at ${price:,.2f})"
    except RuntimeError:
        price_str = ""
    return (f"Open Interest — {symbol}USDT: {oi:,.2f} {symbol}{price_str}\n"
            f"  Rising OI + rising price = strong trend confirmation\n"
            f"  Rising OI + falling price = potential reversal ahead")


def _fetch_ohlc_binance(symbol: str, interval: str = "1d", limit: int = 200):
    """Fetch OHLC from Binance klines API. Supports: 1m,5m,15m,1h,4h,1d,1w."""
    sym = _extract_crypto_symbol(symbol).upper()
    if not sym.endswith("USDT"):
        sym += "USDT"
    url = (f"https://api.binance.com/api/v3/klines"
           f"?symbol={sym}&interval={interval}&limit={limit}")
    try:
        data = _http_get(url, timeout=15)
    except RuntimeError:
        return None
    if not data or not isinstance(data, list) or len(data) < 20:
        return None
    opens = [float(c[1]) for c in data]
    highs = [float(c[2]) for c in data]
    lows = [float(c[3]) for c in data]
    closes = [float(c[4]) for c in data]
    volumes = [float(c[5]) for c in data]
    name = sym.replace("USDT", "")
    return {"opens": opens, "highs": highs, "lows": lows, "closes": closes,
            "volumes": volumes, "name": name}


def _fetch_ohlc_crypto(query: str):
    """Fetch OHLC data from CoinGecko for a crypto asset."""
    coin_id = _extract_coingecko_id(query)
    url = (f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
           f"?vs_currency=usd&days=180")
    data = _http_get(url, timeout=20)
    if not data or not isinstance(data, list) or len(data) < 30:
        return None, coin_id
    highs = [c[2] for c in data]
    lows = [c[3] for c in data]
    closes = [c[4] for c in data]
    return {"highs": highs, "lows": lows, "closes": closes, "name": coin_id.title()}, coin_id


def _fetch_ohlc_yahoo(symbol: str, interval: str = "1d", range_: str = "6mo"):
    """Fetch OHLC data from Yahoo Finance for stocks, commodities, forex."""
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
           f"?interval={interval}&range={range_}")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0.0.0 Safari/537.36",
    }
    data = _http_get(url, headers=headers, timeout=15)
    if not data:
        return None
    result = data.get("chart", {}).get("result", [])
    if not result:
        return None
    quote = result[0].get("indicators", {}).get("quote", [{}])[0]
    opens = [o for o in (quote.get("open") or []) if o is not None]
    highs = [h for h in (quote.get("high") or []) if h is not None]
    lows = [l for l in (quote.get("low") or []) if l is not None]
    closes = [c for c in (quote.get("close") or []) if c is not None]
    volumes = [v for v in (quote.get("volume") or []) if v is not None]
    if len(closes) < 20:
        return None
    name = result[0].get("meta", {}).get("shortName") or symbol
    result_dict = {"highs": highs, "lows": lows, "closes": closes, "name": name}
    if opens:
        result_dict["opens"] = opens
    if volumes:
        result_dict["volumes"] = volumes
    return result_dict


# Yahoo symbols for common commodities and forex
_COMMODITY_YAHOO = {
    "gold": "GC=F", "silver": "SI=F", "platinum": "PL=F",
    "palladium": "PA=F", "crude oil": "CL=F", "crudeoil": "CL=F",
    "oil": "CL=F", "natural gas": "NG=F", "naturalgas": "NG=F",
    "copper": "HG=F", "zinc": "ZN=F", "aluminium": "ALI=F",
    "aluminum": "ALI=F", "nickel": "NI=F", "lead": "PB=F",
    # MCX commodity symbols (Upstox) → Yahoo futures equivalents
    "goldm": "GC=F", "goldguinea": "GC=F",
    "silverm": "SI=F", "silvermic": "SI=F",
    "crudeoilm": "CL=F",
    "cottoncandy": "CT=F", "cotton": "CT=F",
    "menthaoil": "GC=F",  # No direct Yahoo equivalent, use gold as proxy
}
_FOREX_YAHOO = {
    "eur/usd": "EURUSD=X", "gbp/usd": "GBPUSD=X", "usd/jpy": "USDJPY=X",
    "usd/cad": "USDCAD=X", "aud/usd": "AUDUSD=X", "usd/chf": "USDCHF=X",
}


def _format_ta_output(ohlc: dict) -> str:
    """Format technical analysis output from OHLC data."""
    closes = ohlc["closes"]
    highs = ohlc["highs"]
    lows = ohlc["lows"]
    name = ohlc["name"]
    current = closes[-1]

    rsi = _calc_rsi(closes, 14)
    ema21 = _calc_ema(closes, 21)
    ema50 = _calc_ema(closes, 50)
    macd_val, signal_val, hist_val = _calc_macd(closes)
    bb_mid, bb_upper, bb_lower = _calc_bollinger(closes, 20, 2.0)
    atr = _calc_atr(highs, lows, closes, 14)

    rsi_interp = ("oversold — buy signal" if rsi < 30
                  else "overbought — sell signal" if rsi > 70
                  else "neutral")
    ema_interp = ("bullish (price > EMA21 > EMA50)" if current > ema21 > ema50
                  else "bearish (price < EMA21 < EMA50)" if current < ema21 < ema50
                  else "mixed")
    macd_interp = ("bullish crossover" if hist_val > 0
                   else "bearish crossover")
    bb_interp = ("near upper band — potential resistance" if current > bb_mid + (bb_upper - bb_mid) * 0.8
                 else "near lower band — potential support" if current < bb_mid - (bb_mid - bb_lower) * 0.8
                 else "within bands — normal range")
    atr_pct = (atr / current * 100) if current else 0

    lines = [
        f"Technical Analysis — {name} (${current:,.2f}):",
        f"",
        f"  RSI (14): {rsi:.1f} — {rsi_interp}",
        f"  EMA 21: ${ema21:,.2f} | EMA 50: ${ema50:,.2f} — {ema_interp}",
        f"  MACD: {macd_val:.2f} | Signal: {signal_val:.2f} | Hist: {hist_val:.2f} — {macd_interp}",
        f"  Bollinger Bands: Upper ${bb_upper:,.2f} | Mid ${bb_mid:,.2f} | Lower ${bb_lower:,.2f}",
        f"    Current position: {bb_interp}",
        f"  ATR (14): ${atr:,.2f} ({atr_pct:.1f}% of price) — daily volatility measure",
    ]
    return "\n".join(lines)


def get_technical_analysis(query: str) -> str:
    """Technical indicators (RSI, EMA, MACD, Bollinger, ATR) for any asset. FREE."""
    q = query.lower()

    # 1. Try crypto (CoinGecko) — check for known crypto names
    coin_id = _extract_coingecko_id(query)
    is_crypto = any(tok in q for tok in (
        "bitcoin", "btc", "ethereum", "eth", "solana", "sol", "doge",
        "xrp", "cardano", "ada", "bnb", "avax", "matic", "link",
        "atom", "near", "sui", "arb", "pepe", "ltc", "dot",
    ))
    if is_crypto:
        try:
            ohlc, _ = _fetch_ohlc_crypto(query)
            if ohlc:
                return _format_ta_output(ohlc)
        except Exception:
            pass

    # 2. Try commodity (Yahoo Finance futures)
    for name, symbol in _COMMODITY_YAHOO.items():
        if name in q:
            try:
                ohlc = _fetch_ohlc_yahoo(symbol)
                if ohlc:
                    return _format_ta_output(ohlc)
            except Exception:
                pass
            break

    # 3. Try forex (Yahoo Finance)
    for pair, symbol in _FOREX_YAHOO.items():
        if pair in q or pair.replace("/", "") in q:
            try:
                ohlc = _fetch_ohlc_yahoo(symbol)
                if ohlc:
                    return _format_ta_output(ohlc)
            except Exception:
                pass
            break

    # 4. Try stock (Yahoo Finance)
    stock_symbol = _extract_stock_symbol(query)
    if stock_symbol:
        try:
            ohlc = _fetch_ohlc_yahoo(stock_symbol)
            if ohlc:
                return _format_ta_output(ohlc)
        except Exception:
            pass

    # 5. Fallback: try as crypto on CoinGecko (might be an unknown coin name)
    try:
        ohlc, cid = _fetch_ohlc_crypto(query)
        if ohlc:
            return _format_ta_output(ohlc)
        return f"Not enough OHLC data for '{query}' to compute indicators"
    except Exception as e:
        return f"Technical analysis not available for '{query}': {e}"


def _compute_tf_line(ohlc: dict, label: str) -> str:
    """Compute TA indicators for one timeframe and return a formatted line."""
    closes = ohlc["closes"]
    highs = ohlc["highs"]
    lows = ohlc["lows"]
    opens = ohlc.get("opens", closes)
    volumes = ohlc.get("volumes", [])
    current = closes[-1]

    # Core indicators (existing)
    rsi = _calc_rsi(closes, 14)
    ema21 = _calc_ema(closes, 21)
    ema50 = _calc_ema(closes, 50)
    macd_val, signal_val, hist_val = _calc_macd(closes)
    bb_mid, bb_upper, bb_lower = _calc_bollinger(closes, 20, 2.0)
    atr = _calc_atr(highs, lows, closes, 14)
    atr_pct = (atr / current * 100) if current else 0

    # Extended indicators (Trading.md features)
    rsi6 = _calc_rsi(closes, 6)
    ema200 = _calc_ema(closes, 200) if len(closes) >= 200 else _calc_ema(closes, len(closes))
    adx = _calc_adx(highs, lows, closes, 14)

    # Keltner Channel: EMA20 ± 2×ATR
    ema20 = _calc_ema(closes, 20)
    keltner_upper = ema20 + 2 * atr
    keltner_lower = ema20 - 2 * atr
    keltner_range = max(keltner_upper - keltner_lower, 0.01)
    keltner_pos = (current - keltner_lower) / keltner_range * 2 - 1  # -1 to 1

    # Volume-based indicators
    rel_vol = 0.0
    obv_dir = 0.0
    vwap = current
    if volumes and len(volumes) >= 2:
        vol_window = volumes[-20:] if len(volumes) >= 20 else volumes
        avg_vol = sum(vol_window) / len(vol_window) if vol_window else 1.0
        rel_vol = volumes[-1] / avg_vol if avg_vol > 0 else 0.0
        obv_dir = _calc_obv_direction(closes, volumes)
        vwap = _calc_vwap(closes, highs, lows, volumes)

    # Candlestick pattern features (latest candle)
    o, h, l, c = opens[-1], highs[-1], lows[-1], closes[-1]
    candle_range = max(h - l, 0.0001)
    body = abs(c - o)
    body_ratio = body / candle_range
    upper_wick = (h - max(o, c)) / candle_range
    lower_wick = (min(o, c) - l) / candle_range

    rsi_tag = ("oversold" if rsi < 30 else "overbought" if rsi > 70
               else "neutral")
    trend = ("BULLISH" if current > ema21 > ema50
             else "BEARISH" if current < ema21 < ema50 else "MIXED")
    macd_tag = "bullish" if hist_val > 0 else "bearish"

    bb_pos = ("near upper band" if current > bb_mid + (bb_upper - bb_mid) * 0.8
              else "near lower band" if current < bb_mid - (bb_mid - bb_lower) * 0.8
              else "within bands")

    # FVG (Fair Value Gaps)
    fvg_data = _find_fvg(highs, lows, closes, opens, lookback=30)
    fvg_bias = fvg_data["fvg_bias"]

    # Fibonacci retracement levels
    fib_data = _calc_fibonacci_levels(highs, lows, closes, lookback=50)
    fib_bias = fib_data["fib_bias"]

    # Pivot Points (Classic)
    pivot_data = _calc_pivot_points(highs, lows, closes)
    pivot_bias = pivot_data["pivot_bias"]

    # Divergence detection (RSI + MACD + OBV vs price)
    # Build full RSI and MACD histogram series for divergence check
    rsi_series = []
    for i in range(14, len(closes)):
        rsi_series.append(_calc_rsi(closes[:i + 1], 14))
    # Pad front with 50 so indices align with closes[14:]
    rsi_series = [50.0] * 14 + rsi_series

    macd_series = []
    for i in range(26, len(closes)):
        _, _, h_val = _calc_macd(closes[:i + 1])
        macd_series.append(h_val)
    macd_series = [0.0] * 26 + macd_series

    # Build OBV series for volume divergence
    obv_series = [0.0]
    if volumes and len(volumes) >= 2:
        obv_val = 0.0
        for i in range(1, min(len(closes), len(volumes))):
            vol = volumes[i] if i < len(volumes) else 0.0
            if closes[i] > closes[i - 1]:
                obv_val += vol
            elif closes[i] < closes[i - 1]:
                obv_val -= vol
            obv_series.append(obv_val)

    div_data = _detect_divergence(closes, highs, lows,
                                  rsi_series, macd_series,
                                  obv_series=obv_series, lookback=40)

    stats = {
        "rsi": rsi, "ema21": ema21, "ema50": ema50,
        "macd_hist": hist_val, "bb_upper": bb_upper, "bb_lower": bb_lower,
        "atr": atr, "current": current,
        # Extended features
        "rsi6": rsi6, "ema200": ema200, "adx": adx,
        "keltner_pos": max(-1.0, min(1.0, keltner_pos)),
        "rel_vol": min(rel_vol, 5.0), "obv_dir": obv_dir,
        "vwap": vwap, "body_ratio": body_ratio,
        "upper_wick": upper_wick, "lower_wick": lower_wick,
        # FVG + Fibonacci
        "fvg_bias": fvg_bias,
        "fvg_bull_count": len(fvg_data["bullish_fvgs"]),
        "fvg_bear_count": len(fvg_data["bearish_fvgs"]),
        "fvg_nearest_bull": fvg_data["nearest_bullish"],
        "fvg_nearest_bear": fvg_data["nearest_bearish"],
        "fib_bias": fib_bias,
        "fib_direction": fib_data["direction"],
        "fib_levels": fib_data["levels"],
        "fib_nearest_support": fib_data["nearest_support"],
        "fib_nearest_resistance": fib_data["nearest_resistance"],
        "fib_swing_high": fib_data["swing_high"],
        "fib_swing_low": fib_data["swing_low"],
        # Pivot Points
        "pivot_bias": pivot_bias,
        "pp": pivot_data["pp"], "s1": pivot_data["s1"], "s2": pivot_data["s2"],
        "r1": pivot_data["r1"], "r2": pivot_data["r2"],
        # Divergence
        "rsi_div": div_data["rsi_div"], "macd_div": div_data["macd_div"],
        "obv_div": div_data["obv_div"], "div_score": div_data["div_score"],
    }

    # Multi-indicator score for this TF (uses ALL indicators including FVG + Fib)
    tf_score = _score_tf(stats)
    tf_bias = "BULLISH" if tf_score > 10 else "BEARISH" if tf_score < -10 else "NEUTRAL"
    stats["tf_score"] = tf_score

    # FVG text
    fvg_parts = []
    if fvg_data["nearest_bullish"]:
        b = fvg_data["nearest_bullish"]
        fvg_parts.append(f"bull ${b['bottom']:,.2f}-${b['top']:,.2f}")
    if fvg_data["nearest_bearish"]:
        b = fvg_data["nearest_bearish"]
        fvg_parts.append(f"bear ${b['bottom']:,.2f}-${b['top']:,.2f}")
    fvg_str = " | ".join(fvg_parts) if fvg_parts else "none"

    # Fibonacci text
    fib_parts = []
    if fib_data["nearest_support"] > 0:
        fib_parts.append(f"sup ${fib_data['nearest_support']:,.2f}")
    if fib_data["nearest_resistance"] > 0:
        fib_parts.append(f"res ${fib_data['nearest_resistance']:,.2f}")
    fib_str = " | ".join(fib_parts) if fib_parts else "none"

    # Pivot text
    pp = pivot_data["pp"]
    pivot_str = (f"PP=${pp:,.2f} S1=${pivot_data['s1']:,.2f} "
                 f"R1=${pivot_data['r1']:,.2f}") if pp > 0 else "n/a"

    # Divergence text
    div_parts = []
    if div_data["rsi_div"] != "none":
        div_parts.append(f"RSI {div_data['rsi_div']}")
    if div_data["macd_div"] != "none":
        div_parts.append(f"MACD {div_data['macd_div']}")
    if div_data["obv_div"] != "none":
        div_parts.append(f"OBV {div_data['obv_div']}")
    div_str = " + ".join(div_parts) if div_parts else "none"

    lines = [
        f"  {label}:",
        f"    RSI: {rsi:.1f} — {rsi_tag} | EMA21: ${ema21:,.2f} | "
        f"EMA50: ${ema50:,.2f} | Trend: {trend}",
        f"    MACD: {macd_tag} (H: {hist_val:+.2f}) | BB: {bb_pos} | "
        f"ATR: ${atr:,.2f} ({atr_pct:.1f}%)",
        f"    Pivot: {pivot_str} | FVG: {fvg_str} | Fib: {fib_str}",
        f"    Divergence: {div_str} | Score: {tf_score:+d} ({tf_bias})",
    ]
    return "\n".join(lines), stats


def get_multi_timeframe_analysis(query: str) -> str:
    """Multi-timeframe technical analysis with trade levels for any asset. FREE."""
    q = query.lower()

    # Detect asset type
    is_crypto = any(tok in q for tok in (
        "bitcoin", "btc", "ethereum", "eth", "solana", "sol", "doge",
        "xrp", "cardano", "ada", "bnb", "avax", "matic", "link",
        "atom", "near", "sui", "arb", "pepe", "ltc", "dot",
    ))

    # Commodity / forex / stock detection
    commodity_sym = None
    for name, sym in _COMMODITY_YAHOO.items():
        if name in q:
            commodity_sym = sym
            break
    forex_sym = None
    for pair, sym in _FOREX_YAHOO.items():
        if pair in q or pair.replace("/", "") in q:
            forex_sym = sym
            break
    stock_sym = None
    if not is_crypto and not commodity_sym and not forex_sym:
        stock_sym = _extract_stock_symbol(query)

    # ── Fetch multi-timeframe OHLC data ──
    tf_data = {}  # label -> ohlc dict

    if is_crypto:
        # Binance klines for crypto — all timeframes available
        crypto_sym = _extract_crypto_symbol(query)
        tf_map = [
            ("WEEKLY (Major Trend)",  "1w",  100),
            ("DAILY (Swing)",         "1d",  200),
            ("4-HOUR (Intraday)",     "4h",  200),
            ("1-HOUR (Entry Timing)", "1h",  200),
            ("30-MIN (Scalp+)",       "30m", 200),
            ("15-MIN (Scalp)",        "15m", 200),
            ("5-MIN (Precision)",     "5m",  200),
        ]
        for label, interval, limit in tf_map:
            ohlc = _fetch_ohlc_binance(crypto_sym, interval, limit)
            if ohlc:
                tf_data[label] = ohlc
        asset_name = crypto_sym
    else:
        # Yahoo Finance for stocks/commodities/forex
        yahoo_sym = commodity_sym or forex_sym or stock_sym
        if not yahoo_sym:
            # Fallback to crypto via Binance
            crypto_sym = _extract_crypto_symbol(query)
            ohlc = _fetch_ohlc_binance(crypto_sym, "1d", 200)
            if ohlc:
                tf_data["DAILY (Swing)"] = ohlc
            asset_name = crypto_sym
        else:
            tf_map_yahoo = [
                ("WEEKLY (Major Trend)",  "1wk", "2y"),
                ("DAILY (Swing)",         "1d",  "6mo"),
                ("1-HOUR (Intraday)",     "60m", "5d"),
                ("15-MIN (Scalp)",        "15m", "1d"),
                ("5-MIN (Precision)",     "5m",  "1d"),
            ]
            for label, interval, range_ in tf_map_yahoo:
                ohlc = _fetch_ohlc_yahoo(yahoo_sym, interval, range_)
                if ohlc:
                    tf_data[label] = ohlc
            asset_name = yahoo_sym

    if not tf_data:
        return f"Multi-timeframe analysis not available for '{query}'"

    # Get the daily (or best available) for trade levels
    daily_key = None
    for k in tf_data:
        if "DAILY" in k:
            daily_key = k
            break
    if not daily_key:
        daily_key = next(iter(tf_data))
    daily_ohlc = tf_data[daily_key]
    current = daily_ohlc["closes"][-1]

    # ── Build output ──
    out = [f"Multi-Timeframe Analysis — {asset_name} (${current:,.2f})", ""]

    all_tf_stats = {}  # label -> stats dict
    daily_stats = None
    for label in tf_data:
        ohlc = tf_data[label]
        line_text, stats = _compute_tf_line(ohlc, label)
        out.append(line_text)
        all_tf_stats[label] = stats
        if label == daily_key:
            daily_stats = stats

    # Support / Resistance from daily
    supports, resistances = _find_support_resistance(
        daily_ohlc["highs"], daily_ohlc["lows"], daily_ohlc["closes"])

    # FVG and Fibonacci from daily TF (used for SL/TP refinement)
    daily_fvg = _find_fvg(daily_ohlc["highs"], daily_ohlc["lows"],
                          daily_ohlc["closes"],
                          daily_ohlc.get("opens", daily_ohlc["closes"]),
                          lookback=30)
    daily_fib = _calc_fibonacci_levels(daily_ohlc["highs"], daily_ohlc["lows"],
                                       daily_ohlc["closes"], lookback=50)

    # Pivot Points from daily TF
    daily_pivot = _calc_pivot_points(daily_ohlc["highs"], daily_ohlc["lows"],
                                     daily_ohlc["closes"])

    out.append("")
    out.append("KEY LEVELS:")
    res_str = " | ".join(f"${r:,.2f}" for r in resistances) if resistances else "none detected"
    sup_str = " | ".join(f"${s:,.2f}" for s in supports) if supports else "none detected"
    out.append(f"  Resistance: {res_str}")
    out.append(f"  Support: {sup_str}")

    # Pivot Points
    if daily_pivot["pp"] > 0:
        pp = daily_pivot
        out.append(f"  Pivot: PP=${pp['pp']:,.2f} | "
                   f"S1=${pp['s1']:,.2f} S2=${pp['s2']:,.2f} S3=${pp['s3']:,.2f} | "
                   f"R1=${pp['r1']:,.2f} R2=${pp['r2']:,.2f} R3=${pp['r3']:,.2f}")

    # Fibonacci levels
    if daily_fib["levels"]:
        fib_strs = []
        for ratio, price_level in sorted(daily_fib["levels"].items()):
            fib_strs.append(f"{ratio*100:.1f}%=${price_level:,.2f}")
        out.append(f"  Fibonacci ({daily_fib['direction']}): {' | '.join(fib_strs)}")
        if daily_fib["nearest_support"] > 0:
            out.append(f"  Fib Support: ${daily_fib['nearest_support']:,.2f}")
        if daily_fib["nearest_resistance"] > 0:
            out.append(f"  Fib Resistance: ${daily_fib['nearest_resistance']:,.2f}")

    # FVG gaps
    fvg_lines = []
    if daily_fvg["nearest_bullish"]:
        b = daily_fvg["nearest_bullish"]
        fvg_lines.append(f"bull gap ${b['bottom']:,.2f}-${b['top']:,.2f}")
    if daily_fvg["nearest_bearish"]:
        b = daily_fvg["nearest_bearish"]
        fvg_lines.append(f"bear gap ${b['bottom']:,.2f}-${b['top']:,.2f}")
    if fvg_lines:
        out.append(f"  FVG: {' | '.join(fvg_lines)}")

    # Pick the best TF for trade levels: prefer 1H > 4H > 30M > 15M > daily
    # Short TF gives tighter, more actionable SL/TP levels
    trade_stats = daily_stats
    for pref_label in ["1-HOUR (Entry Timing)", "4-HOUR (Intraday)",
                        "30-MIN (Scalp+)", "15-MIN (Scalp)"]:
        if pref_label in all_tf_stats:
            trade_stats = all_tf_stats[pref_label]
            break

    # Trading strategy from best available TF stats
    if trade_stats:
        levels = _calc_trade_levels(
            trade_stats["current"], trade_stats["atr"],
            trade_stats["ema21"], trade_stats["ema50"], trade_stats["rsi"],
            trade_stats["bb_upper"], trade_stats["bb_lower"],
            supports, resistances,
            fib_data=daily_fib, fvg_data=daily_fvg)

        # Determine multi-TF confluence using full indicator scoring
        # Each TF is scored -100..+100 using RSI, EMA, MACD, BB, ADX, OBV, EMA200
        bullish_tfs = 0
        bearish_tfs = 0
        total_score = 0
        for label in all_tf_stats:
            s = all_tf_stats[label].get("tf_score", 0)
            total_score += s
            if s > 10:
                bullish_tfs += 1
            elif s < -10:
                bearish_tfs += 1
            # else: neutral TF — doesn't count for either side
        total_tfs = len(all_tf_stats)
        confluence = (f"{bullish_tfs}/{total_tfs} TFs bullish"
                      if bullish_tfs > bearish_tfs
                      else f"{bearish_tfs}/{total_tfs} TFs bearish"
                      if bearish_tfs > bullish_tfs
                      else "mixed across TFs")

        out.append("")
        out.append("TRADING STRATEGY:")
        out.append(f"  Bias: {levels['bias']} ({confluence}, score: {total_score:+d})")
        out.append(f"  Entry: ${levels['entry']:,.2f}")
        out.append(f"  Stop Loss: ${levels['stop_loss']:,.2f}")
        out.append(f"  Take Profit 1: ${levels['tp1']:,.2f} — "
                   f"R:R {levels['risk_reward']:.1f}:1")
        out.append(f"  Take Profit 2: ${levels['tp2']:,.2f}")

        # --- AI features line (structured data for CORTEX 20-dim vector) ---
        funding_rate = 0.0
        oi_change_pct = 0.0
        if is_crypto:
            # Fetch funding rate (Binance perpetuals)
            try:
                fr_url = (f"https://fapi.binance.com/fapi/v1/fundingRate"
                          f"?symbol={asset_name}USDT&limit=2")
                fr_data = _http_get(fr_url, timeout=10)
                if fr_data and len(fr_data) >= 1:
                    funding_rate = float(fr_data[-1].get("fundingRate", 0))
            except Exception:
                pass
            # Fetch open interest (current vs previous for % change)
            try:
                oi_url = (f"https://fapi.binance.com/fapi/v1/openInterest"
                          f"?symbol={asset_name}USDT")
                oi_data = _http_get(oi_url, timeout=10)
                if oi_data:
                    oi_current = float(oi_data.get("openInterest", 0))
                    # Use daily volume ratio as proxy for OI change direction
                    if daily_ohlc.get("volumes") and len(daily_ohlc["volumes"]) >= 2:
                        vol_now = daily_ohlc["volumes"][-1]
                        vol_prev = daily_ohlc["volumes"][-2]
                        if vol_prev > 0:
                            oi_change_pct = (vol_now - vol_prev) / vol_prev
                            oi_change_pct = max(-0.5, min(0.5, oi_change_pct))
            except Exception:
                pass

        ds = daily_stats
        feat_parts = [
            f"rsi6={ds.get('rsi6', 50.0):.2f}",
            f"ema200={ds.get('ema200', 0.0):.2f}",
            f"adx={ds.get('adx', 25.0):.2f}",
            f"keltner_pos={ds.get('keltner_pos', 0.0):.4f}",
            f"rel_vol={ds.get('rel_vol', 0.0):.4f}",
            f"obv_dir={ds.get('obv_dir', 0.0):.4f}",
            f"vwap={ds.get('vwap', 0.0):.2f}",
            f"body_ratio={ds.get('body_ratio', 0.5):.4f}",
            f"upper_wick={ds.get('upper_wick', 0.25):.4f}",
            f"lower_wick={ds.get('lower_wick', 0.25):.4f}",
            f"funding_rate={funding_rate:.6f}",
            f"oi_change={oi_change_pct:.4f}",
            f"fvg_bias={daily_fvg.get('fvg_bias', 0)}",
            f"fib_bias={daily_fib.get('fib_bias', 0)}",
            f"fib_sup={daily_fib.get('nearest_support', 0.0):.2f}",
            f"fib_res={daily_fib.get('nearest_resistance', 0.0):.2f}",
            f"pivot_bias={daily_pivot.get('pivot_bias', 0)}",
            f"pp={daily_pivot.get('pp', 0.0):.2f}",
            f"pivot_s1={daily_pivot.get('s1', 0.0):.2f}",
            f"pivot_r1={daily_pivot.get('r1', 0.0):.2f}",
            f"pivot_s2={daily_pivot.get('s2', 0.0):.2f}",
            f"pivot_r2={daily_pivot.get('r2', 0.0):.2f}",
            f"div_score={ds.get('div_score', 0)}",
            f"confluence_score={total_score}",
        ]
        out.append("")
        out.append(f"AI_FEATURES: {'|'.join(feat_parts)}")

        # --- CORTEX Brain analysis (3 modes: scalp / stable / secure) ---
        if _shared_cortex and _shared_cortex._brain._trained:
            try:
                # Build analysis dict matching parse_analysis format
                cortex_analysis = {
                    "current": current, "rsi": ds.get("rsi", 50.0),
                    "rsi6": ds.get("rsi6", 50.0),
                    "ema21": ds.get("ema21", current),
                    "ema50": ds.get("ema50", current),
                    "ema200": ds.get("ema200", current),
                    "atr_daily": ds.get("atr", 0.0),
                    "atr": ds.get("atr", 0.0),
                    "macd_hist": ds.get("macd_hist", 0.0),
                    "bb_upper": ds.get("bb_upper", current * 1.02),
                    "bb_lower": ds.get("bb_lower", current * 0.98),
                    "adx": ds.get("adx", 25.0),
                    "keltner_pos": ds.get("keltner_pos", 0.0),
                    "rel_vol": ds.get("rel_vol", 0.0),
                    "obv_dir": ds.get("obv_dir", 0.0),
                    "vwap": ds.get("vwap", current),
                    "body_ratio": ds.get("body_ratio", 0.5),
                    "upper_wick": ds.get("upper_wick", 0.25),
                    "lower_wick": ds.get("lower_wick", 0.25),
                    "funding_rate": funding_rate,
                    "oi_change": oi_change_pct,
                    "fear_greed_value": 50,  # neutral default (tool doesn't fetch)
                    "poly_up_prob": 0.5,
                    "total_tfs": total_tfs,
                }
                # Per-TF scores for recount_confluence
                _tf_label_to_key = {
                    "5-MIN (Precision)": "score_5m",
                    "15-MIN (Scalp)": "score_15m",
                    "30-MIN (Scalp+)": "score_30m",
                    "1-HOUR (Entry Timing)": "score_1h",
                    "4-HOUR (Intraday)": "score_4h",
                    "DAILY (Swing)": "score_daily",
                    "WEEKLY (Major Trend)": "score_weekly",
                }
                for label, stats_d in all_tf_stats.items():
                    sk = _tf_label_to_key.get(label)
                    if sk:
                        cortex_analysis[sk] = stats_d.get("tf_score", 0)

                from qor.trading import recount_confluence, TRIPLE_SCREEN
                out.append("")
                out.append("CORTEX BRAIN:")
                for mode in ("scalp", "stable", "secure"):
                    # Deep-copy analysis for each mode
                    mode_analysis = dict(cortex_analysis)
                    # Recount confluence using mode-specific TFs
                    mode_analysis["bullish_tfs"] = bullish_tfs
                    mode_analysis["bearish_tfs"] = bearish_tfs
                    recount_confluence(mode_analysis, mode)
                    # Run CORTEX
                    instance = f"{asset_name}_{mode}"
                    result = _shared_cortex.analyze(mode_analysis, instance)
                    sig = result["signal"]
                    conf = result["confidence"]
                    label = result["label"]
                    b = mode_analysis.get("bullish_tfs", 0)
                    t = mode_analysis.get("total_tfs", 1) or 1
                    veto = mode_analysis.get("trend_veto", "NEUTRAL")
                    out.append(
                        f"  {mode:7s}: {label:12s} "
                        f"(signal={sig:+.3f}, conf={conf:.0%}) "
                        f"[{b}/{t} TFs bullish, veto={veto}]")
            except Exception as e:
                out.append(f"  CORTEX: error — {e}")

    return "\n".join(out)


def get_economic_calendar(query: str) -> str:
    """Upcoming economic events from ForexFactory. FREE, no key."""
    url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    try:
        data = _http_get(url, timeout=20)
    except RuntimeError:
        return "Economic calendar not available"
    if not data or not isinstance(data, list):
        return "No economic events data available"
    # Filter high-impact events by default
    q_lower = query.lower()
    show_all = "all" in q_lower or "medium" in q_lower or "low" in q_lower
    events = []
    for ev in data:
        impact = ev.get("impact", "").lower()
        if not show_all and impact not in ("high", "holiday"):
            continue
        events.append(ev)
    if not events:
        # If no high-impact, show top medium
        events = [ev for ev in data if ev.get("impact", "").lower() in ("high", "medium")][:10]
    if not events:
        return "No high-impact economic events this week"
    lines = ["Economic Calendar (This Week — High Impact):"]
    for ev in events[:12]:
        title = ev.get("title", "?")
        impact = ev.get("impact", "?")
        country = ev.get("country", "?")
        date = ev.get("date", "?")
        forecast = ev.get("forecast", "")
        previous = ev.get("previous", "")
        # Format date shorter
        if isinstance(date, str) and len(date) > 10:
            date = date[:16]
        line = f"  [{impact.upper()}] {date} — {country}: {title}"
        if forecast or previous:
            details = []
            if forecast:
                details.append(f"Forecast: {forecast}")
            if previous:
                details.append(f"Previous: {previous}")
            line += f" ({', '.join(details)})"
        lines.append(line)
    return "\n".join(lines)


def get_crypto_trending(query: str) -> str:
    """Trending cryptocurrencies from CoinGecko. FREE, no key."""
    url = "https://api.coingecko.com/api/v3/search/trending"
    data = _http_get(url)
    coins = data.get("coins", [])
    if not coins:
        return "No trending data available"
    lines = ["Trending Cryptocurrencies (CoinGecko):"]
    for i, entry in enumerate(coins[:7]):
        item = entry.get("item", {})
        name = item.get("name", "?")
        symbol = item.get("symbol", "?")
        rank = item.get("market_cap_rank", "?")
        price_btc = item.get("price_btc", 0)
        # Some versions have score
        score = item.get("score", i)
        lines.append(f"  {i + 1}. {name} ({symbol}) — Market Cap Rank: #{rank}")
    return "\n".join(lines)


def get_global_market(query: str) -> str:
    """Global crypto market overview from CoinGecko. FREE, no key."""
    url = "https://api.coingecko.com/api/v3/global"
    data = _http_get(url)
    d = data.get("data", {})
    if not d:
        return "Global market data not available"
    total_mcap = d.get("total_market_cap", {}).get("usd", 0)
    total_vol = d.get("total_volume", {}).get("usd", 0)
    btc_dom = d.get("market_cap_percentage", {}).get("btc", 0)
    eth_dom = d.get("market_cap_percentage", {}).get("eth", 0)
    active = d.get("active_cryptocurrencies", 0)
    mcap_change = d.get("market_cap_change_percentage_24h_usd", 0)
    direction = "up" if mcap_change > 0 else "down"
    lines = [
        "Global Crypto Market Overview:",
        f"  Total Market Cap: ${total_mcap:,.0f} ({direction} {abs(mcap_change):.1f}% 24h)",
        f"  24h Volume: ${total_vol:,.0f}",
        f"  BTC Dominance: {btc_dom:.1f}%",
        f"  ETH Dominance: {eth_dom:.1f}%",
        f"  Active Cryptocurrencies: {active:,}",
    ]
    # Add top 5 dominance breakdown
    mcap_pct = d.get("market_cap_percentage", {})
    top_coins = sorted(mcap_pct.items(), key=lambda x: x[1], reverse=True)[:5]
    if top_coins:
        lines.append("  Dominance Breakdown:")
        for coin, pct in top_coins:
            lines.append(f"    {coin.upper()}: {pct:.1f}%")
    return "\n".join(lines)


# ==============================================================================
# FINANCIAL MARKETS (FREE APIs, no keys)
# ==============================================================================

def get_commodities(query: str) -> str:
    """Precious metals & commodity prices. FREE, no key."""
    url = "https://data-asg.goldprice.org/dbXRates/USD"
    try:
        data = _http_get(url, timeout=15)
    except RuntimeError:
        return "Commodity price data not available"
    items = data.get("items", [])
    if not items:
        return "No commodity data available"
    lines = ["Commodity Prices (USD):"]
    # goldprice.org returns items with xauPrice, xagPrice, etc.
    for item in items:
        date = item.get("date", "")
        # Gold (XAU)
        gold = item.get("xauPrice")
        if gold and isinstance(gold, (int, float)):
            lines.append(f"  Gold (XAU): ${gold:,.2f}/oz")
        # Silver (XAG)
        silver = item.get("xagPrice")
        if silver and isinstance(silver, (int, float)):
            lines.append(f"  Silver (XAG): ${silver:,.2f}/oz")
        # Platinum (XPT)
        plat = item.get("xptPrice")
        if plat and isinstance(plat, (int, float)):
            lines.append(f"  Platinum (XPT): ${plat:,.2f}/oz")
        # Palladium (XPD)
        pall = item.get("xpdPrice")
        if pall and isinstance(pall, (int, float)):
            lines.append(f"  Palladium (XPD): ${pall:,.2f}/oz")
        # Copper if present
        copper = item.get("xcuPrice")
        if copper and isinstance(copper, (int, float)):
            lines.append(f"  Copper (XCU): ${copper:,.2f}/lb")
        if date:
            lines.append(f"  Updated: {date}")
        break  # Only first item (latest)
    if len(lines) == 1:
        return "No commodity prices available"
    return "\n".join(lines)


def get_stock_quote(query: str) -> str:
    """Stock price quote via Yahoo Finance. FREE, no key."""
    symbol = _extract_stock_symbol(query)
    if not symbol:
        return ("Specify a stock ticker or company name. "
                "Examples: 'stock AAPL', 'apple stock price', 'stock MSFT'")
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
           f"?interval=1d&range=5d")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0.0.0 Safari/537.36",
    }
    try:
        data = _http_get(url, headers=headers, timeout=15)
    except RuntimeError as e:
        return f"Stock data not available for {symbol}: {e}"
    chart = data.get("chart", {})
    results = chart.get("result", [])
    if not results:
        err = chart.get("error", {})
        desc = err.get("description", "Symbol not found")
        return f"No stock data for {symbol}: {desc}"
    meta = results[0].get("meta", {})
    price = meta.get("regularMarketPrice", 0)
    prev_close = meta.get("previousClose", meta.get("chartPreviousClose", 0))
    currency = meta.get("currency", "USD")
    name = meta.get("shortName", meta.get("longName", symbol))
    exchange = meta.get("exchangeName", "?")
    mkt_time = meta.get("regularMarketTime", 0)
    # Calculate change
    change = price - prev_close if prev_close else 0
    change_pct = (change / prev_close * 100) if prev_close else 0
    direction = "+" if change >= 0 else ""
    # Format market time
    time_str = ""
    if mkt_time:
        try:
            dt = datetime.fromtimestamp(mkt_time, tz=timezone.utc)
            time_str = f" (as of {dt.strftime('%Y-%m-%d %H:%M UTC')})"
        except (ValueError, OSError):
            pass
    # Get volume from indicators if available
    indicators = results[0].get("indicators", {})
    quotes = indicators.get("quote", [{}])
    volumes = quotes[0].get("volume", []) if quotes else []
    last_vol = None
    for v in reversed(volumes):
        if v is not None:
            last_vol = v
            break
    vol_str = f" | Volume: {last_vol:,.0f}" if last_vol else ""
    return (f"{name} ({symbol}) — {exchange}\n"
            f"  Price: {price:,.2f} {currency} ({direction}{change:,.2f}, "
            f"{direction}{change_pct:.2f}%){time_str}\n"
            f"  Previous Close: {prev_close:,.2f}{vol_str}")


def get_forex_rates(query: str) -> str:
    """Major forex currency rates via Frankfurter. FREE, no key."""
    base, target = _extract_forex_base(query)
    if target:
        # Specific pair
        url = f"https://api.frankfurter.app/latest?from={base}&to={target}"
        data = _http_get(url)
        rates = data.get("rates", {})
        if target in rates:
            rate = rates[target]
            inverse = 1.0 / rate if rate else 0
            return (f"Forex {base}/{target}: {rate:.4f}\n"
                    f"  1 {base} = {rate:.4f} {target}\n"
                    f"  1 {target} = {inverse:.4f} {base}\n"
                    f"  Date: {data.get('date', '?')}")
        return f"Rate not available for {base}/{target}"
    # Show major pairs from base currency
    major = "EUR,GBP,JPY,CHF,CAD,AUD,CNY,INR,BRL,KRW"
    if base != "USD":
        major = "USD," + major.replace(f"{base},", "").replace(f",{base}", "")
    url = f"https://api.frankfurter.app/latest?from={base}&to={major}"
    data = _http_get(url)
    rates = data.get("rates", {})
    if not rates:
        return f"Forex rates not available for {base}"
    lines = [f"Forex Rates from {base} ({data.get('date', '?')}):"]
    for curr, rate in sorted(rates.items()):
        lines.append(f"  {base}/{curr}: {rate:.4f}")
    return "\n".join(lines)


def get_world_economy(query: str) -> str:
    """Top world economies by GDP from World Bank. FREE, no key."""
    # GDP (current US$) — indicator NY.GDP.MKTP.CD
    # mrv=1 = most recent value, per_page=300 to get all entries
    url = ("https://api.worldbank.org/v2/country/all/indicator/NY.GDP.MKTP.CD"
           "?format=json&per_page=300&mrv=1")
    try:
        data = _http_get(url, timeout=20)
    except RuntimeError:
        return "World economy data not available (World Bank API unreachable)"
    if not isinstance(data, list) or len(data) < 2:
        return "No world economy data available"
    entries = data[1] if len(data) > 1 else []
    if not entries:
        return "No GDP data available"
    # World Bank uses 2-letter codes for BOTH countries AND aggregates
    # Aggregates use special codes — filter them by known set
    aggregate_ids = {
        "1A", "1W", "4E", "7E", "8S", "B8", "EU", "F1", "OE", "S1",
        "S2", "S3", "S4", "T2", "T3", "T4", "T5", "T6", "T7",
        "V1", "V2", "V3", "V4", "XC", "XD", "XE", "XF", "XG", "XH",
        "XI", "XJ", "XL", "XM", "XN", "XO", "XP", "XQ", "XR", "XS",
        "XT", "XU", "XY", "Z4", "Z7", "ZB", "ZF", "ZG", "ZH", "ZI",
        "ZJ", "ZQ", "ZT",
    }
    countries = []
    for entry in entries:
        country = entry.get("country", {})
        code = country.get("id", "")
        if code in aggregate_ids:
            continue
        name = country.get("value", "?")
        gdp = entry.get("value")
        year = entry.get("date", "?")
        if gdp and isinstance(gdp, (int, float)):
            countries.append((name, gdp, year, code))
    if not countries:
        return "No country GDP data available"
    # Sort by GDP descending
    countries.sort(key=lambda x: x[1], reverse=True)
    # Show top 15
    lines = [f"Top World Economies by GDP (USD, {countries[0][2]}):"]
    for i, (name, gdp, year, code) in enumerate(countries[:15]):
        if gdp >= 1e12:
            gdp_str = f"${gdp / 1e12:.2f} trillion"
        elif gdp >= 1e9:
            gdp_str = f"${gdp / 1e9:.1f} billion"
        else:
            gdp_str = f"${gdp:,.0f}"
        lines.append(f"  {i + 1}. {name}: {gdp_str}")
    return "\n".join(lines)


def get_market_indices(query: str) -> str:
    """Major global stock market indices via Yahoo Finance. FREE, no key."""
    indices = {
        "^GSPC": "S&P 500",
        "^DJI": "Dow Jones",
        "^IXIC": "NASDAQ",
        "^FTSE": "FTSE 100",
        "^N225": "Nikkei 225",
        "^HSI": "Hang Seng",
        "^GDAXI": "DAX",
        "^FCHI": "CAC 40",
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0.0.0 Safari/537.36",
    }
    lines = ["Global Market Indices:"]
    for symbol, name in indices.items():
        try:
            url = (f"https://query1.finance.yahoo.com/v8/finance/chart/"
                   f"{urllib.parse.quote(symbol)}?interval=1d&range=2d")
            data = _http_get(url, headers=headers, timeout=10)
            meta = data.get("chart", {}).get("result", [{}])[0].get("meta", {})
            price = meta.get("regularMarketPrice", 0)
            prev = meta.get("previousClose", meta.get("chartPreviousClose", 0))
            if price and prev:
                change_pct = (price - prev) / prev * 100
                direction = "+" if change_pct >= 0 else ""
                lines.append(f"  {name}: {price:,.2f} ({direction}{change_pct:.2f}%)")
            elif price:
                lines.append(f"  {name}: {price:,.2f}")
        except (RuntimeError, KeyError, IndexError):
            continue
    if len(lines) == 1:
        return "Market indices data not available"
    return "\n".join(lines)


# ==============================================================================
# HISTORICAL KNOWLEDGE (FREE APIs, no keys)
# ==============================================================================

def wikidata_search(query: str) -> str:
    """Search Wikidata for structured facts about anything. FREE, no key.
    Covers: people, events, places, companies, discoveries — 100M+ items."""
    q = _extract_query_text(query)
    # Step 1: Search for entity
    url = (f"https://www.wikidata.org/w/api.php"
           f"?action=wbsearchentities&search={urllib.parse.quote(q)}"
           f"&language=en&format=json&limit=3")
    data = _http_get(url)
    results = data.get("search", [])
    if not results:
        return f"No Wikidata results for: {q}"

    # Step 2: Get details for top result
    entity_id = results[0]["id"]
    desc = results[0].get("description", "")
    label = results[0].get("label", q)

    # Get key properties
    detail_url = (f"https://www.wikidata.org/w/api.php"
                  f"?action=wbgetentities&ids={entity_id}"
                  f"&props=claims|descriptions&languages=en&format=json")
    details = _http_get(detail_url)
    entity = details.get("entities", {}).get(entity_id, {})
    claims = entity.get("claims", {})

    facts = [f"{label}: {desc}"]

    # Extract common properties
    prop_names = {
        "P569": "Born", "P570": "Died", "P580": "Start date",
        "P582": "End date", "P17": "Country", "P131": "Located in",
        "P27": "Nationality", "P106": "Occupation", "P31": "Type",
        "P585": "Date", "P1566": "GeoNames ID",
    }
    for prop_id, prop_label in prop_names.items():
        if prop_id in claims:
            val = claims[prop_id][0].get("mainsnak", {}).get("datavalue", {})
            val_type = val.get("type", "")
            if val_type == "time":
                time_str = val.get("value", {}).get("time", "")
                if time_str:
                    facts.append(f"  {prop_label}: {time_str[1:11]}")  # +YYYY-MM-DD → YYYY-MM-DD
            elif val_type == "wikibase-entityid":
                ref_id = val.get("value", {}).get("id", "")
                facts.append(f"  {prop_label}: [{ref_id}]")
            elif val_type == "string":
                facts.append(f"  {prop_label}: {val.get('value', '')}")

    # Also show other search results
    if len(results) > 1:
        facts.append("Related:")
        for r in results[1:]:
            facts.append(f"  - {r.get('label', '?')}: {r.get('description', '')}")

    return "\n".join(facts)


def on_this_day(query: str) -> str:
    """Get historical events that happened on a specific date. FREE, no key.
    Uses Wikipedia's On This Day API."""
    # Extract month/day from query
    now = datetime.now(timezone.utc)
    month, day = now.month, now.day

    # Try to parse date from query
    date_match = re.search(r'(\w+)\s+(\d{1,2})', query)
    if date_match:
        months = {"january": 1, "february": 2, "march": 3, "april": 4,
                  "may": 5, "june": 6, "july": 7, "august": 8,
                  "september": 9, "october": 10, "november": 11, "december": 12,
                  "jan": 1, "feb": 2, "mar": 3, "apr": 4,
                  "jun": 6, "jul": 7, "aug": 8, "sep": 9,
                  "oct": 10, "nov": 11, "dec": 12}
        m_name = date_match.group(1).lower()
        if m_name in months:
            month = months[m_name]
            day = int(date_match.group(2))
    else:
        # Try MM-DD or DD/MM
        num_match = re.search(r'(\d{1,2})[/-](\d{1,2})', query)
        if num_match:
            month, day = int(num_match.group(1)), int(num_match.group(2))

    url = (f"https://api.wikimedia.org/feed/v1/wikipedia/en/onthisday/all"
           f"/{month:02d}/{day:02d}")
    data = _http_get(url, headers={"User-Agent": "QOR-AI-Agent/1.0 (qora.ai)"})

    events = data.get("events", [])[:7]
    births = data.get("births", [])[:3]
    deaths = data.get("deaths", [])[:3]

    lines = [f"On This Day ({month}/{day}):"]

    if events:
        lines.append("Events:")
        for e in events:
            lines.append(f"  {e.get('year', '?')}: {e.get('text', '?')}")

    if births:
        lines.append("Born:")
        for b in births:
            lines.append(f"  {b.get('year', '?')}: {b.get('text', '?')}")

    if deaths:
        lines.append("Died:")
        for d in deaths:
            lines.append(f"  {d.get('year', '?')}: {d.get('text', '?')}")

    return "\n".join(lines) if len(lines) > 1 else f"No events found for {month}/{day}"


def gdelt_search(query: str) -> str:
    """Search GDELT for global news events since 1979. FREE, no key.
    250M+ events from 65 languages — the world's largest news event database."""
    q = _extract_query_text(query)
    url = (f"https://api.gdeltproject.org/api/v2/doc/doc"
           f"?query={urllib.parse.quote(q)}"
           f"&mode=ArtList&maxrecords=7&format=json")
    try:
        data = _http_get(url, timeout=20)
        articles = data.get("articles", [])
        if not articles:
            return f"No GDELT results for: {q}"
        lines = [f"GDELT News Events ({len(articles)} results):"]
        for a in articles[:7]:
            title = a.get("title", "?")
            date = a.get("seendate", "?")[:10]
            source = a.get("domain", "?")
            lines.append(f"  [{date}] {title} ({source})")
        return "\n".join(lines)
    except RuntimeError:
        return f"GDELT search failed for: {q}"


# ==============================================================================
# KNOWLEDGE & SEARCH
# ==============================================================================

def wikipedia_search(query: str) -> str:
    """Search Wikipedia. FREE, no key needed."""
    q = _extract_query_text(query)
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(q)}"
    try:
        data = _http_get(url)
        title = data.get("title", "")
        extract = data.get("extract", "")
        if len(extract) > 600:
            extract = extract[:600] + "..."
        return f"{title}: {extract}"
    except RuntimeError:
        return f"Wikipedia article not found: {q}"


def duckduckgo_search(query: str) -> str:
    """Instant answers from DuckDuckGo. FREE, no key needed."""
    q = _extract_query_text(query)
    url = (f"https://api.duckduckgo.com/?q={urllib.parse.quote(q)}"
           f"&format=json&no_html=1&skip_disambig=1")
    data = _http_get(url)
    abstract = data.get("AbstractText", "")
    if abstract:
        return f"{data.get('Heading', '')}: {abstract}"
    # Fall back to related topics
    topics = data.get("RelatedTopics", [])
    lines = []
    for t in topics[:5]:
        if isinstance(t, dict) and "Text" in t:
            lines.append(f"- {t['Text']}")
    return "\n".join(lines) if lines else f"No results for: {q}"


def web_search(query: str) -> str:
    """Web search via DuckDuckGo python library (if installed) or API."""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if results:
                return "\n".join(
                    f"- {r['title']}: {r['body']}" for r in results
                )
    except (ImportError, Exception):
        pass
    # Fallback to instant answer API
    return duckduckgo_search(query)


def get_definition(query: str) -> str:
    """Word definition from Free Dictionary API."""
    word = re.sub(r'.*(define|meaning of|definition of)\s*', '', query,
                  flags=re.IGNORECASE).strip().split()[0].strip("?.,!")
    url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{urllib.parse.quote(word)}"
    try:
        data = _http_get(url)
        if isinstance(data, list) and data:
            entry = data[0]
            meanings = entry.get("meanings", [])
            lines = [f"{word}:"]
            for m in meanings[:2]:
                pos = m.get("partOfSpeech", "")
                defs = m.get("definitions", [])
                if defs:
                    lines.append(f"  ({pos}) {defs[0].get('definition', '')}")
                    if "example" in defs[0]:
                        lines.append(f"    Example: {defs[0]['example']}")
            return "\n".join(lines)
    except RuntimeError:
        pass
    return f"Definition not found: {word}"


# ==============================================================================
# WEATHER & LOCATION
# ==============================================================================

def get_weather(query: str) -> str:
    """Get weather from Open-Meteo. FREE, no key needed."""
    city = _extract_city(query)
    if not city:
        return "Please specify a city. Example: 'weather in Tokyo'"

    # Geocode city first
    geo_url = (f"https://nominatim.openstreetmap.org/search"
               f"?q={urllib.parse.quote(city)}&format=json&limit=1")
    geo = _http_get(geo_url)
    if not geo:
        return f"City not found: {city}"

    lat = float(geo[0]["lat"])
    lon = float(geo[0]["lon"])
    name = geo[0].get("display_name", city).split(",")[0]

    # Get weather
    url = (f"https://api.open-meteo.com/v1/forecast"
           f"?latitude={lat:.4f}&longitude={lon:.4f}"
           f"&current_weather=true&timezone=auto")
    data = _http_get(url)
    cw = data.get("current_weather", {})

    temp = cw.get("temperature", "?")
    wind = cw.get("windspeed", "?")
    code = cw.get("weathercode", 0)
    weather_desc = _weather_code(code)

    return (f"Weather in {name}: {temp}°C, {weather_desc}, "
            f"wind {wind} km/h")


def geocode(query: str) -> str:
    """Convert address/city to coordinates."""
    q = _extract_query_text(query)
    url = (f"https://nominatim.openstreetmap.org/search"
           f"?q={urllib.parse.quote(q)}&format=json&limit=1")
    data = _http_get(url)
    if not data:
        return f"Location not found: {q}"
    r = data[0]
    return (f"{r.get('display_name', q)}: "
            f"lat={r['lat']}, lon={r['lon']}")


# ==============================================================================
# NEWS & TECH
# ==============================================================================

def hacker_news(query: str) -> str:
    """Top stories from Hacker News. FREE."""
    url = "https://hacker-news.firebaseio.com/v0/topstories.json"
    ids = _http_get(url)[:10]
    lines = ["Top Hacker News Stories:"]
    for i, sid in enumerate(ids[:7]):
        try:
            story = _http_get(
                f"https://hacker-news.firebaseio.com/v0/item/{sid}.json"
            )
            lines.append(
                f"  {i+1}. {story.get('title', '?')} "
                f"({story.get('score', 0)} pts)"
            )
        except RuntimeError:
            continue
    return "\n".join(lines)


def get_news(query: str) -> str:
    """Latest news from free API mirror."""
    category = "general"
    for cat in ["business", "technology", "science", "health",
                "sports", "entertainment"]:
        if cat in query.lower():
            category = cat
            break
    url = (f"https://saurav.tech/NewsAPI/top-headlines/"
           f"category/{category}/us.json")
    try:
        data = _http_get(url)
        articles = data.get("articles", [])[:7]
        lines = [f"Top {category.title()} News:"]
        for a in articles:
            lines.append(f"  - {a.get('title', '?')}")
        return "\n".join(lines)
    except RuntimeError:
        return f"News not available for category: {category}"


# ==============================================================================
# CODE & PACKAGES
# ==============================================================================

def github_repo(query: str) -> str:
    """Get GitHub repo info. FREE."""
    repo = re.search(r'([a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+)', query)
    if not repo:
        return "Specify repo as owner/name (e.g. 'github torvalds/linux')"
    url = f"https://api.github.com/repos/{repo.group(1)}"
    data = _http_get(url)
    return (f"{data.get('full_name', '?')}: {data.get('description', '')}\n"
            f"  Stars: {data.get('stargazers_count', 0):,} | "
            f"Forks: {data.get('forks_count', 0):,} | "
            f"Language: {data.get('language', '?')}")


def pypi_package(query: str) -> str:
    """Get PyPI package info. FREE."""
    name = _extract_query_text(query).replace(" ", "-")
    url = f"https://pypi.org/pypi/{urllib.parse.quote(name)}/json"
    try:
        data = _http_get(url)
        info = data["info"]
        return (f"{info['name']} v{info['version']}: {info.get('summary', '')}\n"
                f"  Author: {info.get('author', '?')} | "
                f"License: {info.get('license', '?')}")
    except RuntimeError:
        return f"Package not found: {name}"


def npm_package(query: str) -> str:
    """Get npm package info. FREE."""
    name = _extract_query_text(query)
    url = f"https://registry.npmjs.org/{urllib.parse.quote(name)}"
    try:
        data = _http_get(url)
        tags = data.get("dist-tags", {})
        return (f"{data.get('name', '?')}: {data.get('description', '')}\n"
                f"  Latest: {tags.get('latest', '?')} | "
                f"License: {data.get('license', '?')}")
    except RuntimeError:
        return f"Package not found: {name}"


def arxiv_search(query: str) -> str:
    """Search arXiv for research papers. FREE."""
    q = _extract_query_text(query)
    url = (f"https://export.arxiv.org/api/query"
           f"?search_query=all:{urllib.parse.quote(q)}&max_results=5")
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "QOR-AI-Agent/1.0")
    with urllib.request.urlopen(req, timeout=15) as resp:
        text = resp.read().decode()

    # Simple XML parsing (no lxml dependency)
    titles = re.findall(r'<title>(.*?)</title>', text, re.DOTALL)
    summaries = re.findall(r'<summary>(.*?)</summary>', text, re.DOTALL)

    lines = ["arXiv Papers:"]
    for i, (t, s) in enumerate(zip(titles[1:6], summaries[:5])):  # skip feed title
        title = t.strip().replace('\n', ' ')
        summary = s.strip().replace('\n', ' ')[:150]
        lines.append(f"  {i+1}. {title}\n     {summary}...")
    return "\n".join(lines) if len(lines) > 1 else f"No papers found for: {q}"


def huggingface_models(query: str) -> str:
    """Search HuggingFace for AI models. FREE."""
    q = _extract_query_text(query)
    url = f"https://huggingface.co/api/models?limit=5&search={urllib.parse.quote(q)}"
    data = _http_get(url)
    lines = ["HuggingFace Models:"]
    for m in data[:5]:
        lines.append(
            f"  - {m.get('modelId', '?')} "
            f"(downloads: {m.get('downloads', 0):,})"
        )
    return "\n".join(lines)


# ==============================================================================
# CODE EXECUTION
# ==============================================================================

def run_code(query: str) -> str:
    """Execute code via Piston API. FREE, no key needed.
    Supports: Python, JavaScript, Go, Rust, C, Java, Ruby, Bash."""
    # Extract language and code
    lang_match = re.search(r'(python|javascript|js|go|rust|c\+\+|java|ruby|bash)',
                           query.lower())
    lang = lang_match.group(1) if lang_match else "python"

    # Extract code (between ``` or after "run:")
    code_match = re.search(r'```(?:\w+)?\n?(.*?)```', query, re.DOTALL)
    if code_match:
        code = code_match.group(1).strip()
    else:
        code = re.sub(r'^.*?(run|execute|eval)\s*:?\s*', '', query,
                      flags=re.IGNORECASE).strip()

    if not code:
        return "Provide code to execute. Example: run python: print('hello')"

    lang_map = {"js": "javascript", "py": "python", "c++": "c++",
                "rb": "ruby", "sh": "bash", "ts": "typescript"}
    lang = lang_map.get(lang, lang)

    payload = {
        "language": lang, "version": "*",
        "files": [{"content": code}]
    }
    data = _http_post("https://emkc.org/api/v2/piston/execute", payload)
    if "message" in data:
        return f"Error: {data['message']}"
    run = data.get("run", {})
    output = run.get("output", run.get("stdout", ""))
    stderr = run.get("stderr", "")
    if stderr and not output:
        return f"Error:\n{stderr}"
    return f"Output ({lang}):\n{output}"


# ==============================================================================
# ENTERTAINMENT
# ==============================================================================

def get_joke(query: str) -> str:
    """Random joke from JokeAPI. FREE."""
    url = "https://v2.jokeapi.dev/joke/Any?safe-mode"
    data = _http_get(url)
    if data.get("type") == "single":
        return data.get("joke", "No joke found")
    return f"{data.get('setup', '')}\n{data.get('delivery', '')}"


def get_trivia(query: str) -> str:
    """Trivia questions from OpenTDB. FREE."""
    url = "https://opentdb.com/api.php?amount=3&type=multiple"
    data = _http_get(url)
    lines = ["Trivia:"]
    for q in data.get("results", []):
        lines.append(f"  Q: {q['question']}")
        lines.append(f"  A: {q['correct_answer']}")
    return "\n".join(lines)


def search_recipes(query: str) -> str:
    """Search recipes from TheMealDB. FREE."""
    q = _extract_query_text(query)
    url = f"https://www.themealdb.com/api/json/v1/1/search.php?s={urllib.parse.quote(q)}"
    data = _http_get(url)
    meals = data.get("meals") or []
    if not meals:
        return f"No recipes found for: {q}"
    lines = ["Recipes:"]
    for m in meals[:3]:
        instructions = (m.get("strInstructions") or "")[:200]
        lines.append(f"  {m['strMeal']} ({m.get('strArea', '?')})")
        lines.append(f"    {instructions}...")
    return "\n".join(lines)


def reddit_posts(query: str) -> str:
    """Top posts from a subreddit. FREE."""
    sub = re.search(r'r/(\w+)', query)
    subreddit = sub.group(1) if sub else "news"
    url = f"https://www.reddit.com/r/{subreddit}/top.json?limit=7"
    data = _http_get(url, headers={"User-Agent": "QOR-AI-Agent/1.0"})
    children = data.get("data", {}).get("children", [])
    lines = [f"Top r/{subreddit}:"]
    for c in children[:7]:
        d = c.get("data", {})
        lines.append(f"  - {d.get('title', '?')} ({d.get('score', 0)} pts)")
    return "\n".join(lines)


def open_library_search(query: str) -> str:
    """Search books on Open Library. FREE."""
    q = _extract_query_text(query)
    url = f"https://openlibrary.org/search.json?q={urllib.parse.quote(q)}&limit=5"
    data = _http_get(url)
    lines = ["Books:"]
    for doc in data.get("docs", [])[:5]:
        authors = doc.get("author_name", ["Unknown"])
        lines.append(
            f"  - {doc.get('title', '?')} by {authors[0]} "
            f"({doc.get('first_publish_year', '?')})"
        )
    return "\n".join(lines)


def nasa_apod(query: str) -> str:
    """NASA Astronomy Picture of the Day. FREE (DEMO_KEY)."""
    url = "https://api.nasa.gov/planetary/apod?api_key=DEMO_KEY"
    data = _http_get(url)
    return (f"{data.get('title', '?')} ({data.get('date', '?')})\n"
            f"{data.get('explanation', '')[:400]}...\n"
            f"Image: {data.get('url', '')}")


# ==============================================================================
# UTILITY
# ==============================================================================

def get_country_info(query: str) -> str:
    """Country information from RestCountries. FREE."""
    name = _extract_query_text(query)
    url = f"https://restcountries.com/v3.1/name/{urllib.parse.quote(name)}"
    try:
        data = _http_get(url)
        if isinstance(data, list) and data:
            c = data[0]
            common = c.get("name", {}).get("common", "?")
            capital = (c.get("capital") or ["?"])[0]
            pop = c.get("population", 0)
            region = c.get("region", "?")
            langs = ", ".join((c.get("languages") or {}).values())
            return (f"{common}: Capital={capital}, Pop={pop:,}, "
                    f"Region={region}, Languages={langs}")
    except RuntimeError:
        pass
    return f"Country not found: {name}"


def read_file(query: str) -> str:
    """Read a local file mentioned in the query."""
    import os as _os
    filenames = re.findall(r'[\w\-\.\/\\]+\.(?:txt|md|csv|json|py|yaml|yml|log|ini|cfg|toml)', query)
    if not filenames:
        return "Specify a filename. Example: 'read notes.txt'"
    for fname in filenames:
        search_dirs = [".", "data", "knowledge", "learn", "documents", "qor-data"]
        for dir_path in search_dirs:
            full_path = _os.path.join(dir_path, fname)
            if _os.path.exists(full_path):
                with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()[:2000]
                return f"Contents of {fname}:\n{content}"
        # Try as absolute/relative path directly
        if _os.path.exists(fname):
            with open(fname, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()[:2000]
            return f"Contents of {fname}:\n{content}"
    return f"File not found: {filenames[0]}"


def news_search(query: str) -> str:
    """Search recent news via DuckDuckGo (if installed) or free API."""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.news(query, max_results=5))
            if results:
                lines = ["Recent News:"]
                for r in results:
                    lines.append(f"  - {r['title']} ({r.get('date', 'recent')})")
                return "\n".join(lines)
    except (ImportError, Exception):
        pass
    # Fallback to free news API
    return get_news(query)


def _safe_eval(expr: str):
    """Safe math evaluator using AST — no code execution."""
    import ast
    import operator

    ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def _eval_node(node):
        if isinstance(node, ast.Expression):
            return _eval_node(node.body)
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Unsupported constant: {node.value}")
        elif isinstance(node, ast.BinOp):
            op_func = ops.get(type(node.op))
            if op_func is None:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            return op_func(_eval_node(node.left), _eval_node(node.right))
        elif isinstance(node, ast.UnaryOp):
            op_func = ops.get(type(node.op))
            if op_func is None:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            return op_func(_eval_node(node.operand))
        else:
            raise ValueError(f"Unsupported expression: {type(node).__name__}")

    tree = ast.parse(expr, mode='eval')
    return _eval_node(tree)


def calculate(query: str) -> str:
    """Simple calculator."""
    expr = re.findall(r'[\d\.\+\-\*/\(\)\s\%]+', query)
    if expr:
        try:
            # Safe eval — only math operations
            cleaned = expr[0].strip()
            allowed = set("0123456789.+-*/() %")
            if all(c in allowed for c in cleaned):
                result = _safe_eval(cleaned)
                return f"{cleaned} = {result}"
        except Exception:
            pass
    return "Could not parse math expression"


def get_time(query: str) -> str:
    """Get current time and date."""
    now = datetime.now(timezone.utc)
    return (f"UTC: {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Unix: {int(now.timestamp())}\n"
            f"Day: {now.strftime('%A')}")


def ip_lookup(query: str) -> str:
    """IP geolocation. FREE."""
    ip = re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', query)
    if not ip:
        return "Provide an IP address"
    url = f"http://ip-api.com/json/{ip.group()}"
    data = _http_get(url)
    if data.get("status") == "fail":
        return f"IP lookup failed: {data.get('message', '?')}"
    return (f"IP {data.get('query', '?')}: "
            f"{data.get('city', '?')}, {data.get('regionName', '?')}, "
            f"{data.get('country', '?')} | ISP: {data.get('isp', '?')}")


# ==============================================================================
# CACHING & RATE LIMITING
# ==============================================================================

class TTLCache:
    """Time-based cache with expiration. Default 5 minutes."""

    def __init__(self, ttl_seconds: int = 300):
        self.ttl = ttl_seconds
        self._cache = {}  # key -> (value, timestamp)

    def get(self, key: str):
        """Get cached value or None if expired/missing."""
        if key in self._cache:
            value, ts = self._cache[key]
            if time.time() - ts < self.ttl:
                return value
            del self._cache[key]
        return None

    def set(self, key: str, value):
        """Store value with current timestamp."""
        self._cache[key] = (value, time.time())

    def clear(self):
        """Clear all cached entries."""
        self._cache.clear()


class RateLimiter:
    """Token bucket rate limiter. Default 30 calls/minute per tool."""

    def __init__(self, max_calls: int = 30, period_seconds: int = 60):
        self.max_calls = max_calls
        self.period = period_seconds
        self._calls = {}  # tool_name -> [timestamps]

    def allow(self, tool_name: str) -> bool:
        """Check if a call is allowed for this tool."""
        now = time.time()
        if tool_name not in self._calls:
            self._calls[tool_name] = []

        # Remove old timestamps
        self._calls[tool_name] = [
            ts for ts in self._calls[tool_name]
            if now - ts < self.period
        ]

        if len(self._calls[tool_name]) >= self.max_calls:
            return False

        self._calls[tool_name].append(now)
        return True

    def wait_time(self, tool_name: str) -> float:
        """Seconds until next call is allowed."""
        if tool_name not in self._calls or not self._calls[tool_name]:
            return 0.0
        oldest = min(self._calls[tool_name])
        wait = self.period - (time.time() - oldest)
        return max(wait, 0.0)


# ==============================================================================
# SPORTS — Free ESPN / TheSportsDB API
# ==============================================================================

def get_sports_scores(query: str) -> str:
    """Get live sports scores and results via TheSportsDB (free, no API key)."""
    try:
        import urllib.request
        import json as _json

        # TheSportsDB free API
        q = urllib.parse.quote(query)
        # Search for events by team/league name
        url = f"https://www.thesportsdb.com/api/v1/json/3/searchevents.php?e={q}"
        req = urllib.request.Request(url, headers={"User-Agent": "QOR/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = _json.loads(resp.read())

        events = data.get("event") or []
        if not events:
            # Try searching by team name
            url2 = f"https://www.thesportsdb.com/api/v1/json/3/searchteams.php?t={q}"
            req2 = urllib.request.Request(url2, headers={"User-Agent": "QOR/1.0"})
            with urllib.request.urlopen(req2, timeout=10) as resp2:
                team_data = _json.loads(resp2.read())
            teams = team_data.get("teams") or []
            if teams:
                team = teams[0]
                # Get last 5 events for this team
                team_id = team["idTeam"]
                url3 = f"https://www.thesportsdb.com/api/v1/json/3/eventslast.php?id={team_id}"
                req3 = urllib.request.Request(url3, headers={"User-Agent": "QOR/1.0"})
                with urllib.request.urlopen(req3, timeout=10) as resp3:
                    events_data = _json.loads(resp3.read())
                events = (events_data.get("results") or [])[:5]
                if not events:
                    return f"Team found: {team['strTeam']} ({team.get('strSport', '?')}), " \
                           f"League: {team.get('strLeague', '?')}, " \
                           f"Stadium: {team.get('strStadium', '?')}. No recent results."

        if not events:
            return f"No sports results found for: {query}"

        lines = []
        for ev in events[:5]:
            home = ev.get("strHomeTeam", "?")
            away = ev.get("strAwayTeam", "?")
            h_score = ev.get("intHomeScore", "?")
            a_score = ev.get("intAwayScore", "?")
            date = ev.get("dateEvent", "?")
            league = ev.get("strLeague", "")
            status = ev.get("strStatus", "")
            line = f"{home} {h_score} - {a_score} {away}"
            if date != "?":
                line += f" ({date})"
            if league:
                line += f" [{league}]"
            if status:
                line += f" Status: {status}"
            lines.append(line)

        return "Sports results:\n" + "\n".join(lines)

    except Exception as e:
        return f"Sports lookup error: {e}"


# ==============================================================================
# TOOL EXECUTOR — Registers all tools with ConfidenceGate
# ==============================================================================

# Master registry: tool_name → (function, description, categories)
ALL_TOOLS = {
    # Crypto & Finance
    "crypto_price":    (get_crypto_price,   "Get cryptocurrency price",
                        ["price", "crypto", "bitcoin", "btc", "eth"]),
    "binance_price":   (get_binance_price,  "Get Binance exchange price",
                        ["binance", "trading"]),
    "crypto_market":   (get_crypto_market,  "Top cryptocurrencies by market cap",
                        ["market", "top crypto"]),
    "convert_currency":(convert_currency,   "Convert currency with live rates",
                        ["currency", "exchange", "convert"]),
    "crypto_history":  (crypto_history,     "Historical crypto price on a past date",
                        ["crypto history", "price history", "bitcoin history"]),

    # Trading Intelligence
    "fear_greed":      (get_fear_greed,     "Crypto Fear & Greed Index (0-100 sentiment)",
                        ["fear", "greed", "fear and greed", "market sentiment"]),
    "polymarket":      (get_polymarket,     "Prediction market odds from Polymarket",
                        ["polymarket", "prediction market", "probability", "odds"]),
    "funding_rate":    (get_funding_rate,   "Crypto perpetual funding rate (Binance)",
                        ["funding rate", "funding", "perpetual"]),
    "open_interest":   (get_open_interest,  "Crypto futures open interest (Binance)",
                        ["open interest", "oi", "futures positions"]),
    "technical_analysis": (get_technical_analysis, "Technical indicators (RSI, EMA, MACD, BB, ATR) for crypto, stocks, commodities, forex",
                        ["rsi", "ema", "macd", "bollinger", "atr", "technical"]),
    "multi_tf_analysis": (get_multi_timeframe_analysis,
        "Multi-timeframe technical analysis with trade levels (weekly to 5min)",
        ["multi timeframe", "all timeframes", "trading strategy", "buy price",
         "sell price", "stop loss", "entry", "take profit", "full analysis"]),
    "economic_calendar": (get_economic_calendar, "Upcoming high-impact economic events",
                        ["economic calendar", "fed meeting", "cpi", "fomc"]),
    "crypto_trending": (get_crypto_trending, "Trending cryptocurrencies on CoinGecko",
                        ["trending crypto", "trending coins", "hot crypto"]),
    "global_market":   (get_global_market,  "Global crypto market overview",
                        ["global market", "total market cap", "btc dominance"]),
    "volume_profile":  (get_volume_profile, "Volume profile analysis (POC, VAH, VAL, HVN, LVN) for any asset",
                        ["volume profile", "vp", "poc", "value area", "hvn", "lvn"]),
    "quant_signals":   (get_quant_signals, "Live quant signals (Hurst, Z-Score, volatility) for any asset",
                        ["hurst", "z-score", "zscore", "mean reversion", "trending", "regime"]),

    # Financial Markets
    "commodities":     (get_commodities,    "Precious metals spot prices (gold, silver, platinum)",
                        ["gold", "silver", "platinum", "palladium", "commodities", "metals"]),
    "stock_quote":     (get_stock_quote,    "Stock price quote (any ticker or company name)",
                        ["stock", "share", "equity", "ticker", "nyse", "nasdaq"]),
    "forex_rates":     (get_forex_rates,    "Forex currency exchange rates",
                        ["forex", "fx", "currency pair", "eur/usd", "exchange rates"]),
    "world_economy":   (get_world_economy,  "Top world economies by GDP (World Bank)",
                        ["gdp", "economy", "top economies", "world economy"]),
    "market_indices":  (get_market_indices, "Global stock market indices (S&P, Dow, NASDAQ, FTSE...)",
                        ["indices", "dow jones", "s&p 500", "nasdaq index", "ftse", "nikkei"]),

    # Historical & Knowledge
    "wikidata":        (wikidata_search,    "Search Wikidata structured facts",
                        ["wikidata", "historical fact", "who was", "when was"]),
    "on_this_day":     (on_this_day,        "Historical events on a given date",
                        ["on this day", "history today", "what happened"]),
    "gdelt":           (gdelt_search,       "Search global news events since 1979",
                        ["gdelt", "global events", "world events", "geopolitical"]),

    # Knowledge & Search
    "wikipedia":       (wikipedia_search,   "Search Wikipedia",
                        ["wiki", "encyclopedia"]),
    "web_search":      (web_search,         "Search the web",
                        ["search", "find", "look up"]),
    "duckduckgo":      (duckduckgo_search,  "DuckDuckGo instant answers",
                        ["search"]),
    "definition":      (get_definition,     "Get word definition",
                        ["define", "dictionary", "meaning"]),

    # Weather & Location
    "weather":         (get_weather,        "Get current weather",
                        ["weather", "temperature", "forecast"]),
    "geocode":         (geocode,            "Convert address to coordinates",
                        ["coordinates", "location", "geocode"]),
    "ip_lookup":       (ip_lookup,          "IP address geolocation",
                        ["ip", "geolocation"]),

    # News
    "hacker_news":     (hacker_news,        "Top Hacker News stories",
                        ["hacker news", "tech news", "hn"]),
    "news":            (get_news,           "Latest news headlines",
                        ["news", "headlines"]),

    # Code & Packages
    "github":          (github_repo,        "GitHub repository info",
                        ["github", "repo"]),
    "pypi":            (pypi_package,       "PyPI Python package info",
                        ["pypi", "python package", "pip"]),
    "npm":             (npm_package,        "npm JavaScript package info",
                        ["npm", "node", "javascript package"]),
    "arxiv":           (arxiv_search,       "Search arXiv research papers",
                        ["arxiv", "research", "paper", "academic"]),
    "huggingface":     (huggingface_models, "Search HuggingFace AI models",
                        ["huggingface", "ai model", "ml model"]),
    "run_code":        (run_code,           "Execute code (Python, JS, Go...)",
                        ["run", "execute", "code"]),

    # Entertainment
    "joke":            (get_joke,           "Get a random joke",
                        ["joke", "funny"]),
    "trivia":          (get_trivia,         "Trivia questions",
                        ["trivia", "quiz"]),
    "recipe":          (search_recipes,     "Search for recipes",
                        ["recipe", "cook", "food"]),
    "reddit":          (reddit_posts,       "Top Reddit posts",
                        ["reddit", "subreddit"]),
    "books":           (open_library_search,"Search books on Open Library",
                        ["book", "library", "author"]),
    "nasa":            (nasa_apod,          "NASA Astronomy Picture of the Day",
                        ["nasa", "astronomy", "space"]),

    # Utility
    "calculate":       (calculate,          "Math calculator",
                        ["math", "calculate", "compute"]),
    "time":            (get_time,           "Current time and date",
                        ["time", "date", "clock"]),
    "country":         (get_country_info,   "Country information",
                        ["country", "nation", "population"]),
    "file_reader":     (read_file,          "Read contents of a local file",
                        ["file", "read file", "document"]),
    "news_search":     (news_search,        "Search recent news articles",
                        ["latest news", "news search", "recent news"]),

    # Documents & Video
    "read_document":   (_read_document,     "Read text from PDF/DOCX documents",
                        ["pdf", "docx", "document", "read document"]),
    "read_video":      (_read_video,        "Extract frames from video file or URL",
                        ["video", "youtube", "video frames", "analyze video"]),

    # Browser
    "browse_web":      (_browse_web,        "Browse webpage and extract text",
                        ["browse", "visit", "navigate", "web page", "website"]),
    "browse_screenshot":(_browse_screenshot, "Screenshot a webpage",
                        ["screenshot", "capture page", "screencap"]),

    # Sports
    "sports":          (get_sports_scores,  "Live sports scores and results",
                        ["sports", "score", "nfl", "nba", "mlb", "soccer",
                         "football", "basketball", "baseball", "tennis",
                         "hockey", "nhl", "premier league", "champions league",
                         "world cup", "olympics"]),
}


class ToolExecutor:
    """Manages and executes all tools (thread-safe for parallel calls)."""

    def __init__(self):
        self.tools = dict(ALL_TOOLS)
        self.cache = TTLCache(ttl_seconds=300)
        self.rate_limiter = RateLimiter(max_calls=30, period_seconds=60)
        self._lock = threading.Lock()

    def register_all(self, gate):
        """Register all tools with a ConfidenceGate."""
        for name, (func, desc, cats) in self.tools.items():
            gate.tools.register(name, desc, func, cats)
        print(f"[Tools] Registered {len(self.tools)} tools")

    def call(self, name: str, query: str) -> str:
        """Call a tool by name with caching and rate limiting (thread-safe)."""
        with self._lock:
            if name not in self.tools:
                return f"Tool not found: {name}"

            cache_key = f"{name}:{query}"
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

            if not self.rate_limiter.allow(name):
                wait = self.rate_limiter.wait_time(name)
                return f"Rate limit reached for {name}. Try again in {wait:.0f}s."

        # HTTP call outside lock — I/O bound, runs in parallel
        func = self.tools[name][0]
        try:
            result = func(query)
            with self._lock:
                self.cache.set(cache_key, result)
            return result
        except Exception as e:
            return f"Tool error ({name}): {e}"

    def detect_intent(self, query: str) -> Optional[str]:
        """Detect which tool should handle a query.

        Order matters: specific multi-word patterns BEFORE generic single-word.
        This prevents "El Salvador bitcoin reserves" from matching "bitcoin"
        (crypto_price) when it should match "reserves" (web_search).
        """
        query_lower = query.lower()

        # ── TIER 1: Very specific multi-word patterns (highest priority) ──
        keyword_map = {
            # Trading intelligence
            "multi_tf_analysis": [
                "multi timeframe", "all timeframes", "trading strategy",
                "buy price", "sell price", "stop loss", "take profit",
                "entry point", "full analysis", "in depth analysis",
                "complete analysis", "trade levels", "multi tf",
                "should i buy", "should i sell", "should i trade",
                "good time to buy", "good time to sell", "trade setup",
                "entry and exit", "long or short", "scalp trade",
            ],
            "technical_analysis": [
                "rsi", "ema ", "macd", "bollinger", "atr",
                "technical analysis", "technical indicator", "indicators for",
                "support and resistance", "moving average", "ichimoku",
                "fibonacci retrace", "chart pattern", "candlestick pattern",
                "head and shoulders", "double top", "double bottom",
                "golden cross", "death cross", "volume profile",
                "obv ", "adx ", "stochastic", "vwap",
            ],
            "funding_rate": [
                "funding rate", "perpetual rate", "perp rate",
                "funding fee", "funding cost",
            ],
            "open_interest": [
                "open interest", "oi ", "futures positions",
                "derivative positions", "long short ratio",
            ],
            "fear_greed": [
                "fear and greed", "fear & greed", "market sentiment",
                "sentiment index", "fear greed", "greed index",
                "investor sentiment", "market mood", "market fear",
            ],
            "polymarket": [
                "polymarket", "prediction market", "probability of",
                "odds of", "will trump", "will biden", "who will win",
                "election odds", "election prediction", "chances of",
                "bet on", "betting odds", "what are the odds",
                "will there be", "prediction for", "market prediction",
                "forecast odds", "kalshi", "predictit",
            ],
            "economic_calendar": [
                "economic calendar", "fed meeting", "cpi data", "cpi report",
                "nfp", "fomc", "economic event", "fed rate",
                "interest rate decision", "jobs report", "unemployment rate",
                "ppi ", "retail sales", "gdp report", "fed decision",
                "central bank", "rate hike", "rate cut", "tapering",
                "quantitative easing", "quantitative tightening",
            ],
            "crypto_trending": [
                "trending crypto", "trending coin", "hot crypto",
                "popular crypto", "what's trending", "is trending",
                "trending token", "most gained", "biggest gainer",
                "biggest loser", "top movers",
            ],
            "global_market": [
                "global market", "total market cap", "btc dominance",
                "crypto dominance", "market overview",
                "crypto market overview", "crypto market cap",
                "total crypto", "defi tvl",
            ],

            # ── Financial markets (specific) ──
            "market_indices": [
                "dow jones", "s&p 500", "s&p500", "nasdaq index",
                "nasdaq composite", "ftse", "nikkei", "dax", "cac 40",
                "hang seng", "stock market index", "market indices",
                "stock indices", "russell 2000", "wilshire", "stoxx",
                "kospi", "sensex", "bse", "nse",
            ],
            "commodities": [
                "gold price", "silver price", "platinum price",
                "palladium", "precious metal", "commodities",
                "gold spot", "silver spot", "xau", "xag",
                "crude oil", "brent", "wti", "natural gas",
                "copper price", "wheat price", "corn price",
                "oil price", "commodity price",
            ],
            "world_economy": [
                "gdp of", "gdp per capita", "top economies",
                "world economy", "richest countries", "largest economies",
                "world gdp", "economy ranking", "economic ranking",
                "trade balance", "current account", "national debt",
                "debt to gdp", "economic growth",
            ],
            "forex_rates": [
                "forex", "fx rate", "currency pair",
                "eur/usd", "gbp/usd", "usd/jpy", "aud/usd", "usd/cad",
                "usd/chf", "nzd/usd", "forex rate", "exchange rate for",
                "dollar to rupee", "dollar to euro", "dollar to pound",
                "dollar to yen", "euro to dollar", "pound to dollar",
            ],
            "stock_quote": [
                "stock price", "stock quote", "share price",
                "stock of", "shares of", "ticker",
                "nyse", "nasdaq listed", "stock market",
                "apple stock", "tesla stock", "google stock",
                "amazon stock", "microsoft stock", "nvidia stock",
                "meta stock", "earning", "quarterly result",
                "eps ", "p/e ratio", "market cap of",
            ],

            # ── Browser ──
            "browse_web": [
                "browse ", "visit ", "go to ", "open website",
                "navigate to", "open url", "web page", "open page",
                "open site", "check website", "load page",
                # Social media automation triggers
                "linkedin", "delete posts", "remove posts",
                "clean my", "my feed", "my profile page",
                "my connections", "bulk delete", "remove all posts",
                "open gmail", "open youtube", "open twitter",
                "open facebook", "open instagram", "open reddit",
            ],
            "browse_screenshot": [
                "screenshot", "capture page", "take screenshot", "screencap",
            ],

            # ── Factual lookup (BEFORE generic crypto/search catches) ──
            "web_search": [
                # Research / lookup intent
                "search for", "look up", "find info", "google ",
                "search about", "find out", "tell me about",
                "i want to know", "can you find", "research ",
                # Quantitative facts
                "how many", "how much does", "how much did",
                "how much is the", "how much was",
                "reserves", "holdings", "total value", "net worth",
                "who owns", "who holds", "how big is",
                "in total", "altogether",
                # Comparison / ranking
                "largest", "smallest", "biggest", "tallest",
                "shortest", "richest", "poorest", "fastest",
                "slowest", "oldest", "youngest", "most popular",
                "most expensive", "cheapest", "highest paid",
                "top 10", "top 5", "top 20", "top 100",
                "best ", "worst ", "ranked", "ranking",
                "compared to", "difference between", "vs ",
                # Stats / data
                "statistics", "data on", "number of", "total number",
                "count of", "rate of", "percent", "percentage",
                "life expectancy", "literacy rate", "unemployment",
                "inflation rate", "salary of", "revenue of",
                "market share", "budget of", "debt of",
                # People
                "who is", "who was", "who are", "who founded",
                "who invented", "who discovered", "who created",
                "who leads", "who built", "who runs",
                "who killed", "who married", "who wrote",
                "who directed", "who designed", "who said",
                "biography", "born in", "died in",
                # What / Which / Where / When
                "what country", "what city", "what year", "what date",
                "what language", "what religion",
                "which country", "which city", "which company",
                "which president", "which government", "which team",
                "where is", "where was", "where are", "where did",
                "when did", "when was", "when is",
                # Historical
                "history of", "origin of", "founded in",
                "established in", "invented in", "discovered in",
                "battle of", "war of", "treaty of",
                # Science
                "speed of", "boiling point", "melting point",
                "chemical formula", "atomic number", "scientific name",
                "discovery of", "theory of", "law of",
                # Geography
                "located in", "borders", "timezone of",
                "coordinates of", "elevation of", "area of",
                "continent of", "region of",
                # Government / politics
                "president of", "prime minister of", "king of",
                "queen of", "chancellor of", "governor of",
                "government of", "political party", "constitution",
                "law in", "policy on", "regulation",
                # Business
                "ceo of", "founder of", "headquarter", "ipo of",
                "valuation of", "acquisition", "merger",
                "founded by", "owned by", "subsidiary",
                # Health
                "symptoms of", "treatment for", "cure for",
                "side effects of", "causes of", "diagnosis",
                "vaccine for", "mortality rate",
                # Culture / entertainment
                "directed by", "written by", "author of",
                "singer of", "actor in", "played by",
                "released in", "published in", "won the",
                # Misc factual
                "meaning of", "stand for", "abbreviation",
                "acronym", "full form of", "symbol of",
                "flag of", "anthem of", "motto of",
                "what does", "what did", "what caused",
                "explain ", "why did", "why does", "why is",
                "is it true", "fact about", "facts about",
            ],

            # ── Crypto & Finance (generic catches) ──
            "crypto_price": [
                "crypto", "bitcoin", "btc", "eth", "ethereum",
                "solana", "sol ", "xrp", "ripple", "cardano", "ada ",
                "dogecoin", "doge", "shiba", "polkadot", "dot ",
                "avalanche", "avax", "chainlink", "link ",
                "polygon", "matic", "litecoin", "ltc",
                "price of", "how much is",
                "tether", "usdt", "usdc", "bnb", "binance coin",
                "tron", "trx", "near ", "sui ", "aptos",
                "arbitrum", "optimism", "base chain",
            ],
            "binance_price": ["binance price", "on binance"],
            "crypto_market": [
                "crypto market", "market cap", "top crypto",
                "top coins", "top tokens", "altcoin",
                "defi market", "nft market",
            ],
            "convert_currency": [
                "convert ", "currency convert", "exchange rate",
                "how many dollars", "how many euros",
                "how many pounds", "how many yen",
                "in usd", "in euros", "in gbp", "in rupees",
            ],
            "crypto_history": [
                "price in ", "price on ", "crypto history",
                "bitcoin in ", "btc in ", "eth in ",
                "price history", "historical price",
                "price last year", "price last month",
                "all time high of", "ath of",
            ],

            # ── Knowledge & historical ──
            "wikidata": [
                "wikidata", "historical fact",
                "structured data about", "entity data",
            ],
            "on_this_day": [
                "on this day", "history today", "what happened on",
                "events on", "what happened in",
                "born on this day", "died on this day",
                "today in history",
            ],
            "gdelt": [
                "gdelt", "global events", "world events",
                "geopolitical", "international news", "world news",
                "global conflict", "international crisis",
                "geopolitics", "world affairs",
            ],

            # ── Documents & Video ──
            "read_document": [
                "read pdf", "open pdf", "pdf", "docx",
                "document", "read document", "extract text from",
                "summarize pdf", "summarize document",
            ],
            "read_video": [
                "video", "watch video", "video frames",
                "youtube", "extract frames", "analyze video",
                "video url", "clip ", "footage",
            ],

            # ── Knowledge lookup ──
            "wikipedia": [
                "wikipedia", "wiki", "wiki page", "wiki article",
            ],
            "definition": [
                "define ", "definition of", "meaning of",
                "what does it mean", "etymology", "synonym",
                "antonym", "thesaurus",
            ],

            # ── Weather ──
            "weather": [
                "weather", "temperature", "forecast", "rain",
                "sunny", "cloudy", "storm", "humidity", "wind",
                "snow", "heatwave", "cold front", "uv index",
                "air quality", "pollen", "is it cold", "is it hot",
                "is it raining", "will it rain", "weather tomorrow",
            ],

            # ── News ──
            "hacker_news": [
                "hacker news", "hn ", "tech news", "ycombinator",
                "y combinator", "hackernews", "show hn", "ask hn",
            ],
            "news_search": [
                "latest news", "recent news", "news about",
                "news search", "breaking news", "headline",
                "news today", "news on", "news for",
                "what's happening", "current affairs",
            ],
            "news": [
                "news", "headlines", "current events",
                "top stories", "world news",
            ],

            # ── Code & packages ──
            "github": [
                "github", "repository", "repo ", "git repo",
                "open source", "github repo", "pull request",
                "github project", "star count",
            ],
            "pypi": [
                "pypi", "pip install", "python package",
                "python library", "python module",
            ],
            "npm": [
                "npm", "node package", "npm install",
                "javascript package", "js library", "yarn add",
            ],
            "arxiv": [
                "arxiv", "research paper", "academic paper",
                "scientific paper", "preprint", "machine learning paper",
                "ai paper", "deep learning paper",
            ],
            "huggingface": [
                "huggingface", "hugging face", "ai model",
                "transformer model", "llm model", "pretrained model",
                "model card", "model hub",
            ],
            "run_code": [
                "run code", "execute code", "```",
                "run this", "execute this", "run python",
                "run javascript", "compile", "test code",
            ],

            # ── Entertainment ──
            "joke": [
                "joke", "funny", "humor", "make me laugh",
                "tell me a joke", "something funny",
            ],
            "trivia": [
                "trivia", "quiz", "fun fact", "random fact",
                "did you know", "interesting fact",
            ],
            "recipe": [
                "recipe", "cook", "how to make", "how to cook",
                "ingredients for", "bake ", "cooking time",
                "meal prep", "dish called",
            ],
            "reddit": [
                "reddit", "r/", "subreddit", "redditor",
                "reddit post", "reddit thread",
            ],
            "books": [
                "book ", "books about", "library search",
                "author ", "novel ", "read about",
                "book recommendation", "best book",
                "published by", "isbn",
            ],
            "nasa": [
                "nasa", "astronomy picture", "space photo",
                "apod", "astronomy photo", "hubble",
                "james webb", "jwst",
            ],

            # ── Utility ──
            "calculate": [
                "calculate", "math", "compute", "what is ",
                "how much is ", "solve ", "equation",
                "percentage of", "square root", "factorial",
                "convert ", "formula",
            ],
            "time": [
                "what time", "current time", "date today",
                "what date", "today's date", "time in",
                "time zone", "utc time", "local time",
            ],
            "country": [
                "country info", "population of", "capital of",
                "capital city of", "flag of", "language of",
                "languages spoken", "currency of", "area of",
                "continent of", "government of", "leader of",
                "national anthem", "country code",
                "demonym", "calling code",
            ],
            "ip_lookup": [
                "ip address", "ip lookup", "my ip",
                "what is my ip", "ip location", "geolocation",
            ],
            "geocode": [
                "coordinates", "geocode", "latitude of",
                "longitude of", "lat long", "gps coordinates",
            ],
            "file_reader": [
                "read file", "open file", "contents of",
                "show file", "cat file", "display file",
            ],
        }

        for tool_name, keywords in keyword_map.items():
            for kw in keywords:
                if kw in query_lower:
                    return tool_name
        return None

    def list_tools(self) -> List[dict]:
        """List all available tools."""
        return [{"name": n, "description": d, "categories": c}
                for n, (_, d, c) in self.tools.items()]


# ==============================================================================
# EXTRACTION HELPERS
# ==============================================================================

def _extract_crypto(query: str) -> str:
    symbols = {
        "bitcoin": "bitcoin", "btc": "bitcoin",
        "ethereum": "ethereum", "eth": "ethereum",
        "solana": "solana", "sol": "solana",
        "cardano": "cardano", "ada": "cardano",
        "dogecoin": "dogecoin", "doge": "dogecoin",
        "xrp": "ripple", "ripple": "ripple",
        "polkadot": "polkadot", "dot": "polkadot",
        "qor": "qora", "qora": "qora",
    }
    q = query.lower()
    for key, coin in symbols.items():
        if key in q:
            return coin
    return "bitcoin"


def _extract_trading_symbol(query: str) -> str:
    q = query.upper()
    match = re.search(r'([A-Z]{2,5})[-/]?(USD|USDT|USDC)?', q)
    if match:
        return match.group(0)
    return "BTCUSDT"


def _extract_query_text(query: str) -> str:
    prefixes = ["search for", "look up", "find", "what is", "who is",
                "define", "meaning of", "tell me about"]
    q = query
    for p in prefixes:
        if query.lower().startswith(p):
            q = query[len(p):].strip()
            break
    return q.strip("?.,! ")


def _extract_city(query: str) -> str:
    for prefix in ["weather in", "temperature in", "forecast for",
                   "weather at", "forecast in"]:
        idx = query.lower().find(prefix)
        if idx != -1:
            return query[idx + len(prefix):].strip().strip("?.,!")
    return ""


def _weather_code(code: int) -> str:
    codes = {
        0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy",
        3: "Overcast", 45: "Foggy", 48: "Depositing rime fog",
        51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
        61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
        71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
        80: "Slight showers", 81: "Moderate showers", 82: "Violent showers",
        95: "Thunderstorm", 96: "Thunderstorm + hail",
    }
    return codes.get(code, f"Code {code}")

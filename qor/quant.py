"""
QOR Quant Engine
================
1. Asset classification — detect crypto/stock/commodity/forex from symbol name
2. Hidden Markov Model — market regime detection for buy/sell signals
3. QuantMetrics — all institutional quant formulas in one place:
   - Sharpe Ratio, Sortino Ratio, Expectancy, Risk of Ruin,
   - Optimal f, CAPM Alpha, Z-Score, Hurst Exponent,
   - Max Drawdown, Information Ratio, Volatility Position Sizing,
   - Edge Decay

Asset classification is used by:
  - Ingestion daemon (auto-create sources per asset type)
  - Session tracker (know which tool fetches price for each asset)
  - Trading engines (future: multi-exchange support)

HMM is used by:
  - Spot PositionManager (regime veto for LONG entries)
  - Futures PositionManager (regime veto for LONG + SHORT entries)

QuantMetrics is used by:
  - TradeStore.get_stats() — full quant report on closed trades
  - Trading engines — edge validation before entering trades
  - Runtime status — live quant health dashboard
"""

import hashlib
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


# =============================================================================
# Asset Classification — detect type + correct tool for any symbol
# =============================================================================

# Crypto tickers/names → canonical name (for CoinGecko API)
_CRYPTO_MAP = {
    "btc": "bitcoin", "bitcoin": "bitcoin",
    "eth": "ethereum", "ethereum": "ethereum",
    "sol": "solana", "solana": "solana",
    "doge": "dogecoin", "dogecoin": "dogecoin",
    "xrp": "xrp", "ripple": "xrp",
    "ada": "cardano", "cardano": "cardano",
    "bnb": "bnb", "binance": "bnb",
    "dot": "polkadot", "polkadot": "polkadot",
    "avax": "avalanche", "avalanche": "avalanche",
    "ltc": "litecoin", "litecoin": "litecoin",
    "link": "chainlink", "chainlink": "chainlink",
    "matic": "polygon", "polygon": "polygon",
    "uni": "uniswap", "uniswap": "uniswap",
    "shib": "shiba-inu", "shiba": "shiba-inu",
    "trx": "tron", "tron": "tron",
    "near": "near",
    "apt": "aptos", "aptos": "aptos",
    "sui": "sui",
    "arb": "arbitrum", "arbitrum": "arbitrum",
    "op": "optimism", "optimism": "optimism",
    "atom": "cosmos", "cosmos": "cosmos",
    "xlm": "stellar", "stellar": "stellar",
    "fil": "filecoin", "filecoin": "filecoin",
    "hbar": "hedera", "hedera": "hedera",
    "pepe": "pepe",
    "kas": "kaspa", "kaspa": "kaspa",
    "ton": "toncoin", "toncoin": "toncoin",
    "icp": "internet-computer",
    "fet": "fetch-ai",
    "ren": "ren", "sand": "the-sandbox",
    "mana": "decentraland", "aave": "aave",
}

# Commodity names → canonical name
_COMMODITY_MAP = {
    "gold": "gold", "xau": "gold", "xauusd": "gold",
    "silver": "silver", "xag": "silver", "xagusd": "silver",
    "platinum": "platinum", "xpt": "platinum",
    "palladium": "palladium",
    "oil": "crude oil", "crude": "crude oil", "crude oil": "crude oil",
    "wti": "crude oil", "brent": "brent crude",
    "natural gas": "natural gas", "natgas": "natural gas",
    "copper": "copper",
    # MCX commodity symbols (Upstox / Indian exchanges)
    "goldm": "gold", "goldguinea": "gold",
    "silverm": "silver", "silvermic": "silver",
    "crudeoil": "crude oil", "crudeoilm": "crude oil",
    "naturalgas": "natural gas",
    "zinc": "zinc", "lead": "lead",
    "aluminium": "aluminium", "nickel": "nickel",
    "cottoncandy": "cotton", "cotton": "cotton",
    "menthaoil": "mentha oil",
}

# Forex pairs
_FOREX_MAP = {
    "eur/usd": "EUR/USD", "eurusd": "EUR/USD",
    "gbp/usd": "GBP/USD", "gbpusd": "GBP/USD",
    "usd/jpy": "USD/JPY", "usdjpy": "USD/JPY",
    "usd/cad": "USD/CAD", "usdcad": "USD/CAD",
    "aud/usd": "AUD/USD", "audusd": "AUD/USD",
    "nzd/usd": "NZD/USD", "nzdusd": "NZD/USD",
    "usd/chf": "USD/CHF", "usdchf": "USD/CHF",
    "eur/gbp": "EUR/GBP", "eurgbp": "EUR/GBP",
    "eur/jpy": "EUR/JPY", "eurjpy": "EUR/JPY",
    # INR pairs (Upstox currency derivatives)
    "usd/inr": "USD/INR", "usdinr": "USD/INR",
    "eur/inr": "EUR/INR", "eurinr": "EUR/INR",
    "gbp/inr": "GBP/INR", "gbpinr": "GBP/INR",
    "jpy/inr": "JPY/INR", "jpyinr": "JPY/INR",
}

# Well-known stock tickers (subset — unknown tickers default to stock)
_KNOWN_STOCKS = {
    # US stocks
    "aapl", "msft", "googl", "goog", "amzn", "meta", "tsla", "nvda",
    "amd", "intc", "nflx", "dis", "ba", "jpm", "gs", "v", "ma",
    "pypl", "sq", "shop", "crm", "orcl", "ibm", "csco", "adbe",
    "uber", "lyft", "abnb", "coin", "hood", "pltr", "snow",
    "spy", "qqq", "dia", "iwm",  # ETFs
    # Indian stocks (NSE)
    "reliance", "tcs", "infy", "hdfcbank", "icicibank", "sbin",
    "bhartiartl", "hindunilvr", "itc", "kotakbank", "wipro", "hcltech",
    "tatamotors", "maruti", "bajfinance", "bajfinsv", "axisbank",
    "sunpharma", "titan", "ultracemco", "ntpc", "powergrid",
    "ongc", "adanient", "adaniports", "techm", "tatasteel",
    "jswsteel", "indusindbk", "asianpaint", "nestleind", "drreddy",
    "divislab", "cipla", "bpcl", "coalindia", "grasim", "heromotoco",
    "eichermot", "lt", "m&m", "britannia", "apollohosp",
    # Indian indices
    "nifty", "nifty 50", "banknifty", "nifty bank", "sensex",
    "finnifty", "nifty fin service",
}


@dataclass
class AssetType:
    """Classified asset with tool routing info."""
    symbol: str          # Original symbol as given (e.g., "BTC", "AAPL")
    asset_type: str      # "crypto", "stock", "commodity", "forex"
    canonical: str       # Canonical name for API calls
    display: str         # Human-readable display name
    price_tool: str      # Tool to call for price
    price_query: str     # Query arg for price tool
    ta_tool: str         # Tool for TA (multi_tf_analysis works for all)
    ta_query: str        # Query arg for TA tool
    is_crypto: bool      # True if traded on Binance as USDT pair


def classify_asset(symbol: str) -> AssetType:
    """Classify any symbol into asset type with correct tool routing.

    Args:
        symbol: "BTC", "AAPL", "gold", "EUR/USD", etc.

    Returns:
        AssetType with all routing info for ingestion/sessions/trading.
    """
    sym_lower = symbol.lower().strip()

    # Check crypto
    if sym_lower in _CRYPTO_MAP:
        canonical = _CRYPTO_MAP[sym_lower]
        return AssetType(
            symbol=symbol, asset_type="crypto", canonical=canonical,
            display=symbol.upper(), price_tool="crypto_price",
            price_query=canonical, ta_tool="multi_tf_analysis",
            ta_query=symbol.upper(), is_crypto=True,
        )

    # Check commodity
    if sym_lower in _COMMODITY_MAP:
        canonical = _COMMODITY_MAP[sym_lower]
        return AssetType(
            symbol=symbol, asset_type="commodity", canonical=canonical,
            display=canonical.title(), price_tool="commodities",
            price_query=canonical, ta_tool="multi_tf_analysis",
            ta_query=canonical, is_crypto=False,
        )

    # Check forex
    if sym_lower in _FOREX_MAP:
        canonical = _FOREX_MAP[sym_lower]
        return AssetType(
            symbol=symbol, asset_type="forex", canonical=canonical,
            display=canonical, price_tool="forex_rates",
            price_query="USD", ta_tool="multi_tf_analysis",
            ta_query=canonical, is_crypto=False,
        )

    # Default: treat as stock
    canonical = symbol.upper()
    return AssetType(
        symbol=symbol, asset_type="stock", canonical=canonical,
        display=canonical, price_tool="stock_quote",
        price_query=canonical, ta_tool="multi_tf_analysis",
        ta_query=canonical, is_crypto=False,
    )


def build_ingestion_sources(assets: List[str]) -> List[dict]:
    """Build ingestion source configs for a list of assets.

    Creates sources for ALL relevant tools per asset — not just price + TA,
    but also sentiment, news, funding, polymarket, etc. This means the
    knowledge tree has COMPREHENSIVE data and the answer path doesn't need
    to call tools at query time.

    Returns list of dicts ready for IngestionSource creation:
        [{"name": ..., "tool": ..., "query": ..., "interval": ...,
          "priority": ..., "asset_type": ...}, ...]
    """
    # Full tool groups per asset type — mirrors asset_context.ASSET_TOOL_GROUPS
    # but with ingestion-specific intervals and priorities.
    # Format: (tool_name, query_template, interval_minutes, priority)
    _INGESTION_GROUPS = {
        "crypto": [
            ("crypto_price",      "{price_query}",          5,   1),
            ("binance_price",     "{price_query}",          5,   1),
            ("multi_tf_analysis", "{symbol}",               5,   1),
            ("fear_greed",        "crypto",                 5,   2),
            ("funding_rate",      "{symbol}",               5,   2),
            ("open_interest",     "{symbol}",               5,   2),
            ("global_market",     "crypto",                 5,   2),
            ("news_search",       "{symbol} crypto news",   15,  3),
            ("polymarket",        "{symbol} crypto",        15,  3),
            ("economic_calendar", "upcoming",               60,  3),
        ],
        "stock": [
            ("stock_quote",       "{price_query}",          15,  2),
            ("multi_tf_analysis", "{symbol}",               15,  2),
            ("market_indices",    "indices",                60,  2),
            ("news_search",       "{symbol} stock news",    15,  3),
            ("polymarket",        "{symbol}",               60,  3),
            ("economic_calendar", "upcoming",               60,  3),
        ],
        "commodity": [
            ("commodities",       "{price_query}",          5,   2),
            ("multi_tf_analysis", "{symbol}",               5,   2),
            ("forex_rates",       "USD",                    5,   2),
            ("news_search",       "{symbol} price news",    15,  3),
            ("polymarket",        "{symbol}",               60,  3),
            ("economic_calendar", "upcoming",               60,  3),
            ("world_economy",     "top economies",          1440, 4),
        ],
        "forex": [
            ("forex_rates",       "{price_query}",          5,   2),
            ("multi_tf_analysis", "{symbol}",               5,   2),
            ("news_search",       "{symbol} forex news",    15,  3),
            ("polymarket",        "{symbol}",               60,  3),
            ("economic_calendar", "upcoming",               60,  3),
        ],
    }

    sources = []
    seen = set()  # Deduplicate by tool:query key

    for symbol in assets:
        at = classify_asset(symbol)
        group = _INGESTION_GROUPS.get(at.asset_type, [])
        slug = at.display.lower().replace(' ', '_').replace('/', '_')

        for tool_name, query_tmpl, interval, priority in group:
            # Fill in query template
            query = (query_tmpl
                     .replace("{price_query}", at.price_query)
                     .replace("{symbol}", at.display))
            dedup_key = f"{tool_name}:{query}"
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            sources.append({
                "name": f"{slug}_{tool_name}",
                "tool": tool_name,
                "query": query,
                "interval": interval,
                "priority": priority,
                "asset_type": at.asset_type,
            })

    return sources


def build_session_assets(assets: List[str]) -> List[Tuple[str, str, str, bool]]:
    """Build session tracker asset list from configured symbols.

    Returns list of (tool_name, query, display_name, is_crypto) tuples
    matching the format expected by SessionTracker._ASSETS.
    """
    result = []
    seen = set()

    for symbol in assets:
        at = classify_asset(symbol)
        key = at.canonical
        if key in seen:
            continue
        seen.add(key)
        result.append((at.price_tool, at.price_query, at.display, at.is_crypto))

    return result


# =============================================================================
# Market Hours — is the market open for this asset type?
# =============================================================================

# US stock market holidays (NYSE/NASDAQ) — fixed dates for 2025-2026
# Updated annually. Format: (month, day).
_US_MARKET_HOLIDAYS = {
    # 2025
    (1, 1), (1, 20), (2, 17), (4, 18), (5, 26), (6, 19),
    (7, 4), (9, 1), (11, 27), (12, 25),
    # 2026
    (1, 1), (1, 19), (2, 16), (4, 3), (5, 25), (6, 19),
    (7, 3), (9, 7), (11, 26), (12, 25),
}


def is_market_open(asset_type: str, now_utc=None) -> dict:
    """Check if the market for a given asset type is currently open.

    Args:
        asset_type: "crypto", "stock", "commodity", "forex"
        now_utc: datetime in UTC (default: now)

    Returns:
        {"open": bool, "reason": str, "next_open": str}
        - open: True if market is active right now
        - reason: human-readable explanation
        - next_open: when it opens next (approximate)
    """
    from datetime import datetime, timezone

    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    weekday = now_utc.weekday()  # 0=Mon, 6=Sun
    hour = now_utc.hour
    month_day = (now_utc.month, now_utc.day)

    # Crypto: 24/7/365 — always open
    if asset_type == "crypto":
        return {"open": True, "reason": "Crypto markets trade 24/7"}

    # Stocks: Mon-Fri, 14:30-21:00 UTC (9:30 AM - 4:00 PM ET)
    # Pre-market: 09:00-14:30 UTC, After-hours: 21:00-01:00 UTC
    if asset_type == "stock":
        if weekday >= 5:  # Saturday or Sunday
            return {"open": False,
                    "reason": "Stock market closed (weekend)",
                    "next_open": "Monday 14:30 UTC (9:30 AM ET)"}
        if month_day in _US_MARKET_HOLIDAYS:
            return {"open": False,
                    "reason": "Stock market closed (US holiday)",
                    "next_open": "Next business day 14:30 UTC"}
        if 14 <= hour < 21:
            return {"open": True,
                    "reason": "US stock market open (regular hours)"}
        elif 9 <= hour < 14:
            return {"open": True,
                    "reason": "US stock market pre-market hours"}
        elif hour >= 21 or hour < 1:
            return {"open": True,
                    "reason": "US stock market after-hours"}
        return {"open": False,
                "reason": "Stock market closed (outside trading hours)",
                "next_open": "14:30 UTC (9:30 AM ET)"}

    # Forex: Sun 22:00 UTC - Fri 22:00 UTC (continuous)
    if asset_type == "forex":
        # Closed: Friday 22:00 UTC → Sunday 22:00 UTC
        if weekday == 5:  # Saturday — fully closed
            return {"open": False,
                    "reason": "Forex market closed (weekend)",
                    "next_open": "Sunday 22:00 UTC"}
        if weekday == 6 and hour < 22:  # Sunday before 22:00
            return {"open": False,
                    "reason": "Forex market closed (weekend)",
                    "next_open": "Sunday 22:00 UTC"}
        if weekday == 4 and hour >= 22:  # Friday after 22:00
            return {"open": False,
                    "reason": "Forex market closed (weekend)",
                    "next_open": "Sunday 22:00 UTC"}
        return {"open": True,
                "reason": "Forex market open (24h Mon-Fri)"}

    # Commodities: similar to stocks, varies by exchange
    # COMEX gold/silver: Mon-Fri, 23:00-22:00 UTC (nearly 24h with 1h break)
    # Simplified: Mon-Fri trading hours
    if asset_type == "commodity":
        if weekday >= 5:  # Weekend
            return {"open": False,
                    "reason": "Commodity market closed (weekend)",
                    "next_open": "Sunday 23:00 UTC"}
        if month_day in _US_MARKET_HOLIDAYS:
            return {"open": False,
                    "reason": "Commodity market closed (US holiday)",
                    "next_open": "Next business day"}
        return {"open": True,
                "reason": "Commodity market open (Mon-Fri)"}

    # Unknown — assume open
    return {"open": True, "reason": "Market hours unknown for this type"}


def market_status_note(asset_type: str, asset_name: str = "",
                       now_utc=None) -> str:
    """Get a human-readable market status note for prompt injection.

    Returns empty string if market is open, or a note like:
    "Note: Gold commodity market is closed (weekend). Prices shown are
    from last market close."
    """
    status = is_market_open(asset_type, now_utc)
    if status["open"]:
        return ""
    name_str = f"{asset_name} " if asset_name else ""
    return (f"Note: {name_str}{asset_type} market is currently closed — "
            f"{status['reason']}. Prices shown are from last market close. "
            f"Next open: {status.get('next_open', 'next business day')}.")


# =============================================================================
# Constants — HMM
# =============================================================================

# Hidden states
STRONG_BULL = 0
BULL = 1
CHOPPY = 2
BEAR = 3
STRONG_BEAR = 4

STATE_NAMES = ["STRONG_BULL", "BULL", "CHOPPY", "BEAR", "STRONG_BEAR"]
N_STATES = 5

# Observation dimensions
OBS_RETURN = 0       # Log return (%)
OBS_VOLATILITY = 1   # ATR / price (normalized)
OBS_RSI = 2          # RSI normalized to [-1, 1]  (0 = RSI 50)
OBS_MACD = 3         # MACD histogram / ATR (normalized)
OBS_VOLUME = 4       # Relative volume (vs average)
OBS_MOMENTUM = 5     # EMA21 - EMA50 direction (normalized)
N_OBS = 6


# =============================================================================
# GaussianHMM — Core HMM with continuous Gaussian emissions
# =============================================================================

class GaussianHMM:
    """Hidden Markov Model with multivariate Gaussian emissions.

    Pure numpy implementation — no external ML libraries needed.

    Parameters:
        n_states: Number of hidden states
        n_obs: Observation dimensionality
        pi: Initial state distribution (n_states,)
        A: Transition matrix (n_states, n_states) — A[i,j] = P(j|i)
        means: Emission means (n_states, n_obs)
        covars: Emission covariances (n_states, n_obs, n_obs) — diagonal
    """

    def __init__(self, n_states: int = N_STATES, n_obs: int = N_OBS):
        if not _HAS_NUMPY:
            raise ImportError("numpy required for GaussianHMM")

        self.n_states = n_states
        self.n_obs = n_obs

        # Initial state distribution — start uniform
        self.pi = np.ones(n_states) / n_states

        # Transition matrix — initialized with persistence bias
        # Markets tend to stay in their current regime
        self.A = np.full((n_states, n_states), 0.05 / (n_states - 1))
        np.fill_diagonal(self.A, 0.85)
        # Normalize rows
        self.A /= self.A.sum(axis=1, keepdims=True)

        # Emission means — each state has characteristic observation profile
        self.means = np.zeros((n_states, n_obs))
        self._init_emission_means()

        # Emission variances — diagonal only (independent observations)
        # Shape: (n_states, n_obs) instead of (n_states, n_obs, n_obs)
        self.covars = np.full((n_states, n_obs), 0.5)  # moderate variance

        self._trained = False
        self._train_count = 0

    def _init_emission_means(self):
        """Set sensible initial emission means per state.

        Observation vector: [return, volatility, rsi_norm, macd_norm,
                             rel_volume, momentum]
        """
        # STRONG_BULL: high returns, low vol, high RSI, positive MACD
        self.means[STRONG_BULL] = [0.5, -0.3, 0.5, 0.5, 0.3, 0.6]
        # BULL: moderate positive returns
        self.means[BULL] = [0.2, -0.1, 0.2, 0.2, 0.1, 0.3]
        # CHOPPY: near-zero returns, moderate vol
        self.means[CHOPPY] = [0.0, 0.1, 0.0, 0.0, 0.0, 0.0]
        # BEAR: negative returns, higher vol
        self.means[BEAR] = [-0.2, 0.2, -0.2, -0.2, 0.1, -0.3]
        # STRONG_BEAR: large negative returns, high vol, panic
        self.means[STRONG_BEAR] = [-0.5, 0.5, -0.5, -0.5, 0.5, -0.6]

    def _log_gaussian_pdf(self, x, mean, var):
        """Log probability of x under multivariate Gaussian (diagonal covariance).

        Args:
            x: observation vector (n_obs,)
            mean: state mean (n_obs,)
            var: diagonal variances (n_obs,) — NOT full covariance matrix
        """
        d = self.n_obs
        diff = x - mean
        var = np.clip(var, 1e-6, None)  # prevent log(0)
        log_det = np.sum(np.log(var))
        mahal = np.sum(diff ** 2 / var)
        return -0.5 * (d * np.log(2 * np.pi) + log_det + mahal)

    def _compute_log_emission(self, observations):
        """Compute log emission probabilities for all states and time steps.

        Vectorized: computes all (T, n_states) at once using diagonal covariance.

        Args:
            observations: (T, n_obs) array

        Returns:
            log_B: (T, n_states) array of log P(o_t | state_s)
        """
        T = len(observations)
        d = self.n_obs
        var = np.clip(self.covars, 1e-6, None)  # (n_states, n_obs)
        log_det = np.sum(np.log(var), axis=1)   # (n_states,)
        log_B = np.zeros((T, self.n_states))
        for s in range(self.n_states):
            diff = observations - self.means[s]  # (T, n_obs)
            mahal = np.sum(diff ** 2 / var[s], axis=1)  # (T,)
            log_B[:, s] = -0.5 * (d * np.log(2 * np.pi) + log_det[s] + mahal)
        return log_B

    def _forward(self, log_B):
        """Forward algorithm in log space (vectorized inner loop).

        Returns:
            log_alpha: (T, n_states) — forward probabilities
            log_likelihood: scalar — log P(observations | model)
        """
        T = log_B.shape[0]
        N = self.n_states
        log_alpha = np.full((T, N), -np.inf)
        log_A = np.log(np.clip(self.A, 1e-300, None))
        log_pi = np.log(np.clip(self.pi, 1e-300, None))

        # t=0
        log_alpha[0] = log_pi + log_B[0]

        # t=1..T-1  (vectorized: broadcast alpha over transition matrix)
        for t in range(1, T):
            # log_alpha[t-1][:, None] + log_A is (N, N), logsumexp over axis=0
            M = log_alpha[t - 1][:, None] + log_A  # (N, N)
            c = M.max(axis=0)
            log_alpha[t] = c + np.log(np.sum(np.exp(M - c), axis=0)) + log_B[t]

        log_likelihood = _logsumexp(log_alpha[-1])
        return log_alpha, log_likelihood

    def _backward(self, log_B):
        """Backward algorithm in log space (vectorized inner loop).

        Returns:
            log_beta: (T, n_states) — backward probabilities
        """
        T = log_B.shape[0]
        N = self.n_states
        log_beta = np.full((T, N), -np.inf)
        log_A = np.log(np.clip(self.A, 1e-300, None))

        # t=T-1
        log_beta[-1] = 0.0  # log(1) = 0

        # t=T-2..0  (vectorized: broadcast over states)
        for t in range(T - 2, -1, -1):
            # log_A + (log_B[t+1] + log_beta[t+1]) is (N, N), logsumexp over axis=1
            M = log_A + (log_B[t + 1] + log_beta[t + 1])  # (N, N) broadcast
            c = M.max(axis=1)
            log_beta[t] = c + np.log(np.sum(np.exp(M - c[:, None]), axis=1))

        return log_beta

    def viterbi(self, observations):
        """Viterbi algorithm — find most likely state sequence.

        Args:
            observations: (T, n_obs) array

        Returns:
            states: (T,) array of most likely states
            log_prob: log probability of the best path
        """
        observations = np.asarray(observations, dtype=np.float64)
        T = len(observations)
        if T == 0:
            return np.array([], dtype=int), -np.inf

        log_B = self._compute_log_emission(observations)
        log_A = np.log(np.clip(self.A, 1e-300, None))
        log_pi = np.log(np.clip(self.pi, 1e-300, None))

        # Viterbi tables
        V = np.full((T, self.n_states), -np.inf)
        backptr = np.zeros((T, self.n_states), dtype=int)

        # t=0
        V[0] = log_pi + log_B[0]

        # t=1..T-1  (vectorized over states)
        for t in range(1, T):
            scores = V[t - 1][:, None] + log_A  # (N, N)
            backptr[t] = np.argmax(scores, axis=0)
            V[t] = scores[backptr[t], np.arange(self.n_states)] + log_B[t]

        # Backtrace
        states = np.zeros(T, dtype=int)
        states[-1] = np.argmax(V[-1])
        for t in range(T - 2, -1, -1):
            states[t] = backptr[t + 1, states[t + 1]]

        return states, float(np.max(V[-1]))

    def predict_proba(self, observations):
        """Get state probabilities for the last time step.

        Args:
            observations: (T, n_obs) array

        Returns:
            proba: (n_states,) array — P(state | observations)
        """
        observations = np.asarray(observations, dtype=np.float64)
        if len(observations) == 0:
            return self.pi.copy()

        log_B = self._compute_log_emission(observations)
        log_alpha, _ = self._forward(log_B)

        # Normalize last alpha to get posterior
        log_last = log_alpha[-1]
        log_last -= _logsumexp(log_last)  # normalize
        return np.exp(log_last)

    def fit(self, observations, n_iter: int = 20, tol: float = 1e-4):
        """Train HMM using Baum-Welch (EM) algorithm.

        Args:
            observations: (T, n_obs) array of training data
            n_iter: Maximum EM iterations
            tol: Convergence tolerance (log-likelihood change)

        Returns:
            log_likelihoods: list of log-likelihoods per iteration
        """
        observations = np.asarray(observations, dtype=np.float64)
        T = len(observations)
        if T < 3:
            logger.warning("HMM fit: need at least 3 observations")
            return []

        log_lls = []
        prev_ll = -np.inf

        for iteration in range(n_iter):
            # E-step: compute responsibilities
            log_B = self._compute_log_emission(observations)
            log_alpha, log_ll = self._forward(log_B)
            log_beta = self._backward(log_B)

            log_lls.append(log_ll)

            # Check convergence
            if abs(log_ll - prev_ll) < tol and iteration > 0:
                break
            prev_ll = log_ll

            # Posterior state probabilities: gamma[t,s] = P(state=s at t | obs)
            log_gamma = log_alpha + log_beta
            for t in range(T):
                log_gamma[t] -= _logsumexp(log_gamma[t])
            gamma = np.exp(log_gamma)

            # Transition responsibilities: xi[t,i,j] = P(i at t, j at t+1 | obs)
            # Vectorized: compute all (i,j) pairs at once per timestep
            log_A = np.log(np.clip(self.A, 1e-300, None))
            xi = np.zeros((T - 1, self.n_states, self.n_states))
            for t in range(T - 1):
                # (N,1) + (N,N) + (1,N) + (1,N) - scalar = (N,N)
                log_xi_t = (log_alpha[t, :, None] + log_A +
                            log_B[t + 1, None, :] + log_beta[t + 1, None, :] - log_ll)
                xi[t] = np.exp(log_xi_t)

            # M-step: update parameters
            # Initial distribution
            self.pi = gamma[0] + 1e-10
            self.pi /= self.pi.sum()

            # Transition matrix
            for i in range(self.n_states):
                denom = gamma[:-1, i].sum() + 1e-10
                for j in range(self.n_states):
                    self.A[i, j] = xi[:, i, j].sum() / denom
            # Normalize + floor
            self.A = np.clip(self.A, 1e-6, None)
            self.A /= self.A.sum(axis=1, keepdims=True)

            # Emission means and covariances
            for s in range(self.n_states):
                weight = gamma[:, s]
                denom = weight.sum() + 1e-10

                # Mean
                self.means[s] = (weight[:, None] * observations).sum(
                    axis=0) / denom

                # Variance (diagonal only — stored as 1D vector per state)
                diff = observations - self.means[s]
                weighted_sq = (weight[:, None] * diff ** 2).sum(axis=0) / denom
                self.covars[s] = np.clip(weighted_sq, 1e-4, None)

        self._trained = True
        self._train_count += 1
        return log_lls

    def save(self, path: str):
        """Save model parameters to JSON."""
        data = {
            "n_states": self.n_states,
            "n_obs": self.n_obs,
            "pi": self.pi.tolist(),
            "A": self.A.tolist(),
            "means": self.means.tolist(),
            "covars": [self.covars[s].tolist() for s in range(self.n_states)],
            "trained": self._trained,
            "train_count": self._train_count,
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str) -> bool:
        """Load model parameters from JSON. Returns True if loaded."""
        if not os.path.exists(path):
            return False
        try:
            with open(path) as f:
                data = json.load(f)
            self.pi = np.array(data["pi"])
            self.A = np.array(data["A"])
            self.means = np.array(data["means"])
            covars = np.array(data["covars"])
            # Backward compat: old format stored full (n_states, n_obs, n_obs)
            if covars.ndim == 3:
                # Extract diagonal from each state's covariance matrix
                self.covars = np.array([np.diag(covars[s]) for s in range(covars.shape[0])])
            else:
                self.covars = covars
            self._trained = data.get("trained", True)
            self._train_count = data.get("train_count", 1)
            return True
        except Exception as e:
            logger.warning("HMM load failed: %s", e)
            return False


def _logsumexp(x):
    """Numerically stable log-sum-exp."""
    x = np.asarray(x)
    c = x.max()
    if c == -np.inf:
        return -np.inf
    return c + np.log(np.sum(np.exp(x - c)))


# =============================================================================
# MarketHMM — Trading-specific HMM wrapper
# =============================================================================

@dataclass
class HMMSignal:
    """Output from MarketHMM.get_signal()."""
    state: int                    # Current state index (0-4)
    state_name: str               # Human-readable state name
    prev_state: int               # Previous state index
    prev_state_name: str          # Previous state name
    probabilities: Dict[str, float]  # State name → probability
    confidence: float             # Probability of current state (0-1)
    signal: str                   # "BUY", "SELL", "SHORT", "HOLD"
    signal_strength: float        # -1.0 (strong sell) to +1.0 (strong buy)
    regime_age: int               # How many ticks in current regime
    transition: bool              # True if state just changed


class MarketHMM:
    """Trading-specific HMM for market regime detection.

    Wraps GaussianHMM with:
    - Observation builder from TA analysis dict
    - Per-symbol state tracking (regime age, transitions)
    - Signal generation for entry/exit decisions
    - Training from graph snapshot nodes
    - Periodic retraining

    Usage:
        hmm = MarketHMM(data_dir="qor-data")
        hmm.load()  # Load saved model (if exists)

        # Every 5-min tick:
        signal = hmm.get_signal(analysis_dict, symbol="BTC")
        if signal.signal == "BUY" and signal.confidence > 0.7:
            # Enter long
        elif signal.signal == "SHORT" and signal.confidence > 0.7:
            # Enter short (futures)

        # Periodic retraining:
        hmm.train_from_graph(graph)
        hmm.save()
    """

    # Minimum observations needed before HMM output is trusted.
    # 30 provides stable z-score normalization (was 10, too noisy for warmup).
    MIN_HISTORY = 30
    # Retrain after this many new observations
    RETRAIN_INTERVAL = 500

    def __init__(self, data_dir: str = "qor-data"):
        self._data_dir = data_dir
        self._model = GaussianHMM(N_STATES, N_OBS) if _HAS_NUMPY else None

        # Per-symbol observation history (rolling window)
        self._obs_history: Dict[str, list] = {}
        self._state_history: Dict[str, list] = {}
        self._max_history = 1000

        # Per-symbol regime tracking
        self._current_state: Dict[str, int] = {}
        self._regime_age: Dict[str, int] = {}

        # Training tracking
        self._obs_since_train = 0
        self._last_train_time = 0.0

        # Per-symbol previous prices for actual return calculation
        self._prev_prices: Dict[str, float] = {}

    @property
    def is_available(self) -> bool:
        """True if numpy is available and model exists."""
        return self._model is not None

    @property
    def is_trained(self) -> bool:
        """True if model has been trained at least once."""
        return self._model is not None and self._model._trained

    def _model_path(self) -> str:
        return os.path.join(self._data_dir, "hmm_model.json")

    def save(self):
        """Save trained model to disk."""
        if self._model:
            self._model.save(self._model_path())
            logger.info("[HMM] Model saved to %s", self._model_path())

    def load(self) -> bool:
        """Load model from disk. Returns True if loaded."""
        if not self._model:
            return False
        loaded = self._model.load(self._model_path())
        if loaded:
            logger.info("[HMM] Model loaded (trained=%s, count=%d)",
                        self._model._trained, self._model._train_count)
        return loaded

    # -----------------------------------------------------------------
    # Pre-warm from historical klines (skip 2.5h cold start)
    # -----------------------------------------------------------------

    def warm_from_klines(self, klines: list, symbol: str = "BTC"):
        """Feed historical klines to build observation history, skipping cold start.

        Args:
            klines: Binance kline format list:
                [[open_time, open, high, low, close, volume, ...], ...]
            symbol: Trading symbol for per-symbol state tracking.

        After calling this, get_signal() will produce real signals immediately
        instead of returning neutral for the first 30 ticks (~2.5 hours).
        """
        if not self.is_available or not _HAS_NUMPY:
            return
        if not klines:
            return

        import math
        fed = 0
        prev_close = 0.0
        for k in klines:
            try:
                o, h, low, close, vol = (
                    float(k[1]), float(k[2]), float(k[3]),
                    float(k[4]), float(k[5]),
                )
            except (IndexError, ValueError, TypeError):
                continue
            if close <= 0:
                continue

            # Build a minimal analysis dict from kline OHLCV
            atr_approx = h - low  # single-candle range as ATR proxy
            rsi_approx = 50.0     # neutral default (no multi-period RSI from single kline)
            if prev_close > 0:
                change = close - prev_close
                # Rough RSI proxy: positive change → RSI > 50, negative → < 50
                pct = change / prev_close
                rsi_approx = 50 + pct * 500  # scale: 1% move ≈ 5 RSI points
                rsi_approx = max(10, min(90, rsi_approx))

            ema_proxy = (o + close) / 2  # midpoint as EMA stand-in
            analysis = {
                "current": close,
                "atr": atr_approx,
                "rsi": rsi_approx,
                "macd_hist": 0,
                "ema21": ema_proxy,
                "ema50": ema_proxy,
                "bullish_tfs": 4 if close > o else 3,
                "total_tfs": 7,
            }
            obs = self.build_observation(analysis, symbol=symbol)
            if obs is not None:
                if symbol not in self._obs_history:
                    self._obs_history[symbol] = []
                    self._state_history[symbol] = []
                self._obs_history[symbol].append(obs)
                if len(self._obs_history[symbol]) > self._max_history:
                    self._obs_history[symbol] = self._obs_history[symbol][-self._max_history:]
                fed += 1
            prev_close = close

        if fed > 0:
            logger.info("[HMM] Warmed %s with %d historical candles "
                        "(need %d for signals)", symbol, fed, self.MIN_HISTORY)

    # -----------------------------------------------------------------
    # Observation builder — converts TA analysis dict to 6-dim vector
    # -----------------------------------------------------------------

    def build_observation(self, analysis: dict, symbol: str = "default") -> Optional[list]:
        """Convert TA analysis dict to 6-dim observation vector.

        Args:
            analysis: Parsed TA dict with RSI, ATR, MACD, etc.
                      (from PositionManager._parse_analysis)
            symbol: Trading symbol for per-symbol price tracking.

        Returns:
            [return_norm, vol_norm, rsi_norm, macd_norm, vol_rel, momentum]
            or None if insufficient data.
        """
        current = analysis.get("current", 0)
        if current <= 0:
            return None

        # 1. Actual return (log return from previous price, not RSI proxy)
        import math
        prev_price = self._prev_prices.get(symbol, current)
        if prev_price > 0 and current > 0:
            return_norm = math.log(current / prev_price) * 20  # scale for HMM
            return_norm = max(-2.0, min(2.0, return_norm))
        else:
            return_norm = 0.0
        self._prev_prices[symbol] = current

        # 2. Volatility (ATR as % of price, normalized)
        atr = (analysis.get("atr_1h") or analysis.get("atr_daily")
               or analysis.get("atr", 0))
        if atr > 0 and current > 0:
            vol_pct = atr / current  # typically 0.01 - 0.05
            vol_norm = (vol_pct - 0.02) / 0.02  # center at 2%, scale
            vol_norm = max(-2.0, min(2.0, vol_norm))
        else:
            vol_norm = 0.0

        # 3. RSI normalized to [-1, 1] where 0 = RSI 50
        rsi = analysis.get("rsi", 50)
        rsi_norm = (rsi - 50) / 50.0  # RSI 30→-0.4, RSI 70→+0.4

        # 4. MACD histogram normalized by ATR
        macd_hist = analysis.get("macd_hist", 0)
        if atr > 0 and macd_hist != 0:
            macd_norm = macd_hist / atr
            macd_norm = max(-2.0, min(2.0, macd_norm))
        else:
            macd_norm = 0.0

        # 5. Relative volume
        rel_vol = analysis.get("rel_vol", 0)
        if rel_vol == 0:
            # Proxy: if we have OBV direction, use that
            obv_dir = analysis.get("obv_dir", 0)
            rel_vol = obv_dir * 0.3  # mild signal

        # 6. Momentum (EMA21 vs EMA50 spread, normalized by price)
        ema21 = analysis.get("ema21", 0)
        ema50 = analysis.get("ema50", 0)
        if ema21 > 0 and ema50 > 0 and current > 0:
            momentum = (ema21 - ema50) / current * 100  # percentage spread
            momentum = max(-2.0, min(2.0, momentum))
        else:
            # Fallback: use TF confluence direction
            bullish = analysis.get("bullish_tfs", 0)
            total = analysis.get("total_tfs", 1)
            if total > 0:
                momentum = (bullish / total - 0.5) * 2  # [-1, 1]
            else:
                momentum = 0.0

        return [return_norm, vol_norm, rsi_norm, macd_norm, rel_vol, momentum]

    def _normalize_observation(self, obs: list, symbol: str) -> list:
        """Apply rolling z-score normalization using per-symbol history.

        Stabilizes inputs to the HMM so all dimensions have comparable scales
        regardless of the underlying asset's price or volatility characteristics.
        Falls back to raw observation if insufficient history.
        """
        if not _HAS_NUMPY:
            return obs
        history = self._obs_history.get(symbol, [])
        if len(history) < 20:
            return obs  # Not enough history for meaningful stats
        arr = np.array(history[-200:])  # Rolling window of last 200
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        std = np.clip(std, 1e-6, None)  # Prevent division by zero
        normalized = ((np.array(obs) - mean) / std).tolist()
        return normalized

    # -----------------------------------------------------------------
    # Signal generation
    # -----------------------------------------------------------------

    def get_signal(self, analysis: dict, symbol: str = "BTC") -> HMMSignal:
        """Get regime signal for a symbol.

        Args:
            analysis: Parsed TA analysis dict
            symbol: Trading symbol (for per-symbol state tracking)

        Returns:
            HMMSignal with state, confidence, signal, and metadata
        """
        # Default neutral signal if HMM not available
        if not self.is_available:
            return self._neutral_signal(symbol)

        # Build observation (pass symbol for per-symbol price tracking)
        obs = self.build_observation(analysis, symbol=symbol)
        if obs is None:
            return self._neutral_signal(symbol)

        # Add to per-symbol history
        if symbol not in self._obs_history:
            self._obs_history[symbol] = []
            self._state_history[symbol] = []
        self._obs_history[symbol].append(obs)
        if len(self._obs_history[symbol]) > self._max_history:
            self._obs_history[symbol] = self._obs_history[symbol][-self._max_history:]

        self._obs_since_train += 1

        # Need minimum history for reliable signal
        history = self._obs_history[symbol]
        if len(history) < self.MIN_HISTORY:
            return self._neutral_signal(symbol, reason="warming up")

        # Get state probabilities from forward algorithm
        # Apply rolling z-score normalization for stable HMM inputs
        recent = history[-100:]  # Use last 100 for speed
        normalized = [self._normalize_observation(o, symbol) for o in recent]
        obs_array = np.array(normalized)
        proba = self._model.predict_proba(obs_array)

        # Current state = highest probability
        current_state = int(np.argmax(proba))
        confidence = float(proba[current_state])

        # Track regime transitions
        prev_state = self._current_state.get(symbol, CHOPPY)
        transition = current_state != prev_state
        if transition:
            self._regime_age[symbol] = 0
        else:
            self._regime_age[symbol] = self._regime_age.get(symbol, 0) + 1
        self._current_state[symbol] = current_state

        # Track state history
        self._state_history[symbol].append(current_state)
        if len(self._state_history[symbol]) > self._max_history:
            self._state_history[symbol] = self._state_history[symbol][-self._max_history:]

        # Generate signal based on state + transition
        signal, strength = self._state_to_signal(
            current_state, prev_state, confidence, transition)

        return HMMSignal(
            state=current_state,
            state_name=STATE_NAMES[current_state],
            prev_state=prev_state,
            prev_state_name=STATE_NAMES[prev_state],
            probabilities={
                STATE_NAMES[i]: round(float(proba[i]), 4)
                for i in range(N_STATES)
            },
            confidence=round(confidence, 4),
            signal=signal,
            signal_strength=round(strength, 4),
            regime_age=self._regime_age.get(symbol, 0),
            transition=transition,
        )

    def _state_to_signal(self, state: int, prev_state: int,
                         confidence: float, transition: bool,
                         ) -> Tuple[str, float]:
        """Convert state + transition to trading signal.

        Returns:
            (signal_name, signal_strength)
            signal_strength: -1.0 (strong sell) to +1.0 (strong buy)
        """
        # Base strength from state
        state_strength = {
            STRONG_BULL: 0.9,
            BULL: 0.5,
            CHOPPY: 0.0,
            BEAR: -0.5,
            STRONG_BEAR: -0.9,
        }
        strength = state_strength[state] * confidence

        # Transition bonus — regime change is a stronger signal
        if transition:
            if state in (STRONG_BULL, BULL) and prev_state in (CHOPPY, BEAR, STRONG_BEAR):
                strength = max(strength, 0.6 * confidence)  # bullish transition
            elif state in (STRONG_BEAR, BEAR) and prev_state in (CHOPPY, BULL, STRONG_BULL):
                strength = min(strength, -0.6 * confidence)  # bearish transition

        # Map strength to signal
        if strength >= 0.3:
            signal = "BUY"
        elif strength <= -0.3:
            signal = "SHORT"  # Futures can use this; spot ignores it
        else:
            signal = "HOLD"

        return signal, strength

    def _neutral_signal(self, symbol: str, reason: str = "") -> HMMSignal:
        """Return a neutral HOLD signal."""
        return HMMSignal(
            state=CHOPPY,
            state_name="CHOPPY",
            prev_state=CHOPPY,
            prev_state_name="CHOPPY",
            probabilities={name: 0.2 for name in STATE_NAMES},
            confidence=0.2,
            signal="HOLD",
            signal_strength=0.0,
            regime_age=0,
            transition=False,
        )

    # -----------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------

    def train_from_observations(self, observations: list,
                                n_iter: int = 30) -> Dict[str, Any]:
        """Train HMM on a list of observation vectors.

        Args:
            observations: List of 6-dim observation vectors
            n_iter: Baum-Welch iterations

        Returns:
            {"trained": bool, "iterations": int, "observations": int,
             "final_ll": float}
        """
        if not self.is_available:
            return {"trained": False, "reason": "numpy not available"}

        if len(observations) < 20:
            return {"trained": False, "reason": "need at least 20 observations",
                    "observations": len(observations)}

        obs_array = np.array(observations, dtype=np.float64)
        log_lls = self._model.fit(obs_array, n_iter=n_iter)

        self._obs_since_train = 0
        self._last_train_time = time.time()

        result = {
            "trained": True,
            "iterations": len(log_lls),
            "observations": len(observations),
            "final_ll": round(log_lls[-1], 2) if log_lls else 0,
        }
        logger.info("[HMM] Trained on %d observations (%d iterations, "
                     "LL=%.2f)", len(observations), len(log_lls),
                     result["final_ll"])
        return result

    def train_from_graph(self, graph, symbols: Optional[List[str]] = None,
                         max_nodes: int = 5000) -> Dict[str, Any]:
        """Train HMM from historical snapshot nodes in the knowledge graph.

        Extracts observations from snapshot nodes that contain price/TA data.

        Args:
            graph: QORGraph instance
            symbols: List of symbols to train on (None = all)
            max_nodes: Maximum snapshot nodes to read

        Returns:
            Training result dict
        """
        if not self.is_available:
            return {"trained": False, "reason": "numpy not available"}

        if not graph or not graph.is_open:
            return {"trained": False, "reason": "graph not available"}

        observations = []
        try:
            nodes = graph.list_nodes(node_type="snapshot")
            count = 0
            for item in nodes:
                if count >= max_nodes:
                    break
                if isinstance(item, tuple):
                    nid, data = item
                else:
                    nid = item
                    data = graph.get_node(nid)
                if not data:
                    continue

                props = data.get("properties", {})
                content = props.get("content", "")

                # Filter by symbol if specified
                entity = props.get("entity", "")
                if symbols and entity not in symbols:
                    continue

                # Try to extract a pseudo-analysis dict from content
                obs = self._observation_from_content(content, entity)
                if obs is not None:
                    observations.append(obs)
                count += 1

        except Exception as e:
            logger.warning("[HMM] Graph scan error: %s", e)

        if not observations:
            # Fallback: use collected per-symbol history
            for sym_obs in self._obs_history.values():
                observations.extend(sym_obs)

        return self.train_from_observations(observations)

    def _observation_from_content(self, content: str,
                                  entity: str) -> Optional[list]:
        """Extract observation vector from snapshot node content text.

        Tries to parse price, RSI, ATR, MACD from the content string.
        Returns 6-dim observation or None.
        """
        import re

        # Try to extract key values
        price = 0.0
        rsi = 50.0
        atr = 0.0
        macd = 0.0

        # Price
        m = re.search(r'\$([0-9,]+\.?\d*)', content)
        if m:
            try:
                price = float(m.group(1).replace(",", ""))
            except ValueError:
                pass

        # RSI
        m = re.search(r'RSI[:\s]*([0-9.]+)', content, re.IGNORECASE)
        if m:
            try:
                rsi = float(m.group(1))
            except ValueError:
                pass

        # ATR
        m = re.search(r'ATR[:\s]*\$?([0-9,]+\.?\d*)', content, re.IGNORECASE)
        if m:
            try:
                atr = float(m.group(1).replace(",", ""))
            except ValueError:
                pass

        # MACD histogram
        m = re.search(r'(?:MACD|Hist)[:\s]*([-0-9,.]+)', content, re.IGNORECASE)
        if m:
            try:
                macd = float(m.group(1).replace(",", ""))
            except ValueError:
                pass

        if price <= 0:
            return None

        # Build observation
        rsi_norm = (rsi - 50) / 50.0
        vol_norm = (atr / price - 0.02) / 0.02 if price > 0 and atr > 0 else 0
        vol_norm = max(-2.0, min(2.0, vol_norm))
        macd_norm = macd / atr if atr > 0 else 0
        macd_norm = max(-2.0, min(2.0, macd_norm))

        # Return proxy from RSI (no previous price to compute actual return)
        return_norm = rsi_norm * 0.5

        return [return_norm, vol_norm, rsi_norm, macd_norm, 0.0, rsi_norm * 0.3]

    def should_retrain(self) -> bool:
        """Check if model should be retrained."""
        if not self.is_available:
            return False
        if not self._model._trained:
            return self._obs_since_train >= 50  # First train after 50 obs
        return self._obs_since_train >= self.RETRAIN_INTERVAL

    # -----------------------------------------------------------------
    # Status / info
    # -----------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        """Return HMM status for runtime status report."""
        result = {
            "available": self.is_available,
            "trained": self.is_trained,
            "train_count": self._model._train_count if self._model else 0,
            "obs_since_train": self._obs_since_train,
            "symbols_tracked": len(self._obs_history),
        }

        # Per-symbol current state
        states = {}
        for sym, state_id in self._current_state.items():
            hist_len = len(self._obs_history.get(sym, []))
            states[sym] = {
                "state": STATE_NAMES[state_id],
                "regime_age": self._regime_age.get(sym, 0),
                "history_len": hist_len,
            }
        result["symbol_states"] = states

        return result


# =============================================================================
# QuantMetrics — All institutional quant formulas
# =============================================================================

class QuantMetrics:
    """Institutional quant analytics computed from trade history.

    All formulas from the quant stack:
      1. Sharpe Ratio         — risk-adjusted return
      2. Sortino Ratio        — downside-risk-adjusted return
      3. Expectancy           — expected R per trade
      4. Risk of Ruin         — probability of account blowup
      5. Optimal f            — optimal fraction to risk (Ralph Vince)
      6. CAPM Alpha           — skill vs market return
      7. Z-Score              — mean reversion signal for a price series
      8. Hurst Exponent       — trending vs mean-reverting regime
      9. Max Drawdown         — worst peak-to-trough equity loss
     10. Information Ratio    — alpha consistency vs benchmark
     11. Volatility Sizing    — position size from target vol
     12. Edge Decay           — how fast the strategy edge erodes

    Usage:
        qm = QuantMetrics()
        report = qm.full_report(closed_trades)
        # report = {"sharpe": 1.8, "sortino": 2.1, "expectancy": 0.65, ...}

        # Individual formulas:
        sharpe = qm.sharpe_ratio(returns)
        z = qm.z_score(prices)
        h = qm.hurst_exponent(prices)
    """

    def __init__(self, risk_free_rate: float = 0.04,
                 annualize_factor: float = 365.0):
        """
        Args:
            risk_free_rate: Annual risk-free rate (default 4% — US T-bills).
            annualize_factor: Trading days per year.
                              365 for crypto (24/7), 252 for stocks.
        """
        self.rf = risk_free_rate
        self.ann = annualize_factor

    # -----------------------------------------------------------------
    # 1. Sharpe Ratio — return per unit of total risk
    # -----------------------------------------------------------------

    def sharpe_ratio(self, returns: List[float]) -> float:
        """Sharpe = (Rp - Rf) / σp.

        Args:
            returns: List of per-trade or per-period % returns.

        Returns:
            Annualized Sharpe ratio. >1 good, >2 strong, >3 elite.
        """
        if len(returns) < 2:
            return 0.0
        mean_r = sum(returns) / len(returns)
        std_r = _std(returns)
        if std_r <= 0:
            return 0.0
        # Annualize: multiply by sqrt(N) where N = trades/periods per year
        per_trade = (mean_r - self.rf / self.ann) / std_r
        return round(per_trade * math.sqrt(min(len(returns), self.ann)), 4)

    # -----------------------------------------------------------------
    # 2. Sortino Ratio — return per unit of downside risk only
    # -----------------------------------------------------------------

    def sortino_ratio(self, returns: List[float]) -> float:
        """Sortino = (Rp - Rf) / σd (downside deviation only).

        Better than Sharpe — ignores upside volatility.
        """
        if len(returns) < 2:
            return 0.0
        mean_r = sum(returns) / len(returns)
        downside = [r for r in returns if r < 0]
        if not downside:
            return 10.0  # No losses = perfect (capped)
        down_dev = _std(downside)
        if down_dev <= 0:
            return 10.0
        per_trade = (mean_r - self.rf / self.ann) / down_dev
        return round(per_trade * math.sqrt(min(len(returns), self.ann)), 4)

    # -----------------------------------------------------------------
    # 3. Expectancy — expected profit per trade in R-multiples
    # -----------------------------------------------------------------

    def expectancy(self, wins: int, losses: int,
                   avg_win: float, avg_loss: float) -> float:
        """E = (WinRate × AvgWin) - (LossRate × AvgLoss).

        Args:
            wins/losses: Trade counts.
            avg_win: Average winning trade % (positive).
            avg_loss: Average losing trade % (positive, absolute).

        Returns:
            Expectancy per trade. >0 = profitable system.
        """
        total = wins + losses
        if total == 0:
            return 0.0
        win_rate = wins / total
        loss_rate = losses / total
        return round(win_rate * avg_win - loss_rate * avg_loss, 4)

    # -----------------------------------------------------------------
    # 4. Risk of Ruin — probability of account blowup
    # -----------------------------------------------------------------

    def risk_of_ruin(self, win_rate: float, payoff_ratio: float,
                     risk_per_trade: float = 0.02) -> float:
        """RoR = ((1 - edge) / (1 + edge)) ^ (capital / risk).

        Args:
            win_rate: Fraction (0-1), e.g. 0.55.
            payoff_ratio: avg_win / avg_loss (e.g. 2.0).
            risk_per_trade: Fraction of capital risked per trade (e.g. 0.02 = 2%).

        Returns:
            Probability of ruin (0-1). <0.01 = safe, >0.10 = dangerous.
        """
        if win_rate <= 0 or win_rate >= 1 or payoff_ratio <= 0:
            return 1.0
        # Edge = expected return per unit risked
        edge = win_rate * payoff_ratio - (1 - win_rate)
        if edge <= 0:
            return 1.0  # Negative edge = guaranteed ruin
        ratio = (1 - edge) / (1 + edge)
        if ratio <= 0:
            return 0.0
        # capital / risk = how many consecutive losses to blow up
        units = 1.0 / max(risk_per_trade, 0.001)
        return round(max(0.0, min(1.0, ratio ** units)), 6)

    # -----------------------------------------------------------------
    # 5. Optimal f — Ralph Vince position sizing
    # -----------------------------------------------------------------

    def optimal_f(self, win_rate: float, payoff_ratio: float) -> float:
        """f = ((B × p) - q) / B — Kelly/Vince optimal fraction.

        Args:
            win_rate: p = probability of win (0-1).
            payoff_ratio: B = avg_win / avg_loss.

        Returns:
            Optimal fraction of capital to risk (0-1).
            Real-world: use f/2 or f/3 for safety (half-Kelly).
        """
        if payoff_ratio <= 0 or win_rate <= 0 or win_rate >= 1:
            return 0.0
        q = 1 - win_rate
        f = (payoff_ratio * win_rate - q) / payoff_ratio
        return round(max(0.0, min(1.0, f)), 4)

    # -----------------------------------------------------------------
    # 6. CAPM Alpha — strategy skill vs market
    # -----------------------------------------------------------------

    def capm_alpha(self, strategy_returns: List[float],
                   market_returns: List[float]) -> dict:
        """α = Rp - [Rf + β(Rm - Rf)].

        Args:
            strategy_returns: Per-period strategy returns.
            market_returns: Per-period market/benchmark returns (same length).

        Returns:
            {"alpha": float, "beta": float}
            alpha > 0 = real edge beyond market exposure.
        """
        n = min(len(strategy_returns), len(market_returns))
        if n < 5:
            return {"alpha": 0.0, "beta": 0.0}
        sr = strategy_returns[:n]
        mr = market_returns[:n]
        mean_s = sum(sr) / n
        mean_m = sum(mr) / n
        # β = Cov(Rs, Rm) / Var(Rm)
        cov = sum((sr[i] - mean_s) * (mr[i] - mean_m) for i in range(n)) / n
        var_m = sum((mr[i] - mean_m) ** 2 for i in range(n)) / n
        beta = cov / var_m if var_m > 0 else 0.0
        # α = Rp - [Rf + β(Rm - Rf)]
        rf_period = self.rf / self.ann
        alpha = mean_s - (rf_period + beta * (mean_m - rf_period))
        return {"alpha": round(alpha, 6), "beta": round(beta, 4)}

    # -----------------------------------------------------------------
    # 7. Z-Score — mean reversion signal
    # -----------------------------------------------------------------

    @staticmethod
    def z_score(prices: List[float], window: int = 20) -> float:
        """Z = (Price - Mean) / StdDev.

        Args:
            prices: Recent price series.
            window: Lookback period for mean/std.

        Returns:
            Z-score. >+2 = overbought (sell), <-2 = oversold (buy).
        """
        if len(prices) < window:
            return 0.0
        subset = prices[-window:]
        mean = sum(subset) / len(subset)
        std = _std(subset)
        if std <= 0:
            return 0.0
        return round((prices[-1] - mean) / std, 4)

    # -----------------------------------------------------------------
    # 8. Hurst Exponent — trending vs mean-reverting
    # -----------------------------------------------------------------

    @staticmethod
    def hurst_exponent(prices: List[float]) -> float:
        """H = log(R/S) / log(N) — Rescaled Range method.

        Returns:
            H > 0.5 = trending, H < 0.5 = mean-reverting, H ≈ 0.5 = random.
        """
        if len(prices) < 20:
            return 0.5
        # Log returns
        log_returns = []
        for i in range(1, len(prices)):
            if prices[i] > 0 and prices[i - 1] > 0:
                log_returns.append(math.log(prices[i] / prices[i - 1]))
        if len(log_returns) < 20:
            return 0.5

        # R/S analysis over multiple sub-periods
        n = len(log_returns)
        rs_values = []
        ns_values = []

        for size in [int(n / k) for k in (2, 4, 8, 16) if n / k >= 8]:
            rs_list = []
            for start in range(0, n - size + 1, size):
                chunk = log_returns[start:start + size]
                mean_c = sum(chunk) / len(chunk)
                # Cumulative deviation from mean
                cum_dev = []
                running = 0.0
                for r in chunk:
                    running += r - mean_c
                    cum_dev.append(running)
                # Range
                R = max(cum_dev) - min(cum_dev)
                # Standard deviation
                S = _std(chunk)
                if S > 0:
                    rs_list.append(R / S)
            if rs_list:
                avg_rs = sum(rs_list) / len(rs_list)
                if avg_rs > 0:
                    rs_values.append(math.log(avg_rs))
                    ns_values.append(math.log(size))

        if len(rs_values) < 2:
            return 0.5

        # Linear regression: log(R/S) = H × log(N) + c
        n_pts = len(rs_values)
        sum_x = sum(ns_values)
        sum_y = sum(rs_values)
        sum_xy = sum(ns_values[i] * rs_values[i] for i in range(n_pts))
        sum_x2 = sum(x ** 2 for x in ns_values)
        denom = n_pts * sum_x2 - sum_x ** 2
        if denom == 0:
            return 0.5
        H = (n_pts * sum_xy - sum_x * sum_y) / denom
        return round(max(0.0, min(1.0, H)), 4)

    # -----------------------------------------------------------------
    # 9. Maximum Drawdown — worst peak-to-trough loss
    # -----------------------------------------------------------------

    @staticmethod
    def max_drawdown(equity_curve: List[float]) -> dict:
        """MDD = (Peak - Trough) / Peak.

        Args:
            equity_curve: Cumulative equity values (e.g. [1000, 1050, 980, ...]).

        Returns:
            {"mdd_pct": float, "peak": float, "trough": float,
             "peak_idx": int, "trough_idx": int}
            mdd_pct < 20% = safe, > 40% = dangerous.
        """
        if len(equity_curve) < 2:
            return {"mdd_pct": 0.0, "peak": 0.0, "trough": 0.0,
                    "peak_idx": 0, "trough_idx": 0}
        peak = equity_curve[0]
        peak_idx = 0
        max_dd = 0.0
        dd_peak = peak
        dd_trough = peak
        dd_peak_idx = 0
        dd_trough_idx = 0

        for i, val in enumerate(equity_curve):
            if val > peak:
                peak = val
                peak_idx = i
            dd = (peak - val) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
                dd_peak = peak
                dd_trough = val
                dd_peak_idx = peak_idx
                dd_trough_idx = i

        return {
            "mdd_pct": round(max_dd * 100, 2),
            "peak": round(dd_peak, 2),
            "trough": round(dd_trough, 2),
            "peak_idx": dd_peak_idx,
            "trough_idx": dd_trough_idx,
        }

    # -----------------------------------------------------------------
    # 10. Information Ratio — alpha consistency
    # -----------------------------------------------------------------

    def information_ratio(self, strategy_returns: List[float],
                          benchmark_returns: List[float]) -> float:
        """IR = (Rp - Rb) / TrackingError.

        Measures consistency of beating benchmark.
        """
        n = min(len(strategy_returns), len(benchmark_returns))
        if n < 5:
            return 0.0
        excess = [strategy_returns[i] - benchmark_returns[i] for i in range(n)]
        mean_excess = sum(excess) / n
        te = _std(excess)
        if te <= 0:
            return 0.0
        return round(mean_excess / te * math.sqrt(min(n, self.ann)), 4)

    # -----------------------------------------------------------------
    # 11. Volatility Position Sizing
    # -----------------------------------------------------------------

    @staticmethod
    def volatility_position_size(target_vol: float, asset_vol: float,
                                 capital: float) -> float:
        """PositionSize = (TargetVolatility / AssetVolatility) × Capital.

        Args:
            target_vol: Target portfolio volatility (e.g. 0.10 = 10%).
            asset_vol: Asset's annualized volatility (e.g. 0.50 = 50%).
            capital: Total capital in USD.

        Returns:
            Dollar amount to allocate to this asset.
        """
        if asset_vol <= 0:
            return 0.0
        return round(target_vol / asset_vol * capital, 2)

    # -----------------------------------------------------------------
    # 12. Edge Decay — how fast edge erodes
    # -----------------------------------------------------------------

    @staticmethod
    def edge_decay(initial_edge: float, decay_rate: float,
                   time_periods: int) -> float:
        """Edge_t = Edge_0 × e^(-λt).

        Args:
            initial_edge: Starting edge (e.g. expectancy = 0.65).
            decay_rate: λ — decay constant (e.g. 0.01 per period).
            time_periods: t — number of periods elapsed.

        Returns:
            Remaining edge after t periods.
        """
        return round(initial_edge * math.exp(-decay_rate * time_periods), 6)

    # -----------------------------------------------------------------
    # Full report — compute everything from closed trades
    # -----------------------------------------------------------------

    def full_report(self, closed_trades: List[dict],
                    equity_start: float = 10000.0,
                    market_returns: Optional[List[float]] = None,
                    risk_per_trade: float = 0.02) -> dict:
        """Compute all quant metrics from a list of closed trades.

        Args:
            closed_trades: List of trade dicts with at least:
                - "pnl": float (USD profit/loss)
                - "pnl_pct": float (% return)
                - "entry_time": int (microseconds timestamp)
            equity_start: Starting equity for drawdown curve.
            market_returns: Optional benchmark returns for alpha/IR.
            risk_per_trade: Fraction risked per trade for Risk of Ruin.

        Returns:
            Dict with all 12 metrics + interpretation.
        """
        if not closed_trades:
            return {"trades": 0, "message": "no closed trades"}

        # Extract returns
        returns = [t.get("pnl_pct", 0.0) for t in closed_trades]
        wins = [t for t in closed_trades if t.get("pnl", 0) > 0]
        losses = [t for t in closed_trades if t.get("pnl", 0) <= 0]
        n_wins = len(wins)
        n_losses = len(losses)
        total = n_wins + n_losses

        avg_win = (sum(t["pnl_pct"] for t in wins) / n_wins) if n_wins else 0.0
        avg_loss = (abs(sum(t["pnl_pct"] for t in losses) / n_losses)
                    if n_losses else 0.0)
        win_rate = n_wins / total if total > 0 else 0.0
        payoff = avg_win / avg_loss if avg_loss > 0 else 0.0

        # Build equity curve
        equity = [equity_start]
        for t in closed_trades:
            pnl = t.get("pnl", 0)
            equity.append(equity[-1] + pnl)

        # Compute all metrics
        sharpe = self.sharpe_ratio(returns)
        sortino = self.sortino_ratio(returns)
        expect = self.expectancy(n_wins, n_losses, avg_win, avg_loss)
        ror = self.risk_of_ruin(win_rate, payoff, risk_per_trade)
        opt_f = self.optimal_f(win_rate, payoff)
        mdd = self.max_drawdown(equity)

        report = {
            "trades": total,
            "win_rate": round(win_rate * 100, 1),
            "avg_win_pct": round(avg_win, 2),
            "avg_loss_pct": round(avg_loss, 2),
            "payoff_ratio": round(payoff, 2),

            # Core quant metrics
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "expectancy": expect,
            "risk_of_ruin": ror,
            "optimal_f": opt_f,
            "optimal_f_half": round(opt_f / 2, 4),  # Half-Kelly (safer)
            "max_drawdown_pct": mdd["mdd_pct"],
            "max_drawdown_peak": mdd["peak"],
            "max_drawdown_trough": mdd["trough"],
            "equity_final": round(equity[-1], 2),

            # Interpretation
            "sharpe_grade": ("elite" if sharpe >= 3 else "strong" if sharpe >= 2
                            else "good" if sharpe >= 1 else "weak" if sharpe >= 0
                            else "negative"),
            "system_edge": "profitable" if expect > 0 else "losing",
            "ruin_risk": ("safe" if ror < 0.01 else "moderate" if ror < 0.10
                         else "dangerous"),
            "drawdown_risk": ("safe" if mdd["mdd_pct"] < 20
                             else "warning" if mdd["mdd_pct"] < 40
                             else "dangerous"),
        }

        # CAPM Alpha + Information Ratio (if benchmark provided)
        if market_returns and len(market_returns) >= 5:
            capm = self.capm_alpha(returns, market_returns)
            report["alpha"] = capm["alpha"]
            report["beta"] = capm["beta"]
            report["information_ratio"] = self.information_ratio(
                returns, market_returns)
            report["has_alpha"] = capm["alpha"] > 0

        return report

    def quick_edge_check(self, win_rate: float, avg_win: float,
                         avg_loss: float) -> dict:
        """Fast edge validation — call before entering a trade.

        Args:
            win_rate: Historical win rate (0-1).
            avg_win: Average win % (positive).
            avg_loss: Average loss % (positive, absolute).

        Returns:
            {"has_edge": bool, "expectancy": float, "risk_of_ruin": float,
             "optimal_risk_pct": float, "verdict": str}
        """
        payoff = avg_win / avg_loss if avg_loss > 0 else 0.0
        expect = self.expectancy(
            int(win_rate * 100), int((1 - win_rate) * 100),
            avg_win, avg_loss)
        ror = self.risk_of_ruin(win_rate, payoff)
        opt_f = self.optimal_f(win_rate, payoff)

        has_edge = expect > 0 and ror < 0.10
        if expect <= 0:
            verdict = "NO EDGE — negative expectancy"
        elif ror > 0.10:
            verdict = f"RISKY — ruin probability {ror:.1%}"
        elif expect > 0.5:
            verdict = f"STRONG EDGE — E={expect:.2f}R per trade"
        else:
            verdict = f"MODERATE EDGE — E={expect:.2f}R per trade"

        return {
            "has_edge": has_edge,
            "expectancy": expect,
            "risk_of_ruin": ror,
            "optimal_risk_pct": round(opt_f * 100, 1),
            "half_kelly_pct": round(opt_f * 50, 1),  # Safer
            "payoff_ratio": round(payoff, 2),
            "verdict": verdict,
        }


# -- Helper --

def _std(values: List[float]) -> float:
    """Population standard deviation (no numpy dependency)."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance)

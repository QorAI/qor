"""
QOR Asset Context — Parallel Asset Data Gathering
====================================================
When a user asks about an asset (crypto, commodity, stock, forex),
detect the asset and run ALL related tools in parallel so that
follow-up questions (indicators, sentiment, news) are instant from cache.

Usage:
    from qor.asset_context import detect_asset, gather_asset_context

    asset = detect_asset("What's the price of Bitcoin?")
    # AssetInfo(name='bitcoin', asset_type='crypto', query='bitcoin')

    results = gather_asset_context(asset, executor, cache=cache_store)
    # [(tool_name, result_text), ...]
"""

import re
import atexit
from dataclasses import dataclass
from typing import Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed


# ── Shared worker pool (reused across calls, cheaper on small CPU/GPU) ──
# Workers are I/O-bound (HTTP calls), so 6 threads is fine even on 1 CPU core.
_shared_pool = ThreadPoolExecutor(max_workers=12)
atexit.register(_shared_pool.shutdown, wait=False)


@dataclass
class AssetInfo:
    """Detected asset from a user question."""
    name: str        # "bitcoin", "gold", "AAPL"
    asset_type: str  # "crypto", "commodity", "stock", "forex"
    query: str       # cleaned query string for tool calls


# ── Crypto name mapping (ticker/name → canonical) ──

CRYPTO_NAMES = {
    "bitcoin": "bitcoin", "btc": "bitcoin",
    "ethereum": "ethereum", "eth": "ethereum",
    "solana": "solana", "sol": "solana",
    "dogecoin": "dogecoin", "doge": "dogecoin",
    "xrp": "xrp", "ripple": "xrp",
    "cardano": "cardano", "ada": "cardano",
    "bnb": "bnb", "binance coin": "bnb",
    "polkadot": "polkadot", "dot": "polkadot",
    "avalanche": "avalanche", "avax": "avalanche",
    "litecoin": "litecoin", "ltc": "litecoin",
    "chainlink": "chainlink", "link": "chainlink",
    "polygon": "polygon", "matic": "polygon",
    "uniswap": "uniswap", "uni": "uniswap",
    "shiba inu": "shiba-inu", "shib": "shiba-inu",
    "tron": "tron", "trx": "tron",
    "near": "near", "near protocol": "near",
    "aptos": "aptos", "apt": "aptos",
    "sui": "sui",
    "arbitrum": "arbitrum", "arb": "arbitrum",
    "optimism": "optimism", "op": "optimism",
    "cosmos": "cosmos", "atom": "cosmos",
    "stellar": "stellar", "xlm": "stellar",
    "filecoin": "filecoin", "fil": "filecoin",
    "hedera": "hedera", "hbar": "hedera",
    "pepe": "pepe",
    "kaspa": "kaspa", "kas": "kaspa",
}

COMMODITY_NAMES = {
    "gold": "gold", "silver": "silver", "platinum": "platinum",
    "palladium": "palladium", "xau": "gold", "xag": "silver",
    "crude oil": "crude oil", "oil": "crude oil",
    "natural gas": "natural gas", "copper": "copper",
}

FOREX_PAIRS = {
    "usd/eur": "USD/EUR", "eur/usd": "EUR/USD",
    "gbp/usd": "GBP/USD", "usd/jpy": "USD/JPY",
    "usd/cad": "USD/CAD", "aud/usd": "AUD/USD",
}

STOCK_PATTERNS = ["stock", "share", "ticker", "nyse", "nasdaq"]
_TICKER_RE = re.compile(r'\$([A-Z]{1,5})\b')

# ── General market overview triggers ──
# These match when the user asks about "the market" without a specific asset.
MARKET_OVERVIEW_KEYWORDS = [
    "market overview", "market condition", "market update",
    "how is the market", "how's the market", "how are the market",
    "market sentiment", "market today", "market right now",
    "market looking", "market doing", "what's the market",
    "whats the market", "market summary", "financial market",
    "market report", "market status", "market analysis",
    "all markets", "global markets",
]


# ── Tool groups per asset type ──

ASSET_TOOL_GROUPS = {
    "crypto": [
        ("crypto_price",        "{name}"),
        ("binance_price",       "{name}"),
        ("multi_tf_analysis",   "{name}"),
        ("fear_greed",          "crypto"),
        ("funding_rate",        "{name}"),
        ("open_interest",       "{name}"),
        ("global_market",       "crypto"),
        ("news_search",         "{name} crypto news"),
        ("polymarket",          "{name} crypto prediction"),
        ("economic_calendar",   "upcoming"),
    ],
    "commodity": [
        ("commodities",         "{name}"),
        ("multi_tf_analysis",   "{name}"),
        ("forex_rates",         "USD"),
        ("news_search",         "{name} price news"),
        ("polymarket",          "{name}"),
        ("economic_calendar",   "upcoming"),
        ("world_economy",       "top economies"),
    ],
    "stock": [
        ("stock_quote",         "{name}"),
        ("multi_tf_analysis",   "{name}"),
        ("market_indices",      "indices"),
        ("news_search",         "{name} stock news"),
        ("polymarket",          "{name}"),
        ("economic_calendar",   "upcoming"),
    ],
    "forex": [
        ("forex_rates",         "{name}"),
        ("multi_tf_analysis",   "{name}"),
        ("polymarket",          "{name}"),
        ("economic_calendar",   "upcoming"),
        ("news_search",         "{name} forex news"),
    ],
    "market_overview": [
        ("crypto_price",        "bitcoin"),
        ("crypto_price",        "ethereum"),
        ("fear_greed",          "crypto"),
        ("global_market",       "crypto"),
        ("market_indices",      "indices"),
        ("commodities",         "gold"),
        ("forex_rates",         "EUR/USD"),
        ("multi_tf_analysis",   "bitcoin"),
        ("polymarket",          "crypto prediction"),
        ("economic_calendar",   "upcoming"),
        ("news_search",         "market news today"),
    ],
}


def detect_asset(question: str) -> Optional[AssetInfo]:
    """
    Detect an asset (crypto, commodity, stock, forex) from a question.
    Returns AssetInfo or None if no asset detected.
    """
    q = question.lower().strip()

    # 0. General market overview — no specific asset, wants broad market data
    for kw in MARKET_OVERVIEW_KEYWORDS:
        if kw in q:
            return AssetInfo(name="market", asset_type="market_overview", query="market")

    # 1. Crypto — most common, check first
    for token, canonical in CRYPTO_NAMES.items():
        if token in q:
            return AssetInfo(name=canonical, asset_type="crypto", query=canonical)

    # 2. Commodities
    for token, canonical in COMMODITY_NAMES.items():
        if token in q:
            return AssetInfo(name=canonical, asset_type="commodity", query=canonical)

    # 3. Forex pairs
    for token, canonical in FOREX_PAIRS.items():
        if token in q:
            return AssetInfo(name=canonical, asset_type="forex", query=canonical)

    # 4. Stock — $TICKER pattern
    ticker_match = _TICKER_RE.search(question)  # search original (uppercase)
    if ticker_match:
        ticker = ticker_match.group(1)
        return AssetInfo(name=ticker, asset_type="stock", query=ticker)

    # 5. Stock — keyword pattern ("AAPL stock", "Tesla shares")
    for kw in STOCK_PATTERNS:
        if kw in q:
            # Extract the word before the keyword as potential ticker/name
            idx = q.index(kw)
            before = q[:idx].strip().split()
            if before:
                name = before[-1].upper()
                return AssetInfo(name=name, asset_type="stock", query=name)

    return None


def gather_asset_context(
    asset: AssetInfo,
    executor,
    cache=None,
    memory=None,
    verbose: bool = True,
) -> List[Tuple[str, str]]:
    """
    Run all tools for an asset type in parallel.

    Args:
        asset: Detected AssetInfo
        executor: ToolExecutor instance (must have .call(name, query))
        cache: CacheStore instance (preferred) or None
        memory: MemoryStore instance (fallback) or None
        verbose: Print progress

    Returns:
        List of (tool_name, result_text) for tools that returned data.
    """
    tools = ASSET_TOOL_GROUPS.get(asset.asset_type, [])
    if not tools:
        return []

    results = []

    def _call_one(tool_name: str, query: str) -> Tuple[str, str, Optional[str]]:
        """Call a single tool. Returns (name, query, result) or (name, query, None)."""
        # Cache key includes query to distinguish e.g. crypto_price:bitcoin vs crypto_price:ethereum
        ck = f"tool:{tool_name}:{query}"
        if cache is not None:
            cached = cache.get_fresh(ck)
            if cached:
                return (tool_name, query, cached.content)

        try:
            result = executor.call(tool_name, query)
            if result and not result.startswith(("[Tool", "Tool error", "Rate limit")):
                return (tool_name, query, result)
        except Exception:
            pass
        return (tool_name, query, None)

    # Run all tools in parallel via shared pool (reuses threads across calls)
    futures = []
    for tool_name, query_template in tools:
        query = query_template.format(name=asset.name)
        futures.append(_shared_pool.submit(_call_one, tool_name, query))

    for future in as_completed(futures, timeout=20):
        try:
            name, query_used, result = future.result(timeout=15)
            if result:
                results.append((name, query_used, result))
        except Exception:
            pass

    # Save all results to cache/memory for instant follow-ups
    for tool_name, query_used, result in results:
        save_key = f"tool:{tool_name}:{query_used}"
        if cache is not None:
            cache.store(save_key, result, f"tool:{tool_name}", tool_name,
                        confidence=0.85)
        elif memory is not None:
            memory.store(save_key, result, f"tool:{tool_name}",
                         category="live", confidence=0.85)

    if verbose:
        print(f"  |- Asset context: {len(results)}/{len(tools)} tools "
              f"returned data for {asset.name}")

    # Return (tool_name, result) pairs — callers don't need the query
    return [(name, result) for name, _query, result in results]

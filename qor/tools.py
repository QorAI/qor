"""
QOR Tools — 30+ Free API Tools (No Keys Required)
=====================================================
Ported from Go tools.go into native Python.

All tools use FREE public APIs — no API keys needed.

Categories:
  - Crypto & Finance: CoinGecko, Binance, Frankfurter
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

import re
import json
import time
import math
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable


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
    """Get real-time price from Binance."""
    symbol = _extract_trading_symbol(query).replace("-", "")
    if not symbol.endswith("USDT"):
        symbol += "USDT"
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
    data = _http_get(url)
    price = float(data.get("price", 0))
    return f"{data.get('symbol', symbol)}: ${price:,.2f} (Binance)"


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


def calculate(query: str) -> str:
    """Simple calculator."""
    expr = re.findall(r'[\d\.\+\-\*/\(\)\s\%]+', query)
    if expr:
        try:
            # Safe eval — only math operations
            cleaned = expr[0].strip()
            allowed = set("0123456789.+-*/() %")
            if all(c in allowed for c in cleaned):
                result = eval(cleaned)
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
}


class ToolExecutor:
    """Manages and executes all tools."""

    def __init__(self):
        self.tools = dict(ALL_TOOLS)

    def register_all(self, gate):
        """Register all tools with a ConfidenceGate."""
        for name, (func, desc, cats) in self.tools.items():
            gate.tools.register(name, desc, func, cats)
        print(f"[Tools] Registered {len(self.tools)} tools")

    def call(self, name: str, query: str) -> str:
        """Call a tool by name."""
        if name not in self.tools:
            return f"Tool not found: {name}"
        func = self.tools[name][0]
        try:
            return func(query)
        except Exception as e:
            return f"Tool error ({name}): {e}"

    def detect_intent(self, query: str) -> Optional[str]:
        """Detect which tool should handle a query."""
        query_lower = query.lower()

        # Keyword matching (same logic as Go matchBuiltinTools)
        keyword_map = {
            "crypto_price":    ["crypto", "bitcoin", "btc", "eth", "ethereum",
                                "price of", "how much is"],
            "binance_price":   ["binance"],
            "crypto_market":   ["crypto market", "market cap", "top crypto"],
            "convert_currency":["convert", "currency", "exchange rate"],
            "wikipedia":       ["wikipedia", "wiki"],
            "web_search":      ["search for", "look up", "find info"],
            "definition":      ["define", "definition", "meaning of"],
            "weather":         ["weather", "temperature", "forecast"],
            "hacker_news":     ["hacker news", "hn", "tech news"],
            "news":            ["news", "headlines", "current events"],
            "github":          ["github", "repository"],
            "pypi":            ["pypi", "pip install", "python package"],
            "npm":             ["npm", "node package"],
            "arxiv":           ["arxiv", "research paper"],
            "huggingface":     ["huggingface", "ai model"],
            "run_code":        ["run code", "execute", "```"],
            "joke":            ["joke", "funny", "humor"],
            "trivia":          ["trivia", "quiz"],
            "recipe":          ["recipe", "cook", "how to make"],
            "reddit":          ["reddit", "r/"],
            "books":           ["book", "library search"],
            "nasa":            ["nasa", "astronomy picture"],
            "calculate":       ["calculate", "math", "compute", "what is "],
            "time":            ["what time", "current time", "date today"],
            "country":         ["country info", "population of", "capital of"],
            "ip_lookup":       ["ip address", "ip lookup"],
            "geocode":         ["coordinates", "geocode"],
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

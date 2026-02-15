"""
QOR Browser Tool — Playwright-based headless browsing
======================================================
Two tools matching existing sync handler pattern:
  browse_web(query) -> str       — Navigate + extract text
  browse_screenshot(query) -> str — Navigate + capture PNG

Uses singleton BrowserEngine with lazy Chromium launch
and 5-minute idle auto-close to conserve resources.

Requires: pip install playwright && playwright install chromium
Graceful fallback if playwright not installed.
"""

import os
import re
import time
import threading
from typing import Optional


# ==============================================================================
# URL EXTRACTION
# ==============================================================================

_SITE_SHORTCUTS = {
    "coinmarketcap": "https://coinmarketcap.com",
    "etherscan": "https://etherscan.io",
    "uniswap": "https://app.uniswap.org",
    "defillama": "https://defillama.com",
    "coingecko": "https://www.coingecko.com",
    "polymarket": "https://polymarket.com",
    "dexscreener": "https://dexscreener.com",
    "snapshot": "https://snapshot.org",
    "opensea": "https://opensea.io",
    "aave": "https://app.aave.com",
    "compound": "https://app.compound.finance",
    "makerdao": "https://makerdao.com",
    "lido": "https://lido.fi",
    "curve": "https://curve.fi",
    "github": "https://github.com",
    "arxiv": "https://arxiv.org",
}

_URL_PATTERN = re.compile(r'https?://\S+')


def _extract_url(query: str) -> str:
    """Extract URL from query. Priority: explicit URL > site shortcut > search."""
    # 1. Explicit URL in query
    m = _URL_PATTERN.search(query)
    if m:
        return m.group(0).rstrip('.,;:!?)')

    # 2. Site shortcut
    q_lower = query.lower()
    for shortcut, url in _SITE_SHORTCUTS.items():
        if shortcut in q_lower:
            return url

    # 3. DuckDuckGo search fallback
    # Strip common prefixes to get the search query
    search_q = q_lower
    for prefix in ("browse ", "visit ", "go to ", "open ", "navigate to ",
                   "show me ", "check ", "look at "):
        if search_q.startswith(prefix):
            search_q = search_q[len(prefix):]
            break

    from urllib.parse import quote_plus
    return f"https://duckduckgo.com/?q={quote_plus(search_q.strip())}"


# ==============================================================================
# BROWSER ENGINE (Singleton, lazy-start, idle auto-close)
# ==============================================================================

class BrowserEngine:
    """Singleton headless Chromium browser with idle auto-close."""

    def __init__(self, idle_timeout_seconds: int = 300,
                 screenshots_dir: str = "qor-data/screenshots"):
        self._lock = threading.Lock()
        self._playwright = None
        self._browser = None
        self._idle_timeout = idle_timeout_seconds
        self._screenshots_dir = screenshots_dir
        self._last_used = 0.0
        self._idle_timer: Optional[threading.Timer] = None

    def _ensure_browser(self):
        """Lazy-start Chromium. Raises ImportError if playwright missing."""
        if self._browser is not None and self._browser.is_connected():
            self._touch()
            return

        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            raise ImportError(
                "Browser tool requires playwright. "
                "Install with: pip install playwright && playwright install chromium"
            )

        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        self._touch()

    def _touch(self):
        """Reset idle timer."""
        self._last_used = time.time()
        if self._idle_timer is not None:
            self._idle_timer.cancel()
        self._idle_timer = threading.Timer(self._idle_timeout, self._idle_close)
        self._idle_timer.daemon = True
        self._idle_timer.start()

    def _idle_close(self):
        """Auto-close browser after idle timeout."""
        with self._lock:
            if self._browser is not None:
                elapsed = time.time() - self._last_used
                if elapsed >= self._idle_timeout - 1:
                    self._shutdown()

    def _shutdown(self):
        """Internal shutdown (must hold lock)."""
        if self._browser is not None:
            try:
                self._browser.close()
            except Exception:
                pass
            self._browser = None
        if self._playwright is not None:
            try:
                self._playwright.stop()
            except Exception:
                pass
            self._playwright = None
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None

    def navigate_and_extract(self, url: str, max_chars: int = 4000) -> str:
        """Navigate to URL, strip noise, return clean text."""
        with self._lock:
            self._ensure_browser()
            context = self._browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/120.0.0.0 Safari/537.36"
            )
            try:
                page = context.new_page()
                page.set_default_timeout(30000)
                page.goto(url, wait_until="domcontentloaded")
                # Wait a bit for JS rendering
                page.wait_for_timeout(1500)

                # Remove noisy elements
                page.evaluate("""
                    for (const sel of ['script', 'style', 'nav', 'footer',
                                       'header', 'iframe', 'noscript',
                                       '.cookie-banner', '.popup', '.modal',
                                       '[role="banner"]', '[role="navigation"]']) {
                        document.querySelectorAll(sel).forEach(el => el.remove());
                    }
                """)

                # Extract text from body
                text = page.inner_text("body")
                # Collapse whitespace
                text = re.sub(r'\n{3,}', '\n\n', text)
                text = re.sub(r'[ \t]{2,}', ' ', text)
                text = text.strip()

                if len(text) > max_chars:
                    text = text[:max_chars] + "\n...(truncated)"

                title = page.title() or ""
                final_url = page.url

                return f"Page: {title}\nURL: {final_url}\n\n{text}"
            finally:
                context.close()

    def take_screenshot(self, url: str) -> str:
        """Navigate to URL, save PNG screenshot, return file path."""
        with self._lock:
            self._ensure_browser()
            os.makedirs(self._screenshots_dir, exist_ok=True)

            context = self._browser.new_context(
                viewport={"width": 1280, "height": 720},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/120.0.0.0 Safari/537.36"
            )
            try:
                page = context.new_page()
                page.set_default_timeout(30000)
                page.goto(url, wait_until="domcontentloaded")
                page.wait_for_timeout(2000)

                # Generate filename from URL
                safe_name = re.sub(r'[^\w\-.]', '_', url.split("//")[-1][:60])
                ts = int(time.time())
                filename = f"{safe_name}_{ts}.png"
                filepath = os.path.join(self._screenshots_dir, filename)

                page.screenshot(path=filepath, full_page=False)
                return f"Screenshot saved: {filepath}"
            finally:
                context.close()

    def smart_extract(self, url: str, max_chars: int = 6000,
                       wait_for: str = None, scroll: bool = True) -> str:
        """Navigate to URL with smarter extraction for JS-heavy sites.

        - Waits longer for dynamic content (tables, lists)
        - Scrolls to trigger lazy-loading
        - Extracts tables as structured text
        - Falls back to inner_text if no tables found
        """
        with self._lock:
            self._ensure_browser()
            context = self._browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/120.0.0.0 Safari/537.36"
            )
            try:
                page = context.new_page()
                page.set_default_timeout(30000)
                page.goto(url, wait_until="domcontentloaded")

                # Wait for specific element if requested
                if wait_for:
                    try:
                        page.wait_for_selector(wait_for, timeout=8000)
                    except Exception:
                        pass

                # Wait for JS rendering (longer for dynamic sites)
                page.wait_for_timeout(3000)

                # Scroll down to trigger lazy-loading
                if scroll:
                    page.evaluate("""
                        window.scrollBy(0, 800);
                    """)
                    page.wait_for_timeout(1500)

                # Remove noisy elements
                page.evaluate("""
                    for (const sel of ['script', 'style', 'nav', 'footer',
                                       'header', 'iframe', 'noscript',
                                       '.cookie-banner', '.popup', '.modal',
                                       '.ad', '.advertisement', '.sidebar',
                                       '[role="banner"]', '[role="navigation"]',
                                       '[class*="cookie"]', '[class*="consent"]',
                                       '[class*="popup"]', '[class*="modal"]']) {
                        document.querySelectorAll(sel).forEach(el => el.remove());
                    }
                """)

                # Try to extract tables as structured data
                tables_text = page.evaluate("""
                    () => {
                        const tables = document.querySelectorAll('table');
                        if (tables.length === 0) return '';
                        let result = '';
                        tables.forEach((table, ti) => {
                            if (ti > 2) return;  // max 3 tables
                            const rows = table.querySelectorAll('tr');
                            rows.forEach((row, ri) => {
                                if (ri > 25) return;  // max 25 rows
                                const cells = row.querySelectorAll('td, th');
                                const values = Array.from(cells).map(c =>
                                    c.innerText.trim().replace(/\\n/g, ' ')
                                ).filter(v => v.length > 0);
                                if (values.length > 0) {
                                    result += values.join(' | ') + '\\n';
                                }
                            });
                            result += '\\n';
                        });
                        return result;
                    }
                """)

                # Also try list-based layouts (used by CoinMarketCap, etc.)
                lists_text = ""
                if not tables_text.strip():
                    lists_text = page.evaluate("""
                        () => {
                            // Try common data row patterns
                            const selectors = [
                                'tr', '[class*="row"]', '[class*="item"]',
                                '[class*="coin"]', '[class*="token"]',
                                '[class*="asset"]', 'li'
                            ];
                            let best = '';
                            for (const sel of selectors) {
                                const items = document.querySelectorAll(sel);
                                if (items.length >= 5 && items.length <= 200) {
                                    let text = '';
                                    let count = 0;
                                    items.forEach(item => {
                                        if (count >= 30) return;
                                        const t = item.innerText.trim()
                                            .replace(/\\n+/g, ' | ')
                                            .replace(/\\s{2,}/g, ' ');
                                        if (t.length > 10 && t.length < 500) {
                                            text += t + '\\n';
                                            count++;
                                        }
                                    });
                                    if (text.length > best.length) {
                                        best = text;
                                    }
                                }
                            }
                            return best;
                        }
                    """)

                # Fall back to full body text
                body_text = page.inner_text("body")
                body_text = re.sub(r'\n{3,}', '\n\n', body_text)
                body_text = re.sub(r'[ \t]{2,}', ' ', body_text)
                body_text = body_text.strip()

                # Choose best content: tables > lists > body
                content = ""
                if tables_text.strip():
                    content = f"[TABLE DATA]\n{tables_text.strip()}"
                    if len(content) < max_chars // 2:
                        content += f"\n\n[PAGE TEXT]\n{body_text}"
                elif lists_text.strip():
                    content = f"[STRUCTURED DATA]\n{lists_text.strip()}"
                    if len(content) < max_chars // 2:
                        content += f"\n\n[PAGE TEXT]\n{body_text}"
                else:
                    content = body_text

                if len(content) > max_chars:
                    content = content[:max_chars] + "\n...(truncated)"

                title = page.title() or ""
                final_url = page.url

                return f"Page: {title}\nURL: {final_url}\n\n{content}"
            finally:
                context.close()

    def click_and_extract(self, url: str, selector: str,
                          max_chars: int = 4000) -> str:
        """Navigate, click an element, then extract page text."""
        with self._lock:
            self._ensure_browser()
            context = self._browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/120.0.0.0 Safari/537.36"
            )
            try:
                page = context.new_page()
                page.set_default_timeout(30000)
                page.goto(url, wait_until="domcontentloaded")
                page.wait_for_timeout(2000)
                try:
                    page.click(selector, timeout=5000)
                    page.wait_for_timeout(1500)
                except Exception as e:
                    pass  # Click failed — still extract what we can
                text = page.inner_text("body")
                text = re.sub(r'\n{3,}', '\n\n', text)
                text = re.sub(r'[ \t]{2,}', ' ', text)
                text = text.strip()
                if len(text) > max_chars:
                    text = text[:max_chars] + "\n...(truncated)"
                return f"Page: {page.title()}\nURL: {page.url}\n\n{text}"
            finally:
                context.close()

    def search_and_extract(self, url: str, search_selector: str,
                           search_text: str, max_chars: int = 4000) -> str:
        """Navigate, fill a search box, submit, and extract results."""
        with self._lock:
            self._ensure_browser()
            context = self._browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/120.0.0.0 Safari/537.36"
            )
            try:
                page = context.new_page()
                page.set_default_timeout(30000)
                page.goto(url, wait_until="domcontentloaded")
                page.wait_for_timeout(2000)
                try:
                    page.fill(search_selector, search_text)
                    page.press(search_selector, "Enter")
                    page.wait_for_timeout(3000)
                except Exception:
                    pass
                text = page.inner_text("body")
                text = re.sub(r'\n{3,}', '\n\n', text)
                text = re.sub(r'[ \t]{2,}', ' ', text)
                text = text.strip()
                if len(text) > max_chars:
                    text = text[:max_chars] + "\n...(truncated)"
                return f"Page: {page.title()}\nURL: {page.url}\n\n{text}"
            finally:
                context.close()

    def close(self):
        """Shutdown browser (called by cmd_run cleanup)."""
        with self._lock:
            self._shutdown()

    def status(self) -> dict:
        """Return browser status for status command."""
        running = self._browser is not None and self._browser.is_connected()
        idle_seconds = time.time() - self._last_used if self._last_used > 0 else 0
        try:
            from playwright.sync_api import sync_playwright  # noqa: F401
            available = True
        except ImportError:
            available = False
        return {
            "running": running,
            "available": available,
            "idle_seconds": round(idle_seconds, 1) if running else 0,
        }


# ==============================================================================
# SINGLETON ACCESS
# ==============================================================================

_engine: Optional[BrowserEngine] = None
_engine_lock = threading.Lock()


def get_engine(screenshots_dir: str = "qor-data/screenshots") -> BrowserEngine:
    """Get or create the singleton BrowserEngine."""
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = BrowserEngine(screenshots_dir=screenshots_dir)
        return _engine


# Model + skill loader references (set once by cmd_run via set_browse_model)
_browse_model = None
_browse_tokenizer = None
_browse_skill_loader = None


def set_browse_model(model, tokenizer, skill_loader=None):
    """Wire model + tokenizer + skills into the browse agent.

    Called once by cmd_run() after loading the model. Without this,
    BrowseAgent falls back to heuristic rules.
    """
    global _browse_model, _browse_tokenizer, _browse_skill_loader
    _browse_model = model
    _browse_tokenizer = tokenizer
    _browse_skill_loader = skill_loader


# ==============================================================================
# TOOL HANDLERS (match existing sync pattern: def handler(query: str) -> str)
# ==============================================================================

def browse_web(query: str) -> str:
    """Navigate to URL or search, extract page text.

    Uses smart_extract: waits for JS rendering, scrolls for lazy-loading,
    extracts tables and structured data (not just body text).
    """
    try:
        url = _extract_url(query)
        return get_engine().smart_extract(url)
    except ImportError as e:
        return str(e)
    except Exception as e:
        return f"Browse error: {e}"


def browse_screenshot(query: str) -> str:
    """Navigate to URL, save screenshot, return path."""
    try:
        url = _extract_url(query)
        return get_engine().take_screenshot(url)
    except ImportError as e:
        return str(e)
    except Exception as e:
        return f"Screenshot error: {e}"


# ==============================================================================
# BROWSER AGENT — Multi-step browsing with observe → decide → act loop
# ==============================================================================
# Inspired by nanobot's agent loop pattern but adapted for QOR's frozen model.
# The model sees page state as text and decides next actions via structured prompts.

# Action types the agent loop understands
_BROWSER_ACTIONS = {
    "DONE":     "Task complete, return collected data",
    "SCROLL":   "Scroll down to load more content",
    "CLICK":    "Click on a link or button (provide text/selector)",
    "SEARCH":   "Type into search box and submit",
    "NAVIGATE": "Go to a different URL",
    "EXTRACT":  "Extract specific data from current page",
}


class BrowseAgent:
    """Multi-step browser agent that can navigate, interact, and extract data.

    Loop: navigate → observe page → ask model what to do → execute action → repeat.
    Max 5 steps to prevent runaway browsing. Each step produces an observation
    (page text) that feeds into the next decision.
    """

    def __init__(self, model=None, tokenizer=None, max_steps: int = 5):
        self.model = model
        self.tokenizer = tokenizer
        self.max_steps = max_steps
        self.engine = get_engine()

    def run(self, task: str, verbose: bool = False) -> str:
        """Execute a multi-step browse task.

        Args:
            task: User's browse request (e.g. "go to coinmarketcap and get top 10 coins")
            verbose: Print step-by-step actions

        Returns:
            Collected data from all browsing steps as a string.
        """
        try:
            from playwright.sync_api import sync_playwright  # noqa: F401
        except ImportError:
            return ("Browser agent requires playwright. "
                    "Install with: pip install playwright && playwright install chromium")

        url = _extract_url(task)
        collected_data = []
        visited_urls = []
        current_page_text = ""

        for step in range(self.max_steps):
            if verbose:
                print(f"  ├─ Browse step {step + 1}/{self.max_steps}: {url[:80]}")

            # ── OBSERVE: Navigate and extract page state ──
            try:
                current_page_text = self.engine.smart_extract(url, max_chars=4000)
                visited_urls.append(url)
            except Exception as e:
                collected_data.append(f"[Error loading {url}: {e}]")
                break

            # First step: always collect the initial page data
            if step == 0:
                collected_data.append(current_page_text)

            # ── DECIDE: What should we do next? ──
            action, action_arg = self._decide_next_action(
                task, current_page_text, collected_data, step)

            if verbose:
                print(f"  │  Action: {action}"
                      f"{f' ({action_arg[:60]})' if action_arg else ''}")

            # ── ACT: Execute the decided action ──
            if action == "DONE":
                # Task complete — we have enough data
                break

            elif action == "SCROLL":
                # Scroll and re-extract (get content below the fold)
                try:
                    scroll_text = self._scroll_and_extract(url)
                    if scroll_text and scroll_text not in current_page_text:
                        collected_data.append(f"[After scrolling]\n{scroll_text}")
                except Exception:
                    pass
                # Stay on same URL, next iteration will re-extract
                continue

            elif action == "CLICK" and action_arg:
                # Click a link/button and navigate to result
                new_url = self._find_link(current_page_text, action_arg, url)
                if new_url and new_url not in visited_urls:
                    url = new_url
                    # Next iteration will navigate to this new URL
                else:
                    # Couldn't find the link — try extracting more from current page
                    collected_data.append(current_page_text)
                    break

            elif action == "SEARCH" and action_arg:
                # Search on the current page
                try:
                    search_result = self.engine.search_and_extract(
                        url, 'input[type="search"], input[type="text"], '
                             'input[name="q"], input[placeholder*="earch"]',
                        action_arg)
                    collected_data.append(f"[Search: {action_arg}]\n{search_result}")
                except Exception:
                    pass
                break  # Search results are usually the final answer

            elif action == "NAVIGATE" and action_arg:
                # Go to a completely new URL
                new_url = action_arg if action_arg.startswith("http") else \
                    _extract_url(action_arg)
                if new_url not in visited_urls:
                    url = new_url
                else:
                    break  # Already visited

            elif action == "EXTRACT":
                # Already have the data, just make sure it's collected
                collected_data.append(current_page_text)
                break

            else:
                # Unknown action or no argument — stop
                break

        # ── COMBINE: Merge all collected data ──
        if not collected_data:
            return f"Could not extract data from {url}"

        # Deduplicate and join
        seen = set()
        unique_parts = []
        for part in collected_data:
            # Use first 200 chars as dedup key
            key = part[:200]
            if key not in seen:
                seen.add(key)
                unique_parts.append(part)

        result = "\n\n".join(unique_parts)
        # Cap total output
        if len(result) > 8000:
            result = result[:8000] + "\n...(truncated)"

        return result

    def _decide_next_action(self, task: str, page_text: str,
                            collected_so_far: list, step: int
                            ) -> tuple:
        """Decide what to do next based on task + current page state.

        Uses the model if available, otherwise uses heuristic rules.
        Returns (action, argument) tuple.
        """
        # If model is available, ask it to decide
        if self.model is not None and self.tokenizer is not None:
            return self._model_decide(task, page_text, collected_so_far, step)

        # ── Heuristic fallback (no model) ──
        return self._heuristic_decide(task, page_text, collected_so_far, step)

    def _model_decide(self, task: str, page_text: str,
                      collected: list, step: int) -> tuple:
        """Ask the frozen model what to do next."""
        # Truncate page text to fit in context
        page_snippet = page_text[:2000]
        collected_summary = f"{len(collected)} data segments collected so far"

        system_msg = (
            "You are a browser agent deciding what to do next. "
            "Reply with EXACTLY one line — one of these actions:\n"
            "DONE — if the page already has the data we need\n"
            "SCROLL — if we need to scroll down for more data\n"
            "CLICK <link text> — to click a specific link\n"
            "SEARCH <query> — to search on this page\n"
            "NAVIGATE <url> — to go to a different page\n"
            "EXTRACT — to save current page data and stop"
        )
        user_msg = (
            f"Task: {task}\n"
            f"Step: {step + 1}/{self.max_steps}\n"
            f"Collected: {collected_summary}\n"
            f"Current page:\n{page_snippet}"
        )

        try:
            import torch
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ]
            # Use chat template for instruct model (SmolLM3 Llama-3 format)
            if hasattr(self.tokenizer, 'format_chat'):
                input_ids = self.tokenizer.format_chat(
                    messages, add_generation_prompt=True)
            else:
                input_ids = self.tokenizer.encode(
                    f"{system_msg}\n\n{user_msg}\nReply:")

            device = next(self.model.parameters()).device
            tokens = torch.tensor([input_ids], device=device)
            with torch.no_grad():
                output = self.model.generate(tokens, max_new_tokens=30,
                                             temperature=0.3)
            response = self.tokenizer.decode(
                output[0][len(input_ids):].tolist())
            response = response.strip().split("\n")[0].strip()

            return self._parse_action(response)
        except Exception:
            return self._heuristic_decide(task, page_text, collected, step)

    def _heuristic_decide(self, task: str, page_text: str,
                          collected: list, step: int) -> tuple:
        """Rule-based decision when model is unavailable."""
        task_lower = task.lower()
        page_lower = page_text.lower()

        # Step 0: we just loaded the page — check if we have enough data
        if step == 0:
            # Check if the page has substantial structured data
            has_numbers = len(re.findall(r'\$[\d,.]+', page_text)) >= 3
            has_table = '[TABLE DATA]' in page_text or '[STRUCTURED DATA]' in page_text
            has_enough_text = len(page_text) > 1000

            if has_table or (has_numbers and has_enough_text):
                return ("DONE", "")

            # Page might need scrolling (lazy-loaded content)
            return ("SCROLL", "")

        # Step 1+: we've scrolled or navigated — check again
        if step >= 1:
            # If we have collected data with numbers/tables, we're done
            all_text = " ".join(collected)
            if len(all_text) > 1500 and (
                '$' in all_text or '[TABLE' in all_text or
                '[STRUCTURED' in all_text
            ):
                return ("DONE", "")

            # Try to find a relevant link to click
            if "top 10" in task_lower or "top coins" in task_lower:
                if "coinmarketcap" in page_lower:
                    return ("DONE", "")  # CMC homepage IS the top coins

            # Nothing useful after scrolling — extract what we have and stop
            return ("EXTRACT", "")

        return ("DONE", "")

    def _parse_action(self, response: str) -> tuple:
        """Parse model response into (action, argument).

        Only uppercases the action keyword, preserves original case
        for arguments (URLs, search terms, link text).
        """
        raw = response.strip()
        raw_upper = raw.upper()

        for action in _BROWSER_ACTIONS:
            if raw_upper.startswith(action):
                # Extract argument from ORIGINAL case (not uppercased)
                arg = raw[len(action):].strip().strip("—-:").strip()
                return (action, arg)

        # Default: if response looks like it has data, we're done
        return ("DONE", "")

    def _scroll_and_extract(self, url: str, scroll_amount: int = 1500) -> str:
        """Scroll page down and extract new content."""
        with self.engine._lock:
            self.engine._ensure_browser()
            context = self.engine._browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/120.0.0.0 Safari/537.36"
            )
            try:
                page = context.new_page()
                page.set_default_timeout(30000)
                page.goto(url, wait_until="domcontentloaded")
                page.wait_for_timeout(2000)
                # Scroll further down
                page.evaluate(f"window.scrollBy(0, {scroll_amount})")
                page.wait_for_timeout(2000)
                page.evaluate(f"window.scrollBy(0, {scroll_amount})")
                page.wait_for_timeout(1500)
                text = page.inner_text("body")
                text = re.sub(r'\n{3,}', '\n\n', text)
                text = re.sub(r'[ \t]{2,}', ' ', text)
                return text.strip()[:3000]
            finally:
                context.close()

    def _find_link(self, page_text: str, link_text: str,
                   current_url: str) -> Optional[str]:
        """Find a URL in page text matching the link text.

        This is a heuristic — looks for URLs near the link text.
        """
        # Extract all URLs from page text
        urls = re.findall(r'https?://[^\s<>"\']+', page_text)
        link_lower = link_text.lower()

        # Look for URL containing the link text
        for url in urls:
            if link_lower.replace(" ", "") in url.lower().replace(" ", ""):
                return url

        # Try to construct URL from current site
        from urllib.parse import urlparse
        parsed = urlparse(current_url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        slug = link_text.lower().replace(" ", "-")
        return f"{base}/{slug}"


def browse_agent(query: str, model=None, tokenizer=None,
                 verbose: bool = False) -> str:
    """Multi-step browser agent tool handler.

    For simple pages: single navigate + extract (fast path).
    For complex tasks: observe → decide → act loop (up to 5 steps).

    Model/tokenizer auto-discovered from set_browse_model() if not passed.
    """
    # Auto-discover model from module-level reference (set by cmd_run)
    if model is None:
        model = _browse_model
    if tokenizer is None:
        tokenizer = _browse_tokenizer

    try:
        agent = BrowseAgent(model=model, tokenizer=tokenizer, max_steps=5)
        return agent.run(query, verbose=verbose)
    except ImportError as e:
        return str(e)
    except Exception as e:
        return f"Browse agent error: {e}"


# ==============================================================================
# BROWSER AUTOMATION — Visible browser, user's choice, real task execution
# ==============================================================================
# Separate from BrowserEngine (headless data extraction).
# This opens a REAL visible browser window using the user's installed browser
# with persistent login sessions, so the AI can perform tasks like:
#   "go to LinkedIn and remove 100 posts"
#   "go to Gmail and archive all promotions"
#   "open YouTube and subscribe to these channels"

# Supported browsers (Playwright channels)
SUPPORTED_BROWSERS = {
    "chrome":   {"channel": "chrome",  "label": "Google Chrome"},
    "edge":     {"channel": "msedge",  "label": "Microsoft Edge"},
    "firefox":  {"type": "firefox",    "label": "Mozilla Firefox"},
    "safari":   {"type": "webkit",     "label": "Safari (WebKit)"},
    "chromium": {"channel": None,      "label": "Chromium (built-in)"},
}


def _find_real_chrome_profile() -> Optional[str]:
    """Find the user's real Chrome User Data directory (platform-specific)."""
    import sys
    home = os.path.expanduser("~")
    candidates = []
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA", os.path.join(home, "AppData", "Local"))
        candidates = [
            os.path.join(local, "Google", "Chrome", "User Data"),
            os.path.join(local, "Microsoft", "Edge", "User Data"),
        ]
    elif sys.platform == "darwin":
        candidates = [
            os.path.join(home, "Library", "Application Support", "Google", "Chrome"),
            os.path.join(home, "Library", "Application Support", "Microsoft Edge"),
        ]
    else:  # Linux
        candidates = [
            os.path.join(home, ".config", "google-chrome"),
            os.path.join(home, ".config", "chromium"),
            os.path.join(home, ".config", "microsoft-edge"),
        ]
    for path in candidates:
        if os.path.isdir(path):
            # Verify it has a Default profile
            default = os.path.join(path, "Default")
            if os.path.isdir(default):
                return path
    return None


def _sync_cookies_from_real_profile(real_profile_dir: str,
                                     automation_profile_dir: str) -> int:
    """Copy login/cookie data from user's real Chrome to our automation profile.

    Chrome locks its profile when running, but the cookie/login files can
    still be copied. This gives the automation browser the user's logins
    without needing to close Chrome.

    Returns: number of files synced.
    """
    import shutil
    # Files that contain login sessions and cookies
    # These are in the "Default" subfolder of the User Data dir
    FILES_TO_SYNC = [
        "Cookies",
        "Cookies-journal",
        "Login Data",
        "Login Data-journal",
        "Web Data",
        "Web Data-journal",
        "Local State",               # encryption state
        "Preferences",
        "Secure Preferences",
        "Network",                    # folder: network cookies
    ]
    src_default = os.path.join(real_profile_dir, "Default")
    dst_default = os.path.join(automation_profile_dir, "Default")
    if not os.path.isdir(src_default):
        return 0
    os.makedirs(dst_default, exist_ok=True)
    synced = 0
    # Copy Local State from root of User Data (not Default subfolder)
    local_state = os.path.join(real_profile_dir, "Local State")
    if os.path.isfile(local_state):
        dst_ls = os.path.join(automation_profile_dir, "Local State")
        if not os.path.isfile(dst_ls):
            try:
                shutil.copy2(local_state, dst_ls)
                synced += 1
            except Exception:
                pass
    for fname in FILES_TO_SYNC:
        src = os.path.join(src_default, fname)
        dst = os.path.join(dst_default, fname)
        if os.path.isdir(src):
            # Copy directory (e.g., Network/)
            if not os.path.isdir(dst):
                try:
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                    synced += 1
                except Exception:
                    pass
        elif os.path.isfile(src):
            # Only copy if destination doesn't exist or is older
            try:
                if (not os.path.isfile(dst) or
                        os.path.getmtime(src) > os.path.getmtime(dst)):
                    shutil.copy2(src, dst)
                    synced += 1
            except (PermissionError, OSError):
                # Chrome may lock some files — skip those
                pass
    return synced

# Actions the automation agent can perform
_AUTOMATION_ACTIONS = {
    "NAVIGATE": "Go to a URL",
    "CLICK":    "Click element by visible text or CSS selector",
    "TYPE":     "Type text into an input field",
    "SCROLL":   "Scroll the page up or down",
    "WAIT":     "Wait for page to load or element to appear",
    "SCREENSHOT": "Take screenshot of current state",
    "DONE":     "Task complete",
}


class BrowserAutomation:
    """Visible browser automation — user watches AI perform real tasks.

    Key differences from BrowserEngine (headless):
    - Opens a VISIBLE browser window (user can watch)
    - Uses user's installed browser (Chrome/Edge/Firefox)
    - Persistent profile — login sessions survive between runs
    - Slow mode — delays between actions so user can follow
    - Multi-step task execution with model-based decisions

    Usage:
        auto = BrowserAutomation(browser="chrome")
        auto.start()
        result = auto.execute_task("go to linkedin and remove 10 posts")
        auto.stop()
    """

    def __init__(self, browser: str = "chrome",
                 profile_dir: str = "qor-data/browser-profile",
                 slow_mo: int = 500,
                 max_steps: int = 50,
                 model=None, tokenizer=None):
        self.browser_name = browser.lower()
        self.profile_dir = profile_dir
        self.slow_mo = slow_mo  # ms between actions
        self.max_steps = max_steps
        self.model = model or _browse_model
        self.tokenizer = tokenizer or _browse_tokenizer
        self.skill_loader = _browse_skill_loader
        self._skill_instructions = ""  # matched skill for current task
        self._playwright = None
        self._context = None
        self._page = None
        self._running = False
        self._task_log = []  # step-by-step log of actions taken
        self._lock = threading.Lock()
        self._screenshots_dir = os.path.join(
            os.path.dirname(profile_dir), "screenshots")

    @property
    def is_running(self) -> bool:
        return self._running and self._page is not None

    def start(self) -> str:
        """Launch visible browser with persistent profile.

        Auto-syncs cookies/logins from the user's real Chrome profile
        so the automation browser is already logged in everywhere.
        Subsequent runs keep their own session state too.
        """
        with self._lock:
            if self._running:
                return "Browser already running"

            try:
                from playwright.sync_api import sync_playwright
            except ImportError:
                return ("Browser automation requires playwright. "
                        "Install: pip install playwright && playwright install chromium")

            os.makedirs(self.profile_dir, exist_ok=True)
            os.makedirs(self._screenshots_dir, exist_ok=True)

            # Auto-sync cookies from user's real Chrome profile
            synced = 0
            if self.browser_name in ("chrome", "edge", "chromium"):
                real_profile = _find_real_chrome_profile()
                if real_profile:
                    try:
                        synced = _sync_cookies_from_real_profile(
                            real_profile, self.profile_dir)
                        if synced > 0:
                            logger.info(
                                f"[BrowserAuto] Synced {synced} files from "
                                f"real Chrome profile: {real_profile}")
                    except Exception as e:
                        logger.warning(
                            f"[BrowserAuto] Cookie sync failed: {e}")

            self._playwright = sync_playwright().start()

            # Select browser type
            browser_info = SUPPORTED_BROWSERS.get(self.browser_name, {})

            if self.browser_name == "firefox":
                browser_type = self._playwright.firefox
                channel = None
            elif self.browser_name == "safari":
                browser_type = self._playwright.webkit
                channel = None
            else:
                browser_type = self._playwright.chromium
                channel = browser_info.get("channel")

            # launch_persistent_context keeps cookies/sessions between runs
            launch_args = {
                "user_data_dir": self.profile_dir,
                "headless": False,
                "slow_mo": self.slow_mo,
                "viewport": {"width": 1366, "height": 768},
                "args": ["--no-sandbox", "--start-maximized"],
            }
            # WebKit (Safari) doesn't support --no-sandbox or --start-maximized
            if self.browser_name in ("safari", "firefox"):
                launch_args.pop("args", None)
            if channel:
                launch_args["channel"] = channel

            try:
                self._context = browser_type.launch_persistent_context(
                    **launch_args)
            except Exception as e:
                # If user's browser not found, fall back to bundled Chromium
                if channel:
                    launch_args.pop("channel", None)
                    self._context = self._playwright.chromium \
                        .launch_persistent_context(**launch_args)
                else:
                    self._playwright.stop()
                    self._playwright = None
                    return f"Failed to launch browser: {e}"

            # Get or create first page
            if self._context.pages:
                self._page = self._context.pages[0]
            else:
                self._page = self._context.new_page()

            self._running = True
            self._task_log = []
            label = browser_info.get("label", self.browser_name)
            sync_msg = (f", synced {synced} login files from your {label}"
                        if synced > 0 else "")
            return f"Browser opened: {label}{sync_msg}"

    def stop(self) -> str:
        """Close the visible browser. Profile/cookies are saved."""
        with self._lock:
            if not self._running:
                return "Browser not running"
            try:
                if self._context:
                    self._context.close()
            except Exception:
                pass
            try:
                if self._playwright:
                    self._playwright.stop()
            except Exception:
                pass
            self._context = None
            self._page = None
            self._playwright = None
            self._running = False
            return "Browser closed (sessions saved)"

    def status(self) -> dict:
        """Current automation status."""
        return {
            "running": self._running,
            "browser": self.browser_name,
            "profile_dir": self.profile_dir,
            "current_url": self._page.url if self._page else None,
            "current_title": (self._page.title()
                              if self._page else None),
            "steps_taken": len(self._task_log),
            "task_log": self._task_log[-10:],  # last 10 steps
        }

    def execute_task(self, task: str, callback=None) -> dict:
        """Execute a multi-step automation task.

        Args:
            task: Natural language task description
            callback: Optional fn(step_info) called after each step for live UI updates

        Returns:
            {ok, steps, summary, log}
        """
        if not self._running or not self._page:
            return {"ok": False, "error": "Browser not running. Call start() first."}

        # Match a skill for this task (e.g. "linkedin" → linkedin-automation skill)
        self._skill_instructions = ""
        if self.skill_loader:
            try:
                matched = self.skill_loader.match(task)
                if matched:
                    self._skill_instructions = matched.instructions
            except Exception:
                pass

        self._task_log = []
        steps_done = 0

        for step in range(self.max_steps):
            # ── OBSERVE: Get current page state ──
            try:
                page_state = self._observe()
            except Exception as e:
                self._log_step(step, "ERROR", f"Observe failed: {e}")
                break

            # ── DECIDE: Ask model what to do next ──
            action, arg = self._decide(task, page_state, step)
            self._log_step(step, action, arg)

            if callback:
                try:
                    callback({"step": step, "action": action, "arg": arg,
                              "url": self._page.url, "title": self._page.title()})
                except Exception:
                    pass

            # ── ACT: Execute the action ──
            if action == "DONE":
                steps_done = step + 1
                break

            try:
                self._act(action, arg)
            except Exception as e:
                self._log_step(step, "ERROR", f"Action failed: {e}")
                # Don't break — model can try a different action next step

            steps_done = step + 1

        return {
            "ok": True,
            "steps": steps_done,
            "summary": self._task_log[-1]["arg"] if self._task_log else "No action taken",
            "log": list(self._task_log),
            "final_url": self._page.url if self._page else None,
        }

    # ── Internal: Observe ──

    def _observe(self) -> dict:
        """Extract current page state for the model."""
        page = self._page
        title = page.title() or ""
        url = page.url or ""

        # Remove noisy elements before extraction
        try:
            page.evaluate("""
                for (const sel of ['script', 'style', 'noscript',
                                   '.cookie-banner', '[class*="cookie"]',
                                   '[class*="consent"]', '.popup', '.modal']) {
                    document.querySelectorAll(sel).forEach(el => el.remove());
                }
            """)
        except Exception:
            pass

        # Get visible text (truncated)
        try:
            text = page.inner_text("body")
            text = re.sub(r'\n{3,}', '\n\n', text)
            text = re.sub(r'[ \t]{2,}', ' ', text)
            text = text.strip()[:3000]
        except Exception:
            text = ""

        # Get clickable elements (links, buttons)
        try:
            clickables = page.evaluate("""
                () => {
                    const items = [];
                    const els = document.querySelectorAll(
                        'a, button, [role="button"], [onclick], input[type="submit"]'
                    );
                    els.forEach((el, i) => {
                        if (i >= 30) return;
                        const text = (el.innerText || el.getAttribute('aria-label')
                                     || el.getAttribute('title') || '').trim();
                        if (text && text.length < 80) {
                            items.push(text.substring(0, 60));
                        }
                    });
                    return items;
                }
            """)
        except Exception:
            clickables = []

        # Get input fields
        try:
            inputs = page.evaluate("""
                () => {
                    const items = [];
                    document.querySelectorAll(
                        'input, textarea, [contenteditable="true"]'
                    ).forEach((el, i) => {
                        if (i >= 10) return;
                        const label = el.getAttribute('placeholder')
                            || el.getAttribute('aria-label')
                            || el.getAttribute('name') || '';
                        const type = el.getAttribute('type') || 'text';
                        if (type !== 'hidden') {
                            items.push({label: label.substring(0, 40), type});
                        }
                    });
                    return items;
                }
            """)
        except Exception:
            inputs = []

        return {
            "title": title,
            "url": url,
            "text": text,
            "clickables": clickables[:20],
            "inputs": inputs[:10],
        }

    # ── Internal: Decide ──

    def _decide(self, task: str, state: dict, step: int) -> tuple:
        """Ask model what action to take next."""
        if self.model is not None and self.tokenizer is not None:
            return self._model_decide(task, state, step)
        return self._heuristic_decide(task, state, step)

    def _model_decide(self, task: str, state: dict, step: int) -> tuple:
        """Use the LLM to decide next action."""
        clickables_str = "\n".join(
            f"  [{i}] {c}" for i, c in enumerate(state["clickables"][:15]))
        inputs_str = "\n".join(
            f"  [{i}] {inp['label']} ({inp['type']})"
            for i, inp in enumerate(state["inputs"][:5]))
        history_str = "\n".join(
            f"  Step {s['step']}: {s['action']} {s['arg']}"
            for s in self._task_log[-5:])

        # Build system prompt — inject skill instructions if matched
        skill_section = ""
        if self._skill_instructions:
            # Truncate skill to fit in context alongside page state
            skill_text = self._skill_instructions[:1500]
            skill_section = f"\n\n## Skill Instructions (follow these steps):\n{skill_text}\n"

        system_msg = (
            "You are a browser automation agent controlling a visible browser. "
            "The user is watching. Perform the task step by step."
            f"{skill_section}\n\n"
            "Reply with EXACTLY one line — one of these actions:\n"
            "NAVIGATE <url> — go to a URL\n"
            "CLICK <visible text of button/link> — click an element\n"
            "TYPE <placeholder or label> | <text to type> — type into a field\n"
            "SCROLL down or SCROLL up — scroll the page\n"
            "WAIT <seconds> — wait for loading\n"
            "DONE <summary of what was accomplished> — task is complete"
        )
        user_msg = (
            f"Task: {task}\n"
            f"Step: {step + 1}/{self.max_steps}\n"
            f"Current page: {state['title']}\n"
            f"URL: {state['url']}\n\n"
            f"Clickable elements:\n{clickables_str or '  (none visible)'}\n\n"
            f"Input fields:\n{inputs_str or '  (none visible)'}\n\n"
            f"Page text (excerpt):\n{state['text'][:1500]}\n\n"
            f"Actions taken so far:\n{history_str or '  (none yet)'}\n\n"
            f"What should I do next?"
        )

        try:
            import torch
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ]
            if hasattr(self.tokenizer, 'format_chat'):
                input_ids = self.tokenizer.format_chat(
                    messages, add_generation_prompt=True)
            else:
                input_ids = self.tokenizer.encode(
                    f"{system_msg}\n\n{user_msg}\nReply:")

            device = next(self.model.parameters()).device
            tokens = torch.tensor([input_ids], device=device)
            with torch.no_grad():
                output = self.model.generate(tokens, max_new_tokens=60,
                                             temperature=0.3)
            response = self.tokenizer.decode(
                output[0][len(input_ids):].tolist())
            response = response.strip().split("\n")[0].strip()

            return self._parse_automation_action(response)
        except Exception:
            return self._heuristic_decide(task, state, step)

    def _heuristic_decide(self, task: str, state: dict, step: int) -> tuple:
        """Rule-based fallback when model is unavailable."""
        task_lower = task.lower()
        url = state["url"].lower()
        text = state["text"].lower()
        clickables = [c.lower() for c in state.get("clickables", [])]

        # Step 0: navigate to the target site if not there yet
        if step == 0:
            # ALWAYS check for explicit URLs in the task FIRST
            # (user may provide a specific URL like company admin page)
            m = _URL_PATTERN.search(task)
            if m:
                return ("NAVIGATE", m.group(0))
            # LinkedIn: go to activity page for post deletion tasks
            if "linkedin" in task_lower and any(
                    w in task_lower for w in ["delete", "remove", "clean",
                                               "activity"]):
                return ("NAVIGATE",
                        "https://www.linkedin.com/in/me/recent-activity/all/")
            # LinkedIn: messaging
            if "linkedin" in task_lower and any(
                    w in task_lower for w in ["message", "send", "dm"]):
                return ("NAVIGATE",
                        "https://www.linkedin.com/messaging/")
            # LinkedIn: connections / invitations
            if "linkedin" in task_lower and any(
                    w in task_lower for w in ["accept", "reject", "invitation",
                                               "connection request"]):
                return ("NAVIGATE",
                        "https://www.linkedin.com/mynetwork/invitation-manager/")
            # LinkedIn: profile update
            if "linkedin" in task_lower and any(
                    w in task_lower for w in ["profile", "headline", "about",
                                               "experience", "education"]):
                return ("NAVIGATE",
                        "https://www.linkedin.com/in/me/")
            # LinkedIn: post / share / create
            if "linkedin" in task_lower and any(
                    w in task_lower for w in ["post", "share", "create",
                                               "publish", "write"]):
                return ("NAVIGATE",
                        "https://www.linkedin.com/feed/")
            # LinkedIn general
            if "linkedin" in task_lower:
                return ("NAVIGATE", "https://www.linkedin.com/feed/")
            # Extract target site from task
            for site, site_url in _SITE_SHORTCUTS.items():
                if site in task_lower:
                    if site not in url:
                        return ("NAVIGATE", site_url)
            # Check common site names
            for name in ["gmail", "youtube", "twitter",
                         "facebook", "instagram", "reddit", "github"]:
                if name in task_lower:
                    return ("NAVIGATE", f"https://www.{name}.com")

        # LinkedIn: check login state
        if "linkedin" in url:
            if "sign in" in text or "join now" in text:
                return ("DONE", "Please log into LinkedIn in the browser window, "
                        "then run the task again")

        # LinkedIn: delete post flow
        if "linkedin" in url and any(w in task_lower for w in
                                      ["delete", "remove", "clean"]):
            # Try to click confirmation "Delete" button first (in dialog)
            for c in clickables:
                if c.strip().lower() == "delete":
                    return ("CLICK", c)
            # Try to find and click three-dot menu
            for c in clickables:
                if c in ("...", "more", "open control menu"):
                    return ("CLICK", c)
            # Try to find Delete in dropdown
            for c in clickables:
                if "delete" in c.lower():
                    return ("CLICK", c)
            # Scroll to find posts
            return ("SCROLL", "down")

        # LinkedIn: accept connection requests
        if "linkedin" in url and any(w in task_lower for w in
                                      ["accept", "invitation"]):
            for c in clickables:
                if "accept" in c.lower():
                    return ("CLICK", c)
            return ("SCROLL", "down")

        # LinkedIn: like posts
        if "linkedin" in url and any(w in task_lower for w in
                                      ["like", "react"]):
            for c in clickables:
                if "like" in c.lower() and "unlike" not in c.lower():
                    return ("CLICK", c)
            return ("SCROLL", "down")

        # LinkedIn: comment on posts
        if "linkedin" in url and "comment" in task_lower:
            for c in clickables:
                if "comment" in c.lower():
                    return ("CLICK", c)
            return ("SCROLL", "down")

        # LinkedIn: create a post
        if "linkedin" in url and any(w in task_lower for w in
                                      ["post", "create", "publish", "write"]):
            for c in clickables:
                if "start a post" in c.lower():
                    return ("CLICK", c)

        # LinkedIn: send message
        if "linkedin" in url and any(w in task_lower for w in
                                      ["message", "send", "dm"]):
            for c in clickables:
                if "new message" in c.lower() or "compose" in c.lower():
                    return ("CLICK", c)

        # LinkedIn: connect with someone
        if "linkedin" in url and "connect" in task_lower:
            for c in clickables:
                if "connect" in c.lower():
                    return ("CLICK", c)

        # Generic: scroll on the right page
        if step > 0 and step < 10:
            return ("SCROLL", "down")

        # After many steps, done
        if step >= 10:
            return ("DONE", "Heuristic completed — model needed for complex tasks")

        return ("DONE", "Could not determine next action")

    def _parse_automation_action(self, response: str) -> tuple:
        """Parse model response into (action, argument)."""
        raw = response.strip()
        raw_upper = raw.upper()

        for action in _AUTOMATION_ACTIONS:
            if raw_upper.startswith(action):
                arg = raw[len(action):].strip().strip("—-:").strip()
                return (action, arg)

        return ("DONE", "Could not parse action")

    # ── Internal: Act ──

    def _act(self, action: str, arg: str):
        """Execute an action on the visible browser."""
        page = self._page

        if action == "NAVIGATE":
            url = arg if arg.startswith("http") else f"https://{arg}"
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            page.wait_for_timeout(2000)

        elif action == "CLICK":
            # Try multiple strategies to find the element
            clicked = False
            # 1. Try by exact text
            try:
                page.get_by_text(arg, exact=False).first.click(timeout=5000)
                clicked = True
            except Exception:
                pass
            # 2. Try by role + name
            if not clicked:
                try:
                    page.get_by_role("button", name=arg).first.click(
                        timeout=3000)
                    clicked = True
                except Exception:
                    pass
            # 3. Try by link text
            if not clicked:
                try:
                    page.get_by_role("link", name=arg).first.click(
                        timeout=3000)
                    clicked = True
                except Exception:
                    pass
            # 4. Try CSS selector directly
            if not clicked and (arg.startswith(".") or arg.startswith("#")
                                or arg.startswith("[")):
                try:
                    page.click(arg, timeout=3000)
                    clicked = True
                except Exception:
                    pass
            # 5. Try aria-label
            if not clicked:
                try:
                    page.click(f'[aria-label*="{arg}"]', timeout=3000)
                    clicked = True
                except Exception:
                    pass

            if not clicked:
                raise RuntimeError(f"Could not find element: {arg}")

            page.wait_for_timeout(1500)

        elif action == "TYPE":
            # Format: "placeholder or label | text to type"
            parts = arg.split("|", 1)
            if len(parts) == 2:
                selector_hint = parts[0].strip()
                text = parts[1].strip()
            else:
                selector_hint = ""
                text = arg

            typed = False
            if selector_hint:
                # Try by placeholder
                try:
                    page.get_by_placeholder(selector_hint, exact=False) \
                        .first.fill(text)
                    typed = True
                except Exception:
                    pass
                # Try by label
                if not typed:
                    try:
                        page.get_by_label(selector_hint, exact=False) \
                            .first.fill(text)
                        typed = True
                    except Exception:
                        pass

            if not typed:
                # Type into currently focused element
                try:
                    page.keyboard.type(text)
                    typed = True
                except Exception:
                    pass

            if not typed:
                raise RuntimeError(f"Could not type into: {selector_hint}")

            page.wait_for_timeout(500)

        elif action == "SCROLL":
            direction = arg.lower() if arg else "down"
            amount = 600 if direction == "down" else -600
            page.evaluate(f"window.scrollBy(0, {amount})")
            page.wait_for_timeout(1000)

        elif action == "WAIT":
            try:
                seconds = float(arg) if arg else 2
                seconds = min(seconds, 10)  # cap at 10s
            except ValueError:
                seconds = 2
            page.wait_for_timeout(int(seconds * 1000))

        elif action == "SCREENSHOT":
            ts = int(time.time())
            path = os.path.join(self._screenshots_dir, f"auto_{ts}.png")
            page.screenshot(path=path)

    # ── Internal: Logging ──

    def _log_step(self, step: int, action: str, arg: str):
        """Record a step in the task log."""
        self._task_log.append({
            "step": step,
            "action": action,
            "arg": arg or "",
            "time": time.strftime("%H:%M:%S"),
            "url": self._page.url if self._page else "",
        })


# ==============================================================================
# AUTOMATION SINGLETON (separate from headless BrowserEngine)
# ==============================================================================

_automation: Optional[BrowserAutomation] = None
_automation_lock = threading.Lock()

# Default browser preference (can be changed via API/UI)
_default_browser = "chrome"


def get_automation() -> Optional[BrowserAutomation]:
    """Get the current automation instance (None if not started)."""
    return _automation


def set_default_browser(browser: str) -> str:
    """Set the default browser for automation. Returns confirmation."""
    global _default_browser
    browser = browser.lower()
    if browser not in SUPPORTED_BROWSERS:
        return (f"Unknown browser: {browser}. "
                f"Choose from: {', '.join(SUPPORTED_BROWSERS.keys())}")
    _default_browser = browser
    label = SUPPORTED_BROWSERS[browser]["label"]
    return f"Default browser set to: {label}"


def start_automation(browser: str = None,
                     profile_dir: str = "qor-data/browser-profile") -> str:
    """Start a visible browser automation session."""
    global _automation
    with _automation_lock:
        if _automation and _automation.is_running:
            return "Browser automation already running"
        _automation = BrowserAutomation(
            browser=browser or _default_browser,
            profile_dir=profile_dir,
            model=_browse_model,
            tokenizer=_browse_tokenizer,
        )
        return _automation.start()


def stop_automation() -> str:
    """Stop the visible browser."""
    global _automation
    with _automation_lock:
        if not _automation or not _automation.is_running:
            return "Browser automation not running"
        result = _automation.stop()
        _automation = None
        return result


def run_automation_task(task: str, callback=None) -> dict:
    """Execute a task on the visible browser."""
    if not _automation or not _automation.is_running:
        return {"ok": False, "error": "Browser not running. Start it first."}
    return _automation.execute_task(task, callback=callback)


def has_real_chrome_profile() -> bool:
    """Check if the user's real Chrome profile exists (for cookie sync)."""
    return _find_real_chrome_profile() is not None


def automation_status() -> dict:
    """Get automation status."""
    if not _automation:
        return {
            "running": False,
            "browser": _default_browser,
            "available_browsers": list(SUPPORTED_BROWSERS.keys()),
        }
    s = _automation.status()
    s["available_browsers"] = list(SUPPORTED_BROWSERS.keys())
    return s

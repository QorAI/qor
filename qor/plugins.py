"""
QOR Tool Plugins — Add/Edit/Remove Tools WITHOUT Touching Model Code
======================================================================
NO new model release needed. NO code changes needed.

How it works:
  1. Tools live in a `plugins/` folder as simple Python files
  2. Config lives in `tools_config.yaml` (or .json)
  3. QOR auto-loads everything at startup
  4. Hot-reload: change a file → tool updates instantly (no restart needed)

To add a new tool:
  1. Drop a .py file in plugins/    (or)
  2. Add an entry to tools_config.yaml

That's it. No model retraining. No code editing. No redeployment.

Architecture:
  ┌─────────────────────────────────────────────────────────┐
  │                                                         │
  │  MODEL (frozen weights)                                 │
  │  - Trained ONCE                                         │
  │  - Never changes when you add/remove tools              │
  │  - Only retrain to improve language understanding       │
  │                                                         │
  ├─────────────────────────────────────────────────────────┤
  │                                                         │
  │  CONFIDENCE GATE (routing logic)                        │
  │  - Decides: answer internally OR call a tool            │
  │  - Reads tool registry at runtime                       │
  │  - No changes needed when tools change                  │
  │                                                         │
  ├─────────────────────────────────────────────────────────┤
  │                                                         │
  │  TOOL REGISTRY (dynamic, hot-reloadable)      ← HERE   │
  │  - Loads from plugins/ folder                           │
  │  - Loads from tools_config.yaml                         │
  │  - Loads from tools_config.json                         │
  │  - Add/edit/remove anytime — no model change            │
  │  - Hot reload: file change → tool updates               │
  │                                                         │
  └─────────────────────────────────────────────────────────┘

3 ways to add tools (pick any):

  METHOD 1: Plugin file (most flexible)
  ──────────────────────────────────────
  Create plugins/my_tool.py:
  
    def handler(query: str) -> str:
        # call your API here
        return "result"
    
    TOOL_NAME = "my_tool"
    TOOL_DESCRIPTION = "What this tool does"
    TOOL_KEYWORDS = ["keyword1", "keyword2"]
    TOOL_CATEGORIES = ["general"]

  METHOD 2: Config file (easiest, no Python needed)
  ─────────────────────────────────────────────────
  Add to tools_config.yaml:
  
    - name: my_api
      url: "https://api.example.com/search?q={query}"
      method: GET
      description: "Search my API"
      keywords: [keyword1, keyword2]
      response_path: "data.results[0].text"

  METHOD 3: Runtime (from code)
  ─────────────────────────────
    from qor.plugins import add_tool
    
    def my_func(query):
        return requests.get(f"https://api.com?q={query}").json()
    
    add_tool(gate, "my_func", "Description", my_func, ["keyword"])
"""

import os
import sys
import json
import time
import importlib
import importlib.util
import urllib.request
import urllib.parse
import re
from typing import Dict, List, Optional, Callable, Any
from pathlib import Path
from datetime import datetime


# ==============================================================================
# TOOL PLUGIN — A single loaded tool
# ==============================================================================

class ToolPlugin:
    """A dynamically loaded tool."""

    def __init__(self, name: str, description: str, handler: Callable,
                 keywords: List[str] = None, categories: List[str] = None,
                 source: str = "code", config: dict = None):
        self.name = name
        self.description = description
        self.handler = handler
        self.keywords = keywords or []
        self.categories = categories or ["general"]
        self.source = source       # "plugin_file", "config", "builtin", "code"
        self.config = config or {}
        self.loaded_at = datetime.now().isoformat()
        self.call_count = 0
        self.last_error = None
        self.enabled = True

    def call(self, query: str) -> str:
        """Execute this tool."""
        if not self.enabled:
            return f"[Tool '{self.name}' is disabled]"
        try:
            self.call_count += 1
            result = self.handler(query)
            self.last_error = None
            return str(result)
        except Exception as e:
            self.last_error = str(e)
            return f"[Tool error: {e}]"

    def matches(self, query: str) -> bool:
        """Check if this tool should handle a query."""
        q = query.lower()
        return any(kw in q for kw in self.keywords)


# ==============================================================================
# PLUGIN MANAGER — Loads tools from files, configs, and code
# ==============================================================================

class PluginManager:
    """
    Manages tool plugins from multiple sources.
    
    Sources (in load order):
      1. Built-in tools (from tools.py)
      2. Plugin .py files (from plugins/ directory)
      3. Config file (tools_config.yaml or .json)
      4. Runtime additions (via add_tool())
    """

    def __init__(self, plugins_dir: str = "plugins",
                 config_path: str = "tools_config"):
        self.plugins_dir = plugins_dir
        self.config_path = config_path
        self.tools: Dict[str, ToolPlugin] = {}
        self._file_timestamps: Dict[str, float] = {}  # for hot reload
        self._watchers_started = False

    # ------------------------------------------------------------------
    # LOADING
    # ------------------------------------------------------------------

    def load_all(self, include_builtins: bool = True):
        """Load tools from all sources."""
        if include_builtins:
            self._load_builtins()
        self._load_plugin_files()
        self._load_config_file()
        print(f"  [Plugins] {len(self.tools)} tools loaded "
              f"(builtins + {self._count_by_source('plugin_file')} plugins "
              f"+ {self._count_by_source('config')} from config)")

    def _load_builtins(self):
        """Load built-in tools from tools.py."""
        try:
            from qor.tools import ALL_TOOLS
            for name, (func, desc, cats) in ALL_TOOLS.items():
                self.tools[name] = ToolPlugin(
                    name=name,
                    description=desc,
                    handler=func,
                    keywords=cats,
                    categories=cats,
                    source="builtin",
                )
        except ImportError:
            print("  [Plugins] Warning: Could not load builtin tools")

    def _load_plugin_files(self):
        """
        Load .py files from plugins/ directory.
        
        Each plugin file should have:
          - handler(query: str) -> str  (required)
          - TOOL_NAME = "name"          (optional, defaults to filename)
          - TOOL_DESCRIPTION = "..."    (optional)
          - TOOL_KEYWORDS = [...]       (optional)
          - TOOL_CATEGORIES = [...]     (optional)
        """
        if not os.path.isdir(self.plugins_dir):
            os.makedirs(self.plugins_dir, exist_ok=True)
            self._create_example_plugin()
            return

        for fname in sorted(os.listdir(self.plugins_dir)):
            if not fname.endswith('.py') or fname.startswith('_'):
                continue

            fpath = os.path.join(self.plugins_dir, fname)
            self._load_single_plugin(fpath)

    def _load_single_plugin(self, fpath: str) -> bool:
        """Load a single plugin file."""
        fname = os.path.basename(fpath)
        mod_name = f"qor_plugin_{os.path.splitext(fname)[0]}"

        try:
            # Record timestamp for hot reload
            self._file_timestamps[fpath] = os.path.getmtime(fpath)

            # Load the module
            spec = importlib.util.spec_from_file_location(mod_name, fpath)
            if spec is None or spec.loader is None:
                return False

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Extract tool info
            handler = getattr(module, 'handler', None)
            if handler is None:
                # Try alternate names
                for attr_name in ['run', 'execute', 'call', 'main']:
                    handler = getattr(module, attr_name, None)
                    if handler:
                        break

            if handler is None:
                print(f"  [Plugins] Warning: {fname} has no handler() function")
                return False

            name = getattr(module, 'TOOL_NAME',
                           os.path.splitext(fname)[0])
            description = getattr(module, 'TOOL_DESCRIPTION',
                                  f"Plugin: {name}")
            keywords = getattr(module, 'TOOL_KEYWORDS', [name])
            categories = getattr(module, 'TOOL_CATEGORIES', ["general"])

            self.tools[name] = ToolPlugin(
                name=name,
                description=description,
                handler=handler,
                keywords=keywords,
                categories=categories,
                source="plugin_file",
                config={"file": fpath},
            )
            print(f"  [Plugins] Loaded: {name} (from {fname})")
            return True

        except Exception as e:
            print(f"  [Plugins] Error loading {fname}: {e}")
            return False

    def _load_config_file(self):
        """
        Load tools from config file (YAML or JSON).
        
        Supports two types:
          1. Simple URL tools (just an API endpoint)
          2. Custom handler tools (inline Python)
        
        Example YAML:
          tools:
            - name: my_api
              url: "https://api.example.com/data?q={query}"
              method: GET
              description: "My custom API"
              keywords: ["my_api", "custom"]
              headers:
                Authorization: "Bearer YOUR_KEY"
              response_path: "results[0].text"
        
        Example JSON:
          {"tools": [{"name": "my_api", "url": "...", ...}]}
        """
        # Try YAML first, then JSON
        config = None
        config_file = None

        for ext in ['.yaml', '.yml', '.json']:
            path = self.config_path + ext
            if os.path.exists(path):
                config_file = path
                break

        if not config_file:
            return

        try:
            with open(config_file, 'r') as f:
                content = f.read()

            if config_file.endswith('.json'):
                config = json.loads(content)
            else:
                # Simple YAML parser (no pyyaml dependency needed)
                config = self._parse_simple_yaml(content)

            if not config:
                return

            tools_list = config if isinstance(config, list) else \
                         config.get('tools', [])

            for tool_config in tools_list:
                self._load_config_tool(tool_config)

            self._file_timestamps[config_file] = os.path.getmtime(config_file)

        except Exception as e:
            print(f"  [Plugins] Error loading config {config_file}: {e}")

    def _load_config_tool(self, cfg: dict):
        """Load a single tool from config."""
        name = cfg.get('name', '')
        if not name:
            return

        url = cfg.get('url', '')
        method = cfg.get('method', 'GET').upper()
        headers = cfg.get('headers', {})
        response_path = cfg.get('response_path', '')
        body_template = cfg.get('body', '')
        description = cfg.get('description', f"API: {name}")
        keywords = cfg.get('keywords', [name])
        categories = cfg.get('categories', ['general'])
        timeout = cfg.get('timeout', 15)
        enabled = cfg.get('enabled', True)

        # Create handler function for this API config
        def make_handler(url, method, headers, response_path,
                         body_template, timeout):
            def handler(query: str) -> str:
                # Fill {query} placeholder in URL
                final_url = url.replace('{query}',
                                        urllib.parse.quote(query))

                # Fill any other {field} placeholders
                for match in re.findall(r'\{(\w+)\}', final_url):
                    # Try to extract from query
                    val = _extract_field(query, match)
                    final_url = final_url.replace(f'{{{match}}}', val)

                try:
                    if method == 'GET':
                        req = urllib.request.Request(final_url)
                    else:
                        body = body_template.replace('{query}', query) \
                            if body_template else ''
                        req = urllib.request.Request(
                            final_url,
                            data=body.encode() if body else None,
                            method=method
                        )

                    req.add_header('User-Agent', 'QOR-AI-Agent/1.0')
                    for k, v in headers.items():
                        req.add_header(k, v)

                    with urllib.request.urlopen(req, timeout=timeout) as resp:
                        data = json.loads(resp.read().decode())

                    # Extract response using path
                    if response_path:
                        return _extract_path(data, response_path)

                    # Auto-format JSON response
                    if isinstance(data, dict):
                        return _format_dict(data)
                    return json.dumps(data, indent=2)[:1000]

                except Exception as e:
                    return f"API error ({name}): {e}"

            return handler

        handler = make_handler(url, method, headers, response_path,
                               body_template, timeout)

        self.tools[name] = ToolPlugin(
            name=name,
            description=description,
            handler=handler,
            keywords=keywords,
            categories=categories,
            source="config",
            config=cfg,
        )
        self.tools[name].enabled = enabled

    # ------------------------------------------------------------------
    # HOT RELOAD
    # ------------------------------------------------------------------

    def check_for_updates(self) -> List[str]:
        """
        Check if any plugin files have changed.
        Call this periodically for hot-reload.
        Returns list of updated tool names.
        """
        updated = []

        # Check plugin files
        if os.path.isdir(self.plugins_dir):
            for fname in os.listdir(self.plugins_dir):
                if not fname.endswith('.py') or fname.startswith('_'):
                    continue
                fpath = os.path.join(self.plugins_dir, fname)
                mtime = os.path.getmtime(fpath)
                if fpath in self._file_timestamps:
                    if mtime > self._file_timestamps[fpath]:
                        print(f"  [Plugins] Reloading: {fname}")
                        self._load_single_plugin(fpath)
                        updated.append(fname)
                else:
                    # New file
                    print(f"  [Plugins] New plugin: {fname}")
                    self._load_single_plugin(fpath)
                    updated.append(fname)

        # Check config files
        for ext in ['.yaml', '.yml', '.json']:
            path = self.config_path + ext
            if os.path.exists(path):
                mtime = os.path.getmtime(path)
                if path in self._file_timestamps and \
                   mtime > self._file_timestamps[path]:
                    print(f"  [Plugins] Config changed, reloading...")
                    self._load_config_file()
                    updated.append(path)

        return updated

    def reload(self, name: str = None):
        """Force reload a specific tool or all tools."""
        if name:
            tool = self.tools.get(name)
            if tool and tool.source == "plugin_file":
                fpath = tool.config.get("file", "")
                if fpath and os.path.exists(fpath):
                    self._load_single_plugin(fpath)
                    print(f"  [Plugins] Reloaded: {name}")
        else:
            self.tools.clear()
            self.load_all()

    # ------------------------------------------------------------------
    # TOOL ACCESS
    # ------------------------------------------------------------------

    def get(self, name: str) -> Optional[ToolPlugin]:
        """Get a tool by name."""
        return self.tools.get(name)

    def call(self, name: str, query: str) -> str:
        """Call a tool by name."""
        tool = self.tools.get(name)
        if not tool:
            return f"Tool not found: {name}"
        return tool.call(query)

    def detect_intent(self, query: str) -> Optional[str]:
        """Detect which tool should handle a query."""
        q = query.lower()
        best_match = None
        best_count = 0

        for name, tool in self.tools.items():
            if not tool.enabled:
                continue
            match_count = sum(1 for kw in tool.keywords if kw in q)
            if match_count > best_count:
                best_count = match_count
                best_match = name

        return best_match

    def register_with_gate(self, gate):
        """Register all tools with a ConfidenceGate."""
        for name, tool in self.tools.items():
            if tool.enabled:
                gate.tools.register(
                    name, tool.description, tool.handler, tool.categories
                )
        print(f"  [Plugins] Registered {len(self.tools)} tools with gate")

    # ------------------------------------------------------------------
    # MANAGEMENT
    # ------------------------------------------------------------------

    def add(self, name: str, description: str, handler: Callable,
            keywords: List[str] = None, categories: List[str] = None):
        """Add a tool at runtime."""
        self.tools[name] = ToolPlugin(
            name=name, description=description, handler=handler,
            keywords=keywords or [name],
            categories=categories or ["general"],
            source="code",
        )
        print(f"  [Plugins] Added: {name}")

    def remove(self, name: str):
        """Remove a tool."""
        if name in self.tools:
            del self.tools[name]
            print(f"  [Plugins] Removed: {name}")

    def enable(self, name: str):
        """Enable a tool."""
        if name in self.tools:
            self.tools[name].enabled = True

    def disable(self, name: str):
        """Disable a tool (without removing)."""
        if name in self.tools:
            self.tools[name].enabled = False

    def list_tools(self) -> List[dict]:
        """List all tools."""
        return [{
            "name": t.name,
            "description": t.description,
            "source": t.source,
            "enabled": t.enabled,
            "calls": t.call_count,
            "keywords": t.keywords,
        } for t in self.tools.values()]

    def stats(self):
        """Print plugin statistics."""
        by_source = {}
        for t in self.tools.values():
            by_source[t.source] = by_source.get(t.source, 0) + 1
        total_calls = sum(t.call_count for t in self.tools.values())
        disabled = sum(1 for t in self.tools.values() if not t.enabled)

        print(f"\n  Plugin Stats:")
        print(f"    Total tools:  {len(self.tools)}")
        print(f"    Enabled:      {len(self.tools) - disabled}")
        print(f"    Disabled:     {disabled}")
        print(f"    Total calls:  {total_calls}")
        for src, count in sorted(by_source.items()):
            print(f"    {src}: {count}")

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    def _count_by_source(self, source: str) -> int:
        return sum(1 for t in self.tools.values() if t.source == source)

    def _create_example_plugin(self):
        """Create example plugin file."""
        example = '''"""
Example QOR Plugin — Copy this to create your own tools!
Drop .py files in this plugins/ folder. QOR loads them automatically.
"""

# REQUIRED: The function that handles queries
def handler(query: str) -> str:
    """
    Your tool logic here.
    
    Args:
        query: The user's question/request
    Returns:
        String with the answer/data
    """
    # Example: call an API
    # import urllib.request, json
    # url = f"https://api.example.com/search?q={query}"
    # data = json.loads(urllib.request.urlopen(url).read())
    # return data["answer"]
    
    return f"Example plugin received: {query}"


# OPTIONAL: Tool metadata (defaults to filename if not set)
TOOL_NAME = "example_plugin"
TOOL_DESCRIPTION = "Example plugin — replace with your own"
TOOL_KEYWORDS = ["example", "test"]
TOOL_CATEGORIES = ["general"]
'''
        path = os.path.join(self.plugins_dir, "_example_plugin.py")
        os.makedirs(self.plugins_dir, exist_ok=True)
        with open(path, 'w') as f:
            f.write(example)

        # Also create example config
        example_config = {
            "tools": [
                {
                    "name": "example_api",
                    "url": "https://api.duckduckgo.com/?q={query}&format=json&no_html=1",
                    "method": "GET",
                    "description": "Example: DuckDuckGo search",
                    "keywords": ["example_search"],
                    "categories": ["general"],
                    "response_path": "AbstractText",
                    "enabled": False,
                    "_comment": "Set enabled: true to activate. "
                                "Change url to your API."
                }
            ]
        }
        config_path = self.config_path + ".json"
        with open(config_path, 'w') as f:
            json.dump(example_config, f, indent=2)

        print(f"  [Plugins] Created example plugin: {self.plugins_dir}/")
        print(f"  [Plugins] Created example config: {config_path}")

    def _parse_simple_yaml(self, content: str) -> dict:
        """
        Minimal YAML parser (handles simple tool configs).
        For complex YAML, install pyyaml.
        """
        try:
            import yaml
            return yaml.safe_load(content)
        except ImportError:
            pass

        # Fallback: treat as JSON-like
        # This handles the most common case
        content = content.strip()
        if content.startswith('{') or content.startswith('['):
            return json.loads(content)

        # Very basic YAML → dict conversion
        print("  [Plugins] Note: Install pyyaml for YAML support. "
              "Using JSON config instead.")
        return {}


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def _extract_field(query: str, field: str) -> str:
    """Try to extract a named field from a query."""
    # Common extractions
    if field == "query":
        return urllib.parse.quote(query)
    if field == "city":
        for prefix in ["weather in", "temperature in"]:
            if prefix in query.lower():
                return urllib.parse.quote(
                    query.lower().split(prefix)[1].strip())
        return urllib.parse.quote(query)
    if field == "symbol":
        symbols = {"bitcoin": "BTC", "btc": "BTC", "ethereum": "ETH",
                   "eth": "ETH", "solana": "SOL"}
        for k, v in symbols.items():
            if k in query.lower():
                return v
        return "BTC"
    return urllib.parse.quote(query)


def _extract_path(data: Any, path: str) -> str:
    """Extract a value from nested dict/list using dot notation.
    Example: 'data.results[0].text'
    """
    current = data
    for part in path.replace('[', '.[').split('.'):
        if not part:
            continue
        if part.startswith('[') and part.endswith(']'):
            # Array index
            idx = int(part[1:-1])
            if isinstance(current, list) and idx < len(current):
                current = current[idx]
            else:
                return f"[Index {idx} out of range]"
        elif isinstance(current, dict):
            current = current.get(part, f"[Key '{part}' not found]")
        else:
            return str(current)
    return str(current) if current else "[Empty]"


def _format_dict(data: dict, max_len: int = 800) -> str:
    """Format a dict as readable text."""
    lines = []
    for k, v in data.items():
        if isinstance(v, (dict, list)):
            v = json.dumps(v)[:200]
        lines.append(f"  {k}: {v}")
    result = "\n".join(lines)
    return result[:max_len]


# ==============================================================================
# CONVENIENCE — Quick add tool to a running agent
# ==============================================================================

def add_tool(gate, name: str, description: str, handler: Callable,
             keywords: List[str] = None):
    """Add a tool to a running agent. No restart needed."""
    gate.tools.register(
        name, description, handler, keywords or ["general"]
    )
    if hasattr(gate, '_plugin_manager'):
        gate._plugin_manager.add(name, description, handler, keywords)
    print(f"  ✓ Tool added: {name}")


def remove_tool(gate, name: str):
    """Remove a tool from a running agent."""
    if name in gate.tools.tools:
        del gate.tools.tools[name]
    if hasattr(gate, '_plugin_manager'):
        gate._plugin_manager.remove(name)
    print(f"  ✓ Tool removed: {name}")


def add_api_tool(gate, name: str, url: str, description: str = "",
                 keywords: List[str] = None, headers: dict = None,
                 response_path: str = ""):
    """
    Add a simple API tool — just provide the URL.
    
    Example:
        add_api_tool(gate,
            name="my_api",
            url="https://api.example.com/search?q={query}",
            description="Search my API",
            keywords=["my_search"],
            response_path="results[0].text"
        )
    """
    _headers = headers or {}

    def handler(query: str) -> str:
        final_url = url.replace('{query}', urllib.parse.quote(query))
        try:
            req = urllib.request.Request(final_url)
            req.add_header('User-Agent', 'QOR-AI-Agent/1.0')
            for k, v in _headers.items():
                req.add_header(k, v)
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
            if response_path:
                return _extract_path(data, response_path)
            return _format_dict(data) if isinstance(data, dict) \
                else json.dumps(data)[:800]
        except Exception as e:
            return f"API error: {e}"

    add_tool(gate, name, description or f"API: {name}", handler,
             keywords or [name])

"""
QOR — Qora Neuran AI Project
====================
Main entry point. All commands in one place.

SETUP:
    python -m qor setup                     # Create folders and sample data
    python -m qor tokenizer                  # Train tokenizer on your data

TRAINING:
    python -m qor train                      # Train the model
    python -m qor train --size medium        # Train a bigger model
    python -m qor train --resume             # Resume from checkpoint

EVALUATION:
    python -m qor eval                       # Evaluate the model
    python -m qor test                       # Run the Mind Test

GENERATION:
    python -m qor chat                       # Interactive chat
    python -m qor generate "Your prompt"     # One-off generation

CONTINUAL LEARNING:
    python -m qor learn                      # Learn from files in learn/ folder
    python -m qor watch                      # Auto-learn new files (live)

RUNTIME (continuous background learning):
    python -m qor run                        # Read loop + consolidation + chat
    python -m qor consolidate                # Manual batch consolidation

SERVING:
    python -m qor serve                      # Start API server
    python -m qor serve --fastapi            # Production API server
"""

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime, timedelta

# Fix Windows console encoding — prevents 'charmap' codec errors with Unicode characters
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


SIZE_CHOICES = ["small", "medium", "large", "qor3b", "qor3b_multimodal"]


# ==============================================================================
# TOPIC DETECTION — Keyword matching for user interest tracking
# ==============================================================================

# Known topic categories — keyword → canonical topic name
TOPIC_MAP = {
    # Crypto
    "bitcoin": ["bitcoin", "btc"],
    "ethereum": ["ethereum", "eth"],
    "crypto": ["crypto", "cryptocurrency", "altcoin", "defi", "nft"],
    "trading": ["trading", "trade", "position", "stop loss", "take profit", "dca"],
    "solana": ["solana", "sol"],
    # Finance
    "stocks": ["stock", "shares", "equity", "s&p", "nasdaq", "dow"],
    "forex": ["forex", "currency", "eur/usd", "exchange rate"],
    "gold": ["gold", "silver", "commodities"],
    # Tech
    "ai": ["ai", "artificial intelligence", "machine learning", "neural", "llm", "gpt"],
    "programming": ["python", "javascript", "coding", "programming", "code", "github"],
    # General
    "weather": ["weather", "temperature", "rain", "forecast"],
    "news": ["news", "headline", "breaking"],
    "science": ["science", "research", "study", "discovery", "arxiv"],
    "sports": ["sports", "football", "basketball", "soccer", "game score"],
}


def _detect_topics(text):
    """Detect topics/interests from user text.

    Returns:
        List of canonical topic names found in the text.
    """
    t = text.lower()
    topics = []
    for topic, keywords in TOPIC_MAP.items():
        if any(kw in t for kw in keywords):
            topics.append(topic)
    return topics


def _update_interests(profile, question, result, data_dir, graph=None, user_id=None):
    """Extract topics from question and update user interest profile."""
    interests = profile.setdefault("interests", {})
    today = datetime.now().strftime("%Y-%m-%d")

    # Detect topics from the question
    topics = _detect_topics(question)
    for topic in topics:
        topic_lower = topic.lower()
        if topic_lower not in interests:
            interests[topic_lower] = {"count": 0, "last": "", "score": 0.0}
        entry = interests[topic_lower]
        entry["count"] += 1
        entry["last"] = today
        # Score: frequency-weighted, capped at 1.0
        entry["score"] = min(1.0, entry["count"] / 50.0 + 0.1)

    # Decay old interests (haven't been mentioned in 30+ days)
    cutoff = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    for topic in list(interests.keys()):
        entry = interests[topic]
        if entry.get("last") and entry["last"] < cutoff:
            entry["score"] *= 0.9  # Gentle decay
            if entry["score"] < 0.05:
                del interests[topic]

    # Save to tree (primary) — no profile.json write
    if graph is not None and user_id:
        try:
            from .knowledge_tree import update_profile_field
            update_profile_field(graph, user_id, "interests", interests)
        except Exception:
            pass


def _build_system_prompt(user_name, profile, model=None):
    """Build system prompt with user identity, interests, cautions, and capabilities."""
    top_interests = sorted(
        profile.get("interests", {}).items(),
        key=lambda x: x[1].get("score", 0), reverse=True
    )[:5]
    interest_str = ", ".join(name for name, _ in top_interests) if top_interests else "unknown"
    cautions = profile.get("cautions", [])
    caution_str = ", ".join(cautions) if cautions else ""

    if user_name:
        prompt = (
            f"You are QOR (Qora Neuran AI), a friendly and helpful AI companion. "
            f"The user's name is {user_name}. Address them by name naturally — "
            f"be warm and personal but not excessive. "
            f"The user's main interests are: {interest_str}. "
            f"Tailor your responses to these interests when relevant — "
            f"e.g., if they're into crypto, relate examples to crypto when natural."
        )
    else:
        prompt = (
            "You are QOR (Qora Neuran AI), a friendly and helpful AI companion. "
            f"The user's main interests are: {interest_str}. "
            f"Tailor your responses to these interests when relevant."
        )

    # Multimodal capabilities — tell the model what it can perceive
    if model is not None:
        capabilities = []
        if getattr(model, 'vision_encoder', None) is not None:
            capabilities.append(
                "You CAN see and analyze images. When the user provides an image file path "
                "(e.g. /path/to/image.png), you will see the image contents. "
                "Describe what you see in detail — charts, text, objects, patterns, colors."
            )
        if getattr(model, 'audio_encoder', None) is not None:
            capabilities.append(
                "You CAN hear and analyze audio. When the user provides an audio file path "
                "(e.g. /path/to/audio.wav), you will hear the audio contents."
            )
        if capabilities:
            prompt += "\n" + " ".join(capabilities)

    if caution_str:
        prompt += (
            f"\nBe careful with these sensitive topics: {caution_str}. "
            f"Stay neutral and factual on them."
        )

    return prompt


def _load_profile(data_dir):
    """Load user profile from qor-data/profile.json."""
    path = os.path.join(data_dir, "profile.json")
    profile = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                profile = json.load(f)
        except Exception:
            pass
    # Ensure default fields exist
    profile.setdefault("interests", {})
    profile.setdefault("cautions", [])
    profile.setdefault("preferred_detail_level", "detailed")
    return profile


def _save_profile(data_dir, profile):
    """Save user profile to qor-data/profile.json."""
    path = os.path.join(data_dir, "profile.json")
    with open(path, 'w') as f:
        json.dump(profile, f, indent=2)


def _extract_name(text):
    """Extract user name from common patterns."""
    text = text.strip()
    patterns = [
        r"(?:my name is|i'm|i am|call me|it's|its|name's|names)\s+(\w+)",
        r"^(\w+)$",  # Just a single word = the name itself
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            name = m.group(1).strip().title()
            # Filter out common non-name words
            if name.lower() not in {"yes", "no", "ok", "okay", "sure", "hey",
                                     "hi", "hello", "quit", "skip", "none",
                                     "nah", "nope", "what", "why", "how"}:
                return name
    return None


# ==============================================================================
# TRADING CREDENTIAL MANAGEMENT — Encrypted API key storage
# ==============================================================================

def _parse_api_command(text):
    """Parse API key commands from chat input.

    Supported patterns (Binance — backward compatible):
      "set api key <KEY>"
      "set api secret <SECRET>"
      "set api key <KEY> secret <SECRET>"
      "remove api key" / "clear api key" / "delete api key"
      "show api status" / "api status"
      "set trading live" / "set trading demo"
      "confirm live trading"

    Multi-exchange patterns:
      "set <EXCHANGE> api key <KEY> secret <SECRET>"
      "set <EXCHANGE> api key <KEY> secret <SECRET> passphrase <PASS>"
      "set <EXCHANGE> api key <KEY>"
      "set <EXCHANGE> api secret <SECRET>"
      "set <EXCHANGE> passphrase <PASS>"
      "remove <EXCHANGE> api key"
      "<EXCHANGE> api status"
      "exchanges" / "list exchanges"

    Returns: {"action": str, ...} or None
    """
    t = text.strip().lower()
    raw = text.strip()

    # List all configured exchanges
    if t in ("exchanges", "list exchanges", "show exchanges"):
        return {"action": "list_exchanges"}

    # Status check (Binance default)
    if t in ("api status", "show api status", "api keys", "show api keys"):
        return {"action": "status"}

    # --- Multi-exchange commands (must be checked BEFORE generic set api) ---

    # Exchange-specific status: "<exchange> api status" or "<exchange> status"
    m = re.match(r"(\w+)\s+(?:api\s+)?status$", t)
    if m and m.group(1) not in ("api", "show", "trading", "futures"):
        return {"action": "exchange_status", "exchange": m.group(1)}

    # Remove exchange creds: "remove <exchange> api key"
    m = re.match(r"(?:remove|clear|delete)\s+(\w+)\s+api\s*keys?", t)
    if m:
        return {"action": "remove_exchange", "exchange": m.group(1)}

    # Set exchange key+secret+passphrase: "set <exchange> api key X secret Y passphrase Z"
    m = re.match(
        r"set\s+(\w+)\s+api\s+key\s+(\S+)\s+secret\s+(\S+)\s+passphrase\s+(\S+)",
        raw, re.IGNORECASE)
    if m and m.group(1).lower() not in ("api", "trading", "futures", "spot"):
        return {"action": "set_exchange", "exchange": m.group(1).lower(),
                "key": m.group(2), "secret": m.group(3), "passphrase": m.group(4)}

    # Set exchange key+secret: "set <exchange> api key X secret Y"
    m = re.match(
        r"set\s+(\w+)\s+api\s+key\s+(\S+)\s+secret\s+(\S+)",
        raw, re.IGNORECASE)
    if m and m.group(1).lower() not in ("api", "trading", "futures", "spot"):
        return {"action": "set_exchange", "exchange": m.group(1).lower(),
                "key": m.group(2), "secret": m.group(3)}

    # Set exchange key only: "set <exchange> api key X"
    m = re.match(r"set\s+(\w+)\s+api\s+key\s+(\S+)", raw, re.IGNORECASE)
    if m and m.group(1).lower() not in ("api", "trading", "futures", "spot"):
        return {"action": "set_exchange_key", "exchange": m.group(1).lower(),
                "key": m.group(2)}

    # Set exchange secret only: "set <exchange> api secret X"
    m = re.match(r"set\s+(\w+)\s+api\s+secret\s+(\S+)", raw, re.IGNORECASE)
    if m and m.group(1).lower() not in ("api", "trading", "futures", "spot"):
        return {"action": "set_exchange_secret", "exchange": m.group(1).lower(),
                "secret": m.group(2)}

    # Set exchange passphrase: "set <exchange> passphrase X"
    m = re.match(r"set\s+(\w+)\s+passphrase\s+(\S+)", raw, re.IGNORECASE)
    if m and m.group(1).lower() not in ("api", "trading", "futures", "spot"):
        return {"action": "set_exchange_passphrase", "exchange": m.group(1).lower(),
                "passphrase": m.group(2)}

    # Set exchange demo/live: "set <exchange> demo" / "set <exchange> live"
    m = re.match(r"set\s+(\w+)\s+(demo|live|testnet|production)", t)
    if m and m.group(1) not in ("api", "trading", "futures", "spot"):
        return {"action": "set_exchange_mode", "exchange": m.group(1),
                "testnet": m.group(2) in ("demo", "testnet")}

    # --- Binance default commands (backward compatible) ---

    # Remove credentials
    if any(t.startswith(p) for p in ("remove api key", "clear api key", "delete api key",
                                      "remove api keys", "clear api keys", "delete api keys")):
        return {"action": "remove"}

    # Set both key and secret: "set api key <KEY> secret <SECRET>"
    m = re.match(r"set\s+api\s+key\s+(\S+)\s+secret\s+(\S+)", raw, re.IGNORECASE)
    if m:
        return {"action": "set_both", "key": m.group(1), "secret": m.group(2)}

    # Set key only: "set api key <KEY>"
    m = re.match(r"set\s+api\s+key\s+(\S+)", raw, re.IGNORECASE)
    if m:
        return {"action": "set_key", "key": m.group(1)}

    # Set secret only: "set api secret <SECRET>"
    m = re.match(r"set\s+api\s+secret\s+(\S+)", raw, re.IGNORECASE)
    if m:
        return {"action": "set_secret", "secret": m.group(1)}

    # Start/stop trading (keep keys)
    if t in ("start trading", "trading start", "resume trading"):
        return {"action": "start"}
    if t in ("stop trading", "trading stop", "pause trading"):
        return {"action": "stop"}

    # Trading mode
    if t in ("set trading live", "set trading production"):
        return {"action": "set_live"}
    if t in ("set trading demo", "set trading testnet"):
        return {"action": "set_demo"}
    if t == "confirm live trading":
        return {"action": "confirm_live"}
    m = re.match(r"set\s+(?:trading|spot)\s+mode\s+(\w+)", t)
    if m:
        return {"action": "set_spot_mode", "mode": m.group(1).lower()}

    # Futures commands
    if t in ("start futures", "futures start"):
        return {"action": "start_futures"}
    if t in ("stop futures", "futures stop"):
        return {"action": "stop_futures"}
    m = re.match(r"set\s+futures\s+leverage\s+(\d+)", t)
    if m:
        return {"action": "set_futures_leverage", "leverage": int(m.group(1))}
    m = re.match(r"set\s+futures\s+mode\s+(\w+)", t)
    if m:
        return {"action": "set_futures_mode", "mode": m.group(1).lower()}

    # OAuth2 token commands: "set <exchange> token <TOKEN>"
    m = re.match(r"set\s+(\w+)\s+token\s+(\S+)", raw, re.IGNORECASE)
    if m and m.group(1).lower() not in ("api", "trading", "futures", "spot"):
        return {"action": "set_exchange_token", "exchange": m.group(1).lower(),
                "token": m.group(2)}

    # OAuth2 auth code exchange: "set <exchange> auth <CODE> redirect <URI>"
    m = re.match(r"set\s+(\w+)\s+auth\s+(\S+)\s+redirect\s+(\S+)", raw, re.IGNORECASE)
    if m and m.group(1).lower() not in ("api", "trading", "futures", "spot"):
        return {"action": "exchange_auth_code", "exchange": m.group(1).lower(),
                "code": m.group(2), "redirect_uri": m.group(3)}

    # OAuth2 auth code (default redirect): "set <exchange> auth <CODE>"
    m = re.match(r"set\s+(\w+)\s+auth\s+(\S+)", raw, re.IGNORECASE)
    if m and m.group(1).lower() not in ("api", "trading", "futures", "spot"):
        return {"action": "exchange_auth_code", "exchange": m.group(1).lower(),
                "code": m.group(2), "redirect_uri": ""}

    return None


def _load_trading_credentials(profile, crypto, session_id):
    """Decrypt trading credentials from profile if session matches.
    Returns {"api_key": str, "api_secret": str, "testnet": bool} or None.
    """
    creds = profile.get("trading_credentials")
    if not creds:
        return None
    if creds.get("session_id", "") != session_id:
        return None
    if "api_key" not in creds or "api_secret" not in creds:
        return None
    try:
        return {
            "api_key": crypto.decrypt_str(creds["api_key"]),
            "api_secret": crypto.decrypt_str(creds["api_secret"]),
            "testnet": creds.get("testnet", True),
        }
    except Exception:
        return None


def _clear_trading_credentials(data_dir, profile):
    """Remove stored trading credentials from profile.json."""
    if "trading_credentials" in profile:
        del profile["trading_credentials"]
        _save_profile(data_dir, profile)  # Credentials → profile.json only
        return True
    return False


def _mask_key(key):
    """Show first 4 and last 4 chars of a key, mask the rest."""
    if len(key) <= 8:
        return key[:2] + "..." + key[-2:]
    return key[:4] + "..." + key[-4:]


# ── Multi-exchange credential helpers ────────────────────────────────
# Stored in profile.json["exchange_credentials"][name] — encrypted.

def _save_exchange_creds(data_dir, profile, name, crypto,
                         api_key=None, api_secret=None, passphrase=None,
                         access_token=None, testnet=True, session_id=""):
    """Encrypt and save exchange credentials to profile.json."""
    all_ex = profile.setdefault("exchange_credentials", {})
    creds = all_ex.get(name, {})
    if api_key is not None:
        creds["api_key"] = crypto.encrypt_str(api_key)
    if api_secret is not None:
        creds["api_secret"] = crypto.encrypt_str(api_secret)
    if passphrase is not None:
        creds["passphrase"] = crypto.encrypt_str(passphrase)
    if access_token is not None:
        creds["access_token"] = crypto.encrypt_str(access_token)
    creds["testnet"] = testnet
    creds["session_id"] = session_id
    creds["set_at"] = datetime.now().isoformat()
    all_ex[name] = creds
    profile["exchange_credentials"] = all_ex
    _save_profile(data_dir, profile)


def _load_exchange_creds(profile, name, crypto):
    """Decrypt exchange credentials from profile.json.

    Returns {"api_key": str, "api_secret": str, "passphrase": str,
             "access_token": str, "testnet": bool} or None.
    """
    all_ex = profile.get("exchange_credentials", {})
    creds = all_ex.get(name)
    if not creds or "api_key" not in creds:
        return None
    try:
        result = {
            "api_key": crypto.decrypt_str(creds["api_key"]),
            "api_secret": crypto.decrypt_str(creds.get("api_secret", "")),
            "testnet": creds.get("testnet", True),
        }
        if "passphrase" in creds:
            result["passphrase"] = crypto.decrypt_str(creds["passphrase"])
        if "access_token" in creds:
            result["access_token"] = crypto.decrypt_str(creds["access_token"])
        return result
    except Exception:
        return None


def _clear_exchange_creds(data_dir, profile, name):
    """Remove credentials for one exchange from profile.json."""
    all_ex = profile.get("exchange_credentials", {})
    if name in all_ex:
        del all_ex[name]
        profile["exchange_credentials"] = all_ex
        _save_profile(data_dir, profile)
        return True
    return False


def _list_exchange_creds(profile):
    """List all configured exchanges (names only, no secrets)."""
    all_ex = profile.get("exchange_credentials", {})
    result = []
    for name, creds in all_ex.items():
        result.append({
            "name": name,
            "has_key": "api_key" in creds,
            "has_secret": "api_secret" in creds,
            "has_passphrase": "passphrase" in creds,
            "testnet": creds.get("testnet", True),
            "set_at": creds.get("set_at", "?"),
        })
    return result


def _restart_trading_engine(runtime, config, gate):
    """Stop existing trading engine and start a new one with current config."""
    if runtime._trading_engine:
        runtime._trading_engine.stop()
        runtime._trading_engine = None
    if config.trading.api_key and config.trading.api_secret:
        try:
            from qor.trading import TradingEngine
            tex = getattr(gate, '_tool_executor', None)
            runtime._trading_engine = TradingEngine(config, tool_executor=tex)
            runtime._trading_engine.start()
            mode = "DEMO" if config.trading.testnet else "PRODUCTION"
            return mode
        except Exception as e:
            return f"ERROR: {e}"
    return None


def _restart_futures_engine(runtime, config, gate):
    """Stop existing futures engine and start a new one with current config."""
    if runtime._futures_engine:
        runtime._futures_engine.stop()
        runtime._futures_engine = None
    if config.futures.api_key and config.futures.api_secret:
        try:
            from qor.futures import FuturesEngine
            tex = getattr(gate, '_tool_executor', None)
            runtime._futures_engine = FuturesEngine(config, tool_executor=tex)
            runtime._futures_engine.start()
            mode = "TESTNET" if config.futures.testnet else "PRODUCTION"
            lev = config.futures.leverage
            return f"{mode} {lev}x"
        except Exception as e:
            return f"ERROR: {e}"
    return None


def cmd_load_donor(args):
    """Load donor weights into QOR architecture."""
    import subprocess
    script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "scripts", "load_qor3b.py")
    cmd = [sys.executable, script]
    if args.model_dir:
        cmd += ["--model-dir", args.model_dir]
    if args.output:
        cmd += ["--output", args.output]
    if args.dtype:
        cmd += ["--dtype", args.dtype]
    subprocess.run(cmd)


def cmd_build_multimodal(args):
    """Build unified multimodal checkpoint (SmolLM3 + SigLIP + Whisper)."""
    import subprocess
    script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "scripts", "load_qor3b_multimodal.py")
    cmd = [sys.executable, script]
    if args.donor_checkpoint:
        cmd += ["--donor-checkpoint", args.donor_checkpoint]
    if args.model_dir:
        cmd += ["--model-dir", args.model_dir]
    if args.output:
        cmd += ["--output", args.output]
    if args.dtype:
        cmd += ["--dtype", args.dtype]
    subprocess.run(cmd)


def cmd_setup(args):
    """Create project folders and sample training data."""
    from qor.config import QORConfig

    config = QORConfig()
    config.resolve_data_paths()
    data_dir = config.runtime.data_dir

    # Training data directory (outside qor-data — it's a build input)
    os.makedirs("data", exist_ok=True)
    print(f"  Created data/")

    # qor-data/ — ALL runtime data lives here
    runtime_dirs = [
        config.runtime.historical_dir,                         # qor-data/historical
        config.train.checkpoint_dir,                           # qor-data/checkpoints
        config.continual.learn_dir,                            # qor-data/learn
        config.get_data_path("knowledge"),                     # qor-data/knowledge
        config.get_data_path("knowledge", "graph.rocksdb"),    # qor-data/knowledge/graph.rocksdb
        config.get_data_path("plugins"),                       # qor-data/plugins
        config.get_data_path("skills"),                        # qor-data/skills
        config.get_data_path("logs"),                          # qor-data/logs
        config.get_data_path("screenshots"),                   # qor-data/screenshots (video frames)
        config.get_data_path("documents"),                     # qor-data/documents
        config.get_data_path("trading"),                       # qor-data/trading
        config.get_data_path("futures"),                       # qor-data/futures
    ]
    for d in runtime_dirs:
        os.makedirs(d, exist_ok=True)
        print(f"  Created {d}/")

    # Create sample training data
    sample_path = os.path.join("data", "sample.txt")
    if not os.path.exists(sample_path):
        sample_text = """The sun rises in the east and sets in the west. Every morning brings new light to the world.

Cats are curious animals that love to explore. They have four legs, sharp claws, and excellent night vision. A cat's purr is one of the most soothing sounds in nature.

Dogs are loyal companions that have lived alongside humans for thousands of years. They come in many breeds, from tiny Chihuahuas to giant Great Danes.

Water covers about seventy percent of the Earth's surface. The oceans are home to millions of species, from tiny plankton to massive blue whales.

The human brain contains approximately eighty-six billion neurons. Each neuron can form thousands of connections with other neurons, creating an incredibly complex network.

Mathematics is the language of the universe. Two plus two equals four. The square root of nine is three. Pi is approximately three point one four one five nine.

Music has the power to move emotions. A melody can make us feel joy, sadness, excitement, or peace. Rhythm connects us to something primal and universal.

Trees produce oxygen through photosynthesis. They absorb carbon dioxide and release oxygen, making life possible for animals and humans. A single large tree can provide oxygen for up to four people.

The speed of light is approximately three hundred thousand kilometers per second. Nothing in the universe can travel faster than light. Einstein's theory of relativity depends on this fundamental constant.

Cooking is both an art and a science. Heat transforms ingredients through chemical reactions. The Maillard reaction creates the brown, flavorful crust on bread and meat.
"""
        with open(sample_path, 'w') as f:
            f.write(sample_text * 20)  # Repeat for enough training data
        print(f"  Created {sample_path} (sample training data)")
    else:
        print(f"  {sample_path} already exists")

    print(f"\n  Project ready!")
    print(f"  Runtime data directory: {os.path.abspath(data_dir)}/")
    print(f"\n  Next steps:")
    print(f"    1. Add your own .txt files to data/")
    print(f"    2. python -m qor tokenizer")
    print(f"    3. python -m qor train")


def cmd_tokenizer(args):
    """Train a BPE tokenizer on the data."""
    from qor.config import QORConfig, PRESETS
    from qor.tokenizer import QORTokenizer

    config = PRESETS.get(args.size, PRESETS["small"])()
    tokenizer = QORTokenizer()
    tokenizer.train(
        data_dir=config.train.data_dir,
        vocab_size=config.tokenizer.vocab_size,
        save_path=config.tokenizer.save_path,
    )


def cmd_train(args):
    """Train the QOR model."""
    from qor.config import QORConfig, PRESETS
    from qor.train import Trainer

    config = PRESETS.get(args.size, PRESETS["small"])()

    if args.steps:
        config.train.max_steps = args.steps
    if args.device:
        config.train.device = args.device
    if args.batch_size:
        config.train.batch_size = args.batch_size
    if getattr(args, 'distributed', False):
        config.train.distributed = True
    if getattr(args, 'gradient_checkpointing', False):
        config.train.gradient_checkpointing = True
    if getattr(args, 'compile', False):
        config.model.compile = True

    # Save config for reproducibility
    os.makedirs(config.train.checkpoint_dir, exist_ok=True)
    config.save(os.path.join(config.train.checkpoint_dir, "config.json"))

    trainer = Trainer(config)
    resume = os.path.join(config.train.checkpoint_dir, "final_model.pt") if args.resume else None
    trainer.train(resume_from=resume)


def cmd_eval(args):
    """Evaluate the model."""
    from qor.config import QORConfig, PRESETS
    from qor.evaluate import Evaluator

    config = PRESETS.get(args.size, PRESETS["small"])()
    evaluator = Evaluator(config)

    ckpt = args.checkpoint or os.path.join(config.train.checkpoint_dir, "best_model.pt")
    eval_text = args.eval_file

    evaluator.full_report(ckpt, eval_text)


def cmd_test(args):
    """Run the Mind Test (continual learning test)."""
    from qor.config import QORConfig, PRESETS
    from qor.evaluate import Evaluator
    from qor.tokenizer import QORTokenizer

    config = PRESETS.get(args.size, PRESETS["small"])()

    # Need a tokenizer for the test
    tokenizer = QORTokenizer()
    tok_path = config.tokenizer.save_path
    if os.path.exists(tok_path):
        tokenizer.load(tok_path)
        config.model.vocab_size = tokenizer.vocab_size
    else:
        print("Train a tokenizer first: python -m qor tokenizer")
        print("Or run setup first: python -m qor setup && python -m qor tokenizer")
        return

    if args.device:
        config.train.device = args.device

    evaluator = Evaluator(config)
    evaluator.continual_learning_test(tokenizer)


def cmd_chat(args):
    """Interactive chat with the model."""
    from qor.config import QORConfig, PRESETS
    from qor.serve import QORServer

    config = PRESETS.get(args.size, PRESETS["small"])()
    if args.device:
        config.train.device = args.device
    config.resolve_data_paths()

    ckpt = args.checkpoint or os.path.join(config.train.checkpoint_dir, "best_model.pt")

    server = QORServer(config)
    server.load(checkpoint_path=ckpt)

    print(f"\n{'='*60}")
    print(f"  QOR Chat — The Qore Mind")
    print(f"  Type 'quit' to exit, 'reset' to clear memory")
    print(f"{'='*60}\n")

    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not prompt:
            continue
        if prompt.lower() == 'quit':
            break
        if prompt.lower() == 'reset':
            server.model.reset_fast_weights()
            print("  [Memory reset]\n")
            continue

        result = server.generate(prompt, temperature=args.temperature or 0.8)
        print(f"QOR: {result['generated_text']}")
        print(f"  [{result['tokens_generated']} tokens, {result['time_seconds']}s]\n")


def cmd_generate(args):
    """Generate text from a prompt."""
    from qor.config import QORConfig, PRESETS
    from qor.serve import QORServer

    config = PRESETS.get(args.size, PRESETS["small"])()
    if args.device:
        config.train.device = args.device

    ckpt = args.checkpoint or os.path.join(config.train.checkpoint_dir, "best_model.pt")

    server = QORServer(config)
    server.load(checkpoint_path=ckpt)

    result = server.generate(args.prompt, temperature=args.temperature or 0.8,
                              max_tokens=args.max_tokens or 200)
    print(result['output'])


def _learn_files_to_tree(config, folder):
    """Read .txt files from folder → save as knowledge nodes in tree → delete files."""
    from qor.graph import QORGraph, GraphConfig
    from qor.knowledge_tree import get_or_create_system_id
    from qor.confidence import _extract_entities_and_edges
    import hashlib as _hl

    graph_config = config.graph if hasattr(config, 'graph') else GraphConfig()
    graph = QORGraph(graph_config)
    graph.open()
    user_id = f"user:{get_or_create_system_id(config.runtime.data_dir)}"
    learned = 0
    for fname in os.listdir(folder):
        if not fname.endswith('.txt'):
            continue
        fpath = os.path.join(folder, fname)
        if not os.path.isfile(fpath):
            continue
        with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read().strip()
        if not content:
            os.remove(fpath)
            continue
        know_hash = _hl.sha256(
            (fname + ":" + content[:200]).encode()).hexdigest()[:8]
        know_id = f"know:{know_hash}"
        graph.add_node(know_id, node_type="knowledge", properties={
            "content": content[:2000], "source": f"learn:{fname}",
            "question": fname.replace('.txt', ''),
            "timestamp": datetime.now().isoformat(),
        })
        graph.add_edge(user_id, "learned", know_id,
                       confidence=0.9, source="learn_file")
        for subj, pred, obj in _extract_entities_and_edges(content)[:15]:
            graph.add_edge(subj, pred, obj, confidence=0.7,
                           source="learn_file")
        os.remove(fpath)
        learned += 1
        print(f"  + {fname} → tree ({len(content)} chars)")
    graph.close()
    return learned


def cmd_learn(args):
    """Learn from .txt files — saves to knowledge tree. No model weight changes."""
    from qor.config import QORConfig, PRESETS

    config = PRESETS.get(args.size, PRESETS["small"])()
    config.resolve_data_paths()
    folder = args.folder if args.folder != "learn" else config.continual.learn_dir

    if not os.path.isdir(folder):
        print(f"  Folder not found: {folder}")
        return
    count = _learn_files_to_tree(config, folder)
    print(f"  Done: {count} files learned into tree")


def cmd_watch(args):
    """Watch folder for new .txt files — saves to tree on arrival."""
    from qor.config import QORConfig, PRESETS
    import time as _time

    config = PRESETS.get(args.size, PRESETS["small"])()
    config.resolve_data_paths()
    folder = args.folder if args.folder != "learn" else config.continual.learn_dir

    if not os.path.isdir(folder):
        print(f"  Folder not found: {folder}")
        return
    interval = args.interval or 10
    print(f"  Watching {folder} every {interval}s — drop .txt files to learn")
    print(f"  Press Ctrl+C to stop\n")
    try:
        while True:
            count = _learn_files_to_tree(config, folder)
            if count > 0:
                print(f"  [{datetime.now().strftime('%H:%M:%S')}] Learned {count} files")
            _time.sleep(interval)
    except KeyboardInterrupt:
        print("\n  Stopped.")


def cmd_run(args):
    """Start the full QOR runtime: background read loop + consolidation + interactive chat.

    Wires ALL knowledge systems: Model, ConfidenceGate, Knowledge Graph,
    Knowledge Base, RAG, Plugins, ToolExecutor — following create_agent() pattern.
    """
    from qor.config import QORConfig, PRESETS
    from qor.continual import ContinualLearner
    from qor.runtime import QORRuntime
    from qor.confidence import ConfidenceGate
    from qor.tools import ToolExecutor
    from qor.graph import QORGraph, GraphConfig
    from qor.knowledge import KnowledgeBase
    from qor.rag import QORRag
    from qor.plugins import PluginManager
    # setup.py no longer pollutes memory with psychology — data comes from read loop

    config = PRESETS.get(args.size, PRESETS["small"])()
    if args.device:
        config.train.device = args.device

    # Resolve all runtime paths under qor-data/
    config.resolve_data_paths()
    data_dir = config.runtime.data_dir

    ckpt = args.checkpoint or os.path.join(config.train.checkpoint_dir, "best_model.pt")

    # Ensure qor-data/ directory tree exists
    for d in [config.runtime.historical_dir,
              config.train.checkpoint_dir, config.continual.learn_dir,
              config.get_data_path("knowledge"), config.get_data_path("plugins"),
              config.get_data_path("logs"), config.get_data_path("screenshots"),
              config.get_data_path("trading"),
              config.get_data_path("futures")]:
        os.makedirs(d, exist_ok=True)
    print(f"  Data directory: {os.path.abspath(data_dir)}")

    # ------------------------------------------------------------------
    # 1. Load model via ContinualLearner (mmap for 3B)
    # ------------------------------------------------------------------
    learner = ContinualLearner(config)
    learner.load(ckpt)
    print(f"  Model loaded from {ckpt}")

    # Report multimodal capabilities
    if getattr(learner.model, 'vision_encoder', None) is not None:
        print(f"  Vision: ENABLED (SigLIP encoder loaded)")
    else:
        print(f"  Vision: disabled (use --size qor3b_multimodal for image support)")
    if getattr(learner.model, 'audio_encoder', None) is not None:
        print(f"  Audio:  ENABLED (Whisper encoder loaded)")

    # ------------------------------------------------------------------
    # 2. ConfidenceGate (the brain)
    # ------------------------------------------------------------------
    memory_path = config.get_data_path("memory.parquet")
    gate = ConfidenceGate(learner.model, learner.tokenizer, config)
    gate.memory.path = memory_path
    print(f"  Confidence Gate initialized (memory: {memory_path})")

    # ------------------------------------------------------------------
    # 3. Knowledge Graph (RocksDB)
    # ------------------------------------------------------------------
    graph = None
    try:
        graph_config = config.graph if hasattr(config, 'graph') else GraphConfig()
        graph = QORGraph(graph_config)
        graph.open()
        gate.set_graph(graph)
        gs = graph.stats()
        print(f"  Graph connected: {gs['node_count']} nodes, {gs['edge_count']} edges ({gs['backend']})")
    except Exception as e:
        print(f"  Graph: skipped ({e})")
        graph = None

    # ------------------------------------------------------------------
    # 4. Knowledge Base (psychology + documents)
    # ------------------------------------------------------------------
    knowledge_dir = config.get_data_path("knowledge")
    kb = None
    try:
        kb = KnowledgeBase(knowledge_dir)
        kb.load()
        print(f"  Knowledge Base: {len(kb.nodes)} nodes loaded")
    except Exception as e:
        print(f"  Knowledge Base: skipped ({e})")
        kb = None

    # ------------------------------------------------------------------
    # 5. RAG (document retrieval)
    # ------------------------------------------------------------------
    rag = None
    try:
        rag = QORRag()
        if os.path.isdir(knowledge_dir):
            rag.add_folder(knowledge_dir)
        gate.set_rag(rag)
        chunk_count = len(rag.store.chunks) if hasattr(rag, 'store') else 0
        print(f"  RAG connected: {chunk_count} chunks")
    except Exception as e:
        print(f"  RAG: skipped ({e})")
        rag = None

    # ------------------------------------------------------------------
    # 6. Plugins + Tools
    # ------------------------------------------------------------------
    plugin_mgr = None
    tool_executor = None
    try:
        plugin_mgr = PluginManager(
            plugins_dir=config.get_data_path("plugins"),
            config_path=config.get_data_path("tools_config"),
        )
        plugin_mgr.load_all(include_builtins=True)
        plugin_mgr.register_with_gate(gate)
        print(f"  Plugins: {len(plugin_mgr.tools)} tools loaded")
    except Exception as e:
        print(f"  Plugins: skipped ({e})")

    # custom_tools.py is a USER TEMPLATE file — not called at startup.
    # All real tools are loaded via PluginManager (builtins + plugins + config).
    # custom_tools.py was overwriting real working tools (weather, news_search)
    # with placeholders. Users who add custom tools should use plugins/ folder
    # or tools_config.json instead.

    try:
        tool_executor = ToolExecutor()
        gate._tool_executor = tool_executor
    except Exception as e:
        print(f"  ToolExecutor: skipped ({e})")

    # Store references on gate for status access
    gate._knowledge_base = kb
    gate._graph = graph
    if plugin_mgr is not None:
        gate._plugin_manager = plugin_mgr

    # ------------------------------------------------------------------
    # 6b. Skills (SKILL.md text files)
    # ------------------------------------------------------------------
    skill_loader = None
    try:
        from qor.skills import SkillLoader
        skills_dir = config.get_data_path("skills")
        os.makedirs(skills_dir, exist_ok=True)
        skill_loader = SkillLoader(skills_dir)
        skill_loader.load_all()
        gate._skill_loader = skill_loader
        print(f"  Skills: {len(skill_loader.skills)} loaded from {skills_dir}")
    except Exception as e:
        print(f"  Skills: skipped ({e})")

    # Wire model + skills into browse agent for model-based decisions
    try:
        from qor.browser import set_browse_model
        set_browse_model(learner.model, learner.tokenizer, skill_loader)
    except Exception:
        pass  # Browser not available — browse agent uses heuristic fallback

    # ------------------------------------------------------------------
    # 6c. NGRE Brain (4-layer pipeline: Mamba → Graph → Search → Reasoning)
    # ------------------------------------------------------------------
    ngre_brain = None
    try:
        from qor.ngre import create_ngre_brain
        ckpt_dir = config.train.checkpoint_dir
        ngre_brain = create_ngre_brain(
            checkpoint_dir=ckpt_dir,
            config=config.ngre if hasattr(config, 'ngre') else None,
        )
        if graph is not None:
            ngre_brain.set_graph(graph)
        gate.set_ngre_brain(ngre_brain)
        total_p = sum(p.numel() for p in ngre_brain.parameters())
        train_p = sum(p.numel() for p in ngre_brain.parameters() if p.requires_grad)
        print(f"  NGRE Brain: {total_p:,} params ({train_p:,} trainable)")
    except Exception as e:
        print(f"  NGRE Brain: skipped ({e})")

    # ------------------------------------------------------------------
    # 7. CacheStore + ChatStore
    # ------------------------------------------------------------------
    from qor.cache import CacheStore
    from qor.chat import ChatStore
    from qor.crypto import QORCrypto

    # Initialize crypto for chat encryption
    crypto = None
    try:
        crypto = QORCrypto(key_path=config.runtime.encryption_key_path)
        print(f"  Crypto: AES encryption ready ({config.runtime.encryption_key_path})")
    except Exception as e:
        print(f"  Crypto: skipped ({e})")

    cache_store = None
    chat_store = None
    try:
        cache_path = config.get_data_path("cache.parquet")
        cache_store = CacheStore(path=cache_path,
                                  secret=config.runtime.integrity_secret)
        gate.set_cache(cache_store)
        print(f"  Cache Store: {cache_store.count()} entries ({cache_path})")
    except Exception as e:
        print(f"  Cache Store: skipped ({e})")

    try:
        chat_path = config.get_data_path("chat.parquet")
        chat_store = ChatStore(path=chat_path,
                                secret=config.runtime.integrity_secret,
                                crypto=crypto)
        print(f"  Chat Store: {chat_store.count()} messages, "
              f"{chat_store.session_count()} sessions ({chat_path})")
    except Exception as e:
        print(f"  Chat Store: skipped ({e})")

    # ------------------------------------------------------------------
    # 8. User profile + decrypt stored trading credentials
    # ------------------------------------------------------------------
    # System ID = REAL identity (machine-level, generated once, permanent)
    # Name = display label (user can change it, doesn't change identity)
    from .knowledge_tree import (
        get_profile_from_tree, save_profile_to_tree, update_profile_field,
        get_or_create_system_id,
    )
    system_id = get_or_create_system_id(data_dir)
    user_id = f"user:{system_id}"
    print(f"  System ID: {system_id}")

    # Tree first — profile.json is bootstrap only (first run or tree empty)
    profile = get_profile_from_tree(graph, user_id)
    if profile is None:
        # Bootstrap from profile.json (first run or migration)
        profile = _load_profile(data_dir)
        if graph is not None:
            save_profile_to_tree(graph, user_id, profile)
            print(f"  Profile: bootstrapped into tree ({user_id})")
    user_name = profile.get("user_name", "")
    session_id = user_name.lower() if user_name else "cli_default"

    # Auto-load encrypted trading credentials from profile
    if crypto is not None:
        stored_creds = _load_trading_credentials(profile, crypto, session_id)
        if stored_creds:
            config.trading.api_key = stored_creds["api_key"]
            config.trading.api_secret = stored_creds["api_secret"]
            config.trading.testnet = stored_creds["testnet"]
            config.trading.enabled = True
            # Same keys work for futures (Binance uses one key pair)
            config.futures.api_key = stored_creds["api_key"]
            config.futures.api_secret = stored_creds["api_secret"]
            config.futures.testnet = stored_creds["testnet"]
            print(f"  Trading credentials: loaded from profile (encrypted)")

        # Load additional exchange credentials into config.exchanges
        ex_loaded = 0
        for ex_info in _list_exchange_creds(profile):
            ex_name = ex_info["name"]
            loaded = _load_exchange_creds(profile, ex_name, crypto)
            if loaded:
                from qor.config import ExchangeKeys
                # Check if already in config.exchanges
                existing = config.get_exchange(ex_name) if hasattr(config, 'get_exchange') else None
                if existing:
                    existing.api_key = loaded["api_key"]
                    existing.api_secret = loaded["api_secret"]
                    if "passphrase" in loaded:
                        existing.passphrase = loaded["passphrase"]
                    if "access_token" in loaded:
                        existing.access_token = loaded["access_token"]
                    existing.testnet = loaded["testnet"]
                    existing.enabled = True
                else:
                    config.exchanges.append(ExchangeKeys(
                        name=ex_name, api_key=loaded["api_key"],
                        api_secret=loaded["api_secret"],
                        passphrase=loaded.get("passphrase", ""),
                        access_token=loaded.get("access_token", ""),
                        testnet=loaded["testnet"], enabled=True,
                    ))
                ex_loaded += 1
        if ex_loaded:
            print(f"  Exchange credentials: loaded {ex_loaded} from profile (encrypted)")

    # Load saved trading symbols from profile (all exchanges)
    saved_syms = profile.get("trading_symbols")
    if saved_syms and isinstance(saved_syms, dict):
        spot_syms = saved_syms.get("spot")
        if spot_syms and isinstance(spot_syms, list):
            config.trading.symbols = spot_syms
            print(f"  Trading symbols (spot): {', '.join(spot_syms)}")
        futures_syms = saved_syms.get("futures")
        if futures_syms and isinstance(futures_syms, list) and hasattr(config, 'futures'):
            config.futures.symbols = futures_syms
            print(f"  Trading symbols (futures): {', '.join(futures_syms)}")
        # Load symbols for other exchanges (upstox, alpaca, oanda, etc.)
        for ex_name, ex_syms in saved_syms.items():
            if ex_name in ("spot", "futures", "updated_at"):
                continue
            if not isinstance(ex_syms, list) or not ex_syms:
                continue
            for ex in getattr(config, 'exchanges', []):
                if ex.name == ex_name:
                    ex.symbols = ex_syms
                    ex.enabled = True
                    print(f"  Trading symbols ({ex_name}): {', '.join(ex_syms)}")
                    break

    # ------------------------------------------------------------------
    # 9. Start runtime with ALL systems
    # ------------------------------------------------------------------
    runtime = QORRuntime(config)
    runtime.start(learner, gate=gate, graph=graph, rag=rag,
                  cache_store=cache_store, chat_store=chat_store,
                  tool_executor=tool_executor)

    # ------------------------------------------------------------------
    # 10. Start background API server for UI
    # ------------------------------------------------------------------
    try:
        from qor.serve import start_api_thread
        api_thread = start_api_thread(
            config, runtime, learner, gate, graph=graph, rag=rag,
            kb=kb, cache_store=cache_store, chat_store=chat_store,
            plugin_mgr=plugin_mgr, skill_loader=skill_loader,
            tool_executor=tool_executor, crypto=crypto,
            profile=profile, profile_path=os.path.join(data_dir, "profile.json"),
        )
        if api_thread:
            print(f"  API server: http://localhost:{config.serve.port} (background)")
    except Exception as e:
        print(f"  API server: skipped ({e})")

    # ------------------------------------------------------------------
    # 11. Personalized greeting
    # ------------------------------------------------------------------

    # Set system prompt with user name + interests + cautions
    gate.system_prompt = _build_system_prompt(user_name, profile, model=learner.model)

    print(f"\n{'='*60}")
    print(f"  QOR Runtime — Direct DB Storage (No Batching)")
    print(f"  Background read loop: every {config.runtime.read_interval}s")
    print(f"  Cleanup: every {config.runtime.cleanup_every_hours}h")
    print(f"  Sources: {len(config.runtime.read_sources)}")
    print(f"{'='*60}")
    print(f"  Commands:")
    print(f"    quit         — Exit")
    print(f"    status       — Show runtime stats (all systems)")
    print(f"    consolidate  — Force cleanup now")
    print(f"    memory       — Show memory stats")
    print(f"    cache        — Show cache stats")
    print(f"    history      — Show last 10 conversation turns")
    print(f"    verify       — Run hash chain verification")
    print(f"    tools        — List available tools")
    print(f"    skills       — List available skills")
    print(f"    interests    — Show tracked user interests")
    print(f"    name         — Show or change your name")
    print(f"    reset        — Clear fast memory")
    print(f"    verbose      — Toggle reasoning display")
    print(f"    trading      — Show trading engine status")
    print(f"    positions    — Show open positions")
    print(f"    trades       — Show recent closed trades")
    print(f"    api status   — Show API key status")
    print(f"    set api key <KEY> [secret <SECRET>] — Store API keys (encrypted)")
    print(f"    set trading mode X — scalp / stable / secure")
    print(f"    start trading — Start trading engine")
    print(f"    stop trading  — Stop trading engine (keys kept)")
    print(f"    remove api key — Clear stored API keys")
    print(f"    futures           — Show futures engine status")
    print(f"    futures positions — Show open futures positions")
    print(f"    futures trades    — Show recent closed futures trades")
    print(f"    start futures     — Start futures engine")
    print(f"    stop futures      — Stop futures engine")
    print(f"    set futures leverage N — Set leverage (1-10)")
    print(f"    set futures mode X    — scalp / stable / secure")
    print(f"{'='*60}")

    # --- Personalized greeting ---
    if user_name:
        print(f"\n  Hi {user_name}! Welcome back. I'm QOR, your AI companion.")
        print(f"  Ask me anything, or type a command.\n")
    else:
        print(f"\n  Hey buddy! I am QOR, your AI companion.")
        print(f"  What should I call you?\n")

        # Onboarding: ask for name (non-blocking — user can skip)
        while True:
            try:
                reply = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not reply:
                continue
            if reply.lower() in ("quit", "exit"):
                break
            if reply.lower() in ("skip", "later", "no", "nah", "nope"):
                print("  No worries! You can tell me your name anytime.\n")
                break
            name = _extract_name(reply)
            if name:
                user_name = name
                session_id = user_name.lower()
                profile["user_name"] = user_name
                save_profile_to_tree(graph, user_id, profile)  # user_id = system_id
                gate.system_prompt = _build_system_prompt(user_name, profile, model=learner.model)
                print(f"\n  Nice to meet you, {user_name}! Let's get started.\n")
                break
            else:
                print("  I didn't catch that. What's your name? (or type 'skip')\n")

    # ------------------------------------------------------------------
    # Interactive chat with full ConfidenceGate
    # ------------------------------------------------------------------
    verbose = True
    turn_count = 0
    _pending_live_confirm = False
    _last_result = None  # Track last answer for feedback detection

    # Set user_id on gate for knowledge tree — uses system_id (permanent)
    gate._user_id = user_id  # "user:sys-xxxxxxxxxxxx"

    # Initialize query pool for parallel chat processing
    query_pool = None
    query_pool_output = None
    if config.runtime.query_pool_enabled:
        try:
            from qor.query_pool import QueryPool, OutputManager
            query_pool_output = OutputManager()
            query_pool = QueryPool(
                gate, chat_store, session_id, query_pool_output,
                max_workers=config.runtime.query_pool_workers,
                verbose=verbose,
            )
            print(f"  Query pool: {config.runtime.query_pool_workers} workers")
        except Exception as e:
            print(f"  Query pool: disabled ({e})")
            query_pool = None

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not question:
            continue
        if question.lower() == "quit":
            # Wait for any in-flight queries before exiting
            if query_pool is not None:
                active = query_pool.active_count
                if active > 0:
                    print(f"  Waiting for {active} in-flight queries...")
                query_pool.wait_all()
                query_pool.shutdown()
                query_pool = None
            break
        if question.lower() == "status":
            s = runtime.status()
            print(f"  Memory entries:       {s.get('memory_entries', 'N/A')}")
            print(f"  Cache entries:        {s.get('cache_entries', 'N/A')}")
            print(f"  Chat messages:        {s.get('chat_messages', 'N/A')}")
            print(f"  Chat sessions:        {s.get('chat_sessions', 'N/A')}")
            print(f"  Last cleanup:         {s.get('last_cleanup') or 'never'}")
            print(f"  Cleanup count:        {s.get('cleanup_count', 0)}")
            print(f"  Read stats:           {s.get('read_stats', {})}")
            print(f"  Graph nodes:          {s.get('graph_nodes', 'N/A')}")
            print(f"  Graph edges:          {s.get('graph_edges', 'N/A')}")
            print(f"  RAG chunks:           {s.get('rag_chunks', 'N/A')}")
            print(f"  Historical entries:   {s.get('historical_entries', 'N/A')}")
            print(f"  Checkpoint snapshots: {s.get('checkpoint_snapshots', 'N/A')}")
            # Trading status
            ts = s.get("trading")
            if ts:
                stats = ts.get("stats", {})
                print(f"  Trading:              ACTIVE ({ts.get('mode', '?')}) — "
                      f"{ts.get('open_positions', 0)} open, "
                      f"{stats.get('total_trades', 0)} total, "
                      f"win {stats.get('win_rate', 0):.0f}%, "
                      f"P&L ${stats.get('total_pnl_usdt', 0):+,.2f}")
            else:
                print(f"  Trading:              disabled")
            # Futures status
            fs = s.get("futures")
            if fs:
                fstats = fs.get("stats", {})
                print(f"  Futures:              ACTIVE ({fs.get('mode', '?')}) "
                      f"{fs.get('leverage', '?')}x — "
                      f"{fs.get('open_positions', 0)} open, "
                      f"{fstats.get('total_trades', 0)} total, "
                      f"win {fstats.get('win_rate', 0):.0f}%, "
                      f"P&L ${fstats.get('total_pnl_usdt', 0):+,.2f}")
            else:
                print(f"  Futures:              disabled")
            # CORTEX brain status
            cx = s.get("cortex")
            if cx:
                trained = cx.get("trained", False)
                instances = cx.get("active_instances", [])
                candles = cx.get("history_candles", {})
                buf = cx.get("train_buffer", 0)
                total_candles = sum(candles.values()) if candles else 0
                if trained:
                    print(f"  CORTEX:               TRAINED — "
                          f"{len(instances)} symbols tracked, "
                          f"{total_candles} candles buffered, "
                          f"{buf} pending training samples")
                else:
                    print(f"  CORTEX:               NOT TRAINED — "
                          f"{buf} pending samples (needs 10 to train)")
            else:
                print(f"  CORTEX:               not loaded")
            # Knowledge tree stats
            kt = s.get("knowledge_tree")
            if kt:
                print(f"  Knowledge tree:       "
                      f"{kt.get('corrections', 0)} corrections, "
                      f"{kt.get('lessons', 0)} lessons, "
                      f"{kt.get('blocked_facts', 0)} blocked, "
                      f"{kt.get('preferences', 0)} prefs")
            # NGRE Brain status
            if gate._ngre_brain is not None:
                rs = gate.get_routing_stats()
                total_r = rs.get("total", 0)
                if total_r > 0:
                    print(f"  NGRE routing:         "
                          f"Template {rs.get('TEMPLATE', 0)} "
                          f"({rs.get('TEMPLATE_pct', 0):.0f}%), "
                          f"Fast {rs.get('LLM_FAST', 0)}, "
                          f"Think {rs.get('LLM_THINK', 0)}, "
                          f"Cloud {rs.get('CLOUD', 0)}")
                else:
                    print(f"  NGRE routing:         active (no queries yet)")
            else:
                print(f"  NGRE Brain:           not loaded")
            # Browser status
            try:
                from qor.browser import get_engine
                bs = get_engine(config.get_data_path("screenshots")).status()
                state = "running" if bs["running"] else ("available" if bs["available"] else "not installed")
                idle_str = f" (idle {bs['idle_seconds']}s)" if bs["running"] else ""
                print(f"  Browser:              {state}{idle_str}")
            except Exception:
                print(f"  Browser:              not available")
            print()
            continue
        if question.lower() == "consolidate":
            print("  Running cleanup...")
            result = runtime.cleanup_now()
            print(f"  Status:           {result.get('status')}")
            print(f"  Memory removed:   {result.get('memory_removed', 0)}")
            print(f"  Graph removed:    {result.get('graph_removed', 0)}")
            print(f"  Timestamp:        {result.get('timestamp', '')}")
            print()
            continue
        if question.lower() == "memory":
            gate.memory.stats()
            print()
            continue
        if question.lower() == "tools":
            for tool in gate.tools.list_tools():
                print(f"  {tool['name']}: {tool['description']}")
            print()
            continue
        if question.lower() == "skills":
            if skill_loader and skill_loader.skills:
                for name, skill in sorted(skill_loader.skills.items()):
                    print(f"  {name}: {skill.description}")
                    print(f"    keywords: {', '.join(skill.keywords)}")
            else:
                print("  No skills loaded")
            print()
            continue
        if question.lower() == "reset":
            learner.model.reset_fast_weights()
            print("  [Fast memory reset]\n")
            continue
        if question.lower() == "cache":
            if cache_store is not None:
                cache_store.stats()
            else:
                print("  Cache store not available")
            print()
            continue
        if question.lower() == "history":
            if chat_store is not None:
                hist_session = user_name.lower() if user_name else "cli_default"
                msgs = chat_store.get_history(hist_session, last_n=20)
                if msgs:
                    for m in msgs:
                        role_tag = "You" if m.role == "user" else "QOR"
                        print(f"  [{role_tag}] {m.content[:120]}")
                else:
                    print("  No conversation history yet")
            else:
                print("  Chat store not available")
            print()
            continue
        if question.lower() == "verify":
            print("  Hash chain verification:")
            if cache_store is not None:
                cv = cache_store.verify_chain()
                status = "VALID" if cv["valid"] else f"BROKEN at record {cv['broken_at']}"
                print(f"    Cache: {status} ({cv['checked']} records checked)")
            if chat_store is not None:
                sessions = chat_store.list_sessions()
                for sess in sessions:
                    sv = chat_store.verify_chain(sess["session_id"])
                    status = "VALID" if sv["valid"] else f"BROKEN at record {sv['broken_at']}"
                    print(f"    Chat [{sess['session_id']}]: {status} ({sv['checked']} records)")
                if not sessions:
                    print(f"    Chat: no sessions to verify")
            print()
            continue
        if question.lower() == "verbose":
            verbose = not verbose
            print(f"  [Verbose: {'ON' if verbose else 'OFF'}]\n")
            continue
        if question.lower() == "trading":
            te = runtime._trading_engine
            if te:
                s = te.status()
                stats = s.get("stats", {})
                print(f"  Trading:         ACTIVE ({s.get('mode', '?')})")
                print(f"  Symbols:         {', '.join(s.get('symbols', []))}")
                print(f"  Open positions:  {s.get('open_positions', 0)}")
                print(f"  Total trades:    {stats.get('total_trades', 0)}")
                print(f"  Win rate:        {stats.get('win_rate', 0):.1f}%")
                print(f"  Total P&L:       ${stats.get('total_pnl_usdt', 0):,.2f}")
                pf = stats.get('profit_factor', 0)
                pf_str = f"{pf:.2f}" if pf != float('inf') else "inf"
                print(f"  Profit factor:   {pf_str}")
                print(f"  Tick count:      {s.get('tick_count', 0)}")
                print(f"  Last tick:       {s.get('last_tick') or 'never'}")
                if s.get("last_error"):
                    print(f"  Last error:      {s['last_error']}")
                by_sym = stats.get("by_symbol", {})
                if by_sym:
                    print(f"  Per symbol:")
                    for sym, ss in by_sym.items():
                        print(f"    {sym}: {ss['trades']} trades, "
                              f"{ss['win_rate']:.0f}% win, ${ss['pnl']:+,.2f}")
            else:
                print("  Trading: DISABLED")
                print("  To enable: set trading.enabled=True in config + API keys")
            print()
            continue
        if question.lower() == "positions":
            te = runtime._trading_engine
            if te:
                opens = te.store.get_open_trades()
                if not opens:
                    print("  No open positions")
                else:
                    for t in opens:
                        pair = t["symbol"] + "USDT"
                        try:
                            current = te.client.get_price(pair)
                        except Exception:
                            current = 0
                        if current > 0 and t["entry_price"] > 0:
                            unrealized = (current - t["entry_price"]) * t["quantity"]
                            pct = ((current / t["entry_price"]) - 1) * 100
                            tp2 = t.get("take_profit2", 0)
                            tp2_str = f" TP2: ${tp2:,.2f}" if tp2 else ""
                            dca = t.get("dca_count", 0)
                            dca_str = f" DCA: {dca}x" if dca else ""
                            tp1_str = " [TP1 taken]" if t.get("tp1_hit") else ""
                            print(f"  {t['symbol']}: entry ${t['entry_price']:,.2f} "
                                  f"now ${current:,.2f} ({pct:+.2f}%) "
                                  f"SL: ${t['stop_loss']:,.2f} TP1: ${t['take_profit']:,.2f}"
                                  f"{tp2_str}{dca_str}{tp1_str} "
                                  f"P&L: ${unrealized:+,.2f}")
                        else:
                            print(f"  {t['symbol']}: entry ${t['entry_price']:,.2f} "
                                  f"SL: ${t['stop_loss']:,.2f} TP: ${t['take_profit']:,.2f}")
            else:
                print("  Trading engine not active")
            print()
            continue
        if question.lower() == "trades":
            te = runtime._trading_engine
            if te:
                closed = [t for t in te.store.trades.values()
                          if t["status"] != "open"]
                closed.sort(key=lambda t: t.get("exit_time", 0))
                closed = closed[-10:]
                if not closed:
                    print("  No closed trades yet")
                else:
                    for t in closed:
                        tag = "W" if t["pnl"] > 0 else "L"
                        print(f"  [{tag}] {t['symbol']}: {t['side']} "
                              f"${t['entry_price']:,.2f} -> ${t['exit_price']:,.2f} "
                              f"P&L: ${t['pnl']:+,.2f} ({t['pnl_pct']:+.2f}%) "
                              f"-- {t['exit_reason']}")
            else:
                print("  Trading engine not active")
            print()
            continue
        if question.lower() == "futures":
            fe = runtime._futures_engine
            if fe:
                s = fe.status()
                stats = s.get("stats", {})
                print(f"  Futures:         ACTIVE ({s.get('mode', '?')})")
                print(f"  Leverage:        {s.get('leverage', '?')}x {s.get('margin_type', '?')}")
                print(f"  Symbols:         {', '.join(s.get('symbols', []))}")
                print(f"  Open positions:  {s.get('open_positions', 0)}")
                print(f"  Total trades:    {stats.get('total_trades', 0)}")
                print(f"  Win rate:        {stats.get('win_rate', 0):.1f}%")
                print(f"  Total P&L:       ${stats.get('total_pnl_usdt', 0):,.2f}")
                print(f"  Funding paid:    ${s.get('total_funding_paid', 0):,.4f}")
                pf = stats.get('profit_factor', 0)
                pf_str = f"{pf:.2f}" if pf != float('inf') else "inf"
                print(f"  Profit factor:   {pf_str}")
                print(f"  Tick count:      {s.get('tick_count', 0)}")
                print(f"  Last tick:       {s.get('last_tick') or 'never'}")
                if s.get("last_error"):
                    print(f"  Last error:      {s['last_error']}")
                by_sym = stats.get("by_symbol", {})
                if by_sym:
                    print(f"  Per symbol:")
                    for sym, ss in by_sym.items():
                        print(f"    {sym}: {ss['trades']} trades, "
                              f"{ss['win_rate']:.0f}% win, ${ss['pnl']:+,.2f}")
            else:
                print("  Futures: DISABLED")
                print("  To enable: start futures (requires API keys)")
            print()
            continue
        if question.lower() == "futures positions":
            fe = runtime._futures_engine
            if fe:
                opens = fe.store.get_open_trades()
                if not opens:
                    print("  No open futures positions")
                else:
                    for t in opens:
                        pair = t["symbol"] + "USDT"
                        try:
                            current = fe.client.get_price(pair)
                        except Exception:
                            current = 0
                        direction = t.get("direction", "LONG")
                        lev = t.get("leverage", 1)
                        liq = t.get("liquidation_price", 0)
                        funding = t.get("funding_paid", 0)
                        if current > 0 and t["entry_price"] > 0:
                            if direction == "LONG":
                                unrealized = (current - t["entry_price"]) * t["quantity"]
                                pct = ((current / t["entry_price"]) - 1) * 100
                            else:
                                unrealized = (t["entry_price"] - current) * t["quantity"]
                                pct = ((t["entry_price"] / current) - 1) * 100
                            tp2 = t.get("take_profit2", 0)
                            tp2_str = f" TP2: ${tp2:,.2f}" if tp2 else ""
                            dca = t.get("dca_count", 0)
                            dca_str = f" DCA: {dca}x" if dca else ""
                            tp1_str = " [TP1 taken]" if t.get("tp1_hit") else ""
                            liq_str = f" Liq: ${liq:,.2f}" if liq > 0 else ""
                            print(f"  {t['symbol']} {direction} {lev}x: "
                                  f"entry ${t['entry_price']:,.2f} "
                                  f"now ${current:,.2f} ({pct:+.2f}%) "
                                  f"SL: ${t['stop_loss']:,.2f} TP1: ${t['take_profit']:,.2f}"
                                  f"{tp2_str}{dca_str}{tp1_str}{liq_str} "
                                  f"P&L: ${unrealized:+,.2f}")
                        else:
                            print(f"  {t['symbol']} {direction} {lev}x: "
                                  f"entry ${t['entry_price']:,.2f} "
                                  f"SL: ${t['stop_loss']:,.2f} TP: ${t['take_profit']:,.2f}")
            else:
                print("  Futures engine not active")
            print()
            continue
        if question.lower() == "futures trades":
            fe = runtime._futures_engine
            if fe:
                closed = [t for t in fe.store.trades.values()
                          if t["status"] != "open"]
                closed.sort(key=lambda t: t.get("exit_time", 0))
                closed = closed[-10:]
                if not closed:
                    print("  No closed futures trades yet")
                else:
                    for t in closed:
                        tag = "W" if t["pnl"] > 0 else "L"
                        direction = t.get("direction", "?")
                        lev = t.get("leverage", 1)
                        print(f"  [{tag}] {t['symbol']} {direction} {lev}x: "
                              f"${t['entry_price']:,.2f} -> ${t['exit_price']:,.2f} "
                              f"P&L: ${t['pnl']:+,.2f} ({t['pnl_pct']:+.2f}%) "
                              f"-- {t['exit_reason']}")
            else:
                print("  Futures engine not active")
            print()
            continue
        if question.lower() == "name":
            if user_name:
                print(f"  Current name: {user_name}")
                print(f"  (Say 'call me <new name>' to change)")
            else:
                print(f"  No name set. Tell me your name!")
            print()
            continue
        if question.lower() == "interests":
            interests = profile.get("interests", {})
            if interests:
                sorted_interests = sorted(interests.items(),
                                           key=lambda x: x[1].get("score", 0),
                                           reverse=True)
                print(f"  Your tracked interests:")
                for topic, data in sorted_interests:
                    score_bar = "#" * int(data["score"] * 10) + "." * (10 - int(data["score"] * 10))
                    print(f"    {topic:<16} [{score_bar}] "
                          f"count={data['count']} last={data.get('last', '?')}")
            else:
                print(f"  No interests tracked yet. Just keep chatting!")
            cautions = profile.get("cautions", [])
            if cautions:
                print(f"  Caution topics: {', '.join(cautions)}")
            # Show chat-derived topic summary if available
            if chat_store is not None:
                session_id = user_name.lower() if user_name else "cli_default"
                topic_summary = chat_store.get_topic_summary(session_id, last_n=100)
                if topic_summary:
                    print(f"  Chat topic frequency (last 100 msgs):")
                    for topic, count in list(topic_summary.items())[:10]:
                        print(f"    {topic}: {count} mentions")
            print()
            continue
        if question.lower() == "tree":
            try:
                from qor.knowledge_tree import tree_status
                tree_status(graph, gate._user_id)
            except Exception as e:
                print(f"  Knowledge tree error: {e}")
            print()
            continue
        if question.lower() == "bootstrap":
            try:
                from qor.knowledge_tree import bootstrap_tree
                # Get trade stores from runtime engines
                spot_store = runtime._trading_engine.store if runtime._trading_engine else None
                futures_store = runtime._futures_engine.store if runtime._futures_engine else None
                bootstrap_tree(
                    graph, gate._user_id,
                    profile=profile,
                    chat_store=chat_store,
                    trade_store=spot_store,
                    futures_store=futures_store,
                    memory_store=gate.memory,
                )
            except Exception as e:
                print(f"  Bootstrap error: {e}")
            print()
            continue

        # --- API key management commands ---
        api_cmd = _parse_api_command(question)
        if api_cmd is not None:
            if crypto is None:
                print("  Crypto not available — cannot encrypt API keys.")
                print("  Check that 'cryptography' package is installed.\n")
                continue

            action = api_cmd["action"]

            if action == "status":
                creds = profile.get("trading_credentials")
                if creds and "api_key" in creds and "api_secret" in creds:
                    mode = "DEMO (testnet)" if creds.get("testnet", True) else "PRODUCTION"
                    te = runtime._trading_engine
                    engine_status = "ACTIVE" if te else "INACTIVE"
                    print(f"  API Status:")
                    print(f"    Keys:       set (encrypted)")
                    print(f"    Mode:       {mode}")
                    print(f"    Linked to:  {creds.get('session_id', '?')}")
                    print(f"    Set at:     {creds.get('set_at', '?')}")
                    print(f"    Engine:     {engine_status}")
                elif creds and "api_key" in creds:
                    print(f"  API Status:")
                    print(f"    Key:        set (encrypted)")
                    print(f"    Secret:     NOT SET")
                    print(f"    Use: set api secret <YOUR_SECRET>")
                else:
                    print(f"  API Status: no keys set")
                    print(f"    Use: set api key <KEY> secret <SECRET>")
                print()
                continue

            if action == "remove":
                if _clear_trading_credentials(data_dir, profile):
                    config.trading.api_key = ""
                    config.trading.api_secret = ""
                    config.trading.enabled = False
                    config.futures.api_key = ""
                    config.futures.api_secret = ""
                    config.futures.enabled = False
                    if runtime._trading_engine:
                        runtime._trading_engine.stop()
                        runtime._trading_engine = None
                    if runtime._futures_engine:
                        runtime._futures_engine.stop()
                        runtime._futures_engine = None
                    print("  Trading credentials removed. Spot + futures engines stopped.\n")
                else:
                    print("  No trading credentials to remove.\n")
                continue

            if action == "stop":
                if runtime._trading_engine:
                    runtime._trading_engine.stop()
                    runtime._trading_engine = None
                    config.trading.enabled = False
                    print("  Trading engine stopped. Keys still saved.\n")
                else:
                    print("  Trading engine is not running.\n")
                continue

            if action == "start":
                if runtime._trading_engine:
                    print("  Trading engine is already running.\n")
                    continue
                # Try loading keys from config (already decrypted) or profile
                if not config.trading.api_key or not config.trading.api_secret:
                    stored_creds = _load_trading_credentials(profile, crypto, session_id)
                    if stored_creds:
                        config.trading.api_key = stored_creds["api_key"]
                        config.trading.api_secret = stored_creds["api_secret"]
                        config.trading.testnet = stored_creds["testnet"]
                    else:
                        print("  No API keys found. Set them first:")
                        print("  Use: set api key <KEY> secret <SECRET>\n")
                        continue
                config.trading.enabled = True
                mode = _restart_trading_engine(runtime, config, gate)
                if mode and not mode.startswith("ERROR"):
                    print(f"  Trading engine started ({mode}).\n")
                elif mode:
                    print(f"  Trading engine: {mode}\n")
                else:
                    print("  Failed to start trading engine.\n")
                continue

            if action == "set_both":
                creds = profile.get("trading_credentials", {})
                creds["api_key"] = crypto.encrypt_str(api_cmd["key"])
                creds["api_secret"] = crypto.encrypt_str(api_cmd["secret"])
                creds.setdefault("testnet", True)
                creds["session_id"] = session_id
                creds["set_at"] = datetime.now().isoformat()
                profile["trading_credentials"] = creds
                _save_profile(data_dir, profile)  # Credentials → profile.json only
                # Inject and start
                config.trading.api_key = api_cmd["key"]
                config.trading.api_secret = api_cmd["secret"]
                config.trading.testnet = creds["testnet"]
                config.trading.enabled = True
                # Same keys for futures
                config.futures.api_key = api_cmd["key"]
                config.futures.api_secret = api_cmd["secret"]
                config.futures.testnet = creds["testnet"]
                mode = _restart_trading_engine(runtime, config, gate)
                if mode and not mode.startswith("ERROR"):
                    print(f"  API credentials stored (encrypted). Trading engine started ({mode}).")
                elif mode:
                    print(f"  API credentials stored (encrypted). Engine: {mode}")
                else:
                    print(f"  API credentials stored (encrypted).")
                print()
                continue

            if action == "set_key":
                creds = profile.get("trading_credentials", {})
                creds["api_key"] = crypto.encrypt_str(api_cmd["key"])
                creds.setdefault("testnet", True)
                creds["session_id"] = session_id
                creds["set_at"] = datetime.now().isoformat()
                profile["trading_credentials"] = creds
                _save_profile(data_dir, profile)  # Credentials → profile.json only
                if "api_secret" in creds:
                    # Both keys exist — decrypt secret, inject and start
                    try:
                        secret = crypto.decrypt_str(creds["api_secret"])
                        config.trading.api_key = api_cmd["key"]
                        config.trading.api_secret = secret
                        config.trading.testnet = creds["testnet"]
                        config.trading.enabled = True
                        mode = _restart_trading_engine(runtime, config, gate)
                        if mode and not mode.startswith("ERROR"):
                            print(f"  API key updated (encrypted). Trading engine started ({mode}).")
                        elif mode:
                            print(f"  API key updated (encrypted). Engine: {mode}")
                        else:
                            print(f"  API key updated (encrypted).")
                    except Exception:
                        print(f"  API key stored (encrypted). Existing secret could not be decrypted.")
                        print(f"  Use: set api secret <YOUR_SECRET>")
                else:
                    print(f"  API key stored (encrypted). Now set your secret:")
                    print(f"  Use: set api secret <YOUR_SECRET>")
                print()
                continue

            if action == "set_secret":
                creds = profile.get("trading_credentials", {})
                creds["api_secret"] = crypto.encrypt_str(api_cmd["secret"])
                creds.setdefault("testnet", True)
                creds["session_id"] = session_id
                creds["set_at"] = datetime.now().isoformat()
                profile["trading_credentials"] = creds
                _save_profile(data_dir, profile)  # Credentials → profile.json only
                if "api_key" in creds:
                    # Both keys exist — decrypt key, inject and start
                    try:
                        key = crypto.decrypt_str(creds["api_key"])
                        config.trading.api_key = key
                        config.trading.api_secret = api_cmd["secret"]
                        config.trading.testnet = creds["testnet"]
                        config.trading.enabled = True
                        mode = _restart_trading_engine(runtime, config, gate)
                        if mode and not mode.startswith("ERROR"):
                            print(f"  API secret stored. Trading credentials complete!")
                            tmode = "DEMO (testnet)" if creds["testnet"] else "PRODUCTION"
                            print(f"  Mode: {tmode}. Use 'set trading live' for production.")
                            print(f"  Trading engine started ({mode}).")
                        elif mode:
                            print(f"  API secret stored (encrypted). Engine: {mode}")
                        else:
                            print(f"  API secret stored (encrypted).")
                    except Exception:
                        print(f"  API secret stored (encrypted). Existing key could not be decrypted.")
                        print(f"  Use: set api key <YOUR_KEY>")
                else:
                    print(f"  API secret stored (encrypted). Now set your key:")
                    print(f"  Use: set api key <YOUR_KEY>")
                print()
                continue

            if action == "set_live":
                print(f"  WARNING: PRODUCTION MODE — This will use REAL money on Binance!")
                print(f"  Type 'confirm live trading' to proceed.\n")
                _pending_live_confirm = True
                continue

            if action == "set_demo":
                creds = profile.get("trading_credentials", {})
                if creds:
                    creds["testnet"] = True
                    profile["trading_credentials"] = creds
                    _save_profile(data_dir, profile)  # Credentials → profile.json only
                    config.trading.testnet = True
                    if config.trading.api_key and config.trading.api_secret:
                        mode = _restart_trading_engine(runtime, config, gate)
                        print(f"  Trading mode set to DEMO (testnet). Engine restarted.\n")
                    else:
                        print(f"  Trading mode set to DEMO (testnet).\n")
                else:
                    print(f"  No trading credentials set. Set API keys first.\n")
                continue

            if action == "confirm_live":
                if not _pending_live_confirm:
                    print(f"  Use 'set trading live' first.\n")
                    continue
                _pending_live_confirm = False
                creds = profile.get("trading_credentials", {})
                if creds:
                    creds["testnet"] = False
                    profile["trading_credentials"] = creds
                    _save_profile(data_dir, profile)  # Credentials → profile.json only
                    config.trading.testnet = False
                    if config.trading.api_key and config.trading.api_secret:
                        mode = _restart_trading_engine(runtime, config, gate)
                        print(f"  Trading mode set to PRODUCTION. Be careful!")
                        if mode and not mode.startswith("ERROR"):
                            print(f"  Trading engine restarted ({mode}).")
                        elif mode:
                            print(f"  Engine: {mode}")
                    else:
                        print(f"  Trading mode set to PRODUCTION. Set API keys to start.")
                else:
                    print(f"  No trading credentials set. Set API keys first.")
                print()
                continue

            # --- Futures commands ---
            if action == "start_futures":
                if runtime._futures_engine:
                    print("  Futures engine is already running.\n")
                    continue
                # Use same keys as spot (Binance uses one key pair)
                if not config.futures.api_key or not config.futures.api_secret:
                    if config.trading.api_key and config.trading.api_secret:
                        config.futures.api_key = config.trading.api_key
                        config.futures.api_secret = config.trading.api_secret
                        config.futures.testnet = config.trading.testnet
                    else:
                        stored_creds = _load_trading_credentials(profile, crypto, session_id)
                        if stored_creds:
                            config.futures.api_key = stored_creds["api_key"]
                            config.futures.api_secret = stored_creds["api_secret"]
                            config.futures.testnet = stored_creds["testnet"]
                        else:
                            print("  No API keys found. Set them first:")
                            print("  Use: set api key <KEY> secret <SECRET>\n")
                            continue
                config.futures.enabled = True
                mode = _restart_futures_engine(runtime, config, gate)
                if mode and not mode.startswith("ERROR"):
                    print(f"  Futures engine started ({mode}).")
                    print(f"  NOTE: Make sure Hedge Mode and Multi-Asset Mode are")
                    print(f"  enabled in your Binance Futures settings.\n")
                elif mode:
                    print(f"  Futures engine: {mode}\n")
                else:
                    print("  Failed to start futures engine.\n")
                continue

            if action == "stop_futures":
                if runtime._futures_engine:
                    runtime._futures_engine.stop()
                    runtime._futures_engine = None
                    config.futures.enabled = False
                    print("  Futures engine stopped.\n")
                else:
                    print("  Futures engine is not running.\n")
                continue

            if action == "set_futures_leverage":
                lev = api_cmd["leverage"]
                lev = max(1, min(lev, config.futures.max_leverage))
                config.futures.leverage = lev
                if runtime._futures_engine:
                    runtime._futures_engine.set_leverage(lev)
                    print(f"  Futures leverage set to {lev}x (applied to all symbols).\n")
                else:
                    print(f"  Futures leverage set to {lev}x (will apply on next start).\n")
                continue

            if action == "set_futures_mode":
                mode = api_cmd["mode"]
                if mode not in ("scalp", "stable", "secure"):
                    print(f"  Invalid mode '{mode}'. Use: scalp, stable, or secure\n")
                    print("    scalp  — SL/TP from 5m-30m ATR (tight, fast trades)")
                    print("    stable — SL/TP from 30m-4h ATR (medium swings)")
                    print("    secure — SL/TP from 4h-1w ATR (wide, safe)\n")
                else:
                    config.futures.trade_mode = mode
                    desc = {"scalp": "5m-30m ATR", "stable": "30m-4h ATR", "secure": "4h-1w ATR"}
                    print(f"  Futures trade mode set to {mode.upper()} ({desc[mode]}).\n")
                continue

            if action == "set_spot_mode":
                mode = api_cmd["mode"]
                if mode not in ("scalp", "stable", "secure"):
                    print(f"  Invalid mode '{mode}'. Use: scalp, stable, or secure\n")
                    print("    scalp  — SL/TP from 5m-30m ATR (tight, fast trades)")
                    print("    stable — SL/TP from 30m-4h ATR (medium swings)")
                    print("    secure — SL/TP from 4h-1w ATR (wide, safe)\n")
                else:
                    config.trading.trade_mode = mode
                    desc = {"scalp": "5m-30m ATR", "stable": "30m-4h ATR", "secure": "4h-1w ATR"}
                    print(f"  Spot trade mode set to {mode.upper()} ({desc[mode]}).\n")
                continue

            # --- Multi-exchange commands ---

            if action == "list_exchanges":
                ex_list = _list_exchange_creds(profile)
                # Also show Binance (from trading_credentials)
                binance_creds = profile.get("trading_credentials")
                print("  Configured Exchanges:")
                if binance_creds and "api_key" in binance_creds:
                    mode = "DEMO" if binance_creds.get("testnet", True) else "LIVE"
                    te = runtime._trading_engine
                    status = "ACTIVE" if te else "inactive"
                    print(f"    binance: keys=set  mode={mode}  engine={status}")
                else:
                    print(f"    binance: not configured")
                if ex_list:
                    for ex in ex_list:
                        mode = "DEMO" if ex["testnet"] else "LIVE"
                        parts = []
                        if ex["has_key"]:
                            parts.append("key=set")
                        if ex["has_secret"]:
                            parts.append("secret=set")
                        if ex["has_passphrase"]:
                            parts.append("passphrase=set")
                        print(f"    {ex['name']}: {' '.join(parts)}  mode={mode}")
                elif not binance_creds:
                    print(f"    (none)")
                print()
                print("  Add exchange:  set <exchange> api key <KEY> secret <SECRET>")
                print("  With passphrase: set <exchange> api key <KEY> secret <SECRET> passphrase <PASS>")
                print("  Remove:        remove <exchange> api key")
                from qor.config import EXCHANGE_DEFAULTS
                print(f"  Supported: {', '.join(sorted(EXCHANGE_DEFAULTS.keys()))}\n")
                continue

            if action == "exchange_status":
                ex_name = api_cmd["exchange"]
                loaded = _load_exchange_creds(profile, ex_name, crypto)
                if loaded:
                    mode = "DEMO (testnet)" if loaded["testnet"] else "LIVE"
                    print(f"  {ex_name.upper()} API Status:")
                    print(f"    Key:    {_mask_key(loaded['api_key'])}")
                    print(f"    Secret: {_mask_key(loaded['api_secret'])}")
                    if "passphrase" in loaded:
                        print(f"    Pass:   {_mask_key(loaded['passphrase'])}")
                    print(f"    Mode:   {mode}")
                    # Check if it's in config.exchanges
                    ex_cfg = config.get_exchange(ex_name) if hasattr(config, 'get_exchange') else None
                    if ex_cfg and ex_cfg.enabled:
                        print(f"    Config: enabled")
                    else:
                        print(f"    Config: not in config.exchanges (keys stored, not active)")
                else:
                    print(f"  {ex_name.upper()}: no API keys set")
                    print(f"  Use: set {ex_name} api key <KEY> secret <SECRET>")
                print()
                continue

            if action == "set_exchange":
                ex_name = api_cmd["exchange"]
                _save_exchange_creds(
                    data_dir, profile, ex_name, crypto,
                    api_key=api_cmd["key"],
                    api_secret=api_cmd["secret"],
                    passphrase=api_cmd.get("passphrase"),
                    testnet=True, session_id=session_id,
                )
                # Also push into config.exchanges
                from qor.config import ExchangeKeys
                existing = config.get_exchange(ex_name) if hasattr(config, 'get_exchange') else None
                if existing:
                    existing.api_key = api_cmd["key"]
                    existing.api_secret = api_cmd["secret"]
                    if "passphrase" in api_cmd:
                        existing.passphrase = api_cmd["passphrase"]
                    existing.enabled = True
                else:
                    config.exchanges.append(ExchangeKeys(
                        name=ex_name, api_key=api_cmd["key"],
                        api_secret=api_cmd["secret"],
                        passphrase=api_cmd.get("passphrase", ""),
                        testnet=True, enabled=True,
                    ))
                print(f"  {ex_name.upper()} API credentials stored (encrypted).")
                print(f"  Mode: DEMO (testnet). Use 'set {ex_name} live' for production.\n")
                continue

            if action == "set_exchange_key":
                ex_name = api_cmd["exchange"]
                _save_exchange_creds(
                    data_dir, profile, ex_name, crypto,
                    api_key=api_cmd["key"], session_id=session_id,
                )
                print(f"  {ex_name.upper()} API key stored (encrypted).")
                loaded = _load_exchange_creds(profile, ex_name, crypto)
                if loaded and loaded["api_secret"]:
                    print(f"  Both key + secret set.\n")
                else:
                    print(f"  Now set secret: set {ex_name} api secret <SECRET>\n")
                continue

            if action == "set_exchange_secret":
                ex_name = api_cmd["exchange"]
                _save_exchange_creds(
                    data_dir, profile, ex_name, crypto,
                    api_secret=api_cmd["secret"], session_id=session_id,
                )
                print(f"  {ex_name.upper()} API secret stored (encrypted).")
                loaded = _load_exchange_creds(profile, ex_name, crypto)
                if loaded and loaded["api_key"]:
                    print(f"  Both key + secret set.\n")
                else:
                    print(f"  Now set key: set {ex_name} api key <KEY>\n")
                continue

            if action == "set_exchange_passphrase":
                ex_name = api_cmd["exchange"]
                _save_exchange_creds(
                    data_dir, profile, ex_name, crypto,
                    passphrase=api_cmd["passphrase"], session_id=session_id,
                )
                print(f"  {ex_name.upper()} passphrase stored (encrypted).\n")
                continue

            if action == "set_exchange_mode":
                ex_name = api_cmd["exchange"]
                testnet = api_cmd["testnet"]
                all_ex = profile.get("exchange_credentials", {})
                if ex_name in all_ex:
                    all_ex[ex_name]["testnet"] = testnet
                    profile["exchange_credentials"] = all_ex
                    _save_profile(data_dir, profile)
                    mode_str = "DEMO (testnet)" if testnet else "LIVE (production)"
                    if not testnet:
                        print(f"  WARNING: {ex_name.upper()} set to PRODUCTION — real money!")
                    else:
                        print(f"  {ex_name.upper()} set to {mode_str}.")
                else:
                    print(f"  No credentials for {ex_name}. Set keys first:")
                    print(f"  Use: set {ex_name} api key <KEY> secret <SECRET>")
                print()
                continue

            if action == "set_exchange_token":
                ex_name = api_cmd["exchange"]
                token = api_cmd["token"]
                _save_exchange_creds(
                    data_dir, profile, ex_name, crypto,
                    access_token=token, session_id=session_id,
                )
                # Push into config.exchanges
                from qor.config import ExchangeKeys
                existing = config.get_exchange(ex_name) if hasattr(config, 'get_exchange') else None
                if existing:
                    existing.access_token = token
                    existing.enabled = True
                else:
                    config.exchanges.append(ExchangeKeys(
                        name=ex_name, access_token=token,
                        testnet=True, enabled=True,
                    ))
                print(f"  {ex_name.upper()} access token stored (encrypted).")
                print(f"  Token valid until ~3:30 AM IST next day.\n")
                continue

            if action == "exchange_auth_code":
                ex_name = api_cmd["exchange"]
                auth_code = api_cmd["code"]
                redirect_uri = api_cmd.get("redirect_uri", "")
                # Load existing creds to get client_id/secret
                loaded = _load_exchange_creds(profile, ex_name, crypto)
                if not loaded or not loaded.get("api_key") or not loaded.get("api_secret"):
                    print(f"  Set API key+secret first:")
                    print(f"  Use: set {ex_name} api key <CLIENT_ID> secret <CLIENT_SECRET>")
                    print()
                    continue
                if not redirect_uri:
                    redirect_uri = "https://localhost"
                    print(f"  Using default redirect_uri: {redirect_uri}")
                try:
                    if ex_name == "upstox":
                        from qor.upstox import UpstoxClient
                        token_data = UpstoxClient.get_access_token(
                            auth_code, loaded["api_key"], loaded["api_secret"], redirect_uri)
                        token = token_data.get("access_token", "")
                        if not token:
                            print(f"  Token exchange failed: no access_token in response")
                            print(f"  Response: {token_data}\n")
                            continue
                        _save_exchange_creds(
                            data_dir, profile, ex_name, crypto,
                            access_token=token, session_id=session_id,
                        )
                        from qor.config import ExchangeKeys
                        existing = config.get_exchange(ex_name) if hasattr(config, 'get_exchange') else None
                        if existing:
                            existing.access_token = token
                            existing.enabled = True
                        else:
                            config.exchanges.append(ExchangeKeys(
                                name=ex_name, api_key=loaded["api_key"],
                                api_secret=loaded["api_secret"],
                                access_token=token, testnet=True, enabled=True,
                            ))
                        print(f"  {ex_name.upper()} access token obtained and stored (encrypted).")
                        print(f"  Token valid until ~3:30 AM IST next day.\n")
                    else:
                        print(f"  OAuth2 auth code exchange not implemented for {ex_name}.\n")
                except Exception as e:
                    print(f"  Token exchange error: {e}\n")
                continue

            if action == "remove_exchange":
                ex_name = api_cmd["exchange"]
                if _clear_exchange_creds(data_dir, profile, ex_name):
                    # Also remove from config.exchanges
                    config.exchanges = [
                        ex for ex in config.exchanges
                        if (ex.name if isinstance(ex, object) and hasattr(ex, 'name')
                            else ex.get("name", "")) != ex_name
                    ]
                    print(f"  {ex_name.upper()} credentials removed.\n")
                else:
                    print(f"  No credentials for {ex_name}.\n")
                continue

        # Detect name introduction or change mid-conversation
        _name_keywords = ["my name", "call me", "i'm ", "i am "]
        if any(kw in question.lower() for kw in _name_keywords):
            detected = _extract_name(question)
            if detected:
                user_name = detected
                session_id = user_name.lower()
                profile["user_name"] = user_name
                # gate._user_id stays the same — system_id is permanent
                # Name is just a display property on the user node
                save_profile_to_tree(graph, gate._user_id, profile)
                gate.system_prompt = _build_system_prompt(user_name, profile, model=learner.model)
                print(f"  [Got it! I'll call you {user_name} from now on.]\n")

        # Get recent chat context for follow-up resolution
        chat_ctx = None
        if chat_store is not None:
            try:
                session_id = user_name.lower() if user_name else "cli_default"
                chat_ctx = chat_store.get_context(session_id, last_n=6)
            except Exception:
                pass

        # --- Parallel query mode: submit to pool, print asynchronously ---
        if query_pool is not None:
            # Clean up completed queries
            query_pool.check_done()

            # Update pool's session_id and verbose in case they changed
            query_pool.session_id = session_id
            query_pool.verbose = verbose

            ticket = query_pool.submit(
                question, chat_context=chat_ctx,
                graph=graph, profile=profile,
                data_dir=data_dir, user_id=gate._user_id,
            )
            print(f"  [Processing query #{ticket.query_id}...]")
            turn_count += 1

            # Rebuild system prompt every 10 turns
            if turn_count % 10 == 0:
                gate.system_prompt = _build_system_prompt(
                    user_name, profile, model=learner.model)
            continue

        # --- Synchronous mode (query_pool disabled or unavailable) ---
        result = gate.answer(question, verbose=verbose, chat_context=chat_ctx)
        print(f"\nQOR: {result['answer']}")
        if not verbose:
            conf = result.get('confidence', 0)
            bar = "#" * int(conf * 10) + "." * (10 - int(conf * 10))
            print(f"  [{bar}] {result.get('source', '?')}")
        print()

        # Feedback detection — check if user is correcting/rating the last answer
        if graph is not None:
            try:
                from qor.knowledge_tree import FeedbackDetector, FeedbackProcessor
                feedback = FeedbackDetector.detect(question, _last_result, chat_ctx)
                if feedback and feedback.confidence >= 0.6:
                    msg = FeedbackProcessor.process(
                        feedback, graph, gate._user_id,
                        memory_store=gate.memory,
                        cache_store=cache_store)
                    if msg:
                        print(f"  [Noted: {msg}]")

                    # Re-fetch: correction/stale/soft_correction → re-answer
                    # the previous question with force_tool=True
                    if (feedback.event_type in ("correction", "stale",
                                                "soft_correction")
                            and _last_result
                            and _last_result.get("question")):
                        prev_q = _last_result["question"]
                        print(f"  [Re-checking: {prev_q[:60]}...]")
                        corrected = gate.answer(
                            prev_q, verbose=verbose,
                            chat_context=chat_ctx,
                            force_tool=True)
                        if corrected.get("answer"):
                            print(f"\nQOR (corrected): {corrected['answer']}\n")
                            result = corrected  # Update result for chat save
                        else:
                            print()
                    else:
                        print()
            except Exception:
                pass
        _last_result = result

        # Save conversation turn
        if chat_store is not None:
            try:
                session_id = user_name.lower() if user_name else "cli_default"
                chat_store.add_turn(session_id, question, result)
            except Exception:
                pass

        # Track user interests from this question
        _update_interests(profile, question, result, data_dir,
                          graph=graph, user_id=gate._user_id)
        turn_count += 1

        # Rebuild system prompt every 10 turns to reflect updated interests
        if turn_count % 10 == 0:
            gate.system_prompt = _build_system_prompt(user_name, profile, model=learner.model)

    # Cleanup
    if query_pool is not None:
        try:
            query_pool.wait_all(timeout=10)
            query_pool.shutdown()
        except Exception:
            pass
    if gate.memory._dirty:
        gate.memory.save()
    if cache_store is not None and cache_store._dirty:
        cache_store.save()
    if chat_store is not None and chat_store._dirty:
        chat_store.save()
    runtime.stop()
    # Close browser if it was used
    try:
        from qor.browser import get_engine
        get_engine().close()
    except Exception:
        pass
    if graph is not None:
        try:
            graph.close()
        except Exception:
            pass
    print("\n  Runtime stopped. Goodbye.")


def cmd_consolidate(args):
    """Manually run cleanup — prune old live data, rotate checkpoints, compact graph."""
    from qor.config import QORConfig, PRESETS
    from qor.continual import ContinualLearner
    from qor.runtime import QORRuntime
    from qor.confidence import ConfidenceGate
    from qor.graph import QORGraph, GraphConfig
    from qor.rag import QORRag

    config = PRESETS.get(args.size, PRESETS["small"])()
    if args.device:
        config.train.device = args.device

    # Resolve all runtime paths under qor-data/
    config.resolve_data_paths()

    ckpt = args.checkpoint or os.path.join(config.train.checkpoint_dir, "best_model.pt")

    learner = ContinualLearner(config)
    learner.load(ckpt)

    # Wire knowledge systems
    gate = ConfidenceGate(learner.model, learner.tokenizer, config)
    gate.memory.path = config.get_data_path("memory.parquet")

    graph = None
    try:
        graph_config = config.graph if hasattr(config, 'graph') else GraphConfig()
        graph = QORGraph(graph_config)
        graph.open()
    except Exception as e:
        print(f"  Graph: skipped ({e})")

    # Create runtime and run cleanup
    runtime = QORRuntime(config)
    runtime._learner = learner
    runtime._gate = gate
    runtime._graph = graph

    from qor.runtime import CheckpointRotator
    if config.runtime.checkpoint_rotation:
        runtime._checkpoint_rotator = CheckpointRotator(config.train.checkpoint_dir)

    print(f"  Running cleanup (memory: {len(gate.memory.entries)} entries)...")
    result = runtime.cleanup_now()
    print(f"  Status:           {result.get('status')}")
    print(f"  Memory removed:   {result.get('memory_removed', 0)}")
    print(f"  Graph removed:    {result.get('graph_removed', 0)}")
    print(f"  Timestamp:        {result.get('timestamp', '')}")

    # Save
    if gate.memory._dirty:
        gate.memory.save()
    if graph is not None:
        try:
            graph.close()
        except Exception:
            pass


def cmd_ui(args):
    """Launch Gradio web UI."""
    from qor.config import QORConfig, PRESETS
    from qor.hub import create_gradio_app

    config = PRESETS.get(args.size, PRESETS["small"])()
    ckpt = args.checkpoint or os.path.join(config.train.checkpoint_dir, "best_model.pt")

    demo = create_gradio_app(config, ckpt)
    if demo:
        demo.launch(server_port=args.port or 7860, share=args.share)


def cmd_graph(args):
    """Knowledge graph operations."""
    from qor.config import QORConfig
    from qor.graph import QORGraph, GraphConfig

    qor_config = QORConfig()
    qor_config.resolve_data_paths()
    default_db = qor_config.graph.db_path if hasattr(qor_config.graph, 'db_path') else "graph_db"
    config = GraphConfig(db_path=args.db_path or default_db)
    graph = QORGraph(config)
    graph.open()

    try:
        if args.action == "stats":
            s = graph.stats()
            print(f"\n  Knowledge Graph Stats:")
            print(f"    Nodes:      {s['node_count']}")
            print(f"    Edges:      {s['edge_count']}")
            print(f"    Synonyms:   {s['synonym_count']}")
            print(f"    Predicates: {s['predicate_count']}")
            print(f"    Backend:    {s['backend']}")
            print(f"    Serializer: {s['serializer']}")
            if s['db_size_bytes'] > 0:
                print(f"    DB Size:    {s['db_size_bytes'] / 1024:.1f} KB")
            print()

        elif args.action == "add":
            # Parse "Subject predicate Object" from args.text
            parts = args.text.split()
            if len(parts) < 3:
                print("Usage: python -m qor graph --action add --text 'Ravi manages ML_Group'")
                return
            subject = parts[0]
            predicate = parts[1]
            obj = "_".join(parts[2:])
            graph.add_edge(subject, predicate, obj, confidence=1.0, source="cli")
            print(f"  Added: {subject} -[{predicate}]-> {obj}")

        elif args.action == "query":
            result = graph.semantic_query(args.text)
            print(f"\n  Query: {args.text}")
            print(f"  Answer: {result['answer']}")
            print(f"  Confidence: {result['confidence']:.2f}")
            print(f"  Entities found: {result.get('entities_found', [])}")
            print(f"  Edges matched: {result.get('edge_count', 0)}")
            print()

        elif args.action == "export":
            output = args.output or "graph_export.json"
            graph.export_json(output)
            print(f"  Graph exported to {output}")

        elif args.action == "import":
            if not args.input:
                print("Usage: python -m qor graph --action import --input graph.json")
                return
            graph.import_json(args.input)
            print(f"  Graph imported from {args.input}")

        elif args.action == "nodes":
            nodes = graph.list_nodes()
            print(f"\n  Nodes ({len(nodes)}):")
            for nid, data in nodes:
                print(f"    {nid} ({data.get('type', '?')})")
            print()

        elif args.action == "edges":
            if not args.entity:
                print("Usage: python -m qor graph --action edges --entity ravi")
                return
            edges = graph.get_edges(args.entity, direction="both")
            print(f"\n  Edges for '{args.entity}' ({len(edges)}):")
            for e in edges:
                print(f"    {e['subject']} -[{e['predicate']}]-> {e['object']} (conf={e.get('confidence', 0):.2f})")
            print()
        else:
            print("Unknown action. Use: stats, add, query, export, import, nodes, edges")
    finally:
        graph.close()


def cmd_train_vision(args):
    """Train with vision (image-text) data."""
    from qor.config import QORConfig, PRESETS
    from qor.train import Trainer

    if getattr(args, 'pretrained', False):
        config = QORConfig.qor3b_multimodal()
    else:
        config = QORConfig.small_multimodal()

    # Override vision settings
    config.vision.enabled = True
    if getattr(args, 'pretrained', False):
        config.vision.use_pretrained = True
    if args.image_size:
        config.vision.image_size = args.image_size
    if args.patch_size:
        config.vision.patch_size = args.patch_size
    if args.channels:
        config.vision.in_channels = args.channels
    if args.steps:
        config.train.max_steps = args.steps
    if args.device:
        config.train.device = args.device
    if args.batch_size:
        config.train.batch_size = args.batch_size

    os.makedirs(config.train.checkpoint_dir, exist_ok=True)

    trainer = Trainer(config)
    trainer.train_multimodal("vision", args.data)


def cmd_train_audio(args):
    """Train with audio-text data."""
    from qor.config import QORConfig, PRESETS
    from qor.train import Trainer

    if getattr(args, 'pretrained', False):
        config = QORConfig.qor3b_multimodal()
    else:
        config = QORConfig.small_multimodal()

    config.audio.enabled = True
    if getattr(args, 'pretrained', False):
        config.audio.use_pretrained = True
    if args.steps:
        config.train.max_steps = args.steps
    if args.device:
        config.train.device = args.device
    if args.batch_size:
        config.train.batch_size = args.batch_size

    os.makedirs(config.train.checkpoint_dir, exist_ok=True)

    trainer = Trainer(config)
    trainer.train_multimodal("audio", args.data)


def cmd_test_vision(args):
    """Test vision model on a single image."""
    from qor.config import QORConfig
    from qor.model import QORModel
    from qor.tokenizer import QORTokenizer
    from qor.config import VisionConfig

    config = QORConfig.small_multimodal()
    if args.device:
        config.train.device = args.device

    device = config.get_device()

    # Load tokenizer
    tokenizer = QORTokenizer()
    tok_path = config.tokenizer.save_path
    if os.path.exists(tok_path):
        tokenizer.load(tok_path)
    else:
        print("Train a tokenizer first: python -m qor tokenizer")
        return

    config.model.vocab_size = tokenizer.vocab_size

    # Load model
    ckpt_path = args.checkpoint
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return

    import torch
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Reconstruct vision config from checkpoint if available
    vision_config = config.vision
    encoder_hf_configs = checkpoint.get("encoder_hf_configs", {})
    if "config" in checkpoint and "vision" in checkpoint["config"]:
        from qor.config import VisionConfig
        vision_config = VisionConfig()
        for k, v in checkpoint["config"]["vision"].items():
            if hasattr(vision_config, k):
                setattr(vision_config, k, v)

    model = QORModel(
        config.model,
        vision_config=vision_config,
        _from_checkpoint=True,
        _encoder_hf_configs=encoder_hf_configs,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=False)

    # Load and process image
    try:
        from PIL import Image
        from torchvision import transforms
    except ImportError:
        print("Requires: pip install Pillow torchvision")
        return

    t_list = [
        transforms.Resize((config.vision.image_size, config.vision.image_size)),
        transforms.ToTensor(),
    ]
    if config.vision.in_channels == 1:
        t_list.insert(0, transforms.Grayscale(num_output_channels=1))
        t_list.append(transforms.Normalize([0.5], [0.5]))
    else:
        t_list.append(transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
    transform = transforms.Compose(t_list)

    mode = 'L' if config.vision.in_channels == 1 else 'RGB'
    image = transform(Image.open(args.image).convert(mode)).unsqueeze(0).to(device)

    # Build input with image tokens
    n_patches = config.vision.n_patches
    token_ids = tokenizer.encode_with_image("", n_patches, "")
    # Remove trailing EOS so model can generate
    if token_ids[-1] == tokenizer.eos_id:
        token_ids = token_ids[:-1]

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    patch_id = tokenizer.image_patch_id
    image_positions = (input_ids[0] == patch_id).nonzero(as_tuple=True)[0].unsqueeze(0)
    generated = model.generate(
        input_ids,
        max_new_tokens=args.max_tokens or 100,
        temperature=args.temperature or 0.8,
        images=image,
        image_positions=image_positions,
        stop_tokens=[tokenizer.eos_id],
    )

    output_text = tokenizer.decode(generated, skip_special_tokens=True)
    print(f"\nImage: {args.image}")
    print(f"Model says: {output_text}")


def cmd_serve(args):
    """Start the API server."""
    from qor.config import QORConfig, PRESETS

    config = PRESETS.get(args.size, PRESETS["small"])()
    if args.port:
        config.serve.port = args.port
    if args.quantize:
        config.serve.quantize = True
    if args.graph:
        config.serve.graph_enabled = True
        config.serve.graph_db_path = args.graph_db
    if getattr(args, 'device', None):
        config.train.device = args.device

    # Full runtime mode — wires ALL systems (tools, trading, graph, knowledge)
    if getattr(args, 'full', False):
        config.resolve_data_paths()
        ckpt = getattr(args, 'checkpoint', None) or os.path.join(
            config.train.checkpoint_dir, "best_model.pt")
        from qor.serve import run_full_server
        run_full_server(config, checkpoint_path=ckpt)
        return

    if args.fastapi:
        from qor.serve import run_fastapi_server
        run_fastapi_server(config)
    else:
        from qor.serve import run_flask_server
        run_flask_server(config)


def cmd_optimize(args):
    """Run QSearch optimization for a QOR subsystem."""
    target = args.target
    print(f"\n{'='*60}")
    print(f"QSearch Quantum-Inspired Optimizer — target: {target}")
    print(f"{'='*60}\n")

    if target == "trading":
        # Load trade history
        from qor.config import QORConfig, PRESETS
        config = PRESETS.get(args.size, PRESETS["small"])()
        config.resolve_data_paths()
        data_dir = config.trading.data_dir
        trades_path = os.path.join(data_dir, "trades.parquet")
        if not os.path.exists(trades_path):
            # Try futures
            data_dir = config.futures.data_dir
            trades_path = os.path.join(data_dir, "trades.parquet")
        if not os.path.exists(trades_path):
            print("ERROR: No trade history found. Need >= 10 closed trades.")
            print(f"  Looked in: {config.trading.data_dir}/trades.parquet")
            print(f"             {config.futures.data_dir}/trades.parquet")
            return
        from qor.trading import TradeStore
        store = TradeStore(trades_path)
        from qor.qsearch import optimize
        try:
            best = optimize(
                "trading", trade_store=store,
                n_workers=args.workers, generations=args.generations,
                branches=args.branches, survivors=args.survivors,
            )
        except ValueError as e:
            print(f"ERROR: {e}")
            return

    elif target in ("cortex", "mamba"):
        from qor.qsearch import optimize
        best = optimize(
            "mamba",  # qsearch uses "mamba" key internally
            n_workers=args.workers, generations=args.generations,
            branches=args.branches, survivors=args.survivors,
        )

    elif target == "cms":
        from qor.qsearch import optimize
        best = optimize(
            "cms",
            n_workers=args.workers, generations=args.generations,
            branches=args.branches, survivors=args.survivors,
        )

    else:
        print(f"Unknown target: {target}. Use: trading, cortex, cms")
        return

    # Show results
    print(f"\n{'='*60}")
    print(f"OPTIMAL PARAMETERS ({target})")
    print(f"{'='*60}")
    for k, v in sorted(best.data.items()):
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")
    print(f"\n  Fitness: {best.fitness:+.6f}")
    print(f"  Generation: {best.generation}")

    # Show how to apply
    if target == "trading":
        print(f"\nTo apply these settings, update TradingConfig in config.py:")
        for k, v in sorted(best.data.items()):
            if isinstance(v, float):
                print(f"  {k} = {v:.4f}")
            else:
                print(f"  {k} = {v}")
    elif target in ("cortex", "mamba"):
        print(f"\nTo apply, update ModelConfig in config.py:")
        for k, v in sorted(best.data.items()):
            print(f"  {k} = {int(v) if isinstance(v, float) and v == int(v) else v}")
    elif target == "cms":
        print(f"\nTo apply, update ModelConfig in config.py:")
        for k, v in sorted(best.data.items()):
            print(f"  {k} = {int(v)}")


def cmd_ngre_train(args):
    """Run the 4-phase NGRE training pipeline."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from scripts.train_ngre import (
        phase1_extract_embeddings, phase2_bootstrap_brain,
        phase3_joint_finetune, phase4_export,
    )
    from qor.config import QORConfig, PRESETS

    config = PRESETS.get(args.size, PRESETS["small"])()
    config.resolve_data_paths()

    data_dir = config.get_data_path("knowledge")
    ngre_dir = config.get_data_path("ngre")
    os.makedirs(ngre_dir, exist_ok=True)

    phases = ["1", "2", "3", "4"] if args.phase == "all" else [args.phase]
    results = {}

    for phase in phases:
        print(f"\n{'='*60}")
        print(f"  NGRE Training — Phase {phase}")
        print(f"{'='*60}")

        if phase == "1":
            results["phase1"] = phase1_extract_embeddings(
                data_dir=data_dir, output_dir=ngre_dir)
        elif phase == "2":
            results["phase2"] = phase2_bootstrap_brain(
                ngre_dir=ngre_dir, d_model=args.d_model,
                epochs=args.epochs, lr=args.lr)
        elif phase == "3":
            results["phase3"] = phase3_joint_finetune(
                ngre_dir=ngre_dir, epochs=args.epochs, lr=args.lr)
        elif phase == "4":
            output = os.path.join(config.train.checkpoint_dir, "ngre_brain_final.pt")
            results["phase4"] = phase4_export(ngre_dir=ngre_dir, output=output)

    print(f"\n{'='*60}")
    print("  Training complete!")
    for k, v in results.items():
        print(f"  {k}: {v}")


def cmd_graph_health(args):
    """Check knowledge graph health metrics."""
    from qor.config import QORConfig, PRESETS
    from qor.graph import QORGraph, GraphConfig

    config = PRESETS.get(args.size, PRESETS["small"])()
    config.resolve_data_paths()

    graph_config = config.graph if hasattr(config, 'graph') else GraphConfig()
    if args.db_path:
        graph_config.db_path = args.db_path
    graph = QORGraph(graph_config)
    graph.open()

    stats = graph.stats()
    print(f"\n  Graph Health Report")
    print(f"  {'─'*50}")
    print(f"  Nodes:       {stats.get('node_count', 0)}")
    print(f"  Edges:       {stats.get('edge_count', 0)}")

    # Hot tier stats
    ht = stats.get("hot_tier", {})
    if ht:
        print(f"  Hot tier:    {ht.get('size', 0)} nodes "
              f"(hits={ht.get('hits', 0)}, misses={ht.get('misses', 0)})")

    # Embedding stats
    emb = stats.get("embedding_count", 0)
    if emb:
        print(f"  Embeddings:  {emb}")

    # Type distribution
    td = stats.get("type_distribution", {})
    if td:
        print(f"\n  Node Types:")
        for ntype, count in sorted(td.items(), key=lambda x: -x[1]):
            print(f"    {ntype}: {count}")

    # NGRE health monitor if available
    try:
        from qor.health import GraphHealthMonitor
        monitor = GraphHealthMonitor(graph)
        health = monitor.check_all()
        print(f"\n  Health Metrics:")
        for metric, value in health.items():
            if isinstance(value, float):
                print(f"    {metric}: {value:.4f}")
            else:
                print(f"    {metric}: {value}")
    except Exception as e:
        print(f"\n  (Health monitor not available: {e})")

    graph.close()


def cmd_ngre_status(args):
    """Show NGRE brain status and parameter counts."""
    from qor.config import QORConfig, PRESETS

    config = PRESETS.get(args.size, PRESETS["small"])()
    config.resolve_data_paths()

    # Check for NGRE brain checkpoint
    ngre_dir = config.get_data_path("ngre")
    candidates = [
        os.path.join(config.train.checkpoint_dir, "ngre_brain_final.pt"),
        os.path.join(ngre_dir, "ngre_brain_finetuned.pt"),
        os.path.join(ngre_dir, "ngre_brain.pt"),
    ]

    print(f"\n  NGRE Brain Status")
    print(f"  {'─'*50}")

    found = None
    for c in candidates:
        if os.path.exists(c):
            found = c
            break

    if found:
        size_mb = os.path.getsize(found) / (1024 * 1024)
        print(f"  Checkpoint:  {found} ({size_mb:.1f} MB)")
        try:
            import torch
            ckpt = torch.load(found, map_location="cpu", weights_only=False)
            phase = ckpt.get("phase", "?")
            loss = ckpt.get("final_loss", "?")
            d_model = ckpt.get("config", {}).get("d_model", 768)
            print(f"  Phase:       {phase}")
            print(f"  Final loss:  {loss}")
            print(f"  d_model:     {d_model}")
            if ckpt.get("export_time"):
                print(f"  Exported:    {ckpt['export_time']}")
        except Exception as e:
            print(f"  (Cannot load checkpoint: {e})")
    else:
        print(f"  Checkpoint:  NOT FOUND")
        print(f"  Searched:    {', '.join(candidates)}")

    # Show module info
    try:
        from qor.ngre import NGREBrain
        brain = NGREBrain(d_hidden=768)
        total = sum(p.numel() for p in brain.parameters())
        trainable = sum(p.numel() for p in brain.parameters() if p.requires_grad)
        frozen = total - trainable
        print(f"\n  Architecture:")
        print(f"    Total params:     {total:,}")
        print(f"    Trainable:        {trainable:,}")
        print(f"    Frozen (Mamba):   {frozen:,}")
    except Exception as e:
        print(f"\n  (Cannot load NGRE module: {e})")

    # Routing stats
    try:
        gate_stats_path = config.get_data_path("routing_stats.json")
        if os.path.exists(gate_stats_path):
            with open(gate_stats_path) as f:
                rs = json.load(f)
            print(f"\n  Routing Stats:")
            for k, v in rs.items():
                print(f"    {k}: {v}")
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="QOR — The Qore Mind",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m qor setup                    # Create project structure
  python -m qor tokenizer                # Train tokenizer
  python -m qor train                    # Train small model
  python -m qor train --size medium      # Train medium model
  python -m qor test                     # Run the Mind Test
  python -m qor chat                     # Chat with model
  python -m qor serve                    # Start API server
  python -m qor learn                    # Learn new files
  python -m qor watch                    # Auto-learn (live)
        """,
    )

    sub = parser.add_subparsers(dest="command", help="Command to run")

    # === Setup ===
    p = sub.add_parser("setup", help="Create project folders and sample data")

    # === Tokenizer ===
    p = sub.add_parser("tokenizer", help="Train BPE tokenizer")
    p.add_argument("--size", default="small", choices=SIZE_CHOICES)

    # === Train ===
    p = sub.add_parser("train", help="Train the model")
    p.add_argument("--size", default="small", choices=SIZE_CHOICES)
    p.add_argument("--steps", type=int, help="Training steps")
    p.add_argument("--device", type=str, help="Device: cpu/cuda/mps")
    p.add_argument("--batch-size", type=int, help="Batch size")
    p.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    p.add_argument("--distributed", action="store_true", help="Multi-GPU DDP training")
    p.add_argument("--gradient-checkpointing", action="store_true", help="Save VRAM with gradient checkpointing")
    p.add_argument("--compile", action="store_true", help="Enable torch.compile for speedup")

    # === Eval ===
    p = sub.add_parser("eval", help="Evaluate the model")
    p.add_argument("--size", default="small", choices=SIZE_CHOICES)
    p.add_argument("--checkpoint", type=str, help="Checkpoint path")
    p.add_argument("--eval-file", type=str, help="Text file for perplexity")

    # === Test ===
    p = sub.add_parser("test", help="Run the Mind Test (continual learning)")
    p.add_argument("--size", default="small", choices=SIZE_CHOICES)
    p.add_argument("--device", type=str, help="Device")

    # === Chat ===
    p = sub.add_parser("chat", help="Interactive chat")
    p.add_argument("--size", default="small", choices=SIZE_CHOICES)
    p.add_argument("--device", type=str, help="Device")
    p.add_argument("--checkpoint", type=str, help="Checkpoint path")
    p.add_argument("--temperature", type=float, help="Sampling temperature")

    # === Generate ===
    p = sub.add_parser("generate", help="Generate text")
    p.add_argument("prompt", type=str, help="Text prompt")
    p.add_argument("--size", default="small", choices=SIZE_CHOICES)
    p.add_argument("--device", type=str, help="Device")
    p.add_argument("--checkpoint", type=str, help="Checkpoint path")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max-tokens", type=int, default=200)

    # === Learn ===
    p = sub.add_parser("learn", help="Learn from new text files")
    p.add_argument("--size", default="small", choices=SIZE_CHOICES)
    p.add_argument("--device", type=str, help="Device")
    p.add_argument("--checkpoint", type=str, help="Checkpoint path")
    p.add_argument("--folder", type=str, default="learn", help="Folder with new files")

    # === Watch ===
    p = sub.add_parser("watch", help="Auto-learn new files (live)")
    p.add_argument("--size", default="small", choices=SIZE_CHOICES)
    p.add_argument("--device", type=str, help="Device")
    p.add_argument("--checkpoint", type=str, help="Checkpoint path")
    p.add_argument("--folder", type=str, default="learn", help="Folder to watch")
    p.add_argument("--interval", type=int, default=10, help="Check interval (seconds)")

    # === Run (full runtime) ===
    p = sub.add_parser("run", help="Start full runtime: read loop + consolidation + chat")
    p.add_argument("--size", default="small", choices=SIZE_CHOICES)
    p.add_argument("--device", type=str, help="Device: cpu/cuda/mps")
    p.add_argument("--checkpoint", type=str, help="Checkpoint path")
    p.add_argument("--temperature", type=float, help="Sampling temperature")

    # === Consolidate (manual) ===
    p = sub.add_parser("consolidate", help="Manually consolidate current batch")
    p.add_argument("--size", default="small", choices=SIZE_CHOICES)
    p.add_argument("--device", type=str, help="Device: cpu/cuda/mps")
    p.add_argument("--checkpoint", type=str, help="Checkpoint path")

    # === Serve ===
    p = sub.add_parser("serve", help="Start API server")
    p.add_argument("--size", default="small", choices=SIZE_CHOICES)
    p.add_argument("--port", type=int, help="Server port")
    p.add_argument("--fastapi", action="store_true", help="Use FastAPI (production)")
    p.add_argument("--quantize", action="store_true", help="Enable INT8 quantization (CPU)")
    p.add_argument("--graph", action="store_true", help="Enable graph-backed knowledge")
    p.add_argument("--graph-db", type=str, default="graph_db", help="Graph database path for --graph")
    p.add_argument("--full", action="store_true", help="Full runtime mode (tools, trading, graph, all systems)")
    p.add_argument("--checkpoint", type=str, help="Checkpoint path")
    p.add_argument("--device", type=str, help="Device: cpu/cuda/mps")

    # === Graph ===
    p = sub.add_parser("graph", help="Knowledge graph operations")
    p.add_argument("--action", default="stats", choices=["stats", "add", "query", "export", "import", "nodes", "edges"])
    p.add_argument("--text", type=str, default="", help="Text for add/query actions")
    p.add_argument("--entity", type=str, help="Entity ID for edges action")
    p.add_argument("--output", type=str, help="Output file for export")
    p.add_argument("--input", type=str, help="Input file for import")
    p.add_argument("--db-path", type=str, default="graph_db", help="Graph database path")

    # === NGRE Train ===
    p = sub.add_parser("ngre-train", help="Run NGRE 4-phase training pipeline")
    p.add_argument("--phase", default="all", choices=["1", "2", "3", "4", "all"],
                   help="Training phase (1=extract, 2=bootstrap, 3=finetune, 4=export)")
    p.add_argument("--size", default="small", choices=SIZE_CHOICES)
    p.add_argument("--epochs", type=int, default=5, help="Training epochs per phase")
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    p.add_argument("--d-model", type=int, default=768, help="Brain hidden dimension")

    # === Graph Health ===
    p = sub.add_parser("graph-health", help="Check knowledge graph health metrics")
    p.add_argument("--size", default="small", choices=SIZE_CHOICES)
    p.add_argument("--db-path", type=str, default="", help="Graph database path override")

    # === NGRE Status ===
    p = sub.add_parser("ngre-status", help="Show NGRE brain status and parameters")
    p.add_argument("--size", default="small", choices=SIZE_CHOICES)

    # === Train Vision ===
    p = sub.add_parser("train-vision", help="Train with image-text data")
    p.add_argument("--data", type=str, required=True, help="Path to image data directory")
    p.add_argument("--steps", type=int, help="Training steps")
    p.add_argument("--device", type=str, help="Device: cpu/cuda/mps")
    p.add_argument("--batch-size", type=int, help="Batch size")
    p.add_argument("--image-size", type=int, help="Image size (default: 28 for MNIST)")
    p.add_argument("--patch-size", type=int, help="Patch size (default: 7 for MNIST)")
    p.add_argument("--channels", type=int, help="Image channels (1=grayscale, 3=RGB)")
    p.add_argument("--pretrained", action="store_true", help="Use pretrained SigLIP encoder (frozen + bridge)")

    # === Train Audio ===
    p = sub.add_parser("train-audio", help="Train with audio-text data")
    p.add_argument("--data", type=str, required=True, help="Path to audio data directory")
    p.add_argument("--steps", type=int, help="Training steps")
    p.add_argument("--device", type=str, help="Device: cpu/cuda/mps")
    p.add_argument("--batch-size", type=int, help="Batch size")
    p.add_argument("--pretrained", action="store_true", help="Use pretrained Whisper encoder (frozen + bridge)")

    # === Test Vision ===
    p = sub.add_parser("test-vision", help="Test vision model on an image")
    p.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    p.add_argument("--image", type=str, required=True, help="Image file to test")
    p.add_argument("--device", type=str, help="Device")
    p.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")
    p.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")

    # === Load Donor ===
    p = sub.add_parser("load-donor", help="Load 3B donor weights into QOR")
    p.add_argument("--model-dir", type=str, help="Local donor model directory (downloads if not set)")
    p.add_argument("--output", type=str, default="checkpoints/qor3b.pt", help="Output checkpoint path")
    p.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32", "bfloat16"])

    # === Build Multimodal ===
    p = sub.add_parser("build-multimodal", help="Build unified multimodal checkpoint (SmolLM3 + SigLIP + Whisper)")
    p.add_argument("--donor-checkpoint", type=str, help="Existing qor3b checkpoint (skips re-downloading)")
    p.add_argument("--model-dir", type=str, help="Local donor model directory")
    p.add_argument("--output", type=str, default="checkpoints/qor3b_multimodal.pt", help="Output checkpoint path")
    p.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32", "bfloat16"])

    # === Optimize (QSearch) ===
    p = sub.add_parser("optimize", help="QSearch quantum-inspired parameter optimization")
    p.add_argument("--target", required=True, choices=["trading", "cortex", "cms"],
                   help="What to optimize: trading params, CORTEX hyperparams, or CMS frequencies")
    p.add_argument("--size", default="small", choices=SIZE_CHOICES)
    p.add_argument("--workers", type=int, default=None, help="Parallel workers (default: CPU count)")
    p.add_argument("--generations", type=int, default=30, help="Max search generations")
    p.add_argument("--branches", type=int, default=50, help="Variants per survivor per generation")
    p.add_argument("--survivors", type=int, default=8, help="Survivors per generation")

    # === UI ===
    p = sub.add_parser("ui", help="Launch Gradio web UI")
    p.add_argument("--size", default="small", choices=SIZE_CHOICES)
    p.add_argument("--checkpoint", type=str, help="Checkpoint path")
    p.add_argument("--port", type=int, default=7860, help="UI port")
    p.add_argument("--share", action="store_true", help="Create public link")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        print("\n  Start with: python -m qor setup")
        return

    # Setup logging
    log_level = getattr(args, 'verbose', False) and logging.DEBUG or logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    commands = {
        "setup": cmd_setup,
        "tokenizer": cmd_tokenizer,
        "train": cmd_train,
        "train-vision": cmd_train_vision,
        "train-audio": cmd_train_audio,
        "test-vision": cmd_test_vision,
        "eval": cmd_eval,
        "test": cmd_test,
        "chat": cmd_chat,
        "generate": cmd_generate,
        "learn": cmd_learn,
        "watch": cmd_watch,
        "run": cmd_run,
        "consolidate": cmd_consolidate,
        "serve": cmd_serve,
        "ui": cmd_ui,
        "graph": cmd_graph,
        "load-donor": cmd_load_donor,
        "build-multimodal": cmd_build_multimodal,
        "optimize": cmd_optimize,
        "ngre-train": cmd_ngre_train,
        "graph-health": cmd_graph_health,
        "ngre-status": cmd_ngre_status,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()

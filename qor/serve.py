"""
QOR API Server — Serve the Qore Mind
======================================

Functions:
  _register_dashboard_routes()  — Dashboard trading terminal endpoints (10 routes)
  _register_routes()            — Shared API routes (32 routes), calls dashboard routes
  run_flask_server()            — Simple Flask server
  run_fastapi_server()          — FastAPI with OpenAPI docs
  run_full_server()             — Full runtime server
  start_api_thread()            — Background daemon thread (for cmd_run)
"""

import os
import json
import time
import torch
from typing import Optional

from .config import QORConfig
from .model import QORModel
from .tokenizer import QORTokenizer


def _validate_generate_params(prompt, max_tokens=None, temperature=None, top_k=None, top_p=None):
    """Validate generation parameters. Returns error message or None if valid."""
    if not isinstance(prompt, str):
        return "prompt must be a string"
    if not prompt.strip():
        return "prompt must not be empty"
    if len(prompt) > 10000:
        return "prompt must be 10000 characters or less"
    if max_tokens is not None and (not isinstance(max_tokens, int) or max_tokens < 1 or max_tokens > 2048):
        return "max_tokens must be an integer between 1 and 2048"
    if temperature is not None and (not isinstance(temperature, (int, float)) or temperature < 0.01 or temperature > 5.0):
        return "temperature must be a number between 0.01 and 5.0"
    if top_k is not None and (not isinstance(top_k, int) or top_k < 1 or top_k > 100):
        return "top_k must be an integer between 1 and 100"
    if top_p is not None and (not isinstance(top_p, (int, float)) or top_p < 0.0 or top_p > 1.0):
        return "top_p must be a number between 0.0 and 1.0"
    return None


class ChatSession:
    """Manages conversation history for a chat session."""

    def __init__(self, system_prompt: str = "You are QOR, a helpful AI assistant.",
                 max_history_turns: int = 10):
        self.system_prompt = system_prompt
        self.max_history_turns = max_history_turns
        self.history = []  # list of {"role": "user"|"assistant", "content": str}

    def add_user(self, message: str):
        self.history.append({"role": "user", "content": message})
        self._trim()

    def add_assistant(self, message: str):
        self.history.append({"role": "assistant", "content": message})
        self._trim()

    def format_prompt(self) -> str:
        parts = [f"<|system|>\n{self.system_prompt}"]
        for turn in self.history:
            role = turn["role"]
            parts.append(f"<|{role}|>\n{turn['content']}")
        parts.append("<|assistant|>")
        return "\n".join(parts)

    def _trim(self):
        # Keep system + last N turns (each turn = user + assistant = 2 entries)
        max_entries = self.max_history_turns * 2
        if len(self.history) > max_entries:
            self.history = self.history[-max_entries:]


class QORServer:
    """Serves QOR model via REST API."""

    def __init__(self, config: QORConfig):
        self.config = config
        self.device = config.get_device()
        self.model = None
        self.tokenizer = None
        self.request_count = 0
        self.start_time = None
        self.chat_sessions = {}
        self.graph = None
        self.chat_store = None  # ChatStore for persistent history

    def load(self, checkpoint_path: Optional[str] = None,
             tokenizer_path: Optional[str] = None):
        """Load model and tokenizer."""
        ckpt_path = checkpoint_path or self.config.serve.model_path
        tok_path = tokenizer_path or self.config.serve.tokenizer_path

        print(f"Loading QOR model from {ckpt_path}")

        # Load tokenizer — check checkpoint dir first
        self.tokenizer = QORTokenizer()
        ckpt_dir = os.path.dirname(ckpt_path)
        donor_tok = os.path.join(ckpt_dir, "tokenizer.json")
        if os.path.exists(donor_tok):
            self.tokenizer.load(donor_tok)
        elif os.path.exists(tok_path):
            self.tokenizer.load(tok_path)
        else:
            self.tokenizer.load(tok_path)
        self.config.model.vocab_size = self.tokenizer.vocab_size

        # Load model — mmap to avoid double memory on large models
        import gc
        checkpoint = torch.load(ckpt_path, map_location=self.device,
                                weights_only=False, mmap=True)

        # Use config from checkpoint if available (handles custom model sizes)
        if "config" in checkpoint and "model" in checkpoint["config"]:
            saved = checkpoint["config"]["model"]
            from .config import ModelConfig
            model_cfg = ModelConfig()
            for k, v in saved.items():
                if hasattr(model_cfg, k):
                    setattr(model_cfg, k, v)
            model_cfg.vocab_size = self.config.model.vocab_size
            self.config.model = model_cfg

        # Reconstruct multimodal configs from checkpoint if available
        vision_config = None
        audio_config = None
        encoder_hf_configs = checkpoint.get("encoder_hf_configs", {})
        if "config" in checkpoint:
            if "vision" in checkpoint["config"]:
                from .config import VisionConfig
                vision_config = VisionConfig()
                for k, v in checkpoint["config"]["vision"].items():
                    if hasattr(vision_config, k):
                        setattr(vision_config, k, v)
            if "audio" in checkpoint["config"]:
                from .config import AudioConfig
                audio_config = AudioConfig()
                for k, v in checkpoint["config"]["audio"].items():
                    if hasattr(audio_config, k):
                        setattr(audio_config, k, v)

        self.model = QORModel(
            self.config.model,
            vision_config=vision_config,
            audio_config=audio_config,
            _from_checkpoint=True,
            _encoder_hf_configs=encoder_hf_configs,
        )
        # strict=False: allows new layers (CfC etc) not in old checkpoints
        # assign=True: avoids creating duplicate tensors (critical for 3B+)
        self.model.load_state_dict(checkpoint["model_state"],
                                   strict=False, assign=True)
        # Add modality tokens for multimodal models (no-op if already present)
        if vision_config is not None or audio_config is not None:
            num_added = self.tokenizer.ensure_modality_tokens()
            if num_added > 0 and self.tokenizer.vocab_size > self.model.config.vocab_size:
                self.model.resize_token_embeddings(self.tokenizer.vocab_size)

        self.model.to(self.device)
        self.model.eval()
        del checkpoint
        gc.collect()

        # Quantized inference (CPU only)
        if getattr(self.config.serve, 'quantize', False) and str(self.device) == 'cpu':
            self.model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8
            )
            print(f"  Quantized model (INT8) for faster CPU inference")

        # torch.compile for speedup
        if getattr(self.config.model, 'compile', False):
            mode = getattr(self.config.model, 'compile_mode', 'reduce-overhead')
            self.model.compile_model(mode=mode)

        print(f"  Parameters: {self.model.n_params:,}")
        print(f"  Device: {self.device}")
        print(f"  Ready to serve!")

    def generate(self, prompt: str,
                 max_tokens: int = None,
                 temperature: float = None,
                 top_k: int = None,
                 top_p: float = None) -> dict:
        """Generate text from a prompt."""
        if self.model is None:
            return {"error": "Model not loaded"}

        max_tokens = max_tokens or self.config.serve.max_new_tokens
        temperature = temperature or self.config.serve.temperature
        top_k = top_k or self.config.serve.top_k
        top_p = top_p or self.config.serve.top_p

        # Encode — use chat template if available
        system_prompt = getattr(self.config.serve, 'system_prompt',
                                "You are QOR (Qora Neuran AI), a helpful AI assistant.")
        if hasattr(self.tokenizer, 'format_chat'):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            ids = self.tokenizer.format_chat(messages, add_generation_prompt=True)
        else:
            ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        prompt_len = len(ids)
        input_ids = torch.tensor([ids], device=self.device)

        # Generate
        start = time.time()
        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stop_tokens=[self.tokenizer.eos_id],
        )
        elapsed = time.time() - start

        # Decode only the generated tokens (skip prompt)
        new_ids = output_ids[prompt_len:]
        generated_text = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        new_tokens = len(new_ids)

        self.request_count += 1

        return {
            "prompt": prompt,
            "output": generated_text,
            "generated_text": generated_text,
            "tokens_generated": new_tokens,
            "time_seconds": round(elapsed, 3),
            "tokens_per_second": round(new_tokens / max(elapsed, 0.001), 1),
        }

    def generate_stream(self, prompt: str,
                        max_tokens: int = None,
                        temperature: float = None,
                        top_k: int = None,
                        top_p: float = None):
        """Generate text token-by-token as a generator (for SSE streaming)."""
        if self.model is None:
            return

        max_tokens = max_tokens or self.config.serve.max_new_tokens
        temperature = temperature or self.config.serve.temperature
        top_k = top_k or self.config.serve.top_k
        top_p = top_p or self.config.serve.top_p

        ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = torch.tensor([ids], device=self.device)

        self.model.eval()
        with torch.no_grad():
            result = self.model.forward(input_ids, enable_self_mod=False, use_cache=True)
            kv_cache = result["kv_cache"]
            logits = result["logits"][:, -1, :]

            generated = list(ids)
            prev_text = self.tokenizer.decode(generated, skip_special_tokens=True)

            for _ in range(max_tokens):
                logits = logits / max(temperature, 1e-8)

                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')

                probs = torch.nn.functional.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                token_id = next_token.item()
                generated.append(token_id)

                eos_id = getattr(self.tokenizer, 'eos_id', None)
                if eos_id is not None and token_id == eos_id:
                    break

                # Decode and yield the new text delta
                full_text = self.tokenizer.decode(generated, skip_special_tokens=True)
                delta = full_text[len(prev_text):]
                prev_text = full_text
                if delta:
                    yield delta

                result = self.model.forward(next_token, enable_self_mod=False,
                                            kv_cache=kv_cache, use_cache=True)
                kv_cache = result["kv_cache"]
                logits = result["logits"][:, -1:, :].squeeze(1)

    def get_info(self) -> dict:
        """Return model information."""
        info = {
            "name": "QOR — The Qore Mind",
            "parameters": self.model.n_params if self.model else 0,
            "vocab_size": self.tokenizer.vocab_size if self.tokenizer else 0,
            "device": str(self.device),
            "requests_served": self.request_count,
            "cms_levels": self.config.model.cms_levels,
            "self_mod_enabled": self.config.model.self_mod_lr > 0,
        }
        if self.graph and self.graph.is_open:
            try:
                info["graph"] = self.graph.stats()
            except Exception:
                info["graph"] = {"error": "stats unavailable"}
        return info


# ========================================================================
# MODULE-LEVEL HELPERS (shared by run_full_server + start_api_thread)
# ========================================================================

_price_cache = {}  # {pair: (price, timestamp)}
_PRICE_TTL = 60    # seconds


def _cached_price(client, pair):
    """Get price from cache or Binance (max once per 60s per pair)."""
    now = time.time()
    cached = _price_cache.get(pair)
    if cached and (now - cached[1]) < _PRICE_TTL:
        return cached[0]
    try:
        p = client.get_price(pair)
        _price_cache[pair] = (p, now)
        return p
    except Exception:
        return cached[0] if cached else 0


def _enrich_open_trade(t, client):
    """Add current_price, opened_at (ISO), duration, unrealized_pnl to an open trade dict."""
    sym = t["symbol"]
    pair = sym if sym.endswith("USDT") else f"{sym}USDT"
    t["current_price"] = _cached_price(client, pair)
    entry_us = t.get("entry_time", 0)
    if entry_us > 0:
        from datetime import datetime, timezone
        entry_dt = datetime.fromtimestamp(entry_us / 1_000_000, tz=timezone.utc)
        t["opened_at"] = entry_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        now = datetime.now(timezone.utc)
        delta = now - entry_dt
        hours = delta.total_seconds() / 3600
        if hours < 1:
            t["duration"] = f"{int(delta.total_seconds() / 60)}m"
        elif hours < 24:
            t["duration"] = f"{hours:.1f}h"
        else:
            t["duration"] = f"{hours / 24:.1f}d"
    else:
        t["opened_at"] = ""
        t["duration"] = ""
    # Unrealized PnL
    entry_price = t.get("entry_price", 0)
    qty = t.get("quantity", 0)
    cur = t["current_price"]
    if entry_price > 0 and cur > 0 and qty > 0:
        direction = t.get("direction", "LONG")
        if direction == "SHORT":
            t["unrealized_pnl"] = round((entry_price - cur) * qty, 4)
        else:
            t["unrealized_pnl"] = round((cur - entry_price) * qty, 4)
    else:
        t["unrealized_pnl"] = 0.0
    return t


def _build_identity(profile, gate):
    """Build system prompt identity from user profile."""
    user_name = profile.get("user_name", "") if profile else ""
    identity = ("You are QOR (Qora Neuran AI), a helpful AI assistant with access to "
                "real-time tools, a knowledge graph, and memory systems. "
                "Answer questions accurately using your available context.")
    if user_name:
        identity = (f"You are QOR (Qora Neuran AI), a personal AI assistant for {user_name}. "
                     + identity[len("You are QOR (Qora Neuran AI), "):])
    if profile and profile.get("interests"):
        interests_str = ", ".join(str(k) for k in profile["interests"].keys())
        identity += f"\n\nThe user's interests include: {interests_str}. Use this to provide more relevant answers."
    if profile and profile.get("cautions"):
        identity += f"\nImportant cautions for this user: {', '.join(str(c) for c in profile['cautions'])}."
    detail = profile.get("preferred_detail_level", "detailed") if profile else "detailed"
    if detail and detail != "detailed":
        identity += f"\nThe user prefers {detail} responses."
    identity += ("\n\nYou have access to: user profile, chat history, memory database, "
                 "knowledge graph, RAG, tool cache, and 52+ real-time tools. "
                 "When answering, always provide comprehensive analysis with real data.")
    if getattr(gate.model, 'vision_encoder', None) is not None:
        identity += (" You CAN see and analyze images. When the user provides an image file path, "
                      "you will see the image contents. Describe what you see in detail.")
    if getattr(gate.model, 'audio_encoder', None) is not None:
        identity += (" You CAN hear and analyze audio. When the user provides an audio file path, "
                      "you will hear the audio contents.")
    return identity


# ========================================================================
# DASHBOARD ROUTE REGISTRATION (trading terminal endpoints)
# ========================================================================

def _register_dashboard_routes(app, config, runtime):
    """Register dashboard-only API routes (CORTEX, multi-TF, activity, analytics).

    Called by _register_routes() so these work in both run_full_server and
    start_api_thread.
    """
    from flask import request, jsonify

    # --- Helpers ---

    def _get_configured_symbols(engine_type='any'):
        """Get configured trading symbols from engines or config."""
        syms = set()
        if engine_type in ('any', 'spot'):
            eng = runtime._trading_engine
            if eng and hasattr(eng, 'config') and hasattr(eng.config, 'symbols'):
                syms.update(eng.config.symbols)
        if engine_type in ('any', 'futures'):
            eng = runtime._futures_engine
            if eng and hasattr(eng, 'config') and hasattr(eng.config, 'symbols'):
                syms.update(eng.config.symbols)
        if not syms:
            if hasattr(config, 'trading') and hasattr(config.trading, 'symbols'):
                syms.update(config.trading.symbols)
            if hasattr(config, 'futures') and hasattr(config.futures, 'symbols'):
                syms.update(config.futures.symbols)
        return sorted(syms)

    def _compute_indicators_json(ohlc):
        """Compute all TA indicators from OHLC dict, return structured JSON."""
        from qor.tools import (
            _calc_rsi, _calc_ema, _calc_ema_series, _calc_macd,
            _calc_bollinger, _calc_atr, _calc_adx, _find_support_resistance,
        )
        closes = ohlc["closes"]
        highs = ohlc["highs"]
        lows = ohlc["lows"]
        current = closes[-1] if closes else 0

        rsi_val = _calc_rsi(closes, 14)
        macd_val, signal_val, hist_val = _calc_macd(closes)
        bb_mid, bb_upper, bb_lower = _calc_bollinger(closes, 20, 2.0)
        atr_val = _calc_atr(highs, lows, closes, 14)
        adx_val = _calc_adx(highs, lows, closes, 14)
        ema21 = _calc_ema(closes, 21)
        ema50 = _calc_ema(closes, 50)
        ema200 = _calc_ema(closes, 200) if len(closes) >= 200 else _calc_ema(closes, len(closes))

        # Series for charts
        rsi_series = []
        for i in range(14, len(closes)):
            rsi_series.append(_calc_rsi(closes[:i+1], 14))

        ema21_s = _calc_ema_series(closes, 21)
        ema50_s = _calc_ema_series(closes, 50)

        macd_line_series = []
        ema12 = _calc_ema_series(closes, 12)
        ema26 = _calc_ema_series(closes, 26)
        if len(closes) >= 26:
            macd_raw = [ema12[i] - ema26[i] for i in range(len(closes))]
            sig_s = _calc_ema_series(macd_raw[25:], 9)
            for i in range(len(sig_s)):
                macd_line_series.append({
                    "macd": round(macd_raw[25 + i], 4),
                    "signal": round(sig_s[i], 4),
                    "hist": round(macd_raw[25 + i] - sig_s[i], 4),
                })

        supports, resistances = _find_support_resistance(highs, lows, closes)

        rsi_tag = "oversold" if rsi_val < 30 else "overbought" if rsi_val > 70 else "neutral"
        return {
            "symbol": ohlc.get("name", ""),
            "current_price": current,
            "rsi": {"value": round(rsi_val, 2), "series": [round(v, 2) for v in rsi_series], "interpretation": rsi_tag},
            "macd": {"line": round(macd_val, 4), "signal": round(signal_val, 4), "histogram": round(hist_val, 4), "series": macd_line_series},
            "ema": {"ema21": round(ema21, 2), "ema50": round(ema50, 2), "ema200": round(ema200, 2),
                    "series": {"ema21": [round(v, 2) for v in ema21_s], "ema50": [round(v, 2) for v in ema50_s]}},
            "bollinger": {"upper": round(bb_upper, 2), "middle": round(bb_mid, 2), "lower": round(bb_lower, 2)},
            "atr": {"value": round(atr_val, 2), "pct": round(atr_val / current * 100, 2) if current else 0},
            "adx": {"value": round(adx_val, 2)},
            "supports": [round(s, 2) for s in supports],
            "resistances": [round(r, 2) for r in resistances],
        }

    def _get_exchange_open_positions():
        """Fetch open positions directly from Binance exchange API."""
        positions = []
        # Futures positions from Binance API
        feng = runtime._futures_engine
        if feng and hasattr(feng, 'client'):
            try:
                raw = feng.client.get_positions()
                for p in (raw or []):
                    amt = float(p.get("positionAmt", 0))
                    if abs(amt) < 1e-9:
                        continue
                    sym_raw = p.get("symbol", "")
                    sym = sym_raw.replace("USDT", "") if sym_raw.endswith("USDT") else sym_raw
                    entry_price = float(p.get("entryPrice", 0))
                    mark_price = float(p.get("markPrice", 0))
                    unrealized = float(p.get("unRealizedProfit", 0))
                    lev = int(p.get("leverage", 1))
                    direction = "LONG" if amt > 0 else "SHORT"
                    positions.append({
                        "symbol": sym, "direction": direction,
                        "entry_price": entry_price, "current_price": mark_price,
                        "quantity": abs(amt), "unrealized_pnl": unrealized,
                        "leverage": lev, "engine": "FUTURES",
                    })
            except Exception:
                pass
        # Spot open trades from store
        seng = runtime._trading_engine
        if seng and hasattr(seng, 'store'):
            for t in seng.store.trades.values():
                if t.get("status") == "open":
                    sym = t.get("symbol", "")
                    pair = sym if sym.endswith("USDT") else f"{sym}USDT"
                    try:
                        cp = _cached_price(seng.client, pair)
                    except Exception:
                        cp = 0
                    ep = t.get("entry_price", 0)
                    qty = t.get("quantity", 0)
                    upnl = (cp - ep) * qty if cp and ep else 0
                    positions.append({
                        "symbol": sym, "direction": "LONG",
                        "entry_price": ep, "current_price": cp,
                        "quantity": qty, "unrealized_pnl": upnl,
                        "leverage": 1, "engine": "SPOT",
                    })
        return positions

    # --- Routes ---

    @app.route('/api/trading/symbols', methods=['GET'])
    def api_trading_symbols():
        """Get configured trading symbols (optionally per engine)."""
        engine_filter = request.args.get('engine', 'any')
        spot_syms = list(config.trading.symbols) if hasattr(config, 'trading') else []
        futures_syms = list(config.futures.symbols) if hasattr(config, 'futures') else []
        return jsonify({
            "symbols": _get_configured_symbols(engine_filter),
            "spot": spot_syms,
            "futures": futures_syms,
        })

    @app.route('/api/trading/symbols', methods=['POST'])
    def api_trading_symbols_set():
        """Set which symbols to trade.

        Body: {"symbols": ["BTC", "ETH", "SOL"], "engine": "both"}
        engine: "spot", "futures", or "both" (default)
        """
        data = request.get_json()
        if not data:
            return jsonify({"error": "Missing request body"}), 400

        symbols = data.get('symbols')
        if not symbols or not isinstance(symbols, list):
            return jsonify({"error": "'symbols' must be a non-empty list"}), 400

        # Normalize: uppercase, deduplicate, strip whitespace
        symbols = list(dict.fromkeys(
            s.strip().upper() for s in symbols if isinstance(s, str) and s.strip()
        ))
        if not symbols:
            return jsonify({"error": "No valid symbols provided"}), 400

        engine = data.get('engine', 'both').lower()
        updated = []

        if engine in ('spot', 'both'):
            config.trading.symbols = symbols
            eng = runtime._trading_engine
            if eng and hasattr(eng, 'config'):
                eng.config.symbols = symbols
            updated.append('spot')

        if engine in ('futures', 'both'):
            config.futures.symbols = symbols
            eng = runtime._futures_engine
            if eng and hasattr(eng, 'config'):
                eng.config.symbols = symbols
                # Re-run leverage/margin setup for new symbols
                eng._leverage_set = False
                try:
                    eng._setup_symbols()
                except Exception:
                    pass
            updated.append('futures')

        if not updated:
            return jsonify({"error": f"Unknown engine: {engine}"}), 400

        # Persist to profile.json
        saved = False
        try:
            if profile is not None:
                profile["trading_symbols"] = {
                    "spot": list(config.trading.symbols),
                    "futures": list(config.futures.symbols),
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
            if profile_path:
                with open(profile_path, 'w') as pf:
                    json.dump(profile, pf, indent=2)
            saved = True
        except Exception:
            pass

        return jsonify({
            "ok": True,
            "symbols": symbols,
            "engines_updated": updated,
            "saved": saved,
            "message": f"Now trading {', '.join(symbols)} on {', '.join(updated)}",
        })

    @app.route('/api/chart/indicators')
    def api_chart_indicators():
        from qor.tools import (
            _fetch_ohlc_binance, _fetch_ohlc_yahoo,
            _COMMODITY_YAHOO, _FOREX_YAHOO, _extract_stock_symbol,
        )
        from qor.quant import classify_asset
        symbol = request.args.get('symbol', 'BTC')
        period = request.args.get('period', '24h')
        period_map = {"1h": ("5m", 60), "4h": ("15m", 100), "24h": ("1h", 100),
                      "7d": ("4h", 100), "30d": ("1d", 30)}
        yahoo_period_map = {"1h": ("1m", "1d"), "4h": ("5m", "1d"), "24h": ("15m", "5d"),
                            "7d": ("60m", "7d"), "30d": ("1d", "1mo")}
        interval, limit = period_map.get(period, ("1h", 100))
        y_interval, y_range = yahoo_period_map.get(period, ("15m", "5d"))
        # Classify asset first — don't blindly try Binance for everything
        asset = classify_asset(symbol)
        ohlc = None
        if asset.asset_type == "crypto":
            ohlc = _fetch_ohlc_binance(symbol, interval, limit)
        else:
            # Commodity, forex, or stock → Yahoo Finance
            q = symbol.lower().strip()
            yahoo_sym = _COMMODITY_YAHOO.get(q) or _FOREX_YAHOO.get(q)
            if not yahoo_sym:
                yahoo_sym = _extract_stock_symbol(symbol)
            if yahoo_sym:
                try:
                    ohlc = _fetch_ohlc_yahoo(yahoo_sym, y_interval, y_range)
                except Exception:
                    ohlc = None
        if not ohlc:
            return jsonify({"error": "No OHLC data available"}), 404
        result = _compute_indicators_json(ohlc)
        result["period"] = period
        return jsonify(result)

    @app.route('/api/trading/analysis/<symbol>')
    def api_trading_analysis(symbol):
        from qor.tools import (
            _fetch_ohlc_binance, _calc_rsi, _calc_ema, _calc_macd,
            _calc_bollinger, _calc_atr, _calc_adx, _calc_trade_levels,
            _find_support_resistance, _compute_tf_line, _score_tf,
            _fetch_ohlc_yahoo, _COMMODITY_YAHOO, _FOREX_YAHOO, _extract_stock_symbol,
        )
        from qor.quant import classify_asset
        # Classify asset FIRST to avoid crypto/stock confusion
        asset = classify_asset(symbol)
        is_crypto = asset.asset_type == "crypto"

        ohlc = None
        yahoo_sym = None
        if is_crypto:
            ohlc = _fetch_ohlc_binance(symbol, "1d", 25)
        else:
            # Yahoo — commodity, forex, or stock (with Indian .NS mapping)
            q = symbol.lower()
            yahoo_sym = _COMMODITY_YAHOO.get(q) or _FOREX_YAHOO.get(q)
            if not yahoo_sym:
                yahoo_sym = _extract_stock_symbol(symbol)
            try:
                ohlc = _fetch_ohlc_yahoo(yahoo_sym, "1d", "6mo")
            except Exception:
                ohlc = None

        if not ohlc:
            from qor.quant import classify_asset, is_market_open
            asset = classify_asset(symbol)
            market = is_market_open(asset.asset_type)
            if not market["open"]:
                return jsonify({
                    "error": f"No data for {symbol}",
                    "market_closed": True,
                    "message": f"{asset.asset_type.title()} market is closed — {market['reason']}. "
                               f"Try again when market reopens ({market.get('next_open', 'next session')}).",
                }), 200
            return jsonify({"error": f"No data for {symbol}"}), 404

        closes = ohlc["closes"]
        highs = ohlc["highs"]
        lows = ohlc["lows"]
        current = closes[-1]

        rsi = _calc_rsi(closes, 14)
        ema21 = _calc_ema(closes, 21)
        ema50 = _calc_ema(closes, 50)
        ema200 = _calc_ema(closes, 200) if len(closes) >= 200 else _calc_ema(closes, len(closes))
        macd_val, signal_val, hist_val = _calc_macd(closes)
        bb_mid, bb_upper, bb_lower = _calc_bollinger(closes, 20, 2.0)
        atr = _calc_atr(highs, lows, closes, 14)
        adx = _calc_adx(highs, lows, closes, 14)
        supports, resistances = _find_support_resistance(highs, lows, closes)
        levels = _calc_trade_levels(current, atr, ema21, ema50, rsi, bb_upper, bb_lower, supports, resistances)

        # Multi-TF confluence — full scoring with ALL indicators via _score_tf()
        tf_results = []
        if is_crypto:
            tf_map = [("W", "1w", 100), ("D", "1d", 200), ("4H", "4h", 200),
                      ("1H", "1h", 200), ("15m", "15m", 200), ("5m", "5m", 200)]
        else:
            tf_map = [("W", "1wk", "2y"), ("D", "1d", "6mo"), ("1H", "60m", "5d"),
                      ("15m", "15m", "1d"), ("5m", "5m", "1d")]

        bullish_tfs = 0
        bearish_tfs = 0
        total_tfs = 0
        total_score = 0
        for tf_info in tf_map:
            tf_name = tf_info[0]
            try:
                if is_crypto:
                    tf_ohlc = _fetch_ohlc_binance(symbol, tf_info[1], tf_info[2])
                else:
                    tf_ohlc = _fetch_ohlc_yahoo(yahoo_sym, tf_info[1], tf_info[2])
                if tf_ohlc and len(tf_ohlc["closes"]) >= 20:
                    total_tfs += 1
                    _, tf_stats = _compute_tf_line(tf_ohlc, tf_name)
                    tf_score = tf_stats.get("tf_score", 0)
                    total_score += tf_score
                    tf_bias = "BULLISH" if tf_score > 10 else "BEARISH" if tf_score < -10 else "NEUTRAL"
                    if tf_score > 10:
                        bullish_tfs += 1
                    elif tf_score < -10:
                        bearish_tfs += 1
                    tf_results.append({
                        "name": tf_name, "bias": tf_bias, "score": tf_score,
                        "rsi": round(tf_stats.get("rsi", 50), 1),
                        "ema_trend": "above" if tf_stats["current"] > tf_stats["ema21"] else "below",
                        "macd_trend": "bullish" if tf_stats.get("macd_hist", 0) > 0 else "bearish",
                        "adx": round(tf_stats.get("adx", 0), 1),
                    })
            except Exception:
                pass
        if bullish_tfs > bearish_tfs:
            overall_bias = "BULLISH"
        elif bearish_tfs > bullish_tfs:
            overall_bias = "BEARISH"
        else:
            overall_bias = "NEUTRAL"

        result = {
            "symbol": symbol, "timestamp": time.time(),
            "bias": overall_bias,
            "bullish_tfs": bullish_tfs, "bearish_tfs": bearish_tfs, "total_tfs": total_tfs, "total_score": total_score,
            "entry": round(levels["entry"], 2), "stop_loss": round(levels["stop_loss"], 2),
            "tp1": round(levels["tp1"], 2), "tp2": round(levels["tp2"], 2),
            "risk_reward": round(levels["risk_reward"], 2),
            "rsi": round(rsi, 2), "ema21": round(ema21, 2), "ema50": round(ema50, 2), "ema200": round(ema200, 2),
            "macd_hist": round(hist_val, 4), "bb_upper": round(bb_upper, 2), "bb_lower": round(bb_lower, 2),
            "adx": round(adx, 2),
            "supports": [round(s, 2) for s in supports],
            "resistances": [round(r, 2) for r in resistances],
            "timeframes": tf_results,
        }

        # Append CORTEX signal if available (check ALL engines)
        _all_engines = [runtime._futures_engine, runtime._trading_engine]
        _all_engines += list(getattr(runtime, '_exchange_engines', {}).values())
        for eng in _all_engines:
            if eng and hasattr(eng, 'manager') and hasattr(eng.manager, 'cortex') and eng.manager.cortex:
                cached = eng.manager.cortex._last_results.get(symbol)
                if cached:
                    result["cortex"] = cached
                    break

        return jsonify(result)

    @app.route('/api/trading/activity')
    def api_trading_activity():
        limit = int(request.args.get('limit', 100))
        engine_type = request.args.get('engine', 'any')
        activity = []
        if engine_type in ('any', 'spot'):
            eng = runtime._trading_engine
            if eng:
                for a in eng._activity_log:
                    entry = dict(a)
                    entry["engine"] = "SPOT"
                    activity.append(entry)
        if engine_type in ('any', 'futures'):
            eng = runtime._futures_engine
            if eng:
                for a in eng._activity_log:
                    entry = dict(a)
                    entry["engine"] = "FUTURES"
                    activity.append(entry)
        # Sort by time descending, limit
        activity.sort(key=lambda x: x.get("time", ""), reverse=True)
        return jsonify(activity[:limit])

    @app.route('/api/trading/performance/<symbol>')
    def api_trading_performance(symbol):
        n = int(request.args.get('n', 20))
        all_trades = []
        for label, eng in [("spot", runtime._trading_engine), ("futures", runtime._futures_engine)]:
            if eng and hasattr(eng, 'store'):
                for t in eng.store.trades.values():
                    if t.get("symbol") == symbol:
                        td = dict(t)
                        td["engine"] = label
                        all_trades.append(td)
        closed = [t for t in all_trades if t.get("status") != "open"]
        recent = closed[-n:]
        wins = [t for t in recent if (t.get("pnl") or 0) > 0]
        losses = [t for t in recent if (t.get("pnl") or 0) <= 0]
        return jsonify({
            "symbol": symbol, "trades": len(recent),
            "wins": len(wins), "losses": len(losses),
            "win_rate": round(len(wins) / len(recent) * 100, 1) if recent else 0,
            "total_pnl": round(sum(t.get("pnl", 0) for t in recent), 2),
            "avg_pnl_pct": round(sum(t.get("pnl_pct", 0) for t in recent) / len(recent), 2) if recent else 0,
            "best_pct": round(max((t.get("pnl_pct", 0) for t in recent), default=0), 2),
            "worst_pct": round(min((t.get("pnl_pct", 0) for t in recent), default=0), 2),
        })

    @app.route('/api/chart/trades-overlay')
    def api_chart_trades_overlay():
        symbol = request.args.get('symbol', 'BTC')
        entries, exits = [], []
        for label, eng in [("spot", runtime._trading_engine), ("futures", runtime._futures_engine)]:
            if eng and hasattr(eng, 'store'):
                for t in eng.store.trades.values():
                    if t.get("symbol") != symbol:
                        continue
                    if t.get("entry_time") and t.get("entry_price"):
                        entries.append({
                            "time": t["entry_time"], "price": t["entry_price"],
                            "direction": t.get("direction", t.get("side", "LONG")),
                        })
                    if t.get("exit_time") and t.get("exit_price") and t.get("status") != "open":
                        exits.append({
                            "time": t["exit_time"], "price": t["exit_price"],
                            "pnl": t.get("pnl", 0), "status": t.get("status", ""),
                        })
        return jsonify({"symbol": symbol, "entries": entries, "exits": exits})

    @app.route('/api/chart/distribution')
    def api_chart_distribution():
        all_trades = []
        for label, eng in [("spot", runtime._trading_engine), ("futures", runtime._futures_engine)]:
            if eng and hasattr(eng, 'store'):
                for t in eng.store.trades.values():
                    td = dict(t)
                    td["engine"] = label
                    all_trades.append(td)
        # Include open positions from exchange
        open_positions = _get_exchange_open_positions()
        closed = [t for t in all_trades if t.get("status") != "open"]
        wins = [t for t in closed if (t.get("pnl") or 0) > 0]
        losses = [t for t in closed if (t.get("pnl") or 0) <= 0]

        by_dir = {}
        for t in all_trades:
            d = (t.get("direction") or t.get("side") or "LONG").upper()
            if d not in by_dir:
                by_dir[d] = {"wins": 0, "losses": 0, "pnl": 0, "open": 0}
            if t.get("status") == "open":
                by_dir[d]["open"] += 1
            elif (t.get("pnl") or 0) > 0:
                by_dir[d]["wins"] += 1
            else:
                by_dir[d]["losses"] += 1
            by_dir[d]["pnl"] = round(by_dir[d]["pnl"] + (t.get("pnl") or 0), 2)
        # Count exchange open positions by direction too
        for p in open_positions:
            d = p.get("direction", "LONG")
            if d not in by_dir:
                by_dir[d] = {"wins": 0, "losses": 0, "pnl": 0, "open": 0}

        by_status = {}
        for t in all_trades:
            s = t.get("status", "unknown")
            by_status[s] = by_status.get(s, 0) + 1
        # Count exchange open positions
        open_count = len(open_positions) + sum(1 for t in all_trades if t.get("status") == "open")

        return jsonify({
            "win_loss_pie": {
                "wins": len(wins), "losses": len(losses),
                "open": open_count,
            },
            "by_direction": by_dir,
            "by_status": by_status,
        })

    @app.route('/api/chart/by-symbol')
    def api_chart_by_symbol():
        all_trades = []
        for label, eng in [("spot", runtime._trading_engine), ("futures", runtime._futures_engine)]:
            if eng and hasattr(eng, 'store'):
                for t in eng.store.trades.values():
                    td = dict(t)
                    td["engine"] = label
                    all_trades.append(td)
        # Include exchange positions
        open_positions = _get_exchange_open_positions()

        by_sym = {}
        for t in all_trades:
            s = t.get("symbol", "?")
            if s not in by_sym:
                by_sym[s] = {"symbol": s, "trades": 0, "wins": 0, "losses": 0, "total_pnl": 0, "open": 0}
            by_sym[s]["trades"] += 1
            if t.get("status") == "open":
                by_sym[s]["open"] += 1
            elif (t.get("pnl") or 0) > 0:
                by_sym[s]["wins"] += 1
            else:
                by_sym[s]["losses"] += 1
            by_sym[s]["total_pnl"] += t.get("pnl", 0)
        for p in open_positions:
            s = p.get("symbol", "?")
            if s not in by_sym:
                by_sym[s] = {"symbol": s, "trades": 0, "wins": 0, "losses": 0, "total_pnl": 0, "open": 0}
            # Add unrealized P&L
            by_sym[s]["total_pnl"] += p.get("unrealized_pnl", 0)

        symbols = []
        for d in by_sym.values():
            total = d["wins"] + d["losses"]
            d["win_rate"] = round(d["wins"] / total * 100, 1) if total > 0 else 0
            d["total_pnl"] = round(d["total_pnl"], 2)
            symbols.append(d)
        symbols.sort(key=lambda x: x["total_pnl"], reverse=True)
        return jsonify({"symbols": symbols})

    @app.route('/api/chart/equity')
    def api_chart_equity():
        all_trades = []
        for label, eng in [("spot", runtime._trading_engine), ("futures", runtime._futures_engine)]:
            if eng and hasattr(eng, 'store'):
                for t in eng.store.trades.values():
                    if t.get("status") != "open" and t.get("exit_time"):
                        all_trades.append(t)
        all_trades.sort(key=lambda x: x.get("exit_time", 0))

        initial = 10000.0
        balance = initial
        series = []
        max_bal = initial
        max_dd = 0

        for t in all_trades:
            balance += t.get("pnl", 0)
            max_bal = max(max_bal, balance)
            dd = (max_bal - balance) / max_bal * 100 if max_bal > 0 else 0
            max_dd = max(max_dd, dd)
            series.append({
                "time": t.get("exit_time", 0),
                "balance": round(balance, 2),
                "drawdown_pct": round(dd, 2),
            })

        # Add unrealized P&L from open exchange positions
        open_positions = _get_exchange_open_positions()
        unrealized = sum(p.get("unrealized_pnl", 0) for p in open_positions)

        return jsonify({
            "initial_balance": initial,
            "series": series,
            "max_drawdown_pct": round(max_dd, 2),
            "current_balance": round(balance, 2),
            "unrealized_pnl": round(unrealized, 2),
        })

    @app.route('/api/trading/cortex/<symbol>')
    def api_trading_cortex(symbol):
        try:
            _all_engines = [runtime._futures_engine, runtime._trading_engine]
            _all_engines += list(getattr(runtime, '_exchange_engines', {}).values())
            for eng in _all_engines:
                if eng and hasattr(eng, 'manager') and hasattr(eng.manager, 'cortex') and eng.manager.cortex:
                    cortex = eng.manager.cortex
                    cached = cortex._last_results.get(symbol)
                    if cached:
                        result = dict(cached)
                        result["symbol"] = symbol
                        result["observation_dim"] = cortex.OBS_DIM
                        return jsonify(result)
            # No CORTEX data — check why
            from qor.quant import classify_asset, is_market_open
            asset = classify_asset(symbol)
            market = is_market_open(asset.asset_type)
            if not market["open"]:
                return jsonify({
                    "symbol": symbol,
                    "available": False,
                    "market_closed": True,
                    "message": f"{asset.asset_type.title()} market is closed — {market['reason']}. "
                               f"CORTEX data available when market reopens ({market.get('next_open', 'next session')}).",
                }), 200
            # Market open but no engine running for this symbol
            return jsonify({
                "symbol": symbol,
                "available": False,
                "market_closed": False,
                "message": f"No trading engine active for {symbol}. "
                           f"Start an engine with this symbol to get CORTEX signals.",
            }), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500


# ========================================================================
# SHARED ROUTE REGISTRATION (called by both run_full_server + start_api_thread)
# ========================================================================

def _register_routes(app, config, runtime, learner, gate, graph, rag,
                     cache_store, chat_store, plugin_mgr, skill_loader,
                     tool_executor, crypto, profile, profile_path, server):
    """Register ALL shared API routes on the Flask app.

    Both run_full_server() and start_api_thread() call this once.
    """
    from flask import request, jsonify, Response, send_file

    # Resolve data_dir to absolute path once so it works regardless of CWD
    _data_dir = os.path.abspath(config.runtime.data_dir)

    def _browser_available():
        try:
            from qor.browser import SUPPORTED_BROWSERS
            return list(SUPPORTED_BROWSERS.keys())
        except Exception:
            return []

    def _browser_default():
        try:
            from qor.browser import _default_browser
            return _default_browser
        except Exception:
            return "chrome"

    # ---- Health / Info ----

    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({"status": "ok", "model_loaded": True, "full_runtime": True})

    @app.route('/info', methods=['GET'])
    def info():
        return jsonify(server.get_info())

    # ---- Status ----

    @app.route('/api/status', methods=['GET'])
    def api_status():
        s = runtime.status()
        s["model_parameters"] = getattr(learner.model, 'n_params', 0) if learner.model else 0
        s["device"] = str(config.get_device())
        s["uptime_seconds"] = round(time.time() - server.start_time, 1) if server.start_time else 0
        return jsonify(s)

    @app.route('/api/tools', methods=['GET'])
    def api_tools():
        tools = []
        if plugin_mgr and hasattr(plugin_mgr, 'tools'):
            for name, t in plugin_mgr.tools.items():
                tools.append({"name": name, "description": getattr(t, 'description', ''),
                              "enabled": getattr(t, 'enabled', True)})
        elif hasattr(gate, 'tools') and gate.tools:
            for t in gate.tools.list_tools():
                tools.append(t)
        return jsonify(tools)

    @app.route('/api/skills', methods=['GET'])
    def api_skills():
        skills = []
        if skill_loader and skill_loader.skills:
            for name, s in sorted(skill_loader.skills.items()):
                skills.append({"name": name, "description": s.description,
                              "keywords": getattr(s, 'keywords', [])})
        return jsonify(skills)

    # ---- NGRE Health ----

    @app.route('/api/graph-health', methods=['GET'])
    def api_graph_health():
        """Graph health metrics from NGRE GraphHealthMonitor."""
        result = {}
        if hasattr(runtime, '_health_monitor') and runtime._health_monitor:
            try:
                result = runtime._health_monitor.check_all()
            except Exception as e:
                result = {"error": str(e)}
        else:
            result = {"status": "health monitor not available"}
        return jsonify(result)

    @app.route('/api/ngre-status', methods=['GET'])
    def api_ngre_status():
        """NGRE brain status and routing statistics."""
        result = {"ngre_available": False}
        if gate and hasattr(gate, '_ngre_brain') and gate._ngre_brain is not None:
            result["ngre_available"] = True
            result["treegate_active"] = gate._treegate is not None
        if gate and hasattr(gate, '_routing_stats'):
            result["routing_stats"] = gate.get_routing_stats()
        # Source reliability
        if hasattr(runtime, '_source_reliability') and runtime._source_reliability:
            try:
                result["source_reliability"] = runtime._source_reliability.get_all_stats()
            except Exception:
                pass
        # Compressor
        if hasattr(runtime, '_compressor') and runtime._compressor:
            try:
                result["compressor"] = runtime._compressor.stats()
            except Exception:
                pass
        # Cold tier
        if hasattr(runtime, '_cold_tier') and runtime._cold_tier:
            try:
                result["cold_tier"] = runtime._cold_tier.stats()
            except Exception:
                pass
        return jsonify(result)

    @app.route('/api/routing-stats', methods=['GET'])
    def api_routing_stats():
        """ComplexityGate routing statistics."""
        if gate and hasattr(gate, 'get_routing_stats'):
            return jsonify(gate.get_routing_stats())
        return jsonify({"error": "routing stats not available"})

    # ---- Command execution ----

    @app.route('/api/command', methods=['POST'])
    def api_command():
        data = request.get_json()
        if not data or 'command' not in data:
            return jsonify({"error": "Missing 'command'"}), 400
        cmd = data['command'].strip().lower()
        result_lines = []
        if cmd == "status":
            s = runtime.status()
            result_lines.append(f"Memory entries: {s.get('memory_entries', 'N/A')}")
            result_lines.append(f"Cache entries: {s.get('cache_entries', 'N/A')}")
            result_lines.append(f"Chat messages: {s.get('chat_messages', 'N/A')}")
            result_lines.append(f"Chat sessions: {s.get('chat_sessions', 'N/A')}")
            result_lines.append(f"Graph nodes: {s.get('graph_nodes', 'N/A')}")
            result_lines.append(f"Graph edges: {s.get('graph_edges', 'N/A')}")
            result_lines.append(f"RAG chunks: {s.get('rag_chunks', 'N/A')}")
            result_lines.append(f"Historical: {s.get('historical_entries', 'N/A')}")
            result_lines.append(f"Last cleanup: {s.get('last_cleanup') or 'never'}")
            ts = s.get("trading")
            if ts:
                stats = ts.get("stats", {})
                result_lines.append(f"Trading: ACTIVE — {ts.get('open_positions', 0)} open, "
                                    f"P&L ${stats.get('total_pnl_usdt', 0):+,.2f}")
            else:
                result_lines.append("Trading: disabled")
            fs = s.get("futures")
            if fs:
                fstats = fs.get("stats", {})
                result_lines.append(f"Futures: ACTIVE {fs.get('leverage', '?')}x — "
                                    f"{fs.get('open_positions', 0)} open, "
                                    f"P&L ${fstats.get('total_pnl_usdt', 0):+,.2f}")
            else:
                result_lines.append("Futures: disabled")
        elif cmd == "consolidate":
            result = runtime.cleanup_now()
            result_lines.append(f"Status: {result.get('status')}")
            result_lines.append(f"Memory removed: {result.get('memory_removed', 0)}")
            result_lines.append(f"Graph removed: {result.get('graph_removed', 0)}")
        elif cmd == "memory":
            if gate.memory:
                count = gate.memory.count() if hasattr(gate.memory, 'count') else 0
                result_lines.append(f"Memory entries: {count}")
            else:
                result_lines.append("Memory not available")
        elif cmd == "cache":
            if cache_store:
                result_lines.append(f"Cache entries: {cache_store.count()}")
            else:
                result_lines.append("Cache not available")
        elif cmd == "tools":
            if hasattr(gate, 'tools') and gate.tools:
                for t in gate.tools.list_tools():
                    result_lines.append(f"  {t['name']}: {t['description']}")
            else:
                result_lines.append("No tools loaded")
        elif cmd == "skills":
            if skill_loader and skill_loader.skills:
                for name, skill in sorted(skill_loader.skills.items()):
                    result_lines.append(f"  {name}: {skill.description}")
            else:
                result_lines.append("No skills loaded")
        elif cmd == "verify":
            results = []
            if cache_store:
                try:
                    v = cache_store.verify_chain()
                    results.append(f"Cache: {'OK' if v.get('valid') else 'BROKEN'} ({v.get('checked', 0)} entries)")
                except Exception as e:
                    results.append(f"Cache: error ({e})")
            if chat_store:
                try:
                    sessions = chat_store.list_sessions() if hasattr(chat_store, 'list_sessions') else []
                    verified = 0
                    for s in sessions:
                        sid = s.get("session_id", s) if isinstance(s, dict) else str(s)
                        try:
                            v = chat_store.verify_chain(sid)
                            if v.get("valid"):
                                verified += 1
                        except Exception:
                            pass
                    results.append(f"Chat: {verified}/{len(sessions)} sessions verified OK")
                except Exception as e:
                    results.append(f"Chat: error ({e})")
            result_lines = results if results else ["No stores to verify"]
        elif cmd == "reset":
            learner.model.reset_fast_weights()
            result_lines.append("Fast memory reset")
        elif cmd.startswith("history"):
            if chat_store:
                sessions = chat_store.list_sessions() if hasattr(chat_store, 'list_sessions') else []
                if sessions:
                    for s in sessions[:10]:
                        result_lines.append(f"  Session: {s}")
                else:
                    result_lines.append("No sessions found")
            else:
                result_lines.append("Chat store not available")
        elif cmd.startswith("set futures leverage"):
            try:
                n = int(cmd.split()[-1])
                n = max(1, min(10, n))
                config.futures.leverage = n
                if runtime._futures_engine and hasattr(runtime._futures_engine, 'set_leverage'):
                    runtime._futures_engine.set_leverage(n)
                result_lines.append(f"Futures leverage set to {n}x")
            except (ValueError, IndexError):
                result_lines.append("Usage: set futures leverage N (1-10)")
        elif cmd.startswith("set futures mode"):
            try:
                mode = cmd.split()[-1].lower()
                if mode not in ("scalp", "stable", "secure"):
                    result_lines.append("Invalid mode. Use: scalp, stable, or secure")
                    result_lines.append("  scalp  — SL/TP from 5m-30m ATR (tight, fast trades)")
                    result_lines.append("  stable — SL/TP from 30m-4h ATR (medium swings)")
                    result_lines.append("  secure — SL/TP from 4h-1w ATR (wide, safe)")
                else:
                    config.futures.trade_mode = mode
                    desc = {"scalp": "5m-30m ATR", "stable": "30m-4h ATR", "secure": "4h-1w ATR"}
                    result_lines.append(f"Futures trade mode set to {mode.upper()} ({desc[mode]})")
            except (ValueError, IndexError):
                result_lines.append("Usage: set futures mode scalp/stable/secure")
        elif cmd.startswith("set trading mode") or cmd.startswith("set spot mode"):
            try:
                mode = cmd.split()[-1].lower()
                if mode not in ("scalp", "stable", "secure"):
                    result_lines.append("Invalid mode. Use: scalp, stable, or secure")
                    result_lines.append("  scalp  — SL/TP from 5m-30m ATR (tight, fast trades)")
                    result_lines.append("  stable — SL/TP from 30m-4h ATR (medium swings)")
                    result_lines.append("  secure — SL/TP from 4h-1w ATR (wide, safe)")
                else:
                    config.trading.trade_mode = mode
                    desc = {"scalp": "5m-30m ATR", "stable": "30m-4h ATR", "secure": "4h-1w ATR"}
                    result_lines.append(f"Spot trade mode set to {mode.upper()} ({desc[mode]})")
            except (ValueError, IndexError):
                result_lines.append("Usage: set trading mode scalp/stable/secure")
        else:
            result_lines.append(f"Unknown command: {cmd}")
        return jsonify({"result": "\n".join(result_lines), "command": cmd})

    # ---- Chat ----

    @app.route('/api/chat', methods=['POST'])
    def api_chat():
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"error": "Missing 'message'"}), 400
        message = data['message']
        session_id = data.get('session_id', 'web_default')
        chat_ctx = None
        if chat_store:
            try:
                chat_ctx = chat_store.get_context(session_id, last_n=6)
            except Exception:
                pass
        sources_used = []
        try:
            result = gate.answer(message, chat_context=chat_ctx)
            answer = result.get("answer", "")
            confidence = result.get("confidence", 0.0)
            source = result.get("source", "unknown")
            tool_context = result.get("tool_context", [])
            sources_used = result.get("sources_used", [])
        except Exception as e:
            answer = f"Error: {e}"
            confidence = 0.0
            source = "error"
            tool_context = []
        if chat_store:
            try:
                chat_store.add_turn(session_id, message, {
                    "answer": answer, "confidence": confidence,
                    "source": source, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                })
            except Exception:
                pass
        return jsonify({
            "response": answer,
            "confidence": round(confidence, 3) if confidence else 0,
            "source": source or "unknown",
            "session_id": session_id,
            "tool_context": tool_context,
            "sources_used": sources_used,
        })

    @app.route('/api/chat/stream', methods=['POST'])
    def api_chat_stream():
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"error": "Missing 'message'"}), 400
        message = data['message']
        session_id = data.get('session_id', 'web_default')
        chat_ctx = None
        if chat_store:
            try:
                chat_ctx = chat_store.get_context(session_id, last_n=6)
            except Exception:
                pass

        def event_stream():
            answer = ""
            confidence = 0.0
            source = "unknown"
            tool_context = []
            sources_used = []
            try:
                result = gate.answer(message, chat_context=chat_ctx)
                answer = result.get("answer", "")
                confidence = result.get("confidence", 0.0)
                source = result.get("source", "unknown")
                tool_context = result.get("tool_context", [])
                sources_used = result.get("sources_used", [])
                chunk_size = 4
                for i in range(0, len(answer), chunk_size):
                    yield f"data: {json.dumps({'token': answer[i:i + chunk_size]})}\n\n"
                yield f"data: {json.dumps({'done': True, 'confidence': round(confidence, 3) if confidence else 0, 'source': source or 'unknown', 'tool_context': tool_context, 'sources_used': sources_used})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            if chat_store:
                try:
                    chat_store.add_turn(session_id, message, {
                        "answer": answer, "confidence": confidence,
                        "source": source, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    })
                except Exception:
                    pass

        return Response(event_stream(), mimetype='text/event-stream',
                        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

    @app.route('/api/chat/sessions', methods=['GET'])
    def api_chat_sessions():
        if not chat_store:
            return jsonify([])
        try:
            sessions = chat_store.list_sessions() if hasattr(chat_store, 'list_sessions') else []
            for s in sessions:
                sid = s.get("session_id", "")
                try:
                    msgs = chat_store.get_history(sid, last_n=50)
                    first_user = next((m for m in msgs if m.role == "user"), None)
                    if first_user:
                        content = first_user.content or ""
                        s["title"] = content[:60] + ("..." if len(content) > 60 else "")
                        s["preview"] = content[:100]
                    last_msg = msgs[-1] if msgs else None
                    if last_msg:
                        s["last_timestamp"] = str(getattr(last_msg, 'timestamp', ''))
                except Exception:
                    pass
            return jsonify(sessions)
        except Exception:
            return jsonify([])

    @app.route('/api/chat/history/<session_id>', methods=['GET'])
    def api_chat_history(session_id):
        if not chat_store:
            return jsonify([])
        try:
            msgs = chat_store.get_history(session_id, last_n=50)
            return jsonify([{"role": m.role, "content": m.content,
                             "timestamp": str(getattr(m, 'timestamp', '')),
                             "source": getattr(m, 'source', '') or '',
                             "confidence": getattr(m, 'confidence', 0) or 0}
                            for m in msgs])
        except Exception:
            return jsonify([])

    # ---- Profile ----

    @app.route('/api/profile', methods=['GET'])
    def api_profile():
        # Read from tree first, fall back to in-memory profile
        _prof = profile or {}
        try:
            from qor.knowledge_tree import get_profile_from_tree, get_or_create_system_id
            _sys_id = get_or_create_system_id(_data_dir)
            _uid = f"user:{_sys_id}"
            _tree_prof = get_profile_from_tree(graph, _uid)
            if _tree_prof:
                _prof = _tree_prof
        except Exception:
            _sys_id = profile.get("system_id", "") if profile else ""
        return jsonify({
            "system_id": _sys_id,
            "user_name": _prof.get("user_name", ""),
            "interests": _prof.get("interests", {}),
            "cautions": _prof.get("cautions", []),
            "preferred_detail_level": _prof.get("preferred_detail_level", "detailed"),
        })

    @app.route('/api/profile/name', methods=['POST'])
    def api_profile_name():
        data = request.get_json()
        if not data or 'name' not in data:
            return jsonify({"error": "Missing 'name'"}), 400
        new_name = str(data['name']).strip()
        if profile is not None:
            profile["user_name"] = new_name
        # Save to tree (name goes in tree, not profile.json)
        try:
            from qor.knowledge_tree import save_profile_to_tree, get_or_create_system_id
            _uid = f"user:{get_or_create_system_id(_data_dir)}"
            save_profile_to_tree(graph, _uid, profile or {"user_name": new_name})
        except Exception:
            pass
        gate.system_prompt = _build_identity(profile, gate)
        gate._user_name = new_name
        return jsonify({"ok": True, "user_name": new_name})

    @app.route('/api/profile', methods=['POST'])
    def api_profile_update():
        data = request.get_json()
        if not data:
            return jsonify({"error": "Missing request body"}), 400
        if profile is None:
            return jsonify({"error": "Profile not available"}), 500
        if 'interests' in data:
            profile["interests"] = data["interests"]
        if 'cautions' in data:
            profile["cautions"] = data["cautions"]
        if 'preferred_detail_level' in data:
            profile["preferred_detail_level"] = data["preferred_detail_level"]
        if 'user_name' in data:
            profile["user_name"] = str(data["user_name"]).strip()
        # Save to tree (no credentials)
        try:
            from qor.knowledge_tree import save_profile_to_tree, get_or_create_system_id
            _uid = f"user:{get_or_create_system_id(_data_dir)}"
            save_profile_to_tree(graph, _uid, profile)
        except Exception:
            pass
        gate.system_prompt = _build_identity(profile, gate)
        gate._user_name = profile.get("user_name", "")
        return jsonify({"ok": True})

    # ---- Knowledge ----

    @app.route('/api/graph/stats', methods=['GET'])
    def api_graph_stats():
        if graph and graph.is_open:
            try:
                return jsonify(graph.stats())
            except Exception:
                pass
        return jsonify({"node_count": 0, "edge_count": 0, "status": "unavailable"})

    @app.route('/api/graph/query', methods=['POST'])
    def api_graph_query():
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({"error": "Missing 'question'"}), 400
        if graph and graph.is_open:
            try:
                # Use tree_search — single unified search
                from qor.knowledge_tree import tree_search, get_or_create_system_id
                _uid = f"user:{get_or_create_system_id(_data_dir)}"
                results = tree_search(data['question'], graph, _uid)
                return jsonify({"results": results, "count": len(results)})
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        return jsonify({"error": "Graph not available"}), 503

    @app.route('/api/memory/stats', methods=['GET'])
    def api_memory_stats():
        """Knowledge tree stats — the tree IS the memory now."""
        stats = {
            "source": "knowledge_tree",
            "knowledge": 0, "correction": 0, "lesson": 0,
            "blocked_fact": 0, "preference": 0, "trade_pattern": 0,
            "historical_event": 0, "topic": 0, "user": 0,
        }
        if graph and getattr(graph, 'is_open', False):
            try:
                for ntype in stats:
                    if ntype == "source":
                        continue
                    nodes = graph.list_nodes(node_type=ntype)
                    stats[ntype] = len(nodes) if nodes else 0
                stats["total"] = sum(v for k, v in stats.items()
                                     if k != "source")
            except Exception:
                pass
        return jsonify(stats)

    # ---- Trading & Portfolio ----

    @app.route('/api/trading/status', methods=['GET'])
    def api_trading_status():
        s = runtime.status()
        ts = s.get("trading")
        if ts:
            return jsonify({"active": True, "enabled": True, **ts})
        return jsonify({"active": False, "enabled": False, "mode": "disabled",
                        "trade_mode": getattr(config.trading, 'trade_mode', 'scalp').upper()})

    @app.route('/api/trading/positions', methods=['GET'])
    def api_trading_positions():
        eng = runtime._trading_engine
        if eng and hasattr(eng, 'store'):
            open_trades = eng.store.get_open_trades()
            for t in open_trades:
                _enrich_open_trade(t, eng.client)
            return jsonify(open_trades)
        return jsonify([])

    @app.route('/api/trading/trades', methods=['GET'])
    def api_trading_trades():
        limit = request.args.get('limit', 50, type=int)
        include_open = request.args.get('include_open', 'true').lower() == 'true'
        eng = runtime._trading_engine
        if eng and hasattr(eng, 'store'):
            all_trades = list(eng.store.trades.values())
            if not include_open:
                all_trades = [t for t in all_trades if t.get("status", "").startswith("closed")]
            # Sort: open first (by entry_time desc), then closed (by exit_time desc)
            all_trades.sort(key=lambda x: (
                0 if x.get("status") == "open" else 1,
                -(x.get("exit_time") or x.get("entry_time") or 0)
            ))
            return jsonify(all_trades[:limit])
        return jsonify([])

    @app.route('/api/futures/status', methods=['GET'])
    def api_futures_status():
        s = runtime.status()
        fs = s.get("futures")
        if fs:
            return jsonify({"active": True, "enabled": True, **fs})
        return jsonify({"active": False, "enabled": False, "mode": "disabled",
                        "trade_mode": getattr(config.futures, 'trade_mode', 'scalp').upper() if hasattr(config, 'futures') else "SCALP"})

    @app.route('/api/futures/positions', methods=['GET'])
    def api_futures_positions():
        eng = runtime._futures_engine
        if eng and hasattr(eng, 'client'):
            try:
                positions = eng.client.get_positions()
                all_orders = {}
                try:
                    orders = eng.client.get_open_orders()
                    for o in orders:
                        sym = o.get("symbol", "")
                        all_orders.setdefault(sym, []).append(o)
                except Exception:
                    pass
                result = []
                for p in positions:
                    amt = float(p.get("positionAmt", 0))
                    if amt == 0:
                        continue
                    mark = float(p.get("markPrice", 0))
                    notional = abs(amt) * mark if mark > 0 else abs(float(p.get("notional", 0)))
                    if notional < 1.0:  # Skip dust positions (< $1)
                        continue
                    sym = p["symbol"]
                    entry = float(p.get("entryPrice", 0))
                    mark = float(p.get("markPrice", 0))
                    pnl = float(p.get("unRealizedProfit", 0))
                    direction = "LONG" if amt > 0 else "SHORT"
                    upd = int(p.get("updateTime", 0))
                    sl = 0.0
                    tp = 0.0
                    for o in all_orders.get(sym, []):
                        stop = float(o.get("stopPrice", 0))
                        if o.get("type") == "STOP_MARKET" and stop > 0:
                            sl = stop
                        elif o.get("type") == "TAKE_PROFIT_MARKET" and stop > 0:
                            tp = stop
                    result.append({
                        "symbol": sym.replace("USDT", ""), "pair": sym,
                        "direction": direction,
                        "side": "BUY" if direction == "LONG" else "SELL",
                        "quantity": abs(amt), "entry_price": entry,
                        "current_price": mark, "unrealized_pnl": round(pnl, 2),
                        "unrealized_pnl_pct": round(pnl / (abs(amt) * entry) * 100, 2) if entry and amt else 0,
                        "stop_loss": sl, "take_profit": tp,
                        "leverage": int(p.get("leverage", 1)),
                        "margin_type": p.get("marginType", "").upper(),
                        "liquidation_price": float(p.get("liquidationPrice", 0)),
                        "notional": abs(float(p.get("notional", 0))),
                        "entry_time": upd * 1000 if upd else 0,
                        "status": "open",
                    })
                return jsonify(result)
            except Exception:
                pass
        return jsonify([])

    @app.route('/api/futures/trades', methods=['GET'])
    def api_futures_trades():
        limit = request.args.get('limit', 50, type=int)
        include_open = request.args.get('include_open', 'true').lower() == 'true'
        eng = runtime._futures_engine
        if eng and hasattr(eng, 'store'):
            all_trades = list(eng.store.trades.values())
            if not include_open:
                all_trades = [t for t in all_trades if t.get("status", "").startswith("closed")]
            all_trades.sort(key=lambda x: (
                0 if x.get("status") == "open" else 1,
                -(x.get("exit_time") or x.get("entry_time") or 0)
            ))
            return jsonify(all_trades[:limit])
        return jsonify([])

    @app.route('/api/portfolio', methods=['GET'])
    def api_portfolio():
        result = {
            "usdt_balance": 0, "total_pnl": 0.0, "total_trades": 0,
            "wins": 0, "losses": 0, "win_rate": 0.0, "open_positions": 0,
            "profit_factor": 0.0, "avg_win_pct": 0.0, "avg_loss_pct": 0.0,
            "best_trade_pct": 0.0, "worst_trade_pct": 0.0, "avg_hold_hours": 0.0,
            "spot": {}, "futures": {},
        }
        total_balance = 0.0
        holdings = []
        # --- Spot: USDT + all token balances converted to USD ---
        spot_eng = runtime._trading_engine
        if spot_eng:
            try:
                account = spot_eng.client.get_account()
                for b in account.get("balances", []):
                    free = float(b.get("free", 0))
                    locked = float(b.get("locked", 0))
                    total_qty = free + locked
                    if total_qty <= 0:
                        continue
                    asset = b["asset"]
                    if asset in ("USDT", "BUSD", "USD", "USDC"):
                        usd_val = total_qty
                    else:
                        try:
                            price = spot_eng.client.get_price(asset + "USDT")
                            usd_val = total_qty * price
                        except Exception:
                            try:
                                price = spot_eng.client.get_price(asset + "BUSD")
                                usd_val = total_qty * price
                            except Exception:
                                continue
                    if usd_val < 0.01:
                        continue
                    total_balance += usd_val
                    holdings.append({"asset": asset, "qty": round(total_qty, 8), "usd": round(usd_val, 2)})
                result["spot_balance"] = round(total_balance, 2)
            except Exception:
                pass
        # --- Futures: USDT balance + unrealized PnL from positions ---
        fut_eng = runtime._futures_engine
        fut_balance = 0.0
        if fut_eng:
            try:
                account = fut_eng.client.get_account()
                # totalWalletBalance includes margin + realized PnL
                fut_balance = float(account.get("totalWalletBalance", 0))
                unrealized = float(account.get("totalUnrealizedProfit", 0))
                fut_total = fut_balance + unrealized
                total_balance += fut_total
                result["futures_balance"] = round(fut_total, 2)
                result["futures_unrealized"] = round(unrealized, 2)
            except Exception:
                try:
                    bal = fut_eng.client.get_balance("USDT")
                    total_balance += bal
                    result["futures_balance"] = round(bal, 2)
                except Exception:
                    pass
        result["usdt_balance"] = round(total_balance, 2)
        result["holdings"] = sorted(holdings, key=lambda h: h["usd"], reverse=True)
        total_wins = 0
        total_losses = 0
        total_pnl = 0.0
        all_win_pcts = []
        all_loss_pcts = []
        gross_profit = 0.0
        gross_loss = 0.0
        best_pct = 0.0
        worst_pct = 0.0
        open_count = 0
        for label, engine in [("spot", runtime._trading_engine),
                               ("futures", runtime._futures_engine)]:
            if not engine or not hasattr(engine, 'store'):
                continue
            stats = engine.store.get_stats()
            result[label] = stats
            w = stats.get("wins", 0)
            l = stats.get("losses", 0)
            total_wins += w
            total_losses += l
            total_pnl += stats.get("total_pnl_usdt", 0.0)
            open_count += len(engine.store.get_open_trades())
            if w > 0 and stats.get("avg_win_pct", 0):
                all_win_pcts.extend([stats["avg_win_pct"]] * w)
            if l > 0 and stats.get("avg_loss_pct", 0):
                all_loss_pcts.extend([stats["avg_loss_pct"]] * l)
            if stats.get("best_trade_pct", 0) > best_pct:
                best_pct = stats["best_trade_pct"]
            if stats.get("worst_trade_pct", 0) < worst_pct:
                worst_pct = stats["worst_trade_pct"]
            gp = stats.get("avg_win_pct", 0) * w
            gl = abs(stats.get("avg_loss_pct", 0)) * l
            gross_profit += gp
            gross_loss += gl
        total_closed = total_wins + total_losses
        result["total_pnl"] = round(total_pnl, 2)
        result["total_trades"] = total_closed
        result["wins"] = total_wins
        result["losses"] = total_losses
        result["win_rate"] = round((total_wins / total_closed * 100), 1) if total_closed else 0.0
        result["open_positions"] = open_count
        result["avg_win_pct"] = round(sum(all_win_pcts) / len(all_win_pcts), 2) if all_win_pcts else 0.0
        result["avg_loss_pct"] = round(sum(all_loss_pcts) / len(all_loss_pcts), 2) if all_loss_pcts else 0.0
        result["best_trade_pct"] = round(best_pct, 2)
        result["worst_trade_pct"] = round(worst_pct, 2)
        result["profit_factor"] = round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0.0
        # 24h P&L — sum of trades closed in last 24 hours
        day_pnl = 0.0
        now_us = int(time.time() * 1_000_000)
        cutoff_us = now_us - 86400 * 1_000_000  # 24h ago
        for label, engine in [("spot", runtime._trading_engine),
                               ("futures", runtime._futures_engine)]:
            if not engine or not hasattr(engine, 'store'):
                continue
            for t in engine.store.trades.values():
                if t.get("status", "").startswith("closed") and t.get("exit_time", 0) >= cutoff_us:
                    day_pnl += t.get("pnl", 0.0)
        result["day_pnl"] = round(day_pnl, 2)
        balance = result["usdt_balance"]
        result["day_pnl_pct"] = round((day_pnl / balance * 100), 2) if balance > 0 else 0.0
        return jsonify(result)

    # ---- Chart Data ----

    @app.route('/api/chart/pnl', methods=['GET'])
    def api_chart_pnl():
        trades = []
        for label, engine in [("spot", runtime._trading_engine),
                               ("futures", runtime._futures_engine)]:
            if not engine or not hasattr(engine, 'store'):
                continue
            for t in engine.store.trades.values():
                if t.get("status", "").startswith("closed") and t.get("exit_time"):
                    trades.append({
                        "time": t["exit_time"], "pnl": t.get("pnl", 0.0),
                        "pnl_pct": t.get("pnl_pct", 0.0),
                        "symbol": t.get("symbol", ""),
                        "direction": t.get("direction", "LONG"), "engine": label,
                    })
        trades.sort(key=lambda x: x["time"])
        cumulative = 0.0
        series = []
        for t in trades:
            cumulative += t["pnl"]
            series.append({
                "time": t["time"], "pnl": round(t["pnl"], 2),
                "cumulative": round(cumulative, 2), "symbol": t["symbol"],
                "direction": t["direction"], "engine": t["engine"],
            })
        return jsonify({"series": series, "total_pnl": round(cumulative, 2), "trade_count": len(series)})

    @app.route('/api/chart/prices', methods=['GET'])
    def api_chart_prices():
        asset_query = request.args.get('asset', '')
        period = request.args.get('period', '24h')
        if not asset_query:
            return jsonify({"error": "asset param required"}), 400
        try:
            from qor.tools import (
                _fetch_ohlc_yahoo, _fetch_ohlc_binance, _fetch_ohlc_crypto,
                _extract_coingecko_id, _extract_stock_symbol,
                _COMMODITY_YAHOO, _FOREX_YAHOO, _extract_crypto_symbol,
            )
            q = asset_query.lower().strip()
            ohlc = None
            period_map = {
                '1h': ('1m', 60), '4h': ('5m', 48), '24h': ('15m', 96),
                '7d': ('1h', 168), '30d': ('4h', 180), '90d': ('1d', 90), '1y': ('1d', 365),
            }
            yahoo_period_map = {
                '1h': ('1m', '1d'), '4h': ('5m', '1d'), '24h': ('15m', '5d'),
                '7d': ('60m', '7d'), '30d': ('1d', '1mo'), '90d': ('1d', '3mo'), '1y': ('1d', '1y'),
            }
            interval, limit = period_map.get(period, ('15m', 96))
            y_interval, y_range = yahoo_period_map.get(period, ('15m', '5d'))
            from qor.quant import classify_asset as _ca
            _asset = _ca(asset_query)
            is_crypto = _asset.asset_type == "crypto"
            if is_crypto:
                sym = _extract_crypto_symbol(asset_query)
                ohlc = _fetch_ohlc_binance(sym, interval, limit)
                if not ohlc:
                    ohlc, _ = _fetch_ohlc_crypto(asset_query)
            else:
                yahoo_sym = _COMMODITY_YAHOO.get(q) or _FOREX_YAHOO.get(q)
                if not yahoo_sym:
                    yahoo_sym = _extract_stock_symbol(asset_query)
                if yahoo_sym:
                    ohlc = _fetch_ohlc_yahoo(yahoo_sym, y_interval, y_range)
            if not ohlc:
                return jsonify({"error": f"No price data for '{asset_query}'"}), 404
            closes = ohlc.get("closes", [])
            highs = ohlc.get("highs", closes)
            lows = ohlc.get("lows", closes)
            opens = ohlc.get("opens", closes)
            volumes = ohlc.get("volumes", [])
            name = ohlc.get("name", asset_query)
            series = []
            for i, c in enumerate(closes):
                point = {"t": i, "v": round(c, 2)}
                if i < len(highs):
                    point["h"] = round(highs[i], 2)
                if i < len(lows):
                    point["l"] = round(lows[i], 2)
                if i < len(opens):
                    point["o"] = round(opens[i], 2)
                if volumes and i < len(volumes):
                    point["vol"] = volumes[i]
                series.append(point)
            return jsonify({
                "asset": name, "period": period, "series": series,
                "current": round(closes[-1], 2) if closes else 0,
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/chart/fear-greed', methods=['GET'])
    def api_fear_greed():
        try:
            import urllib.request, json as _json
            url = "https://api.alternative.me/fng/?limit=1"
            req = urllib.request.Request(url, headers={"User-Agent": "QOR/1.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = _json.loads(resp.read())
            if data.get("data"):
                entry = data["data"][0]
                return jsonify({"value": int(entry.get("value", 50)),
                                "label": entry.get("value_classification", "Neutral")})
        except Exception:
            pass
        return jsonify({"value": 50, "label": "Neutral"})

    # ---- Settings ----

    @app.route('/api/settings', methods=['GET'])
    def api_settings():
        # System ID
        _sys_id = ""
        try:
            from qor.knowledge_tree import get_or_create_system_id
            _sys_id = get_or_create_system_id(_data_dir)
        except Exception:
            pass

        # Masked API keys (show first 4 + last 4, mask the rest)
        def _mask(key):
            if not key:
                return ""
            if len(key) <= 8:
                return key[:2] + "..." + key[-2:]
            return key[:4] + "..." + key[-4:]

        return jsonify({
            "system_id": _sys_id,
            "model": {
                "size": getattr(config.model, 'd_model', 0),
                "layers": getattr(config.model, 'n_layers', 0),
                "parameters": learner.model.n_params if learner.model else 0,
                "device": str(config.get_device()) if hasattr(config, 'get_device') else "cpu",
            },
            "trading": {
                "enabled": config.trading.enabled,
                "testnet": config.trading.testnet,
                "has_keys": bool(config.trading.api_key),
                "api_key_masked": _mask(config.trading.api_key),
                "api_secret_masked": _mask(config.trading.api_secret),
                "allocated_fund": getattr(config.trading, 'allocated_fund', 0),
                "symbols": config.trading.symbols,
                "trade_mode": getattr(config.trading, 'trade_mode', 'scalp'),
            },
            "futures": {
                "enabled": config.futures.enabled if hasattr(config, 'futures') else False,
                "leverage": config.futures.leverage if hasattr(config, 'futures') else 1,
                "testnet": config.futures.testnet if hasattr(config, 'futures') else True,
                "allocated_fund": getattr(config.futures, 'allocated_fund', 0) if hasattr(config, 'futures') else 0,
                "trade_mode": getattr(config.futures, 'trade_mode', 'scalp') if hasattr(config, 'futures') else 'scalp',
            },
            "serve": {
                "temperature": config.serve.temperature,
                "max_tokens": getattr(config.serve, 'max_new_tokens', 512),
                "host": config.serve.host, "port": config.serve.port,
            },
            "browser": {
                "available": _browser_available(),
                "default": _browser_default(),
            },
        })

    @app.route('/api/settings', methods=['POST'])
    def api_settings_update():
        data = request.get_json()
        if not data or 'key' not in data or 'value' not in data:
            return jsonify({"error": "Missing 'key' and 'value'"}), 400
        key = data['key']
        value = data['value']
        try:
            if key == "temperature":
                config.serve.temperature = float(value)
            elif key == "max_tokens":
                config.serve.max_new_tokens = int(value)
            elif key == "system_prompt":
                gate.system_prompt = str(value)
            elif key == "spot_allocated_fund":
                config.trading.allocated_fund = float(value)
            elif key == "futures_allocated_fund":
                config.futures.allocated_fund = float(value)
            elif key == "spot_trade_mode":
                mode = str(value).lower()
                if mode not in ("scalp", "stable", "secure"):
                    return jsonify({"error": "Invalid mode. Use: scalp, stable, secure"}), 400
                config.trading.trade_mode = mode
            elif key == "futures_trade_mode":
                mode = str(value).lower()
                if mode not in ("scalp", "stable", "secure"):
                    return jsonify({"error": "Invalid mode. Use: scalp, stable, secure"}), 400
                if hasattr(config, 'futures'):
                    config.futures.trade_mode = mode
            else:
                return jsonify({"error": f"Unknown setting: {key}"}), 400
            return jsonify({"ok": True, "key": key, "value": value})
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    @app.route('/api/trading/credentials', methods=['GET'])
    def api_trading_credentials_get():
        """Return masked API keys so the UI can show them."""
        def _mask(key):
            if not key:
                return ""
            if len(key) <= 8:
                return key[:2] + "..." + key[-2:]
            return key[:4] + "..." + key[-4:]
        return jsonify({
            "has_keys": bool(config.trading.api_key and config.trading.api_secret),
            "api_key_masked": _mask(config.trading.api_key),
            "api_secret_masked": _mask(config.trading.api_secret),
            "testnet": config.trading.testnet,
        })

    @app.route('/api/trading/credentials', methods=['POST'])
    def api_trading_credentials():
        data = request.get_json()
        if not data:
            return jsonify({"error": "Missing request body"}), 400
        api_key = data.get('key', '')
        api_secret = data.get('secret', '')
        if not api_key or not api_secret:
            return jsonify({"error": "Both 'key' and 'secret' required"}), 400
        config.trading.api_key = api_key
        config.trading.api_secret = api_secret
        config.trading.enabled = True
        if hasattr(config, 'futures'):
            config.futures.api_key = api_key
            config.futures.api_secret = api_secret
            config.futures.enabled = True
        saved = False
        try:
            creds_data = profile.get("trading_credentials", {}) if profile else {}
            if crypto:
                creds_data["api_key"] = crypto.encrypt_str(api_key)
                creds_data["api_secret"] = crypto.encrypt_str(api_secret)
            else:
                creds_data["api_key"] = api_key
                creds_data["api_secret"] = api_secret
            creds_data["testnet"] = config.trading.testnet
            creds_data["set_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            if profile is not None:
                profile["trading_credentials"] = creds_data
            if profile_path:
                with open(profile_path, 'w') as pf:
                    json.dump(profile, pf, indent=2)
            saved = True
        except Exception:
            pass
        return jsonify({"ok": True, "message": f"Credentials updated{' and saved' if saved else ''}"})

    @app.route('/api/exchange/credentials', methods=['GET'])
    def api_exchange_credentials_list():
        """List ALL supported exchanges with connection status.

        Returns every exchange from EXCHANGE_DEFAULTS, showing which ones
        have API keys configured so the UI can display all available exchanges.
        """
        from qor.config import EXCHANGE_DEFAULTS

        def _mask(key):
            if not key:
                return ""
            if len(key) <= 8:
                return key[:2] + "..." + key[-2:]
            return key[:4] + "..." + key[-4:]

        # Exchange metadata for UI display
        _EXCHANGE_INFO = {
            # Crypto — USDT quote
            "binance":   {"label": "Binance", "type": "crypto", "quote": "USDT",
                          "region": "Global",
                          "markets": ["spot", "futures"],
                          "fields": ["key", "secret"]},
            "coinbase":  {"label": "Coinbase", "type": "crypto", "quote": "USD",
                          "region": "US/Global",
                          "markets": ["spot"],
                          "fields": ["key", "secret", "passphrase"]},
            "okx":       {"label": "OKX", "type": "crypto", "quote": "USDT",
                          "region": "Global",
                          "markets": ["spot", "futures"],
                          "fields": ["key", "secret", "passphrase"]},
            "bybit":     {"label": "Bybit", "type": "crypto", "quote": "USDT",
                          "region": "Global",
                          "markets": ["spot", "futures"],
                          "fields": ["key", "secret"]},
            "kraken":    {"label": "Kraken", "type": "crypto", "quote": "USD",
                          "region": "US/EU",
                          "markets": ["spot", "futures"],
                          "fields": ["key", "secret"]},
            "kucoin":    {"label": "KuCoin", "type": "crypto", "quote": "USDT",
                          "region": "Global",
                          "markets": ["spot", "futures"],
                          "fields": ["key", "secret", "passphrase"]},
            "gate_io":   {"label": "Gate.io", "type": "crypto", "quote": "USDT",
                          "region": "Global",
                          "markets": ["spot", "futures"],
                          "fields": ["key", "secret"]},
            # US Stocks — USD quote
            "alpaca":    {"label": "Alpaca", "type": "stocks", "quote": "USD",
                          "region": "US",
                          "markets": ["us_equities"],
                          "fields": ["key", "secret"]},
            "tradier":   {"label": "Tradier", "type": "stocks", "quote": "USD",
                          "region": "US",
                          "markets": ["us_equities"],
                          "fields": ["key", "secret"]},
            # Forex & US Commodities — USD quote
            "oanda":     {"label": "OANDA", "type": "forex", "quote": "USD",
                          "region": "US/Global",
                          "markets": ["us_commodities", "forex"],
                          "fields": ["key", "secret"]},
            # Indian Markets — INR quote
            "upstox":    {"label": "Upstox", "type": "indian", "quote": "INR",
                          "region": "India",
                          "markets": ["equities", "commodities", "indices"],
                          "fields": ["key", "secret", "access_token"]},
        }

        # Build configured exchanges lookup
        configured = {}
        for ex in getattr(config, 'exchanges', []):
            if ex.api_key:
                configured[ex.name] = ex

        result = []
        for ex_name in EXCHANGE_DEFAULTS:
            info = _EXCHANGE_INFO.get(ex_name, {})
            entry = {
                "name": ex_name,
                "label": info.get("label", ex_name.title()),
                "type": info.get("type", "other"),
                "quote": info.get("quote", "USD"),
                "region": info.get("region", "Global"),
                "markets": info.get("markets", []),
                "required_fields": info.get("fields", ["key", "secret"]),
                "has_testnet": any(k.startswith("testnet_") for k in EXCHANGE_DEFAULTS[ex_name]),
                "connected": False,
                "testnet": True,
                "api_key_masked": "",
                "symbols": [],
            }

            # Check Binance separately (uses config.trading)
            if ex_name == "binance" and config.trading.api_key:
                entry["connected"] = True
                entry["testnet"] = config.trading.testnet
                entry["api_key_masked"] = _mask(config.trading.api_key)
                entry["symbols"] = list(config.trading.symbols)
                # Check if engine is running
                entry["spot_running"] = (runtime._trading_engine is not None
                    and runtime._trading_engine._thread is not None
                    and runtime._trading_engine._thread.is_alive()) if runtime._trading_engine else False
                entry["futures_running"] = (runtime._futures_engine is not None
                    and runtime._futures_engine._thread is not None
                    and runtime._futures_engine._thread.is_alive()) if runtime._futures_engine else False
            elif ex_name in configured:
                ex = configured[ex_name]
                entry["connected"] = True
                entry["testnet"] = ex.testnet
                entry["api_key_masked"] = _mask(ex.api_key)
                entry["symbols"] = list(ex.symbols) if ex.symbols else []
                # Check if engine is running
                engines = getattr(runtime, '_exchange_engines', {})
                eng = engines.get(ex_name)
                entry["running"] = (eng is not None and eng._thread is not None
                    and eng._thread.is_alive()) if eng else False

            result.append(entry)

        return jsonify(result)

    @app.route('/api/exchange/credentials', methods=['POST'])
    def api_exchange_credentials_set():
        """Set API keys for any exchange.

        Body: {"exchange": "upstox", "key": "...", "secret": "...",
               "access_token": "...", "testnet": true}
        For Binance, use POST /api/trading/credentials instead.
        """
        data = request.get_json()
        if not data:
            return jsonify({"error": "Missing request body"}), 400

        ex_name = data.get('exchange', '').lower().strip()
        if not ex_name:
            return jsonify({"error": "'exchange' name required"}), 400

        # Redirect Binance to existing endpoint
        if ex_name == 'binance':
            return jsonify({"error": "Use POST /api/trading/credentials for Binance"}), 400

        api_key = data.get('key', '')
        api_secret = data.get('secret', '')
        access_token = data.get('access_token', '')
        passphrase = data.get('passphrase', '')
        testnet = data.get('testnet', True)

        if not api_key:
            return jsonify({"error": "'key' is required"}), 400

        # Update or create ExchangeKeys in config
        from qor.config import ExchangeKeys
        existing = None
        for ex in getattr(config, 'exchanges', []):
            if ex.name == ex_name:
                existing = ex
                break

        if existing:
            existing.api_key = api_key
            if api_secret:
                existing.api_secret = api_secret
            if access_token:
                existing.access_token = access_token
            if passphrase:
                existing.passphrase = passphrase
            existing.testnet = testnet
            existing.enabled = True
        else:
            if not hasattr(config, 'exchanges'):
                config.exchanges = []
            config.exchanges.append(ExchangeKeys(
                name=ex_name, api_key=api_key, api_secret=api_secret,
                passphrase=passphrase, access_token=access_token,
                testnet=testnet, enabled=True,
            ))

        # Persist encrypted to profile
        saved = False
        try:
            all_ex = profile.setdefault("exchange_credentials", {}) if profile is not None else {}
            creds_data = all_ex.get(ex_name, {})
            if crypto:
                creds_data["api_key"] = crypto.encrypt_str(api_key)
                if api_secret:
                    creds_data["api_secret"] = crypto.encrypt_str(api_secret)
                if access_token:
                    creds_data["access_token"] = crypto.encrypt_str(access_token)
                if passphrase:
                    creds_data["passphrase"] = crypto.encrypt_str(passphrase)
            else:
                creds_data["api_key"] = api_key
                if api_secret:
                    creds_data["api_secret"] = api_secret
                if access_token:
                    creds_data["access_token"] = access_token
                if passphrase:
                    creds_data["passphrase"] = passphrase
            creds_data["testnet"] = testnet
            creds_data["set_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            all_ex[ex_name] = creds_data
            if profile is not None:
                profile["exchange_credentials"] = all_ex
            if profile_path:
                with open(profile_path, 'w') as pf:
                    json.dump(profile, pf, indent=2)
            saved = True
        except Exception:
            pass

        return jsonify({
            "ok": True,
            "exchange": ex_name,
            "message": f"{ex_name.title()} credentials set{' and saved' if saved else ''}",
        })

    @app.route('/api/exchange/upstox/auth-url', methods=['GET'])
    def api_upstox_auth_url():
        """Generate Upstox OAuth2 login URL.

        User opens this in browser → logs in → gets redirected with ?code=...
        Then POST /api/exchange/upstox/token with the code.

        Query params:
            redirect_uri: The redirect URI registered in the Upstox app
                          (default: http://localhost:3000/upstox-callback)
        """
        # Find Upstox API key from config
        api_key = ""
        for ex in getattr(config, 'exchanges', []):
            if ex.name == "upstox" and ex.api_key:
                api_key = ex.api_key
                break
        if not api_key:
            return jsonify({"error": "Set Upstox API key first via Exchange Connections"}), 400
        redirect_uri = request.args.get('redirect_uri', 'http://localhost:3000/upstox-callback')
        import urllib.parse
        auth_url = (
            f"https://api.upstox.com/v2/login/authorization/dialog"
            f"?client_id={api_key}"
            f"&redirect_uri={urllib.parse.quote(redirect_uri, safe='')}"
            f"&response_type=code"
        )
        return jsonify({"auth_url": auth_url, "redirect_uri": redirect_uri})

    @app.route('/api/exchange/upstox/token', methods=['POST'])
    def api_upstox_token():
        """Exchange Upstox OAuth2 code for access token.

        Body: {"code": "auth_code_from_redirect", "redirect_uri": "..."}
        """
        import urllib.parse
        data = request.get_json()
        if not data or not data.get('code'):
            return jsonify({"error": "'code' is required"}), 400

        # Find Upstox creds
        api_key = api_secret = ""
        for ex in getattr(config, 'exchanges', []):
            if ex.name == "upstox":
                api_key = ex.api_key
                api_secret = ex.api_secret
                break
        if not api_key or not api_secret:
            return jsonify({"error": "Set Upstox API key and secret first"}), 400

        redirect_uri = data.get('redirect_uri', 'http://localhost:3000/upstox-callback')
        try:
            from qor.upstox import UpstoxClient
            result = UpstoxClient.get_access_token(
                code=data['code'],
                client_id=api_key,
                client_secret=api_secret,
                redirect_uri=redirect_uri,
            )
            access_token = result.get("access_token", "")
            if not access_token:
                return jsonify({"error": "No access_token in response", "details": result}), 400

            # Save access_token to exchange config
            for ex in getattr(config, 'exchanges', []):
                if ex.name == "upstox":
                    ex.access_token = access_token
                    break

            # Persist to profile
            try:
                all_ex = profile.setdefault("exchange_credentials", {}) if profile is not None else {}
                creds_data = all_ex.get("upstox", {})
                if crypto:
                    creds_data["access_token"] = crypto.encrypt_str(access_token)
                else:
                    creds_data["access_token"] = access_token
                creds_data["token_set_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                all_ex["upstox"] = creds_data
                if profile is not None:
                    profile["exchange_credentials"] = all_ex
                if profile_path:
                    with open(profile_path, 'w') as pf:
                        json.dump(profile, pf, indent=2)
            except Exception:
                pass

            return jsonify({
                "ok": True,
                "message": "Upstox access token obtained and saved",
                "user_id": result.get("user_id", ""),
                "exchanges": result.get("exchanges", []),
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    def _save_trading_symbols(engine_name, symbols_list):
        """Persist trading symbols to profile.json."""
        try:
            if profile is not None:
                saved_syms = profile.get("trading_symbols", {})
                saved_syms[engine_name] = symbols_list
                saved_syms["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                profile["trading_symbols"] = saved_syms
            if profile_path:
                with open(profile_path, 'w') as pf:
                    json.dump(profile, pf, indent=2)
        except Exception:
            pass

    def _start_exchange_engine(ex_name, symbols_list):
        """Start (or update) an exchange engine by name (upstox, alpaca, oanda, etc.)."""
        from .trading import create_exchange_client, TradingEngine
        from qor.config import TradingConfig, ExchangeKeys

        # Find existing exchange config or look for stored creds
        ex_cfg = None
        for ex in getattr(config, 'exchanges', []):
            if ex.name == ex_name:
                ex_cfg = ex
                break

        if ex_cfg is None:
            return None, f"No API keys configured for '{ex_name}'. Set credentials first via POST /api/exchange/credentials"

        if not ex_cfg.api_key:
            return None, f"No API key set for '{ex_name}'"

        # Update symbols
        ex_cfg.symbols = symbols_list
        ex_cfg.enabled = True

        # Check if engine already running
        engines = getattr(runtime, '_exchange_engines', {})
        existing = engines.get(ex_name)
        if existing:
            # Update symbols on running engine
            existing.config.symbols = symbols_list
            if not existing._thread or not existing._thread.is_alive():
                existing.start()
            return existing, None

        # Create new engine
        try:
            client = create_exchange_client(
                ex_name, ex_cfg.api_key, ex_cfg.api_secret,
                passphrase=ex_cfg.passphrase, testnet=ex_cfg.testnet,
                base_url=ex_cfg.base_url,
                access_token=getattr(ex_cfg, 'access_token', ''),
            )
            ex_trading = TradingConfig(
                enabled=True, testnet=ex_cfg.testnet,
                api_key=ex_cfg.api_key, api_secret=ex_cfg.api_secret,
                symbols=symbols_list,
                check_interval_seconds=ex_cfg.check_interval_seconds,
                data_dir=os.path.join(
                    config.runtime.data_dir, "exchanges", ex_name),
            )
            os.makedirs(ex_trading.data_dir, exist_ok=True)
            _ex_cfg = type('_ExCfg', (), {'trading': ex_trading})()
            tex = getattr(gate, '_tool_executor', None)
            engine = TradingEngine(
                _ex_cfg, tool_executor=tex, hmm=getattr(runtime, '_hmm', None),
                client=client)
            engine.start()
            runtime._exchange_engines[ex_name] = engine
            return engine, None
        except Exception as e:
            return None, str(e)

    @app.route('/api/trading/start', methods=['POST'])
    def api_trading_start():
        """Start a trading engine.

        Body: {"engine": "spot"|"futures"|"upstox"|"alpaca"|..., "symbols": ["BTC"]}
        engine: "spot" and "futures" use Binance. Any other name uses
                the matching exchange from config.exchanges.
        symbols: optional — if provided, updates what the engine trades.
        """
        data = request.get_json() or {}
        engine_name = data.get('engine', 'spot')

        # Clean symbols if provided
        req_symbols = data.get('symbols')
        symbols_clean = None
        if req_symbols and isinstance(req_symbols, list):
            symbols_clean = list(dict.fromkeys(
                s.strip().upper() for s in req_symbols
                if isinstance(s, str) and s.strip()
            ))
            if not symbols_clean:
                symbols_clean = None

        try:
            # --- Binance Spot ---
            if engine_name == 'spot':
                if not config.trading.api_key:
                    return jsonify({"error": "Set Binance API keys first"}), 400
                if symbols_clean:
                    config.trading.symbols = symbols_clean
                    _save_trading_symbols('spot', symbols_clean)
                if not runtime._trading_engine:
                    try:
                        from .trading import TradingEngine
                        tex = getattr(gate, '_tool_executor', None)
                        config.trading.enabled = True
                        runtime._trading_engine = TradingEngine(config, tool_executor=tex)
                    except Exception as e:
                        return jsonify({"error": f"Failed to create spot engine: {e}"}), 500
                elif symbols_clean:
                    runtime._trading_engine.config.symbols = config.trading.symbols
                runtime._trading_engine.start()
                mode = "DEMO" if config.trading.testnet else "LIVE"
                trade_mode = getattr(config.trading, 'trade_mode', 'scalp').upper()
                syms = ", ".join(config.trading.symbols)
                return jsonify({"ok": True, "engine": "spot", "status": "started",
                    "symbols": config.trading.symbols,
                    "result": f"Spot engine started ({mode}, {trade_mode})\nSymbols: {syms}"})

            # --- Binance Futures ---
            if engine_name == 'futures':
                if not config.trading.api_key:
                    return jsonify({"error": "Set Binance API keys first"}), 400
                if symbols_clean:
                    config.futures.symbols = symbols_clean
                    _save_trading_symbols('futures', symbols_clean)
                if not runtime._futures_engine:
                    try:
                        from .futures import FuturesEngine
                        tex = getattr(gate, '_tool_executor', None)
                        config.futures.enabled = True
                        config.futures.api_key = config.trading.api_key
                        config.futures.api_secret = config.trading.api_secret
                        runtime._futures_engine = FuturesEngine(config, tool_executor=tex)
                    except Exception as e:
                        return jsonify({"error": f"Failed to create futures engine: {e}"}), 500
                elif symbols_clean:
                    runtime._futures_engine.config.symbols = config.futures.symbols
                    runtime._futures_engine._leverage_set = False
                    try:
                        runtime._futures_engine._setup_symbols()
                    except Exception:
                        pass
                runtime._futures_engine.start()
                mode = "TESTNET" if config.futures.testnet else "LIVE"
                lev = config.futures.leverage if hasattr(config.futures, 'leverage') else 5
                trade_mode = getattr(config.futures, 'trade_mode', 'scalp').upper()
                syms = ", ".join(config.futures.symbols)
                return jsonify({"ok": True, "engine": "futures", "status": "started",
                    "symbols": config.futures.symbols,
                    "result": f"Futures engine started ({mode}, {lev}x, {trade_mode})\nSymbols: {syms}"})

            # --- Any other exchange (upstox, alpaca, oanda, etc.) ---
            syms = symbols_clean or []
            if not syms:
                # Try to get symbols from existing exchange config
                for ex in getattr(config, 'exchanges', []):
                    if ex.name == engine_name and ex.symbols:
                        syms = list(ex.symbols)
                        break
            if not syms:
                return jsonify({"error": f"Provide symbols for '{engine_name}' engine"}), 400

            _save_trading_symbols(engine_name, syms)
            eng, err = _start_exchange_engine(engine_name, syms)
            if err:
                return jsonify({"error": err}), 400
            mode = "DEMO" if getattr(eng.config, 'testnet', True) else "LIVE"
            return jsonify({"ok": True, "engine": engine_name, "status": "started",
                "symbols": syms,
                "result": f"{engine_name.title()} engine started ({mode})\nSymbols: {', '.join(syms)}"})

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/trading/stop', methods=['POST'])
    def api_trading_stop():
        """Stop a trading engine.

        Body: {"engine": "spot"|"futures"|"upstox"|"alpaca"|...}
        """
        data = request.get_json() or {}
        engine_name = data.get('engine', 'spot')
        try:
            if engine_name == 'spot' and runtime._trading_engine:
                runtime._trading_engine.stop()
                return jsonify({"ok": True, "engine": "spot", "status": "stopped",
                    "result": "Spot trading engine stopped."})
            elif engine_name == 'futures' and runtime._futures_engine:
                runtime._futures_engine.stop()
                return jsonify({"ok": True, "engine": "futures", "status": "stopped",
                    "result": "Futures trading engine stopped."})
            else:
                # Check exchange engines
                engines = getattr(runtime, '_exchange_engines', {})
                eng = engines.get(engine_name)
                if eng:
                    eng.stop()
                    return jsonify({"ok": True, "engine": engine_name, "status": "stopped",
                        "result": f"{engine_name.title()} trading engine stopped."})
            return jsonify({"ok": True, "status": "already stopped",
                "result": "Engine is not running."})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/upload', methods=['POST'])
    def api_upload():
        if 'file' not in request.files:
            return jsonify({"error": "No file in request"}), 400
        f = request.files['file']
        if not f.filename:
            return jsonify({"error": "Empty filename"}), 400
        import re as _re
        safe_name = _re.sub(r'[^\w.\-]', '_', f.filename)
        upload_dir = os.path.join(_data_dir, "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        save_path = os.path.join(upload_dir, safe_name)
        base, ext = os.path.splitext(save_path)
        counter = 1
        while os.path.exists(save_path):
            save_path = f"{base}_{counter}{ext}"
            counter += 1
        f.save(save_path)
        ext_lower = ext.lower()
        if ext_lower in ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'):
            ftype = "image"
        elif ext_lower in ('.mp4', '.avi', '.mov', '.mkv'):
            ftype = "video"
        elif ext_lower in ('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.webm', '.aac', '.wma'):
            ftype = "audio"
        elif ext_lower in ('.pdf', '.docx', '.doc', '.txt', '.md'):
            ftype = "document"
        else:
            ftype = "file"
        return jsonify({
            "ok": True, "path": save_path,
            "filename": os.path.basename(save_path),
            "type": ftype, "size": os.path.getsize(save_path),
        })

    # ---- Voice Transcription ----

    _whisper_pipeline = [None]  # lazy-loaded singleton

    @app.route('/api/transcribe', methods=['POST'])
    def api_transcribe():
        """Transcribe audio file to text using Whisper.

        POST multipart: file=<audio>
        Returns: {ok, text, language, duration}
        """
        if 'file' not in request.files:
            return jsonify({"error": "No file in request"}), 400
        f = request.files['file']
        if not f.filename:
            return jsonify({"error": "Empty filename"}), 400

        # Save temp file
        import re as _re
        safe_name = _re.sub(r'[^\w.\-]', '_', f.filename)
        upload_dir = os.path.join(_data_dir, "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        save_path = os.path.join(upload_dir, safe_name)
        f.save(save_path)

        try:
            # Lazy-load Whisper pipeline on first call
            if _whisper_pipeline[0] is None:
                try:
                    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline as hf_pipeline
                    import torch as _torch
                    _model_id = "openai/whisper-small"
                    _proc = AutoProcessor.from_pretrained(_model_id)
                    _model = AutoModelForSpeechSeq2Seq.from_pretrained(
                        _model_id,
                        torch_dtype=_torch.float32,
                        low_cpu_mem_usage=False,
                    )
                    _model = _model.to("cpu")
                    _whisper_pipeline[0] = hf_pipeline(
                        "automatic-speech-recognition",
                        model=_model,
                        tokenizer=_proc.tokenizer,
                        feature_extractor=_proc.feature_extractor,
                        device="cpu",
                        torch_dtype=_torch.float32,
                        generate_kwargs={"task": "transcribe", "language": None},
                    )
                    print("[Transcribe] Whisper-small loaded (auto-detect language)")
                except Exception as e:
                    return jsonify({"error":
                        f"Whisper not available: {e}. "
                        "Install: pip install transformers torch"}), 500

            pipe = _whisper_pipeline[0]
            # Try reading audio with soundfile first (no ffmpeg needed for WAV)
            try:
                import soundfile as sf
                import numpy as np
                audio_data, sr = sf.read(save_path, dtype="float32")
                # Convert to mono if stereo
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.mean(axis=1)
                # Resample to 16kHz if needed
                if sr != 16000:
                    from scipy.signal import resample
                    num_samples = int(len(audio_data) * 16000 / sr)
                    audio_data = resample(audio_data, num_samples).astype(np.float32)
                result = pipe({"raw": audio_data, "sampling_rate": 16000})
            except Exception:
                # Fallback: let pipeline handle the file (needs ffmpeg for non-WAV)
                result = pipe(save_path)
            text = result.get("text", "").strip()

            if not text:
                return jsonify({"ok": False, "error": "No speech detected"})

            return jsonify({
                "ok": True,
                "text": text,
            })
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    # ---- Vision Stream (webcam / screen share) ----

    _vision_state = {"frames": [], "latest": None, "source": None, "count": 0}

    @app.route('/api/vision/frame', methods=['POST'])
    def api_vision_frame():
        """Receive a video frame from webcam or screen share.

        POST JSON: {frame: "data:image/jpeg;base64,...", source: "webcam"|"screen", index: N}
        Stores latest frame + keeps rolling buffer of last 10 frames.
        """
        data = request.get_json(force=True, silent=True) or {}
        frame_data = data.get("frame", "")
        source = data.get("source", "unknown")
        index = data.get("index", 0)

        if not frame_data:
            return jsonify({"error": "No frame data"}), 400

        # Save latest frame to disk for vision pipeline
        try:
            import base64
            # Strip data URL prefix
            if "," in frame_data:
                frame_data = frame_data.split(",", 1)[1]
            img_bytes = base64.b64decode(frame_data)

            frame_dir = os.path.join(_data_dir, "vision_stream")
            os.makedirs(frame_dir, exist_ok=True)

            # Save latest frame (overwrite)
            latest_path = os.path.join(frame_dir, "latest.jpg")
            with open(latest_path, "wb") as f:
                f.write(img_bytes)

            # Save to rolling buffer (last 10 frames)
            buf_path = os.path.join(frame_dir, f"frame_{index % 10:02d}.jpg")
            with open(buf_path, "wb") as f:
                f.write(img_bytes)

            _vision_state["latest"] = latest_path
            _vision_state["source"] = source
            _vision_state["count"] = index

            # Keep metadata for last 10 frames
            _vision_state["frames"] = _vision_state["frames"][-9:] + [{
                "path": buf_path,
                "index": index,
                "time": time.time(),
            }]

            return jsonify({"ok": True, "index": index})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/vision/status', methods=['GET'])
    def api_vision_status():
        """Get current vision stream status."""
        return jsonify({
            "active": _vision_state["latest"] is not None and _vision_state["count"] > 0,
            "source": _vision_state["source"],
            "frame_count": _vision_state["count"],
            "buffer_size": len(_vision_state["frames"]),
        })

    @app.route('/api/vision/describe', methods=['POST'])
    def api_vision_describe():
        """Ask the AI to describe what it sees in the current frame.

        POST JSON: {question: "what do you see?"}
        Returns: {description: "...", frame_index: N}
        """
        data = request.get_json(force=True, silent=True) or {}
        question = data.get("question", "What do you see on screen?")

        if not _vision_state["latest"] or not os.path.exists(_vision_state["latest"]):
            return jsonify({"error": "No vision stream active"}), 400

        try:
            # Use the confidence gate / answer path with the image
            frame_path = _vision_state["latest"]
            # Combine question with frame reference
            full_q = f"{question} [Image: {frame_path}]"

            if gate:
                answer, conf, src = gate.answer(full_q)
            else:
                answer = "Vision stream is active but no AI model loaded for analysis."
                conf, src = 0.5, "vision_stream"

            return jsonify({
                "ok": True,
                "description": answer,
                "confidence": conf,
                "source": src,
                "frame_index": _vision_state["count"],
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # ---- Vision Analysis (continuous frame understanding) ----

    _vision_model = [None]  # lazy-loaded BLIP captioning model
    _vision_prev_hash = [None]  # last processed frame hash for change detection
    _vision_last_caption = [""]  # last caption to detect changes

    @app.route('/api/vision/analyze', methods=['POST'])
    def api_vision_analyze():
        """Analyze the latest frame — returns caption + detected changes.

        POST JSON: {prompt: "optional question about the frame"}
        Returns: {ok, caption, changed, source, frame_index}

        Uses BLIP for image captioning + optional VQA.
        Only processes if frame changed significantly from last analysis.
        """
        import base64, hashlib

        data = request.get_json(force=True, silent=True) or {}
        prompt = data.get("prompt", "")

        if not _vision_state["latest"] or not os.path.exists(_vision_state["latest"]):
            return jsonify({"error": "No vision stream active"}), 400

        try:
            # Read latest frame bytes
            with open(_vision_state["latest"], "rb") as f:
                img_bytes = f.read()

            # Change detection: compare hash of frame
            frame_hash = hashlib.md5(img_bytes).hexdigest()
            if frame_hash == _vision_prev_hash[0] and not prompt:
                # Frame unchanged and no new question — return cached
                return jsonify({
                    "ok": True,
                    "caption": _vision_last_caption[0],
                    "changed": False,
                    "source": _vision_state["source"],
                    "frame_index": _vision_state["count"],
                })

            _vision_prev_hash[0] = frame_hash

            # Lazy-load vision model
            if _vision_model[0] is None:
                try:
                    from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering
                    import torch as _torch

                    print("[Vision] Loading BLIP captioning model...")
                    _proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                    _cap_model = BlipForConditionalGeneration.from_pretrained(
                        "Salesforce/blip-image-captioning-base",
                        dtype=_torch.float32,
                    ).to("cpu")
                    # Also load VQA model for questions
                    print("[Vision] Loading BLIP VQA model...")
                    _vqa_proc = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
                    _vqa_model = BlipForQuestionAnswering.from_pretrained(
                        "Salesforce/blip-vqa-base",
                        dtype=_torch.float32,
                    ).to("cpu")
                    _vision_model[0] = {
                        "cap_proc": _proc, "cap_model": _cap_model,
                        "vqa_proc": _vqa_proc, "vqa_model": _vqa_model,
                    }
                    print("[Vision] BLIP models loaded")
                except Exception as e:
                    print(f"[Vision] BLIP not available: {e}")
                    # Fallback: use the main QOR model via gate
                    _vision_model[0] = "gate_fallback"

            # Open image
            from PIL import Image
            import io
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            caption = ""
            details = {}

            if _vision_model[0] == "gate_fallback":
                # Use QOR model via confidence gate
                q = prompt or "Describe what you see in this image in detail."
                if gate:
                    answer, conf, src = gate.answer(f"{q} [Image: {_vision_state['latest']}]")
                    caption = answer
                else:
                    caption = "Vision model not available."
            else:
                vm = _vision_model[0]
                # Generate caption
                cap_inputs = vm["cap_proc"](image, return_tensors="pt").to("cpu")
                cap_ids = vm["cap_model"].generate(**cap_inputs, max_new_tokens=80)
                caption = vm["cap_proc"].decode(cap_ids[0], skip_special_tokens=True)

                # If user asked a specific question, also do VQA
                if prompt:
                    vqa_inputs = vm["vqa_proc"](image, prompt, return_tensors="pt").to("cpu")
                    vqa_ids = vm["vqa_model"].generate(**vqa_inputs, max_new_tokens=50)
                    details["answer"] = vm["vqa_proc"].decode(vqa_ids[0], skip_special_tokens=True)

                # Run a few standard detections
                standard_qs = [
                    ("gesture", "Is the person making a gesture or waving?"),
                    ("text", "Is there any text visible?"),
                    ("people", "How many people are visible?"),
                ]
                for key, q in standard_qs:
                    try:
                        vqa_in = vm["vqa_proc"](image, q, return_tensors="pt").to("cpu")
                        vqa_out = vm["vqa_model"].generate(**vqa_in, max_new_tokens=20)
                        details[key] = vm["vqa_proc"].decode(vqa_out[0], skip_special_tokens=True)
                    except Exception:
                        pass

            changed = caption != _vision_last_caption[0]
            _vision_last_caption[0] = caption

            return jsonify({
                "ok": True,
                "caption": caption,
                "changed": changed,
                "details": details,
                "source": _vision_state["source"],
                "frame_index": _vision_state["count"],
            })
        except Exception as e:
            print(f"[Vision] Analysis error: {e}")
            return jsonify({"error": str(e)}), 500

    # ---- Task Recording (screen → step detection) ----

    _task_prev_caption = [""]  # last caption for step change detection
    _task_prev_hash = [None]

    @app.route('/api/vision/detect-step', methods=['POST'])
    def api_vision_detect_step():
        """Detect what action/step the user performed on screen.

        Compares current frame to previous, describes the change as a task step.
        Returns: {ok, step, description, url_hint, changed}
        """
        import base64, hashlib

        if not _vision_state["latest"] or not os.path.exists(_vision_state["latest"]):
            return jsonify({"error": "No screen share active"}), 400

        try:
            with open(_vision_state["latest"], "rb") as f:
                img_bytes = f.read()

            frame_hash = hashlib.md5(img_bytes).hexdigest()
            if frame_hash == _task_prev_hash[0]:
                return jsonify({"ok": True, "changed": False, "step": None})
            _task_prev_hash[0] = frame_hash

            # Ensure vision model is loaded
            if _vision_model[0] is None:
                # Trigger lazy load via analyze endpoint
                try:
                    from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering
                    import torch as _torch
                    print("[Vision] Loading BLIP for task recording...")
                    _proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                    _cap_model = BlipForConditionalGeneration.from_pretrained(
                        "Salesforce/blip-image-captioning-base", dtype=_torch.float32).to("cpu")
                    _vqa_proc = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
                    _vqa_model = BlipForQuestionAnswering.from_pretrained(
                        "Salesforce/blip-vqa-base", dtype=_torch.float32).to("cpu")
                    _vision_model[0] = {
                        "cap_proc": _proc, "cap_model": _cap_model,
                        "vqa_proc": _vqa_proc, "vqa_model": _vqa_model,
                    }
                    print("[Vision] BLIP loaded for task recording")
                except Exception as e:
                    _vision_model[0] = "gate_fallback"

            from PIL import Image
            import io
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            caption = ""
            details = {}

            if _vision_model[0] == "gate_fallback":
                if gate:
                    q = "Describe what the user is doing on screen. What did they click or type? What website or app are they using?"
                    answer, _, _ = gate.answer(f"{q} [Image: {_vision_state['latest']}]")
                    caption = answer
                else:
                    caption = "Screen visible but no vision model available."
            else:
                vm = _vision_model[0]
                # Caption: what's on screen
                cap_in = vm["cap_proc"](image, return_tensors="pt").to("cpu")
                cap_ids = vm["cap_model"].generate(**cap_in, max_new_tokens=80)
                caption = vm["cap_proc"].decode(cap_ids[0], skip_special_tokens=True)

                # VQA: detect specifics
                questions = [
                    ("action", "What action is the user performing?"),
                    ("website", "What website or application is visible?"),
                    ("text_input", "What text is the user typing or has typed?"),
                    ("button", "What button or link did the user click?"),
                ]
                for key, q in questions:
                    try:
                        vqa_in = vm["vqa_proc"](image, q, return_tensors="pt").to("cpu")
                        vqa_out = vm["vqa_model"].generate(**vqa_in, max_new_tokens=30)
                        details[key] = vm["vqa_proc"].decode(vqa_out[0], skip_special_tokens=True)
                    except Exception:
                        pass

            # Detect if this is a meaningful new step
            changed = caption != _task_prev_caption[0] and len(caption) > 5
            _task_prev_caption[0] = caption

            # Build step description
            step_desc = caption
            if details.get("action") and details["action"].lower() not in ["no", "none", ""]:
                step_desc = details["action"]
            if details.get("button") and details["button"].lower() not in ["no", "none", ""]:
                step_desc += f" → clicked '{details['button']}'"

            url_hint = details.get("website", "")

            return jsonify({
                "ok": True,
                "changed": changed,
                "step": step_desc if changed else None,
                "caption": caption,
                "details": details,
                "url_hint": url_hint,
                "frame_index": _vision_state["count"],
            })
        except Exception as e:
            print(f"[TaskRecord] Step detection error: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/tasks/save', methods=['POST'])
    def api_save_task():
        """Save a recorded task automation.

        POST JSON: {name, url, steps: [{step, description, time}]}
        Saves to qor-data/tasks/<name>.json
        """
        import json as _json
        data = request.get_json(force=True, silent=True) or {}
        name = data.get("name", "").strip()
        url = data.get("url", "").strip()
        steps = data.get("steps", [])

        if not name:
            return jsonify({"error": "Task name required"}), 400
        if not steps:
            return jsonify({"error": "No steps recorded"}), 400

        # Sanitize name for filename
        import re as _re
        safe_name = _re.sub(r'[^\w\-]', '_', name.lower()).strip('_')

        task_dir = os.path.join(_data_dir, "tasks")
        os.makedirs(task_dir, exist_ok=True)
        task_path = os.path.join(task_dir, f"{safe_name}.json")

        task_data = {
            "name": name,
            "url": url,
            "steps": steps,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "step_count": len(steps),
        }

        with open(task_path, "w", encoding="utf-8") as f:
            _json.dump(task_data, f, indent=2, ensure_ascii=False)

        return jsonify({"ok": True, "path": task_path, "step_count": len(steps)})

    @app.route('/api/tasks/list', methods=['GET'])
    def api_list_tasks():
        """List all saved task automations."""
        import json as _json
        task_dir = os.path.join(_data_dir, "tasks")
        if not os.path.exists(task_dir):
            return jsonify({"tasks": []})

        tasks = []
        for fname in os.listdir(task_dir):
            if fname.endswith(".json"):
                try:
                    with open(os.path.join(task_dir, fname), "r", encoding="utf-8") as f:
                        t = _json.load(f)
                    tasks.append({
                        "name": t.get("name", fname),
                        "url": t.get("url", ""),
                        "step_count": t.get("step_count", 0),
                        "created": t.get("created", ""),
                        "file": fname,
                    })
                except Exception:
                    pass
        return jsonify({"tasks": tasks})

    @app.route('/api/tasks/get/<name>', methods=['GET'])
    def api_get_task(name):
        """Get a specific saved task by filename."""
        import json as _json
        task_dir = os.path.join(_data_dir, "tasks")
        task_path = os.path.join(task_dir, name)
        if not os.path.exists(task_path):
            return jsonify({"error": "Task not found"}), 404
        with open(task_path, "r", encoding="utf-8") as f:
            return jsonify(_json.load(f))

    # ---- Text-to-Speech (edge-tts) ----

    @app.route('/api/tts', methods=['POST'])
    def api_tts():
        """Convert text to speech using edge-tts.

        POST JSON: {text, voice?, rate?}
        Returns: audio/mp3 stream
        """
        data = request.get_json(force=True, silent=True) or {}
        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "No text provided"}), 400
        if len(text) > 5000:
            text = text[:5000]

        voice = data.get("voice", "en-US-AriaNeural")
        rate = data.get("rate", "+0%")  # e.g. "+20%", "-10%"

        try:
            import edge_tts
            import asyncio
            import tempfile

            audio_dir = os.path.join(_data_dir, "tts_cache")
            os.makedirs(audio_dir, exist_ok=True)

            # Hash text+voice+rate for caching
            import hashlib
            cache_key = hashlib.md5(f"{text}|{voice}|{rate}".encode()).hexdigest()
            cache_path = os.path.join(audio_dir, f"{cache_key}.mp3")

            if not os.path.exists(cache_path):
                async def _generate():
                    comm = edge_tts.Communicate(text, voice, rate=rate)
                    await comm.save(cache_path)

                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(_generate())
                finally:
                    loop.close()

            return send_file(cache_path, mimetype="audio/mpeg")
        except ImportError:
            return jsonify({"error": "edge-tts not installed. Run: pip install edge-tts"}), 500
        except Exception as e:
            print(f"[TTS] Error: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/tts/voices', methods=['GET'])
    def api_tts_voices():
        """List available edge-tts voices.

        GET ?lang=en (optional filter by language prefix)
        Returns: [{name, gender, locale}]
        """
        try:
            import edge_tts
            import asyncio

            async def _list():
                return await edge_tts.list_voices()

            loop = asyncio.new_event_loop()
            voices = loop.run_until_complete(_list())
            loop.close()

            lang_filter = request.args.get("lang", "").lower()
            results = []
            for v in voices:
                locale = v.get("Locale", "")
                if lang_filter and not locale.lower().startswith(lang_filter):
                    continue
                results.append({
                    "name": v.get("ShortName", ""),
                    "gender": v.get("Gender", ""),
                    "locale": locale,
                })
            return jsonify(results)
        except ImportError:
            return jsonify({"error": "edge-tts not installed"}), 500
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # ---- Browser Automation ----

    @app.route('/api/browser/status', methods=['GET'])
    def api_browser_status():
        try:
            from qor.browser import automation_status
            return jsonify(automation_status())
        except Exception as e:
            return jsonify({"running": False, "error": str(e)})

    @app.route('/api/browser/config', methods=['GET'])
    def api_browser_config():
        """Get browser automation config (available browsers, current choice)."""
        try:
            from qor.browser import (SUPPORTED_BROWSERS, automation_status,
                                      has_real_chrome_profile)
            status = automation_status()
            browsers = []
            for key, info in SUPPORTED_BROWSERS.items():
                browsers.append({
                    "id": key,
                    "label": info.get("label", key),
                    "selected": key == status.get("browser", "chrome"),
                })
            return jsonify({
                "browsers": browsers,
                "selected": status.get("browser", "chrome"),
                "running": status.get("running", False),
                "has_real_profile": has_real_chrome_profile(),
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/browser/config', methods=['POST'])
    def api_browser_config_update():
        """Set default browser for automation."""
        data = request.get_json()
        if not data or 'browser' not in data:
            return jsonify({"error": "Missing 'browser' field"}), 400
        try:
            from qor.browser import set_default_browser
            result = set_default_browser(data['browser'])
            return jsonify({"ok": True, "message": result})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/browser/start', methods=['POST'])
    def api_browser_start():
        """Start visible browser automation session."""
        data = request.get_json() or {}
        browser = data.get('browser')
        profile = data.get('profile_dir',
                           config.get_data_path("browser-profile"))
        try:
            from qor.browser import start_automation
            result = start_automation(browser=browser, profile_dir=profile)
            failed = any(w in result.lower() for w in
                         ["failed", "requires", "error", "not found"])
            return jsonify({"ok": not failed, "message": result})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route('/api/browser/stop', methods=['POST'])
    def api_browser_stop():
        """Stop visible browser."""
        try:
            from qor.browser import stop_automation
            result = stop_automation()
            return jsonify({"ok": True, "message": result})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route('/api/browser/task', methods=['POST'])
    def api_browser_task():
        """Execute an automation task on the visible browser.

        POST {"task": "go to linkedin and remove 100 posts"}
        Returns: {ok, steps, summary, log, final_url}
        """
        data = request.get_json()
        if not data or 'task' not in data:
            return jsonify({"error": "Missing 'task' field"}), 400
        try:
            from qor.browser import run_automation_task
            result = run_automation_task(data['task'])
            return jsonify(result)
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    # ---- Market Categories ----

    @app.route('/api/markets', methods=['GET'])
    def api_markets():
        """List all market categories (for UI tabs)."""
        from qor.config import MARKET_CATEGORIES
        result = []
        for cat_id, cat in MARKET_CATEGORIES.items():
            exchange = cat["exchange"]
            connected = False
            if exchange == "binance":
                eng = runtime._trading_engine
                connected = eng is not None and hasattr(eng, 'client')
            else:
                connected = exchange in getattr(runtime, '_exchange_engines', {})
            result.append({
                "id": cat_id,
                "label": cat["label"],
                "exchange": exchange,
                "quote": cat["quote"],
                "symbol_count": len(cat["symbols"]),
                "exchange_connected": connected,
            })
        return jsonify(result)

    @app.route('/api/markets/<category>', methods=['GET'])
    def api_market_detail(category):
        """Category detail with symbols."""
        from qor.config import MARKET_CATEGORIES
        cat = MARKET_CATEGORIES.get(category)
        if not cat:
            return jsonify({"error": f"Unknown category: {category}"}), 404
        exchange = cat["exchange"]
        connected = False
        if exchange == "binance":
            eng = runtime._trading_engine
            connected = eng is not None and hasattr(eng, 'client')
        else:
            connected = exchange in getattr(runtime, '_exchange_engines', {})
        return jsonify({
            "id": category,
            "label": cat["label"],
            "exchange": exchange,
            "quote": cat["quote"],
            "exchange_connected": connected,
            "symbols": cat["symbols"],
        })

    @app.route('/api/markets/<category>/prices', methods=['GET'])
    def api_market_prices(category):
        """Live prices for all symbols in a category."""
        from qor.config import MARKET_CATEGORIES
        cat = MARKET_CATEGORIES.get(category)
        if not cat:
            return jsonify({"error": f"Unknown category: {category}"}), 404

        exchange = cat["exchange"]
        quote = cat["quote"]
        prices = []

        if exchange == "binance":
            # Use existing _cached_price with spot engine's client
            eng = runtime._trading_engine
            client = eng.client if eng and hasattr(eng, 'client') else None
            for s in cat["symbols"]:
                sym = s["symbol"]
                pair = f"{sym}{quote}"
                entry = {"symbol": sym, "name": s["name"], "price": None, "source": "binance"}
                if client:
                    try:
                        entry["price"] = _cached_price(client, pair)
                    except Exception:
                        pass
                else:
                    entry["error"] = "exchange not connected"
                prices.append(entry)
        else:
            # Upstox or other exchange engines
            eng = getattr(runtime, '_exchange_engines', {}).get(exchange)
            client = eng.client if eng and hasattr(eng, 'client') else None
            for s in cat["symbols"]:
                sym = s["symbol"]
                entry = {"symbol": sym, "name": s["name"], "price": None, "source": exchange}
                if client:
                    try:
                        entry["price"] = client.get_price(sym)
                    except Exception:
                        pass
                else:
                    entry["error"] = "exchange not connected"
                prices.append(entry)

        return jsonify({
            "category": category,
            "quote": quote,
            "prices": prices,
            "timestamp": int(time.time()),
        })

    # ---- Mobile WebRTC Signaling ----
    # In-memory signaling store: {code: {offer, answer, ice_offer, ice_answer, created}}
    _mobile_sessions = {}

    def _cleanup_mobile_sessions():
        """Remove signaling sessions older than 5 minutes."""
        now = time.time()
        expired = [c for c, s in _mobile_sessions.items() if now - s["created"] > 300]
        for c in expired:
            del _mobile_sessions[c]

    def _get_lan_ip():
        """Auto-detect LAN IP using socket trick."""
        import socket as _socket
        try:
            s = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    @app.route('/api/mobile/signal/new', methods=['GET'])
    def api_mobile_signal_new():
        """Generate a new pairing session with code + LAN IP."""
        import random, string
        _cleanup_mobile_sessions()
        code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        _mobile_sessions[code] = {
            "offer": None, "answer": None,
            "ice_offer": [], "ice_answer": [],
            "created": time.time()
        }
        lan_ip = _get_lan_ip()
        port = config.serve.port if hasattr(config.serve, 'port') else 8000
        turn_cfg = (profile or {}).get("turn_config", {})
        has_turn = bool(turn_cfg.get("turn_key_id")) and bool(turn_cfg.get("api_token"))
        return jsonify({"code": code, "ip": lan_ip, "port": port, "turn": has_turn})

    @app.route('/api/mobile/signal/<code>', methods=['POST'])
    def api_mobile_signal_post(code):
        """Store signaling data (offer, answer, or ICE candidates)."""
        _cleanup_mobile_sessions()
        if code not in _mobile_sessions:
            return jsonify({"error": "Unknown or expired code"}), 404
        data = request.get_json()
        if not data or 'type' not in data:
            return jsonify({"error": "Missing 'type'"}), 400
        session = _mobile_sessions[code]
        sig_type = data['type']
        if sig_type == 'offer':
            session['offer'] = data.get('data')
        elif sig_type == 'answer':
            session['answer'] = data.get('data')
        elif sig_type == 'ice_offer':
            session['ice_offer'].append(data.get('data'))
        elif sig_type == 'ice_answer':
            session['ice_answer'].append(data.get('data'))
        else:
            return jsonify({"error": f"Unknown type: {sig_type}"}), 400
        return jsonify({"ok": True})

    @app.route('/api/mobile/signal/<code>', methods=['GET'])
    def api_mobile_signal_get(code):
        """Poll for signaling data."""
        _cleanup_mobile_sessions()
        if code not in _mobile_sessions:
            return jsonify({"error": "Unknown or expired code"}), 404
        session = _mobile_sessions[code]
        sig_type = request.args.get('type', 'offer')
        if sig_type == 'offer':
            return jsonify({"data": session['offer']}) if session['offer'] else (jsonify({"data": None}), 200)
        elif sig_type == 'answer':
            return jsonify({"data": session['answer']}) if session['answer'] else (jsonify({"data": None}), 200)
        elif sig_type == 'ice_offer':
            return jsonify({"data": session['ice_offer']})
        elif sig_type == 'ice_answer':
            return jsonify({"data": session['ice_answer']})
        return jsonify({"error": f"Unknown type: {sig_type}"}), 400

    @app.route('/api/mobile/signal/<code>', methods=['DELETE'])
    def api_mobile_signal_delete(code):
        """Clean up signaling session after connection established."""
        if code in _mobile_sessions:
            del _mobile_sessions[code]
        return jsonify({"ok": True})

    # ── Mobile heartbeat (tracks active mobile connections via HTTP) ──
    _mobile_heartbeat = {"last_seen": 0, "active": False}

    @app.route('/api/mobile/heartbeat', methods=['POST'])
    def api_mobile_heartbeat():
        """Mobile pings this every 10s to signal it's active."""
        _mobile_heartbeat["last_seen"] = time.time()
        _mobile_heartbeat["active"] = True
        return jsonify({"ok": True})

    @app.route('/api/mobile/status', methods=['GET'])
    def api_mobile_status():
        """Desktop checks this to see if mobile is connected."""
        now = time.time()
        active = _mobile_heartbeat["active"] and (now - _mobile_heartbeat["last_seen"]) < 20
        if not active:
            _mobile_heartbeat["active"] = False
        return jsonify({"mobile_connected": active, "last_seen": _mobile_heartbeat["last_seen"]})

    # ── Cloudflare TURN config ──────────────────────────────────

    @app.route('/api/mobile/turn', methods=['GET'])
    def api_mobile_turn_get():
        """Get saved Cloudflare TURN configuration."""
        turn_cfg = (profile or {}).get("turn_config", {})
        has_key = bool(turn_cfg.get("turn_key_id"))
        has_token = bool(turn_cfg.get("api_token"))
        return jsonify({
            "configured": has_key and has_token,
            "turn_key_id": turn_cfg.get("turn_key_id", ""),
            "api_token_set": has_token,
        })

    @app.route('/api/mobile/turn', methods=['POST'])
    def api_mobile_turn_set():
        """Save Cloudflare TURN credentials (Turn Key ID + API Token)."""
        data = request.get_json()
        if not data:
            return jsonify({"error": "Missing body"}), 400
        turn_key_id = data.get("turn_key_id", "").strip()
        api_token = data.get("api_token", "").strip()
        if not turn_key_id or not api_token:
            return jsonify({"error": "Both turn_key_id and api_token are required"}), 400
        turn_cfg = {}
        turn_cfg["turn_key_id"] = turn_key_id
        if crypto:
            turn_cfg["api_token"] = crypto.encrypt_str(api_token)
        else:
            turn_cfg["api_token"] = api_token
        turn_cfg["set_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        saved = False
        try:
            if profile is not None:
                profile["turn_config"] = turn_cfg
            if profile_path:
                with open(profile_path, 'w') as pf:
                    json.dump(profile, pf, indent=2)
            saved = True
        except Exception:
            pass
        return jsonify({"ok": True, "saved": saved})

    @app.route('/api/mobile/turn', methods=['DELETE'])
    def api_mobile_turn_delete():
        """Remove saved Cloudflare TURN credentials."""
        if profile is not None and "turn_config" in profile:
            del profile["turn_config"]
        saved = False
        try:
            if profile_path:
                with open(profile_path, 'w') as pf:
                    json.dump(profile, pf, indent=2)
            saved = True
        except Exception:
            pass
        return jsonify({"ok": True, "saved": saved})

    @app.route('/api/mobile/turn/credentials', methods=['GET'])
    def api_mobile_turn_credentials():
        """Generate short-lived TURN credentials from Cloudflare API.

        Calls: POST https://rtc.live.cloudflare.com/v1/turn/keys/{id}/credentials/generate
        Returns ICE servers array ready to pass to RTCPeerConnection.
        """
        import urllib.request, urllib.error
        turn_cfg = (profile or {}).get("turn_config", {})
        turn_key_id = turn_cfg.get("turn_key_id", "")
        api_token_raw = turn_cfg.get("api_token", "")
        if not turn_key_id or not api_token_raw:
            return jsonify({"ice_servers": [{"urls": "stun:stun.l.google.com:19302"}]})
        # Decrypt token if needed
        api_token = api_token_raw
        if crypto:
            try:
                api_token = crypto.decrypt_str(api_token_raw)
            except Exception:
                api_token = api_token_raw
        # Call Cloudflare TURN API
        url = f"https://rtc.live.cloudflare.com/v1/turn/keys/{turn_key_id}/credentials/generate"
        payload = json.dumps({"ttl": 86400}).encode()
        req = urllib.request.Request(url, data=payload, method='POST', headers={
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        })
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                cf_data = json.loads(resp.read().decode())
            ice_servers = cf_data.get("iceServers", {})
            turn_urls = ice_servers.get("urls", [])
            username = ice_servers.get("username", "")
            credential = ice_servers.get("credential", "")
            return jsonify({"ice_servers": [
                {"urls": "stun:stun.l.google.com:19302"},
                {"urls": turn_urls, "username": username, "credential": credential},
            ]})
        except Exception as e:
            return jsonify({
                "ice_servers": [{"urls": "stun:stun.l.google.com:19302"}],
                "error": str(e),
            })

    # Register dashboard routes (also defined here so they work in both callers)
    _register_dashboard_routes(app, config, runtime)


def run_flask_server(config: QORConfig):
    """Run a simple Flask API server."""
    try:
        from flask import Flask, request, jsonify
        from flask_cors import CORS
    except ImportError:
        print("Install Flask: pip install flask flask-cors")
        return

    app = Flask(__name__)
    CORS(app, origins=config.serve.cors_origins)

    server = QORServer(config)
    server.load()
    server.start_time = time.time()

    @app.route('/generate', methods=['POST'])
    def generate():
        data = request.get_json()
        if not data or 'prompt' not in data:
            return jsonify({"error": "Missing 'prompt' in request body"}), 400

        error = _validate_generate_params(
            data['prompt'], data.get('max_tokens'), data.get('temperature'),
            data.get('top_k'), data.get('top_p'))
        if error:
            return jsonify({"error": error}), 400

        result = server.generate(
            prompt=data['prompt'],
            max_tokens=data.get('max_tokens'),
            temperature=data.get('temperature'),
            top_k=data.get('top_k'),
            top_p=data.get('top_p'),
        )
        return jsonify(result)

    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({"status": "ok", "model_loaded": server.model is not None})

    @app.route('/info', methods=['GET'])
    def info():
        return jsonify(server.get_info())

    @app.route('/generate/stream', methods=['POST'])
    def generate_stream():
        data = request.get_json()
        if not data or 'prompt' not in data:
            return jsonify({"error": "Missing 'prompt' in request body"}), 400

        def event_stream():
            for token in server.generate_stream(
                prompt=data['prompt'],
                max_tokens=data.get('max_tokens'),
                temperature=data.get('temperature'),
                top_k=data.get('top_k'),
                top_p=data.get('top_p'),
            ):
                yield f"data: {json.dumps({'token': token})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"

        return app.response_class(event_stream(), mimetype='text/event-stream')

    @app.route('/chat', methods=['POST'])
    def chat():
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"error": "Missing 'message' in request body"}), 400

        session_id = data.get('session_id', 'default')
        user_name = data.get('user_name', '')
        if session_id not in server.chat_sessions:
            system = getattr(server.config.serve, 'system_prompt', 'You are QOR, a helpful AI assistant.')
            if user_name:
                system = (
                    f"You are QOR (Qora Neuran AI), a friendly and helpful AI companion. "
                    f"The user's name is {user_name}. Address them by name naturally — "
                    f"be warm and personal but not excessive."
                )
            max_turns = getattr(server.config.serve, 'max_history_turns', 10)
            server.chat_sessions[session_id] = ChatSession(system, max_turns)

        session = server.chat_sessions[session_id]
        session.add_user(data['message'])
        prompt = session.format_prompt()

        result = server.generate(prompt=prompt, max_tokens=data.get('max_tokens'))
        answer = result.get('generated_text', result.get('output', ''))
        session.add_assistant(answer)

        # Save to persistent ChatStore if available
        if server.chat_store is not None:
            try:
                server.chat_store.add_turn(session_id, data['message'], {
                    "answer": answer,
                    "confidence": 0.0,
                    "source": "generate",
                    "timestamp": result.get('timestamp', ''),
                })
            except Exception:
                pass

        return jsonify({
            "session_id": session_id,
            "response": answer,
            "tokens_generated": result.get('tokens_generated', 0),
            "time_seconds": result.get('time_seconds', 0),
        })

    print(f"\n  QOR API Server running at http://{config.serve.host}:{config.serve.port}")
    print(f"  Endpoints:")
    print(f"    POST /generate         — Generate text")
    print(f"    POST /generate/stream  — Streaming generation (SSE)")
    print(f"    POST /chat             — Chat with session memory")
    print(f"    GET  /health    — Health check")
    print(f"    GET  /info      — Model info")
    print(f"\n  Example:")
    print(f'    curl -X POST http://localhost:{config.serve.port}/generate \\')
    print(f'      -H "Content-Type: application/json" \\')
    print(f'      -d \'{{"prompt": "Once upon a time"}}\'')
    print()

    app.run(host=config.serve.host, port=config.serve.port, debug=False)


def run_fastapi_server(config: QORConfig):
    """Run a production FastAPI server."""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel
        import uvicorn
    except ImportError:
        print("Install FastAPI: pip install fastapi uvicorn")
        return

    app = FastAPI(title="QOR — The Qore Mind", version="1.0.0")
    app.add_middleware(CORSMiddleware, allow_origins=config.serve.cors_origins,
                       allow_methods=["*"], allow_headers=["*"])

    server = QORServer(config)
    server.load()

    class GenerateRequest(BaseModel):
        prompt: str
        max_tokens: Optional[int] = None
        temperature: Optional[float] = None
        top_k: Optional[int] = None
        top_p: Optional[float] = None

    @app.post("/generate")
    async def generate(req: GenerateRequest):
        error = _validate_generate_params(
            req.prompt, req.max_tokens, req.temperature, req.top_k, req.top_p)
        if error:
            raise HTTPException(status_code=400, detail=error)

        result = server.generate(
            prompt=req.prompt, max_tokens=req.max_tokens,
            temperature=req.temperature, top_k=req.top_k, top_p=req.top_p
        )
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        return result

    @app.get("/health")
    async def health():
        return {"status": "ok", "model_loaded": server.model is not None}

    @app.get("/info")
    async def info():
        return server.get_info()

    @app.post("/generate/stream")
    async def generate_stream_endpoint(req: GenerateRequest):
        from starlette.responses import StreamingResponse

        async def event_stream():
            for token in server.generate_stream(
                prompt=req.prompt, max_tokens=req.max_tokens,
                temperature=req.temperature, top_k=req.top_k, top_p=req.top_p
            ):
                yield f"data: {json.dumps({'token': token})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    class ChatRequest(BaseModel):
        message: str
        session_id: str = "default"
        max_tokens: Optional[int] = None
        user_name: Optional[str] = None

    @app.post("/chat")
    async def chat(req: ChatRequest):
        if req.session_id not in server.chat_sessions:
            system = getattr(server.config.serve, 'system_prompt', 'You are QOR, a helpful AI assistant.')
            if req.user_name:
                system = (
                    f"You are QOR (Qora Neuran AI), a friendly and helpful AI companion. "
                    f"The user's name is {req.user_name}. Address them by name naturally — "
                    f"be warm and personal but not excessive."
                )
            max_turns = getattr(server.config.serve, 'max_history_turns', 10)
            server.chat_sessions[req.session_id] = ChatSession(system, max_turns)

        session = server.chat_sessions[req.session_id]
        session.add_user(req.message)
        prompt = session.format_prompt()

        result = server.generate(prompt=prompt, max_tokens=req.max_tokens)
        answer = result.get('generated_text', result.get('output', ''))
        session.add_assistant(answer)

        # Save to persistent ChatStore if available
        if server.chat_store is not None:
            try:
                server.chat_store.add_turn(req.session_id, req.message, {
                    "answer": answer,
                    "confidence": 0.0,
                    "source": "generate",
                    "timestamp": result.get('timestamp', ''),
                })
            except Exception:
                pass

        return {
            "session_id": req.session_id,
            "response": answer,
            "tokens_generated": result.get('tokens_generated', 0),
            "time_seconds": result.get('time_seconds', 0),
        }

    print(f"\n  QOR API (FastAPI) at http://{config.serve.host}:{config.serve.port}")
    print(f"  Docs at http://{config.serve.host}:{config.serve.port}/docs\n")

    uvicorn.run(app, host=config.serve.host, port=config.serve.port)


def run_full_server(config: QORConfig, checkpoint_path: str = None,
                    tokenizer_path: str = None):
    """Full runtime API server — wires ALL systems like cmd_run().

    Exposes 20+ endpoints for chat, trading, knowledge, tools, and settings.
    Used by the QOR Terminal UI frontend.
    """
    try:
        from flask import Flask, request, jsonify, Response
        from flask_cors import CORS
    except ImportError:
        print("Install Flask: pip install flask flask-cors")
        return

    import threading
    import uuid
    from .continual import ContinualLearner
    from .runtime import QORRuntime
    from .confidence import ConfidenceGate
    from .tools import ToolExecutor
    from .graph import QORGraph, GraphConfig
    from .knowledge import KnowledgeBase
    from .rag import QORRag
    from .plugins import PluginManager
    from .cache import CacheStore
    from .chat import ChatStore
    from .crypto import QORCrypto

    # Resolve all runtime paths
    config.resolve_data_paths()
    data_dir = config.runtime.data_dir

    ckpt = checkpoint_path or os.path.join(config.train.checkpoint_dir, "best_model.pt")
    tok = tokenizer_path

    # Ensure directories exist
    for d in [config.runtime.historical_dir,
              config.train.checkpoint_dir, config.continual.learn_dir,
              config.get_data_path("knowledge"), config.get_data_path("plugins"),
              config.get_data_path("logs"), config.get_data_path("screenshots"),
              config.get_data_path("trading"), config.get_data_path("futures")]:
        os.makedirs(d, exist_ok=True)

    print(f"\n  QOR Full Runtime API Server")
    print(f"  Data directory: {os.path.abspath(data_dir)}")

    # 1. Load model
    learner = ContinualLearner(config)
    learner.load(ckpt)
    print(f"  Model loaded from {ckpt}")

    # 2. ConfidenceGate
    memory_path = config.get_data_path("memory.parquet")
    gate = ConfidenceGate(learner.model, learner.tokenizer, config)
    gate.memory.path = memory_path
    print(f"  Confidence Gate initialized")

    # 3. Knowledge Graph
    graph = None
    try:
        graph_config = config.graph if hasattr(config, 'graph') else GraphConfig()
        graph = QORGraph(graph_config)
        graph.open()
        gate.set_graph(graph)
        gs = graph.stats()
        print(f"  Graph: {gs['node_count']} nodes, {gs['edge_count']} edges")
    except Exception as e:
        print(f"  Graph: skipped ({e})")

    # 4. Knowledge Base
    knowledge_dir = config.get_data_path("knowledge")
    kb = None
    try:
        kb = KnowledgeBase(knowledge_dir)
        kb.load()
        print(f"  Knowledge Base: {len(kb.nodes)} nodes")
    except Exception as e:
        print(f"  Knowledge Base: skipped ({e})")

    # 5. RAG
    rag = None
    try:
        rag = QORRag()
        if os.path.isdir(knowledge_dir):
            rag.add_folder(knowledge_dir)
        gate.set_rag(rag)
        chunk_count = len(rag.store.chunks) if hasattr(rag, 'store') else 0
        print(f"  RAG: {chunk_count} chunks")
    except Exception as e:
        print(f"  RAG: skipped ({e})")

    # 6. Plugins + Tools
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

    try:
        tool_executor = ToolExecutor()
        gate._tool_executor = tool_executor
    except Exception as e:
        print(f"  ToolExecutor: skipped ({e})")

    gate._knowledge_base = kb
    gate._graph = graph
    if plugin_mgr is not None:
        gate._plugin_manager = plugin_mgr

    # 6b. Skills
    skill_loader = None
    try:
        from .skills import SkillLoader
        skills_dir = config.get_data_path("skills")
        os.makedirs(skills_dir, exist_ok=True)
        skill_loader = SkillLoader(skills_dir)
        skill_loader.load_all()
        gate._skill_loader = skill_loader
        print(f"  Skills: {len(skill_loader.skills)} loaded")
    except Exception as e:
        print(f"  Skills: skipped ({e})")

    # 6c. NGRE Brain (4-layer pipeline: Mamba → Graph → Search → Reasoning)
    ngre_brain = None
    try:
        from .ngre import create_ngre_brain
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

    # 7. CacheStore + ChatStore
    crypto = None
    try:
        crypto = QORCrypto(key_path=config.runtime.encryption_key_path)
    except Exception:
        pass

    cache_store = None
    chat_store = None
    try:
        cache_path = config.get_data_path("cache.parquet")
        cache_store = CacheStore(path=cache_path, secret=config.runtime.integrity_secret)
        gate.set_cache(cache_store)
        print(f"  Cache Store: {cache_store.count()} entries")
    except Exception as e:
        print(f"  Cache Store: skipped ({e})")

    try:
        chat_path = config.get_data_path("chat.parquet")
        chat_store = ChatStore(path=chat_path, secret=config.runtime.integrity_secret, crypto=crypto)
        print(f"  Chat Store: {chat_store.count()} messages")
    except Exception as e:
        print(f"  Chat Store: skipped ({e})")

    # 7b. Load user profile + trading credentials from qor-data/profile.json
    profile_path = config.get_data_path("profile.json")
    profile = {}
    if os.path.exists(profile_path):
        try:
            with open(profile_path) as pf:
                profile = json.load(pf)
        except Exception:
            pass
    user_name = profile.get("user_name", "")

    # Load encrypted trading credentials from profile
    if crypto is not None:
        creds = profile.get("trading_credentials")
        if creds and "api_key" in creds and "api_secret" in creds:
            try:
                dec_key = crypto.decrypt_str(creds["api_key"])
                dec_secret = crypto.decrypt_str(creds["api_secret"])
                config.trading.api_key = dec_key
                config.trading.api_secret = dec_secret
                config.trading.testnet = creds.get("testnet", True)
                config.trading.enabled = True
                if hasattr(config, 'futures'):
                    config.futures.api_key = dec_key
                    config.futures.api_secret = dec_secret
                    config.futures.testnet = creds.get("testnet", True)
                    config.futures.enabled = True
                print(f"  Trading credentials: loaded from profile (encrypted)")
            except Exception as e:
                print(f"  Trading credentials: decrypt failed ({e})")

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

    # 8. Runtime
    runtime = QORRuntime(config)
    runtime.start(learner, gate=gate, graph=graph, rag=rag,
                  cache_store=cache_store, chat_store=chat_store,
                  tool_executor=tool_executor)

    # Build system prompt
    identity = "You are QOR (Qora Neuran AI), a helpful AI assistant with access to real-time tools, a knowledge graph, and memory systems. Answer questions accurately using your available context."
    if user_name:
        identity = f"You are QOR (Qora Neuran AI), a personal AI assistant for {user_name}. " + identity[len("You are QOR (Qora Neuran AI), "):]
    # Include profile data so the AI knows the user
    if profile.get("interests"):
        interests_str = ", ".join(str(k) for k in profile["interests"].keys())
        identity += f"\n\nThe user's interests include: {interests_str}. Use this to provide more relevant answers."
    if profile.get("cautions"):
        identity += f"\nImportant cautions for this user: {', '.join(str(c) for c in profile['cautions'])}."
    detail = profile.get("preferred_detail_level", "detailed")
    if detail and detail != "detailed":
        identity += f"\nThe user prefers {detail} responses."
    # Tell the AI about its data sources
    identity += "\n\nYou have access to: user profile, chat history, memory database, knowledge graph, RAG, tool cache, and 52+ real-time tools. When answering, always provide comprehensive analysis with real data."
    # Tell model about its multimodal capabilities
    if getattr(gate.model, 'vision_encoder', None) is not None:
        identity += (" You CAN see and analyze images. When the user provides an image file path, "
                      "you will see the image contents. Describe what you see in detail.")
    if getattr(gate.model, 'audio_encoder', None) is not None:
        identity += (" You CAN hear and analyze audio. When the user provides an audio file path, "
                      "you will hear the audio contents.")
    gate.system_prompt = identity
    gate._user_name = user_name  # for identity.txt {name} replacement

    # ================================================================
    # Flask App with ALL endpoints
    # ================================================================
    app = Flask(__name__)
    CORS(app, origins=["*"])

    server = QORServer(config)
    server.model = learner.model
    server.tokenizer = learner.tokenizer
    server.graph = graph
    server.chat_store = chat_store
    server.start_time = time.time()

    # Register ALL shared API routes (shared with start_api_thread)
    _register_routes(app, config, runtime, learner, gate, graph, rag,
                     cache_store, chat_store, plugin_mgr, skill_loader,
                     tool_executor, crypto, profile, profile_path, server)

    # ---- Routes unique to run_full_server ----

    @app.route('/generate', methods=['POST'])
    def generate():
        data = request.get_json()
        if not data or 'prompt' not in data:
            return jsonify({"error": "Missing 'prompt'"}), 400
        error = _validate_generate_params(
            data['prompt'], data.get('max_tokens'), data.get('temperature'),
            data.get('top_k'), data.get('top_p'))
        if error:
            return jsonify({"error": error}), 400
        result = server.generate(
            prompt=data['prompt'], max_tokens=data.get('max_tokens'),
            temperature=data.get('temperature'), top_k=data.get('top_k'),
            top_p=data.get('top_p'))
        return jsonify(result)

    @app.route('/api/models', methods=['GET'])
    def api_models():
        from .config import PRESETS
        models = []
        size_info = {
            "small": {"params": "5M", "size_gb": 0.05, "requires": "Any CPU"},
            "medium": {"params": "30M", "size_gb": 0.12, "requires": "Any CPU"},
            "large": {"params": "100M", "size_gb": 0.4, "requires": "4GB RAM"},
            "qor3b": {"params": "4.3B", "size_gb": 8.5, "requires": "16GB RAM or GPU"},
            "small_multimodal": {"params": "5M+", "size_gb": 0.1, "requires": "Any CPU"},
            "qor3b_multimodal": {"params": "4.8B", "size_gb": 10.0, "requires": "16GB RAM or GPU"},
        }
        for name in PRESETS:
            info = size_info.get(name, {"params": "?", "size_gb": 0, "requires": "?"})
            models.append({"id": name, "name": name, **info})
        return jsonify(models)

    # ---- Cleanup on shutdown ----
    import atexit

    def _shutdown():
        print("\n  Shutting down QOR runtime...")
        try:
            runtime.stop()
        except Exception:
            pass
        if graph and graph.is_open:
            try:
                graph.close()
            except Exception:
                pass

    atexit.register(_shutdown)

    # ---- Start server ----
    print(f"\n{'='*60}")
    print(f"  QOR Full Runtime API Server")
    print(f"  http://{config.serve.host}:{config.serve.port}")
    print(f"{'='*60}")
    print(f"  API Endpoints:")
    print(f"    GET  /health                 — Health check")
    print(f"    GET  /api/status             — Full runtime status")
    print(f"    GET  /api/tools              — List tools")
    print(f"    GET  /api/skills             — List skills")
    print(f"    GET  /api/models             — Available model sizes")
    print(f"    POST /api/command            — Execute command")
    print(f"    POST /api/chat               — Chat (full routing)")
    print(f"    POST /api/chat/stream        — Chat (SSE streaming)")
    print(f"    GET  /api/chat/sessions      — List chat sessions")
    print(f"    GET  /api/chat/history/<id>  — Session history")
    print(f"    GET  /api/trading/status      — Spot engine status")
    print(f"    GET  /api/trading/positions   — Open spot positions")
    print(f"    GET  /api/trading/trades      — Recent spot trades")
    print(f"    GET  /api/futures/status      — Futures engine status")
    print(f"    GET  /api/futures/positions   — Open futures positions")
    print(f"    GET  /api/futures/trades      — Recent futures trades")
    print(f"    GET  /api/portfolio           — Combined portfolio")
    print(f"    GET  /api/graph/stats         — Knowledge graph stats")
    print(f"    POST /api/graph/query         — Semantic graph query")
    print(f"    GET  /api/memory/stats        — Memory store stats")
    print(f"    GET  /api/settings            — Current settings")
    print(f"    POST /api/settings            — Update setting")
    print(f"    GET  /api/trading/symbols     — Get trading symbols")
    print(f"    POST /api/trading/symbols     — Set trading symbols")
    print(f"    POST /api/trading/credentials — Set API keys")
    print(f"    POST /api/trading/start       — Start engine")
    print(f"    POST /api/trading/stop        — Stop engine")
    print(f"    POST /api/transcribe          — Voice to text (Whisper)")
    print(f"{'='*60}\n")

    import logging as _logging
    _logging.getLogger("werkzeug").setLevel(_logging.ERROR)
    app.run(host=config.serve.host, port=config.serve.port,
            debug=False, threaded=True)


def start_api_thread(config, runtime, learner, gate, graph=None, rag=None,
                     kb=None, cache_store=None, chat_store=None,
                     plugin_mgr=None, skill_loader=None, tool_executor=None,
                     crypto=None, profile=None, profile_path=None):
    """Start API server in a background daemon thread sharing cmd_run()'s objects.

    This lets the UI connect to the same runtime as the interactive CLI.
    """
    import threading
    import logging as _logging

    try:
        from flask import Flask, request, jsonify, Response, send_file
        from flask_cors import CORS
    except ImportError:
        print("  API server: skipped (pip install flask flask-cors)")
        return None

    _logging.getLogger("werkzeug").setLevel(_logging.ERROR)

    app = Flask(__name__)
    CORS(app, origins=["*"])

    server = QORServer(config)
    server.model = learner.model
    server.tokenizer = learner.tokenizer
    server.graph = graph
    server.chat_store = chat_store
    server.start_time = time.time()

    if profile is None:
        profile = {}
    if profile_path is None:
        profile_path = config.get_data_path("profile.json")


    # Register ALL shared API routes (shared with run_full_server)
    _register_routes(app, config, runtime, learner, gate, graph, rag,
                     cache_store, chat_store, plugin_mgr, skill_loader,
                     tool_executor, crypto, profile, profile_path, server)

    # Dashboard routes are now in _register_dashboard_routes (called by _register_routes)

    # REMOVED: all dashboard routes moved to _register_dashboard_routes()
    # Start in daemon thread
    def _run():
        app.run(host=config.serve.host, port=config.serve.port,
                debug=False, threaded=True)

    t = threading.Thread(target=_run, daemon=True, name="qor-api")
    t.start()
    return t

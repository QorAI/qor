"""
QOR Configuration — The Qore Mind
All settings in one place. Nothing hidden.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class TokenizerConfig:
    """Tokenizer settings."""
    type: str = "bpe"                   # "bpe" (train your own) or "pretrained" (use GPT-2's)
    vocab_size: int = 8192              # BPE vocabulary size (8K for small, 32K for medium, 50K for large)
    min_frequency: int = 2              # Minimum word frequency to include
    pretrained_name: str = "gpt2"       # HuggingFace tokenizer name (if type="pretrained")
    save_path: str = "tokenizer.json"   # Where to save trained tokenizer


@dataclass
class ModelConfig:
    """QOR model architecture."""
    d_model: int = 256                  # Hidden dimension
    n_layers: int = 6                   # Number of QOR blocks
    n_heads: int = 8                    # Attention heads
    d_ff: int = 1024                    # FFN inner dimension (4x d_model)
    dropout: float = 0.1                # Dropout rate
    max_seq_len: int = 512              # Maximum sequence length
    vocab_size: int = 8192              # Must match tokenizer

    # CMS (Continuum Memory System)
    cms_levels: int = 3                 # Memory speed levels
    cms_fast_freq: int = 1              # Fast: every step
    cms_med_freq: int = 16              # Medium: every 16 steps
    cms_slow_freq: int = 64             # Slow: every 64 steps

    # Self-Modification
    self_mod_lr: float = 0.02           # Fast weight learning rate
    self_mod_decay: float = 0.95        # Retention gate (alpha)
    surprise_threshold: float = 0.5     # Min surprise to trigger update

    # CfC Liquid Neurons (replaces linear target_pred with CfC-mmRNN)
    use_cfc: bool = True                # Use CfC liquid neurons for self-modification (requires: pip install ncps)
    cfc_neurons: int = 32               # Liquid neurons per CfC cell (NCP wiring)
    cfc_output_size: int = 8            # CfC motor neuron count (projected to out_features)

    # GQA (Grouped Query Attention)
    n_kv_heads: int = 0                 # KV heads (0 = same as n_heads, i.e. MHA)

    # Sliding Window Attention
    sliding_window: int = 0             # 0 = full attention, >0 = window size

    # RoPE
    rope_theta: float = 10000.0         # RoPE base frequency (qor3b uses 5,000,000)
    no_rope_layer_interval: int = 0     # Skip RoPE every N layers (0 = always use RoPE)

    # CMS per-level FFN sizes (overrides proportional split when set)
    cms_level_ff_sizes: list = field(default_factory=list)  # e.g. [11008, 2752, 1376]

    # S4/Mamba State Space (long-range temporal patterns in QORBlock)
    use_s4: bool = False                # Enable S4/Mamba state-space layer in QORBlock
    s4_d_state: int = 16                # State vector dimension
    s4_d_conv: int = 4                  # Local convolution kernel size

    # torch.compile
    compile: bool = False               # Enable torch.compile
    compile_mode: str = "reduce-overhead"  # Compile mode

    @property
    def head_dim(self):
        return self.d_model // self.n_heads

    def __post_init__(self):
        # n_kv_heads=0 means use n_heads (standard MHA)
        if self.n_kv_heads <= 0:
            self.n_kv_heads = self.n_heads


@dataclass
class TrainConfig:
    """Training settings."""
    # Optimization
    learning_rate: float = 3e-4
    min_lr: float = 1e-5
    weight_decay: float = 0.01
    batch_size: int = 8
    grad_accumulation_steps: int = 1    # Simulate larger batches
    max_grad_norm: float = 1.0

    # Schedule
    max_steps: int = 10000
    warmup_steps: int = 500
    eval_every: int = 500
    save_every: int = 1000
    log_every: int = 50

    # Data
    data_dir: str = "data"              # Where training text files live
    val_split: float = 0.05             # 5% validation

    # Checkpoints
    checkpoint_dir: str = "checkpoints"
    max_checkpoints: int = 5            # Keep last N checkpoints

    # Hardware
    device: str = "auto"                # auto / cpu / cuda / mps
    mixed_precision: bool = True        # FP16 training (faster on GPU)
    num_workers: int = 2                # Data loading workers
    distributed: bool = False           # Multi-GPU DDP training
    streaming: bool = False             # Use streaming dataset
    gradient_checkpointing: bool = False # Trade compute for memory


@dataclass
class ServeConfig:
    """API serving settings."""
    host: str = "0.0.0.0"
    port: int = 8000
    model_path: str = "checkpoints/best_model.pt"
    tokenizer_path: str = "tokenizer.json"
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    cors_origins: list = field(default_factory=lambda: ["*"])
    quantize: bool = False              # INT8 quantization for CPU
    system_prompt: str = "You are QOR, a helpful AI assistant."
    max_history_turns: int = 10         # Chat history length
    chat_template: str = "chatml"       # Chat format template


@dataclass
class VisionConfig:
    """Vision encoder settings."""
    enabled: bool = False               # Only activate when needed
    image_size: int = 224               # Input image size (224 for standard, 28 for MNIST)
    patch_size: int = 16                # Patch size (16 for standard, 7 for MNIST)
    in_channels: int = 3                # 3 for RGB, 1 for grayscale (MNIST)
    use_vqvae: bool = False             # Enable VQ-VAE discrete tokenizer
    codebook_size: int = 512            # VQ-VAE codebook entries
    codebook_dim: int = 64              # VQ-VAE codebook embedding dimension

    # Pretrained encoder (SigLIP)
    use_pretrained: bool = False        # Use frozen pretrained encoder (SigLIP)
    pretrained_model: str = "google/siglip-so400m-patch14-384"
    pretrained_hidden_size: int = 1152  # SigLIP so400m output dim
    pretrained_image_size: int = 384    # SigLIP input resolution
    pretrained_patch_size: int = 14     # SigLIP patch size
    bridge_bottleneck: int = 1024       # Bridge MLP bottleneck dim

    @property
    def n_patches(self) -> int:
        if self.use_pretrained:
            return (self.pretrained_image_size // self.pretrained_patch_size) ** 2
        return (self.image_size // self.patch_size) ** 2


@dataclass
class AudioConfig:
    """Audio encoder settings."""
    enabled: bool = False               # Only activate when needed
    n_mels: int = 80                    # Mel spectrogram frequency bins
    sample_rate: int = 16000            # Audio sample rate
    n_fft: int = 400                    # FFT window size
    hop_length: int = 160               # FFT hop length
    frame_stride: int = 4               # Temporal downsampling stride
    max_audio_tokens: int = 256         # Max audio tokens (5 sec ~ 125 tokens)

    # Pretrained encoder (Whisper)
    use_pretrained: bool = False        # Use frozen pretrained encoder (Whisper)
    pretrained_model: str = "openai/whisper-small"
    pretrained_hidden_size: int = 768   # Whisper small encoder output dim
    bridge_bottleneck: int = 1024       # Bridge MLP bottleneck dim


@dataclass
class ContinualConfig:
    """Continual learning settings."""
    # How to ingest new knowledge
    learn_dir: str = "learn"            # Drop new text files here
    learn_rate: float = 1e-4            # Lower than initial training
    learn_steps_per_file: int = 200     # Steps per new document
    protect_slow_layers: bool = True    # Freeze slow CMS during learning
    surprise_gate: bool = True          # Only learn from surprising tokens
    surprise_threshold: float = 1.0     # Higher = more selective


@dataclass
class TradingConfig:
    """AI automated trading settings (Binance Spot Demo Mode)."""
    enabled: bool = False                           # Must opt-in
    testnet: bool = True                            # True = demo mode (safety default)
    api_key: str = ""                               # Binance API key (env: QOR_BINANCE_KEY)
    api_secret: str = ""                            # Binance API secret (env: QOR_BINANCE_SECRET)
    symbols: list = field(default_factory=lambda: ["BTC", "ETH"])
    check_interval_seconds: int = 300               # 5 minutes
    allocated_fund: float = 0.0                     # Max USDT to use (0 = use full balance)
    max_position_pct: float = 10.0                  # Max % of allocated fund per trade
    max_open_positions: int = 3                     # Max simultaneous positions
    stop_loss_atr_mult: float = 2.0                 # SL = entry - N * ATR
    take_profit_atr_mult: float = 3.0               # TP = entry + N * ATR
    trailing_stop: bool = True                      # Adjust SL as price moves in favor
    min_risk_reward: float = 1.2                    # Minimum R:R ratio to enter
    slippage_pct: float = 0.05                      # Estimated slippage % (deducted from R:R)
    cooldown_minutes: int = 30                      # Wait after closing a trade
    data_dir: str = "trading"                       # Resolved by resolve_data_paths()
    # DCA (Dollar Cost Average)
    dca_enabled: bool = True                        # Add to losing positions at support
    dca_max_adds: int = 2                           # Max DCA orders per position
    dca_drop_pct: float = 5.0                       # Add when price drops N% from entry
    dca_multiplier: float = 1.5                     # Each DCA order is N× the previous size
    # Partial take profit
    partial_tp_enabled: bool = True                 # Sell portions at TP levels
    partial_tp1_pct: float = 50.0                   # Sell N% of position at TP1
    partial_tp2_pct: float = 100.0                  # Sell remaining at TP2 (100 = close all)
    # Break-even protection
    move_sl_to_be: bool = True                      # Move SL to breakeven after TP1 hit
    # Trading mode — Elder Triple Screen: trend TF=veto, setup TFs=confluence, entry TF=timing
    # scalp:  trend=1h, setup=15m+30m, entry=5m | stable: trend=daily, setup=4h, entry=1h
    # secure: trend=weekly, setup=daily, entry=4h
    trade_mode: str = "scalp"                       # "scalp", "stable", "secure"
    # Sentiment gates (Polymarket + Fear & Greed + Calendar)
    poly_block_threshold: float = 0.35              # Block entry if Poly against direction (0-1)
    fg_extreme_greed: int = 85                      # Block LONG if F&G > this
    fg_extreme_fear: int = 15                       # Block SHORT if F&G < this (contrarian)
    calendar_block_minutes: int = 60                # Skip entry if high-impact event within N min
    # Kelly Criterion position sizing
    kelly_enabled: bool = True                      # Use Kelly for dynamic position sizing
    kelly_fraction: float = 0.5                     # Fraction of full Kelly (0.5 = half-kelly)
    kelly_min_trades: int = 10                      # Min closed trades before Kelly kicks in
    kelly_min_pct: float = 2.0                      # Floor: never size below N% if edge exists

    def __post_init__(self):
        if not self.api_key:
            self.api_key = os.environ.get("QOR_BINANCE_KEY", "")
        if not self.api_secret:
            self.api_secret = os.environ.get("QOR_BINANCE_SECRET", "")
        if self.trade_mode not in ("scalp", "stable", "secure"):
            self.trade_mode = "scalp"


@dataclass
class FuturesConfig:
    """Binance USDT-M Futures trading settings."""
    enabled: bool = False
    testnet: bool = True
    api_key: str = ""
    api_secret: str = ""
    symbols: list = field(default_factory=lambda: ["BTC", "ETH"])
    check_interval_seconds: int = 300       # 5 minutes
    leverage: int = 5                       # Default leverage (1-10)
    max_leverage: int = 10                  # Max allowed
    margin_type: str = "ISOLATED"           # ISOLATED or CROSSED
    allocated_fund: float = 0.0             # Max USDT to use (0 = use full balance)
    max_position_pct: float = 10.0          # Max % of allocated fund per trade
    max_open_positions: int = 3
    stop_loss_atr_mult: float = 2.0
    take_profit_atr_mult: float = 3.0
    trailing_stop: bool = True
    min_risk_reward: float = 1.2
    slippage_pct: float = 0.05                  # Estimated slippage % (deducted from R:R)
    cooldown_minutes: int = 30
    data_dir: str = "futures"
    # DCA
    dca_enabled: bool = True
    dca_max_adds: int = 2
    dca_drop_pct: float = 5.0
    dca_multiplier: float = 1.5
    # Partial TP
    partial_tp_enabled: bool = True
    partial_tp1_pct: float = 50.0
    partial_tp2_pct: float = 100.0
    move_sl_to_be: bool = True
    # Futures-specific
    hedge_mode: bool = True                 # Enable hedge mode (simultaneous LONG+SHORT)
    multi_asset_mode: bool = True           # Enable multi-asset margin mode
    funding_rate_threshold: float = 0.03    # Warn if funding > 3%
    max_funding_cost_pct: float = 0.1       # Close if cumulative funding > 0.1% of position
    maintenance_margin_pct: float = 0.006   # Maintenance margin for liq price estimate (0.6%)
    # Trading mode — Elder Triple Screen: trend TF=veto, setup TFs=confluence, entry TF=timing
    # scalp:  trend=1h, setup=15m+30m, entry=5m | stable: trend=daily, setup=4h, entry=1h
    # secure: trend=weekly, setup=daily, entry=4h
    trade_mode: str = "scalp"               # "scalp", "stable", "secure"
    # Sentiment gates (Polymarket + Fear & Greed + Calendar)
    poly_block_threshold: float = 0.35              # Block entry if Poly against direction (0-1)
    fg_extreme_greed: int = 85                      # Block LONG if F&G > this
    fg_extreme_fear: int = 15                       # Block SHORT if F&G < this (contrarian)
    calendar_block_minutes: int = 60                # Skip entry if high-impact event within N min
    # Kelly Criterion position sizing
    kelly_enabled: bool = True                      # Use Kelly for dynamic position sizing
    kelly_fraction: float = 0.5                     # Fraction of full Kelly (0.5 = half-kelly)
    kelly_min_trades: int = 10                      # Min closed trades before Kelly kicks in
    kelly_min_pct: float = 2.0                      # Floor: never size below N% if edge exists

    def __post_init__(self):
        if not self.api_key:
            self.api_key = os.environ.get("QOR_BINANCE_KEY", "")
        if not self.api_secret:
            self.api_secret = os.environ.get("QOR_BINANCE_SECRET", "")
        self.leverage = max(1, min(self.leverage, self.max_leverage))
        if self.trade_mode not in ("scalp", "stable", "secure"):
            self.trade_mode = "scalp"


@dataclass
class ExchangeKeys:
    """API credentials for a trading exchange or data provider.

    Same pattern as Binance (api_key + api_secret + testnet).
    Add one entry per exchange/broker you want to connect.

    Crypto exchanges:  coinbase, okx, bybit, kraken, kucoin, gate_io
    Stock brokers:     alpaca, interactive_brokers, tradier
    Commodity/Forex:   oanda, saxo, ig_markets

    Environment variable fallback:
        QOR_{NAME}_KEY, QOR_{NAME}_SECRET, QOR_{NAME}_PASSPHRASE
        e.g. QOR_OKX_KEY, QOR_COINBASE_SECRET
    """
    name: str = ""              # Exchange identifier: "coinbase", "okx", "bybit", etc.
    api_key: str = ""           # API key
    api_secret: str = ""        # API secret
    passphrase: str = ""        # Some exchanges require this (OKX, Coinbase Pro)
    access_token: str = ""      # OAuth2 access token (Upstox, etc.)
    testnet: bool = True        # Use testnet/sandbox by default (safety)
    enabled: bool = False       # Must opt-in
    base_url: str = ""          # Custom base URL (empty = use exchange default)
    futures_url: str = ""       # Futures API URL (empty = use exchange default)
    # Symbols to trade on this exchange (e.g. ["SOL","AVAX"] for crypto,
    # ["AAPL","MSFT"] for stocks, ["XAU_USD","XAG_USD"] for commodities)
    symbols: list = field(default_factory=list)
    quote_currency: str = "USD" # Quote currency: "USDT" for crypto, "USD" for stocks/forex
    check_interval_seconds: int = 300  # Per-symbol check interval

    def __post_init__(self):
        name_upper = self.name.upper().replace("-", "_").replace(" ", "_")
        if name_upper and not self.api_key:
            self.api_key = os.environ.get(f"QOR_{name_upper}_KEY", "")
        if name_upper and not self.api_secret:
            self.api_secret = os.environ.get(f"QOR_{name_upper}_SECRET", "")
        if name_upper and not self.passphrase:
            self.passphrase = os.environ.get(f"QOR_{name_upper}_PASSPHRASE", "")
        if name_upper and not self.access_token:
            self.access_token = os.environ.get(f"QOR_{name_upper}_TOKEN", "")


# Default exchange URLs (used when ExchangeKeys.base_url is empty)
EXCHANGE_DEFAULTS = {
    # Crypto
    "binance":   {"spot": "https://api.binance.com/api/v3",
                  "futures": "https://fapi.binance.com/fapi/v1",
                  "testnet_spot": "https://testnet.binance.vision/api/v3",
                  "testnet_futures": "https://testnet.binancefuture.com/fapi/v1"},
    "coinbase":  {"spot": "https://api.coinbase.com/v2",
                  "advanced": "https://api.coinbase.com/api/v3",
                  "testnet_spot": "https://api-public.sandbox.exchange.coinbase.com"},
    "okx":       {"spot": "https://www.okx.com/api/v5",
                  "testnet_spot": "https://www.okx.com/api/v5"},  # OKX uses header for demo
    "bybit":     {"spot": "https://api.bybit.com/v5",
                  "testnet_spot": "https://api-testnet.bybit.com/v5"},
    "kraken":    {"spot": "https://api.kraken.com/0",
                  "futures": "https://futures.kraken.com/derivatives/api/v3"},
    "kucoin":    {"spot": "https://api.kucoin.com/api/v1",
                  "futures": "https://api-futures.kucoin.com/api/v1",
                  "testnet_spot": "https://openapi-sandbox.kucoin.com/api/v1"},
    "gate_io":   {"spot": "https://api.gateio.ws/api/v4",
                  "futures": "https://api.gateio.ws/api/v4"},
    # Stock brokers
    "alpaca":    {"spot": "https://api.alpaca.markets/v2",
                  "testnet_spot": "https://paper-api.alpaca.markets/v2"},
    "tradier":   {"spot": "https://api.tradier.com/v1",
                  "testnet_spot": "https://sandbox.tradier.com/v1"},
    # Forex / Commodity
    "oanda":     {"spot": "https://api-fxtrade.oanda.com/v3",
                  "testnet_spot": "https://api-fxpractice.oanda.com/v3"},
    # Indian markets (NSE/BSE/MCX)
    "upstox":    {"spot": "https://api.upstox.com/v2",
                  "hft": "https://api-hft.upstox.com/v3"},
}


# Predefined market categories with top symbols and exchange mapping (for UI tabs)
MARKET_CATEGORIES = {
    "crypto": {
        "label": "Cryptocurrency",
        "exchange": "binance",
        "quote": "USDT",
        "symbols": [
            {"symbol": "BTC", "name": "Bitcoin"},
            {"symbol": "ETH", "name": "Ethereum"},
            {"symbol": "SOL", "name": "Solana"},
            {"symbol": "BNB", "name": "BNB"},
            {"symbol": "XRP", "name": "Ripple"},
            {"symbol": "ADA", "name": "Cardano"},
            {"symbol": "DOGE", "name": "Dogecoin"},
            {"symbol": "AVAX", "name": "Avalanche"},
            {"symbol": "DOT", "name": "Polkadot"},
            {"symbol": "LINK", "name": "Chainlink"},
        ],
    },
    "commodities": {
        "label": "Commodities (MCX)",
        "exchange": "upstox",
        "quote": "INR",
        "symbols": [
            {"symbol": "GOLDM", "name": "Gold Mini"},
            {"symbol": "SILVERM", "name": "Silver Mini"},
            {"symbol": "CRUDEOILM", "name": "Crude Oil Mini"},
            {"symbol": "NATURALGAS", "name": "Natural Gas"},
            {"symbol": "COPPER", "name": "Copper"},
            {"symbol": "ZINC", "name": "Zinc"},
            {"symbol": "LEAD", "name": "Lead"},
            {"symbol": "ALUMINIUM", "name": "Aluminium"},
            {"symbol": "NICKEL", "name": "Nickel"},
            {"symbol": "COTTONCANDY", "name": "Cotton"},
        ],
    },
    "indices": {
        "label": "Indices (NSE/BSE)",
        "exchange": "upstox",
        "quote": "INR",
        "symbols": [
            {"symbol": "NIFTY 50", "name": "Nifty 50"},
            {"symbol": "NIFTY BANK", "name": "Bank Nifty"},
            {"symbol": "SENSEX", "name": "BSE Sensex"},
            {"symbol": "NIFTY IT", "name": "Nifty IT"},
            {"symbol": "NIFTY FIN SERVICE", "name": "Nifty Financial"},
            {"symbol": "NIFTY PHARMA", "name": "Nifty Pharma"},
            {"symbol": "NIFTY AUTO", "name": "Nifty Auto"},
            {"symbol": "NIFTY FMCG", "name": "Nifty FMCG"},
            {"symbol": "NIFTY METAL", "name": "Nifty Metal"},
            {"symbol": "NIFTY ENERGY", "name": "Nifty Energy"},
        ],
    },
    "equities": {
        "label": "Indian Equities (NSE)",
        "exchange": "upstox",
        "quote": "INR",
        "symbols": [
            {"symbol": "RELIANCE", "name": "Reliance Industries"},
            {"symbol": "TCS", "name": "Tata Consultancy"},
            {"symbol": "HDFCBANK", "name": "HDFC Bank"},
            {"symbol": "INFY", "name": "Infosys"},
            {"symbol": "ICICIBANK", "name": "ICICI Bank"},
            {"symbol": "HINDUNILVR", "name": "Hindustan Unilever"},
            {"symbol": "SBIN", "name": "State Bank of India"},
            {"symbol": "BHARTIARTL", "name": "Bharti Airtel"},
            {"symbol": "ITC", "name": "ITC Limited"},
            {"symbol": "KOTAKBANK", "name": "Kotak Mahindra Bank"},
        ],
    },
    "nse_futures": {
        "label": "NSE Futures (F&O)",
        "exchange": "upstox",
        "quote": "INR",
        "symbols": [
            {"symbol": "NIFTY", "name": "Nifty 50 Futures"},
            {"symbol": "BANKNIFTY", "name": "Bank Nifty Futures"},
            {"symbol": "FINNIFTY", "name": "Fin Nifty Futures"},
            {"symbol": "RELIANCE", "name": "Reliance Futures"},
            {"symbol": "TCS", "name": "TCS Futures"},
            {"symbol": "INFY", "name": "Infosys Futures"},
            {"symbol": "HDFCBANK", "name": "HDFC Bank Futures"},
            {"symbol": "ICICIBANK", "name": "ICICI Bank Futures"},
            {"symbol": "SBIN", "name": "SBI Futures"},
            {"symbol": "BHARTIARTL", "name": "Bharti Airtel Futures"},
        ],
    },
    "mcx_futures": {
        "label": "MCX Futures",
        "exchange": "upstox",
        "quote": "INR",
        "symbols": [
            {"symbol": "GOLD", "name": "Gold Futures"},
            {"symbol": "GOLDM", "name": "Gold Mini Futures"},
            {"symbol": "SILVER", "name": "Silver Futures"},
            {"symbol": "SILVERM", "name": "Silver Mini Futures"},
            {"symbol": "CRUDEOIL", "name": "Crude Oil Futures"},
            {"symbol": "CRUDEOILM", "name": "Crude Oil Mini"},
            {"symbol": "NATURALGAS", "name": "Natural Gas Futures"},
            {"symbol": "COPPER", "name": "Copper Futures"},
            {"symbol": "ZINC", "name": "Zinc Futures"},
            {"symbol": "ALUMINIUM", "name": "Aluminium Futures"},
        ],
    },
    "us_equities": {
        "label": "US Equities",
        "exchange": "alpaca",
        "quote": "USD",
        "symbols": [
            {"symbol": "AAPL", "name": "Apple"},
            {"symbol": "MSFT", "name": "Microsoft"},
            {"symbol": "GOOGL", "name": "Alphabet"},
            {"symbol": "AMZN", "name": "Amazon"},
            {"symbol": "NVDA", "name": "NVIDIA"},
            {"symbol": "META", "name": "Meta Platforms"},
            {"symbol": "TSLA", "name": "Tesla"},
            {"symbol": "JPM", "name": "JPMorgan Chase"},
            {"symbol": "V", "name": "Visa"},
            {"symbol": "UNH", "name": "UnitedHealth"},
        ],
    },
    "us_commodities": {
        "label": "US Commodities",
        "exchange": "oanda",
        "quote": "USD",
        "symbols": [
            {"symbol": "XAU_USD", "name": "Gold"},
            {"symbol": "XAG_USD", "name": "Silver"},
            {"symbol": "WTICO_USD", "name": "WTI Crude Oil"},
            {"symbol": "BCO_USD", "name": "Brent Crude Oil"},
            {"symbol": "NATGAS_USD", "name": "Natural Gas"},
            {"symbol": "XCU_USD", "name": "Copper"},
            {"symbol": "XPT_USD", "name": "Platinum"},
            {"symbol": "XPD_USD", "name": "Palladium"},
            {"symbol": "CORN_USD", "name": "Corn"},
            {"symbol": "WHEAT_USD", "name": "Wheat"},
        ],
    },
}


@dataclass
class RuntimeConfig:
    """Background read loop + cleanup settings.
    ReadLoop stores data directly to DB (no batching). Periodic cleanup prunes old data."""
    data_dir: str = "qor-data"           # Root directory for ALL runtime data
    read_interval: int = 30              # Seconds between READ cycles
    cleanup_every_hours: float = 1.0     # Hours between cleanup cycles
    historical_dir: str = "historical"   # Permanent archive (never deleted)
    live_retention_days: int = 7         # Delete live memory entries older than this
    cms_slow_decay_rate: float = 0.001     # Per-consolidation weight decay for slow CMS (small = gentle)
    graph_gc_min_confidence: float = 0.1   # Prune graph edges below this confidence
    checkpoint_rotation: bool = True        # Enable daily/weekly/monthly/yearly rotation
    cache_default_ttl_minutes: int = 60        # Default TTL for cached tool results
    chat_retention_days: int = 90              # Auto-delete chat sessions older than this
    integrity_secret: str = ""                 # Hash chain secret (empty = plain SHA-256)
    encryption_key_path: str = ""              # Path to Fernet key file (empty = no encryption)
    cortex_retrain_hours: float = 6.0       # Hours between CORTEX kline retraining (0 = disabled)
    cortex_retrain_days: int = 7             # Days of recent klines to fetch for retraining
    enable_read_loop: bool = False           # Background API calls (disabled — tools called on-demand)
    enable_ingestion: bool = True             # 24/7 knowledge ingestion daemon (PRD §22) — always on
    query_pool_workers: int = 3              # Concurrent query worker threads (chat queries run in background)
    query_pool_enabled: bool = True          # Enable parallel query pool (False = synchronous fallback)
    chat_io_workers: int = 10               # Chat-dedicated I/O pool (tool calls from gate.answer)
    system_io_workers: int = 6              # System I/O pool (ingestion, background tasks — NOT chat)
    # Watched assets — ingestion + session tracker auto-cover these
    # Trading symbols are auto-merged in. Add non-traded assets here.
    # Format: "BTC", "ETH", "AAPL", "gold", "EUR/USD", etc.
    watch_assets: list = field(default_factory=list)
    read_sources: list = field(default_factory=lambda: [
        {"tool": "crypto_price", "query": "bitcoin", "interval": 60},
        {"tool": "crypto_price", "query": "ethereum", "interval": 60},
        {"tool": "hacker_news", "query": "", "interval": 300},
        {"tool": "news", "query": "technology", "interval": 600},
        {"tool": "weather", "query": "weather in New York", "interval": 1800},
    ])


@dataclass
class NGREConfig:
    """NGRE (Neural Graph Reasoning Engine) brain settings.
    4-layer architecture: Mamba temporal → Graph memory → Interference search → Reasoning."""

    # Layer 1: Mamba temporal module
    d_hidden: int = 768                     # Mamba hidden dimension (matches mamba-130m)
    mamba_checkpoint: str = ""              # Path to mamba_block.safetensors
    mamba_config: str = ""                  # Path to mamba_block_config.json
    enable_mamba: bool = True               # Enable Mamba temporal module

    # Layer 3: Interference search
    d_embedding: int = 768                  # Node embedding dimension
    search_max_iterations: int = 7          # Max Grover iterations
    search_default_k: int = 32             # Default top-k retrieval
    oracle_hidden: int = 768                # Oracle network hidden size

    # Layer 4: Reasoning (Phase C)
    reasoning_steps: int = 5                # Max GRU refinement steps
    reasoning_halt_threshold: float = 0.8   # Confidence to stop reasoning
    reasoning_n_heads: int = 8              # Cross-attention heads

    # Graph bootstrap (Phase 2 training)
    bootstrap_surprise_threshold: float = 2.0  # Surprise to trigger node creation
    bootstrap_batch_size: int = 16          # Sentences per batch during bootstrap
    bootstrap_max_nodes: int = 100_000      # Max nodes to create during bootstrap

    # Hebbian learning
    hebbian_lr: float = 0.001               # Hebbian learning rate (per query)
    hebbian_positive_reward: float = 1.0    # Reward for good answers / thumbs up
    hebbian_negative_reward: float = -1.0   # Penalty for corrections


def _default_graph_config():
    """Lazy import to avoid circular imports with qor.graph."""
    from qor.graph import GraphConfig
    return GraphConfig()


@dataclass
class QORConfig:
    """Master config — everything in one place."""
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    serve: ServeConfig = field(default_factory=ServeConfig)
    continual: ContinualConfig = field(default_factory=ContinualConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    graph: object = field(default_factory=_default_graph_config)
    trading: TradingConfig = field(default_factory=TradingConfig)
    futures: FuturesConfig = field(default_factory=FuturesConfig)
    ngre: NGREConfig = field(default_factory=NGREConfig)
    # Additional exchanges/brokers — list of ExchangeKeys dicts
    # Example: [{"name": "okx", "api_key": "...", "api_secret": "...", "enabled": True}]
    exchanges: list = field(default_factory=list)

    def get_exchange(self, name: str) -> Optional['ExchangeKeys']:
        """Get exchange credentials by name. Returns None if not configured."""
        name_lower = name.lower().replace("-", "_").replace(" ", "_")
        for ex in self.exchanges:
            if isinstance(ex, ExchangeKeys):
                if ex.name.lower() == name_lower:
                    return ex
            elif isinstance(ex, dict):
                if ex.get("name", "").lower() == name_lower:
                    return ExchangeKeys(**ex)
        return None

    def get_exchange_url(self, name: str, market: str = "spot") -> str:
        """Get the correct URL for an exchange (respects testnet setting).

        Args:
            name: Exchange name ("binance", "okx", "bybit", etc.)
            market: "spot" or "futures"
        Returns:
            Base URL string
        """
        ex = self.get_exchange(name)
        # Check for custom URL override
        if ex:
            if market == "futures" and ex.futures_url:
                return ex.futures_url
            if ex.base_url:
                return ex.base_url

        defaults = EXCHANGE_DEFAULTS.get(name.lower(), {})
        if not defaults:
            return ""

        if ex and ex.testnet:
            key = f"testnet_{market}"
            if key in defaults:
                return defaults[key]

        return defaults.get(market, "")

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        print(f"Config saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'QORConfig':
        with open(path) as f:
            data = json.load(f)
        config = cls()
        for section_name, section_data in data.items():
            if section_name == "exchanges" and isinstance(section_data, list):
                config.exchanges = [
                    ExchangeKeys(**ex) if isinstance(ex, dict) else ex
                    for ex in section_data
                ]
                continue
            section = getattr(config, section_name, None)
            if section is None:
                continue
            if isinstance(section_data, dict):
                for key, value in section_data.items():
                    if hasattr(section, key):
                        setattr(section, key, value)
        return config

    @classmethod
    def small(cls) -> 'QORConfig':
        """~5M params — runs on any computer."""
        c = cls()
        c.model.d_model = 256
        c.model.n_layers = 6
        c.model.d_ff = 1024
        c.model.n_heads = 8
        c.model.n_kv_heads = 4            # GQA: 4 KV heads
        c.model.max_seq_len = 512
        c.tokenizer.vocab_size = 8192
        c.model.vocab_size = 8192
        c.train.batch_size = 8
        return c

    @classmethod
    def medium(cls) -> 'QORConfig':
        """~30M params — needs 4GB+ GPU."""
        c = cls()
        c.model.d_model = 512
        c.model.n_layers = 8
        c.model.d_ff = 2048
        c.model.n_heads = 8
        c.model.n_kv_heads = 2            # GQA: 2 KV heads
        c.model.max_seq_len = 512
        c.tokenizer.vocab_size = 16384
        c.model.vocab_size = 16384
        c.train.batch_size = 4
        return c

    @classmethod
    def large(cls) -> 'QORConfig':
        """~100M params — needs 8GB+ GPU."""
        c = cls()
        c.model.d_model = 768
        c.model.n_layers = 12
        c.model.d_ff = 3072
        c.model.n_heads = 12
        c.model.n_kv_heads = 4            # GQA: 4 KV heads
        c.model.max_seq_len = 1024
        c.tokenizer.vocab_size = 32000
        c.model.vocab_size = 32000
        c.train.batch_size = 2
        return c

    @classmethod
    def qor3b(cls) -> 'QORConfig':
        """~4.4B params — initialized from a 3B donor model (Apache 2.0).
        QOR-unique layers (self-mod, medium/slow CMS) start as passthrough
        and grow via self-learning."""
        c = cls()
        c.model.d_model = 2048
        c.model.n_layers = 36
        c.model.n_heads = 16
        c.model.n_kv_heads = 4
        c.model.d_ff = 11008            # Only used as fallback; cms_level_ff_sizes overrides
        c.model.max_seq_len = 8192
        c.model.vocab_size = 128256
        c.model.dropout = 0.0
        c.model.rope_theta = 5000000.0
        c.model.no_rope_layer_interval = 4
        c.model.cms_level_ff_sizes = [11008, 2752, 1376]  # fast=donor, medium=1/4, slow=1/8
        c.tokenizer.type = "pretrained"
        c.tokenizer.vocab_size = 128256
        c.model.use_cfc = True              # CfC liquid neurons in SelfModifyingLinear
        c.model.cfc_neurons = 32             # NCP wiring: sensory → inter → command → motor
        c.model.cfc_output_size = 8          # Motor neuron count
        c.model.use_s4 = True               # S4/Mamba state-space scan in QORBlock
        c.train.batch_size = 1
        c.train.gradient_checkpointing = True
        return c

    @classmethod
    def small_multimodal(cls) -> 'QORConfig':
        """~5M params + vision/audio — MNIST-ready."""
        c = cls.small()
        c.vision.enabled = True
        c.vision.image_size = 28
        c.vision.patch_size = 7
        c.vision.in_channels = 1          # Grayscale
        c.audio.enabled = True
        return c

    @classmethod
    def qor3b_multimodal(cls) -> 'QORConfig':
        """qor3b + pretrained SigLIP vision + Whisper audio encoders.
        Frozen encoders connected via trainable bridge layers."""
        c = cls.qor3b()
        c.vision.enabled = True
        c.vision.use_pretrained = True     # SigLIP so400m (frozen)
        c.audio.enabled = True
        c.audio.use_pretrained = True      # Whisper small (frozen)
        return c

    def resolve_data_paths(self):
        """Rebase all runtime paths under data_dir.

        Call this once at startup (cmd_run, cmd_consolidate, create_agent)
        to consolidate all runtime data into one self-contained directory.

        Before: batches/, historical/, knowledge/, graph_db/, memory.parquet, plugins/ — scattered
        After:  qor-data/batches/, qor-data/historical/, qor-data/knowledge/, etc. — organized
        """
        d = self.runtime.data_dir
        self.runtime.historical_dir = os.path.join(d, "historical")
        self.continual.learn_dir = os.path.join(d, "learn")
        self.train.checkpoint_dir = os.path.join(d, "checkpoints")
        # Encryption key
        if not self.runtime.encryption_key_path:
            self.runtime.encryption_key_path = os.path.join(d, ".keyfile")
        # Graph
        if hasattr(self, 'graph') and hasattr(self.graph, 'db_path'):
            self.graph.db_path = os.path.join(d, "knowledge", "graph.rocksdb")
        # Trading
        if hasattr(self, 'trading'):
            self.trading.data_dir = os.path.join(d, "trading")
        # Futures
        if hasattr(self, 'futures'):
            self.futures.data_dir = os.path.join(d, "futures")
        # NGRE: resolve mamba checkpoint paths — check both original and resolved dirs
        if hasattr(self, 'ngre') and not self.ngre.mamba_checkpoint:
            for ckpt_dir in [self.train.checkpoint_dir, "checkpoints"]:
                mamba_st = os.path.join(ckpt_dir, "mamba_block.safetensors")
                mamba_cfg = os.path.join(ckpt_dir, "mamba_block_config.json")
                if os.path.isfile(mamba_st):
                    self.ngre.mamba_checkpoint = mamba_st
                    self.ngre.mamba_config = mamba_cfg
                    break
        return self

    def get_data_path(self, *parts):
        """Get a path relative to data_dir. e.g. config.get_data_path("memory.parquet")"""
        return os.path.join(self.runtime.data_dir, *parts)

    def get_device(self):
        import torch
        if self.train.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return self.train.device


# Preset configs for common use cases
PRESETS = {
    "small": QORConfig.small,
    "medium": QORConfig.medium,
    "large": QORConfig.large,
    "qor3b": QORConfig.qor3b,
    "small_multimodal": QORConfig.small_multimodal,
    "qor3b_multimodal": QORConfig.qor3b_multimodal,
}

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

    @property
    def head_dim(self):
        return self.d_model // self.n_heads


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
class QORConfig:
    """Master config — everything in one place."""
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    serve: ServeConfig = field(default_factory=ServeConfig)
    continual: ContinualConfig = field(default_factory=ContinualConfig)

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
            section = getattr(config, section_name)
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
        c.model.max_seq_len = 1024
        c.tokenizer.vocab_size = 32000
        c.model.vocab_size = 32000
        c.train.batch_size = 2
        return c

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
}

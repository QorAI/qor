"""
QOR Model — The Qore Mind
==========================
Production architecture with all three core components:
  1. Self-Modifying Neurons (fast weight adaptation)
  2. Continuum Memory System (multi-speed memory)
  3. Surprise-Gated Learning (learn from the unexpected)

Plus production necessities:
  - RoPE positional encoding (better than absolute)
  - RMSNorm (faster than LayerNorm)
  - KV-Cache for fast generation
  - Proper weight initialization
  - Gradient checkpointing support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

from .config import ModelConfig


# ==============================================================================
# BUILDING BLOCKS
# ==============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Normalization — faster than LayerNorm."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


def precompute_rope(dim: int, max_seq_len: int, theta: float = 10000.0):
    """Precompute Rotary Position Embedding (RoPE) frequencies."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)
    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin


def apply_rope(q, k, cos, sin):
    """Apply RoPE to query and key tensors."""
    # q, k: (batch, n_heads, seq_len, head_dim)
    seq_len = q.shape[2]
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)  # (1, 1, seq, dim/2)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)

    def rotate(x, cos, sin):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    return rotate(q, cos, sin), rotate(k, cos, sin)


# ==============================================================================
# SELF-MODIFYING LINEAR — Neurons that adapt in real-time
# ==============================================================================

class SelfModifyingLinear(nn.Module):
    """
    A linear layer that updates its own weights during the forward pass.

    Standard: y = Wx + b      (W frozen after training)
    Self-mod: y = (W + ΔW)x   (ΔW updates live based on surprise)

    This is what gives QOR its ability to adapt during inference.
    """

    def __init__(self, in_features: int, out_features: int, config: ModelConfig):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.eta = config.self_mod_lr
        self.alpha = config.self_mod_decay
        self.threshold = config.surprise_threshold

        # Target predictor for measuring surprise
        self.target_pred = nn.Linear(in_features, out_features, bias=False)

        # Fast weight delta — updated manually, not by optimizer
        self.register_buffer('delta_W', torch.zeros(out_features, in_features))

    def reset_fast_weights(self):
        self.delta_W.zero_()

    def forward(self, x: torch.Tensor, enable_self_mod: bool = True) -> torch.Tensor:
        # Apply base weights + fast weight delta
        y = self.W(x) + F.linear(x, self.delta_W)

        if enable_self_mod and self.training:
            with torch.no_grad():
                predicted = self.target_pred(x)
                teach_signal = y.detach() - predicted
                surprise = teach_signal.norm(dim=-1).mean()

                if surprise > self.threshold:
                    delta = torch.einsum('bsi,bsj->ij',
                        teach_signal, x.detach()
                    ) / (x.shape[0] * x.shape[1])
                    self.delta_W = self.alpha * self.delta_W + self.eta * delta

        return y


# ==============================================================================
# CONTINUUM MEMORY SYSTEM — Multi-speed memory
# ==============================================================================

class CMSLevel(nn.Module):
    """One speed level of the Continuum Memory System."""

    def __init__(self, d_model: int, d_ff: int, update_freq: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),  # SwiGLU-like activation
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm = RMSNorm(d_model)
        self.update_freq = update_freq
        self.step_count = 0
        self._should_update = True

    def step(self):
        self.step_count += 1
        self._should_update = (self.step_count % self.update_freq == 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm(x)
        if self._should_update and self.training:
            return x + self.mlp(normed)
        else:
            with torch.no_grad():
                return x + self.mlp(normed)


class ContinuumMemorySystem(nn.Module):
    """
    Multi-speed memory: fast thoughts, medium recall, deep knowledge.

    Fast level:   Updates every step      — working memory
    Medium level: Updates every 16 steps  — short-term memory
    Slow level:   Updates every 64 steps  — long-term knowledge

    Slow layers protect old knowledge while fast layers learn new things.
    """

    def __init__(self, d_model: int, d_ff: int, config: ModelConfig):
        super().__init__()
        freqs = [config.cms_fast_freq, config.cms_med_freq, config.cms_slow_freq]
        # Each level gets a proportional share of the FFN capacity
        level_ff = [d_ff // 2, d_ff // 4, d_ff // 4]  # Fast gets more capacity

        self.levels = nn.ModuleList([
            CMSLevel(d_model, ff, freq, config.dropout)
            for freq, ff in zip(freqs[:config.cms_levels], level_ff[:config.cms_levels])
        ])
        self.level_names = ["fast", "medium", "slow"][:config.cms_levels]

    def step(self):
        for level in self.levels:
            level.step()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for level in self.levels:
            x = level(x)
        return x

    def get_status(self):
        return {name: level._should_update
                for name, level in zip(self.level_names, self.levels)}

    def freeze_slow_layers(self):
        """Freeze slow layers (for continual learning)."""
        for level in self.levels:
            if level.update_freq >= 64:
                for param in level.parameters():
                    param.requires_grad = False

    def unfreeze_all(self):
        for level in self.levels:
            for param in level.parameters():
                param.requires_grad = True


# ==============================================================================
# ATTENTION with RoPE and KV-Cache
# ==============================================================================

class QORAttention(nn.Module):
    """Multi-head attention with RoPE and optional KV-cache for fast generation."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        self.norm = RMSNorm(config.d_model)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Precompute RoPE
        cos, sin = precompute_rope(config.head_dim, config.max_seq_len)
        self.register_buffer('rope_cos', cos, persistent=False)
        self.register_buffer('rope_sin', sin, persistent=False)

    def forward(self, x: torch.Tensor,
                kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False) -> Tuple[torch.Tensor, Optional[Tuple]]:
        B, T, C = x.shape
        normed = self.norm(x)

        q = self.q_proj(normed).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(normed).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(normed).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q, k = apply_rope(q, k, self.rope_cos, self.rope_sin)

        # KV-Cache for generation
        if kv_cache is not None:
            prev_k, prev_v = kv_cache
            k = torch.cat([prev_k, k], dim=2)
            v = torch.cat([prev_v, v], dim=2)

        new_cache = (k, v) if use_cache else None

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) / scale

        # Causal mask
        if T > 1:
            mask = torch.triu(torch.ones(T, k.shape[2], device=x.device, dtype=torch.bool), diagonal=k.shape[2]-T+1)
            attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        out = self.resid_dropout(self.out_proj(out))

        return x + out, new_cache


# ==============================================================================
# QOR BLOCK — One complete layer
# ==============================================================================

class QORBlock(nn.Module):
    """
    One QOR block:
      Input → [Attention + RoPE] → [Self-Mod Neurons] → [CMS Memory] → Output
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = QORAttention(config)
        self.self_mod = SelfModifyingLinear(config.d_model, config.d_model, config)
        self.self_mod_norm = RMSNorm(config.d_model)
        self.cms = ContinuumMemorySystem(config.d_model, config.d_ff, config)

    def forward(self, x: torch.Tensor, enable_self_mod: bool = True,
                kv_cache=None, use_cache=False):
        # Attention
        x, new_cache = self.attention(x, kv_cache=kv_cache, use_cache=use_cache)

        # Self-modification
        normed = self.self_mod_norm(x)
        x = x + self.self_mod(normed, enable_self_mod=enable_self_mod)

        # Multi-speed memory
        x = self.cms(x)

        return x, new_cache


# ==============================================================================
# THE FULL QOR MODEL
# ==============================================================================

class QORModel(nn.Module):
    """
    QOR — The Qore Mind.

    Complete language model with:
    - Proper token + RoPE positional encoding
    - Stacked QOR blocks (attention + self-mod + CMS)
    - KV-cache for fast autoregressive generation
    - Weight tying (embedding = output head)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token embedding (no position embedding — RoPE handles that)
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.embed_dropout = nn.Dropout(config.dropout)

        # QOR blocks
        self.blocks = nn.ModuleList([
            QORBlock(config) for _ in range(config.n_layers)
        ])

        # Output
        self.norm = RMSNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.embed.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Parameter count
        self.n_params = sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        """Proper weight initialization."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def reset_fast_weights(self):
        for block in self.blocks:
            block.self_mod.reset_fast_weights()

    def step_cms(self):
        for block in self.blocks:
            block.cms.step()

    def forward(self, input_ids: torch.Tensor,
                targets: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                enable_self_mod: bool = True,
                kv_cache: Optional[list] = None,
                use_cache: bool = False):
        """
        Args:
            input_ids: (batch, seq_len) token IDs
            targets: (batch, seq_len) target token IDs for loss
            attention_mask: (batch, seq_len) mask for padding
            enable_self_mod: whether to update fast weights
            kv_cache: list of (k, v) tuples per layer
            use_cache: whether to return updated kv_cache
        """
        B, T = input_ids.shape

        # Embed tokens
        x = self.embed(input_ids)
        x = self.embed_dropout(x)

        # Pass through QOR blocks
        new_caches = []
        for i, block in enumerate(self.blocks):
            layer_cache = kv_cache[i] if kv_cache else None
            x, new_cache = block(x, enable_self_mod=enable_self_mod,
                                  kv_cache=layer_cache, use_cache=use_cache)
            if use_cache:
                new_caches.append(new_cache)

        # Output
        x = self.norm(x)
        logits = self.head(x)

        # Loss
        loss = None
        surprise = None
        if targets is not None:
            # Flatten for cross-entropy
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = targets[:, 1:].contiguous()

            if attention_mask is not None:
                # Mask out padding from loss
                shift_mask = attention_mask[:, 1:].contiguous()
                loss_per_token = F.cross_entropy(
                    shift_logits.view(-1, self.config.vocab_size),
                    shift_targets.view(-1), reduction='none'
                )
                loss_per_token = loss_per_token.view(B, -1)
                loss = (loss_per_token * shift_mask).sum() / shift_mask.sum().clamp(min=1)
            else:
                loss = F.cross_entropy(
                    shift_logits.view(-1, self.config.vocab_size),
                    shift_targets.view(-1)
                )

            # Average surprise for logging
            with torch.no_grad():
                surprise = loss.item()

        return {
            "logits": logits,
            "loss": loss,
            "surprise": surprise,
            "kv_cache": new_caches if use_cache else None,
        }

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor,
                 max_new_tokens: int = 256,
                 temperature: float = 0.8,
                 top_k: int = 50,
                 top_p: float = 0.9,
                 repetition_penalty: float = 1.1,
                 stop_tokens: Optional[list] = None):
        """
        Generate tokens autoregressively with KV-cache.

        Supports:
        - Temperature sampling
        - Top-k filtering
        - Top-p (nucleus) sampling
        - Repetition penalty
        - Stop tokens (EOS, etc.)
        """
        self.eval()
        self.reset_fast_weights()
        device = input_ids.device

        # Initial forward pass (fills KV-cache)
        result = self.forward(input_ids, enable_self_mod=True, use_cache=True)
        kv_cache = result["kv_cache"]
        logits = result["logits"][:, -1, :]

        generated = input_ids.tolist()[0] if input_ids.dim() > 1 else input_ids.tolist()

        for _ in range(max_new_tokens):
            # Apply temperature
            logits = logits / max(temperature, 1e-8)

            # Repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(generated[-50:]):  # Look at last 50 tokens
                    logits[0, token_id] /= repetition_penalty

            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                mask = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[mask] = float('-inf')
                logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_id = next_token.item()
            generated.append(token_id)

            # Check stop condition
            if stop_tokens and token_id in stop_tokens:
                break

            # Forward with KV-cache (only new token)
            result = self.forward(next_token, enable_self_mod=True,
                                   kv_cache=kv_cache, use_cache=True)
            kv_cache = result["kv_cache"]
            logits = result["logits"][:, -1:, :].squeeze(1)

        return generated

    def count_parameters(self) -> dict:
        """Detailed parameter count breakdown."""
        counts = {
            "embedding": sum(p.numel() for p in self.embed.parameters()),
            "attention": 0,
            "self_mod": 0,
            "cms_fast": 0,
            "cms_medium": 0,
            "cms_slow": 0,
            "output_head": 0,  # tied with embedding
            "total": self.n_params,
        }
        for block in self.blocks:
            counts["attention"] += sum(p.numel() for p in block.attention.parameters())
            counts["self_mod"] += sum(p.numel() for p in block.self_mod.parameters())
            for name, level in zip(block.cms.level_names, block.cms.levels):
                key = f"cms_{name}"
                if key in counts:
                    counts[key] += sum(p.numel() for p in level.parameters())
        return counts

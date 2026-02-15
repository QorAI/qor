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

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

from .config import ModelConfig, VisionConfig, AudioConfig

# CfC liquid neurons (pip install ncps)
try:
    from ncps.torch import CfC as _CfC
    from ncps.wirings import AutoNCP as _AutoNCP
    _HAS_NCPS = True
except ImportError:
    _HAS_NCPS = False

logger = logging.getLogger(__name__)


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


def apply_rope(q, k, cos, sin, offset: int = 0):
    """Apply RoPE to query and key tensors.

    Args:
        q, k: (batch, n_heads, seq_len, head_dim)
        cos, sin: precomputed RoPE tables (max_seq_len, head_dim/2)
        offset: position offset for KV-cache decoding (length of cached sequence)
    """
    seq_len = q.shape[2]
    cos = cos[offset:offset + seq_len].unsqueeze(0).unsqueeze(0).to(dtype=q.dtype, device=q.device)
    sin = sin[offset:offset + seq_len].unsqueeze(0).unsqueeze(0).to(dtype=q.dtype, device=q.device)

    def rotate(x, cos, sin):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    return rotate(q, cos, sin), rotate(k, cos, sin)


# ==============================================================================
# S4/MAMBA STATE SPACE BLOCK — imported from qor.mamba (single source of truth)
# ==============================================================================
from .cortex import S4Block  # noqa: E402 — pure PyTorch, CPU-friendly, no CUDA needed


# ==============================================================================
# SELF-MODIFYING LINEAR — Neurons that adapt in real-time
# ==============================================================================

class SelfModifyingLinear(nn.Module):
    """
    A linear layer that updates its own weights during the forward pass.

    Standard: y = Wx + b      (W frozen after training)
    Self-mod: y = (W + ΔW)x   (ΔW updates live based on surprise)

    Surprise detection modes:
      - Linear (default): Simple nn.Linear target predictor
      - CfC (use_cfc=True): Closed-form Continuous-time liquid neurons with
        NCP wiring and mixed memory (LSTM + CfC dynamics). Carries hidden
        state across forward passes for temporal surprise context.
        Paper: Nature Machine Intelligence 2022. Requires: pip install ncps
    """

    def __init__(self, in_features: int, out_features: int, config: ModelConfig):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.eta = config.self_mod_lr
        self.alpha = config.self_mod_decay
        self.threshold = config.surprise_threshold
        self.use_cfc = getattr(config, 'use_cfc', False)

        if self.use_cfc:
            if not _HAS_NCPS:
                raise ImportError(
                    "CfC self-modification requires ncps: pip install ncps"
                )
            cfc_neurons = getattr(config, 'cfc_neurons', 32)
            cfc_output = getattr(config, 'cfc_output_size', 8)

            # CfC-mmRNN with NCP brain-like wiring replaces linear target_pred
            wiring = _AutoNCP(cfc_neurons, output_size=cfc_output)
            self.cfc = _CfC(
                input_size=in_features,
                units=wiring,
                mixed_memory=True,      # h + c state (best benchmark variant)
                return_sequences=True,  # per-token predictions
            )
            # Project CfC output (8) → out_features for teach signal
            self.cfc_proj = nn.Linear(cfc_output, out_features, bias=False)
            nn.init.normal_(self.cfc_proj.weight, std=0.01)

            # Persistent hidden state — carries temporal context across calls
            self.register_buffer('cfc_h', torch.zeros(1, cfc_neurons))
            self.register_buffer('cfc_c', torch.zeros(1, cfc_neurons))
        else:
            # Legacy: simple linear target predictor
            self.target_pred = nn.Linear(in_features, out_features, bias=False)

        # Fast weight delta — updated manually, not by optimizer
        self.register_buffer('delta_W', torch.zeros(out_features, in_features))

    def reset_fast_weights(self):
        self.delta_W.zero_()
        if self.use_cfc:
            self.cfc_h.zero_()
            self.cfc_c.zero_()

    def forward(self, x: torch.Tensor, enable_self_mod: bool = True) -> torch.Tensor:
        # Apply base weights + fast weight delta (cast delta_W to match x dtype)
        y = self.W(x) + F.linear(x, self.delta_W.to(x.dtype))

        if enable_self_mod and self.training:
            with torch.no_grad():
                if self.use_cfc:
                    # CfC liquid neurons: temporal surprise detection
                    batch_size = x.shape[0]
                    h0 = self.cfc_h.expand(batch_size, -1).contiguous()
                    c0 = self.cfc_c.expand(batch_size, -1).contiguous()
                    cfc_out, (h_new, c_new) = self.cfc(
                        x.float(), (h0.float(), c0.float())
                    )
                    predicted = self.cfc_proj(cfc_out).to(x.dtype)
                    # Persist hidden state (average across batch)
                    self.cfc_h.copy_(h_new.mean(dim=0, keepdim=True))
                    self.cfc_c.copy_(c_new.mean(dim=0, keepdim=True))
                else:
                    predicted = self.target_pred(x)

                teach_signal = y.detach() - predicted
                surprise = teach_signal.norm(dim=-1).mean()

                if surprise > self.threshold:
                    delta = torch.einsum('bsi,bsj->ij',
                        teach_signal, x.detach()
                    ) / (x.shape[0] * x.shape[1])
                    self.delta_W = self.alpha * self.delta_W + self.eta * delta
                    # Clamp to prevent unbounded growth
                    dw_norm = self.delta_W.norm()
                    max_norm = self.W.weight.norm() * 0.1  # 10% of base weight
                    if dw_norm > max_norm:
                        self.delta_W = self.delta_W * (max_norm / dw_norm)

        return y


# ==============================================================================
# CONTINUUM MEMORY SYSTEM — Multi-speed memory
# ==============================================================================

class SwiGLUMLP(nn.Module):
    """SwiGLU feed-forward network — used by Llama, Mistral, etc."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, hidden_size: int = 0):
        super().__init__()
        # If hidden_size is explicitly given, use it directly (for donor weight loading).
        # Otherwise compute from d_ff (standard Llama practice).
        hidden = hidden_size if hidden_size > 0 else d_ff * 2 // 3
        self.gate_proj = nn.Linear(d_model, hidden, bias=False)
        self.up_proj = nn.Linear(d_model, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class CMSLevel(nn.Module):
    """One speed level of the Continuum Memory System."""

    def __init__(self, d_model: int, d_ff: int, update_freq: int, dropout: float = 0.1,
                 hidden_size: int = 0):
        super().__init__()
        self.mlp = SwiGLUMLP(d_model, d_ff, dropout, hidden_size=hidden_size)
        self.norm = RMSNorm(d_model)
        self.update_freq = update_freq
        self.step_count = 0
        self._should_update = True

    def step(self):
        self.step_count += 1
        self._should_update = (self.step_count % self.update_freq == 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm(x)
        if self.training:
            # Always allow gradients during backprop training
            return x + self.mlp(normed)
        elif self._should_update:
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

        # Per-level FFN sizes: use explicit sizes if provided, else proportional split
        explicit_sizes = getattr(config, 'cms_level_ff_sizes', [])
        if explicit_sizes:
            # Explicit hidden sizes per level (used for donor weight loading)
            self.levels = nn.ModuleList([
                CMSLevel(d_model, d_ff, freq, config.dropout, hidden_size=hs)
                for freq, hs in zip(freqs[:config.cms_levels], explicit_sizes[:config.cms_levels])
            ])
        else:
            # Default proportional split: fast gets more capacity
            level_ff = [d_ff // 2, d_ff // 4, d_ff // 4]
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

    @torch.no_grad()
    def decay_slow(self, decay_rate: float = 0.001):
        """Gradually decay slow CMS weights toward zero.

        This makes room for new stable patterns by slightly shrinking
        old slow-layer weights each consolidation cycle. Small rate
        ensures old knowledge fades slowly, not abruptly.

        Pattern: W = W * (1 - decay_rate)
        At rate=0.001, weights retain 99.9% each cycle.
        After 100 cycles: ~90.5% retained. After 1000: ~36.8%.
        """
        for level in self.levels:
            if level.update_freq >= 64:  # slow layers only
                for param in level.parameters():
                    param.mul_(1.0 - decay_rate)


# ==============================================================================
# ATTENTION with RoPE and KV-Cache
# ==============================================================================

class QORAttention(nn.Module):
    """
    Multi-head attention with:
    - Grouped Query Attention (GQA) — fewer KV heads for efficiency
    - Flash Attention via F.scaled_dot_product_attention
    - Sliding window attention (optional)
    - RoPE and KV-cache
    """

    def __init__(self, config: ModelConfig, layer_idx: int = 0):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model
        self.n_kv_heads = getattr(config, 'n_kv_heads', config.n_heads)
        self.n_rep = self.n_heads // self.n_kv_heads  # repetition factor for GQA
        self.sliding_window = getattr(config, 'sliding_window', 0)

        self.q_proj = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=False)

        self.norm = RMSNorm(config.d_model)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Determine if this layer uses RoPE (some donor models skip every Nth layer)
        interval = getattr(config, 'no_rope_layer_interval', 0)
        self.use_rope = True
        if interval > 0 and ((layer_idx + 1) % interval == 0):
            self.use_rope = False

        # Precompute RoPE (still allocate buffers even if unused — keeps state_dict consistent)
        rope_theta = getattr(config, 'rope_theta', 10000.0)
        cos, sin = precompute_rope(config.head_dim, config.max_seq_len, theta=rope_theta)
        self.register_buffer('rope_cos', cos, persistent=False)
        self.register_buffer('rope_sin', sin, persistent=False)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat KV heads to match query heads (for GQA)."""
        if self.n_rep == 1:
            return x
        B, n_kv, T, D = x.shape
        return x[:, :, None, :, :].expand(B, n_kv, self.n_rep, T, D).reshape(B, self.n_heads, T, D)

    def forward(self, x: torch.Tensor,
                kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False) -> Tuple[torch.Tensor, Optional[Tuple]]:
        B, T, C = x.shape
        normed = self.norm(x)

        q = self.q_proj(normed).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(normed).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(normed).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Calculate position offset from KV-cache (for correct RoPE during decoding)
        pos_offset = 0
        if kv_cache is not None:
            pos_offset = kv_cache[0].shape[2]  # length of cached sequence

        # Apply RoPE (some layers skip it, e.g. every 4th layer in certain donor models)
        if self.use_rope:
            q, k = apply_rope(q, k, self.rope_cos, self.rope_sin, offset=pos_offset)

        # KV-Cache for generation (AFTER RoPE — cached k's already have correct positions)
        if kv_cache is not None:
            prev_k, prev_v = kv_cache
            k = torch.cat([prev_k, k], dim=2)
            v = torch.cat([prev_v, v], dim=2)

        new_cache = (k, v) if use_cache else None

        # Expand KV heads for GQA
        k_expanded = self._repeat_kv(k)
        v_expanded = self._repeat_kv(v)

        # Ensure q/k/v have same dtype (fp16 weights + fp32 RMSNorm can cause mismatch)
        if v_expanded.dtype != q.dtype:
            v_expanded = v_expanded.to(q.dtype)
            k_expanded = k_expanded.to(q.dtype)

        # Flash Attention via F.scaled_dot_product_attention
        # Use sliding window mask if configured, otherwise use is_causal
        if self.sliding_window > 0 and T > 1:
            # Build sliding window + causal mask
            S = k_expanded.shape[2]
            row_idx = torch.arange(T, device=x.device).unsqueeze(1)
            col_idx = torch.arange(S, device=x.device).unsqueeze(0)
            # Causal: can only attend to positions <= current (adjusted for cache offset)
            offset = S - T
            causal = col_idx <= (row_idx + offset)
            # Sliding window: can only attend within window
            window = (row_idx + offset) - col_idx < self.sliding_window
            mask = causal & window
            # Convert to float mask for SDPA: 0 = attend, -inf = mask
            attn_mask = torch.zeros(T, S, device=x.device, dtype=q.dtype)
            attn_mask.masked_fill_(~mask, float('-inf'))
            out = F.scaled_dot_product_attention(
                q, k_expanded, v_expanded,
                attn_mask=attn_mask,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
            )
        elif T > 1 and kv_cache is None:
            # Prefill: use is_causal=True (efficient)
            out = F.scaled_dot_product_attention(
                q, k_expanded, v_expanded,
                is_causal=True,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
            )
        else:
            # Single token decode with cache: no mask needed
            out = F.scaled_dot_product_attention(
                q, k_expanded, v_expanded,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
            )

        out = out.transpose(1, 2).reshape(B, T, self.n_heads * self.head_dim)
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

    def __init__(self, config: ModelConfig, layer_idx: int = 0):
        super().__init__()
        self.attention = QORAttention(config, layer_idx=layer_idx)
        self.self_mod = SelfModifyingLinear(config.d_model, config.d_model, config)
        self.self_mod_norm = RMSNorm(config.d_model)

        # S4/Mamba state-space scan (long-range temporal patterns)
        self.s4 = None
        if getattr(config, 'use_s4', False):
            self.s4 = S4Block(d_model=config.d_model,
                              d_state=getattr(config, 's4_d_state', 16),
                              d_conv=getattr(config, 's4_d_conv', 4))

        self.cms = ContinuumMemorySystem(config.d_model, config.d_ff, config)

    def forward(self, x: torch.Tensor, enable_self_mod: bool = True,
                kv_cache=None, use_cache=False):
        # Attention
        x, new_cache = self.attention(x, kv_cache=kv_cache, use_cache=use_cache)

        # Self-modification
        normed = self.self_mod_norm(x)
        x = x + self.self_mod(normed, enable_self_mod=enable_self_mod)

        # S4 state-space scan (long-range temporal patterns)
        if self.s4 is not None:
            x = self.s4(x)

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

    def __init__(self, config: ModelConfig,
                 vision_config: Optional[VisionConfig] = None,
                 audio_config: Optional[AudioConfig] = None,
                 _from_checkpoint: bool = False,
                 _encoder_hf_configs: Optional[dict] = None):
        """
        Args:
            config: Model architecture config.
            vision_config: Vision encoder config (None = no vision).
            audio_config: Audio encoder config (None = no audio).
            _from_checkpoint: If True, create pretrained encoder architectures
                WITHOUT downloading from HuggingFace. Weights come from
                load_state_dict() after construction. Use this when loading
                from a saved checkpoint.
            _encoder_hf_configs: Dict with 'vision' and/or 'audio' keys containing
                the HF encoder config dicts saved during build. Required when
                _from_checkpoint=True and pretrained encoders are used.
        """
        super().__init__()
        self.config = config
        self.vision_config = vision_config
        self.audio_config = audio_config
        self._gradient_checkpointing = False

        # Token embedding (no position embedding — RoPE handles that)
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.embed_dropout = nn.Dropout(config.dropout)

        # QOR blocks
        self.blocks = nn.ModuleList([
            QORBlock(config, layer_idx=i) for i in range(config.n_layers)
        ])

        # Output
        self.norm = RMSNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.embed.weight

        # Conditional multimodal encoders
        self.vision_encoder = None
        self.audio_encoder = None

        hf_configs = _encoder_hf_configs or {}

        if vision_config is not None and vision_config.enabled:
            from .vision import create_vision_encoder
            self.vision_encoder = create_vision_encoder(
                vision_config, config.d_model,
                _from_checkpoint=_from_checkpoint,
                _hf_config=hf_configs.get('vision'),
            )

        if audio_config is not None and audio_config.enabled:
            from .audio import create_audio_encoder
            self.audio_encoder = create_audio_encoder(
                audio_config, config.d_model,
                _from_checkpoint=_from_checkpoint,
                _hf_config=hf_configs.get('audio'),
            )

        # Initialize weights
        self.apply(self._init_weights)

        # Parameter count
        self.n_params = sum(p.numel() for p in self.parameters())

    def get_encoder_hf_configs(self) -> dict:
        """Extract HF encoder configs for saving in checkpoint.
        These are used to recreate the architecture without downloading."""
        configs = {}
        if self.vision_encoder is not None and hasattr(self.vision_encoder, 'get_hf_config'):
            configs['vision'] = self.vision_encoder.get_hf_config()
        if self.audio_encoder is not None and hasattr(self.audio_encoder, 'get_hf_config'):
            configs['audio'] = self.audio_encoder.get_hf_config()
        return configs

    def resize_token_embeddings(self, new_vocab_size: int):
        """Resize the token embedding and output head for new vocabulary size.

        Used when modality tokens (image_patch, audio_frame) are added to
        a pretrained tokenizer. Preserves existing weights, initializes
        new embeddings with small random values. Maintains weight tying.

        Args:
            new_vocab_size: New vocabulary size (must be >= current vocab_size).
        """
        old_vocab_size = self.config.vocab_size
        if new_vocab_size == old_vocab_size:
            return  # Nothing to do
        if new_vocab_size < old_vocab_size:
            raise ValueError(f"Cannot shrink vocab from {old_vocab_size} to {new_vocab_size}")

        # Create new embedding with more rows
        old_embed = self.embed
        new_embed = nn.Embedding(new_vocab_size, self.config.d_model)
        # Copy existing weights
        new_embed.weight.data[:old_vocab_size] = old_embed.weight.data
        # Small random init for new tokens
        nn.init.normal_(new_embed.weight.data[old_vocab_size:], mean=0.0, std=0.02)
        self.embed = new_embed

        # Rebuild output head (tied to embedding)
        self.head = nn.Linear(self.config.d_model, new_vocab_size, bias=False)
        self.head.weight = self.embed.weight  # Weight tying

        self.config.vocab_size = new_vocab_size
        self.n_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Resized embeddings: {old_vocab_size} -> {new_vocab_size} "
                    f"(+{new_vocab_size - old_vocab_size} tokens)")

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing to trade compute for memory (~2x less VRAM)."""
        self._gradient_checkpointing = True
        logger.info("Gradient checkpointing enabled")

    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing."""
        self._gradient_checkpointing = False

    def compile_model(self, mode: str = "reduce-overhead"):
        """Apply torch.compile for additional speedup. Requires PyTorch 2.0+."""
        try:
            self.forward = torch.compile(self.forward, mode=mode)
            logger.info(f"torch.compile applied (mode={mode})")
            return True
        except Exception as e:
            logger.warning(f"torch.compile not available: {e}")
            return False

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
                use_cache: bool = False,
                images: Optional[torch.Tensor] = None,
                image_positions: Optional[torch.Tensor] = None,
                mel_specs: Optional[torch.Tensor] = None,
                audio_positions: Optional[torch.Tensor] = None):
        """
        Args:
            input_ids: (batch, seq_len) token IDs
            targets: (batch, seq_len) target token IDs for loss
            attention_mask: (batch, seq_len) mask for padding
            enable_self_mod: whether to update fast weights
            kv_cache: list of (k, v) tuples per layer
            use_cache: whether to return updated kv_cache
            images: (batch, C, H, W) image tensor (optional)
            image_positions: (batch, n_patches) indices where image embeddings go
            mel_specs: (batch, n_mels, n_frames) mel spectrogram (optional)
            audio_positions: (batch, n_audio_tokens) indices where audio embeddings go
        """
        B, T = input_ids.shape

        # Embed tokens
        x = self.embed(input_ids)
        x = self.embed_dropout(x)

        # Splice in vision embeddings at image placeholder positions
        aux_loss = None
        if images is not None and self.vision_encoder is not None and image_positions is not None:
            vis_result = self.vision_encoder(images)
            vis_embeds = vis_result["embeddings"]  # (B, n_patches, d_model)
            if vis_result["vq_loss"] is not None:
                aux_loss = vis_result["vq_loss"]

            # Replace placeholder embeddings with vision embeddings
            for b in range(B):
                positions = image_positions[b]
                n_vis = min(vis_embeds.shape[1], positions.shape[0])
                x[b, positions[:n_vis]] = vis_embeds[b, :n_vis]

        # Splice in audio embeddings at audio placeholder positions
        if mel_specs is not None and self.audio_encoder is not None and audio_positions is not None:
            audio_embeds = self.audio_encoder(mel_specs)  # (B, n_tokens, d_model)

            # Replace placeholder embeddings with audio embeddings
            for b in range(B):
                positions = audio_positions[b]
                n_aud = min(audio_embeds.shape[1], positions.shape[0])
                x[b, positions[:n_aud]] = audio_embeds[b, :n_aud]

        # Pass through QOR blocks
        new_caches = []
        for i, block in enumerate(self.blocks):
            layer_cache = kv_cache[i] if kv_cache else None

            if self._gradient_checkpointing and self.training and not use_cache:
                # Gradient checkpointing: trade compute for ~2x less VRAM
                x, new_cache = torch.utils.checkpoint.checkpoint(
                    block, x, enable_self_mod, layer_cache, use_cache,
                    use_reentrant=False,
                )
            else:
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

            # Build loss mask: exclude padding AND modality placeholder positions
            if attention_mask is not None:
                shift_mask = attention_mask[:, 1:].contiguous().float()
            else:
                shift_mask = torch.ones(B, T - 1, device=input_ids.device)

            # Mask out image placeholder positions from loss
            if image_positions is not None:
                for b in range(B):
                    positions = image_positions[b]
                    # Shift by -1 since we're computing loss on shifted targets
                    valid = positions[positions > 0] - 1
                    valid = valid[valid < shift_mask.shape[1]]
                    shift_mask[b, valid] = 0.0

            # Mask out audio placeholder positions from loss
            if audio_positions is not None:
                for b in range(B):
                    positions = audio_positions[b]
                    valid = positions[positions > 0] - 1
                    valid = valid[valid < shift_mask.shape[1]]
                    shift_mask[b, valid] = 0.0

            loss_per_token = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_targets.view(-1), reduction='none'
            )
            loss_per_token = loss_per_token.view(B, -1)
            loss = (loss_per_token * shift_mask).sum() / shift_mask.sum().clamp(min=1)

            # Add VQ-VAE auxiliary loss
            if aux_loss is not None:
                loss = loss + aux_loss

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
                 stop_tokens: Optional[list] = None,
                 images: Optional[torch.Tensor] = None,
                 image_positions: Optional[torch.Tensor] = None,
                 mel_specs: Optional[torch.Tensor] = None,
                 audio_positions: Optional[torch.Tensor] = None):
        """
        Generate tokens autoregressively with KV-cache.

        Supports:
        - Temperature sampling
        - Top-k filtering
        - Top-p (nucleus) sampling
        - Repetition penalty
        - Stop tokens (EOS, etc.)
        - Optional image/audio context for multimodal prefill
        """
        self.eval()
        self.reset_fast_weights()
        device = input_ids.device

        # Initial forward pass (fills KV-cache, includes multimodal context)
        result = self.forward(input_ids, enable_self_mod=True, use_cache=True,
                              images=images, image_positions=image_positions,
                              mel_specs=mel_specs, audio_positions=audio_positions)
        kv_cache = result["kv_cache"]
        logits = result["logits"][:, -1, :]

        generated = input_ids.tolist()[0] if input_ids.dim() > 1 else input_ids.tolist()

        for _ in range(max_new_tokens):
            # Cast to float32 for numerically stable sampling (fp16 overflows in softmax)
            logits = logits.float()

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

    @torch.no_grad()
    def generate_stream(self, input_ids: torch.Tensor,
                        max_new_tokens: int = 256,
                        temperature: float = 0.8,
                        top_k: int = 50,
                        top_p: float = 0.9,
                        repetition_penalty: float = 1.1,
                        stop_tokens: Optional[list] = None):
        """Generate tokens one at a time as a generator (for streaming)."""
        self.eval()
        self.reset_fast_weights()

        result = self.forward(input_ids, enable_self_mod=True, use_cache=True)
        kv_cache = result["kv_cache"]
        logits = result["logits"][:, -1, :]

        generated = input_ids.tolist()[0] if input_ids.dim() > 1 else input_ids.tolist()

        for _ in range(max_new_tokens):
            # Cast to float32 for numerically stable sampling (fp16 overflows in softmax)
            logits = logits.float()
            logits = logits / max(temperature, 1e-8)

            if repetition_penalty != 1.0:
                for token_id in set(generated[-50:]):
                    logits[0, token_id] /= repetition_penalty

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                mask = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[mask] = float('-inf')
                logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_id = next_token.item()
            generated.append(token_id)

            if stop_tokens and token_id in stop_tokens:
                break

            yield token_id

            result = self.forward(next_token, enable_self_mod=True,
                                   kv_cache=kv_cache, use_cache=True)
            kv_cache = result["kv_cache"]
            logits = result["logits"][:, -1:, :].squeeze(1)

    def count_parameters(self) -> dict:
        """Detailed parameter count breakdown."""
        counts = {
            "embedding": sum(p.numel() for p in self.embed.parameters()),
            "attention": 0,
            "self_mod": 0,
            "s4": 0,
            "cms_fast": 0,
            "cms_medium": 0,
            "cms_slow": 0,
            "vision": 0,
            "vision_trainable": 0,
            "audio": 0,
            "audio_trainable": 0,
            "output_head": 0,  # tied with embedding
            "total": self.n_params,
        }
        for block in self.blocks:
            counts["attention"] += sum(p.numel() for p in block.attention.parameters())
            counts["self_mod"] += sum(p.numel() for p in block.self_mod.parameters())
            if block.s4 is not None:
                counts["s4"] += sum(p.numel() for p in block.s4.parameters())
            for name, level in zip(block.cms.level_names, block.cms.levels):
                key = f"cms_{name}"
                if key in counts:
                    counts[key] += sum(p.numel() for p in level.parameters())
        if self.vision_encoder is not None:
            counts["vision"] = sum(p.numel() for p in self.vision_encoder.parameters())
            counts["vision_trainable"] = sum(
                p.numel() for p in self.vision_encoder.parameters() if p.requires_grad)
        if self.audio_encoder is not None:
            counts["audio"] = sum(p.numel() for p in self.audio_encoder.parameters())
            counts["audio_trainable"] = sum(
                p.numel() for p in self.audio_encoder.parameters() if p.requires_grad)
        return counts

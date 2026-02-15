"""
NGRE — Neural Graph Reasoning Engine
=====================================
4-layer brain architecture separated from the LLM mouth.

Layer 1: MambaTemporalModule  — Frozen pretrained Mamba SSM (768-dim hidden state)
Layer 2: Dynamic Graph Memory  — QORGraph (see graph.py)
Layer 3: InterferenceSearch    — Quantum-like amplitude search on graph
Layer 4: ReasoningLayer        — Cross-attention + GRU (Phase C)

The brain produces structured context; the LLM (SmolLM3) speaks from it.
"""

import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy torch import — only needed when NGRE is actually instantiated
# ---------------------------------------------------------------------------
_torch = None
_nn = None
_F = None


def _ensure_torch():
    global _torch, _nn, _F
    if _torch is None:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        _torch = torch
        _nn = nn
        _F = F
    return _torch, _nn, _F


# ===================================================================
# Layer 1: Mamba Temporal Module (frozen, pretrained)
# ===================================================================

class MambaRMSNorm:
    """RMSNorm matching HuggingFace Mamba implementation."""

    def __new__(cls, d_model: int, eps: float = 1e-6):
        torch, nn, _ = _ensure_torch()

        class _RMSNorm(nn.Module):
            def __init__(self, d, e):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(d))
                self.eps = e

            def forward(self, x):
                rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
                return (x * rms).to(x.dtype) * self.weight

        return _RMSNorm(d_model, eps)


class MambaMixer:
    """
    Pure PyTorch selective SSM mixer — matches mamba-130m-hf layer weights.

    Architecture (from safetensors inspection):
        in_proj:  (768 → 3072)   split into gate(1536) + input(1536)
        conv1d:   (1536, 1, 4)   depthwise, groups=1536
        x_proj:   (1536 → 80)   dt_rank(48) + B(16) + C(16)
        dt_proj:  (48 → 1536)   time-step projection
        A_log:    (1536, 16)     log state-transition matrix
        D:        (1536,)        skip connection
        out_proj: (1536 → 768)  output
    """

    def __new__(cls, d_model: int = 768, d_state: int = 16,
                d_conv: int = 4, expand: int = 2, dt_rank: int = 48):
        torch, nn, F = _ensure_torch()

        class _MambaMixer(nn.Module):
            def __init__(self, d_model, d_state, d_conv, expand, dt_rank):
                super().__init__()
                self.d_model = d_model
                self.d_state = d_state
                self.d_inner = d_model * expand  # 1536
                self.dt_rank = dt_rank

                # Projections
                self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
                self.conv1d = nn.Conv1d(
                    self.d_inner, self.d_inner,
                    kernel_size=d_conv, padding=d_conv - 1,
                    groups=self.d_inner
                )
                self.x_proj = nn.Linear(
                    self.d_inner, dt_rank + d_state * 2, bias=False
                )
                self.dt_proj = nn.Linear(dt_rank, self.d_inner, bias=True)

                # SSM parameters
                self.A_log = nn.Parameter(torch.zeros(self.d_inner, d_state))
                self.D = nn.Parameter(torch.ones(self.d_inner))
                self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

            def forward(self, x):
                """
                x: (batch, seq, d_model)
                Returns: (batch, seq, d_model)
                """
                batch, seq_len, _ = x.shape

                # Split into gate and input
                xz = self.in_proj(x)  # (B, L, 2*d_inner)
                x_in, z = xz.chunk(2, dim=-1)  # each (B, L, d_inner)

                # Causal conv1d
                x_conv = x_in.transpose(1, 2)  # (B, d_inner, L)
                x_conv = self.conv1d(x_conv)[:, :, :seq_len]  # causal: trim
                x_conv = x_conv.transpose(1, 2)  # (B, L, d_inner)
                x_conv = F.silu(x_conv)

                # SSM parameters from input
                x_dbl = self.x_proj(x_conv)  # (B, L, dt_rank + 2*d_state)
                dt, B_param, C_param = x_dbl.split(
                    [self.dt_rank, self.d_state, self.d_state], dim=-1
                )
                dt = F.softplus(self.dt_proj(dt))  # (B, L, d_inner)

                # State transition
                A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

                # Selective scan (sequential — correct on any device)
                y = self._selective_scan(x_conv, dt, A, B_param, C_param)

                # Skip connection
                y = y + self.D.unsqueeze(0).unsqueeze(0) * x_conv

                # Gated output
                y = y * F.silu(z)
                return self.out_proj(y)

            def _selective_scan(self, x, dt, A, B, C):
                """
                Sequential selective scan (pure PyTorch, no CUDA kernel).

                x:  (B, L, D)     input
                dt: (B, L, D)     time-step
                A:  (D, N)        state matrix (negative)
                B:  (B, L, N)     input matrix
                C:  (B, L, N)     output matrix

                Returns: (B, L, D)
                """
                batch, seq_len, d_inner = x.shape
                d_state = A.shape[1]

                # Discretize A: A_bar = exp(A * dt)
                # dt: (B, L, D) -> (B, L, D, 1)
                # A:  (D, N) -> (1, 1, D, N)
                dt_A = dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
                A_bar = torch.exp(dt_A)  # (B, L, D, N)

                # Discretize B: B_bar = dt * B
                # dt: (B, L, D) -> (B, L, D, 1)
                # B:  (B, L, N) -> (B, L, 1, N)
                B_bar = dt.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, D, N)

                # Sequential scan
                h = torch.zeros(batch, d_inner, d_state,
                                device=x.device, dtype=x.dtype)
                outputs = []

                for t in range(seq_len):
                    # h = A_bar * h + B_bar * x
                    h = A_bar[:, t] * h + B_bar[:, t] * x[:, t].unsqueeze(-1)
                    # y = C @ h
                    y_t = (C[:, t].unsqueeze(1) * h).sum(-1)  # (B, D)
                    outputs.append(y_t)

                return torch.stack(outputs, dim=1)  # (B, L, D)

        return _MambaMixer(d_model, d_state, d_conv, expand, dt_rank)


class MambaBlock:
    """
    Single Mamba block: RMSNorm → MambaMixer + residual.
    Loads pretrained weights from safetensors (state-spaces/mamba-130m-hf layer 0).
    """

    def __new__(cls, d_model: int = 768, d_state: int = 16,
                d_conv: int = 4, expand: int = 2, dt_rank: int = 48):
        torch, nn, _ = _ensure_torch()

        class _MambaBlock(nn.Module):
            def __init__(self, d_model, d_state, d_conv, expand, dt_rank):
                super().__init__()
                self.norm = MambaRMSNorm(d_model)
                self.mixer = MambaMixer(d_model, d_state, d_conv, expand, dt_rank)

            def forward(self, x):
                return x + self.mixer(self.norm(x))

            @classmethod
            def from_pretrained(cls, safetensors_path: str, config_path: str = ""):
                """Load from extracted mamba_block.safetensors."""
                if config_path and os.path.isfile(config_path):
                    with open(config_path) as f:
                        cfg = json.load(f)
                    d_model = cfg.get("hidden_size", 768)
                    d_state = cfg.get("state_size", 16)
                    d_conv = cfg.get("conv_kernel", 4)
                    d_inner = cfg.get("intermediate_size", d_model * 2)
                    dt_rank = cfg.get("time_step_rank", 48)
                    expand = d_inner // d_model
                else:
                    d_model, d_state, d_conv, expand, dt_rank = 768, 16, 4, 2, 48

                block = MambaBlock(d_model, d_state, d_conv, expand, dt_rank)

                from safetensors.torch import load_file
                state = load_file(safetensors_path)
                block.load_state_dict(state, strict=True)
                return block

        return _MambaBlock(d_model, d_state, d_conv, expand, dt_rank)


class MambaTemporalModule:
    """
    Layer 1: Frozen Mamba temporal processor.

    Uses the FULL state-spaces/mamba-130m-hf model (all 24 layers, 129M params).
    Downloads from HuggingFace on first use, cached locally afterwards.

    Takes token embeddings → produces:
      - h: 768-dim hidden state (last position)
      - h_seq: full hidden state sequence (all positions)
      - surprise: per-token surprise signal
    """

    # Class-level cache so we only load the HF model once
    _hf_backbone = None
    _hf_norm_f = None
    _hf_embeddings = None  # Embedding(50280, 768) — text tokens → vectors
    _hf_tokenizer = None   # GPTNeoX tokenizer for mamba-130m-hf

    def __new__(cls, mamba_path: str = "", config_path: str = "",
                d_model: int = 768, surprise_threshold: float = 2.0):
        torch, nn, F = _ensure_torch()

        class _MambaTemporalModule(nn.Module):
            def __init__(self, mamba_path, config_path, d_model, surprise_threshold):
                super().__init__()
                self.d_model = d_model
                self.surprise_threshold = surprise_threshold

                # Load the FULL mamba-130m-hf model (all 24 layers)
                self._backbone = None
                self._norm_f = None
                self._load_full_mamba()

                # Simple linear head for next-token prediction (surprise)
                self.surprise_head = nn.Linear(d_model, d_model, bias=False)
                self.surprise_head.weight.requires_grad = False
                nn.init.eye_(self.surprise_head.weight)  # identity init

            def _load_full_mamba(self):
                """Load the full mamba-130m-hf model from HuggingFace."""
                # Use class-level cache
                if MambaTemporalModule._hf_backbone is not None:
                    self._backbone = MambaTemporalModule._hf_backbone
                    self._norm_f = MambaTemporalModule._hf_norm_f
                    self._embeddings = MambaTemporalModule._hf_embeddings
                    self._tokenizer = MambaTemporalModule._hf_tokenizer
                    logger.info("Reusing cached mamba-130m-hf backbone (24 layers)")
                    return

                self._embeddings = None
                self._tokenizer = None

                try:
                    from transformers import AutoModelForCausalLM, AutoTokenizer
                    logger.info("Loading full state-spaces/mamba-130m-hf (24 layers, 129M params)...")
                    hf_model = AutoModelForCausalLM.from_pretrained(
                        "state-spaces/mamba-130m-hf",
                        dtype=torch.float32,
                    )
                    hf_model.eval()

                    # Load tokenizer for text → token_ids
                    self._tokenizer = AutoTokenizer.from_pretrained(
                        "state-spaces/mamba-130m-hf")

                    # Extract backbone (all 24 layers), embeddings, and final norm
                    self._backbone = hf_model.backbone
                    self._norm_f = hf_model.backbone.norm_f
                    self._embeddings = hf_model.backbone.embeddings

                    # Freeze everything
                    for p in self._backbone.parameters():
                        p.requires_grad = False

                    # Cache at class level
                    MambaTemporalModule._hf_backbone = self._backbone
                    MambaTemporalModule._hf_norm_f = self._norm_f
                    MambaTemporalModule._hf_embeddings = self._embeddings
                    MambaTemporalModule._hf_tokenizer = self._tokenizer

                    total_p = sum(p.numel() for p in self._backbone.parameters())
                    logger.info("Loaded mamba-130m-hf: %d params, %d layers",
                                total_p, len(self._backbone.layers))
                except Exception as e:
                    logger.warning("Failed to load mamba-130m-hf: %s — "
                                   "falling back to single block", e)
                    self._backbone = None
                    self._norm_f = None

                # Fallback: use single extracted block if full model unavailable
                if self._backbone is None:
                    mamba_path_attr = getattr(self, '_mamba_path', '')
                    config_path_attr = getattr(self, '_config_path', '')
                    if mamba_path_attr and os.path.isfile(mamba_path_attr):
                        _blk_cls = type(MambaBlock(self.d_model))
                        self._single_block = _blk_cls.from_pretrained(
                            mamba_path_attr, config_path_attr)
                        for p in self._single_block.parameters():
                            p.requires_grad = False
                        logger.info("Fallback: loaded single Mamba block from %s",
                                    mamba_path_attr)
                    else:
                        self._single_block = MambaBlock(self.d_model)
                        for p in self._single_block.parameters():
                            p.requires_grad = False
                        logger.warning("No Mamba model available — using random init")

            def forward(self, x_embed):
                """
                x_embed: (batch, seq, d_model) — already embedded tokens
                Returns dict with h, h_seq, surprise
                """
                with torch.no_grad():
                    if self._backbone is not None:
                        # Full 24-layer Mamba model
                        hidden = x_embed
                        for layer in self._backbone.layers:
                            hidden = layer(hidden)
                            # HF MambaBlock may return tuple (hidden,) or just tensor
                            if isinstance(hidden, tuple):
                                hidden = hidden[0]
                        h_seq = self._norm_f(hidden)  # final norm
                    else:
                        # Fallback: single block
                        h_seq = self._single_block(x_embed)

                h = h_seq[:, -1, :]  # (B, d_model) — last position

                # Compute surprise: how much the hidden state deviates from prediction
                if h_seq.shape[1] > 1:
                    predicted = self.surprise_head(h_seq[:, :-1, :])
                    actual = h_seq[:, 1:, :].detach()
                    surprise = (predicted - actual).pow(2).mean(dim=-1)  # (B, L-1)
                    # Pad first position with zero surprise
                    surprise = F.pad(surprise, (1, 0), value=0.0)
                else:
                    surprise = torch.zeros(
                        h_seq.shape[0], 1, device=h_seq.device)

                return {
                    "h": h,                    # (B, d_model) — summary hidden state
                    "h_seq": h_seq,            # (B, L, d_model) — full sequence
                    "surprise": surprise,       # (B, L) — per-token surprise
                    "mean_surprise": surprise.mean(dim=-1),  # (B,)
                    "high_surprise_mask": surprise > self.surprise_threshold,
                }

            def get_hidden_state(self, x_embed):
                """Shortcut: get just the final hidden state h."""
                with torch.no_grad():
                    if self._backbone is not None:
                        hidden = x_embed
                        for layer in self._backbone.layers:
                            hidden = layer(hidden)
                            if isinstance(hidden, tuple):
                                hidden = hidden[0]
                        h_seq = self._norm_f(hidden)
                    else:
                        h_seq = self._single_block(x_embed)
                return h_seq[:, -1, :]

            def embed_text(self, text: str, max_tokens: int = 512):
                """
                Compute a 768-dim embedding from raw text.

                Full pipeline: tokenize → embed → 24 Mamba layers → norm → mean pool.
                This is the proper way to get graph-storable embeddings from text.

                Args:
                    text: Raw text string
                    max_tokens: Max tokens (truncate longer texts)

                Returns:
                    (768,) float32 tensor — mean-pooled hidden state
                """
                if self._tokenizer is None or self._embeddings is None:
                    return None
                if not text or not text.strip():
                    return None

                with torch.no_grad():
                    # Tokenize
                    token_ids = self._tokenizer.encode(
                        text[:4000], add_special_tokens=False)
                    if not token_ids:
                        return None
                    token_ids = token_ids[:max_tokens]
                    input_ids = torch.tensor(
                        [token_ids], dtype=torch.long,
                        device=next(self._embeddings.parameters()).device)

                    # Embed tokens → (1, L, 768)
                    x_embed = self._embeddings(input_ids)

                    # Run through all 24 layers
                    if self._backbone is not None:
                        hidden = x_embed
                        for layer in self._backbone.layers:
                            hidden = layer(hidden)
                            if isinstance(hidden, tuple):
                                hidden = hidden[0]
                        h_seq = self._norm_f(hidden)  # (1, L, 768)
                    else:
                        h_seq = self._single_block(x_embed)

                    # Mean pool across sequence → (768,)
                    return h_seq.squeeze(0).mean(dim=0)

            def parameters(self, recurse=True):
                """Yield all parameters for param counting."""
                if self._backbone is not None:
                    yield from self._backbone.parameters(recurse)
                elif hasattr(self, '_single_block'):
                    yield from self._single_block.parameters(recurse)
                yield from self.surprise_head.parameters(recurse)

        return _MambaTemporalModule(mamba_path, config_path, d_model, surprise_threshold)


# ===================================================================
# Layer 3: Quantum-Like Interference Search
# ===================================================================

class OracleNetwork:
    """
    Learnable oracle for interference search.
    W_init: initializes amplitudes from Mamba hidden state
    W_query: re-biases amplitudes toward query relevance
    """

    def __new__(cls, d_hidden: int = 768, d_embedding: int = 768):
        torch, nn, _ = _ensure_torch()

        class _OracleNetwork(nn.Module):
            def __init__(self, d_hidden, d_embedding):
                super().__init__()
                self.W_init = nn.Linear(d_hidden, d_embedding, bias=False)
                self.W_query = nn.Linear(d_hidden, d_embedding, bias=False)
                # Complexity gate: predicts query complexity (0-1)
                self.complexity_gate = nn.Sequential(
                    nn.Linear(d_hidden, 256),
                    nn.ReLU(),
                    nn.Linear(256, 1),
                    nn.Sigmoid(),
                )

            def init_amplitudes(self, node_embeddings, h):
                """
                Initialize amplitude vector over graph nodes.
                node_embeddings: (N, d_embedding)
                h: (d_hidden,) or (B, d_hidden)
                Returns: (N,) or (B, N) amplitude vector
                """
                # W_init @ h → (d_embedding,)
                h_proj = self.W_init(h)  # (..., d_embedding)
                # node_embeddings @ h_proj^T → (N,) or (B, N)
                if h_proj.dim() == 1:
                    scores = node_embeddings @ h_proj  # (N,)
                else:
                    scores = node_embeddings @ h_proj.T  # (N, B) → transpose
                    scores = scores.T  # (B, N)
                return torch.softmax(scores.float(), dim=-1)

            def oracle_relevance(self, node_embeddings, h):
                """
                Compute per-node relevance for oracle re-bias.
                Returns: (N,) or (B, N) relevance in [0, 1]
                """
                h_proj = self.W_query(h)
                if h_proj.dim() == 1:
                    scores = node_embeddings @ h_proj
                else:
                    scores = (node_embeddings @ h_proj.T).T
                return torch.sigmoid(scores.float())

            def adaptive_iterations(self, h):
                """
                Predict number of search iterations (1-7).
                h: (d_hidden,) or (B, d_hidden)
                """
                if h.dim() == 1:
                    h = h.unsqueeze(0)
                complexity = self.complexity_gate(h).squeeze(-1)  # (B,)
                iters = (1 + 6 * complexity).clamp(1, 7).int()
                return iters.item() if iters.numel() == 1 else iters

        return _OracleNetwork(d_hidden, d_embedding)


class InterferenceSearch:
    """
    Layer 3: Quantum-like interference search on graph.

    Algorithm (from PRD §5):
    1. Initialize amplitudes from Mamba hidden state via oracle
    2. Iterate (1-7 times, adaptive):
       a. Graph diffusion: amplitudes = adj_matrix @ amplitudes
       b. Oracle re-bias: amplitudes *= sigmoid(embeddings @ W_query @ h)
       c. Confidence gating: amplitudes *= confidence_vector
       d. Grover reflection: amplitudes = 2*mean - amplitudes
       e. Normalize
    3. Collapse: top-k nodes by amplitude
    """

    def __new__(cls, oracle=None, default_k: int = 32,
                max_iterations: int = 7):
        torch, nn, F = _ensure_torch()

        class _InterferenceSearch(nn.Module):
            def __init__(self, oracle, default_k, max_iterations):
                super().__init__()
                self.oracle = oracle
                self.default_k = default_k
                self.max_iterations = max_iterations

            def search(self, h, graph_tensors: Dict, k: int = 0) -> Dict:
                """
                Run interference search.

                Args:
                    h: (d_hidden,) Mamba hidden state
                    graph_tensors: dict from QORGraph.get_search_tensors():
                        - node_ids: list of entity_id strings
                        - embeddings: (N, d) float tensor
                        - adj_indices: (2, E) long tensor (sparse COO)
                        - adj_weights: (E,) float tensor
                        - confidence: (N,) float tensor
                    k: top-k to retrieve (0 = use default)

                Returns dict:
                    - node_ids: list of top-k entity_id strings
                    - amplitudes: (k,) amplitude values
                    - iterations: how many iterations ran
                """
                if k <= 0:
                    k = self.default_k

                node_ids = graph_tensors["node_ids"]
                embeddings = graph_tensors["embeddings"]
                confidence = graph_tensors["confidence"]
                N = len(node_ids)

                if N == 0:
                    return {"node_ids": [], "amplitudes": torch.tensor([]),
                            "iterations": 0}

                k = min(k, N)

                # Build sparse adjacency
                adj_indices = graph_tensors["adj_indices"]
                adj_weights = graph_tensors["adj_weights"]
                adj = torch.sparse_coo_tensor(
                    adj_indices, adj_weights,
                    size=(N, N), dtype=torch.float32
                ).coalesce()

                # 1. Initialize amplitudes
                amplitudes = self.oracle.init_amplitudes(embeddings, h)
                if amplitudes.dim() > 1:
                    amplitudes = amplitudes.squeeze(0)

                # Determine iterations
                num_iter = self.oracle.adaptive_iterations(h)
                num_iter = min(num_iter, self.max_iterations)

                # 2. Interference iterations
                for i in range(num_iter):
                    # Graph diffusion: propagate along edges
                    diffused = torch.sparse.mm(adj, amplitudes.unsqueeze(-1))
                    amplitudes = diffused.squeeze(-1)

                    # Handle case where diffusion zeroes out
                    if amplitudes.abs().sum() < 1e-12:
                        amplitudes = self.oracle.init_amplitudes(
                            embeddings, h)
                        if amplitudes.dim() > 1:
                            amplitudes = amplitudes.squeeze(0)

                    # Oracle re-bias
                    relevance = self.oracle.oracle_relevance(embeddings, h)
                    if relevance.dim() > 1:
                        relevance = relevance.squeeze(0)
                    amplitudes = amplitudes * relevance

                    # Confidence gating
                    amplitudes = amplitudes * confidence

                    # Grover reflection about mean
                    mean_amp = amplitudes.mean()
                    amplitudes = 2 * mean_amp - amplitudes

                    # Normalize
                    norm = amplitudes.norm()
                    if norm > 1e-12:
                        amplitudes = amplitudes / norm

                # 3. Collapse: top-k by amplitude magnitude
                top_vals, top_idx = torch.topk(
                    amplitudes.abs(), k=k, sorted=True)

                result_ids = [node_ids[i] for i in top_idx.tolist()]

                return {
                    "node_ids": result_ids,
                    "amplitudes": top_vals,
                    "all_amplitudes": amplitudes,
                    "iterations": num_iter,
                }

            def forward(self, h, graph_tensors, k=0):
                return self.search(h, graph_tensors, k)

        return _InterferenceSearch(oracle, default_k, max_iterations)


# ===================================================================
# Layer 4: Reasoning Layer (Cross-Attention + GRU)
# ===================================================================

class ReasoningLayer:
    """
    Layer 4: Cross-attention fuses temporal h with retrieved graph context,
    then GRU iterative refinement with adaptive halting.

    Architecture (PRD §6):
        Q = h @ W_Q           (query from temporal state)
        K = C @ W_K           (keys from graph context)
        V = C @ W_V           (values from graph context)
        fused = softmax(Q K^T / sqrt(d)) @ V

        state = fused
        for step in 1..max_steps:
            state = GRU(state, fused)
            if halting_network(state) > threshold: break

    Parameter budget:
        Cross-attention (Q/K/V + out): ~2.4M
        GRU cell (768→768):            ~3.5M
        Halting network:               ~0.1M
        Fusion projection:             ~0.6M
        Layer norms:                   ~0.003M
        Total:                         ~6.6M
    """

    def __new__(cls, d_hidden: int = 768, n_heads: int = 8,
                max_steps: int = 5, halt_threshold: float = 0.8):
        torch, nn, F = _ensure_torch()

        class _ReasoningLayer(nn.Module):
            def __init__(self, d_hidden, n_heads, max_steps, halt_threshold):
                super().__init__()
                self.d_hidden = d_hidden
                self.n_heads = n_heads
                self.max_steps = max_steps
                self.halt_threshold = halt_threshold

                # Cross-attention: Q from temporal h, K/V from retrieved nodes
                self.cross_attn = nn.MultiheadAttention(
                    embed_dim=d_hidden,
                    num_heads=n_heads,
                    batch_first=True,
                )
                self.attn_norm_q = nn.LayerNorm(d_hidden)
                self.attn_norm_kv = nn.LayerNorm(d_hidden)

                # GRU iterative refinement
                self.gru = nn.GRUCell(d_hidden, d_hidden)

                # Halting network: predicts confidence (0-1)
                self.halting = nn.Sequential(
                    nn.Linear(d_hidden, 128),
                    nn.ReLU(),
                    nn.Linear(128, 1),
                    nn.Sigmoid(),
                )

                # Fusion: combine cross-attention output with temporal h
                self.fusion = nn.Linear(d_hidden * 2, d_hidden)
                self.out_norm = nn.LayerNorm(d_hidden)

            def forward(self, h, retrieved_embeddings):
                """
                h: (B, d_hidden) or (d_hidden,) from Mamba temporal module
                retrieved_embeddings: (k, d_hidden) or (B, k, d_hidden)

                Returns: (B, d_hidden) reasoning output
                """
                if h.dim() == 1:
                    h = h.unsqueeze(0)  # (1, d_hidden)
                if retrieved_embeddings.dim() == 2:
                    retrieved_embeddings = retrieved_embeddings.unsqueeze(0)

                B = h.shape[0]

                # Cross-attention: Q from h, K/V from retrieved nodes
                query = self.attn_norm_q(h.unsqueeze(1))    # (B, 1, d)
                kv = self.attn_norm_kv(retrieved_embeddings)  # (B, k, d)

                attn_out, _ = self.cross_attn(
                    query=query, key=kv, value=kv
                )  # (B, 1, d)
                attn_out = attn_out.squeeze(1)  # (B, d)

                # Fusion: combine temporal h with attention context
                fused = self.fusion(
                    torch.cat([h, attn_out], dim=-1))  # (B, d)

                # GRU iterative refinement with adaptive halting
                state = fused
                steps_taken = 0

                for step in range(self.max_steps):
                    state = self.gru(state, fused)  # (B, d)
                    steps_taken += 1

                    # Check halting condition
                    confidence = self.halting(state)  # (B, 1)
                    if confidence.min().item() > self.halt_threshold:
                        break

                output = self.out_norm(state)  # (B, d)
                return output

            def forward_with_info(self, h, retrieved_embeddings):
                """Forward pass that also returns diagnostic info."""
                if h.dim() == 1:
                    h = h.unsqueeze(0)
                if retrieved_embeddings.dim() == 2:
                    retrieved_embeddings = retrieved_embeddings.unsqueeze(0)

                B = h.shape[0]
                query = self.attn_norm_q(h.unsqueeze(1))
                kv = self.attn_norm_kv(retrieved_embeddings)

                attn_out, attn_weights = self.cross_attn(
                    query=query, key=kv, value=kv)
                attn_out = attn_out.squeeze(1)

                fused = self.fusion(torch.cat([h, attn_out], dim=-1))

                state = fused
                steps_taken = 0
                confidences = []

                for step in range(self.max_steps):
                    state = self.gru(state, fused)
                    steps_taken += 1
                    conf = self.halting(state)
                    confidences.append(conf.item() if conf.numel() == 1
                                       else conf.mean().item())
                    if conf.min().item() > self.halt_threshold:
                        break

                output = self.out_norm(state)
                return {
                    "output": output,
                    "steps_taken": steps_taken,
                    "confidences": confidences,
                    "attn_weights": attn_weights,
                    "final_confidence": confidences[-1] if confidences else 0.0,
                }

        return _ReasoningLayer(d_hidden, n_heads, max_steps, halt_threshold)


# ===================================================================
# NGREBrain: Full 4-layer orchestrator
# ===================================================================

class NGREBrain:
    """
    Full NGRE 4-layer brain.

    Layer 1: MambaTemporalModule (frozen) → h ∈ R^768
    Layer 2: QORGraph (external)           → dynamic memory
    Layer 3: InterferenceSearch            → retrieve relevant nodes
    Layer 4: ReasoningLayer                → fuse h + context → output

    Usage:
        brain = NGREBrain(config)
        brain.set_graph(graph)  # connect to QORGraph
        result = brain.process(token_embeddings)
    """

    def __new__(cls, config=None, d_hidden: int = 768,
                mamba_path: str = "", config_path: str = "",
                search_k: int = 32, max_iterations: int = 7):
        torch, nn, _ = _ensure_torch()

        class _NGREBrain(nn.Module):
            def __init__(self, config, d_hidden, mamba_path, config_path,
                         search_k, max_iterations):
                super().__init__()

                # Resolve from config if provided
                if config is not None:
                    d_hidden = getattr(config, 'd_hidden', d_hidden)
                    mamba_path = getattr(config, 'mamba_checkpoint', mamba_path)
                    config_path = getattr(config, 'mamba_config', config_path)
                    search_k = getattr(config, 'search_default_k', search_k)
                    max_iterations = getattr(
                        config, 'search_max_iterations', max_iterations)
                    surprise_th = getattr(
                        config, 'bootstrap_surprise_threshold', 2.0)
                else:
                    surprise_th = 2.0

                self.d_hidden = d_hidden

                # Layer 1: Temporal
                self.temporal = MambaTemporalModule(
                    mamba_path=mamba_path,
                    config_path=config_path,
                    d_model=d_hidden,
                    surprise_threshold=surprise_th,
                )

                # Layer 3: Search
                self.oracle = OracleNetwork(d_hidden, d_hidden)
                self.search = InterferenceSearch(
                    oracle=self.oracle,
                    default_k=search_k,
                    max_iterations=max_iterations,
                )

                # Layer 4: Reasoning (Phase C placeholder)
                self.reasoning = ReasoningLayer(d_hidden)

                # Graph reference (set via set_graph)
                self._graph = None

                # Training state — untrained until explicit training or load
                self._trained = False

            def set_graph(self, graph):
                """Connect to a QORGraph instance (Layer 2)."""
                self._graph = graph

            def save(self, path: str):
                """Save NGRE brain weights + config to file."""
                import os as _os
                _os.makedirs(_os.path.dirname(path) or ".", exist_ok=True)
                torch.save({
                    "state_dict": self.state_dict(),
                    "config": {"d_model": self.d_hidden},
                    "trained": self._trained,
                    "phase": 4,
                }, path)

            def load(self, path: str) -> bool:
                """Load NGRE brain weights. Returns True if loaded.

                Note: weights_only=False is required for state_dict tensors.
                Only load files generated by NGREBrain.save() or train_ngre.py.
                """
                import os as _os
                if not _os.path.exists(path):
                    return False
                try:
                    data = torch.load(path, map_location="cpu",
                                      weights_only=False)
                    sd = data.get("state_dict", data)
                    self.load_state_dict(sd, strict=False)
                    self._trained = data.get("trained", True)
                    self.eval()
                    return True
                except Exception as e:
                    logger.warning("NGRE brain load failed: %s", e)
                    return False

            def process(self, x_embed, graph_tensors: Optional[Dict] = None,
                        k: int = 0) -> Dict:
                """
                Full NGRE forward pass.

                Args:
                    x_embed: (B, L, d_hidden) token embeddings
                    graph_tensors: pre-computed search tensors (optional,
                                   auto-fetched from graph if None)
                    k: top-k retrieval count

                Returns dict:
                    h: temporal hidden state
                    surprise: per-token surprise
                    search_result: interference search output
                    reasoning_output: fused context vector
                """
                # Layer 1: Temporal
                temporal_out = self.temporal(x_embed)
                h = temporal_out["h"]  # (B, d_hidden) or (d_hidden,)

                result = {"temporal": temporal_out, "h": h}

                # Layer 2 + 3: Graph search
                if graph_tensors is None and self._graph is not None:
                    graph_tensors = self._graph.get_search_tensors()

                if graph_tensors is not None and len(
                        graph_tensors.get("node_ids", [])) > 0:
                    h_search = h[0] if h.dim() > 1 else h
                    search_result = self.search(
                        h_search, graph_tensors, k=k)
                    result["search_result"] = search_result

                    # Layer 4: Reasoning
                    if search_result["node_ids"]:
                        emb = graph_tensors["embeddings"]
                        top_idx = []
                        id_list = graph_tensors["node_ids"]
                        for nid in search_result["node_ids"]:
                            if nid in id_list:
                                top_idx.append(id_list.index(nid))
                        if top_idx:
                            idx_t = torch.tensor(
                                top_idx, device=emb.device)
                            retrieved = emb[idx_t]  # (k, d)
                            reasoning_out = self.reasoning(h, retrieved)
                            result["reasoning_output"] = reasoning_out
                else:
                    result["search_result"] = {
                        "node_ids": [], "amplitudes": torch.tensor([]),
                        "iterations": 0}

                return result

            def forward(self, x_embed, graph_tensors=None, k=0):
                return self.process(x_embed, graph_tensors, k)

            def compute_embedding(self, text: str) -> Optional[list]:
                """
                Compute a 768-dim embedding vector from raw text.

                Uses Mamba's own tokenizer + embedding layer + all 24 frozen layers.
                Returns a plain Python list suitable for graph.add_node(embedding=...).

                Args:
                    text: Raw text string

                Returns:
                    list of 768 floats, or None if embedding unavailable
                """
                vec = self.temporal.embed_text(text)
                if vec is None:
                    return None
                return vec.cpu().tolist()

            def create_node_from_surprise(self, graph, text: str,
                                          embedding, surprise: float,
                                          source: str = "mamba_bootstrap"):
                """
                Create a graph node when surprise exceeds threshold.
                Used during Phase 2 graph bootstrap.
                """
                if graph is None:
                    return None

                import hashlib
                content_hash = hashlib.sha256(
                    text[:200].encode()).hexdigest()[:12]
                node_id = f"mamba:{content_hash}"

                emb_list = None
                if embedding is not None:
                    if hasattr(embedding, 'tolist'):
                        emb_list = embedding.detach().cpu().tolist()
                    elif isinstance(embedding, list):
                        emb_list = embedding

                graph.add_node(
                    node_id,
                    node_type="knowledge",
                    properties={
                        "content": text[:2000],
                        "surprise": float(surprise),
                        "source": source,
                    },
                    confidence=min(0.5 + surprise * 0.1, 0.95),
                    source=source,
                    embedding=emb_list,
                    text_summary=text[:200],
                )
                return node_id

        return _NGREBrain(config, d_hidden, mamba_path, config_path,
                          search_k, max_iterations)


# ===================================================================
# Utility: load NGRE brain from checkpoint paths
# ===================================================================

def create_ngre_brain(checkpoint_dir: str = "checkpoints",
                      config=None) -> Any:
    """
    Create an NGREBrain instance, loading trained weights if available.

    Searches for trained checkpoints in order:
      1. {checkpoint_dir}/ngre_brain_final.pt  (Phase 4 export)
      2. qor-data/ngre/ngre_brain_finetuned.pt (Phase 3)
      3. qor-data/ngre/ngre_brain.pt           (Phase 2)

    If no checkpoint found, returns untrained brain (_trained=False).

    Args:
        checkpoint_dir: directory to search for checkpoints
        config: optional NGREConfig dataclass

    Returns:
        NGREBrain instance (nn.Module)
    """
    brain = NGREBrain(
        config=config,
        mamba_path="",   # full model loaded from HF directly
        config_path="",
    )

    # Try to load trained weights
    candidates = [
        os.path.join(checkpoint_dir, "ngre_brain_final.pt"),
        os.path.join("qor-data", "ngre", "ngre_brain_finetuned.pt"),
        os.path.join("qor-data", "ngre", "ngre_brain.pt"),
    ]
    for path in candidates:
        if os.path.exists(path):
            if brain.load(path):
                logger.info("NGRE brain loaded from %s (trained=%s)",
                            path, brain._trained)
            break

    return brain


# ===================================================================
# TreeGate: Central Orchestrator (PRD §9)
# ===================================================================

# Confidence tiers for TreeGate routing
TIER_HIGH = "HIGH"       # >= 0.7 — use tree data directly
TIER_MEDIUM = "MEDIUM"   # 0.3-0.7 — tree + verify with skill
TIER_LOW = "LOW"         # < 0.3 — call skill, write to tree
TIER_NONE = "NONE"       # no data from any source

# Complexity routing levels
ROUTE_TEMPLATE = "TEMPLATE"      # < 2ms, ~70% — high-confidence simple fact
ROUTE_LLM_FAST = "LLM_FAST"     # 100-300ms, ~20% — SmolLM3 /no_think
ROUTE_LLM_THINK = "LLM_THINK"   # 500-2000ms, ~7% — SmolLM3 /think
ROUTE_CLOUD = "CLOUD"            # 1-5s, ~3% — external API


@dataclass
class RoutingDecision:
    """Output of ComplexityGate.route()."""
    level: str = ROUTE_LLM_FAST
    confidence: float = 0.5
    reason: str = ""


@dataclass
class TreeGateResult:
    """Output of TreeGate.evaluate()."""
    tier: str = TIER_NONE
    max_confidence: float = 0.0
    context_parts: List[str] = None
    sources_used: List[str] = None
    routing: RoutingDecision = None
    prompt: str = ""
    adjustments: Any = None  # ContextAdjustments from knowledge_tree
    needs_tool: bool = False
    tree_has_data: bool = False
    feedback_event: Any = None  # FeedbackEvent if user feedback was detected
    feedback_response: str = ""  # Human-readable feedback acknowledgement
    saved_to_tree: int = 0  # Number of triples saved to graph by _save_to_tree

    def __post_init__(self):
        if self.context_parts is None:
            self.context_parts = []
        if self.sources_used is None:
            self.sources_used = []
        if self.routing is None:
            self.routing = RoutingDecision()


class ComplexityGate:
    """
    Routes queries to the cheapest sufficient response level.

    PRD §8.2:
        confidence > 0.85  → Template (no LLM, < 2ms, ~70%)
        0.5-0.85           → SmolLM3 /no_think (100-300ms, ~20%)
        0.2-0.5            → SmolLM3 /think (500-2000ms, ~7%)
        < 0.2 or complex   → Cloud API (1-5s, ~3%)

    Template patterns: simple factual lookups like "AAPL price is $245.30"
    """

    # Patterns that can be answered by template (no LLM needed)
    TEMPLATE_PATTERNS = [
        # price queries  —  "what is the price of X"
        "price", "how much is", "what is .* worth",
        # time queries
        "what time is it", "current time",
        # simple definitions already in context
        "what is the definition of",
    ]

    # Multi-domain / complex query signals → upgrade to /think or cloud
    COMPLEX_SIGNALS = [
        "how might", "what would happen if", "compare", "analyze",
        "strategy", "portfolio", "implications", "relationship between",
        "macro", "forecast", "predict", "explain why", "multi-step",
        "pros and cons", "trade-off", "should i",
    ]

    @staticmethod
    def route(confidence: float, question: str,
              has_context: bool = True) -> RoutingDecision:
        """
        Decide which response route to use.

        Args:
            confidence: max tree node confidence (0-1)
            question: user's question (for complexity detection)
            has_context: whether context was found

        Returns:
            RoutingDecision with level, confidence, reason
        """
        q_lower = question.lower()

        # Check for complex signals → upgrade
        is_complex = any(sig in q_lower
                         for sig in ComplexityGate.COMPLEX_SIGNALS)

        if not has_context:
            if is_complex:
                return RoutingDecision(
                    level=ROUTE_CLOUD,
                    confidence=confidence,
                    reason="no context + complex query",
                )
            return RoutingDecision(
                level=ROUTE_LLM_FAST,
                confidence=confidence,
                reason="no context, simple query",
            )

        # Template: very high confidence + simple pattern
        if confidence > 0.85 and not is_complex:
            is_template = any(
                pat in q_lower for pat in ComplexityGate.TEMPLATE_PATTERNS)
            if is_template:
                return RoutingDecision(
                    level=ROUTE_TEMPLATE,
                    confidence=confidence,
                    reason="high confidence + template pattern",
                )

        # High confidence → fast LLM
        if confidence > 0.85:
            return RoutingDecision(
                level=ROUTE_LLM_FAST,
                confidence=confidence,
                reason="high confidence context",
            )

        # Medium confidence
        if confidence >= 0.5:
            if is_complex:
                return RoutingDecision(
                    level=ROUTE_LLM_THINK,
                    confidence=confidence,
                    reason="medium confidence + complex query",
                )
            return RoutingDecision(
                level=ROUTE_LLM_FAST,
                confidence=confidence,
                reason="medium confidence",
            )

        # Low confidence
        if confidence >= 0.2:
            if is_complex:
                return RoutingDecision(
                    level=ROUTE_CLOUD,
                    confidence=confidence,
                    reason="low confidence + complex query",
                )
            return RoutingDecision(
                level=ROUTE_LLM_THINK,
                confidence=confidence,
                reason="low confidence",
            )

        # Very low confidence
        return RoutingDecision(
            level=ROUTE_CLOUD if is_complex else ROUTE_LLM_THINK,
            confidence=confidence,
            reason="very low confidence",
        )


# Strict LLM prompt template (PRD §9)
STRICT_PROMPT_TEMPLATE = (
    "You are a precise assistant. Answer ONLY from the verified facts below. "
    "This is real-time data from our live database and tools — NOT user-provided. "
    "Use ALL of it to give a comprehensive answer. If the facts don't fully "
    "cover the question, say what you can and note what's missing. "
    "NEVER make up information.\n\n"
    "VERIFIED FACTS:\n{context}\n\n"
    "QUESTION: {question}"
)

# Template response patterns (no LLM needed)
_TEMPLATE_EXTRACTORS = {
    "price": lambda ctx: _extract_price_from_context(ctx),
    "time": lambda ctx: _extract_time_from_context(ctx),
}


def _extract_price_from_context(context: str) -> Optional[str]:
    """Try to extract a clean price answer from context."""
    import re
    # Look for patterns like "$245.30" or "price: 245.30" or "current price is 245.30"
    patterns = [
        r'(?:price|worth|trading at|currently at)[:\s]+\$?([\d,]+\.?\d*)',
        r'\$([\d,]+\.?\d*)',
    ]
    for pat in patterns:
        m = re.search(pat, context, re.IGNORECASE)
        if m:
            return m.group(0).strip()
    return None


def _extract_time_from_context(context: str) -> Optional[str]:
    """Try to extract time from context."""
    import re
    m = re.search(r'\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?', context)
    if m:
        return m.group(0)
    return None


class CloudAPIClient:
    """OpenAI-compatible API client for CLOUD routing level.

    Used by ComplexityGate when confidence < 0.2 and query is complex.
    Supports any OpenAI-compatible endpoint (OpenAI, Anthropic via proxy,
    local vLLM, Ollama, etc.).
    """

    def __init__(self, api_key: str = "", base_url: str = "",
                 model: str = "gpt-4o-mini", timeout: int = 30):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url or os.environ.get(
            "CLOUD_API_URL", "https://api.openai.com/v1")
        self.model = model or os.environ.get("CLOUD_API_MODEL", "gpt-4o-mini")
        self.timeout = timeout
        self._available = bool(self.api_key)

    @property
    def available(self) -> bool:
        return self._available

    def generate(self, prompt: str, system: str = "",
                 max_tokens: int = 500, temperature: float = 0.3) -> Optional[str]:
        """Call cloud API with OpenAI-compatible chat format.

        Returns generated text or None on failure.
        """
        if not self._available:
            return None
        try:
            import urllib.request
            import json as _json

            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            payload = _json.dumps({
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }).encode("utf-8")

            url = f"{self.base_url.rstrip('/')}/chat/completions"
            req = urllib.request.Request(
                url,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = _json.loads(resp.read())
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning("Cloud API error: %s", e)
            return None


class RoutingStatsTracker:
    """Track ComplexityGate routing decisions for monitoring."""

    def __init__(self):
        self._counts = {
            ROUTE_TEMPLATE: 0,
            ROUTE_LLM_FAST: 0,
            ROUTE_LLM_THINK: 0,
            ROUTE_CLOUD: 0,
        }
        self._total = 0

    def record(self, level: str):
        self._counts[level] = self._counts.get(level, 0) + 1
        self._total += 1

    def stats(self) -> Dict:
        result = dict(self._counts)
        result["total"] = self._total
        if self._total > 0:
            for key in (ROUTE_TEMPLATE, ROUTE_LLM_FAST,
                        ROUTE_LLM_THINK, ROUTE_CLOUD):
                result[f"{key}_pct"] = round(
                    100 * self._counts.get(key, 0) / self._total, 1)
        return result

    def reset(self):
        for k in self._counts:
            self._counts[k] = 0
        self._total = 0


class TreeGate:
    """
    Central orchestrator: tree-first, skill-fallback, LLM-last.

    PRD §9 flow:
        0. FEEDBACK DETECTION — check for corrections/preferences/complaints
        1. TREE SEARCH — always search graph first
        2. EVALUATE CONFIDENCE — max confidence from results
        3. ROUTE by confidence tier:
           >=0.7 HIGH:   Use tree data -> LLM
           0.3-0.7 MED: Use tree + call skill to verify -> LLM
           <0.3 LOW:    Call skill -> write to tree -> LLM
           NONE:        No data anywhere -> "I don't know"
        4. BUILD CONTEXT PACKAGE — blocked patterns filtered,
           corrections as warnings, trade lessons, preferences
        5. STRICT LLM PROMPT — "Answer ONLY from facts below"
        6. COMPLEXITY ROUTING — Template / SmolLM3 / Cloud
        7. SELF-GROWING KNOWLEDGE — save extracted triples to graph

    Usage:
        gate = TreeGate(graph, user_id)
        result = gate.evaluate(question, tree_results, adjustments)
        # result.tier, result.context_parts, result.routing, result.prompt
    """

    # Thresholds
    HIGH_CONFIDENCE = 0.7
    MEDIUM_CONFIDENCE = 0.3

    # Token budget allocations for pack_context (percentage of total budget)
    _BUDGET_GRAPH = 0.30       # Graph facts: 30%
    _BUDGET_MEMORY = 0.25      # Memory hits: 25%
    _BUDGET_RAG = 0.20         # RAG chunks: 20%
    _BUDGET_TOOL = 0.15        # Tool results: 15%
    _BUDGET_HISTORICAL = 0.10  # Historical: 10%

    def __init__(self, graph=None, user_id: str = "user:default",
                 memory_store=None):
        self.graph = graph
        self.user_id = user_id
        self._memory_store = memory_store

    def evaluate(self, question: str,
                 tree_results: Optional[List[Dict]] = None,
                 adjustments=None,
                 tool_result: Optional[Tuple[str, str]] = None,
                 last_answer: Optional[Dict] = None,
                 chat_context: Optional[List] = None,
                 ) -> TreeGateResult:
        """
        Evaluate a question through the TreeGate pipeline.

        Args:
            question: user's question
            tree_results: output from tree_search() — list of
                          {content, source, score, timestamp}
            adjustments: ContextAdjustments from AnswerFilter
            tool_result: (tool_name, tool_data) if a tool was called
            last_answer: previous answer dict (for feedback detection)
            chat_context: recent chat messages (for feedback context)

        Returns:
            TreeGateResult with tier, context, routing, prompt
        """
        result = TreeGateResult()

        # ── Step 0: Feedback detection ──
        # Check if the user is giving feedback (correction, preference, complaint)
        # before doing normal query processing. If feedback is detected, route to
        # the FeedbackProcessor to store it in the graph.
        try:
            from .knowledge_tree import FeedbackDetector, FeedbackProcessor
            feedback = FeedbackDetector.detect(
                question, last_answer=last_answer,
                chat_context=chat_context)
            if feedback is not None:
                result.feedback_event = feedback
                # Process feedback: write to graph
                if self.graph is not None:
                    fb_response = FeedbackProcessor.process(
                        feedback, self.graph, self.user_id,
                        memory_store=self._memory_store)
                    if fb_response:
                        result.feedback_response = fb_response
                        logger.info("Feedback processed: %s -> %s",
                                    feedback.event_type, fb_response)
        except Exception as e:
            logger.debug("Feedback detection error: %s", e)

        # ── Step 1: Evaluate tree results ──
        if tree_results:
            result.tree_has_data = True
            result.max_confidence = max(
                r.get("score", 0.0) for r in tree_results)

            # Filter out blocked content
            blocked = set()
            if adjustments and hasattr(adjustments, 'blocked_patterns'):
                blocked = set(
                    p.lower() for p in adjustments.blocked_patterns)

            for r in tree_results:
                content = r.get("content", "")
                # Skip blocked content
                if blocked and any(
                        b in content.lower() for b in blocked):
                    continue
                result.context_parts.append(content)
                src = r.get("source", "tree")
                result.sources_used.append(f"tree:{src}")

        # ── Step 2: Determine confidence tier ──
        if result.max_confidence >= self.HIGH_CONFIDENCE:
            result.tier = TIER_HIGH
            result.needs_tool = False
        elif result.max_confidence >= self.MEDIUM_CONFIDENCE:
            result.tier = TIER_MEDIUM
            result.needs_tool = True  # Verify with skill
        elif result.tree_has_data:
            result.tier = TIER_LOW
            result.needs_tool = True  # Fetch fresh data
        else:
            result.tier = TIER_NONE
            result.needs_tool = True

        # ── Step 3: Include tool result if provided ──
        if tool_result is not None:
            tool_name, tool_data = tool_result
            if tool_data:
                result.context_parts.append(tool_data)
                result.sources_used.append(f"tool:{tool_name}")
                # Tool data boosts confidence
                result.max_confidence = max(
                    result.max_confidence, 0.75)
                result.needs_tool = False

        # ── Step 4: Apply adjustments ──
        if adjustments:
            result.adjustments = adjustments
            # Add corrections as warnings
            if hasattr(adjustments, 'corrections'):
                for corr in adjustments.corrections:
                    result.context_parts.append(
                        f"[CORRECTION WARNING: {corr}]")
            # Add trade lessons
            if hasattr(adjustments, 'trade_lessons'):
                for lesson in adjustments.trade_lessons:
                    result.context_parts.append(
                        f"[TRADE LESSON: {lesson}]")

        # ── Step 5: Complexity routing ──
        has_context = len(result.context_parts) > 0
        result.routing = ComplexityGate.route(
            result.max_confidence, question, has_context)

        # ── Step 6: Build prompt (with structured context packing) ──
        if result.context_parts:
            # Categorize context parts by source for structured packing
            sources = self._categorize_context(
                result.context_parts, result.sources_used)
            context_text = self.pack_context(sources)
            result.prompt = STRICT_PROMPT_TEMPLATE.format(
                context=context_text, question=question)
        elif result.tier == TIER_NONE:
            result.prompt = ""  # No prompt — "I don't know"

        # ── Step 7: Self-growing knowledge cycle ──
        # Extract entities/triples from the assembled context and save to graph.
        # This makes the tree grow with every query — even without explicit
        # user feedback or tool calls.
        try:
            saved = self._save_to_tree(question, result, self.graph)
            result.saved_to_tree = saved
        except Exception as e:
            logger.debug("_save_to_tree error: %s", e)

        # ── Step 8: Per-step maintenance ──
        # Bump access counts and micro-decay edges for accessed nodes
        if self.graph is not None and hasattr(self.graph, 'per_step_maintenance'):
            try:
                # Collect accessed node IDs from tree_results sources
                accessed_node_ids = []
                if tree_results:
                    for r in tree_results:
                        content = r.get("content", "")
                        if content:
                            # Use first few meaningful words as potential node IDs
                            words = [w.strip().lower().replace(" ", "_")
                                     for w in content.split()[:3]
                                     if len(w) > 2]
                            accessed_node_ids.extend(words[:2])
                self.graph.per_step_maintenance(
                    accessed_nodes=accessed_node_ids if accessed_node_ids else None)
            except Exception:
                pass  # Never let maintenance break the query path

        return result

    @staticmethod
    def _categorize_context(context_parts: List[str],
                            sources_used: List[str]) -> Dict[str, str]:
        """Categorize context parts by source type for pack_context.

        Maps each context part to one of the 5 budget categories based
        on its source tag. Falls back to 'memory' for unrecognized sources.

        Args:
            context_parts: list of context strings
            sources_used: parallel list of source tags (e.g. "tree:graph",
                          "tool:crypto_price")

        Returns:
            Dict with keys: graph, memory, rag, tool, historical.
            Values are concatenated context strings per category.
        """
        buckets: Dict[str, List[str]] = {
            "graph": [],
            "memory": [],
            "rag": [],
            "tool": [],
            "historical": [],
        }

        for i, part in enumerate(context_parts):
            src = sources_used[i] if i < len(sources_used) else ""
            src_lower = src.lower()

            if src_lower.startswith("tool:"):
                buckets["tool"].append(part)
            elif "graph" in src_lower:
                buckets["graph"].append(part)
            elif "rag" in src_lower:
                buckets["rag"].append(part)
            elif "historical" in src_lower:
                buckets["historical"].append(part)
            elif part.startswith("[CORRECTION WARNING:"):
                # Corrections go into graph bucket (structured knowledge)
                buckets["graph"].append(part)
            elif part.startswith("[TRADE LESSON:"):
                # Trade lessons go into memory bucket
                buckets["memory"].append(part)
            else:
                # Default: memory bucket
                buckets["memory"].append(part)

        return {k: "\n".join(v) for k, v in buckets.items()}

    def pack_context(self, sources: Dict[str, str],
                     token_budget: int = 4096) -> str:
        """Structured token budgeting across knowledge sources.

        Allocates a token budget across 5 source types, truncates each
        to its allocation, and returns the packed context string.

        Token counting is approximate: 4 characters = 1 token.

        Budget allocation:
            Graph facts:  30% (most important structured knowledge)
            Memory hits:  25%
            RAG chunks:   20%
            Tool results: 15%
            Historical:   10%

        Unused budget from empty sources is redistributed proportionally
        to sources that have content, so no budget is wasted.

        Args:
            sources: dict with keys 'graph', 'memory', 'rag', 'tool',
                     'historical'. Values are raw context strings.
            token_budget: total token budget (default 4096).

        Returns:
            Packed context string within budget.
        """
        CHARS_PER_TOKEN = 4
        total_chars = token_budget * CHARS_PER_TOKEN

        # Base allocations
        allocations = {
            "graph": self._BUDGET_GRAPH,
            "memory": self._BUDGET_MEMORY,
            "rag": self._BUDGET_RAG,
            "tool": self._BUDGET_TOOL,
            "historical": self._BUDGET_HISTORICAL,
        }

        # Calculate which sources actually have content
        active = {}
        inactive_budget = 0.0
        for key, pct in allocations.items():
            text = sources.get(key, "").strip()
            if text:
                active[key] = pct
            else:
                inactive_budget += pct

        # Redistribute unused budget proportionally to active sources
        if active and inactive_budget > 0:
            active_total = sum(active.values())
            if active_total > 0:
                for key in active:
                    active[key] += inactive_budget * (
                        active[key] / active_total)

        # Truncate each source to its character budget
        sections = []
        section_labels = {
            "graph": "GRAPH FACTS",
            "memory": "MEMORY",
            "rag": "DOCUMENTS",
            "tool": "TOOL DATA",
            "historical": "HISTORICAL",
        }

        # Order: graph first (most important), then tool, memory, rag, historical
        order = ["graph", "tool", "memory", "rag", "historical"]
        for key in order:
            text = sources.get(key, "").strip()
            if not text:
                continue
            char_budget = int(total_chars * active.get(key, 0))
            if char_budget <= 0:
                continue
            if len(text) > char_budget:
                text = text[:char_budget] + "..."
            label = section_labels.get(key, key.upper())
            sections.append(f"[{label}]\n{text}")

        packed = "\n\n".join(sections)

        # Final safety cap at total budget
        if len(packed) > total_chars:
            packed = packed[:total_chars] + "\n[... truncated]"

        return packed

    def _save_to_tree(self, query: str, result: "TreeGateResult",
                      graph) -> int:
        """Self-growing knowledge cycle: extract triples and save to graph.

        After TreeGate processes a query and assembles context, this method
        extracts entities and (subject, predicate, object) triples from the
        result text and saves them back into the knowledge graph. This
        implements the feedback loop where every answered query makes the
        tree smarter.

        Only runs when:
            - A graph reference is available and open
            - There are context_parts with actual content
            - The result has reasonable confidence (>= 0.3)

        Args:
            query: the original user question
            result: the TreeGateResult after full evaluation
            graph: QORGraph instance (can be None)

        Returns:
            Number of triples saved to graph.
        """
        if graph is None or not getattr(graph, 'is_open', False):
            return 0
        if not result.context_parts:
            return 0
        if result.max_confidence < 0.3:
            return 0

        saved = 0
        try:
            from .confidence import _extract_entities_and_edges
        except ImportError:
            return 0

        # Combine context parts for entity extraction (limit to avoid
        # excessive processing on very large contexts)
        combined = "\n".join(result.context_parts)[:4000]
        triples = _extract_entities_and_edges(combined)

        for subj, pred, obj in triples[:10]:  # Cap at 10 triples per query
            try:
                graph.add_edge(
                    subj, pred, obj,
                    confidence=min(result.max_confidence, 0.8),
                    source="treegate_self_grow",
                )
                saved += 1
            except Exception:
                continue

        if saved > 0:
            logger.debug("_save_to_tree: %d triples from query '%s'",
                         saved, query[:60])

        return saved

    def try_template_response(self, question: str,
                              context_parts: List[str],
                              ) -> Optional[str]:
        """
        Attempt a template response (no LLM needed).
        Returns formatted answer string, or None if template doesn't match.
        """
        q_lower = question.lower()
        context = "\n".join(context_parts)

        if any(k in q_lower for k in ["price", "how much", "worth"]):
            price = _extract_price_from_context(context)
            if price:
                # Try to extract the asset name
                for word in question.split():
                    if word.upper() == word and len(word) >= 2:
                        return f"{word} is currently at {price}."
                return f"The current price is {price}."

        if any(k in q_lower for k in ["what time", "current time"]):
            time_str = _extract_time_from_context(context)
            if time_str:
                return f"The current time is {time_str}."

        return None

    @staticmethod
    def post_verify(answer_text: str, context_parts: List[str],
                    ) -> Dict[str, Any]:
        """
        Post-verification: check LLM claims against context.

        PRD §9.5: "Check every claim against context package.
        Flag/remove anything not grounded in facts."

        Returns:
            {
                "verified": bool,
                "answer": str (original or cleaned),
                "flags": list of ungrounded claims (if any),
            }
        """
        if not answer_text or not context_parts:
            return {"verified": True, "answer": answer_text, "flags": []}

        context_text = " ".join(context_parts).lower()
        flags = []

        # Extract potential factual claims from answer
        # Heuristic: sentences containing numbers, dates, or proper nouns
        import re
        sentences = re.split(r'(?<=[.!?])\s+', answer_text)
        for sentence in sentences:
            # Check if sentence makes specific claims
            has_number = bool(re.search(r'\$?[\d,]+\.?\d*%?', sentence))
            if has_number:
                # Try to verify: at least part of the numeric claim
                # should appear in context
                numbers = re.findall(r'[\d,]+\.?\d*', sentence)
                found_any = False
                for num in numbers:
                    # Normalize: remove commas
                    num_clean = num.replace(",", "")
                    if num_clean in context_text.replace(",", ""):
                        found_any = True
                        break
                if not found_any and len(numbers) > 0:
                    flags.append(sentence.strip())

        # If too many ungrounded claims, flag the answer
        verified = len(flags) <= 1  # Allow 1 minor discrepancy
        return {
            "verified": verified,
            "answer": answer_text,
            "flags": flags,
        }

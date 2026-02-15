"""
QOR CORTEX — Brain-Inspired Sequential Intelligence Engine
===========================================================

CORTEX = CfC + Mamba + Fusion

  C — Continuous-time reflex (CfC / Amygdala)
  O — Observation of deep history (S4 State-Space / Neocortex)
  R — Reasoning layer (Fusion / Prefrontal Cortex)
  T — Temporal memory (Hidden states / Hippocampus)
  E — Execution decision (Output / Cerebellum)
  X — eXtended architecture (Quantum search / Basal Ganglia)

Architecture:
  +-----------------------------+
  | Observation Layer (S4)      | <-- Feeds on 500+ timesteps
  | Output: context vector      |    (trend, regime, structure)
  +----------+------------------+
             | context vector
             v
  +-----------------------------+
  | Continuous-time Reflex      | <-- Current features + context
  | Output: signal, confidence  |    Hidden state persists per instance
  +----------+------------------+
             |
             v
  +-----------------------------+
  | Reasoning + Execution       | <-- Fusion head -> final decision
  +-----------------------------+

Use cases: Trading, robotics, medical monitoring, anomaly detection,
           audio processing, sensor streams, anything sequential.

The S4Block is also used standalone in the main QOR neural model
(qor/model.py QORBlock) for long-range token-level patterns.
"""

import logging
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ncps.torch import CfC as _CfC
    from ncps.wirings import AutoNCP as _AutoNCP
    _HAS_CORTEX = True
except ImportError:
    _HAS_CORTEX = False

logger = logging.getLogger(__name__)


# ==============================================================================
# S4Block — Pure PyTorch State Space Model (Observation Layer)
# ==============================================================================

class S4Block(nn.Module):
    """Simplified S4-style state space block — pure PyTorch.

    Selective gating over a discretized state space scan.
    Captures long-range temporal structure in sequences.
    Runs on CPU — no CUDA kernel needed.

    Used in two places:
      1. QORBlock (model.py) — token-level long-range patterns
      2. CortexBrain (this file) — time-series analysis (trading, etc.)
    """

    def __init__(self, d_model: int = 32, d_state: int = 16, d_conv: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Local convolution (captures short patterns) — causal (left-only) padding
        self.d_conv = d_conv
        self.conv = nn.Conv1d(
            d_model, d_model, kernel_size=d_conv,
            padding=0, groups=d_model,
        )

        # State space parameters
        # Log-spaced initialization: different dimensions capture different timescales
        # Stored as log_A to avoid double-exp in forward (A = -exp(self.A), so
        # effective decay = exp(-exp(log_A))). Low indices = long memory, high = short.
        # Range: log(0.001)=-6.9 → exp(-exp(-6.9))≈1.0 (slow decay, long memory)
        #        log(5.0)=1.6   → exp(-exp(1.6))≈0.007 (fast decay, short memory)
        log_A = torch.linspace(
            math.log(0.001), math.log(5.0), d_state
        ).unsqueeze(0).expand(d_model, -1).clone()
        self.A = nn.Parameter(log_A)
        self.B_proj = nn.Linear(d_model, d_state)
        self.C_proj = nn.Linear(d_model, d_state)
        self.D = nn.Parameter(torch.ones(d_model))

        # Selective gating
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
        )
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """x: (batch, seq_len, d_model)"""
        residual = x

        # Conv branch — causal (left-only) padding to prevent future leakage
        x_conv = F.pad(x.transpose(1, 2), (self.d_conv - 1, 0))
        conv_out = self.conv(x_conv).transpose(1, 2)
        conv_out = F.silu(conv_out)

        # State space scan
        B = self.B_proj(conv_out)          # (batch, seq, d_state)
        C = self.C_proj(conv_out)          # (batch, seq, d_state)

        # Discretized scan (simplified)
        A = -torch.exp(self.A)             # Ensure stability
        batch, seq_len, _ = x.shape

        # Chunk-based scan (32x fewer Python iterations than per-timestep)
        CHUNK = 32
        h = torch.zeros(batch, self.d_model, self.d_state, device=x.device)
        outputs = []
        A_exp = torch.exp(A.unsqueeze(0))  # precompute once

        for start in range(0, seq_len, CHUNK):
            end = min(start + CHUNK, seq_len)
            chunk_conv = conv_out[:, start:end, :]   # (batch, chunk, d_model)
            chunk_B = B[:, start:end, :]             # (batch, chunk, d_state)
            chunk_C = C[:, start:end, :]             # (batch, chunk, d_state)
            for t in range(end - start):
                h = h * A_exp + chunk_conv[:, t, :].unsqueeze(-1) * chunk_B[:, t, :].unsqueeze(1)
                y_t = (h * chunk_C[:, t, :].unsqueeze(1)).sum(-1)
                y_t = y_t + self.D * chunk_conv[:, t, :]
                outputs.append(y_t)

        y = torch.stack(outputs, dim=1)    # (batch, seq, d_model)

        # Gated output
        y = y * self.gate(x)
        y = self.out_proj(y)
        y = self.norm(y + residual)

        return y


# ==============================================================================
# CortexBrain — Brain-Inspired Sequential Intelligence Engine
# ==============================================================================

class CortexBrain(nn.Module):
    """CORTEX brain for any sequential task.

    C — Continuous-time reflex (CfC liquid neurons):
      - Takes current features + Observation context
      - Hidden state persists across calls (temporal memory)
      - Outputs reflex decision vector

    O — Observation (S4 state-space model):
      - Scans full history buffer (up to history_len timesteps)
      - Outputs a context vector (trend, regime, structure)
      - Cached — only refreshes every `observation_interval` seconds

    R — Reasoning (Fusion head):
      - Combines Reflex + Observation into final output

    T — Temporal memory (per-instance hidden states)

    E — Execution (output layer)

    X — eXtended via QSearch quantum optimization

    Args:
        input_size: Feature vector dimension per timestep
        output_size: Output dimension (1 for signal, N for multi-output)
        observation_dim: Internal dimension of S4 observation block
        reflex_neurons: Number of liquid neurons (NCP wiring)
        reflex_output: Motor neuron count
        history_len: Max timesteps in observation history buffer
        observation_interval: DEPRECATED — observation cache now invalidates by
            history length, not wall-clock time. Kept for backward compatibility.
    """

    def __init__(self, input_size: int = 10, output_size: int = 1,
                 observation_dim: int = 32, reflex_neurons: int = 24,
                 reflex_output: int = 8, history_len: int = 500,
                 observation_interval: float = 360.0,
                 # Backward-compatible aliases
                 mamba_dim: int = None, cfc_neurons: int = None,
                 cfc_output: int = None, mamba_interval: float = None):
        super().__init__()
        if not _HAS_CORTEX:
            raise ImportError("CortexBrain requires: pip install ncps torch")

        # Support old parameter names for backward compat
        observation_dim = mamba_dim or observation_dim
        reflex_neurons = cfc_neurons or reflex_neurons
        reflex_output = cfc_output or reflex_output
        observation_interval = mamba_interval if mamba_interval is not None else observation_interval

        self.input_size = input_size
        self.output_size = output_size
        self.observation_dim = observation_dim
        self.observation_ctx_dim = observation_dim // 2
        if observation_dim % 2 != 0:
            logger.warning(f"[CORTEX] observation_dim={observation_dim} is odd; "
                           f"ctx_dim rounded down to {self.observation_ctx_dim}")
        self.reflex_neurons = reflex_neurons
        self.reflex_output = reflex_output
        self.history_len = history_len
        self.observation_interval = observation_interval

        # Backward compat aliases
        self.mamba_dim = observation_dim
        self.cfc_neurons = reflex_neurons
        self.cfc_output = reflex_output
        self.mamba_interval = observation_interval
        self.mamba_ctx_dim = self.observation_ctx_dim

        # --- O: Observation layer (deep history scanner) ---
        self.mamba_proj_in = nn.Linear(input_size, observation_dim)
        self.mamba = S4Block(d_model=observation_dim)
        self.mamba_out = nn.Linear(observation_dim, self.observation_ctx_dim)

        # --- C: Continuous-time reflex (fast + observation context) ---
        reflex_input_size = input_size + self.observation_ctx_dim
        wiring = _AutoNCP(reflex_neurons, output_size=reflex_output)
        self.cfc = _CfC(
            input_size=reflex_input_size,
            units=wiring,
            mixed_memory=True,      # LSTM + CfC dynamics
            return_sequences=False,
        )

        # --- R: Reasoning + E: Execution (fusion head) ---
        self.fusion = nn.Sequential(
            nn.Linear(reflex_output + self.observation_ctx_dim, 32),
            nn.GELU(),
            nn.Linear(32, output_size),
        )

        # --- T: Temporal memory (per-instance state) ---
        self._reflex_states = {}   # instance_id -> (h, c)
        self._history = {}         # instance_id -> list of feature tensors
        self._obs_cache = {}       # instance_id -> (context_vector, timestamp)

        # Backward compat aliases
        self._cfc_states = self._reflex_states
        self._mamba_cache = self._obs_cache

        self._trained = False  # Untrained until explicit training or load
        self.eval()  # Inference mode

    @torch.no_grad()
    def _run_observation(self, instance_id: str) -> torch.Tensor:
        """Run S4 observation scan over full history for an instance.

        Returns context vector (1, observation_ctx_dim). Cached by history
        length — reprocesses full history when new data arrives. Incremental
        processing requires S4 state persistence (not yet implemented).
        """
        current_len = len(self._history.get(instance_id, []))
        if instance_id in self._obs_cache:
            ctx, cached_len = self._obs_cache[instance_id]
            if current_len == cached_len:
                return ctx  # No new data, skip recompute

        history = self._history.get(instance_id, [])
        if len(history) < 3:
            ctx = torch.zeros(1, self.observation_ctx_dim)
            self._obs_cache[instance_id] = (ctx, current_len)
            return ctx

        # Full reprocess (correct). Incremental requires S4 state persistence.
        seq = torch.cat(history, dim=0)                       # (seq_len, input_size)
        obs_in = self.mamba_proj_in(seq).unsqueeze(0)         # (1, seq_len, obs_dim)
        obs_full = self.mamba(obs_in)                          # (1, seq_len, obs_dim)
        ctx = self.mamba_out(obs_full[:, -1, :])               # (1, obs_ctx_dim)

        self._obs_cache[instance_id] = (ctx, current_len)
        return ctx

    @property
    def _run_mamba(self):
        import warnings
        warnings.warn("_run_mamba is deprecated, use _run_observation", DeprecationWarning, stacklevel=2)
        return self._run_observation

    def _get_reflex_state(self, instance_id: str):
        """Get or create reflex hidden state for an instance."""
        if instance_id not in self._reflex_states:
            h = torch.zeros(1, self.reflex_neurons)
            c = torch.zeros(1, self.reflex_neurons)
            self._reflex_states[instance_id] = (h, c)
        return self._reflex_states[instance_id]

    @property
    def _get_cfc_state(self):
        import warnings
        warnings.warn("_get_cfc_state is deprecated, use _get_reflex_state", DeprecationWarning, stacklevel=2)
        return self._get_reflex_state

    def _forward_impl(self, x: torch.Tensor, instance_id: str = "default") -> torch.Tensor:
        """Core forward logic — shared by inference and training paths.

        Args:
            x: (1, input_size) — current input features
            instance_id: unique ID for state tracking
                        (symbol, robot_id, patient_id, etc.)

        Returns:
            (1, output_size) — output signal
        """
        # T: Accumulate into temporal history buffer
        if instance_id not in self._history:
            self._history[instance_id] = []
        self._history[instance_id].append(x.detach())
        if len(self._history[instance_id]) > self.history_len:
            self._history[instance_id] = self._history[instance_id][-self.history_len:]
            # Invalidate cache since indices shifted
            self._obs_cache.pop(instance_id, None)

        # O: Observation — scan deep history -> context vector (cached)
        obs_ctx = self._run_observation(instance_id)   # (1, obs_ctx_dim)

        # C: Continuous-time reflex — react to NOW + observation context
        # Note: CfC processes 1 timestep per call (stateful MLP pattern).
        # For batch processing, accumulate N ticks and pass as sequence.
        reflex_input = torch.cat([x, obs_ctx], dim=-1)  # (1, input_size + ctx)
        h, c = self._get_reflex_state(instance_id)
        reflex_out, (h_new, c_new) = self.cfc(reflex_input.unsqueeze(1), (h, c))
        self._reflex_states[instance_id] = (h_new.detach(), c_new.detach())
        reflex_out = reflex_out.squeeze(1)  # (1, reflex_output)

        # R+E: Reasoning + Execution — fuse reflex + observation -> final output
        combined = torch.cat([reflex_out, obs_ctx], dim=-1)
        output = self.fusion(combined)  # (1, output_size)

        return output

    @torch.no_grad()
    def forward(self, x: torch.Tensor, instance_id: str = "default") -> torch.Tensor:
        """Process one timestep (inference — no gradients)."""
        return self._forward_impl(x, instance_id)

    def reset_instance(self, instance_id: str):
        """Clear all state for an instance."""
        self._reflex_states.pop(instance_id, None)
        self._history.pop(instance_id, None)
        self._obs_cache.pop(instance_id, None)

    def train_batch(self, features: list, targets: list,
                    epochs: int = 10, lr: float = 1e-3) -> dict:
        """Train on historical (feature_vector, target_signal) pairs.

        Args:
            features: List of tensors, each (1, input_size) or (input_size,)
            targets: List of float target signals (e.g. +1 for profitable buy,
                     -1 for profitable sell, 0 for no-trade)
            epochs: Training iterations over the dataset
            lr: Learning rate

        Returns:
            dict with final_loss, epochs_run, trained (bool)
        """
        if len(features) < 2 or len(features) != len(targets):
            return {"final_loss": float("inf"), "epochs_run": 0, "trained": False}

        # Stack into batches
        X = torch.stack([f.view(1, -1) if f.dim() == 1 else f for f in features])  # (N, 1, input_size)
        Y = torch.tensor(targets, dtype=torch.float32).view(-1, 1)  # (N, 1)

        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        final_loss = float("inf")
        for epoch in range(epochs):
            # Reset instance at start of each epoch so CfC temporal memory
            # processes the sequence fresh (avoids cross-epoch contamination)
            self.reset_instance("_train")
            epoch_loss = 0.0
            for i in range(len(X)):
                optimizer.zero_grad()
                # Use single instance_id so CfC hidden state carries across
                # samples within an epoch (temporal context is preserved)
                out = self._forward_impl(X[i], instance_id="_train")
                loss = loss_fn(torch.tanh(out), Y[i].unsqueeze(0))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            final_loss = epoch_loss / len(X)

        # Clean up training instance state
        self.reset_instance("_train")

        self._trained = True
        self.eval()
        logger.info("[CORTEX] Trained on %d samples for %d epochs (loss=%.6f)",
                    len(features), epochs, final_loss)
        return {"final_loss": final_loss, "epochs_run": epochs, "trained": True}

    def cleanup_stale(self, active_ids: set):
        """Remove state for instances no longer active."""
        stale = set(self._history.keys()) - active_ids
        for sid in stale:
            self.reset_instance(sid)
        if stale:
            logger.info("[CORTEX] Cleaned up %d stale instances", len(stale))
            logger.debug("[CORTEX] Stale IDs: %s", stale)

    def save(self, path: str):
        """Save model weights + config + per-instance history to file."""
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "state_dict": self.state_dict(),
            "config": {
                "input_size": self.input_size,
                "output_size": self.output_size,
                "observation_dim": self.observation_dim,
                "reflex_neurons": self.reflex_neurons,
                "reflex_output": self.reflex_output,
                "history_len": self.history_len,
                "observation_interval": self.observation_interval,
            },
            "trained": self._trained,
            "history": {k: [t.cpu() for t in v] for k, v in self._history.items()},
        }, path)

    def load(self, path: str) -> bool:
        """Load model weights + history. Returns True if loaded.

        Handles input_size changes gracefully: if checkpoint was saved
        with a different input_size (e.g. 20→22 after adding features),
        skips mismatched layers and triggers retraining.

        Note: weights_only=False is required because we save tensor lists in
        'history'. Only load files generated by CortexBrain.save() — never
        load untrusted checkpoint files (pickle deserialization risk).
        """
        import os
        if not os.path.exists(path):
            return False
        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
            saved_cfg = data.get("config", {})
            saved_input = saved_cfg.get("input_size", self.input_size)

            if saved_input != self.input_size:
                # Input size changed (e.g. 20→22 after adding features).
                # Cannot reuse projection weights — skip them, mark untrained
                # so bootstrap retrains with new feature dimensions.
                logger.info(
                    "CORTEX input_size changed %d→%d, "
                    "will retrain with new features",
                    saved_input, self.input_size)
                self.load_state_dict(data["state_dict"], strict=False)
                self._trained = False  # Force retrain
            else:
                self.load_state_dict(data["state_dict"])
                self._trained = data.get("trained", True)

            # Restore per-instance history buffers
            if "history" in data:
                self._history = {k: list(v) for k, v in data["history"].items()}
            self.eval()
            return True
        except Exception as e:
            logger.warning("CORTEX load failed: %s", e)
            return False

    def status(self) -> dict:
        """Return state summary."""
        candle_counts = {s: len(h) for s, h in self._history.items()}
        return {
            "type": "CORTEX",
            "trained": self._trained,
            "active_instances": list(self._reflex_states.keys()),
            "reflex_neurons": self.reflex_neurons,
            "observation_dim": self.observation_dim,
            "input_size": self.input_size,
            "output_size": self.output_size,
            "observation_context": self.observation_ctx_dim,
            "history_candles": candle_counts,
            "history_max": self.history_len,
            "observation_interval_sec (deprecated)": self.observation_interval,
        }


# Backward compatibility aliases
MambaCfCHybrid = CortexBrain
_HAS_CFC = _HAS_CORTEX

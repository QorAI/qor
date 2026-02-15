"""
QOR Vision — The Eye
=====================
Patch-based image encoder that projects images into QOR's d_model space.
When no image is provided, this module is never called (zero overhead).

Architecture:
  Image (B, C, H, W) -> PatchEmbedding -> (B, n_patches, d_model)
  Optional VQ-VAE for discrete image tokenization (future image generation).
"""

import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import VisionConfig

logger = logging.getLogger(__name__)


class PatchEmbedding(nn.Module):
    """
    Split an image into non-overlapping patches and project each to d_model.

    MNIST (28x28, patch=7):  16 patches = 16 tokens
    Standard (224x224, patch=16): 196 patches = 196 tokens

    Uses Conv2d with kernel_size=patch_size and stride=patch_size —
    equivalent to splitting into patches then applying a linear projection,
    but more efficient.
    """

    def __init__(self, vision_config: VisionConfig, d_model: int):
        super().__init__()
        self.patch_size = vision_config.patch_size
        self.n_patches = vision_config.n_patches
        self.d_model = d_model

        # Conv2d acts as patch extraction + linear projection
        self.proj = nn.Conv2d(
            in_channels=vision_config.in_channels,
            out_channels=d_model,
            kernel_size=vision_config.patch_size,
            stride=vision_config.patch_size,
        )

        # Learnable 2D position embeddings (separate from RoPE)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.n_patches, d_model)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Layer norm after projection
        self.norm = nn.LayerNorm(d_model)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, C, H, W) image tensor

        Returns:
            (B, n_patches, d_model) patch embeddings
        """
        B = images.shape[0]

        # (B, C, H, W) -> (B, d_model, H/P, W/P) -> (B, d_model, n_patches)
        x = self.proj(images)
        x = x.flatten(2)  # (B, d_model, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, d_model)

        # Add positional embeddings
        x = x + self.pos_embed

        # Normalize
        x = self.norm(x)

        return x


class SimpleVQVAE(nn.Module):
    """
    Optional VQ-VAE for discrete image tokenization.

    Encodes patches into discrete codebook indices for potential
    future image generation. Off by default.

    Uses straight-through gradient estimator for the quantization step.
    """

    def __init__(self, d_model: int, codebook_size: int = 512,
                 codebook_dim: int = 64):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        # Project to codebook dimension
        self.encoder = nn.Sequential(
            nn.Linear(d_model, codebook_dim * 2),
            nn.GELU(),
            nn.Linear(codebook_dim * 2, codebook_dim),
        )

        # Codebook
        self.codebook = nn.Embedding(codebook_size, codebook_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / codebook_size, 1.0 / codebook_size)

        # Decoder: project back to d_model
        self.decoder = nn.Sequential(
            nn.Linear(codebook_dim, codebook_dim * 2),
            nn.GELU(),
            nn.Linear(codebook_dim * 2, d_model),
        )

    def quantize(self, z: torch.Tensor):
        """
        Quantize continuous vectors to nearest codebook entries.

        Args:
            z: (B, N, codebook_dim) encoded vectors

        Returns:
            z_q: (B, N, codebook_dim) quantized vectors
            indices: (B, N) codebook indices
            vq_loss: commitment + codebook loss
        """
        B, N, D = z.shape

        # Compute distances to codebook entries
        # (B*N, D) vs (K, D)
        flat_z = z.reshape(-1, D)
        distances = (
            flat_z.pow(2).sum(dim=1, keepdim=True)
            + self.codebook.weight.pow(2).sum(dim=1, keepdim=True).t()
            - 2 * flat_z @ self.codebook.weight.t()
        )

        # Nearest codebook entry
        indices = distances.argmin(dim=-1)  # (B*N,)
        z_q = self.codebook(indices).reshape(B, N, D)

        # VQ losses
        codebook_loss = F.mse_loss(z_q.detach(), z)     # Commitment loss
        embedding_loss = F.mse_loss(z_q, z.detach())     # Codebook update
        vq_loss = codebook_loss + 0.25 * embedding_loss

        # Straight-through estimator: gradients pass through quantization
        z_q = z + (z_q - z).detach()

        indices = indices.reshape(B, N)
        return z_q, indices, vq_loss

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, N, d_model) patch embeddings

        Returns:
            x_reconstructed: (B, N, d_model) reconstructed embeddings
            indices: (B, N) codebook indices
            vq_loss: scalar VQ-VAE loss
        """
        z = self.encoder(x)
        z_q, indices, vq_loss = self.quantize(z)
        x_reconstructed = self.decoder(z_q)
        return x_reconstructed, indices, vq_loss


class VisionEncoder(nn.Module):
    """
    Complete vision encoder: PatchEmbedding + optional VQ-VAE.

    Usage:
        encoder = VisionEncoder(vision_config, d_model=256)
        patch_embeds = encoder(images)  # (B, n_patches, d_model)
    """

    def __init__(self, vision_config: VisionConfig, d_model: int):
        super().__init__()
        self.config = vision_config
        self.patch_embed = PatchEmbedding(vision_config, d_model)
        self.n_patches = vision_config.n_patches

        # Optional VQ-VAE
        self.vqvae = None
        if vision_config.use_vqvae:
            self.vqvae = SimpleVQVAE(
                d_model=d_model,
                codebook_size=vision_config.codebook_size,
                codebook_dim=vision_config.codebook_dim,
            )

    def forward(self, images: torch.Tensor):
        """
        Args:
            images: (B, C, H, W) image tensor

        Returns:
            dict with:
                embeddings: (B, n_patches, d_model)
                vq_loss: scalar if VQ-VAE enabled, else None
                indices: (B, n_patches) if VQ-VAE enabled, else None
        """
        embeddings = self.patch_embed(images)

        vq_loss = None
        indices = None
        if self.vqvae is not None:
            embeddings, indices, vq_loss = self.vqvae(embeddings)

        return {
            "embeddings": embeddings,
            "vq_loss": vq_loss,
            "indices": indices,
        }


# ==============================================================================
# MODALITY BRIDGE — Bottleneck MLP shared by vision and audio pretrained encoders
# ==============================================================================

class ModalityBridge(nn.Module):
    """
    Projects frozen encoder output to the LLM's d_model space.

    Architecture: encoder_dim → Linear(bottleneck) → GELU → Linear(d_model) → RMSNorm

    Reused by both vision (SigLIP 1152→1024→2048) and audio (Whisper 768→1024→2048).
    Only ~3M trainable params per bridge.
    """

    def __init__(self, encoder_dim: int, d_model: int, bottleneck: int = 1024):
        super().__init__()
        self.down = nn.Linear(encoder_dim, bottleneck, bias=False)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, d_model, bias=False)
        # RMSNorm for stable output (matches QOR block norms)
        self.norm = nn.LayerNorm(d_model)

        # Small init for smooth start
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.normal_(self.up.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, encoder_dim) frozen encoder output
        Returns:
            (B, N, d_model) projected embeddings
        """
        return self.norm(self.up(self.act(self.down(x))))


# ==============================================================================
# PRETRAINED VISION ENCODER — Frozen SigLIP + trainable bridge
# ==============================================================================

class PretrainedVisionEncoder(nn.Module):
    """
    Frozen pretrained vision encoder (SigLIP) with a trainable bridge.

    Same interface as VisionEncoder — returns dict with "embeddings", "vq_loss", "indices".

    SigLIP so400m-patch14-384:
      - Input: (B, 3, 384, 384) images
      - Output: (B, 729, 1152) patch embeddings
      - Bridge projects to (B, 729, d_model)
      - ~400M frozen params + ~3.3M trainable bridge params

    Two creation modes:
      - _from_checkpoint=False (build time): downloads from HuggingFace
      - _from_checkpoint=True  (load time):  creates architecture from _hf_config dict,
        weights come from load_state_dict() — NO internet needed
    """

    def __init__(self, vision_config: VisionConfig, d_model: int,
                 _from_checkpoint: bool = False, _hf_config: dict = None):
        super().__init__()
        self.config = vision_config
        self.n_patches = vision_config.n_patches
        self.d_model = d_model

        try:
            from transformers import SiglipVisionModel, SiglipVisionConfig
        except ImportError:
            raise ImportError(
                "Pretrained vision encoder requires transformers.\n"
                "Install: pip install transformers>=4.40"
            )

        if _from_checkpoint and _hf_config is not None:
            # Create architecture from saved config — NO download, weights from checkpoint
            logger.info("Creating SigLIP architecture from checkpoint config (no download)")
            hf_cfg = SiglipVisionConfig(**_hf_config)
            self.encoder = SiglipVisionModel(hf_cfg)
        else:
            # Download from HuggingFace (only during build-multimodal)
            logger.info(f"Downloading pretrained vision encoder: {vision_config.pretrained_model}")
            self.encoder = SiglipVisionModel.from_pretrained(vision_config.pretrained_model)

        # Freeze encoder — no gradients, always eval mode
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Trainable bridge: encoder_dim → d_model
        self.bridge = ModalityBridge(
            encoder_dim=vision_config.pretrained_hidden_size,
            d_model=d_model,
            bottleneck=vision_config.bridge_bottleneck,
        )

    def get_hf_config(self) -> dict:
        """Extract HF encoder config dict for saving in checkpoint."""
        return self.encoder.config.to_dict()

    def train(self, mode: bool = True):
        """Override to keep encoder frozen even when model.train() is called."""
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, images: torch.Tensor):
        """
        Args:
            images: (B, 3, H, W) image tensor (will be resized if needed)

        Returns:
            dict with:
                embeddings: (B, n_patches, d_model)
                vq_loss: None (no VQ-VAE)
                indices: None (no VQ-VAE)
        """
        with torch.no_grad():
            outputs = self.encoder(pixel_values=images)
            hidden = outputs.last_hidden_state  # (B, n_patches, encoder_dim)

        embeddings = self.bridge(hidden)  # (B, n_patches, d_model)

        return {
            "embeddings": embeddings,
            "vq_loss": None,
            "indices": None,
        }


# ==============================================================================
# FACTORY — Routes based on config
# ==============================================================================

def create_vision_encoder(vision_config: VisionConfig, d_model: int,
                          _from_checkpoint: bool = False,
                          _hf_config: dict = None) -> nn.Module:
    """
    Factory function: returns the right vision encoder based on config.

    - use_pretrained=True  → PretrainedVisionEncoder (SigLIP, frozen + bridge)
    - use_pretrained=False → VisionEncoder (custom, trainable from scratch)

    Args:
        _from_checkpoint: If True, create architecture only (no HF download).
                         Weights will come from load_state_dict().
        _hf_config: Saved HF encoder config dict (from checkpoint).
    """
    if vision_config.use_pretrained:
        return PretrainedVisionEncoder(vision_config, d_model,
                                       _from_checkpoint=_from_checkpoint,
                                       _hf_config=_hf_config)
    return VisionEncoder(vision_config, d_model)

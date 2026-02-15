"""
QOR Audio — The Ear
====================
Mel-spectrogram encoder that projects audio into QOR's d_model space.
Inspired by Whisper's convolutional frontend.

When no audio is provided, this module is never called (zero overhead).

Architecture:
  Waveform -> Mel Spectrogram (B, 80, n_frames) -> Conv1d layers -> (B, n_tokens, d_model)
  Stride=4 temporal downsampling: 5 sec audio at 16kHz ~ 125 tokens
"""

import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AudioConfig

logger = logging.getLogger(__name__)


class SpectrogramEncoder(nn.Module):
    """
    Two Conv1d layers that downsample mel spectrograms into d_model vectors.

    Input:  (B, n_mels, n_frames) mel spectrogram
    Output: (B, n_tokens, d_model) audio token embeddings

    Inspired by Whisper's audio encoder frontend:
      Conv1d(n_mels, d_model, kernel=3, padding=1) -> GELU
      Conv1d(d_model, d_model, kernel=3, stride=frame_stride, padding=1) -> GELU
    """

    def __init__(self, audio_config: AudioConfig, d_model: int):
        super().__init__()
        self.max_audio_tokens = audio_config.max_audio_tokens
        self.frame_stride = audio_config.frame_stride

        # First conv: project mel bins to d_model
        self.conv1 = nn.Conv1d(
            in_channels=audio_config.n_mels,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
        )

        # Second conv: temporal downsampling
        self.conv2 = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            stride=audio_config.frame_stride,
            padding=1,
        )

        self.gelu = nn.GELU()

        # Learnable position embeddings for audio tokens
        self.pos_embed = nn.Parameter(
            torch.zeros(1, audio_config.max_audio_tokens, d_model)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Layer norm after projection
        self.norm = nn.LayerNorm(d_model)

    def forward(self, mel_specs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel_specs: (B, n_mels, n_frames) mel spectrogram

        Returns:
            (B, n_tokens, d_model) audio token embeddings
        """
        # (B, n_mels, n_frames) -> (B, d_model, n_frames)
        x = self.gelu(self.conv1(mel_specs))

        # (B, d_model, n_frames) -> (B, d_model, n_tokens) with stride downsampling
        x = self.gelu(self.conv2(x))

        # (B, d_model, n_tokens) -> (B, n_tokens, d_model)
        x = x.transpose(1, 2)

        # Truncate to max audio tokens
        if x.shape[1] > self.max_audio_tokens:
            x = x[:, :self.max_audio_tokens, :]

        # Add positional embeddings (slice to actual length)
        n_tokens = x.shape[1]
        x = x + self.pos_embed[:, :n_tokens, :]

        # Normalize
        x = self.norm(x)

        return x


class AudioEncoder(nn.Module):
    """
    Complete audio encoder: mel spectrogram computation + SpectrogramEncoder.

    Usage:
        encoder = AudioEncoder(audio_config, d_model=256)
        # If you have raw waveform:
        mel = AudioEncoder.compute_mel_spectrogram(waveform, audio_config)
        audio_embeds = encoder(mel)  # (B, n_tokens, d_model)
    """

    def __init__(self, audio_config: AudioConfig, d_model: int):
        super().__init__()
        self.config = audio_config
        self.encoder = SpectrogramEncoder(audio_config, d_model)

    def forward(self, mel_specs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel_specs: (B, n_mels, n_frames) mel spectrogram

        Returns:
            (B, n_tokens, d_model) audio token embeddings
        """
        return self.encoder(mel_specs)

    @staticmethod
    def compute_mel_spectrogram(waveform: torch.Tensor,
                                 audio_config: AudioConfig) -> torch.Tensor:
        """
        Convert raw waveform to mel spectrogram.

        Args:
            waveform: (B, n_samples) or (n_samples,) raw audio
            audio_config: AudioConfig with spectrogram parameters

        Returns:
            (B, n_mels, n_frames) mel spectrogram
        """
        try:
            import torchaudio
            import torchaudio.transforms as T
        except ImportError:
            raise ImportError(
                "torchaudio is required for audio processing.\n"
                "Install it: pip install torchaudio"
            )

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        mel_transform = T.MelSpectrogram(
            sample_rate=audio_config.sample_rate,
            n_fft=audio_config.n_fft,
            hop_length=audio_config.hop_length,
            n_mels=audio_config.n_mels,
        ).to(waveform.device)

        mel = mel_transform(waveform)

        # Log mel spectrogram (add small epsilon for numerical stability)
        mel = torch.log(mel.clamp(min=1e-10))

        return mel

    @staticmethod
    def compute_n_tokens(n_samples: int, audio_config: AudioConfig) -> int:
        """Compute number of audio tokens for a given number of samples."""
        n_frames = n_samples // audio_config.hop_length
        n_tokens = math.ceil(n_frames / audio_config.frame_stride)
        return min(n_tokens, audio_config.max_audio_tokens)


# ==============================================================================
# PRETRAINED AUDIO ENCODER — Frozen Whisper encoder + trainable bridge
# ==============================================================================

class PretrainedAudioEncoder(nn.Module):
    """
    Frozen pretrained audio encoder (Whisper) with a trainable bridge.

    Same interface as AudioEncoder — forward takes mel_specs, returns (B, n_tokens, d_model).

    Whisper small encoder:
      - Input: (B, 80, 3000) mel spectrogram (30s window)
      - Output: (B, 1500, 768) encoder hidden states
      - Bridge projects to (B, N, d_model), truncated to max_audio_tokens
      - ~122M frozen params + ~2.9M trainable bridge params

    Two creation modes:
      - _from_checkpoint=False (build time): downloads from HuggingFace
      - _from_checkpoint=True  (load time):  creates architecture from _hf_config dict,
        weights come from load_state_dict() — NO internet needed
    """

    def __init__(self, audio_config: AudioConfig, d_model: int,
                 _from_checkpoint: bool = False, _hf_config: dict = None):
        super().__init__()
        self.config = audio_config
        self.max_audio_tokens = audio_config.max_audio_tokens

        try:
            from transformers import WhisperModel, WhisperConfig
        except ImportError:
            raise ImportError(
                "Pretrained audio encoder requires transformers.\n"
                "Install: pip install transformers>=4.40"
            )

        # Import bridge from vision module (shared implementation)
        from .vision import ModalityBridge

        if _from_checkpoint and _hf_config is not None:
            # Create architecture from saved config — NO download, weights from checkpoint
            logger.info("Creating Whisper architecture from checkpoint config (no download)")
            hf_cfg = WhisperConfig(**_hf_config)
            whisper = WhisperModel(hf_cfg)
            self.encoder = whisper.encoder
        else:
            # Download from HuggingFace (only during build-multimodal)
            logger.info(f"Downloading pretrained audio encoder: {audio_config.pretrained_model}")
            whisper = WhisperModel.from_pretrained(audio_config.pretrained_model)
            self.encoder = whisper.encoder

        # Freeze encoder — no gradients, always eval mode
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Trainable bridge: encoder_dim → d_model
        self.bridge = ModalityBridge(
            encoder_dim=audio_config.pretrained_hidden_size,
            d_model=d_model,
            bottleneck=audio_config.bridge_bottleneck,
        )

    def get_hf_config(self) -> dict:
        """Extract HF encoder config dict for saving in checkpoint."""
        return self.encoder.config.to_dict()

    def train(self, mode: bool = True):
        """Override to keep encoder frozen even when model.train() is called."""
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, mel_specs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel_specs: (B, n_mels, n_frames) mel spectrogram

        Returns:
            (B, n_tokens, d_model) audio token embeddings
        """
        # Whisper expects (B, 80, 3000) — pad or truncate to 3000 frames
        B, n_mels, n_frames = mel_specs.shape
        target_frames = 3000  # Whisper's 30-second window
        if n_frames < target_frames:
            mel_specs = F.pad(mel_specs, (0, target_frames - n_frames))
        elif n_frames > target_frames:
            mel_specs = mel_specs[:, :, :target_frames]

        with torch.no_grad():
            outputs = self.encoder(mel_specs)
            hidden = outputs.last_hidden_state  # (B, ~1500, encoder_dim)

        embeddings = self.bridge(hidden)  # (B, ~1500, d_model)

        # Truncate to max_audio_tokens
        if embeddings.shape[1] > self.max_audio_tokens:
            embeddings = embeddings[:, :self.max_audio_tokens, :]

        return embeddings

    @staticmethod
    def compute_mel_spectrogram(waveform, audio_config: AudioConfig):
        """Compute mel spectrogram using Whisper's feature extractor."""
        try:
            from transformers import WhisperFeatureExtractor
        except ImportError:
            raise ImportError(
                "Whisper mel computation requires transformers.\n"
                "Install: pip install transformers>=4.40"
            )
        import torch

        extractor = WhisperFeatureExtractor(
            feature_size=audio_config.n_mels,
            sampling_rate=audio_config.sample_rate,
        )

        if isinstance(waveform, torch.Tensor):
            waveform = waveform.numpy()
        if waveform.ndim > 1:
            waveform = waveform[0]  # Take first channel

        features = extractor(waveform, sampling_rate=audio_config.sample_rate,
                             return_tensors="pt")
        return features.input_features  # (1, 80, 3000)


# ==============================================================================
# FACTORY — Routes based on config
# ==============================================================================

def create_audio_encoder(audio_config: AudioConfig, d_model: int,
                         _from_checkpoint: bool = False,
                         _hf_config: dict = None) -> nn.Module:
    """
    Factory function: returns the right audio encoder based on config.

    - use_pretrained=True  → PretrainedAudioEncoder (Whisper, frozen + bridge)
    - use_pretrained=False → AudioEncoder (custom, trainable from scratch)

    Args:
        _from_checkpoint: If True, create architecture only (no HF download).
                         Weights will come from load_state_dict().
        _hf_config: Saved HF encoder config dict (from checkpoint).
    """
    if audio_config.use_pretrained:
        return PretrainedAudioEncoder(audio_config, d_model,
                                      _from_checkpoint=_from_checkpoint,
                                      _hf_config=_hf_config)
    return AudioEncoder(audio_config, d_model)

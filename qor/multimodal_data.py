"""
QOR Multimodal Data Pipeline
==============================
Datasets and dataloaders for image-text and audio-text training.

Supports:
  - JSONL format: {"image": "path.jpg", "text": "A cat"}
  - MNIST-style folder: data/images/3/img_001.png -> "This is digit 3"
  - Audio-text pairs: {"audio": "clip.wav", "text": "Hello world"}
  - Paired .wav/.txt files in a directory
"""

import os
import json
import math
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Dict

from .config import VisionConfig, AudioConfig
from .tokenizer import QORTokenizer


class ImageTextDataset(Dataset):
    """
    Dataset for image-text pairs.

    Supports two formats:
      1. JSONL: each line is {"image": "path/to/img.png", "text": "description"}
      2. MNIST-style folders: data/<label>/img_001.png
         Auto-generates text: "This is digit <label>"

    Token sequence:
      <BOS> <|image|> [<|image_patch|> x n_patches] <|/image|> text <EOS>
    """

    def __init__(self, data_dir: str, tokenizer: QORTokenizer,
                 vision_config: VisionConfig, max_seq_len: int = 512,
                 transform=None):
        self.tokenizer = tokenizer
        self.vision_config = vision_config
        self.max_seq_len = max_seq_len
        self.n_patches = vision_config.n_patches
        self.samples = []  # List of (image_path, text)

        # Try to import image loading
        try:
            from PIL import Image
            self._Image = Image
        except ImportError:
            raise ImportError("Pillow is required: pip install Pillow")

        # Set up image transform
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._default_transform(vision_config)

        # Load samples
        self._load_samples(data_dir)

    def _default_transform(self, vision_config: VisionConfig):
        """Create default image transform."""
        try:
            from torchvision import transforms
            t_list = [
                transforms.Resize((vision_config.image_size, vision_config.image_size)),
                transforms.ToTensor(),
            ]
            if vision_config.in_channels == 1:
                t_list.insert(0, transforms.Grayscale(num_output_channels=1))
                t_list.append(transforms.Normalize([0.5], [0.5]))
            else:
                t_list.append(transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
            return transforms.Compose(t_list)
        except ImportError:
            return None

    def _load_samples(self, data_dir: str):
        """Auto-detect and load image-text samples."""
        # Check for JSONL file
        jsonl_path = os.path.join(data_dir, "data.jsonl")
        if os.path.exists(jsonl_path):
            self._load_jsonl(jsonl_path, data_dir)
            return

        # Check for any .jsonl files
        for f in os.listdir(data_dir):
            if f.endswith('.jsonl'):
                self._load_jsonl(os.path.join(data_dir, f), data_dir)
                if self.samples:
                    return

        # MNIST-style folder structure: data_dir/<label>/img_*.png
        self._load_folder_structure(data_dir)

    def _load_jsonl(self, jsonl_path: str, base_dir: str):
        """Load from JSONL format."""
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                img_path = entry.get("image", "")
                text = entry.get("text", "")
                if not os.path.isabs(img_path):
                    img_path = os.path.join(base_dir, img_path)
                if os.path.exists(img_path) and text:
                    self.samples.append((img_path, text))

        print(f"  Loaded {len(self.samples)} image-text pairs from JSONL")

    def _load_folder_structure(self, data_dir: str):
        """Load MNIST-style folder structure."""
        image_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}

        for label_name in sorted(os.listdir(data_dir)):
            label_dir = os.path.join(data_dir, label_name)
            if not os.path.isdir(label_dir):
                continue

            for img_name in sorted(os.listdir(label_dir)):
                ext = os.path.splitext(img_name)[1].lower()
                if ext in image_exts:
                    img_path = os.path.join(label_dir, img_name)
                    text = f"This is digit {label_name}"
                    self.samples.append((img_path, text))

        if self.samples:
            print(f"  Loaded {len(self.samples)} images from folder structure")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, text = self.samples[idx]

        # Load and transform image
        image = self._Image.open(img_path).convert(
            'L' if self.vision_config.in_channels == 1 else 'RGB'
        )
        if self.transform:
            image = self.transform(image)
        else:
            # Fallback: manual tensor conversion
            import numpy as np
            image = torch.from_numpy(np.array(image)).float() / 255.0
            if image.dim() == 2:
                image = image.unsqueeze(0)

        # Build token sequence with image placeholders
        token_ids = self.tokenizer.encode_with_image(
            text_before="",
            n_patches=self.n_patches,
            text_after=text,
        )

        # Truncate to max_seq_len
        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[:self.max_seq_len - 1] + [self.tokenizer.eos_id]

        # Pad to max_seq_len
        pad_len = self.max_seq_len - len(token_ids)
        token_ids = token_ids + [self.tokenizer.pad_id] * pad_len

        input_ids = torch.tensor(token_ids, dtype=torch.long)
        targets = input_ids.clone()

        # Find image patch positions (where <|image_patch|> tokens are)
        patch_id = self.tokenizer.image_patch_id
        if patch_id is not None:
            image_positions = (input_ids == patch_id).nonzero(as_tuple=True)[0]
        else:
            image_positions = torch.zeros(self.n_patches, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "targets": targets,
            "image": image,
            "image_positions": image_positions,
        }


class AudioTextDataset(Dataset):
    """
    Dataset for audio-text pairs.

    Supports two formats:
      1. JSONL: each line is {"audio": "path/to/clip.wav", "text": "transcription"}
      2. Paired files: audio_001.wav + audio_001.txt in same directory

    Token sequence:
      <BOS> <|audio|> [<|audio_frame|> x n_tokens] <|/audio|> text <EOS>
    """

    def __init__(self, data_dir: str, tokenizer: QORTokenizer,
                 audio_config: AudioConfig, max_seq_len: int = 512):
        self.tokenizer = tokenizer
        self.audio_config = audio_config
        self.max_seq_len = max_seq_len
        self.samples = []  # List of (audio_path, text)

        self._load_samples(data_dir)

    def _load_samples(self, data_dir: str):
        """Auto-detect and load audio-text samples."""
        # Check for JSONL
        for f in os.listdir(data_dir):
            if f.endswith('.jsonl'):
                self._load_jsonl(os.path.join(data_dir, f), data_dir)
                if self.samples:
                    return

        # Paired files: *.wav + *.txt
        self._load_paired_files(data_dir)

    def _load_jsonl(self, jsonl_path: str, base_dir: str):
        """Load from JSONL format."""
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                audio_path = entry.get("audio", "")
                text = entry.get("text", "")
                if not os.path.isabs(audio_path):
                    audio_path = os.path.join(base_dir, audio_path)
                if os.path.exists(audio_path) and text:
                    self.samples.append((audio_path, text))

        print(f"  Loaded {len(self.samples)} audio-text pairs from JSONL")

    def _load_paired_files(self, data_dir: str):
        """Load paired .wav/.txt files."""
        audio_exts = {'.wav', '.mp3', '.flac', '.ogg'}

        for fname in sorted(os.listdir(data_dir)):
            name, ext = os.path.splitext(fname)
            if ext.lower() in audio_exts:
                txt_path = os.path.join(data_dir, name + '.txt')
                if os.path.exists(txt_path):
                    audio_path = os.path.join(data_dir, fname)
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                    if text:
                        self.samples.append((audio_path, text))

        if self.samples:
            print(f"  Loaded {len(self.samples)} audio-text pairs from paired files")

    def _load_audio(self, audio_path: str) -> torch.Tensor:
        """Load audio and compute mel spectrogram."""
        try:
            import torchaudio
        except ImportError:
            raise ImportError("torchaudio is required: pip install torchaudio")

        waveform, sample_rate = torchaudio.load(audio_path)

        # Resample if needed
        if sample_rate != self.audio_config.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self.audio_config.sample_rate,
            )
            waveform = resampler(waveform)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        waveform = waveform.squeeze(0)  # (n_samples,)

        from .audio import AudioEncoder
        mel = AudioEncoder.compute_mel_spectrogram(waveform, self.audio_config)
        return mel.squeeze(0)  # (n_mels, n_frames)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio_path, text = self.samples[idx]

        # Load audio mel spectrogram
        mel = self._load_audio(audio_path)  # (n_mels, n_frames)

        # Compute n_audio_tokens
        from .audio import AudioEncoder
        n_samples = mel.shape[1] * self.audio_config.hop_length
        n_audio_tokens = AudioEncoder.compute_n_tokens(n_samples, self.audio_config)
        n_audio_tokens = max(1, n_audio_tokens)

        # Build token sequence with audio placeholders
        token_ids = self.tokenizer.encode_with_audio(
            text_before="",
            n_audio_tokens=n_audio_tokens,
            text_after=text,
        )

        # Truncate
        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[:self.max_seq_len - 1] + [self.tokenizer.eos_id]

        # Pad
        pad_len = self.max_seq_len - len(token_ids)
        token_ids = token_ids + [self.tokenizer.pad_id] * pad_len

        input_ids = torch.tensor(token_ids, dtype=torch.long)
        targets = input_ids.clone()

        # Find audio frame positions
        frame_id = self.tokenizer.audio_frame_id
        if frame_id is not None:
            audio_positions = (input_ids == frame_id).nonzero(as_tuple=True)[0]
        else:
            audio_positions = torch.zeros(n_audio_tokens, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "targets": targets,
            "mel_spec": mel,
            "audio_positions": audio_positions,
        }


def multimodal_collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for multimodal batches.
    Handles variable presence of images/audio and pads positions.
    """
    input_ids = torch.stack([b["input_ids"] for b in batch])
    targets = torch.stack([b["targets"] for b in batch])

    result = {
        "input_ids": input_ids,
        "targets": targets,
    }

    # Collate images
    if "image" in batch[0]:
        result["images"] = torch.stack([b["image"] for b in batch])

        # Pad image_positions to same length
        max_pos = max(b["image_positions"].shape[0] for b in batch)
        padded_positions = []
        for b in batch:
            pos = b["image_positions"]
            if pos.shape[0] < max_pos:
                pad = torch.zeros(max_pos - pos.shape[0], dtype=torch.long)
                pos = torch.cat([pos, pad])
            padded_positions.append(pos)
        result["image_positions"] = torch.stack(padded_positions)

    # Collate audio
    if "mel_spec" in batch[0]:
        # Pad mel spectrograms to same length
        max_frames = max(b["mel_spec"].shape[1] for b in batch)
        padded_mels = []
        for b in batch:
            mel = b["mel_spec"]
            if mel.shape[1] < max_frames:
                pad = torch.zeros(mel.shape[0], max_frames - mel.shape[1])
                mel = torch.cat([mel, pad], dim=1)
            padded_mels.append(mel)
        result["mel_specs"] = torch.stack(padded_mels)

        # Pad audio_positions to same length
        max_pos = max(b["audio_positions"].shape[0] for b in batch)
        padded_positions = []
        for b in batch:
            pos = b["audio_positions"]
            if pos.shape[0] < max_pos:
                pad = torch.zeros(max_pos - pos.shape[0], dtype=torch.long)
                pos = torch.cat([pos, pad])
            padded_positions.append(pos)
        result["audio_positions"] = torch.stack(padded_positions)

    return result


def create_multimodal_dataloaders(
    modality: str,
    data_dir: str,
    tokenizer: QORTokenizer,
    vision_config: Optional[VisionConfig] = None,
    audio_config: Optional[AudioConfig] = None,
    max_seq_len: int = 512,
    batch_size: int = 8,
    val_split: float = 0.05,
    num_workers: int = 0,
    transform=None,
):
    """
    Factory for multimodal train/val dataloaders.

    Args:
        modality: "vision" or "audio"
        data_dir: path to data directory
        tokenizer: QORTokenizer instance
        vision_config: VisionConfig (required if modality="vision")
        audio_config: AudioConfig (required if modality="audio")
        max_seq_len: maximum sequence length
        batch_size: batch size
        val_split: validation split ratio
        num_workers: data loading workers
        transform: optional image transform (vision only)

    Returns:
        (train_loader, val_loader)
    """
    if modality == "vision":
        assert vision_config is not None, "vision_config required for vision modality"
        dataset = ImageTextDataset(
            data_dir, tokenizer, vision_config, max_seq_len, transform
        )
    elif modality == "audio":
        assert audio_config is not None, "audio_config required for audio modality"
        dataset = AudioTextDataset(
            data_dir, tokenizer, audio_config, max_seq_len
        )
    else:
        raise ValueError(f"Unknown modality: {modality}. Use 'vision' or 'audio'")

    # Split into train/val
    n_val = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=multimodal_collate_fn,
        drop_last=True,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=multimodal_collate_fn,
        pin_memory=True,
    )

    print(f"  Dataset: {len(dataset)} samples ({n_train} train, {n_val} val)")
    print(f"  Batch size: {batch_size}")

    return train_loader, val_loader

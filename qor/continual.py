"""
QOR Continual Learning — Teach the Mind New Things
=====================================================
Drop new text files into the 'learn' folder.
QOR will read them and learn without forgetting old knowledge.

This is the whole point of the architecture:
  - Fast memory absorbs new information
  - Slow memory is protected (no catastrophic forgetting)
  - Surprise gating means it only updates on genuinely new info
"""

import os
import glob
import time
import json
import torch
import torch.nn as nn
from datetime import datetime
from typing import Optional

from .config import QORConfig
from .model import QORModel
from .tokenizer import QORTokenizer
from .data import SingleFileDataset


class ContinualLearner:
    """
    Watches a folder for new text files and teaches them to the model.
    Like feeding a mind new books to read.
    """

    def __init__(self, config: QORConfig):
        self.config = config
        self.device = config.get_device()
        self.model = None
        self.tokenizer = None
        self.learned_files = set()
        self.history = []

    def load(self, checkpoint_path: str):
        """Load the trained model to continue learning from."""
        print(f"Loading model from {checkpoint_path}")

        # Tokenizer — check checkpoint dir for donor tokenizer first
        self.tokenizer = QORTokenizer()
        ckpt_dir = os.path.dirname(checkpoint_path)
        donor_tok = os.path.join(ckpt_dir, "tokenizer.json")
        if self.config.tokenizer.type == "pretrained" and os.path.exists(donor_tok):
            self.tokenizer.load(donor_tok)
        elif self.config.tokenizer.type == "pretrained":
            self.tokenizer.load_pretrained(self.config.tokenizer.pretrained_name)
        else:
            self.tokenizer.load(self.config.tokenizer.save_path)
        self.config.model.vocab_size = self.tokenizer.vocab_size

        # Model — use mmap to avoid double memory
        import gc
        checkpoint = torch.load(checkpoint_path, map_location=self.device,
                                weights_only=False, mmap=True)

        # Use config from checkpoint if available (handles custom model sizes)
        if "config" in checkpoint and "model" in checkpoint["config"]:
            from .config import ModelConfig
            saved = checkpoint["config"]["model"]
            model_cfg = ModelConfig()
            for k, v in saved.items():
                if hasattr(model_cfg, k):
                    setattr(model_cfg, k, v)
            model_cfg.vocab_size = self.config.model.vocab_size
            self.config.model = model_cfg

        # Reconstruct vision/audio configs from checkpoint (for pretrained encoders)
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

        # _from_checkpoint=True: creates encoder architecture without downloading
        # from HuggingFace — all weights come from checkpoint via load_state_dict()
        self.model = QORModel(
            self.config.model,
            vision_config=vision_config,
            audio_config=audio_config,
            _from_checkpoint=True,
            _encoder_hf_configs=encoder_hf_configs,
        )
        # strict=False: allows loading old checkpoints into models with new
        # layers (e.g. CfC liquid neurons) — missing keys init as passthrough
        self.model.load_state_dict(checkpoint["model_state"],
                                   assign=True, strict=False)
        # Move non-persistent buffers (RoPE cos/sin) to match weight device
        for block in self.model.blocks:
            attn = block.attention
            if hasattr(attn, 'rope_cos') and attn.rope_cos.device != self.device:
                attn.rope_cos = attn.rope_cos.to(self.device)
                attn.rope_sin = attn.rope_sin.to(self.device)
        del checkpoint
        gc.collect()

        # Add modality tokens for multimodal models (no-op if already present)
        if vision_config is not None or audio_config is not None:
            num_added = self.tokenizer.ensure_modality_tokens()
            if num_added > 0 and self.tokenizer.vocab_size > self.model.config.vocab_size:
                self.model.resize_token_embeddings(self.tokenizer.vocab_size)

        print(f"  Parameters: {self.model.n_params:,}")
        print(f"  Ready for continual learning")

        # Load history of previously learned files
        history_path = os.path.join(self.config.train.checkpoint_dir, "learn_history.json")
        if os.path.exists(history_path):
            with open(history_path) as f:
                self.history = json.load(f)
                self.learned_files = {h["file"] for h in self.history}
            print(f"  Previously learned: {len(self.learned_files)} files")

    def learn_file(self, file_path: str) -> dict:
        """Teach the model a single text file."""
        cfg = self.config.continual
        filename = os.path.basename(file_path)
        print(f"\n  Learning: {filename}")

        # Load data
        dataset = SingleFileDataset(self.tokenizer, file_path,
                                     self.config.model.max_seq_len)
        print(f"    Tokens: {len(dataset.data):,}")

        # Protect slow CMS layers during learning
        if cfg.protect_slow_layers:
            for block in self.model.blocks:
                block.cms.freeze_slow_layers()

        # Optimizer — lower learning rate than initial training
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=cfg.learn_rate,
            weight_decay=0.01,
        )

        # Learning loop
        self.model.train()
        total_loss = 0.0
        n_steps = min(cfg.learn_steps_per_file, len(dataset))
        start_time = time.time()

        for step in range(n_steps):
            self.model.step_cms()
            if step % 20 == 0:
                self.model.reset_fast_weights()

            x, y = dataset[step]
            x = x.unsqueeze(0).to(self.device)
            y = y.unsqueeze(0).to(self.device)

            result = self.model(x, targets=y, enable_self_mod=True)
            loss = result["loss"]

            # Surprise gating — only update if content is surprising enough
            if cfg.surprise_gate and result["surprise"] < cfg.surprise_threshold:
                continue  # Skip boring content

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

            if step % 50 == 0:
                print(f"    Step {step}/{n_steps}: loss={loss.item():.4f} surprise={result['surprise']:.4f}")

        elapsed = time.time() - start_time
        avg_loss = total_loss / max(n_steps, 1)

        # Unfreeze all layers
        if cfg.protect_slow_layers:
            for block in self.model.blocks:
                block.cms.unfreeze_all()

        # Record
        record = {
            "file": filename,
            "path": file_path,
            "tokens": len(dataset.data),
            "steps": n_steps,
            "avg_loss": round(avg_loss, 4),
            "time_seconds": round(elapsed, 1),
            "timestamp": datetime.now().isoformat(),
        }
        self.history.append(record)
        self.learned_files.add(filename)

        print(f"    Done: avg_loss={avg_loss:.4f}, time={elapsed:.1f}s")
        return record

    def learn_folder(self, folder: Optional[str] = None):
        """Learn from all new text files in a folder."""
        folder = folder or self.config.continual.learn_dir

        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
            print(f"\n  Created '{folder}/' — drop .txt files here for QOR to learn!")
            print(f"  Then run this command again.")
            return

        # Find new text files
        files = sorted(glob.glob(os.path.join(folder, '**', '*.txt'), recursive=True))
        new_files = [f for f in files if os.path.basename(f) not in self.learned_files]

        # Find new image files (with paired .txt descriptions)
        image_exts = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
        new_images = []
        for ext in image_exts:
            for img_path in sorted(glob.glob(os.path.join(folder, '**', ext), recursive=True)):
                if os.path.basename(img_path) in self.learned_files:
                    continue
                txt_path = os.path.splitext(img_path)[0] + '.txt'
                if os.path.exists(txt_path):
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        desc = f.read().strip()
                    if desc:
                        new_images.append((img_path, desc))

        # Find new audio files (with paired .txt transcriptions)
        audio_exts = ['*.wav', '*.mp3', '*.flac']
        new_audios = []
        for ext in audio_exts:
            for audio_path in sorted(glob.glob(os.path.join(folder, '**', ext), recursive=True)):
                if os.path.basename(audio_path) in self.learned_files:
                    continue
                txt_path = os.path.splitext(audio_path)[0] + '.txt'
                if os.path.exists(txt_path):
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        desc = f.read().strip()
                    if desc:
                        new_audios.append((audio_path, desc))

        total_new = len(new_files) + len(new_images) + len(new_audios)
        if total_new == 0:
            print(f"\n  No new files in '{folder}/' — already learned {len(self.learned_files)} files")
            print(f"  Drop .txt, .png+.txt, or .wav+.txt files in '{folder}/' and run again!")
            return

        print(f"\n{'='*60}")
        print(f"  QOR Continual Learning")
        print(f"  New text files: {len(new_files)}")
        print(f"  New images:     {len(new_images)}")
        print(f"  New audio:      {len(new_audios)}")
        print(f"  Previously learned: {len(self.learned_files)}")
        print(f"{'='*60}")

        for fpath in new_files:
            self.learn_file(fpath)

        for img_path, desc in new_images:
            self.learn_image(img_path, desc)

        for audio_path, desc in new_audios:
            self.learn_audio(audio_path, desc)

        # Save updated model
        self._save()

        # Save history
        history_path = os.path.join(self.config.train.checkpoint_dir, "learn_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        print(f"\n{'='*60}")
        print(f"  Learning Complete!")
        print(f"  Total files learned: {len(self.learned_files)}")
        print(f"  Model saved with new knowledge")
        print(f"{'='*60}\n")

    def watch(self, folder: Optional[str] = None, interval: int = 10):
        """
        Continuously watch a folder for new files and learn them.
        Like a mind that's always ready to read.

        Press Ctrl+C to stop.
        """
        folder = folder or self.config.continual.learn_dir
        os.makedirs(folder, exist_ok=True)

        print(f"\n  QOR is watching '{folder}/' for new knowledge...")
        print(f"  Drop .txt files there and QOR will learn them automatically.")
        print(f"  Press Ctrl+C to stop.\n")

        try:
            while True:
                files = sorted(glob.glob(os.path.join(folder, '**', '*.txt'), recursive=True))
                new_files = [f for f in files if os.path.basename(f) not in self.learned_files]

                for fpath in new_files:
                    self.learn_file(fpath)
                    self._save()

                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n  Stopped watching. Model saved.")

    def learn_image(self, image_path: str, description: str) -> dict:
        """
        Teach the model an image-description pair.

        Args:
            image_path: path to image file
            description: text description of the image
        """
        cfg = self.config.continual
        filename = os.path.basename(image_path)
        print(f"\n  Learning image: {filename}")

        try:
            from PIL import Image
            from torchvision import transforms
        except ImportError:
            print("  Requires: pip install Pillow torchvision")
            return {}

        if self.model.vision_encoder is None:
            print("  Model has no vision encoder. Load a multimodal model.")
            return {}

        vision_cfg = self.config.vision

        # Load image
        t_list = [
            transforms.Resize((vision_cfg.image_size, vision_cfg.image_size)),
            transforms.ToTensor(),
        ]
        if vision_cfg.in_channels == 1:
            t_list.insert(0, transforms.Grayscale(num_output_channels=1))
            t_list.append(transforms.Normalize([0.5], [0.5]))
        else:
            t_list.append(transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
        transform = transforms.Compose(t_list)

        mode = 'L' if vision_cfg.in_channels == 1 else 'RGB'
        image = transform(Image.open(image_path).convert(mode)).unsqueeze(0).to(self.device)

        n_patches = vision_cfg.n_patches
        token_ids = self.tokenizer.encode_with_image("", n_patches, description)
        if len(token_ids) > self.config.model.max_seq_len:
            token_ids = token_ids[:self.config.model.max_seq_len - 1] + [self.tokenizer.eos_id]

        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        targets = input_ids.clone()

        patch_id = self.tokenizer.image_patch_id
        image_positions = (input_ids[0] == patch_id).nonzero(as_tuple=True)[0].unsqueeze(0)

        # Protect slow layers
        if cfg.protect_slow_layers:
            for block in self.model.blocks:
                block.cms.freeze_slow_layers()

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=cfg.learn_rate, weight_decay=0.01,
        )

        self.model.train()
        total_loss = 0.0
        n_steps = cfg.learn_steps_per_file
        start_time = time.time()

        for step in range(n_steps):
            self.model.step_cms()
            if step % 20 == 0:
                self.model.reset_fast_weights()

            result = self.model(input_ids, targets=targets, enable_self_mod=True,
                                images=image, image_positions=image_positions)
            loss = result["loss"]

            if cfg.surprise_gate and result["surprise"] < cfg.surprise_threshold:
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

            if step % 50 == 0:
                print(f"    Step {step}/{n_steps}: loss={loss.item():.4f}")

        elapsed = time.time() - start_time
        avg_loss = total_loss / max(n_steps, 1)

        if cfg.protect_slow_layers:
            for block in self.model.blocks:
                block.cms.unfreeze_all()

        record = {
            "file": filename, "path": image_path, "type": "image",
            "steps": n_steps, "avg_loss": round(avg_loss, 4),
            "time_seconds": round(elapsed, 1),
            "timestamp": datetime.now().isoformat(),
        }
        self.history.append(record)
        self.learned_files.add(filename)
        print(f"    Done: avg_loss={avg_loss:.4f}, time={elapsed:.1f}s")
        return record

    def learn_audio(self, audio_path: str, transcription: str) -> dict:
        """
        Teach the model an audio-transcription pair.

        Args:
            audio_path: path to audio file (.wav)
            transcription: text transcription of the audio
        """
        cfg = self.config.continual
        filename = os.path.basename(audio_path)
        print(f"\n  Learning audio: {filename}")

        if self.model.audio_encoder is None:
            print("  Model has no audio encoder. Load a multimodal model.")
            return {}

        try:
            import torchaudio
        except ImportError:
            print("  Requires: pip install torchaudio")
            return {}

        from .audio import AudioEncoder
        audio_cfg = self.config.audio

        # Load audio and compute mel
        waveform, sr = torchaudio.load(audio_path)
        if sr != audio_cfg.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, audio_cfg.sample_rate)(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)

        mel = AudioEncoder.compute_mel_spectrogram(waveform, audio_cfg)  # (1, n_mels, n_frames)
        mel = mel.to(self.device)

        n_tokens = AudioEncoder.compute_n_tokens(waveform.shape[0], audio_cfg)
        n_tokens = max(1, n_tokens)

        token_ids = self.tokenizer.encode_with_audio("", n_tokens, transcription)
        if len(token_ids) > self.config.model.max_seq_len:
            token_ids = token_ids[:self.config.model.max_seq_len - 1] + [self.tokenizer.eos_id]

        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        targets = input_ids.clone()

        frame_id = self.tokenizer.audio_frame_id
        audio_positions = (input_ids[0] == frame_id).nonzero(as_tuple=True)[0].unsqueeze(0)

        if cfg.protect_slow_layers:
            for block in self.model.blocks:
                block.cms.freeze_slow_layers()

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=cfg.learn_rate, weight_decay=0.01,
        )

        self.model.train()
        total_loss = 0.0
        n_steps = cfg.learn_steps_per_file
        start_time = time.time()

        for step in range(n_steps):
            self.model.step_cms()
            if step % 20 == 0:
                self.model.reset_fast_weights()

            result = self.model(input_ids, targets=targets, enable_self_mod=True,
                                mel_specs=mel, audio_positions=audio_positions)
            loss = result["loss"]

            if cfg.surprise_gate and result["surprise"] < cfg.surprise_threshold:
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

            if step % 50 == 0:
                print(f"    Step {step}/{n_steps}: loss={loss.item():.4f}")

        elapsed = time.time() - start_time
        avg_loss = total_loss / max(n_steps, 1)

        if cfg.protect_slow_layers:
            for block in self.model.blocks:
                block.cms.unfreeze_all()

        record = {
            "file": filename, "path": audio_path, "type": "audio",
            "steps": n_steps, "avg_loss": round(avg_loss, 4),
            "time_seconds": round(elapsed, 1),
            "timestamp": datetime.now().isoformat(),
        }
        self.history.append(record)
        self.learned_files.add(filename)
        print(f"    Done: avg_loss={avg_loss:.4f}, time={elapsed:.1f}s")
        return record

    def learn_batch(self, texts: list, steps: int = 50, lr: float = 5e-5) -> dict:
        """
        Train on a batch of text strings from the read loop.
        Reuses the same freeze/unfreeze + surprise gating pattern as learn_file().

        Args:
            texts: list of raw text strings to learn from
            steps: training steps for this consolidation
            lr: learning rate (lower than initial training)

        Returns:
            dict with avg_loss, steps, time_seconds
        """
        cfg = self.config.continual

        # Concatenate all batch texts with separator
        combined = "\n\n".join(t for t in texts if t.strip())
        if not combined:
            return {"avg_loss": 0.0, "steps": 0, "time_seconds": 0.0}

        # Tokenize
        token_ids = self.tokenizer.encode(combined, add_special_tokens=True)
        if not token_ids:
            return {"avg_loss": 0.0, "steps": 0, "time_seconds": 0.0}

        seq_len = self.config.model.max_seq_len
        data = torch.tensor(token_ids, dtype=torch.long)

        # fp32 for stable training on small models; for large models (3B+)
        # keep the current dtype to avoid OOM — the delta_W.to(x.dtype) fix
        # in model.py handles mixed dtype gracefully
        param_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
        if param_bytes < 4_000_000_000:  # < 4GB — safe to upcast to fp32
            self.model.float()
        self.model.train()

        if cfg.protect_slow_layers:
            for block in self.model.blocks:
                block.cms.freeze_slow_layers()

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=0.01,
        )

        total_loss = 0.0
        actual_steps = 0
        start_time = time.time()

        for step in range(steps):
            self.model.step_cms()
            if step % 20 == 0:
                self.model.reset_fast_weights()

            # Sliding window over the tokenized data
            offset = (step * seq_len) % max(len(data) - seq_len - 1, 1)
            end = min(offset + seq_len + 1, len(data))
            chunk = data[offset:end]
            if len(chunk) < 2:
                continue

            x = chunk[:-1].unsqueeze(0).to(self.device)
            y = chunk[1:].unsqueeze(0).to(self.device)

            result = self.model(x, targets=y, enable_self_mod=True)
            loss = result["loss"]

            # Surprise gating
            if cfg.surprise_gate and result["surprise"] < cfg.surprise_threshold:
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            actual_steps += 1

            if step % 50 == 0:
                print(f"    [batch] Step {step}/{steps}: loss={loss.item():.4f} surprise={result['surprise']:.4f}")

        elapsed = time.time() - start_time
        avg_loss = total_loss / max(actual_steps, 1)

        # Unfreeze all layers
        if cfg.protect_slow_layers:
            for block in self.model.blocks:
                block.cms.unfreeze_all()

        # Save checkpoint
        self._save()

        record = {
            "type": "batch",
            "texts_count": len(texts),
            "tokens": len(token_ids),
            "steps": actual_steps,
            "avg_loss": round(avg_loss, 4),
            "time_seconds": round(elapsed, 1),
            "timestamp": datetime.now().isoformat(),
        }
        self.history.append(record)
        print(f"    [batch] Done: {len(texts)} texts, avg_loss={avg_loss:.4f}, time={elapsed:.1f}s")
        return record

    def _save(self):
        """Save the model after learning."""
        os.makedirs(self.config.train.checkpoint_dir, exist_ok=True)
        save_path = os.path.join(self.config.train.checkpoint_dir, "learned_model.pt")
        torch.save({
            "model_state": self.model.state_dict(),
            "config": {
                "model": vars(self.config.model),
                "tokenizer": vars(self.config.tokenizer),
            },
            "learned_files": list(self.learned_files),
            "timestamp": datetime.now().isoformat(),
        }, save_path)



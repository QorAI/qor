"""
QOR Training Pipeline
======================
Production training with:
  - Mixed precision (FP16) for GPU acceleration
  - Gradient accumulation for effective larger batches
  - Cosine learning rate schedule with warmup
  - Checkpoint saving/loading with best model tracking
  - Validation loss tracking
  - Training metrics logging to JSON
"""

import os
import json
import glob
import math
import time
import torch
import torch.nn as nn
from typing import Optional
from datetime import datetime

from .config import QORConfig
from .model import QORModel
from .tokenizer import QORTokenizer
from .data import create_dataloaders

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

import logging

logger = logging.getLogger(__name__)


class Trainer:
    """Full-featured QOR training pipeline."""

    def __init__(self, config: QORConfig):
        self.config = config
        self.device = config.get_device()
        self.step = 0
        self.best_val_loss = float('inf')
        self.train_log = []
        self.is_distributed = getattr(config.train, 'distributed', False)
        self.rank = 0
        self.world_size = 1

        if self.is_distributed:
            import torch.distributed as dist
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.device = f"cuda:{self.rank}"
            logger.info(f"DDP rank {self.rank}/{self.world_size} on {self.device}")

    def train(self, resume_from: Optional[str] = None):
        """Main training entry point."""
        cfg = self.config
        print(f"\n{'='*60}")
        print(f"  QOR — The Qore Mind — Training")
        print(f"{'='*60}")
        print(f"  Device: {self.device}")

        # ===== Tokenizer =====
        tokenizer = QORTokenizer()
        tok_path = cfg.tokenizer.save_path
        if os.path.exists(tok_path):
            tokenizer.load(tok_path)
        else:
            if cfg.tokenizer.type == "pretrained":
                tokenizer.load_pretrained(cfg.tokenizer.pretrained_name)
            else:
                tokenizer.train(cfg.train.data_dir, cfg.tokenizer.vocab_size,
                               cfg.tokenizer.min_frequency, tok_path)

        # Sync vocab size
        cfg.model.vocab_size = tokenizer.vocab_size

        # ===== Data =====
        train_loader, val_loader = create_dataloaders(
            tokenizer, cfg.train.data_dir, cfg.model.max_seq_len,
            cfg.train.batch_size, cfg.train.val_split, cfg.train.num_workers,
        )

        # Distributed sampler
        if self.is_distributed:
            from torch.utils.data.distributed import DistributedSampler
            train_sampler = DistributedSampler(train_loader.dataset)
            train_loader = torch.utils.data.DataLoader(
                train_loader.dataset, batch_size=cfg.train.batch_size,
                sampler=train_sampler, num_workers=cfg.train.num_workers,
                pin_memory=True, drop_last=True,
            )

        # ===== Model =====
        model = QORModel(cfg.model).to(self.device)
        params = model.count_parameters()

        print(f"\n  Model Parameters:")
        print(f"    Total:      {params['total']:>12,} ({params['total']/1e6:.1f}M)")
        print(f"    Embedding:  {params['embedding']:>12,}")
        print(f"    Attention:  {params['attention']:>12,}")
        print(f"    Self-Mod:   {params['self_mod']:>12,}")
        print(f"    CMS Fast:   {params['cms_fast']:>12,}")
        print(f"    CMS Medium: {params['cms_medium']:>12,}")
        print(f"    CMS Slow:   {params['cms_slow']:>12,}")
        print(f"\n  Training:")
        print(f"    Steps:      {cfg.train.max_steps}")
        print(f"    Batch size: {cfg.train.batch_size} × {cfg.train.grad_accumulation_steps} accum = {cfg.train.batch_size * cfg.train.grad_accumulation_steps} effective")
        print(f"    Seq length: {cfg.model.max_seq_len}")
        print(f"    Mixed prec: {cfg.train.mixed_precision}")
        print(f"{'='*60}\n")

        # ===== Gradient checkpointing =====
        if getattr(cfg.train, 'gradient_checkpointing', False):
            model.enable_gradient_checkpointing()
            print(f"    Grad ckpt: True (saves ~2x VRAM)")

        # ===== torch.compile =====
        if getattr(cfg.model, 'compile', False):
            mode = getattr(cfg.model, 'compile_mode', 'reduce-overhead')
            model.compile_model(mode=mode)
            print(f"    Compiled:  True (mode={mode})")

        # ===== Distributed wrapping =====
        if self.is_distributed:
            from torch.nn.parallel import DistributedDataParallel as DDP
            model = DDP(model, device_ids=[self.rank])

        # ===== Optimizer =====
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay,
            betas=(0.9, 0.95),
        )

        # ===== Mixed precision =====
        scaler = None
        if cfg.train.mixed_precision and self.device == "cuda":
            scaler = torch.amp.GradScaler('cuda')

        # ===== Resume from checkpoint =====
        if resume_from and os.path.exists(resume_from):
            self._load_checkpoint(model, optimizer, scaler, resume_from)

        # ===== Create checkpoint dir =====
        os.makedirs(cfg.train.checkpoint_dir, exist_ok=True)

        # ===== Training loop =====
        model.train()
        data_iter = iter(train_loader)
        start_time = time.time()
        running_loss = 0.0
        val_loss = float('inf')

        print(f"  {'Step':>6} | {'Train':>8} | {'Val':>8} | {'Surprise':>9} | {'CMS':>12} | {'Speed':>10} | {'LR':>10}")
        print(f"  {'-'*80}")

        step_range = range(self.step + 1, cfg.train.max_steps + 1)
        if HAS_TQDM:
            pbar = tqdm(step_range, desc="Training", initial=self.step,
                        total=cfg.train.max_steps, ncols=100)
        else:
            pbar = step_range

        for step in pbar:
            self.step = step
            step_start = time.time()

            # Get batch (cycle through data)
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                x, y = next(data_iter)

            x, y = x.to(self.device), y.to(self.device)

            # Advance CMS clocks
            raw_model = model.module if self.is_distributed else model
            raw_model.step_cms()
            if step % 50 == 0:
                raw_model.reset_fast_weights()

            # Learning rate schedule
            lr = self._get_lr(step)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            # Forward + backward with mixed precision
            if scaler:
                with torch.amp.autocast('cuda'):
                    result = model(x, targets=y, enable_self_mod=True)
                    loss = result["loss"] / cfg.train.grad_accumulation_steps

                scaler.scale(loss).backward()

                if step % cfg.train.grad_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                result = model(x, targets=y, enable_self_mod=True)
                loss = result["loss"] / cfg.train.grad_accumulation_steps
                loss.backward()

                if step % cfg.train.grad_accumulation_steps == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()

            actual_loss = loss.item() * cfg.train.grad_accumulation_steps
            running_loss += actual_loss

            # ===== Logging =====
            if step % cfg.train.log_every == 0:
                elapsed = time.time() - step_start
                tokens_per_sec = cfg.train.batch_size * cfg.model.max_seq_len / max(elapsed, 1e-6)
                avg_loss = running_loss / cfg.train.log_every
                running_loss = 0.0
                cms_status = raw_model.blocks[0].cms.get_status()
                cms_str = f"F:{'●' if cms_status.get('fast') else '○'} M:{'●' if cms_status.get('medium') else '○'} S:{'●' if cms_status.get('slow') else '○'}"

                val_str = "   —    "
                if HAS_TQDM and hasattr(pbar, 'set_postfix'):
                    pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.6f}",
                                    speed=f"{tokens_per_sec:.0f}t/s")
                print(f"  {step:>6} | {avg_loss:>8.4f} | {val_str} | {result['surprise']:>9.4f} | {cms_str:>12} | {tokens_per_sec:>8.0f}t/s | {lr:>10.6f}")

                self.train_log.append({
                    "step": step,
                    "train_loss": avg_loss,
                    "surprise": result['surprise'],
                    "lr": lr,
                    "tokens_per_sec": tokens_per_sec,
                })

            # ===== Validation =====
            if step % cfg.train.eval_every == 0:
                val_loss = self._validate(model, val_loader)
                model.train()  # Back to training mode

                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss

                print(f"  {'':>6}   {'':>8}   {val_loss:>8.4f}{'*' if is_best else ' '}")

                self.train_log[-1]["val_loss"] = val_loss

            # ===== Checkpointing =====
            if step % cfg.train.save_every == 0 and self.rank == 0:
                self._save_checkpoint(model, optimizer, scaler,
                    os.path.join(cfg.train.checkpoint_dir, f"step_{step}.pt"))

                if val_loss <= self.best_val_loss:
                    self._save_checkpoint(model, optimizer, scaler,
                        os.path.join(cfg.train.checkpoint_dir, "best_model.pt"))

                # Cleanup old checkpoints
                self._cleanup_checkpoints()

        # ===== Final save =====
        total_time = time.time() - start_time
        if self.rank == 0:
            self._save_checkpoint(model, optimizer, scaler,
                os.path.join(cfg.train.checkpoint_dir, "final_model.pt"))

        if self.rank == 0:
            # Save training log
            log_path = os.path.join(cfg.train.checkpoint_dir, "training_log.json")
            with open(log_path, 'w') as f:
                json.dump(self.train_log, f, indent=2)

        if self.rank == 0:
            log_path = os.path.join(cfg.train.checkpoint_dir, "training_log.json")
            print(f"\n{'='*60}")
            print(f"  Training Complete!")
            print(f"  Time:       {total_time:.1f}s ({total_time/60:.1f} min)")
            print(f"  Best val:   {self.best_val_loss:.4f}")
            print(f"  Final model: {cfg.train.checkpoint_dir}/final_model.pt")
            print(f"  Best model:  {cfg.train.checkpoint_dir}/best_model.pt")
            print(f"  Log:         {log_path}")
            print(f"{'='*60}\n")

    def _get_lr(self, step):
        cfg = self.config.train
        if step < cfg.warmup_steps:
            return cfg.learning_rate * step / cfg.warmup_steps
        decay_ratio = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)

    @torch.no_grad()
    def _validate(self, model, val_loader):
        model.eval()
        total_loss = 0.0
        n_batches = 0
        for x, y in val_loader:
            x, y = x.to(self.device), y.to(self.device)
            result = model(x, targets=y, enable_self_mod=False)
            total_loss += result["loss"].item()
            n_batches += 1
            if n_batches >= 50:  # Cap validation batches
                break
        return total_loss / max(n_batches, 1)

    def _save_checkpoint(self, model, optimizer, scaler, path):
        checkpoint = {
            "step": self.step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": json.loads(json.dumps({
                "model": vars(self.config.model),
                "tokenizer": vars(self.config.tokenizer),
            })),
            "best_val_loss": self.best_val_loss,
            "timestamp": datetime.now().isoformat(),
        }
        if scaler:
            checkpoint["scaler_state"] = scaler.state_dict()
        torch.save(checkpoint, path)

    def _load_checkpoint(self, model, optimizer, scaler, path):
        print(f"  Resuming from {path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.step = checkpoint["step"]
        self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))
        if scaler and "scaler_state" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state"])
        print(f"  Resumed at step {self.step}, best val loss: {self.best_val_loss:.4f}")

    def train_multimodal(self, modality: str, data_path: str,
                         resume_from: Optional[str] = None):
        """
        Train with multimodal data (vision or audio).

        Args:
            modality: "vision" or "audio"
            data_path: path to image/audio data directory
            resume_from: optional checkpoint to resume from
        """
        from .multimodal_data import create_multimodal_dataloaders

        cfg = self.config
        print(f"\n{'='*60}")
        print(f"  QOR — Multimodal Training ({modality})")
        print(f"{'='*60}")
        print(f"  Device: {self.device}")
        print(f"  Data: {data_path}")

        # ===== Tokenizer =====
        tokenizer = QORTokenizer()
        tok_path = cfg.tokenizer.save_path
        if os.path.exists(tok_path):
            tokenizer.load(tok_path)
        else:
            if cfg.tokenizer.type == "pretrained":
                tokenizer.load_pretrained(cfg.tokenizer.pretrained_name)
            else:
                tokenizer.train(cfg.train.data_dir, cfg.tokenizer.vocab_size,
                               cfg.tokenizer.min_frequency, tok_path)

        cfg.model.vocab_size = tokenizer.vocab_size

        # ===== Data =====
        vision_config = cfg.vision if modality == "vision" else None
        audio_config = cfg.audio if modality == "audio" else None

        train_loader, val_loader = create_multimodal_dataloaders(
            modality=modality,
            data_dir=data_path,
            tokenizer=tokenizer,
            vision_config=vision_config,
            audio_config=audio_config,
            max_seq_len=cfg.model.max_seq_len,
            batch_size=cfg.train.batch_size,
            val_split=cfg.train.val_split,
            num_workers=0,  # Avoid multiprocessing issues with PIL/torchaudio
        )

        # ===== Model with multimodal encoders =====
        model = QORModel(
            cfg.model,
            vision_config=vision_config,
            audio_config=audio_config,
        ).to(self.device)
        params = model.count_parameters()

        print(f"\n  Model Parameters:")
        print(f"    Total:      {params['total']:>12,} ({params['total']/1e6:.1f}M)")
        print(f"    Embedding:  {params['embedding']:>12,}")
        print(f"    Attention:  {params['attention']:>12,}")
        print(f"    Self-Mod:   {params['self_mod']:>12,}")
        if params.get('vision', 0) > 0:
            print(f"    Vision:     {params['vision']:>12,}")
        if params.get('audio', 0) > 0:
            print(f"    Audio:      {params['audio']:>12,}")
        print(f"\n  Training:")
        print(f"    Steps:      {cfg.train.max_steps}")
        print(f"    Batch size: {cfg.train.batch_size}")
        print(f"    Seq length: {cfg.model.max_seq_len}")
        print(f"{'='*60}\n")

        # ===== Gradient checkpointing =====
        if getattr(cfg.train, 'gradient_checkpointing', False):
            model.enable_gradient_checkpointing()

        # ===== Optimizer (exclude frozen pretrained encoder params) =====
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        total_params = sum(p.numel() for p in model.parameters())
        trainable_count = sum(p.numel() for p in trainable_params)
        frozen_count = total_params - trainable_count
        if frozen_count > 0:
            print(f"    Frozen:     {frozen_count:>12,} ({frozen_count/1e6:.1f}M)")
            print(f"    Trainable:  {trainable_count:>12,} ({trainable_count/1e6:.1f}M)")

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay,
            betas=(0.9, 0.95),
        )

        # ===== Mixed precision =====
        scaler = None
        if cfg.train.mixed_precision and self.device == "cuda":
            scaler = torch.amp.GradScaler('cuda')

        # ===== Resume =====
        if resume_from and os.path.exists(resume_from):
            self._load_checkpoint(model, optimizer, scaler, resume_from)

        os.makedirs(cfg.train.checkpoint_dir, exist_ok=True)

        # ===== Training loop =====
        model.train()
        data_iter = iter(train_loader)
        start_time = time.time()
        running_loss = 0.0

        print(f"  {'Step':>6} | {'Loss':>8} | {'Surprise':>9} | {'Speed':>10} | {'LR':>10}")
        print(f"  {'-'*65}")

        for step in range(self.step + 1, cfg.train.max_steps + 1):
            self.step = step
            step_start = time.time()

            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            # Move to device
            input_ids = batch["input_ids"].to(self.device)
            targets = batch["targets"].to(self.device)
            images = batch.get("images")
            image_positions = batch.get("image_positions")
            mel_specs = batch.get("mel_specs")
            audio_positions = batch.get("audio_positions")

            if images is not None:
                images = images.to(self.device)
            if image_positions is not None:
                image_positions = image_positions.to(self.device)
            if mel_specs is not None:
                mel_specs = mel_specs.to(self.device)
            if audio_positions is not None:
                audio_positions = audio_positions.to(self.device)

            # CMS clock
            raw_model = model.module if self.is_distributed else model
            raw_model.step_cms()
            if step % 50 == 0:
                raw_model.reset_fast_weights()

            # LR schedule
            lr = self._get_lr(step)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            # Forward + backward
            if scaler:
                with torch.amp.autocast('cuda'):
                    result = model(
                        input_ids, targets=targets, enable_self_mod=True,
                        images=images, image_positions=image_positions,
                        mel_specs=mel_specs, audio_positions=audio_positions,
                    )
                    loss = result["loss"] / cfg.train.grad_accumulation_steps

                scaler.scale(loss).backward()

                if step % cfg.train.grad_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                result = model(
                    input_ids, targets=targets, enable_self_mod=True,
                    images=images, image_positions=image_positions,
                    mel_specs=mel_specs, audio_positions=audio_positions,
                )
                loss = result["loss"] / cfg.train.grad_accumulation_steps
                loss.backward()

                if step % cfg.train.grad_accumulation_steps == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()

            actual_loss = loss.item() * cfg.train.grad_accumulation_steps
            running_loss += actual_loss

            # Logging
            if step % cfg.train.log_every == 0:
                elapsed = time.time() - step_start
                avg_loss = running_loss / cfg.train.log_every
                running_loss = 0.0
                tokens_per_sec = cfg.train.batch_size * cfg.model.max_seq_len / max(elapsed, 1e-6)

                print(f"  {step:>6} | {avg_loss:>8.4f} | {result['surprise']:>9.4f} | {tokens_per_sec:>8.0f}t/s | {lr:>10.6f}")

                self.train_log.append({
                    "step": step,
                    "train_loss": avg_loss,
                    "surprise": result['surprise'],
                    "lr": lr,
                })

            # Validation
            if step % cfg.train.eval_every == 0 and val_loader is not None:
                val_loss = self._validate_multimodal(model, val_loader)
                model.train()
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                print(f"  {'':>6}   Val: {val_loss:.4f}{'*' if is_best else ''}")

            # Checkpointing
            if step % cfg.train.save_every == 0 and self.rank == 0:
                self._save_checkpoint(model, optimizer, scaler,
                    os.path.join(cfg.train.checkpoint_dir, f"mm_{modality}_step_{step}.pt"))

        # Final save
        total_time = time.time() - start_time
        if self.rank == 0:
            final_path = os.path.join(cfg.train.checkpoint_dir, f"mm_{modality}_final.pt")
            self._save_checkpoint(model, optimizer, scaler, final_path)

            print(f"\n{'='*60}")
            print(f"  Multimodal Training Complete!")
            print(f"  Time:       {total_time:.1f}s ({total_time/60:.1f} min)")
            print(f"  Best val:   {self.best_val_loss:.4f}")
            print(f"  Model:      {final_path}")
            print(f"{'='*60}\n")

    @torch.no_grad()
    def _validate_multimodal(self, model, val_loader):
        """Validate on multimodal data."""
        model.eval()
        total_loss = 0.0
        n_batches = 0
        for batch in val_loader:
            input_ids = batch["input_ids"].to(self.device)
            targets = batch["targets"].to(self.device)
            images = batch.get("images")
            image_positions = batch.get("image_positions")
            mel_specs = batch.get("mel_specs")
            audio_positions = batch.get("audio_positions")

            if images is not None:
                images = images.to(self.device)
            if image_positions is not None:
                image_positions = image_positions.to(self.device)
            if mel_specs is not None:
                mel_specs = mel_specs.to(self.device)
            if audio_positions is not None:
                audio_positions = audio_positions.to(self.device)

            result = model(
                input_ids, targets=targets, enable_self_mod=False,
                images=images, image_positions=image_positions,
                mel_specs=mel_specs, audio_positions=audio_positions,
            )
            total_loss += result["loss"].item()
            n_batches += 1
            if n_batches >= 50:
                break
        return total_loss / max(n_batches, 1)

    def _cleanup_checkpoints(self):
        """Keep only the N most recent checkpoints."""
        cfg = self.config.train
        checkpoints = sorted(glob.glob(os.path.join(cfg.checkpoint_dir, "step_*.pt")))
        while len(checkpoints) > cfg.max_checkpoints:
            os.remove(checkpoints.pop(0))

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


class Trainer:
    """Full-featured QOR training pipeline."""

    def __init__(self, config: QORConfig):
        self.config = config
        self.device = config.get_device()
        self.step = 0
        self.best_val_loss = float('inf')
        self.train_log = []

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

        print(f"  {'Step':>6} | {'Train':>8} | {'Val':>8} | {'Surprise':>9} | {'CMS':>12} | {'Speed':>10} | {'LR':>10}")
        print(f"  {'-'*80}")

        for step in range(self.step + 1, cfg.train.max_steps + 1):
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
            model.step_cms()
            if step % 50 == 0:
                model.reset_fast_weights()

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
                cms_status = model.blocks[0].cms.get_status()
                cms_str = f"F:{'●' if cms_status.get('fast') else '○'} M:{'●' if cms_status.get('medium') else '○'} S:{'●' if cms_status.get('slow') else '○'}"

                val_str = "   —    "
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
            if step % cfg.train.save_every == 0:
                self._save_checkpoint(model, optimizer, scaler,
                    os.path.join(cfg.train.checkpoint_dir, f"step_{step}.pt"))

                if val_loss <= self.best_val_loss:
                    self._save_checkpoint(model, optimizer, scaler,
                        os.path.join(cfg.train.checkpoint_dir, "best_model.pt"))

                # Cleanup old checkpoints
                self._cleanup_checkpoints()

        # ===== Final save =====
        total_time = time.time() - start_time
        self._save_checkpoint(model, optimizer, scaler,
            os.path.join(cfg.train.checkpoint_dir, "final_model.pt"))

        # Save training log
        log_path = os.path.join(cfg.train.checkpoint_dir, "training_log.json")
        with open(log_path, 'w') as f:
            json.dump(self.train_log, f, indent=2)

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

    def _cleanup_checkpoints(self):
        """Keep only the N most recent checkpoints."""
        cfg = self.config.train
        checkpoints = sorted(glob.glob(os.path.join(cfg.checkpoint_dir, "step_*.pt")))
        while len(checkpoints) > cfg.max_checkpoints:
            os.remove(checkpoints.pop(0))


import glob  # needed for cleanup

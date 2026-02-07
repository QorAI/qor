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

        # Tokenizer
        self.tokenizer = QORTokenizer()
        self.tokenizer.load(self.config.tokenizer.save_path)
        self.config.model.vocab_size = self.tokenizer.vocab_size

        # Model
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model = QORModel(self.config.model).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])

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

        # Find new files
        files = sorted(glob.glob(os.path.join(folder, '**', '*.txt'), recursive=True))
        new_files = [f for f in files if os.path.basename(f) not in self.learned_files]

        if not new_files:
            print(f"\n  No new files in '{folder}/' — already learned {len(self.learned_files)} files")
            print(f"  Drop new .txt files in '{folder}/' and run again!")
            return

        print(f"\n{'='*60}")
        print(f"  QOR Continual Learning")
        print(f"  New files to learn: {len(new_files)}")
        print(f"  Previously learned: {len(self.learned_files)}")
        print(f"{'='*60}")

        for fpath in new_files:
            self.learn_file(fpath)

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

    def _save(self):
        """Save the model after learning."""
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


import glob  # for folder scanning

# Allow folder to be Optional
from typing import Optional

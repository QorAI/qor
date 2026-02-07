"""
QOR Data Pipeline
==================
Handles text data loading, chunking, and batching for training.
Supports: text files, folders, and streaming datasets.
"""

import os
import glob
import random
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional


class TextFileDataset(Dataset):
    """
    Loads all text files from a directory, tokenizes them,
    and serves fixed-length chunks for training.
    """

    def __init__(self, tokenizer, data_dir: str, seq_len: int = 512,
                 val_split: float = 0.05, split: str = "train"):
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        # Find all text files
        files = []
        for ext in ['*.txt', '*.md', '*.text']:
            files.extend(glob.glob(os.path.join(data_dir, '**', ext), recursive=True))

        if not files:
            raise FileNotFoundError(
                f"\nNo text files found in '{data_dir}/'!\n\n"
                f"To train QOR, put your .txt files in the '{data_dir}' folder.\n"
                f"Example:\n"
                f"  {data_dir}/my_book.txt\n"
                f"  {data_dir}/articles.txt\n"
                f"  {data_dir}/wikipedia.txt\n"
            )

        # Read and tokenize everything
        print(f"Loading {len(files)} text files from {data_dir}/")
        all_ids = []
        total_chars = 0
        for fpath in sorted(files):
            with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()
            total_chars += len(text)
            ids = tokenizer.encode(text, add_special_tokens=False)
            all_ids.extend(ids)
            all_ids.append(tokenizer.eos_id)  # Separator between files

        # Split into train/val
        split_idx = int(len(all_ids) * (1 - val_split))
        if split == "train":
            self.data = torch.tensor(all_ids[:split_idx], dtype=torch.long)
        else:
            self.data = torch.tensor(all_ids[split_idx:], dtype=torch.long)

        self.n_chunks = max(1, len(self.data) - seq_len - 1)

        print(f"  Total: {total_chars:,} chars → {len(all_ids):,} tokens")
        print(f"  {split.capitalize()} split: {len(self.data):,} tokens ({self.n_chunks:,} chunks)")

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        # Random chunk from the data
        start = random.randint(0, len(self.data) - self.seq_len - 1)
        x = self.data[start:start + self.seq_len]
        y = self.data[start + 1:start + self.seq_len + 1]
        return x, y


class SingleFileDataset(Dataset):
    """For continual learning — load and learn from a single file."""

    def __init__(self, tokenizer, file_path: str, seq_len: int = 512):
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()

        ids = tokenizer.encode(text, add_special_tokens=False)
        self.data = torch.tensor(ids, dtype=torch.long)
        self.seq_len = seq_len
        self.n_chunks = max(1, len(self.data) - seq_len - 1)

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        start = random.randint(0, len(self.data) - self.seq_len - 1)
        x = self.data[start:start + self.seq_len]
        y = self.data[start + 1:start + self.seq_len + 1]
        return x, y


def create_dataloaders(tokenizer, data_dir: str, seq_len: int = 512,
                       batch_size: int = 8, val_split: float = 0.05,
                       num_workers: int = 2):
    """Create training and validation dataloaders."""
    train_ds = TextFileDataset(tokenizer, data_dir, seq_len, val_split, "train")
    val_ds = TextFileDataset(tokenizer, data_dir, seq_len, val_split, "val")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )

    return train_loader, val_loader

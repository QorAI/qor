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
from torch.utils.data import Dataset, DataLoader, IterableDataset
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

        print(f"  Total: {total_chars:,} chars -> {len(all_ids):,} tokens")
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


class StreamingTextDataset(IterableDataset):
    """
    Streaming dataset for large text files — never loads full data into memory.
    Reads files line-by-line, tokenizes on-the-fly, yields fixed-length chunks.
    """

    def __init__(self, tokenizer, data_dir: str, seq_len: int = 512):
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.files = []
        for ext in ['*.txt', '*.md', '*.text']:
            self.files.extend(glob.glob(os.path.join(data_dir, '**', ext), recursive=True))
        self.files.sort()

    def __iter__(self):
        buffer = []
        for fpath in self.files:
            with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    ids = self.tokenizer.encode(line, add_special_tokens=False)
                    buffer.extend(ids)
                    while len(buffer) >= self.seq_len + 1:
                        x = torch.tensor(buffer[:self.seq_len], dtype=torch.long)
                        y = torch.tensor(buffer[1:self.seq_len + 1], dtype=torch.long)
                        yield x, y
                        buffer = buffer[self.seq_len:]


def dynamic_batch_collate(batch):
    """
    Collate function that pads to max length in batch (not global max).
    Returns attention masks alongside inputs/targets.
    Reduces wasted computation from padding.
    """
    xs, ys = zip(*batch)
    max_len = max(x.size(0) for x in xs)

    padded_x = torch.zeros(len(xs), max_len, dtype=torch.long)
    padded_y = torch.zeros(len(ys), max_len, dtype=torch.long)
    masks = torch.zeros(len(xs), max_len, dtype=torch.float)

    for i, (x, y) in enumerate(zip(xs, ys)):
        length = x.size(0)
        padded_x[i, :length] = x
        padded_y[i, :length] = y
        masks[i, :length] = 1.0

    return padded_x, padded_y, masks


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

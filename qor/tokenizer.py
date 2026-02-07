"""
QOR Tokenizer — Proper BPE tokenization
========================================
Two modes:
  1. Train your own BPE tokenizer on your data (recommended)
  2. Use a pretrained tokenizer (GPT-2, etc.)

A proper tokenizer makes the model 5-10x more efficient than byte-level.
"""

import os
import json
import glob
from typing import List, Optional


class QORTokenizer:
    """
    Production tokenizer for QOR.

    Wraps HuggingFace tokenizers library for BPE training,
    with fallback to a simple character-level tokenizer if
    the library isn't available.
    """

    def __init__(self):
        self.tokenizer = None
        self.vocab_size = 0
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.unk_id = 3
        self._type = None

    def train(self, data_dir: str, vocab_size: int = 8192,
              min_frequency: int = 2, save_path: str = "tokenizer.json"):
        """
        Train a BPE tokenizer on all .txt files in data_dir.

        Args:
            data_dir: Folder containing .txt training files
            vocab_size: Size of vocabulary to learn
            min_frequency: Min frequency for a token to be included
            save_path: Where to save the trained tokenizer
        """
        try:
            from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders
        except ImportError:
            print("=" * 60)
            print("  Install tokenizers: pip install tokenizers")
            print("  Falling back to character-level tokenizer")
            print("=" * 60)
            self._train_char_level(data_dir, save_path)
            return

        print(f"Training BPE tokenizer (vocab_size={vocab_size})...")

        # Collect all text files
        files = []
        for ext in ['*.txt', '*.md', '*.csv', '*.json']:
            files.extend(glob.glob(os.path.join(data_dir, '**', ext), recursive=True))

        if not files:
            raise FileNotFoundError(
                f"No text files found in {data_dir}/\n"
                f"Put your training .txt files in the '{data_dir}' folder first!"
            )

        print(f"  Found {len(files)} text files")

        # Build BPE tokenizer
        tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()

        # Special tokens
        special_tokens = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
            show_progress=True,
        )

        # Train
        tokenizer.train(files, trainer)

        # Add post-processing (BOS/EOS wrapping)
        tokenizer.post_processor = processors.TemplateProcessing(
            single="<BOS> $A <EOS>",
            special_tokens=[
                ("<BOS>", tokenizer.token_to_id("<BOS>")),
                ("<EOS>", tokenizer.token_to_id("<EOS>")),
            ],
        )

        # Enable padding
        tokenizer.enable_padding(
            pad_id=tokenizer.token_to_id("<PAD>"),
            pad_token="<PAD>",
        )

        # Save
        tokenizer.save(save_path)

        # Set internal state
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.get_vocab_size()
        self.pad_id = tokenizer.token_to_id("<PAD>")
        self.bos_id = tokenizer.token_to_id("<BOS>")
        self.eos_id = tokenizer.token_to_id("<EOS>")
        self.unk_id = tokenizer.token_to_id("<UNK>")
        self._type = "bpe"

        print(f"  Vocabulary size: {self.vocab_size}")
        print(f"  Saved to: {save_path}")
        print(f"  Special tokens: PAD={self.pad_id}, BOS={self.bos_id}, EOS={self.eos_id}")

    def load(self, path: str):
        """Load a saved tokenizer."""
        if path.endswith('.json'):
            # Try HuggingFace tokenizers format
            try:
                from tokenizers import Tokenizer
                self.tokenizer = Tokenizer.from_file(path)
                self.vocab_size = self.tokenizer.get_vocab_size()
                self.pad_id = self.tokenizer.token_to_id("<PAD>") or 0
                self.bos_id = self.tokenizer.token_to_id("<BOS>") or 1
                self.eos_id = self.tokenizer.token_to_id("<EOS>") or 2
                self.unk_id = self.tokenizer.token_to_id("<UNK>") or 3
                self._type = "bpe"
                print(f"Loaded BPE tokenizer: {self.vocab_size} tokens")
                return
            except Exception:
                pass

            # Try char-level format
            with open(path) as f:
                data = json.load(f)
            if data.get("type") == "char":
                self._load_char_level(data)
                return

        # Try pretrained HuggingFace
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.vocab_size = len(self.tokenizer)
            self.pad_id = self.tokenizer.pad_token_id or 0
            self.bos_id = self.tokenizer.bos_token_id or 1
            self.eos_id = self.tokenizer.eos_token_id or 2
            self.unk_id = self.tokenizer.unk_token_id or 3
            self._type = "pretrained"
            print(f"Loaded pretrained tokenizer: {self.vocab_size} tokens")
            return
        except Exception:
            pass

        raise ValueError(f"Could not load tokenizer from {path}")

    def load_pretrained(self, name: str = "gpt2"):
        """Load a pretrained tokenizer (GPT-2, etc.)."""
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(name)
            # GPT-2 doesn't have pad token by default
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.vocab_size = len(self.tokenizer)
            self.pad_id = self.tokenizer.pad_token_id
            self.bos_id = self.tokenizer.bos_token_id or 0
            self.eos_id = self.tokenizer.eos_token_id
            self.unk_id = self.tokenizer.unk_token_id or 0
            self._type = "pretrained"
            print(f"Loaded {name} tokenizer: {self.vocab_size} tokens")
        except ImportError:
            print("Install transformers: pip install transformers")
            raise

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs."""
        if self._type == "bpe":
            encoded = self.tokenizer.encode(text)
            ids = encoded.ids
            if not add_special_tokens and len(ids) >= 2:
                ids = ids[1:-1]  # Strip BOS/EOS
            return ids
        elif self._type == "pretrained":
            return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        elif self._type == "char":
            return self._encode_char(text, add_special_tokens)
        else:
            raise RuntimeError("Tokenizer not initialized. Call train() or load() first.")

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""
        if self._type == "bpe":
            return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
        elif self._type == "pretrained":
            return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
        elif self._type == "char":
            return self._decode_char(ids, skip_special_tokens)
        else:
            raise RuntimeError("Tokenizer not initialized.")

    def batch_encode(self, texts: List[str], max_length: int = 512,
                     padding: bool = True, truncation: bool = True) -> dict:
        """Encode a batch of texts with padding and truncation."""
        if self._type == "bpe":
            self.tokenizer.enable_padding(length=max_length, pad_id=self.pad_id)
            self.tokenizer.enable_truncation(max_length=max_length)
            encoded = self.tokenizer.encode_batch(texts)
            input_ids = [e.ids for e in encoded]
            attention_mask = [e.attention_mask for e in encoded]
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        elif self._type == "pretrained":
            return self.tokenizer(
                texts, max_length=max_length, padding=padding,
                truncation=truncation, return_tensors=None
            )
        elif self._type == "char":
            return self._batch_encode_char(texts, max_length)

    # ===== Character-level fallback (if tokenizers not installed) =====

    def _train_char_level(self, data_dir, save_path):
        """Simple character-level tokenizer as fallback."""
        chars = set()
        files = glob.glob(os.path.join(data_dir, '**', '*.txt'), recursive=True)
        for fpath in files:
            with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
                chars.update(f.read())

        # Sort for deterministic ordering
        chars = sorted(chars)

        # Build vocab: special tokens + characters
        self._char_to_id = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self._id_to_char = {0: "<PAD>", 1: "<BOS>", 2: "<EOS>", 3: "<UNK>"}

        for i, ch in enumerate(chars, start=4):
            self._char_to_id[ch] = i
            self._id_to_char[i] = ch

        self.vocab_size = len(self._char_to_id)
        self._type = "char"

        # Save
        data = {
            "type": "char",
            "char_to_id": self._char_to_id,
            "vocab_size": self.vocab_size,
        }
        with open(save_path, 'w') as f:
            json.dump(data, f, ensure_ascii=False)

        print(f"  Character tokenizer: {self.vocab_size} tokens")
        print(f"  Saved to: {save_path}")

    def _load_char_level(self, data):
        self._char_to_id = data["char_to_id"]
        self._id_to_char = {int(v): k for k, v in self._char_to_id.items()}
        self.vocab_size = data["vocab_size"]
        self._type = "char"
        print(f"Loaded character tokenizer: {self.vocab_size} tokens")

    def _encode_char(self, text, add_special_tokens):
        ids = [self._char_to_id.get(ch, self.unk_id) for ch in text]
        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def _decode_char(self, ids, skip_special_tokens):
        special = {self.pad_id, self.bos_id, self.eos_id, self.unk_id}
        chars = []
        for i in ids:
            if skip_special_tokens and i in special:
                continue
            chars.append(self._id_to_char.get(i, '?'))
        return ''.join(chars)

    def _batch_encode_char(self, texts, max_length):
        batch_ids = []
        batch_mask = []
        for text in texts:
            ids = self._encode_char(text, True)[:max_length]
            mask = [1] * len(ids)
            # Pad
            while len(ids) < max_length:
                ids.append(self.pad_id)
                mask.append(0)
            batch_ids.append(ids)
            batch_mask.append(mask)
        return {"input_ids": batch_ids, "attention_mask": batch_mask}

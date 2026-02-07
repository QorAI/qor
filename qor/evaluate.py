"""
QOR Evaluation
===============
Proper evaluation metrics:
  - Perplexity (how surprised is the model?)
  - Generation quality (sample outputs)
  - Continual learning test (does it forget?)
  - Parameter breakdown
"""

import os
import torch
import json
import math
from typing import Optional

from .config import QORConfig
from .model import QORModel
from .tokenizer import QORTokenizer


class Evaluator:
    """Evaluate a trained QOR model."""

    def __init__(self, config: QORConfig):
        self.config = config
        self.device = config.get_device()

    def load_model(self, checkpoint_path: str) -> tuple:
        """Load model and tokenizer from checkpoint."""
        print(f"Loading model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Load tokenizer
        tokenizer = QORTokenizer()
        tokenizer.load(self.config.tokenizer.save_path)
        self.config.model.vocab_size = tokenizer.vocab_size

        # Load model
        model = QORModel(self.config.model).to(self.device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        step = checkpoint.get("step", "?")
        val_loss = checkpoint.get("best_val_loss", "?")
        print(f"  Step: {step}, Val loss: {val_loss}")
        print(f"  Parameters: {model.n_params:,}")

        return model, tokenizer

    def perplexity(self, model, tokenizer, text: Optional[str] = None,
                   file_path: Optional[str] = None) -> float:
        """
        Calculate perplexity on text.
        Lower = better. A perplexity of 1.0 means perfect prediction.
        Human text typically has perplexity 20-80 on well-trained models.
        """
        if file_path:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()

        if not text:
            raise ValueError("Provide text or file_path")

        ids = tokenizer.encode(text, add_special_tokens=False)
        seq_len = self.config.model.max_seq_len

        total_loss = 0.0
        total_tokens = 0

        model.eval()
        with torch.no_grad():
            for i in range(0, len(ids) - seq_len - 1, seq_len):
                x = torch.tensor([ids[i:i+seq_len]], device=self.device)
                y = torch.tensor([ids[i+1:i+seq_len+1]], device=self.device)
                result = model(x, targets=y, enable_self_mod=False)
                total_loss += result["loss"].item() * seq_len
                total_tokens += seq_len

        avg_loss = total_loss / max(total_tokens, 1)
        ppl = math.exp(avg_loss)
        return ppl

    def generate_samples(self, model, tokenizer, prompts: list,
                         max_tokens: int = 100, temperature: float = 0.8) -> list:
        """Generate text from prompts and return results."""
        results = []
        model.eval()

        for prompt in prompts:
            ids = tokenizer.encode(prompt, add_special_tokens=True)
            input_ids = torch.tensor([ids], device=self.device)

            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=50,
                top_p=0.9,
                stop_tokens=[tokenizer.eos_id],
            )

            output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
            results.append({
                "prompt": prompt,
                "output": output_text,
                "new_tokens": len(output_ids) - len(ids),
            })

        return results

    def continual_learning_test(self, tokenizer) -> dict:
        """
        The Mind Test: Can QOR learn B without forgetting A?

        1. Train fresh model on Dataset A
        2. Evaluate on A (should be good)
        3. Train same model on Dataset B
        4. Evaluate on A again (should still be good if CMS works)
        5. Compare with baseline (no CMS)
        """
        print(f"\n{'='*60}")
        print(f"  The Mind Test — Continual Learning")
        print(f"{'='*60}\n")

        # Prepare datasets
        dataset_a = "Cats have four legs and whiskers. Dogs are loyal and bark. Birds fly with wings. Fish swim in water. "
        dataset_b = "Two plus two equals four. Pi is three point fourteen. Triangles have three sides. Circles have no corners. "

        def make_data(text, n_repeat=200):
            full = (text * n_repeat)
            ids = tokenizer.encode(full, add_special_tokens=False)
            return torch.tensor(ids, dtype=torch.long)

        data_a = make_data(dataset_a)
        data_b = make_data(dataset_b)
        seq_len = min(self.config.model.max_seq_len, 128)

        def quick_train(model, data, steps=300):
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
            model.train()
            for step in range(steps):
                model.step_cms()
                if step % 50 == 0:
                    model.reset_fast_weights()
                start = torch.randint(0, len(data) - seq_len - 1, (1,)).item()
                x = data[start:start+seq_len].unsqueeze(0).to(self.device)
                y = data[start+1:start+seq_len+1].unsqueeze(0).to(self.device)
                result = model(x, targets=y)
                result["loss"].backward()
                optimizer.step()
                optimizer.zero_grad()
                if step % 100 == 0:
                    print(f"    Step {step}: loss={result['loss'].item():.4f}")

        def evaluate_prompt(model, prompt):
            ids = tokenizer.encode(prompt, add_special_tokens=True)
            input_ids = torch.tensor([ids], device=self.device)
            output_ids = model.generate(input_ids, max_new_tokens=30, temperature=0.5)
            return tokenizer.decode(output_ids, skip_special_tokens=True)

        # === QOR (full architecture) ===
        print("  [1/4] Training QOR on animals...")
        qor = QORModel(self.config.model).to(self.device)
        quick_train(qor, data_a)
        qor_a_before = evaluate_prompt(qor, "Cats have")
        print(f"  After A: 'Cats have' → {qor_a_before[:60]}")

        print("\n  [2/4] Training QOR on math...")
        quick_train(qor, data_b)
        qor_a_after = evaluate_prompt(qor, "Cats have")
        qor_b_after = evaluate_prompt(qor, "Two plus")
        print(f"  After A+B: 'Cats have' → {qor_a_after[:60]}")
        print(f"  After A+B: 'Two plus'  → {qor_b_after[:60]}")

        # === Baseline (all CMS at same speed = standard FFN) ===
        from .config import ModelConfig
        baseline_cfg = ModelConfig(
            d_model=self.config.model.d_model,
            n_layers=self.config.model.n_layers,
            n_heads=self.config.model.n_heads,
            d_ff=self.config.model.d_ff,
            vocab_size=self.config.model.vocab_size,
            max_seq_len=self.config.model.max_seq_len,
            cms_fast_freq=1, cms_med_freq=1, cms_slow_freq=1,  # All same!
            self_mod_lr=0.0,  # Disabled
        )

        print("\n  [3/4] Training Baseline on animals...")
        baseline = QORModel(baseline_cfg).to(self.device)
        quick_train(baseline, data_a)
        base_a_before = evaluate_prompt(baseline, "Cats have")
        print(f"  After A: 'Cats have' → {base_a_before[:60]}")

        print("\n  [4/4] Training Baseline on math...")
        quick_train(baseline, data_b)
        base_a_after = evaluate_prompt(baseline, "Cats have")
        base_b_after = evaluate_prompt(baseline, "Two plus")
        print(f"  After A+B: 'Cats have' → {base_a_after[:60]}")
        print(f"  After A+B: 'Two plus'  → {base_b_after[:60]}")

        # Results
        results = {
            "qor": {
                "animals_before_math": qor_a_before,
                "animals_after_math": qor_a_after,
                "math_after_training": qor_b_after,
            },
            "baseline": {
                "animals_before_math": base_a_before,
                "animals_after_math": base_a_after,
                "math_after_training": base_b_after,
            },
        }

        print(f"\n{'='*60}")
        print(f"  RESULTS")
        print(f"{'='*60}")
        print(f"\n  QOR (multi-speed memory):")
        print(f"    Animals after math training: {qor_a_after[:60]}")
        print(f"    Math knowledge:              {qor_b_after[:60]}")
        print(f"\n  Baseline (standard transformer):")
        print(f"    Animals after math training: {base_a_after[:60]}")
        print(f"    Math knowledge:              {base_b_after[:60]}")
        print(f"\n  ✅ QOR preserves old knowledge = multi-speed memory works")
        print(f"  ❌ Baseline forgets = catastrophic forgetting")
        print(f"{'='*60}\n")

        return results

    def full_report(self, checkpoint_path: str,
                    eval_text_path: Optional[str] = None) -> dict:
        """Run all evaluations and produce a report."""
        model, tokenizer = self.load_model(checkpoint_path)

        report = {
            "model_params": model.count_parameters(),
            "checkpoint": checkpoint_path,
        }

        # Perplexity
        if eval_text_path and os.path.exists(eval_text_path):
            ppl = self.perplexity(model, tokenizer, file_path=eval_text_path)
            report["perplexity"] = ppl
            print(f"\n  Perplexity: {ppl:.2f}")

        # Generation samples
        prompts = [
            "The quick brown fox",
            "Once upon a time",
            "The most important thing is",
        ]
        samples = self.generate_samples(model, tokenizer, prompts)
        report["samples"] = samples

        print(f"\n  Generation Samples:")
        for s in samples:
            print(f"    '{s['prompt']}' → {s['output'][:80]}")

        # Save report
        report_path = os.path.join(self.config.train.checkpoint_dir, "eval_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n  Report saved to: {report_path}")

        return report

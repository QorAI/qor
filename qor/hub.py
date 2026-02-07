"""
QOR — HuggingFace Integration
================================
Register QOR as a HuggingFace model so you can:
  - Push to HuggingFace Hub (share with others)
  - Load with AutoModel (standard HF interface)
  - Use with Gradio for a web UI
  - Deploy on HuggingFace Spaces (free hosting!)

Usage:
    # Push your trained model to HuggingFace
    python -m qor.hub push --repo your-name/qor-small

    # Load someone else's QOR model
    python -m qor.hub load --repo your-name/qor-small
"""

import os
import json
import shutil
import torch
from typing import Optional

from .config import QORConfig
from .model import QORModel
from .tokenizer import QORTokenizer


def export_for_hub(config: QORConfig,
                   checkpoint_path: str,
                   output_dir: str = "hub_export"):
    """
    Export QOR model in a format ready for HuggingFace Hub.

    Creates:
      hub_export/
      ├── config.json          # Model config
      ├── model.safetensors    # Model weights (safe format)
      ├── tokenizer.json       # Tokenizer
      ├── README.md            # Model card
      └── qor_model.py         # Model class (for custom loading)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    tokenizer = QORTokenizer()
    tokenizer.load(config.tokenizer.save_path)
    config.model.vocab_size = tokenizer.vocab_size

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = QORModel(config.model)
    model.load_state_dict(checkpoint["model_state"])

    # Save config
    model_config = {
        "model_type": "qor",
        "architectures": ["QORModel"],
        "d_model": config.model.d_model,
        "n_layers": config.model.n_layers,
        "n_heads": config.model.n_heads,
        "d_ff": config.model.d_ff,
        "vocab_size": config.model.vocab_size,
        "max_seq_len": config.model.max_seq_len,
        "cms_levels": config.model.cms_levels,
        "cms_fast_freq": config.model.cms_fast_freq,
        "cms_med_freq": config.model.cms_med_freq,
        "cms_slow_freq": config.model.cms_slow_freq,
        "self_mod_lr": config.model.self_mod_lr,
        "self_mod_decay": config.model.self_mod_decay,
        "n_parameters": model.n_params,
    }
    with open(os.path.join(output_dir, "config.json"), 'w') as f:
        json.dump(model_config, f, indent=2)

    # Save weights
    try:
        from safetensors.torch import save_file
        save_file(model.state_dict(), os.path.join(output_dir, "model.safetensors"))
        print("  Saved weights as safetensors")
    except ImportError:
        torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        print("  Saved weights as pytorch_model.bin (install safetensors for safer format)")

    # Copy tokenizer
    if os.path.exists(config.tokenizer.save_path):
        shutil.copy(config.tokenizer.save_path, os.path.join(output_dir, "tokenizer.json"))

    # Create model card
    card = f"""---
tags:
  - qor
  - nested-learning
  - continual-learning
license: apache-2.0
---

# QOR — The Qore Mind

A language model built with the QOR architecture featuring:
- **Self-Modifying Neurons**: Weights adapt during inference
- **Multi-Speed Memory (CMS)**: Fast thoughts, medium recall, deep knowledge
- **Surprise-Gated Learning**: Only learns from unexpected content

## Model Details
- Parameters: {model.n_params:,}
- Dimensions: {config.model.d_model}
- Layers: {config.model.n_layers}
- Attention Heads: {config.model.n_heads}
- CMS Levels: {config.model.cms_levels} (fast={config.model.cms_fast_freq}, med={config.model.cms_med_freq}, slow={config.model.cms_slow_freq})

## Usage

```python
from qor.config import QORConfig
from qor.model import QORModel
from qor.tokenizer import QORTokenizer

# Load
tokenizer = QORTokenizer()
tokenizer.load("tokenizer.json")

config = QORConfig.small()
config.model.vocab_size = tokenizer.vocab_size
model = QORModel(config.model)
# Load weights...

# Generate
ids = tokenizer.encode("Hello world")
output = model.generate(torch.tensor([ids]), max_new_tokens=100)
print(tokenizer.decode(output))
```

## Architecture
Based on the Nested Learning paradigm (Behrouz et al., NeurIPS 2025).
"""
    with open(os.path.join(output_dir, "README.md"), 'w') as f:
        f.write(card)

    print(f"\n  Exported to {output_dir}/")
    print(f"  Files: config.json, model weights, tokenizer.json, README.md")
    return output_dir


def push_to_hub(output_dir: str, repo_id: str, private: bool = False):
    """Push exported model to HuggingFace Hub."""
    try:
        from huggingface_hub import HfApi, login
    except ImportError:
        print("Install: pip install huggingface_hub")
        return

    print(f"\n  Pushing to HuggingFace Hub: {repo_id}")
    print(f"  You may need to login: huggingface-cli login")

    api = HfApi()
    api.create_repo(repo_id, exist_ok=True, private=private)
    api.upload_folder(folder_path=output_dir, repo_id=repo_id)

    print(f"\n  ✓ Published to: https://huggingface.co/{repo_id}")
    print(f"  Others can now download and use your QOR model!")


def load_from_hub(repo_id: str, local_dir: str = "downloaded_model"):
    """Download a QOR model from HuggingFace Hub."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Install: pip install huggingface_hub")
        return None

    print(f"Downloading {repo_id}...")
    path = snapshot_download(repo_id, local_dir=local_dir)
    print(f"  Downloaded to: {path}")
    return path


def create_gradio_app(config: QORConfig, checkpoint_path: str):
    """
    Create a Gradio web interface for QOR.
    Perfect for HuggingFace Spaces deployment.
    """
    try:
        import gradio as gr
    except ImportError:
        print("Install: pip install gradio")
        return

    from .serve import QORServer

    server = QORServer(config)
    server.load(checkpoint_path=checkpoint_path)

    def generate(prompt, max_tokens, temperature, top_k):
        result = server.generate(
            prompt=prompt,
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            top_k=int(top_k),
        )
        info = f"Tokens: {result['tokens_generated']} | Time: {result['time_seconds']}s | Speed: {result['tokens_per_second']} tok/s"
        return result['output'], info

    demo = gr.Interface(
        fn=generate,
        inputs=[
            gr.Textbox(label="Prompt", placeholder="Type your prompt here...", lines=3),
            gr.Slider(10, 500, value=200, step=10, label="Max Tokens"),
            gr.Slider(0.1, 2.0, value=0.8, step=0.1, label="Temperature"),
            gr.Slider(1, 100, value=50, step=1, label="Top-K"),
        ],
        outputs=[
            gr.Textbox(label="QOR Output", lines=10),
            gr.Textbox(label="Stats"),
        ],
        title="QOR — The Qore Mind",
        description=f"A {server.model.n_params/1e6:.1f}M parameter model with multi-speed memory and self-modifying neurons.",
        examples=[
            ["The most important thing about artificial intelligence is", 200, 0.8, 50],
            ["Once upon a time in a world where machines could think", 200, 0.9, 40],
        ],
    )

    return demo

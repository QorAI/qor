"""
QOR Export — Run QOR Everywhere
=================================
Export QOR model to different formats for deployment:

  - ONNX:        Run on any platform (Windows, Linux, Mac, mobile, web)
  - TorchScript:  Run without Python (C++, mobile)
  - Safetensors:  Safe weight format for sharing

Note: Self-modification (fast weights) is a training/inference feature
that requires PyTorch. Exported models run in "static" mode without
live weight adaptation. They still use the CMS architecture, just
without the runtime self-modification.
"""

import os
import torch
from typing import Optional

from .config import QORConfig
from .model import QORModel
from .tokenizer import QORTokenizer


def export_onnx(config: QORConfig, checkpoint_path: str,
                output_path: str = "qor_model.onnx"):
    """
    Export QOR to ONNX format.

    ONNX models can run on:
    - ONNX Runtime (Python, C++, C#, Java)
    - Windows ML
    - TensorRT (NVIDIA)
    - OpenVINO (Intel)
    - CoreML (Apple) via conversion
    - Web browsers (ONNX.js)
    - Android / iOS
    """
    try:
        import onnx
    except ImportError:
        print("Install: pip install onnx onnxruntime")
        return

    print(f"Exporting QOR to ONNX...")

    # Load model
    tokenizer = QORTokenizer()
    tokenizer.load(config.tokenizer.save_path)
    config.model.vocab_size = tokenizer.vocab_size

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = QORModel(config.model)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Create dummy input
    seq_len = 64  # Fixed for ONNX
    dummy_input = torch.randint(0, config.model.vocab_size, (1, seq_len))

    # Export
    torch.onnx.export(
        model,
        (dummy_input,),
        output_path,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    # Verify
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  ✓ Exported to: {output_path} ({size_mb:.1f} MB)")
    print(f"  ✓ ONNX model verified")
    print(f"\n  Run with ONNX Runtime:")
    print(f"    import onnxruntime as ort")
    print(f"    session = ort.InferenceSession('{output_path}')")
    print(f"    result = session.run(None, {{'input_ids': tokens}})")

    return output_path


def export_torchscript(config: QORConfig, checkpoint_path: str,
                        output_path: str = "qor_model.pt"):
    """
    Export to TorchScript for C++ deployment.

    TorchScript models can run:
    - Without Python (pure C++)
    - On mobile (Android/iOS via PyTorch Mobile)
    - On edge devices
    - In production C++ services
    """
    print(f"Exporting QOR to TorchScript...")

    # Load model
    tokenizer = QORTokenizer()
    tokenizer.load(config.tokenizer.save_path)
    config.model.vocab_size = tokenizer.vocab_size

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = QORModel(config.model)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Trace the model
    dummy_input = torch.randint(0, config.model.vocab_size, (1, 64))

    with torch.no_grad():
        traced = torch.jit.trace(model, (dummy_input,),
                                  check_trace=False)  # Custom ops may not trace perfectly

    traced.save(output_path)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  ✓ Exported to: {output_path} ({size_mb:.1f} MB)")
    print(f"\n  Run without Python:")
    print(f"    model = torch.jit.load('{output_path}')")
    print(f"    output = model(input_ids)")

    return output_path


def quantize_model(config: QORConfig, checkpoint_path: str,
                    output_path: str = "qor_model_int8.pt",
                    quantize_type: str = "dynamic"):
    """
    Quantize model to INT8 for faster inference and smaller size.

    Reduces model size by ~4x and speeds up CPU inference by ~2-3x.
    """
    print(f"Quantizing QOR to INT8 ({quantize_type})...")

    # Load model
    tokenizer = QORTokenizer()
    tokenizer.load(config.tokenizer.save_path)
    config.model.vocab_size = tokenizer.vocab_size

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = QORModel(config.model)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    original_size = sum(p.numel() * p.element_size() for p in model.parameters())

    if quantize_type == "dynamic":
        # Dynamic quantization — easiest, works on CPU
        quantized = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},  # Quantize all linear layers
            dtype=torch.qint8,
        )
    else:
        print(f"  Unknown quantize type: {quantize_type}")
        return

    # Save
    torch.save({
        "model_state": quantized.state_dict(),
        "config": vars(config.model),
        "quantized": True,
        "quantize_type": quantize_type,
    }, output_path)

    new_size = os.path.getsize(output_path)
    print(f"  Original: {original_size / 1e6:.1f} MB")
    print(f"  Quantized: {new_size / 1e6:.1f} MB")
    print(f"  Reduction: {(1 - new_size / original_size) * 100:.0f}%")
    print(f"  ✓ Saved to: {output_path}")

    return output_path


def print_deployment_guide():
    """Print a guide for deploying QOR on various platforms."""
    guide = """
╔══════════════════════════════════════════════════════════════╗
║              QOR Deployment Guide                           ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  LOCAL (your PC)                                             ║
║  ─────────────────                                           ║
║  • PyTorch direct:  python -m qor serve                      ║
║  • Quantized (2x faster): export → INT8 → serve             ║
║                                                              ║
║  WEB UI (free hosting)                                       ║
║  ─────────────────────                                       ║
║  • HuggingFace Spaces:                                       ║
║    1. python -m qor.hub export                               ║
║    2. Push to HF Hub                                         ║
║    3. Create Space with Gradio                               ║
║    → Free GPU available!                                     ║
║                                                              ║
║  API SERVER                                                  ║
║  ──────────                                                  ║
║  • Flask:    python -m qor serve                             ║
║  • FastAPI:  python -m qor serve --fastapi                   ║
║  • Docker:   See Dockerfile example below                    ║
║                                                              ║
║  CLOUD GPU                                                   ║
║  ─────────                                                   ║
║  • RunPod:    Upload code → rent GPU → run                   ║
║  • Vast.ai:   Cheapest GPU rental                            ║
║  • Lambda:    Good for training + serving                    ║
║  • AWS/GCP:   SageMaker / Vertex AI endpoints                ║
║                                                              ║
║  MOBILE / EDGE                                               ║
║  ─────────────                                               ║
║  • Export to ONNX → ONNX Runtime Mobile                      ║
║  • Export to TorchScript → PyTorch Mobile                    ║
║  • Quantize to INT8 for smaller + faster                     ║
║                                                              ║
║  BROWSER                                                     ║
║  ────────                                                    ║
║  • Export to ONNX → ONNX.js / Transformers.js                ║
║  • Small models (< 50M) can run in-browser                   ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝

Docker Example:
─────────────────
    FROM python:3.12-slim
    WORKDIR /app
    COPY qor/ ./qor/
    COPY checkpoints/ ./checkpoints/
    COPY tokenizer.json .
    RUN pip install torch flask flask-cors tokenizers
    EXPOSE 8000
    CMD ["python", "-m", "qor", "serve"]

Build and run:
    docker build -t qor-server .
    docker run -p 8000:8000 qor-server
"""
    print(guide)

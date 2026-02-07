"""
QOR API Server — Serve the Qore Mind
======================================
REST API for generating text with a trained QOR model.

Endpoints:
  POST /generate       — Generate text from a prompt
  POST /learn          — Feed new text for continual learning
  GET  /health         — Health check
  GET  /info           — Model information

Runs with Flask (simple) or FastAPI (production).
"""

import os
import json
import time
import torch
from typing import Optional

from .config import QORConfig
from .model import QORModel
from .tokenizer import QORTokenizer


class QORServer:
    """Serves QOR model via REST API."""

    def __init__(self, config: QORConfig):
        self.config = config
        self.device = config.get_device()
        self.model = None
        self.tokenizer = None
        self.request_count = 0
        self.start_time = None

    def load(self, checkpoint_path: Optional[str] = None,
             tokenizer_path: Optional[str] = None):
        """Load model and tokenizer."""
        ckpt_path = checkpoint_path or self.config.serve.model_path
        tok_path = tokenizer_path or self.config.serve.tokenizer_path

        print(f"Loading QOR model from {ckpt_path}")

        # Load tokenizer
        self.tokenizer = QORTokenizer()
        self.tokenizer.load(tok_path)
        self.config.model.vocab_size = self.tokenizer.vocab_size

        # Load model
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model = QORModel(self.config.model).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

        print(f"  Parameters: {self.model.n_params:,}")
        print(f"  Device: {self.device}")
        print(f"  Ready to serve!")

    def generate(self, prompt: str,
                 max_tokens: int = None,
                 temperature: float = None,
                 top_k: int = None,
                 top_p: float = None) -> dict:
        """Generate text from a prompt."""
        if self.model is None:
            return {"error": "Model not loaded"}

        max_tokens = max_tokens or self.config.serve.max_new_tokens
        temperature = temperature or self.config.serve.temperature
        top_k = top_k or self.config.serve.top_k
        top_p = top_p or self.config.serve.top_p

        # Encode
        ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = torch.tensor([ids], device=self.device)

        # Generate
        start = time.time()
        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stop_tokens=[self.tokenizer.eos_id],
        )
        elapsed = time.time() - start

        # Decode
        output_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        new_tokens = len(output_ids) - len(ids)

        self.request_count += 1

        return {
            "prompt": prompt,
            "output": output_text,
            "generated_text": output_text[len(prompt):] if output_text.startswith(prompt) else output_text,
            "tokens_generated": new_tokens,
            "time_seconds": round(elapsed, 3),
            "tokens_per_second": round(new_tokens / max(elapsed, 0.001), 1),
        }

    def get_info(self) -> dict:
        """Return model information."""
        return {
            "name": "QOR — The Qore Mind",
            "parameters": self.model.n_params if self.model else 0,
            "vocab_size": self.tokenizer.vocab_size if self.tokenizer else 0,
            "device": str(self.device),
            "requests_served": self.request_count,
            "cms_levels": self.config.model.cms_levels,
            "self_mod_enabled": self.config.model.self_mod_lr > 0,
        }


def run_flask_server(config: QORConfig):
    """Run a simple Flask API server."""
    try:
        from flask import Flask, request, jsonify
        from flask_cors import CORS
    except ImportError:
        print("Install Flask: pip install flask flask-cors")
        return

    app = Flask(__name__)
    CORS(app, origins=config.serve.cors_origins)

    server = QORServer(config)
    server.load()
    server.start_time = time.time()

    @app.route('/generate', methods=['POST'])
    def generate():
        data = request.get_json()
        if not data or 'prompt' not in data:
            return jsonify({"error": "Missing 'prompt' in request body"}), 400

        result = server.generate(
            prompt=data['prompt'],
            max_tokens=data.get('max_tokens'),
            temperature=data.get('temperature'),
            top_k=data.get('top_k'),
            top_p=data.get('top_p'),
        )
        return jsonify(result)

    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({"status": "ok", "model_loaded": server.model is not None})

    @app.route('/info', methods=['GET'])
    def info():
        return jsonify(server.get_info())

    print(f"\n  QOR API Server running at http://{config.serve.host}:{config.serve.port}")
    print(f"  Endpoints:")
    print(f"    POST /generate  — Generate text")
    print(f"    GET  /health    — Health check")
    print(f"    GET  /info      — Model info")
    print(f"\n  Example:")
    print(f'    curl -X POST http://localhost:{config.serve.port}/generate \\')
    print(f'      -H "Content-Type: application/json" \\')
    print(f'      -d \'{{"prompt": "Once upon a time"}}\'')
    print()

    app.run(host=config.serve.host, port=config.serve.port, debug=False)


def run_fastapi_server(config: QORConfig):
    """Run a production FastAPI server."""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel
        import uvicorn
    except ImportError:
        print("Install FastAPI: pip install fastapi uvicorn")
        return

    app = FastAPI(title="QOR — The Qore Mind", version="1.0.0")
    app.add_middleware(CORSMiddleware, allow_origins=config.serve.cors_origins,
                       allow_methods=["*"], allow_headers=["*"])

    server = QORServer(config)
    server.load()

    class GenerateRequest(BaseModel):
        prompt: str
        max_tokens: Optional[int] = None
        temperature: Optional[float] = None
        top_k: Optional[int] = None
        top_p: Optional[float] = None

    @app.post("/generate")
    async def generate(req: GenerateRequest):
        result = server.generate(
            prompt=req.prompt, max_tokens=req.max_tokens,
            temperature=req.temperature, top_k=req.top_k, top_p=req.top_p
        )
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        return result

    @app.get("/health")
    async def health():
        return {"status": "ok", "model_loaded": server.model is not None}

    @app.get("/info")
    async def info():
        return server.get_info()

    print(f"\n  QOR API (FastAPI) at http://{config.serve.host}:{config.serve.port}")
    print(f"  Docs at http://{config.serve.host}:{config.serve.port}/docs\n")

    uvicorn.run(app, host=config.serve.host, port=config.serve.port)

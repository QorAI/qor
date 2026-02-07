"""
QOR — The Qore Mind
====================
Main entry point. All commands in one place.

SETUP:
    python -m qor setup                     # Create folders and sample data
    python -m qor tokenizer                  # Train tokenizer on your data

TRAINING:
    python -m qor train                      # Train the model
    python -m qor train --size medium        # Train a bigger model
    python -m qor train --resume             # Resume from checkpoint

EVALUATION:
    python -m qor eval                       # Evaluate the model
    python -m qor test                       # Run the Mind Test

GENERATION:
    python -m qor chat                       # Interactive chat
    python -m qor generate "Your prompt"     # One-off generation

CONTINUAL LEARNING:
    python -m qor learn                      # Learn from files in learn/ folder
    python -m qor watch                      # Auto-learn new files (live)

SERVING:
    python -m qor serve                      # Start API server
    python -m qor serve --fastapi            # Production API server
"""

import argparse
import os
import sys


def cmd_setup(args):
    """Create project folders and sample training data."""
    dirs = ["data", "learn", "checkpoints"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"  ✓ Created {d}/")

    # Create sample training data
    sample_path = os.path.join("data", "sample.txt")
    if not os.path.exists(sample_path):
        sample_text = """The sun rises in the east and sets in the west. Every morning brings new light to the world.

Cats are curious animals that love to explore. They have four legs, sharp claws, and excellent night vision. A cat's purr is one of the most soothing sounds in nature.

Dogs are loyal companions that have lived alongside humans for thousands of years. They come in many breeds, from tiny Chihuahuas to giant Great Danes.

Water covers about seventy percent of the Earth's surface. The oceans are home to millions of species, from tiny plankton to massive blue whales.

The human brain contains approximately eighty-six billion neurons. Each neuron can form thousands of connections with other neurons, creating an incredibly complex network.

Mathematics is the language of the universe. Two plus two equals four. The square root of nine is three. Pi is approximately three point one four one five nine.

Music has the power to move emotions. A melody can make us feel joy, sadness, excitement, or peace. Rhythm connects us to something primal and universal.

Trees produce oxygen through photosynthesis. They absorb carbon dioxide and release oxygen, making life possible for animals and humans. A single large tree can provide oxygen for up to four people.

The speed of light is approximately three hundred thousand kilometers per second. Nothing in the universe can travel faster than light. Einstein's theory of relativity depends on this fundamental constant.

Cooking is both an art and a science. Heat transforms ingredients through chemical reactions. The Maillard reaction creates the brown, flavorful crust on bread and meat.
"""
        with open(sample_path, 'w') as f:
            f.write(sample_text * 20)  # Repeat for enough training data
        print(f"  ✓ Created {sample_path} (sample training data)")
    else:
        print(f"  ✓ {sample_path} already exists")

    print(f"\n  Project ready! Next steps:")
    print(f"    1. Add your own .txt files to data/")
    print(f"    2. python -m qor tokenizer")
    print(f"    3. python -m qor train")


def cmd_tokenizer(args):
    """Train a BPE tokenizer on the data."""
    from qor.config import QORConfig, PRESETS
    from qor.tokenizer import QORTokenizer

    config = PRESETS.get(args.size, PRESETS["small"])()
    tokenizer = QORTokenizer()
    tokenizer.train(
        data_dir=config.train.data_dir,
        vocab_size=config.tokenizer.vocab_size,
        save_path=config.tokenizer.save_path,
    )


def cmd_train(args):
    """Train the QOR model."""
    from qor.config import QORConfig, PRESETS
    from qor.train import Trainer

    config = PRESETS.get(args.size, PRESETS["small"])()

    if args.steps:
        config.train.max_steps = args.steps
    if args.device:
        config.train.device = args.device
    if args.batch_size:
        config.train.batch_size = args.batch_size

    # Save config for reproducibility
    os.makedirs(config.train.checkpoint_dir, exist_ok=True)
    config.save(os.path.join(config.train.checkpoint_dir, "config.json"))

    trainer = Trainer(config)
    resume = os.path.join(config.train.checkpoint_dir, "final_model.pt") if args.resume else None
    trainer.train(resume_from=resume)


def cmd_eval(args):
    """Evaluate the model."""
    from qor.config import QORConfig, PRESETS
    from qor.evaluate import Evaluator

    config = PRESETS.get(args.size, PRESETS["small"])()
    evaluator = Evaluator(config)

    ckpt = args.checkpoint or os.path.join(config.train.checkpoint_dir, "best_model.pt")
    eval_text = args.eval_file

    evaluator.full_report(ckpt, eval_text)


def cmd_test(args):
    """Run the Mind Test (continual learning test)."""
    from qor.config import QORConfig, PRESETS
    from qor.evaluate import Evaluator
    from qor.tokenizer import QORTokenizer

    config = PRESETS.get(args.size, PRESETS["small"])()

    # Need a tokenizer for the test
    tokenizer = QORTokenizer()
    tok_path = config.tokenizer.save_path
    if os.path.exists(tok_path):
        tokenizer.load(tok_path)
        config.model.vocab_size = tokenizer.vocab_size
    else:
        print("Train a tokenizer first: python -m qor tokenizer")
        print("Or run setup first: python -m qor setup && python -m qor tokenizer")
        return

    if args.device:
        config.train.device = args.device

    evaluator = Evaluator(config)
    evaluator.continual_learning_test(tokenizer)


def cmd_chat(args):
    """Interactive chat with the model."""
    from qor.config import QORConfig, PRESETS
    from qor.serve import QORServer

    config = PRESETS.get(args.size, PRESETS["small"])()
    if args.device:
        config.train.device = args.device

    ckpt = args.checkpoint or os.path.join(config.train.checkpoint_dir, "best_model.pt")

    server = QORServer(config)
    server.load(checkpoint_path=ckpt)

    print(f"\n{'='*60}")
    print(f"  QOR Chat — The Qore Mind")
    print(f"  Type 'quit' to exit, 'reset' to clear memory")
    print(f"{'='*60}\n")

    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not prompt:
            continue
        if prompt.lower() == 'quit':
            break
        if prompt.lower() == 'reset':
            server.model.reset_fast_weights()
            print("  [Memory reset]\n")
            continue

        result = server.generate(prompt, temperature=args.temperature or 0.8)
        print(f"QOR: {result['generated_text']}")
        print(f"  [{result['tokens_generated']} tokens, {result['time_seconds']}s]\n")


def cmd_generate(args):
    """Generate text from a prompt."""
    from qor.config import QORConfig, PRESETS
    from qor.serve import QORServer

    config = PRESETS.get(args.size, PRESETS["small"])()
    if args.device:
        config.train.device = args.device

    ckpt = args.checkpoint or os.path.join(config.train.checkpoint_dir, "best_model.pt")

    server = QORServer(config)
    server.load(checkpoint_path=ckpt)

    result = server.generate(args.prompt, temperature=args.temperature or 0.8,
                              max_tokens=args.max_tokens or 200)
    print(result['output'])


def cmd_learn(args):
    """Learn from new files (continual learning)."""
    from qor.config import QORConfig, PRESETS
    from qor.continual import ContinualLearner

    config = PRESETS.get(args.size, PRESETS["small"])()
    if args.device:
        config.train.device = args.device

    ckpt = args.checkpoint or os.path.join(config.train.checkpoint_dir, "best_model.pt")

    learner = ContinualLearner(config)
    learner.load(ckpt)
    learner.learn_folder(args.folder)


def cmd_watch(args):
    """Watch folder and auto-learn new files."""
    from qor.config import QORConfig, PRESETS
    from qor.continual import ContinualLearner

    config = PRESETS.get(args.size, PRESETS["small"])()
    if args.device:
        config.train.device = args.device

    ckpt = args.checkpoint or os.path.join(config.train.checkpoint_dir, "best_model.pt")

    learner = ContinualLearner(config)
    learner.load(ckpt)
    learner.watch(args.folder, interval=args.interval or 10)


def cmd_serve(args):
    """Start the API server."""
    from qor.config import QORConfig, PRESETS

    config = PRESETS.get(args.size, PRESETS["small"])()
    if args.port:
        config.serve.port = args.port

    if args.fastapi:
        from qor.serve import run_fastapi_server
        run_fastapi_server(config)
    else:
        from qor.serve import run_flask_server
        run_flask_server(config)


def main():
    parser = argparse.ArgumentParser(
        description="QOR — The Qore Mind",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m qor setup                    # Create project structure
  python -m qor tokenizer                # Train tokenizer
  python -m qor train                    # Train small model
  python -m qor train --size medium      # Train medium model
  python -m qor test                     # Run the Mind Test
  python -m qor chat                     # Chat with model
  python -m qor serve                    # Start API server
  python -m qor learn                    # Learn new files
  python -m qor watch                    # Auto-learn (live)
        """,
    )

    sub = parser.add_subparsers(dest="command", help="Command to run")

    # === Setup ===
    p = sub.add_parser("setup", help="Create project folders and sample data")

    # === Tokenizer ===
    p = sub.add_parser("tokenizer", help="Train BPE tokenizer")
    p.add_argument("--size", default="small", choices=["small", "medium", "large"])

    # === Train ===
    p = sub.add_parser("train", help="Train the model")
    p.add_argument("--size", default="small", choices=["small", "medium", "large"])
    p.add_argument("--steps", type=int, help="Training steps")
    p.add_argument("--device", type=str, help="Device: cpu/cuda/mps")
    p.add_argument("--batch-size", type=int, help="Batch size")
    p.add_argument("--resume", action="store_true", help="Resume from last checkpoint")

    # === Eval ===
    p = sub.add_parser("eval", help="Evaluate the model")
    p.add_argument("--size", default="small", choices=["small", "medium", "large"])
    p.add_argument("--checkpoint", type=str, help="Checkpoint path")
    p.add_argument("--eval-file", type=str, help="Text file for perplexity")

    # === Test ===
    p = sub.add_parser("test", help="Run the Mind Test (continual learning)")
    p.add_argument("--size", default="small", choices=["small", "medium", "large"])
    p.add_argument("--device", type=str, help="Device")

    # === Chat ===
    p = sub.add_parser("chat", help="Interactive chat")
    p.add_argument("--size", default="small", choices=["small", "medium", "large"])
    p.add_argument("--device", type=str, help="Device")
    p.add_argument("--checkpoint", type=str, help="Checkpoint path")
    p.add_argument("--temperature", type=float, help="Sampling temperature")

    # === Generate ===
    p = sub.add_parser("generate", help="Generate text")
    p.add_argument("prompt", type=str, help="Text prompt")
    p.add_argument("--size", default="small", choices=["small", "medium", "large"])
    p.add_argument("--device", type=str, help="Device")
    p.add_argument("--checkpoint", type=str, help="Checkpoint path")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max-tokens", type=int, default=200)

    # === Learn ===
    p = sub.add_parser("learn", help="Learn from new text files")
    p.add_argument("--size", default="small", choices=["small", "medium", "large"])
    p.add_argument("--device", type=str, help="Device")
    p.add_argument("--checkpoint", type=str, help="Checkpoint path")
    p.add_argument("--folder", type=str, default="learn", help="Folder with new files")

    # === Watch ===
    p = sub.add_parser("watch", help="Auto-learn new files (live)")
    p.add_argument("--size", default="small", choices=["small", "medium", "large"])
    p.add_argument("--device", type=str, help="Device")
    p.add_argument("--checkpoint", type=str, help="Checkpoint path")
    p.add_argument("--folder", type=str, default="learn", help="Folder to watch")
    p.add_argument("--interval", type=int, default=10, help="Check interval (seconds)")

    # === Serve ===
    p = sub.add_parser("serve", help="Start API server")
    p.add_argument("--size", default="small", choices=["small", "medium", "large"])
    p.add_argument("--port", type=int, help="Server port")
    p.add_argument("--fastapi", action="store_true", help="Use FastAPI (production)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        print("\n  Start with: python -m qor setup")
        return

    commands = {
        "setup": cmd_setup,
        "tokenizer": cmd_tokenizer,
        "train": cmd_train,
        "eval": cmd_eval,
        "test": cmd_test,
        "chat": cmd_chat,
        "generate": cmd_generate,
        "learn": cmd_learn,
        "watch": cmd_watch,
        "serve": cmd_serve,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()

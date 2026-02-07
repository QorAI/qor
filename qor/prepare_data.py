"""
QOR Data Preparation — Get Data and Prepare for Training
==========================================================
Download free datasets and prepare them for QOR training.

Usage:
    python prepare_data.py --source wikipedia      # Download Wikipedia
    python prepare_data.py --source sample          # Generate sample data
    python prepare_data.py --source custom           # Use your own .txt files
    python prepare_data.py --source all              # Wikipedia + books + sample

After running this, your data/ folder will have .txt files ready for training.
Then just run:
    python -m qor tokenizer
    python -m qor train
"""

import os
import argparse


def download_wikipedia(output_dir="data", max_articles=5000, max_chars=10_000_000):
    """
    Download Wikipedia articles.
    ~10M chars = enough for a small model.
    ~100M chars = good for a medium model.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Install datasets: pip install datasets")
        return

    print(f"Downloading Wikipedia (up to {max_articles} articles)...")
    os.makedirs(output_dir, exist_ok=True)

    ds = load_dataset("wikipedia", "20220301.en", split="train",
                       streaming=True, trust_remote_code=True)

    text = ""
    count = 0
    for article in ds:
        # Skip very short articles
        if len(article["text"]) < 500:
            continue

        text += article["text"] + "\n\n"
        count += 1

        if count >= max_articles or len(text) >= max_chars:
            break

        if count % 500 == 0:
            print(f"  {count} articles, {len(text):,} chars...")

    output_path = os.path.join(output_dir, "wikipedia.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)

    print(f"  Saved {count} articles ({len(text):,} chars) to {output_path}")


def download_tiny_stories(output_dir="data", max_stories=10000):
    """
    Download TinyStories — simple short stories, great for small models.
    This dataset was specifically designed to train very small language models.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Install datasets: pip install datasets")
        return

    print(f"Downloading TinyStories (up to {max_stories} stories)...")
    os.makedirs(output_dir, exist_ok=True)

    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

    text = ""
    count = 0
    for story in ds:
        text += story["text"] + "\n\n"
        count += 1
        if count >= max_stories:
            break
        if count % 2000 == 0:
            print(f"  {count} stories, {len(text):,} chars...")

    output_path = os.path.join(output_dir, "tiny_stories.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)

    print(f"  Saved {count} stories ({len(text):,} chars) to {output_path}")


def download_openwebtext(output_dir="data", max_docs=5000):
    """
    Download OpenWebText — high quality web text (same data source as GPT-2).
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Install datasets: pip install datasets")
        return

    print(f"Downloading OpenWebText (up to {max_docs} documents)...")
    os.makedirs(output_dir, exist_ok=True)

    ds = load_dataset("openwebtext", split="train", streaming=True,
                       trust_remote_code=True)

    text = ""
    count = 0
    for doc in ds:
        if len(doc["text"]) < 200:
            continue
        text += doc["text"] + "\n\n"
        count += 1
        if count >= max_docs:
            break
        if count % 1000 == 0:
            print(f"  {count} documents, {len(text):,} chars...")

    output_path = os.path.join(output_dir, "openwebtext.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)

    print(f"  Saved {count} documents ({len(text):,} chars) to {output_path}")


def generate_sample_data(output_dir="data"):
    """Generate rich sample training data — good enough to see QOR working."""
    os.makedirs(output_dir, exist_ok=True)

    text = """The sun rises in the east and sets in the west. This simple fact has guided travelers and explorers for thousands of years. Ancient civilizations built their temples facing east to greet the morning sun. The Earth rotates on its axis once every twenty-four hours, creating the cycle of day and night that all life depends upon.

Water is essential for all known forms of life. A water molecule consists of two hydrogen atoms bonded to one oxygen atom. Water covers approximately seventy-one percent of Earth's surface, with oceans holding about ninety-six percent of all Earth's water. Fresh water is found in glaciers, ice caps, groundwater, rivers, and lakes. The human body is about sixty percent water by weight.

The human brain is the most complex organ in the known universe. It contains roughly eighty-six billion neurons, each connected to thousands of others through synapses. This vast network enables everything from breathing and heartbeat to abstract thought, creativity, and consciousness. The brain consumes about twenty percent of the body's energy despite being only two percent of its weight.

Mathematics is often called the language of nature. The number pi, approximately three point one four one five nine, appears throughout mathematics, physics, and engineering. The Fibonacci sequence, where each number is the sum of the two preceding ones, appears in the arrangement of leaves on stems, the spirals of shells, and the branching of trees.

Cats are one of the most popular pets worldwide. They were first domesticated in the Near East around ten thousand years ago. Cats have excellent night vision, flexible bodies, and sharp retractable claws. A group of cats is called a clowder. Cats spend approximately seventy percent of their lives sleeping.

Dogs have been human companions for at least fifteen thousand years. They were the first domesticated animal. Dogs can understand up to two hundred and fifty words and gestures, can count up to five, and can perform simple mathematical calculations. A dog's sense of smell is roughly forty times better than a human's.

The ocean covers more than seventy percent of the Earth's surface. The deepest point in the ocean is the Mariana Trench, which reaches a depth of nearly eleven kilometers. More people have walked on the Moon than have visited the deepest parts of the ocean. The ocean produces over fifty percent of the world's oxygen through phytoplankton.

Trees are vital to life on Earth. Through photosynthesis, they convert carbon dioxide into oxygen. A single large tree can provide a day's supply of oxygen for up to four people. Trees also help prevent soil erosion, provide habitat for wildlife, and help regulate the water cycle.

Music has been part of human culture for tens of thousands of years. The oldest known musical instruments are bone flutes dating back over forty thousand years. Music activates more parts of the brain than any other human activity. Studies have shown that learning to play a musical instrument can improve memory, spatial reasoning, and language skills.

Cooking transforms raw ingredients into nourishing meals through the application of heat and chemical reactions. The Maillard reaction, which occurs when proteins and sugars are heated together, creates the brown color and complex flavors in bread crust, seared meat, and roasted coffee. Fermentation is another fundamental cooking process that gives us bread, cheese, yogurt, wine, and beer.

The speed of light in a vacuum is approximately three hundred thousand kilometers per second. Nothing with mass can travel at or faster than the speed of light. Light from the Sun takes about eight minutes to reach Earth. The nearest star system to our Sun is Alpha Centauri, and its light takes over four years to reach us.

Electricity powers modern civilization. It flows through conductors as a stream of electrons. The human body itself runs on electrical signals, with neurons communicating through electrical impulses. Lightning is a natural form of electricity, with a single bolt containing enough energy to power a small town for a day.

The immune system is the body's defense against infection and disease. White blood cells patrol the body looking for foreign invaders like bacteria and viruses. The system has two main parts: the innate immune system, which provides immediate but general defense, and the adaptive immune system, which learns to recognize specific threats and builds lasting immunity.

Climate describes the long-term patterns of temperature, humidity, wind, and precipitation in a region. Weather is the day-to-day state of the atmosphere. Climate change refers to long-term shifts in temperatures and weather patterns. Since the industrial revolution, human activities have been the main driver of climate change, primarily through burning fossil fuels.

Programming is the art of giving instructions to computers. The first programmer is widely considered to be Ada Lovelace, who wrote instructions for Charles Babbage's Analytical Engine in the eighteen forties. Today, there are hundreds of programming languages, from Python and JavaScript to Rust and Swift. At their core, all programs are sequences of simple operations that computers execute incredibly quickly.
"""
    # Repeat to make enough training data
    full_text = text * 30

    output_path = os.path.join(output_dir, "sample_knowledge.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_text)

    print(f"  Generated sample data ({len(full_text):,} chars) to {output_path}")


def generate_qa_data(output_dir="data"):
    """Generate question-answer pairs for training."""
    os.makedirs(output_dir, exist_ok=True)

    qa_pairs = """Question: What color is the sky?
Answer: The sky appears blue during the day because of the way Earth's atmosphere scatters sunlight. Shorter blue wavelengths scatter more than other colors, making the sky look blue.

Question: How many legs does a cat have?
Answer: A cat has four legs. Cats are quadrupeds, meaning they walk on four limbs. Their legs are powerful and flexible, allowing them to jump up to six times their body length.

Question: What is water made of?
Answer: Water is made of hydrogen and oxygen. Each water molecule contains two hydrogen atoms bonded to one oxygen atom, which is why its chemical formula is H2O.

Question: What is the capital of France?
Answer: The capital of France is Paris. Paris is located in northern France on the river Seine. It is the largest city in France with a population of over two million people.

Question: How does photosynthesis work?
Answer: Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen. This happens primarily in the leaves, where chlorophyll captures light energy.

Question: What is gravity?
Answer: Gravity is a fundamental force of nature that attracts objects with mass toward each other. On Earth, gravity gives objects weight and causes them to fall toward the ground at approximately 9.8 meters per second squared.

Question: Why is exercise important?
Answer: Exercise is important because it strengthens the heart, muscles, and bones. Regular physical activity improves mental health, boosts energy levels, helps maintain a healthy weight, and reduces the risk of chronic diseases.

Question: What causes rain?
Answer: Rain is caused by the water cycle. The sun heats water in oceans and lakes, causing it to evaporate. The water vapor rises, cools, and condenses into tiny droplets that form clouds. When these droplets combine and become heavy enough, they fall as rain.

Question: How do computers work?
Answer: Computers work by processing information using electrical signals. At the most basic level, they use transistors that can be either on or off, representing ones and zeros. These binary digits are combined to represent numbers, text, images, and instructions that the computer executes billions of times per second.

Question: What is DNA?
Answer: DNA stands for deoxyribonucleic acid. It is the molecule that carries the genetic instructions for the development, functioning, growth, and reproduction of all known living organisms. DNA is structured as a double helix and is found in nearly every cell of the body.
"""
    full_text = qa_pairs * 50

    output_path = os.path.join(output_dir, "qa_knowledge.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_text)

    print(f"  Generated Q&A data ({len(full_text):,} chars) to {output_path}")


def prepare_custom_data(input_dir="my_data", output_dir="data"):
    """
    Copy your own text files into the training data folder.
    Put your .txt files in my_data/ and run this.
    """
    import glob
    import shutil

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(input_dir):
        os.makedirs(input_dir, exist_ok=True)
        print(f"\n  Created '{input_dir}/' folder.")
        print(f"  Put your .txt files there and run this command again!")
        print(f"\n  You can put ANY text files:")
        print(f"    - Books, articles, blog posts")
        print(f"    - Company documents, manuals")
        print(f"    - Chat logs, emails, notes")
        print(f"    - Code, documentation")
        print(f"    - Anything you want QOR to learn!")
        return

    files = glob.glob(os.path.join(input_dir, "**", "*.txt"), recursive=True)
    if not files:
        print(f"  No .txt files found in '{input_dir}/'")
        return

    for fpath in files:
        dest = os.path.join(output_dir, os.path.basename(fpath))
        shutil.copy2(fpath, dest)
        size = os.path.getsize(fpath)
        print(f"  Copied: {os.path.basename(fpath)} ({size:,} bytes)")

    print(f"\n  Copied {len(files)} files to {output_dir}/")


def show_data_stats(data_dir="data"):
    """Show what's in the data folder."""
    import glob

    files = glob.glob(os.path.join(data_dir, "**", "*.txt"), recursive=True)
    if not files:
        print(f"\n  No data files found in '{data_dir}/'")
        print(f"  Run: python prepare_data.py --source sample")
        return

    total_chars = 0
    total_words = 0

    print(f"\n  {'File':<30} {'Size':>10} {'Words':>12}")
    print(f"  {'-'*30} {'-'*10} {'-'*12}")

    for fpath in sorted(files):
        with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
        chars = len(text)
        words = len(text.split())
        total_chars += chars
        total_words += words
        name = os.path.basename(fpath)
        print(f"  {name:<30} {chars:>10,} {words:>12,}")

    print(f"  {'-'*30} {'-'*10} {'-'*12}")
    print(f"  {'TOTAL':<30} {total_chars:>10,} {total_words:>12,}")

    # Estimate training quality
    print(f"\n  Estimated training quality:")
    if total_chars < 100_000:
        print(f"  ⚠️  Very small — will learn basic patterns only")
        print(f"     Add more data for better results")
    elif total_chars < 1_000_000:
        print(f"  ◆  Small — good for testing and simple tasks")
    elif total_chars < 10_000_000:
        print(f"  ◆  Medium — good for a focused domain model")
    elif total_chars < 100_000_000:
        print(f"  ✅ Large — good for general knowledge")
    else:
        print(f"  ✅ Very large — excellent for broad knowledge")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QOR Data Preparation")
    parser.add_argument("--source", type=str, default="sample",
                        choices=["sample", "wikipedia", "tinystories",
                                 "openwebtext", "custom", "all", "stats"],
                        help="Data source to download/prepare")
    parser.add_argument("--max-articles", type=int, default=5000,
                        help="Max articles for Wikipedia")
    parser.add_argument("--max-chars", type=int, default=10_000_000,
                        help="Max characters to download")
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"  QOR Data Preparation")
    print(f"{'='*50}\n")

    if args.source == "stats":
        show_data_stats()
    elif args.source == "sample":
        generate_sample_data()
        generate_qa_data()
        show_data_stats()
    elif args.source == "wikipedia":
        download_wikipedia(max_articles=args.max_articles, max_chars=args.max_chars)
        show_data_stats()
    elif args.source == "tinystories":
        download_tiny_stories()
        show_data_stats()
    elif args.source == "openwebtext":
        download_openwebtext()
        show_data_stats()
    elif args.source == "custom":
        prepare_custom_data()
        show_data_stats()
    elif args.source == "all":
        generate_sample_data()
        generate_qa_data()
        download_wikipedia(max_articles=args.max_articles, max_chars=args.max_chars)
        download_tiny_stories()
        show_data_stats()

    print(f"\n  Next steps:")
    print(f"    1. python -m qor tokenizer     # Train tokenizer on this data")
    print(f"    2. python -m qor train          # Train QOR model")
    print(f"    3. python -m qor chat           # Talk to your model")
    print()

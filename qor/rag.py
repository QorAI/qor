"""
QOR RAG — Retrieval-Augmented Generation
==========================================
Give QOR access to documents it hasn't been trained on.

How it works:
  1. You add documents to the knowledge base (any .txt files)
  2. User asks a question
  3. RAG searches for relevant chunks
  4. Relevant chunks are pasted into the prompt
  5. QOR reads the chunks and generates an answer

This is the "external memory" that complements QOR's
internal multi-speed memory (CMS).

Think of it like this:
  - CMS (internal) = knowledge the mind has absorbed (training)
  - RAG (external) = a reference book the mind can look things up in

Usage:
    from qor.rag import QORRag
    
    rag = QORRag()
    rag.add_folder("documents/")
    
    answer = rag.query("What is the return policy?", model, tokenizer)
"""

import os
import json
import math
import glob
from typing import List, Optional, Tuple
from collections import Counter


class SimpleVectorStore:
    """
    Dead-simple vector similarity search.
    No dependencies — just TF-IDF with cosine similarity.
    
    For production, replace this with:
    - ChromaDB (pip install chromadb) 
    - FAISS (pip install faiss-cpu)
    - Pinecone, Weaviate, etc.
    
    But this works fine for small-to-medium knowledge bases (<100K chunks).
    """

    def __init__(self):
        self.chunks = []          # The actual text chunks
        self.vectors = []         # TF-IDF vectors
        self.metadata = []        # Source file, position, etc.
        self.vocab = {}           # word → index mapping
        self.idf = {}             # word → IDF score

    def _tokenize(self, text: str) -> List[str]:
        """Simple word tokenization."""
        text = text.lower()
        # Keep alphanumeric and spaces
        cleaned = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in text)
        words = cleaned.split()
        # Remove very short words
        return [w for w in words if len(w) > 2]

    def _compute_tfidf(self, words: List[str]) -> dict:
        """Compute TF-IDF vector for a list of words."""
        tf = Counter(words)
        total = len(words) if words else 1
        vector = {}
        for word, count in tf.items():
            if word in self.idf:
                vector[word] = (count / total) * self.idf[word]
        return vector

    def _cosine_similarity(self, vec_a: dict, vec_b: dict) -> float:
        """Cosine similarity between two sparse vectors."""
        common = set(vec_a.keys()) & set(vec_b.keys())
        if not common:
            return 0.0
        
        dot = sum(vec_a[w] * vec_b[w] for w in common)
        norm_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
        norm_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def build_index(self):
        """Build TF-IDF index from all chunks."""
        # Tokenize all chunks
        all_words = []
        chunk_words = []
        for chunk in self.chunks:
            words = self._tokenize(chunk)
            chunk_words.append(words)
            all_words.extend(set(words))  # Unique words per doc for IDF

        # Build vocabulary
        word_counts = Counter(all_words)
        self.vocab = {word: i for i, word in enumerate(word_counts.keys())}

        # Compute IDF
        n_docs = len(self.chunks)
        self.idf = {}
        for word, count in word_counts.items():
            self.idf[word] = math.log(n_docs / (1 + count))

        # Compute TF-IDF vectors
        self.vectors = [self._compute_tfidf(words) for words in chunk_words]

        print(f"  Index built: {len(self.chunks)} chunks, {len(self.vocab)} unique terms")

    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float, dict]]:
        """
        Search for most relevant chunks.
        Returns: [(chunk_text, similarity_score, metadata), ...]
        """
        query_words = self._tokenize(query)
        query_vec = self._compute_tfidf(query_words)

        # Score all chunks
        scores = []
        for i, vec in enumerate(self.vectors):
            sim = self._cosine_similarity(query_vec, vec)
            scores.append((i, sim))

        # Sort by similarity (highest first)
        scores.sort(key=lambda x: x[1], reverse=True)

        # Return top results
        results = []
        for idx, score in scores[:top_k]:
            if score > 0:
                results.append((self.chunks[idx], score, self.metadata[idx]))

        return results


class QORRag:
    """
    Retrieval-Augmented Generation for QOR.
    
    Adds external knowledge that QOR can look up at query time.
    Works alongside QOR's internal CMS memory.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.store = SimpleVectorStore()
        self.chunk_size = chunk_size      # Characters per chunk
        self.chunk_overlap = chunk_overlap
        self.is_indexed = False

    def _split_into_chunks(self, text: str, source: str) -> List[Tuple[str, dict]]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        chunk_id = 0

        while start < len(text):
            end = start + self.chunk_size

            # Try to end at a sentence boundary
            if end < len(text):
                # Look for period, newline, or other boundary
                for boundary in ['. ', '.\n', '\n\n', '\n', '? ', '! ']:
                    pos = text[end-50:end+50].rfind(boundary)
                    if pos != -1:
                        end = end - 50 + pos + len(boundary)
                        break

            chunk_text = text[start:end].strip()
            if chunk_text and len(chunk_text) > 50:  # Skip tiny chunks
                meta = {
                    "source": source,
                    "chunk_id": chunk_id,
                    "start_char": start,
                    "end_char": end,
                }
                chunks.append((chunk_text, meta))
                chunk_id += 1

            start = end - self.chunk_overlap

        return chunks

    def add_text(self, text: str, source: str = "unknown"):
        """Add text to the knowledge base."""
        chunks = self._split_into_chunks(text, source)
        for chunk_text, meta in chunks:
            self.store.chunks.append(chunk_text)
            self.store.metadata.append(meta)
        self.is_indexed = False
        print(f"  Added {len(chunks)} chunks from '{source}'")

    def add_file(self, file_path: str):
        """Add a text file to the knowledge base."""
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
        source = os.path.basename(file_path)
        self.add_text(text, source)

    def add_folder(self, folder_path: str):
        """Add all text files in a folder to the knowledge base."""
        files = sorted(glob.glob(os.path.join(folder_path, '**', '*.txt'), recursive=True))
        if not files:
            print(f"  No .txt files found in '{folder_path}/'")
            return

        print(f"  Loading {len(files)} files from '{folder_path}/'")
        for fpath in files:
            self.add_file(fpath)

        self.build_index()

    def build_index(self):
        """Build the search index. Call after adding all documents."""
        self.store.build_index()
        self.is_indexed = True

    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float, dict]]:
        """Search for relevant chunks."""
        if not self.is_indexed:
            self.build_index()
        return self.store.search(query, top_k)

    def query(self, question: str, model, tokenizer,
              top_k: int = 3, max_context_chars: int = 1500,
              max_new_tokens: int = 200, temperature: float = 0.7) -> dict:
        """
        Full RAG pipeline:
        1. Search for relevant chunks
        2. Build prompt with context
        3. Generate answer with QOR
        
        Args:
            question: User's question
            model: QOR model
            tokenizer: QOR tokenizer
            top_k: Number of chunks to retrieve
            max_context_chars: Max context to include in prompt
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            
        Returns:
            {
                "question": "...",
                "answer": "...",
                "sources": [...],
                "chunks_used": 3,
            }
        """
        import torch

        # Step 1: Retrieve relevant chunks
        results = self.search(question, top_k=top_k)

        if not results:
            # No relevant context found — let QOR answer from its own knowledge
            prompt = f"Question: {question}\nAnswer:"
            sources = []
        else:
            # Step 2: Build context from chunks
            context_parts = []
            total_chars = 0
            sources = []

            for chunk_text, score, meta in results:
                if total_chars + len(chunk_text) > max_context_chars:
                    break
                context_parts.append(chunk_text)
                total_chars += len(chunk_text)
                sources.append({
                    "source": meta["source"],
                    "score": round(score, 3),
                    "preview": chunk_text[:100] + "...",
                })

            context = "\n\n".join(context_parts)

            # Step 3: Build prompt
            prompt = f"""Context information:
{context}

Based on the context above, answer the following question.
Question: {question}
Answer:"""

        # Step 4: Generate with QOR
        ids = tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = torch.tensor([ids], device=next(model.parameters()).device)

        model.eval()
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=50,
            stop_tokens=[tokenizer.eos_id],
        )

        full_output = tokenizer.decode(output_ids, skip_special_tokens=True)

        # Extract just the answer part
        answer = full_output
        if "Answer:" in full_output:
            answer = full_output.split("Answer:")[-1].strip()

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "chunks_used": len(sources),
            "prompt_tokens": len(ids),
        }

    def save(self, path: str = "rag_index.json"):
        """Save the knowledge base."""
        data = {
            "chunks": self.store.chunks,
            "metadata": self.store.metadata,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }
        with open(path, 'w') as f:
            json.dump(data, f)
        print(f"  Knowledge base saved to {path} ({len(self.store.chunks)} chunks)")

    def load(self, path: str = "rag_index.json"):
        """Load a saved knowledge base."""
        with open(path) as f:
            data = json.load(f)
        self.store.chunks = data["chunks"]
        self.store.metadata = data["metadata"]
        self.chunk_size = data.get("chunk_size", 500)
        self.chunk_overlap = data.get("chunk_overlap", 50)
        self.build_index()
        print(f"  Loaded knowledge base: {len(self.store.chunks)} chunks")

    def stats(self):
        """Print knowledge base statistics."""
        sources = set(m["source"] for m in self.store.metadata)
        total_chars = sum(len(c) for c in self.store.chunks)

        print(f"\n  Knowledge Base Stats:")
        print(f"    Total chunks:  {len(self.store.chunks)}")
        print(f"    Total chars:   {total_chars:,}")
        print(f"    Sources:       {len(sources)}")
        for src in sorted(sources):
            n = sum(1 for m in self.store.metadata if m["source"] == src)
            print(f"      {src}: {n} chunks")

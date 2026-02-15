"""
QOR ChatStore — Persistent Conversation History
=================================================
Stores all conversation turns (user + assistant) with per-session
hash chains for tamper detection.

Location: qor-data/chat.parquet

Features:
  - Per-session message grouping
  - Hash chain integrity per session
  - Configurable retention (default 90 days)
  - get_context() returns messages ready for format_chat()
"""

import os
import uuid
import threading
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import Optional, List, Dict

import pyarrow as pa
import pyarrow.parquet as pq

from .integrity import HashChain


# ==============================================================================
# Schema
# ==============================================================================

CHAT_SCHEMA = pa.schema([
    ("id", pa.string()),
    ("session_id", pa.string()),
    ("role", pa.string()),
    ("content", pa.string()),
    ("timestamp", pa.timestamp("us", tz="UTC")),
    ("confidence", pa.float32()),
    ("source", pa.string()),
    ("data_hash", pa.string()),
    ("prev_hash", pa.string()),
])


@dataclass
class ChatMessage:
    """A single chat message."""
    id: str
    session_id: str
    role: str            # "user" or "assistant"
    content: str
    timestamp: str       # ISO format
    confidence: float    # Model confidence (assistant only)
    source: str          # Answer source (assistant only)
    data_hash: str
    prev_hash: str


class ChatStore:
    """
    Persistent conversation history with per-session hash chains.

    Uses Arrow/Parquet for storage (same pattern as MemoryStore/CacheStore).
    Each session gets its own HashChain for independent integrity verification.
    Optionally encrypts content at rest via QORCrypto (Fernet AES).
    """

    def __init__(self, path: str = "chat.parquet", secret: str = "", crypto=None):
        self.path = path
        self._secret = secret
        self._crypto = crypto        # QORCrypto instance for at-rest encryption
        self.messages = []           # List[ChatMessage] — ordered by insertion
        self._chains = {}            # session_id -> HashChain
        self._dirty = False
        self._dirty_count = 0
        self._flush_every = 20
        self._lock = threading.RLock()  # Thread safety for concurrent reads/writes (reentrant)
        self._load()

    def _get_chain(self, session_id: str) -> HashChain:
        """Get or create a HashChain for a session."""
        if session_id not in self._chains:
            self._chains[session_id] = HashChain(f"chat:{session_id}",
                                                  secret=self._secret)
        return self._chains[session_id]

    def _load(self):
        """Load messages from Parquet file."""
        if not os.path.exists(self.path):
            return
        try:
            table = pq.read_table(self.path, schema=CHAT_SCHEMA)
            ids = table.column("id")
            session_ids = table.column("session_id")
            roles = table.column("role")
            contents = table.column("content")
            timestamps = table.column("timestamp").cast(pa.int64())
            confidences = table.column("confidence")
            sources = table.column("source")
            data_hashes = table.column("data_hash")
            prev_hashes = table.column("prev_hash")

            # Track last hash per session to restore chain heads
            session_last_hash = {}

            for i in range(table.num_rows):
                ts_us = timestamps[i].as_py()
                ts_iso = ""
                if ts_us and ts_us > 0:
                    ts_iso = datetime.fromtimestamp(
                        ts_us / 1_000_000, tz=timezone.utc
                    ).isoformat()

                sid = session_ids[i].as_py() or ""
                dh = data_hashes[i].as_py() or ""

                # Decrypt content if encrypted at rest
                raw_content = contents[i].as_py() or ""
                if self._crypto and raw_content:
                    try:
                        from .crypto import QORCrypto
                        if QORCrypto.is_encrypted(raw_content):
                            raw_content = self._crypto.decrypt_str(raw_content)
                    except Exception:
                        pass  # Leave as-is if decryption fails

                msg = ChatMessage(
                    id=ids[i].as_py() or "",
                    session_id=sid,
                    role=roles[i].as_py() or "",
                    content=raw_content,
                    timestamp=ts_iso,
                    confidence=confidences[i].as_py() or 0.0,
                    source=sources[i].as_py() or "",
                    data_hash=dh,
                    prev_hash=prev_hashes[i].as_py() or "",
                )
                self.messages.append(msg)
                if dh:
                    session_last_hash[sid] = dh

            # Restore chain heads
            for sid, last_hash in session_last_hash.items():
                chain = self._get_chain(sid)
                chain.set_head(last_hash)
        except Exception:
            pass

    @staticmethod
    def _iso_to_us(iso_str: str) -> int:
        """Convert ISO timestamp string to microseconds since epoch."""
        if not iso_str:
            return 0
        try:
            dt = datetime.fromisoformat(iso_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1_000_000)
        except (ValueError, TypeError):
            return 0

    def add_turn(self, session_id: str, question: str, answer_result: dict):
        """
        Add a complete Q&A turn (user message + assistant response).

        Args:
            session_id: Conversation session identifier
            question: User's question text
            answer_result: Dict from gate.answer() with keys:
                           answer, confidence, source, timestamp
        """
        with self._lock:
            chain = self._get_chain(session_id)
            now = datetime.now(timezone.utc).isoformat()

            # User message — hash on PLAINTEXT, store encrypted
            user_prev = chain.head
            user_hash = chain.compute_hash(question)
            user_msg = ChatMessage(
                id=str(uuid.uuid4()),
                session_id=session_id,
                role="user",
                content=question,
                timestamp=now,
                confidence=0.0,
                source="",
                data_hash=user_hash,
                prev_hash=user_prev,
            )
            self.messages.append(user_msg)

            # Assistant message — hash on PLAINTEXT, store encrypted
            answer_text = answer_result.get("answer", "")
            asst_prev = chain.head
            asst_hash = chain.compute_hash(answer_text)
            asst_msg = ChatMessage(
                id=str(uuid.uuid4()),
                session_id=session_id,
                role="assistant",
                content=answer_text,
                timestamp=answer_result.get("timestamp", now),
                confidence=answer_result.get("confidence", 0.0),
                source=answer_result.get("source", ""),
                data_hash=asst_hash,
                prev_hash=asst_prev,
            )
            self.messages.append(asst_msg)

            self._dirty = True
            self._dirty_count += 2
            if self._dirty_count >= self._flush_every:
                self.save()

    def get_history(self, session_id: str, last_n: int = 0) -> List[ChatMessage]:
        """
        Get messages for a session.

        Args:
            session_id: Session to query
            last_n: Return only last N messages (0 = all)

        Returns:
            List of ChatMessage in chronological order.
        """
        with self._lock:
            session_msgs = [m for m in self.messages if m.session_id == session_id]
            if last_n > 0:
                session_msgs = session_msgs[-last_n:]
            return session_msgs

    def get_context(self, session_id: str, last_n: int = 10) -> List[dict]:
        """
        Get conversation context ready for format_chat().

        Returns:
            List of {"role": "user"|"assistant", "content": "..."} dicts.
        """
        with self._lock:
            msgs = self.get_history(session_id, last_n=last_n)
            return [{"role": m.role, "content": m.content} for m in msgs]

    def list_sessions(self) -> List[dict]:
        """
        List all sessions with summary info.

        Returns:
            List of {"session_id": str, "message_count": int,
                     "first_message": str, "last_message": str}
        """
        sessions = {}
        for msg in self.messages:
            sid = msg.session_id
            if sid not in sessions:
                sessions[sid] = {
                    "session_id": sid,
                    "message_count": 0,
                    "first_message": msg.timestamp,
                    "last_message": msg.timestamp,
                }
            sessions[sid]["message_count"] += 1
            sessions[sid]["last_message"] = msg.timestamp
        return list(sessions.values())

    def delete_session(self, session_id: str) -> int:
        """
        Delete all messages for a session.

        Returns:
            Number of messages deleted.
        """
        before = len(self.messages)
        self.messages = [m for m in self.messages if m.session_id != session_id]
        removed = before - len(self.messages)

        if session_id in self._chains:
            del self._chains[session_id]

        if removed > 0:
            self._dirty = True
            self.save()
        return removed

    def cleanup_old(self, max_age_days: int = 90) -> int:
        """
        Remove sessions older than max_age_days.

        A session is considered old if ALL its messages are older than the cutoff.

        Returns:
            Number of messages removed.
        """
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=max_age_days)

        # Find sessions where the NEWEST message is older than cutoff
        session_newest = {}
        for msg in self.messages:
            sid = msg.session_id
            if not msg.timestamp:
                continue
            try:
                ts = datetime.fromisoformat(msg.timestamp)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if sid not in session_newest or ts > session_newest[sid]:
                    session_newest[sid] = ts
            except (ValueError, TypeError):
                pass

        old_sessions = set()
        for sid, newest_ts in session_newest.items():
            if newest_ts < cutoff:
                old_sessions.add(sid)

        if not old_sessions:
            return 0

        before = len(self.messages)
        self.messages = [m for m in self.messages
                         if m.session_id not in old_sessions]
        removed = before - len(self.messages)

        for sid in old_sessions:
            if sid in self._chains:
                del self._chains[sid]

        if removed > 0:
            self._dirty = True
            self.save()
        return removed

    def verify_chain(self, session_id: str) -> dict:
        """
        Verify hash chain integrity for a specific session.

        Returns:
            {"valid": bool, "checked": int, "broken_at": int|None}
        """
        session_msgs = self.get_history(session_id)
        records = []
        for msg in session_msgs:
            records.append({
                "content": msg.content,
                "data_hash": msg.data_hash,
                "prev_hash": msg.prev_hash,
            })
        chain = HashChain(f"chat:{session_id}", secret=self._secret)
        return chain.verify_chain(records)

    def save(self):
        """Flush all messages to Parquet."""
        if not self.messages:
            # Write empty file
            table = pa.table({
                "id": pa.array([], type=pa.string()),
                "session_id": pa.array([], type=pa.string()),
                "role": pa.array([], type=pa.string()),
                "content": pa.array([], type=pa.string()),
                "timestamp": pa.array([], type=pa.timestamp("us", tz="UTC")),
                "confidence": pa.array([], type=pa.float32()),
                "source": pa.array([], type=pa.string()),
                "data_hash": pa.array([], type=pa.string()),
                "prev_hash": pa.array([], type=pa.string()),
            })
            pq.write_table(table, self.path)
            self._dirty = False
            self._dirty_count = 0
            return

        ids = []
        session_ids = []
        roles = []
        contents = []
        timestamps = []
        confidences = []
        sources = []
        data_hashes = []
        prev_hashes = []

        for msg in self.messages:
            ids.append(msg.id)
            session_ids.append(msg.session_id)
            roles.append(msg.role)
            # Encrypt content at rest if crypto is available
            stored_content = msg.content
            if self._crypto and stored_content:
                try:
                    stored_content = self._crypto.encrypt_str(stored_content)
                except Exception:
                    pass  # Store plaintext if encryption fails
            contents.append(stored_content)
            timestamps.append(self._iso_to_us(msg.timestamp))
            confidences.append(msg.confidence)
            sources.append(msg.source)
            data_hashes.append(msg.data_hash)
            prev_hashes.append(msg.prev_hash)

        batch = pa.RecordBatch.from_pydict(
            {
                "id": ids,
                "session_id": session_ids,
                "role": roles,
                "content": contents,
                "timestamp": timestamps,
                "confidence": confidences,
                "source": sources,
                "data_hash": data_hashes,
                "prev_hash": prev_hashes,
            },
            schema=CHAT_SCHEMA,
        )
        table = pa.Table.from_batches([batch])
        pq.write_table(table, self.path)
        self._dirty = False
        self._dirty_count = 0

    def count(self) -> int:
        """Total number of messages."""
        return len(self.messages)

    def session_count(self) -> int:
        """Number of unique sessions."""
        return len(set(m.session_id for m in self.messages))

    def get_topic_summary(self, session_id: str, last_n: int = 100) -> dict:
        """Analyze recent conversations to find frequently discussed topics.

        Scans user messages for known topic keywords and counts occurrences.

        Args:
            session_id: Session to analyze.
            last_n: Number of recent messages to scan (0 = all).

        Returns:
            {"bitcoin": 12, "trading": 8, "weather": 3, ...}
        """
        from collections import Counter
        msgs = self.get_history(session_id, last_n=last_n)
        user_msgs = [m.content for m in msgs if m.role == "user"]
        topic_counts = Counter()
        for msg in user_msgs:
            topics = _detect_topics_from_text(msg)
            topic_counts.update(topics)
        return dict(topic_counts.most_common(20))


# ==============================================================================
# TOPIC DETECTION (standalone helper for ChatStore)
# ==============================================================================

# Same topic map as __main__.py — duplicated here to avoid circular imports
_TOPIC_MAP = {
    "bitcoin": ["bitcoin", "btc"],
    "ethereum": ["ethereum", "eth"],
    "crypto": ["crypto", "cryptocurrency", "altcoin", "defi", "nft"],
    "trading": ["trading", "trade", "position", "stop loss", "take profit", "dca"],
    "solana": ["solana", "sol"],
    "stocks": ["stock", "shares", "equity", "s&p", "nasdaq", "dow"],
    "forex": ["forex", "currency", "eur/usd", "exchange rate"],
    "gold": ["gold", "silver", "commodities"],
    "ai": ["ai", "artificial intelligence", "machine learning", "neural", "llm", "gpt"],
    "programming": ["python", "javascript", "coding", "programming", "code", "github"],
    "weather": ["weather", "temperature", "rain", "forecast"],
    "news": ["news", "headline", "breaking"],
    "science": ["science", "research", "study", "discovery", "arxiv"],
    "sports": ["sports", "football", "basketball", "soccer", "game score"],
}


def _detect_topics_from_text(text: str) -> list:
    """Detect topics from text using keyword matching."""
    t = text.lower()
    topics = []
    for topic, keywords in _TOPIC_MAP.items():
        if any(kw in t for kw in keywords):
            topics.append(topic)
    return topics

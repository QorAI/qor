"""
QOR Integrity — Hash Chain Tamper Detection
=============================================
SHA-256 hash chain for verifying data integrity on CacheStore and ChatStore.
Each record's hash depends on its data + the previous record's hash, forming
an append-only chain. Any tampered record breaks the chain at that point.

Pure stdlib — no external dependencies (hashlib only).
"""

import hashlib


class HashChain:
    """
    SHA-256 hash chain for tamper detection.

    Each hash = SHA-256(data + "|" + prev_hash + "|" + secret).
    The chain head (most recent hash) is tracked internally.

    Usage:
        chain = HashChain("cache")
        h1 = chain.compute_hash("first record")     # prev_hash = "genesis"
        h2 = chain.compute_hash("second record")     # prev_hash = h1
        result = chain.verify_chain([
            {"data_hash": h1, "prev_hash": "genesis"},
            {"data_hash": h2, "prev_hash": h1},
        ])
        assert result["valid"] is True
    """

    GENESIS = "genesis"

    def __init__(self, name: str, secret: str = ""):
        self.name = name
        self.secret = secret
        self._prev_hash = self.GENESIS

    @property
    def head(self) -> str:
        """Current chain head (most recent hash)."""
        return self._prev_hash

    def compute_hash(self, data: str, prev_hash: str = None) -> str:
        """
        Compute SHA-256 hash for a record and advance the chain.

        Args:
            data: Record content to hash.
            prev_hash: Override previous hash (default: use chain head).

        Returns:
            Hex digest of SHA-256(data + "|" + prev_hash + "|" + secret).
        """
        if prev_hash is None:
            prev_hash = self._prev_hash

        payload = f"{data}|{prev_hash}|{self.secret}"
        h = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        self._prev_hash = h
        return h

    def set_head(self, hash_value: str):
        """Restore chain state from disk (e.g., last record's hash)."""
        self._prev_hash = hash_value

    def verify_chain(self, records: list) -> dict:
        """
        Walk records and verify hash chain integrity.

        Each record must have:
            - "data_hash": the stored hash
            - "prev_hash": the hash of the previous record
            - One of: "content", "data", or "key" — the data that was hashed

        For recomputation, the data field is assembled as the store would
        have assembled it. If no recognizable data field is found, the
        record is verified by link-only (prev_hash matches prior data_hash).

        Returns:
            {
                "valid": bool,       # True if entire chain is intact
                "checked": int,      # Number of records checked
                "broken_at": int|None  # Index of first broken link, or None
            }
        """
        if not records:
            return {"valid": True, "checked": 0, "broken_at": None}

        prev = self.GENESIS

        for i, rec in enumerate(records):
            stored_hash = rec.get("data_hash", "")
            stored_prev = rec.get("prev_hash", "")

            # Check link: this record's prev_hash should match the prior record's data_hash
            if stored_prev != prev:
                return {"valid": False, "checked": i, "broken_at": i}

            # Recompute hash if we have the data
            data = rec.get("content") or rec.get("data") or rec.get("key") or ""
            if data:
                expected = self._hash_raw(data, stored_prev)
                if expected != stored_hash:
                    return {"valid": False, "checked": i, "broken_at": i}

            prev = stored_hash

        return {"valid": True, "checked": len(records), "broken_at": None}

    def reset(self):
        """Reset chain to genesis state."""
        self._prev_hash = self.GENESIS

    def _hash_raw(self, data: str, prev_hash: str) -> str:
        """Compute hash without advancing the chain (for verification)."""
        payload = f"{data}|{prev_hash}|{self.secret}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

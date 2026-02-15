"""
QOR Crypto — AES Encryption + SHA-256 Integrity
==================================================
Provides Fernet (AES-128-CBC) encryption for chat content at rest
and SHA-256 hashing for data integrity across all stores.

- ChatStore uses both encrypt + hash (personal data)
- MemoryStore, HistoricalStore, KnowledgeGraph, RAG use hash only (API data)

Dependencies: cryptography (pip install cryptography)
"""

import os
import hashlib

from cryptography.fernet import Fernet


class QORCrypto:
    """
    AES encryption (Fernet) + SHA-256 hashing.

    Key management:
        - If key_path exists, loads key from file
        - If key_path doesn't exist, generates new key and saves it
        - If key is provided directly, uses that (for testing)

    Usage:
        crypto = QORCrypto(key_path="qor-data/.keyfile")
        encrypted = crypto.encrypt_str("secret message")
        decrypted = crypto.decrypt_str(encrypted)
        h = QORCrypto.hash_sha256("data to hash")
    """

    def __init__(self, key_path: str = None, key: bytes = None):
        if key:
            self._fernet = Fernet(key)
        elif key_path and os.path.exists(key_path):
            with open(key_path, 'rb') as f:
                self._fernet = Fernet(f.read().strip())
        else:
            new_key = Fernet.generate_key()
            if key_path:
                os.makedirs(os.path.dirname(key_path) or '.', exist_ok=True)
                with open(key_path, 'wb') as f:
                    f.write(new_key)
            self._fernet = Fernet(new_key)

    def encrypt_str(self, text: str) -> str:
        """Encrypt string -> base64 Fernet token string."""
        return self._fernet.encrypt(text.encode('utf-8')).decode('ascii')

    def decrypt_str(self, token: str) -> str:
        """Decrypt Fernet token string -> original string."""
        return self._fernet.decrypt(token.encode('ascii')).decode('utf-8')

    @staticmethod
    def hash_sha256(data: str) -> str:
        """SHA-256 hex digest of a string."""
        return hashlib.sha256(data.encode('utf-8')).hexdigest()

    @staticmethod
    def is_encrypted(text: str) -> bool:
        """Check if string looks like a Fernet token (starts with gAAAAA)."""
        return isinstance(text, str) and text.startswith('gAAAAA')

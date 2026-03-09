"""
encrypt.py — AES-256-GCM encryption layer for SASOK Emotional ID.
Key is loaded from the path specified in SASOK_AES_KEY_PATH env var.
Never store the key inside the repository.
"""
import os
import base64
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

KEY_ENV = "SASOK_AES_KEY_PATH"
_cached_key: bytes | None = None


def _load_key() -> bytes:
    global _cached_key
    if _cached_key is not None:
        return _cached_key
    key_path = os.getenv(KEY_ENV)
    if not key_path:
        raise EnvironmentError(
            f"credentials error: environment variable {KEY_ENV} is not set. "
            "Generate a key with: openssl rand -out aes.key 32"
        )
    try:
        with open(key_path, "rb") as fh:
            key = fh.read()
    except FileNotFoundError:
        raise EnvironmentError(
            f"credentials error: AES key file not found at '{key_path}'"
        )
    if len(key) != 32:
        raise ValueError(
            f"AES key must be exactly 32 bytes (AES-256). "
            f"Got {len(key)} bytes from '{key_path}'"
        )
    _cached_key = key
    return key


def encrypt_bytes(plaintext: bytes) -> bytes:
    """
    Encrypt plaintext using AES-256-GCM.
    Returns: nonce (16 bytes) + tag (16 bytes) + ciphertext.
    """
    key = _load_key()
    nonce = get_random_bytes(16)
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    ciphertext, tag = cipher.encrypt_and_digest(plaintext)
    return nonce + tag + ciphertext


def decrypt_bytes(blob: bytes) -> bytes:
    """
    Decrypt a blob produced by encrypt_bytes().
    Returns plaintext bytes.
    """
    if len(blob) < 32:
        raise ValueError("Encrypted blob is too short to be valid.")
    key = _load_key()
    nonce = blob[:16]
    tag = blob[16:32]
    ciphertext = blob[32:]
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    return cipher.decrypt_and_verify(ciphertext, tag)


def encrypt_text(text: str) -> str:
    """Encrypt a UTF-8 string; return hex-encoded blob."""
    return encrypt_bytes(text.encode("utf-8")).hex()


def decrypt_text(hex_blob: str) -> str:
    """Decrypt a hex-encoded blob; return UTF-8 string."""
    return decrypt_bytes(bytes.fromhex(hex_blob)).decode("utf-8")


def generate_key(path: str) -> None:
    """Generate a fresh 32-byte AES-256 key and write it to *path*."""
    key = get_random_bytes(32)
    with open(path, "wb") as fh:
        fh.write(key)
    os.chmod(path, 0o600)
    print(f"[encrypt] AES-256 key written to {path}")

"""
security/passwords.py — Secure Password Hashing Module
======================================================
Uses NIST-recommended PBKDF2-HMAC-SHA256 with 100,000 iterations and per-user 16-byte random salts.
"""

import hashlib
import hmac
import secrets

ITERATIONS = 100_000
HASH_NAME = "sha256"


def hash_password(password: str, salt: str = None) -> tuple[str, str]:
    """
    Hashes a plain-text password using PBKDF2-HMAC-SHA256.
    
    Returns:
        tuple[str, str]: (password_hash_hex, salt_hex)
    """
    if not salt:
        salt = secrets.token_hex(16)
    
    hash_bytes = hashlib.pbkdf2_hmac(
        HASH_NAME,
        password.encode("utf-8"),
        salt.encode("utf-8"),
        ITERATIONS,
    )
    return hash_bytes.hex(), salt


def verify_password(plain_password: str, stored_hash: str, salt: str) -> bool:
    """
    Verifies a plain-text password against a stored hash and salt in constant time.
    """
    new_hash, _ = hash_password(plain_password, salt)
    return hmac.compare_digest(new_hash, stored_hash)

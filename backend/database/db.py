"""
database/db.py — SQLite Database Manager
========================================
Handles database connection management, schema initialization, and SQL operations
for user authentication and search history tracking.
"""

import os
import sqlite3
from typing import Dict, List, Optional
from datetime import datetime, timezone

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DB_PATH = os.path.join(DB_DIR, "axion.db")


def get_db_connection() -> sqlite3.Connection:
    """Returns a SQLite connection configured with row factory and safe journal mode."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
    except Exception:
        try:
            conn.execute("PRAGMA journal_mode=DELETE;")
        except Exception:
            pass
    try:
        conn.execute("PRAGMA foreign_keys=ON;")
    except Exception:
        pass
    return conn


def init_db() -> None:
    """Initializes the database schema if tables do not already exist."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
        
        # Users table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        
        # Search History table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS search_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                ticker TEXT NOT NULL,
                exchange TEXT NOT NULL,
                company_name TEXT NOT NULL,
                score REAL NOT NULL,
                grade TEXT NOT NULL,
                recommendation TEXT NOT NULL,
                searched_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            );
            """
        )
        
        # MFA codes table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS mfa_codes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                code_hash TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                used INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            );
            """
        )

        # WebAuthn credentials table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS webauthn_credentials (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                credential_id TEXT UNIQUE NOT NULL,
                public_key TEXT NOT NULL,
                sign_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            );
            """
        )

        # WebAuthn challenges table (temporary, used during auth flow)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS webauthn_challenges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                challenge TEXT NOT NULL,
                expires_at TEXT NOT NULL
            );
            """
        )
        # Face recognition encodings table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS face_encodings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                encoding TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            );
            """
        )

        # Indexes for fast lookup
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_history_user_id ON search_history(user_id, searched_at DESC);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_mfa_codes_user_id ON mfa_codes(user_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_webauthn_creds_user_id ON webauthn_credentials(user_id);")
        conn.commit()
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("Could not initialize database schema: %s", e)


# ==============================================================================
# User Database Operations
# ==============================================================================

def create_user(username: str, email: str, password_hash: str, salt: str) -> Dict:
    """Inserts a new user into the database."""
    now_iso = datetime.now(timezone.utc).isoformat()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO users (username, email, password_hash, salt, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (username.lower().strip(), email.lower().strip(), password_hash, salt, now_iso),
        )
        conn.commit()
        user_id = cursor.lastrowid
        return {
            "id": user_id,
            "username": username.lower().strip(),
            "email": email.lower().strip(),
            "created_at": now_iso,
        }


def get_user_by_username_or_email(identifier: str) -> Optional[Dict]:
    """Fetches a user record by username or email."""
    identifier_clean = identifier.lower().strip()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, username, email, password_hash, salt, created_at
            FROM users
            WHERE username = ? OR email = ?
            """,
            (identifier_clean, identifier_clean),
        )
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None


def get_user_by_id(user_id: int) -> Optional[Dict]:
    """Fetches a user profile by ID."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, username, email, created_at FROM users WHERE id = ?",
            (user_id,),
        )
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None


# ==============================================================================
# Search History Database Operations
# ==============================================================================

def record_search_history(
    user_id: Optional[int],
    ticker: str,
    exchange: str,
    company_name: str,
    score: float,
    grade: str,
    recommendation: str,
) -> Dict:
    """Saves a stock search event to the search_history table."""
    now_iso = datetime.now(timezone.utc).isoformat()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO search_history (user_id, ticker, exchange, company_name, score, grade, recommendation, searched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (user_id, ticker, exchange, company_name, score, grade, recommendation, now_iso),
        )
        conn.commit()
        return {
            "id": cursor.lastrowid,
            "user_id": user_id,
            "ticker": ticker,
            "exchange": exchange,
            "company_name": company_name,
            "score": score,
            "grade": grade,
            "recommendation": recommendation,
            "searched_at": now_iso,
        }


def get_user_search_history(user_id: Optional[int], limit: int = 50) -> List[Dict]:
    """Retrieves recent search history for a user ordered by timestamp descending."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if user_id is not None:
            cursor.execute(
                """
                SELECT id, ticker, exchange, company_name, score, grade, recommendation, searched_at
                FROM search_history
                WHERE user_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (user_id, limit),
            )
        else:
            # Guest search history
            cursor.execute(
                """
                SELECT id, ticker, exchange, company_name, score, grade, recommendation, searched_at
                FROM search_history
                WHERE user_id IS NULL
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def clear_user_search_history(user_id: Optional[int]) -> int:
    """Deletes search history records for a user."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if user_id is not None:
            cursor.execute("DELETE FROM search_history WHERE user_id = ?", (user_id,))
        else:
            cursor.execute("DELETE FROM search_history WHERE user_id IS NULL")
        conn.commit()
        return cursor.rowcount


# ==============================================================================
# MFA Code Database Operations
# ==============================================================================

def create_mfa_code(user_id: int, code_hash: str, expires_at: str) -> None:
    """Stores a new hashed MFA OTP code."""
    now_iso = datetime.now(timezone.utc).isoformat()
    with get_db_connection() as conn:
        # Invalidate any previous unused codes for this user
        conn.execute("DELETE FROM mfa_codes WHERE user_id = ? AND used = 0", (user_id,))
        conn.execute(
            "INSERT INTO mfa_codes (user_id, code_hash, expires_at, used, created_at) VALUES (?, ?, ?, 0, ?)",
            (user_id, code_hash, expires_at, now_iso),
        )
        conn.commit()


def get_latest_mfa_code(user_id: int) -> Optional[Dict]:
    """Fetches the latest unused, unexpired MFA code for a user."""
    now_iso = datetime.now(timezone.utc).isoformat()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, code_hash, expires_at, used FROM mfa_codes
            WHERE user_id = ? AND used = 0 AND expires_at > ?
            ORDER BY id DESC LIMIT 1
            """,
            (user_id, now_iso),
        )
        row = cursor.fetchone()
        return dict(row) if row else None


def mark_mfa_code_used(code_id: int) -> None:
    """Marks an MFA code as used after successful verification."""
    with get_db_connection() as conn:
        conn.execute("UPDATE mfa_codes SET used = 1 WHERE id = ?", (code_id,))
        conn.commit()


# ==============================================================================
# WebAuthn Credential Database Operations
# ==============================================================================

def store_webauthn_credential(user_id: int, credential_id: str, public_key: str) -> Dict:
    """Saves a new WebAuthn public key credential for a user."""
    now_iso = datetime.now(timezone.utc).isoformat()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO webauthn_credentials (user_id, credential_id, public_key, sign_count, created_at)
            VALUES (?, ?, ?, 0, ?)
            """,
            (user_id, credential_id, public_key, now_iso),
        )
        conn.commit()
        return {"id": cursor.lastrowid, "user_id": user_id, "credential_id": credential_id}


def get_webauthn_credential_by_credential_id(credential_id: str) -> Optional[Dict]:
    """Fetches a WebAuthn credential record by its credential_id."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, user_id, credential_id, public_key, sign_count FROM webauthn_credentials WHERE credential_id = ?",
            (credential_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None


def get_webauthn_credentials_by_user(user_id: int) -> List[Dict]:
    """Returns all WebAuthn credentials registered for a user."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, credential_id, public_key, sign_count FROM webauthn_credentials WHERE user_id = ?",
            (user_id,),
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def update_webauthn_sign_count(credential_id: str, new_count: int) -> None:
    """Updates the signature counter to prevent replay attacks."""
    with get_db_connection() as conn:
        conn.execute(
            "UPDATE webauthn_credentials SET sign_count = ? WHERE credential_id = ?",
            (new_count, credential_id),
        )
        conn.commit()


def store_webauthn_challenge(user_id: Optional[int], challenge: str, expires_at: str) -> None:
    """Stores a temporary challenge for WebAuthn registration/authentication."""
    with get_db_connection() as conn:
        # Clean up old expired challenges
        now_iso = datetime.now(timezone.utc).isoformat()
        conn.execute("DELETE FROM webauthn_challenges WHERE expires_at < ?", (now_iso,))
        conn.execute(
            "INSERT INTO webauthn_challenges (user_id, challenge, expires_at) VALUES (?, ?, ?)",
            (user_id, challenge, expires_at),
        )
        conn.commit()


def get_webauthn_challenge(challenge: str) -> Optional[Dict]:
    """Retrieves and deletes a stored challenge (one-time use)."""
    now_iso = datetime.now(timezone.utc).isoformat()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, user_id, challenge FROM webauthn_challenges WHERE challenge = ? AND expires_at > ?",
            (challenge, now_iso),
        )
        row = cursor.fetchone()
        if row:
            data = dict(row)
            conn.execute("DELETE FROM webauthn_challenges WHERE id = ?", (data["id"],))
            conn.commit()
            return data
        return None


# ==============================================================================
# Face Recognition CRUD
# ==============================================================================

def store_face_encoding(user_id: int, encoding_json: str) -> int:
    """Stores a face encoding (JSON-serialized numpy array) for a user."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO face_encodings (user_id, encoding, created_at) VALUES (?, ?, ?)",
            (user_id, encoding_json, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        return cursor.lastrowid


def get_face_encodings_by_user(user_id: int) -> List[Dict]:
    """Returns all face encodings for a given user."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, user_id, encoding, created_at FROM face_encodings WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        )
        return [dict(row) for row in cursor.fetchall()]


def get_all_face_encodings() -> List[Dict]:
    """Returns all face encodings for all users (used during login matching)."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, user_id, encoding FROM face_encodings")
        return [dict(row) for row in cursor.fetchall()]


def delete_face_encodings_by_user(user_id: int) -> int:
    """Deletes all face encodings for a user."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM face_encodings WHERE user_id = ?", (user_id,))
        conn.commit()
        return cursor.rowcount


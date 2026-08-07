"""
tests/test_auth.py — Authentication Unit & Integration Tests
===========================================================
Tests registration, duplicate check, password hashing & verification,
and login token generation.
"""

import pytest
from fastapi.testclient import TestClient
from app import app
from database.db import init_db, get_db_connection

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_test_db():
    init_db()
    with get_db_connection() as conn:
        conn.execute("DELETE FROM search_history;")
        conn.execute("DELETE FROM users;")
        conn.commit()
    yield


def test_user_registration_and_login():
    username = "testuser_auth"
    email = "testuser_auth@example.com"
    password = "SecurePassword123!"

    # 1. Register user
    reg_resp = client.post(
        "/api/v1/auth/register",
        json={"username": username, "email": email, "password": password},
    )
    assert reg_resp.status_code == 201
    data = reg_resp.json()
    assert "access_token" in data
    assert data["user"]["username"] == username
    assert data["user"]["email"] == email

    # 2. Duplicate registration attempt
    dup_resp = client.post(
        "/api/v1/auth/register",
        json={"username": username, "email": "other@example.com", "password": password},
    )
    assert dup_resp.status_code == 400

    # 3. Login with correct credentials
    login_resp = client.post(
        "/api/v1/auth/login",
        json={"username_or_email": username, "password": password},
    )
    assert login_resp.status_code == 200
    login_data = login_resp.json()
    assert "access_token" in login_data
    token = login_data["access_token"]

    # 4. Login with invalid password
    bad_login = client.post(
        "/api/v1/auth/login",
        json={"username_or_email": username, "password": "WrongPassword!"},
    )
    assert bad_login.status_code == 401

    # 5. Fetch user profile with token
    me_resp = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert me_resp.status_code == 200
    me_data = me_resp.json()
    assert me_data["username"] == username


def test_mfa_flow(monkeypatch):
    from config import get_settings
    monkeypatch.setattr(get_settings(), "MFA_ENABLED", True)

    username = "mfauser"
    email = "mfauser@example.com"
    password = "SecureMfaPassword123!"

    # 1. Register user
    reg_resp = client.post(
        "/api/v1/auth/register",
        json={"username": username, "email": email, "password": password},
    )
    assert reg_resp.status_code == 201

    # 2. Login - should indicate MFA is required and provide pending_user_id
    login_resp = client.post(
        "/api/v1/auth/login",
        json={"username_or_email": username, "password": password},
    )
    assert login_resp.status_code == 200
    login_data = login_resp.json()
    assert login_data["mfa_required"] is True
    pending_user_id = login_data["pending_user_id"]
    assert pending_user_id is not None

    # 3. Send MFA code
    send_resp = client.post(
        "/api/v1/auth/mfa/send",
        json={"pending_user_id": pending_user_id},
    )
    assert send_resp.status_code == 200
    assert send_resp.json()["status"] == "sent"

    # Fetch OTP from database directly to verify
    from database.db import get_db_connection
    import hashlib
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT code_hash FROM mfa_codes WHERE user_id = ? AND used = 0",
            (pending_user_id,)
        ).fetchone()
        assert row is not None
        # Since we mock SMTP or check logs, we can verify with correct token manually by looking up code table.
        # But we don't have the original code here easily. Let's insert a dummy code in DB to verify.
        dummy_code = "123456"
        dummy_hash = hashlib.sha256(dummy_code.encode()).hexdigest()
        conn.execute(
            "UPDATE mfa_codes SET code_hash = ? WHERE user_id = ? AND used = 0",
            (dummy_hash, pending_user_id)
        )
        conn.commit()

    # 4. Verify with incorrect code
    verify_bad = client.post(
        "/api/v1/auth/mfa/verify",
        json={"pending_user_id": pending_user_id, "otp_code": "000000"},
    )
    assert verify_bad.status_code == 400

    # 5. Verify with correct code
    verify_good = client.post(
        "/api/v1/auth/mfa/verify",
        json={"pending_user_id": pending_user_id, "otp_code": dummy_code},
    )
    assert verify_good.status_code == 200
    assert "access_token" in verify_good.json()


def test_webauthn_flow():
    username = "webauthnuser"
    email = "webauthnuser@example.com"
    password = "WebAuthnPassword123!"

    # Register & get session
    reg_resp = client.post(
        "/api/v1/auth/register",
        json={"username": username, "email": email, "password": password},
    )
    token = reg_resp.json()["access_token"]

    # 1. Start biometric registration - should return PublicKeyCredentialCreationOptions
    begin_resp = client.post(
        "/api/v1/auth/webauthn/register/begin",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert begin_resp.status_code == 200
    options = begin_resp.json()
    assert "challenge" in options
    assert "user" in options
    assert options["user"]["name"] == username


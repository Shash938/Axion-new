"""
tests/test_history.py — Search History Integration Tests
=========================================================
Tests recording of stock searches and retrieval / clearing of search history.
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


def test_search_history_recording_and_fetching():
    # 1. Register & login a test user
    username = "history_user"
    email = "history_user@example.com"
    password = "HistoryPassword123!"

    reg_resp = client.post(
        "/api/v1/auth/register",
        json={"username": username, "email": email, "password": password},
    )
    assert reg_resp.status_code == 201
    token = reg_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 2. Get initial history (should be empty for new user)
    hist_resp = client.get("/api/v1/history", headers=headers)
    assert hist_resp.status_code == 200
    assert len(hist_resp.json()["history"]) == 0

    # 3. Perform a stock search
    analyze_resp = client.post(
        "/api/v1/analyze",
        json={"ticker": "TCS", "exchange": "NSE"},
        headers=headers,
    )
    assert analyze_resp.status_code == 200

    # 4. Fetch history again — should record TCS
    hist_resp2 = client.get("/api/v1/history", headers=headers)
    assert hist_resp2.status_code == 200
    items = hist_resp2.json()["history"]
    assert len(items) >= 1
    assert items[0]["ticker"] == "TCS"
    assert items[0]["exchange"] == "NSE"

    # 5. Clear history
    del_resp = client.delete("/api/v1/history", headers=headers)
    assert del_resp.status_code == 200

    # 6. Verify history is cleared
    hist_resp3 = client.get("/api/v1/history", headers=headers)
    assert hist_resp3.status_code == 200
    assert len(hist_resp3.json()["history"]) == 0

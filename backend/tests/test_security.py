import json
import pytest
from fastapi.testclient import TestClient
from app import app
from config import get_settings


client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_security_headers_present():
    response = client.get("/health")
    assert response.status_code == 200
    assert "x-content-type-options" in response.headers
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "x-frame-options" in response.headers
    assert response.headers["x-frame-options"] == "DENY"
    assert "content-security-policy" in response.headers

def test_ticker_input_validation_malicious():
    # Test malicious ticker that fails regex validation
    response = client.post("/api/v1/analyze", json={"ticker": "<script>alert(1)</script>", "exchange": "NSE"})
    assert response.status_code == 422 # Pydantic validation error

    response2 = client.post("/api/v1/analyze", json={"ticker": "../etc/passwd", "exchange": "NSE"})
    assert response2.status_code == 422

def test_api_key_authentication_enforced():
    # Temporarily enable require_api_key
    settings = get_settings()
    settings.REQUIRE_API_KEY = True
    
    # Missing key
    response = client.post("/api/v1/analyze", json={"ticker": "RELIANCE", "exchange": "NSE"})
    assert response.status_code == 401
    
    # Invalid key
    response = client.post("/api/v1/analyze", json={"ticker": "RELIANCE", "exchange": "NSE"}, headers={"X-API-Key": "invalid-key"})
    assert response.status_code == 401
    
    # Valid key (might return 422/503 depending on network, but shouldn't be 401)
    response = client.post("/api/v1/analyze", json={"ticker": "RELIANCE", "exchange": "NSE"}, headers={"X-API-Key": settings.API_KEYS[0]})
    assert response.status_code != 401
    
    # Reset for other tests
    settings.REQUIRE_API_KEY = False

def test_rate_limiting_exceeded():
    settings = get_settings()
    settings.RATE_LIMIT_PER_MINUTE = 2
    from security.rate_limiter import reset_rate_limiter
    reset_rate_limiter()
    
    # Request 1
    res1 = client.get("/api/v1/analyze/TCS")
    assert res1.status_code != 429
    
    # Request 2
    res2 = client.get("/api/v1/analyze/TCS")
    assert res2.status_code != 429
    
    # Request 3 should be rate limited
    res3 = client.get("/api/v1/analyze/TCS")
    assert res3.status_code == 429
    assert "retry-after" in res3.headers
    assert "x-ratelimit-limit" in res3.headers
    
    # Reset
    settings.RATE_LIMIT_PER_MINUTE = 60
    reset_rate_limiter()

def test_payload_size_limit():
    settings = get_settings()
    original_limit = settings.MAX_PAYLOAD_SIZE_BYTES
    settings.MAX_PAYLOAD_SIZE_BYTES = 100 * 1024  # 100 KB for testing
    try:
        large_payload = json.dumps({"ticker": "A" * (1024 * 150)})
        response = client.post(
            "/api/v1/analyze",
            content=large_payload,
            headers={"Content-Type": "application/json", "Content-Length": str(len(large_payload))}
        )
        assert response.status_code == 413
    finally:
        settings.MAX_PAYLOAD_SIZE_BYTES = original_limit




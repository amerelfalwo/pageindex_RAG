import uuid
import pytest
from fastapi.testclient import TestClient

from app import app
from core.rate_limiter import get_rate_limiter, get_client_ip


@pytest.fixture(autouse=True)
def reset_limiter():
    """Reset in-memory rate limiter buckets before and after each test."""
    limiter = get_rate_limiter()
    limiter.reset()
    yield
    limiter.reset()


@pytest.fixture
def client():
    return TestClient(app)


def test_first_hop_ip_extraction():
    """Verifies that only the first hop of X-Forwarded-For is extracted (ignoring proxy hops)."""
    from starlette.requests import Request

    scope = {
        "type": "http",
        "headers": [
            (b"x-forwarded-for", b"198.51.100.42, 10.0.0.1, 172.16.0.2"),
        ],
        "client": ("10.0.0.1", 54321),
    }
    req = Request(scope)
    assert get_client_ip(req) == "198.51.100.42"


def test_rate_limiter_blocks_after_limit_and_sets_retry_after(client, monkeypatch):
    """Enforces rate limit and checks HTTP 429 and Retry-After header."""
    monkeypatch.setenv("RATE_LIMIT_TOOLS_PER_MIN", "3")

    client_id = str(uuid.uuid4())
    headers = {
        "X-Client-ID": client_id,
        "X-Forwarded-For": "203.0.113.10",
    }

    # 3 allowed requests
    for i in range(3):
        r = client.post("/tools/calculate-math", headers=headers, json={"expression": f"{i} + 1"})
        assert r.status_code == 200

    # 4th request must be rejected with 429
    r4 = client.post("/tools/calculate-math", headers=headers, json={"expression": "99 + 1"})
    assert r4.status_code == 429
    assert "Rate limit exceeded" in r4.json()["detail"]
    assert "Retry-After" in r4.headers
    retry_after = int(r4.headers["Retry-After"])
    assert retry_after > 0


def test_rotating_uuids_cannot_bypass_rate_limit(client, monkeypatch):
    """
    Rotating client_id UUIDs from the SAME IP must NOT bypass rate limiting.
    The IP-level bucket must catch and throttle the attacker.
    """
    monkeypatch.setenv("RATE_LIMIT_TOOLS_PER_MIN", "3")
    attacker_ip = "198.51.100.99"

    # Make 3 requests, each with a different UUID but same IP
    for i in range(3):
        headers = {
            "X-Client-ID": str(uuid.uuid4()),
            "X-Forwarded-For": f"{attacker_ip}, 10.0.0.1",
        }
        r = client.post("/tools/calculate-math", headers=headers, json={"expression": f"{i} + 10"})
        assert r.status_code == 200

    # 4th request with yet another brand-new UUID from same IP must be rejected
    headers_new_uuid = {
        "X-Client-ID": str(uuid.uuid4()),
        "X-Forwarded-For": f"{attacker_ip}, 10.0.0.1",
    }
    r4 = client.post("/tools/calculate-math", headers=headers_new_uuid, json={"expression": "100 + 1"})
    assert r4.status_code == 429
    assert "Rate limit exceeded" in r4.json()["detail"]


def test_distinct_clients_on_distinct_ips_have_independent_limits(client, monkeypatch):
    """Client A hitting limit does not block Client B on a different IP."""
    monkeypatch.setenv("RATE_LIMIT_TOOLS_PER_MIN", "2")

    headers_a = {
        "X-Client-ID": str(uuid.uuid4()),
        "X-Forwarded-For": "192.0.2.1",
    }
    headers_b = {
        "X-Client-ID": str(uuid.uuid4()),
        "X-Forwarded-For": "192.0.2.2",
    }

    # Exhaust Client A
    client.post("/tools/calculate-math", headers=headers_a, json={"expression": "1 + 1"})
    client.post("/tools/calculate-math", headers=headers_a, json={"expression": "2 + 2"})
    r_a_blocked = client.post("/tools/calculate-math", headers=headers_a, json={"expression": "3 + 3"})
    assert r_a_blocked.status_code == 429

    # Client B should still succeed
    r_b = client.post("/tools/calculate-math", headers=headers_b, json={"expression": "4 + 4"})
    assert r_b.status_code == 200

import os

from fastapi.testclient import TestClient

os.environ.setdefault("HTTP_REQUIRE_AUTH", "true")
os.environ.setdefault("HTTP_AUTH_TOKEN", "test-token")
os.environ.setdefault("RATE_LIMIT_ENABLED", "true")
os.environ.setdefault("RATE_LIMIT_MAX_RPM", "1")

from src.http_server import app  # noqa: E402


def test_health():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "healthy"


def test_auth_required_for_schemas():
    with TestClient(app) as client:
        r = client.get("/v1/schemas")
        assert r.status_code == 401


def test_auth_ok_for_schemas():
    with TestClient(app) as client:
        r = client.get("/v1/schemas", headers={"x-api-key": os.environ["HTTP_AUTH_TOKEN"]})
        assert r.status_code == 200
        assert isinstance(r.json(), dict)


def test_rate_limit_enforced():
    with TestClient(app) as client:
        headers = {"x-api-key": os.environ["HTTP_AUTH_TOKEN"]}
        r1 = client.get("/v1/schemas", headers=headers)
        assert r1.status_code == 200
        r2 = client.get("/v1/schemas", headers=headers)
        assert r2.status_code == 429

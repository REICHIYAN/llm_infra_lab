from fastapi.testclient import TestClient

from serving.api_server import app


client = TestClient(app)


def test_health() -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("status") == "ok"


def test_generate() -> None:
    resp = client.post(
        "/generate",
        json={"prompt": "hello", "max_new_tokens": 8},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "request_id" in data
    assert "text" in data
    assert isinstance(data["request_id"], int)
    assert isinstance(data["text"], str)
    assert "request=" in data["text"]
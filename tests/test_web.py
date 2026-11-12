import base64
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from mapper.web import Upstream, load_env, make_app

PLY = b"ply\nfake model bytes"


class FakeRouter:
    """Answers like the mapper behind the router would."""

    def __init__(self):
        self.requests = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer secret"
        body = json.loads(request.content)
        self.requests.append(body)
        parts = body["messages"][0]["content"]
        text = parts[0]["text"]
        images = len(parts) - 1
        if text.startswith("step"):
            answer = {"frames": images, "frontiers": [], "map_png": "iVBO"}
        elif text.startswith("finish"):
            answer = {"frames": 4, "frames_used": 4, "parts": 2, "points": 9, "units": "cm", "ply_path": "/secret"}
        elif text.startswith("queue"):
            answer = {"queue": {"gpu_busy": True, "waiting": 1, "other_sessions_active": 2, "message": "busy", "asked": text}}
        elif text.startswith("download"):
            half = len(PLY) // 2
            chunk = PLY[:half] if text.endswith("part=0") else PLY[half:]
            answer = {"data": base64.b64encode(chunk).decode()}
        else:
            return httpx.Response(400, json={"error": {"message": "unknown command"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(answer)}}]})


@pytest.fixture
def web(tmp_path):
    router = FakeRouter()
    upstream = Upstream("https://router.example/v1", "secret", "alias-visual-geometry", httpx.MockTransport(router))
    client = TestClient(make_app(upstream, tmp_path))
    client.router = router
    return client


def test_page(web):
    r = web.get("/")
    assert r.status_code == 200 and "Room Mapper" in r.text
    assert "secret" not in r.text


def test_step_forwards_photos_and_height(web):
    r = web.post("/api/step", json={"session": "hall", "camera_height_cm": "120",
                                    "images": ["data:image/jpeg;base64,AAAA"] * 3})
    assert r.status_code == 200 and r.json()["frames"] == 3
    sent = web.router.requests[-1]
    assert sent["model"] == "alias-visual-geometry"
    assert sent["messages"][0]["content"][0]["text"] == "step session=hall camera_height_cm=120.0"


def test_finish_downloads_the_model(web, tmp_path):
    r = web.post("/api/finish", json={"session": "hall"})
    assert r.status_code == 200
    data = r.json()
    assert data["ply_url"] == "/api/model/hall.ply" and "ply_path" not in data
    assert web.get(data["ply_url"]).content == PLY


@pytest.mark.parametrize("path,body", [
    ("/api/step", {"session": "../x", "images": ["data:image/jpeg;base64,AA"]}),
    ("/api/step", {"session": "hall", "images": ["http://evil/x.jpg"]}),
])
def test_bad_requests(web, path, body):
    assert web.post(path, json=body).status_code == 400


def test_missing_model(web):
    assert web.get("/api/model/nothing.ply").status_code == 404


def test_load_env(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\nOPENAI_API_KEY='abc'\nOPENAI_BASE_URL=https://x/v1\n")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://from-shell/v1")
    monkeypatch.setenv("MAPPER_MODEL", "other")
    values = load_env(env)
    assert values["OPENAI_API_KEY"] == "abc" and values["OPENAI_BASE_URL"] == "https://x/v1"
    assert values["MAPPER_MODEL"] == "other"  # not in .env, so the shell's value is used


def test_queue(web):
    q = web.get("/api/queue", params={"session": "hall"}).json()["queue"]
    assert q["asked"] == "queue session=hall" and q["waiting"] == 1
    assert web.get("/api/queue", params={"session": "../x"}).json()["queue"]["asked"] == "queue"

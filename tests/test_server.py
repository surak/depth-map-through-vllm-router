import base64
import json

import pytest
from fastapi.testclient import TestClient

from mapper.server import Mapper, make_app

from .conftest import FakeRunner

JPEG = b"\xff\xd8 not really a jpeg"


@pytest.fixture
def client(tmp_path):
    runner = FakeRunner()
    app = make_app(Mapper(runner, tmp_path, step_frames=3), ["alias-visual-geometry"], api_key="k")
    c = TestClient(app, headers={"Authorization": "Bearer k"})
    c.runner = runner
    return c


def ask(client, text, images=0, **extra):
    parts = [{"type": "text", "text": text}] + [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(JPEG).decode()}}
        for _ in range(images)]
    body = {"model": "alias-visual-geometry", "messages": [{"role": "user", "content": parts}], **extra}
    return client.post("/v1/chat/completions", json=body)


def content(r):
    assert r.status_code == 200, r.text
    return json.loads(r.json()["choices"][0]["message"]["content"])


def test_models_and_auth(client):
    assert client.get("/health").status_code == 200
    assert [m["id"] for m in client.get("/v1/models").json()["data"]] == ["alias-visual-geometry"]
    assert client.get("/v1/models", headers={"Authorization": "Bearer wrong"}).status_code == 401


def test_step_then_finish_then_download(client, tmp_path):
    first = content(ask(client, "step session=room camera_height_cm=10", images=1))
    assert first["frames"] == 1 and "note" in first
    step = content(ask(client, "step session=room", images=3))
    assert step["frames"] == 4 and step["frames_used"] == 3
    assert step["frontiers"] and step["obstacle_ahead_cm"]
    assert base64.b64decode(step["map_png"]).startswith(b"\x89PNG")
    assert client.runner.calls[-1][0].endswith("000000.jpg")  # first photo always kept
    assert client.runner.calls[-1][-1].endswith("000003.jpg")  # and the latest

    done = content(ask(client, "finish session=room"))
    assert done["units"] == "cm" and done["parts"] == 1
    part = content(ask(client, "download session=room part=0"))
    assert base64.b64decode(part["data"]) == (tmp_path / "room" / "model.ply").read_bytes()


def test_streaming(client):
    r = ask(client, "status session=x", stream=True)
    events = [line for line in r.text.splitlines() if line.startswith("data: ")]
    assert events[-1] == "data: [DONE]"
    assert json.loads(json.loads(events[0][6:])["choices"][0]["delta"]["content"])["frames"] == 0


@pytest.mark.parametrize("text,expected", [
    ("step session=../etc", "session"),
    ("dance session=a", "unknown command"),
    ("download session=a", "no model"),
])
def test_errors(client, text, expected):
    r = ask(client, text)
    assert r.status_code == 400 and expected in r.json()["error"]["message"]


def test_unknown_model(client):
    r = client.post("/v1/chat/completions", json={"model": "gpt-4", "messages": []})
    assert r.status_code == 404

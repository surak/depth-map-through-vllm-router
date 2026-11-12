"""A web page for sending photos from a phone to the mapper.

The page resizes the photos in the browser and posts them here. This server adds the API key from
.env and forwards them to the router, so the key never reaches the phone. Finished models are
downloaded once and kept in web-data/ for the 3D viewer.
"""

from __future__ import annotations

import base64
import json
import os
import re
import socket
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse

HERE = Path(__file__).parent
SESSION_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def load_env(path: Path) -> dict[str, str]:
    """KEY=value lines. They win over the environment, where an OpenAI key may be set for other tools."""
    values = {k: v for k, v in os.environ.items() if k.startswith(("OPENAI_", "MAPPER_"))}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip("'\"")
    return values


class Upstream:
    def __init__(self, base_url: str, api_key: str, model: str, transport: httpx.AsyncBaseTransport | None = None):
        self.model = model
        self.client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=600, transport=transport,
                                        headers={"Authorization": f"Bearer {api_key}"})

    async def ask(self, text: str, images: list[str] = ()) -> dict:
        parts = [{"type": "text", "text": text}] + [{"type": "image_url", "image_url": {"url": u}} for u in images]
        r = await self.client.post("/chat/completions", json={
            "model": self.model, "messages": [{"role": "user", "content": parts}]})
        if r.status_code != 200:
            try:
                message = r.json()["error"]["message"]
            except (ValueError, KeyError, TypeError):
                message = r.text[:300] or r.reason_phrase
            raise UpstreamError(r.status_code, message)
        return json.loads(r.json()["choices"][0]["message"]["content"])


class UpstreamError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def make_app(upstream: Upstream, data_dir: Path) -> FastAPI:
    app = FastAPI(title="visual geometry web")
    data_dir.mkdir(parents=True, exist_ok=True)

    def session_of(body: dict) -> str:
        name = str(body.get("session", ""))
        if not SESSION_RE.match(name):
            raise UpstreamError(400, "session names use letters, digits, - and _")
        return name

    @app.exception_handler(UpstreamError)
    async def upstream_error(request: Request, e: UpstreamError):
        return JSONResponse({"error": str(e)}, status_code=e.status if 400 <= e.status < 600 else 502)

    @app.get("/")
    async def index():
        return FileResponse(HERE / "static" / "index.html")

    @app.post("/api/step")
    async def step(request: Request):
        body = await request.json()
        text = f"step session={session_of(body)}"
        if body.get("camera_height_cm"):
            text += f" camera_height_cm={float(body['camera_height_cm'])}"
        images = [u for u in body.get("images", []) if isinstance(u, str) and u.startswith("data:image/")]
        if not images:
            raise UpstreamError(400, "no photos")
        return await upstream.ask(text, images)

    @app.post("/api/finish")
    async def finish(request: Request):
        session = session_of(await request.json())
        result = await upstream.ask(f"finish session={session}")
        if result.get("parts"):
            path = data_dir / f"{session}.ply"
            with path.open("wb") as f:
                for part in range(result["parts"]):
                    piece = await upstream.ask(f"download session={session} part={part}")
                    f.write(base64.b64decode(piece["data"]))
            result["ply_url"] = f"/api/model/{session}.ply"
        result.pop("ply_path", None)
        return result

    @app.post("/api/status")
    async def status(request: Request):
        return await upstream.ask(f"status session={session_of(await request.json())}")

    @app.get("/api/model/{name}.ply")
    async def model(name: str):
        path = data_dir / f"{name}.ply"
        if not SESSION_RE.match(name) or not path.exists():
            return JSONResponse({"error": "no such model"}, status_code=404)
        return FileResponse(path, media_type="application/octet-stream", filename=f"{name}.ply")

    return app


def lan_address() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))  # no packet is sent; this just picks the outgoing interface
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def main() -> None:
    import argparse

    import uvicorn

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="0.0.0.0", help="0.0.0.0 lets phones on the same network connect")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--env", default=".env")
    p.add_argument("--data-dir", default="web-data")
    args = p.parse_args()

    env = load_env(Path(args.env))
    if not env.get("OPENAI_API_KEY"):
        raise SystemExit(f"no OPENAI_API_KEY in {args.env}; copy .env.example to .env and fill it in")
    upstream = Upstream(env.get("OPENAI_BASE_URL", "https://api.blablador.fz-juelich.de/v1"),
                        env["OPENAI_API_KEY"], env.get("MAPPER_MODEL", "alias-visual-geometry"))
    print(f"\n  Open on your phone:  http://{lan_address()}:{args.port}/\n", flush=True)
    uvicorn.run(make_app(upstream, Path(args.data_dir)), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

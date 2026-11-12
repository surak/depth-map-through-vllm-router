"""OpenAI-compatible front for VGGT mapping.

Routers and the synapse client only speak the OpenAI API, so the mapper pretends to be a chat
model. A request is a normal chat completion: the photos are `image_url` parts (base64 data URLs)
and the text is a command, e.g. "step session=kitchen camera_height_cm=9". The answer's content
is JSON. Commands:

  step     add the photos to the session and return pose, frontiers and a top-down map
  finish   reconstruct from all photos of the session and store a coloured point cloud (PLY)
  download fetch the PLY in base64 pieces: "download session=kitchen part=0"
  status   number of photos and files of a session
  queue    how busy the GPU is and how many other sessions are mapping

Every answer has a "queue" part saying how long the request waited for the GPU, how many requests
were ahead of it, and how many other sessions were active in the last 10 minutes. Session names of
other people are never shown, only counts.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import os
import re
import threading
import time
import uuid
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from . import mapping

SESSION_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
PART_BYTES = 3 * 1024 * 1024
MAX_IMAGE_BYTES = 10 * 1024 * 1024


class CommandError(Exception):
    pass


def parse_command(text: str) -> tuple[str, dict[str, str]]:
    words = text.split()
    command = words[0].lower() if words and "=" not in words[0] else ""
    options = dict(w.split("=", 1) for w in words if "=" in w)
    return command, options


def message_parts(body: dict) -> tuple[str, list[bytes]]:
    """Text and decoded images of the last user message."""
    users = [m for m in body.get("messages", []) if m.get("role") == "user"]
    if not users:
        raise CommandError("no user message")
    content = users[-1].get("content", "")
    if isinstance(content, str):
        return content, []
    texts, images = [], []
    for part in content:
        if part.get("type") == "text":
            texts.append(part.get("text", ""))
        elif part.get("type") == "image_url":
            url = part.get("image_url", {})
            url = url.get("url", "") if isinstance(url, dict) else url
            if not url.startswith("data:") or "," not in url:
                raise CommandError("images must be base64 data URLs")
            try:
                data = base64.b64decode(url.split(",", 1)[1], validate=True)
            except binascii.Error as e:
                raise CommandError(f"bad base64 image: {e}") from None
            if len(data) > MAX_IMAGE_BYTES:
                raise CommandError("image larger than 10 MB")
            images.append(data)
    return " ".join(texts), images


def spread(n: int, k: int) -> list[int]:
    """Up to k indices out of n, always with the first and the last ones."""
    if n <= k:
        return list(range(n))
    return sorted(set(np.linspace(0, n - 1, k).round().astype(int).tolist()))


ACTIVE_SECONDS = 600


class GpuQueue:
    """First come, first served access to the GPU, with numbers people can be shown."""

    def __init__(self):
        self.cond = threading.Condition()
        self.next_ticket = 0
        self.serving = 0
        self.busy = False

    @contextmanager
    def turn(self):
        with self.cond:
            ticket = self.next_ticket
            self.next_ticket += 1
            ahead = ticket - self.serving  # requests running or waiting before this one
            started = time.monotonic()
            while self.serving != ticket:
                self.cond.wait()
            self.busy = True
        try:
            yield {"ahead_on_arrival": ahead, "waited_s": round(time.monotonic() - started, 1)}
        finally:
            with self.cond:
                self.busy = False
                self.serving += 1
                self.cond.notify_all()

    def state(self) -> dict:
        with self.cond:
            return {"gpu_busy": self.busy, "waiting": self.next_ticket - self.serving - int(self.busy)}


class Mapper:
    def __init__(self, runner, data_dir: Path, step_frames: int = 32, final_frames: int = 96,
                 default_height_cm: float = 10.0):
        self.gpu = GpuQueue()
        self.session_locks = defaultdict(threading.Lock)  # one request per session at a time
        self.seen: dict[str, float] = {}  # session -> last request time
        self.runner = runner
        self.data_dir = data_dir
        self.step_frames = step_frames
        self.final_frames = final_frames
        self.default_height_cm = default_height_cm

    def session_dir(self, options: dict) -> Path:
        name = options.get("session", "")
        if not SESSION_RE.match(name):
            raise CommandError("give session=<name> (letters, digits, - and _)")
        return self.data_dir / name

    def settings(self, folder: Path, options: dict) -> dict:
        path = folder / "session.json"
        meta = json.loads(path.read_text()) if path.exists() else {"camera_height_cm": self.default_height_cm}
        if "camera_height_cm" in options:
            try:
                meta["camera_height_cm"] = float(options["camera_height_cm"])
            except ValueError:
                raise CommandError("camera_height_cm must be a number") from None
            if not 1 <= meta["camera_height_cm"] <= 300:
                raise CommandError("camera_height_cm must be between 1 and 300")
        folder.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(meta))
        return meta

    def frames(self, folder: Path) -> list[Path]:
        return sorted((folder / "frames").glob("*.jpg"))

    def reconstruct(self, paths: list[str]) -> tuple:
        with self.gpu.turn() as waited:
            return self.runner(paths), waited

    def queue_state(self, session: str | None = None) -> dict:
        now = time.time()
        others = sum(1 for name, t in list(self.seen.items()) if name != session and now - t < ACTIVE_SECONDS)
        state = self.gpu.state()
        state["other_sessions_active"] = others
        if state["gpu_busy"] or state["waiting"]:
            state["message"] = (f"{state['waiting'] + state['gpu_busy']} request(s) are using or waiting for the GPU; "
                                "a step takes a few seconds, a finish up to half a minute")
        elif others:
            state["message"] = f"{others} other session(s) mapped something in the last 10 minutes"
        else:
            state["message"] = "nobody else is mapping right now"
        return state

    def run(self, text: str, images: list[bytes]) -> dict:
        command, options = parse_command(text)
        command = command or ("step" if images else "status")
        if command == "queue":
            return {"queue": self.queue_state(options.get("session"))}
        handler = {"step": self.step, "finish": self.finish, "download": self.download,
                   "status": self.status}.get(command)
        if handler is None:
            raise CommandError(f"unknown command {command!r}; use step, finish, download, status or queue")
        session = self.session_dir(options).name
        self.seen[session] = time.time()
        with self.session_locks[session]:
            result = handler(options, images)
        result["queue"] = {**result.get("queue", {}), **self.queue_state(session)}
        return result

    def step(self, options: dict, images: list[bytes]) -> dict:
        folder = self.session_dir(options)
        meta = self.settings(folder, options)
        (folder / "frames").mkdir(parents=True, exist_ok=True)
        start = len(self.frames(folder))
        for i, data in enumerate(images):
            (folder / "frames" / f"{start + i:06d}.jpg").write_bytes(data)
        frames = self.frames(folder)
        if len(frames) < 2:
            return {"session": folder.name, "frames": len(frames), "note": "need at least 2 photos for a map"}
        picked = [str(frames[i]) for i in spread(len(frames), self.step_frames)]
        rec, waited = self.reconstruct(picked)
        try:
            m = mapping.build_map(rec, meta["camera_height_cm"])
        except ValueError as e:
            return {"session": folder.name, "frames": len(frames), "note": str(e), "queue": waited}
        png = mapping.render_png(m)
        (folder / "map.png").write_bytes(png)
        x, z = m.path_cm[-1]
        return {
            "session": folder.name,
            "frames": len(frames),
            "frames_used": len(picked),
            "pose": {"x_cm": round(float(x), 1), "z_cm": round(float(z), 1),
                     "heading_deg": round(-np.degrees(m.heading), 1)},
            "frontiers": m.frontiers[:10],
            "obstacle_ahead_cm": m.obstacle_ahead_cm,
            "explored_done": not m.frontiers,
            "map_png": base64.b64encode(png).decode(),
            "queue": waited,
        }

    def finish(self, options: dict, images: list[bytes]) -> dict:
        folder = self.session_dir(options)
        meta = self.settings(folder, options)
        frames = self.frames(folder)
        if len(frames) < 2:
            raise CommandError("need at least 2 photos")
        picked = [str(frames[i]) for i in spread(len(frames), self.final_frames)]
        rec, waited = self.reconstruct(picked)
        points, colors = mapping.confident_points(rec)
        result = {"session": folder.name, "frames": len(frames), "frames_used": len(picked), "queue": waited}
        try:
            m = mapping.build_map(rec, meta["camera_height_cm"])
            points = points * m.scale_cm  # centimetres
            png = mapping.render_png(m)
            (folder / "map.png").write_bytes(png)
            result["map_png"] = base64.b64encode(png).decode()
            result["units"] = "cm"
        except ValueError as e:
            result["units"] = "unscaled"
            result["note"] = str(e)
        ply = mapping.write_ply(points, colors)
        (folder / "model.ply").write_bytes(ply)
        centres, _ = mapping.camera_poses(rec.extrinsic)
        (folder / "poses.json").write_text(json.dumps({"frames": [Path(p).name for p in picked],
                                                       "extrinsic": rec.extrinsic.tolist()}))
        result.update({"ply_path": str(folder / "model.ply"), "ply_bytes": len(ply),
                       "parts": -(-len(ply) // PART_BYTES), "points": int(min(len(points), 2_000_000)),
                       "cameras": len(centres)})
        return result

    def download(self, options: dict, images: list[bytes]) -> dict:
        path = self.session_dir(options) / "model.ply"
        if not path.exists():
            raise CommandError("no model yet; send finish first")
        try:
            part = int(options.get("part", "0"))
        except ValueError:
            raise CommandError("part must be a number") from None
        size = path.stat().st_size
        parts = -(-size // PART_BYTES)
        if not 0 <= part < parts:
            raise CommandError(f"part must be between 0 and {parts - 1}")
        with path.open("rb") as f:
            f.seek(part * PART_BYTES)
            chunk = f.read(PART_BYTES)
        return {"session": path.parent.name, "part": part, "parts": parts, "ply_bytes": size,
                "data": base64.b64encode(chunk).decode()}

    def status(self, options: dict, images: list[bytes]) -> dict:
        folder = self.session_dir(options)
        return {"session": folder.name, "frames": len(self.frames(folder)),
                "files": sorted(p.name for p in folder.glob("*") if p.is_file())}


def make_app(mapper: Mapper, names: list[str], api_key: str = "") -> FastAPI:
    app = FastAPI(title="visual geometry mapper")
    stats = {"success": 0, "started": time.time()}

    def unauthorized(request: Request):
        if api_key and request.headers.get("authorization", "") != f"Bearer {api_key}":
            return error(401, "invalid api key", "authentication_error")
        return None

    def error(status, message, kind="invalid_request_error"):
        return JSONResponse({"error": {"message": message, "type": kind, "code": status}}, status_code=status)

    @app.get("/health")
    async def health():
        return PlainTextResponse("")

    @app.get("/metrics")
    async def metrics():
        gpu = mapper.gpu.state()
        lines = [f"vllm:num_requests_running {int(gpu['gpu_busy'])}", f"vllm:num_requests_waiting {gpu['waiting']}",
                 f"vllm:request_success_total {stats['success']}",
                 f"process_start_time_seconds {stats['started']}"]
        return PlainTextResponse("\n".join(lines) + "\n")

    @app.get("/v1/models")
    async def models(request: Request):
        if (denied := unauthorized(request)):
            return denied
        return {"object": "list", "data": [
            {"id": n, "object": "model", "created": int(stats["started"]), "owned_by": "vggt",
             "root": n, "max_model_len": 32768} for n in names]}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        if (denied := unauthorized(request)):
            return denied
        try:
            body = await request.json()
        except ValueError:
            return error(400, "body must be JSON")
        model = body.get("model") or names[0]
        if model not in names:
            return error(404, f"model {model!r} not served here; use one of {names}", "not_found_error")
        try:
            text, images = message_parts(body)
            result = await asyncio.to_thread(mapper.run, text, images)
        except CommandError as e:
            return error(400, str(e))
        stats["success"] += 1
        content = json.dumps(result)
        reply_id, created = f"chatcmpl-{uuid.uuid4().hex}", int(time.time())
        if body.get("stream"):
            def events():
                for delta, finish in (({"role": "assistant", "content": content}, None), ({}, "stop")):
                    chunk = {"id": reply_id, "object": "chat.completion.chunk", "created": created,
                             "model": model, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(events(), media_type="text/event-stream")
        return {"id": reply_id, "object": "chat.completion", "created": created, "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}

    return app


def main() -> None:
    import argparse

    import uvicorn

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="VGGT-1B folder or VGGT-Omega .pt checkpoint")
    p.add_argument("--image-resolution", type=int, default=512, help="VGGT-Omega input size (416 for the reproduce checkpoint)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8030)
    p.add_argument("--served-model-name", nargs="+", default=["alias-visual-geometry"])
    p.add_argument("--data-dir", default="data", help="where sessions (photos, maps, models) are kept")
    p.add_argument("--step-frames", type=int, default=32, help="most photos used for a step")
    p.add_argument("--final-frames", type=int, default=96, help="most photos used for finish")
    args = p.parse_args()

    from .model import load

    mapper = Mapper(load(args.model, args.image_resolution), Path(args.data_dir), args.step_frames, args.final_frames)
    app = make_app(mapper, args.served_model_name, os.environ.get("VLLM_API_KEY", ""))
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

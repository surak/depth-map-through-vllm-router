#!/usr/bin/env python3
"""Send photos to the mapper through any OpenAI-compatible endpoint and save the results.

  export OPENAI_BASE_URL=https://api.blablador.fz-juelich.de/v1 OPENAI_API_KEY=...
  python3 examples/client.py step kitchen photo1.jpg photo2.jpg --camera-height-cm 9
  python3 examples/client.py finish kitchen     # writes kitchen.ply
"""

import argparse
import base64
import json
import mimetypes
import os
import urllib.request

MODEL = "alias-visual-geometry"


def ask(text: str, photos: list[str] = ()) -> dict:
    parts = [{"type": "text", "text": text}]
    for path in photos:
        mime = mimetypes.guess_type(path)[0] or "image/jpeg"
        data = base64.b64encode(open(path, "rb").read()).decode()
        parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}})
    body = {"model": MODEL, "messages": [{"role": "user", "content": parts}]}
    req = urllib.request.Request(
        os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8030/v1").rstrip("/") + "/chat/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json", "Authorization": "Bearer " + os.environ.get("OPENAI_API_KEY", "")})
    reply = json.loads(urllib.request.urlopen(req, timeout=600).read())
    return json.loads(reply["choices"][0]["message"]["content"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["step", "finish", "status"])
    p.add_argument("session")
    p.add_argument("photos", nargs="*")
    p.add_argument("--camera-height-cm", type=float)
    a = p.parse_args()

    text = f"{a.command} session={a.session}"
    if a.camera_height_cm:
        text += f" camera_height_cm={a.camera_height_cm}"
    result = ask(text, a.photos)
    if "map_png" in result:
        open(f"{a.session}-map.png", "wb").write(base64.b64decode(result.pop("map_png")))
        print(f"map saved to {a.session}-map.png")
    if a.command == "finish" and "parts" in result:
        with open(f"{a.session}.ply", "wb") as f:
            for part in range(result["parts"]):
                f.write(base64.b64decode(ask(f"download session={a.session} part={part}")["data"]))
        print(f"model saved to {a.session}.ply")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

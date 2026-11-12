# depth-map-through-vllm-router

[VGGT-Ω](https://github.com/facebookresearch/vggt-omega) and [VGGT-1B](https://github.com/facebookresearch/vggt)
3D mapping served with an OpenAI-compatible API, so it can sit next to vLLM models behind a router
(synapse / Blablador) as **`alias-visual-geometry-omega`** (VGGT-Ω, the default) and
**`alias-visual-geometry`** (VGGT-1B).

Website: https://surak.github.io/depth-map-through-vllm-router/

It is built for a small robot car that maps a room by itself: it stops, takes photos, sends them
with `step`, and gets back where it is, which directions are still unexplored and how far the next
obstacle is. When nothing is left to explore it sends `finish` and downloads a coloured point cloud.

## Talking to it

It is a normal chat completion. The photos go in as base64 `image_url` parts, the text is a
command, and the answer's `content` is JSON.

| Command | What it does |
|---|---|
| `step session=<name> [camera_height_cm=<cm>]` | Add the photos, rebuild the map, answer with pose, frontiers, obstacle and map |
| `finish session=<name>` | Reconstruct from all photos (up to 96) and keep `model.ply` |
| `download session=<name> part=<n>` | The PLY in 3 MB base64 pieces, so it fits through the router |
| `status session=<name>` | Photos and files of a session |
| `queue` | How busy the GPU is and how many other sessions are mapping |

A `step` answer:

```json
{"session": "kitchen", "frames": 12, "frames_used": 12,
 "pose": {"x_cm": 44.0, "z_cm": 35.0, "heading_deg": 77.4},
 "frontiers": [{"angle": 20.9, "dist_cm": 45.2, "size_cells": 12}],
 "obstacle_ahead_cm": 29.5, "explored_done": false, "map_png": "<base64>"}
```

- **Angles** are in degrees, relative to where the camera looks now; positive means left.
- **Positions** are in centimetres from where the first photo was taken.
- **`map_png`** is a top-down map: white floor, black obstacles, grey unknown, orange frontiers,
  blue path and a red marker for the car.
- **Pictures for a vision-language model:** the map is meant to be shown to one, together with
  the latest photo, so it can choose the next move.

### Other people

The GPU works on one request at a time, first come first served. Every answer has a `queue` part,
so callers know when they are sharing:

```json
"queue": {"ahead_on_arrival": 2, "waited_s": 3.4, "gpu_busy": true, "waiting": 1,
          "other_sessions_active": 3,
          "message": "2 request(s) are using or waiting for the GPU; a step takes a few seconds, a finish up to half a minute"}
```

- `ahead_on_arrival` and `waited_s`: how many requests were before this one, and how long it waited.
- `gpu_busy` and `waiting`: the state right now.
- `other_sessions_active`: other sessions with a request in the last 10 minutes. Other people's
  session names are never shown.

`queue` alone answers with just this part. Two requests for the same session never run at the same
time, so photos sent from two devices at once can't overwrite each other. `/metrics` reports the
running and waiting counts like vLLM does.

[`examples/client.py`](examples/client.py) does this from the command line.

## Sending photos from a phone

```bash
cp .env.example .env      # put your router URL and key in it
./run.sh serve            # prints the address to open on the phone
```

The page on the phone lets you pick or take photos, set the camera height, see the top-down map
after each batch and build the 3D model. A banner at the top shows whether other people are mapping
or the GPU is busy, and the page says how long your photos waited. The model opens in a viewer you can turn with a finger, and
can be downloaded as PLY.

- **Photo size:** photos are shrunk to 1024 px in the browser before they are sent.
- **Key:** the server adds the API key from `.env`, so the key never reaches the phone. Anyone on
  your network who can open the page can use it, though.
- **Tests:** `./run.sh test` runs them.

## How the map is made

VGGT returns camera poses and a point cloud in the first camera's frame (x right, y down,
z forward), but without a scale.

1. **Scale.** The camera is assumed to be roughly level and at a known height, `camera_height_cm`,
   which is remembered per session. The densest height band below the cameras is taken as the
   floor, and the camera height turns VGGT units into centimetres.
2. **Grid.** Points near the floor become free cells. Points between 4 cm and the camera height
   plus 15 cm become obstacles. Everything else is unknown.
3. **Frontiers.** Free cells that touch unknown ones are grouped in 30° sectors.

Each step reruns VGGT on up to 32 photos: the first, the latest and an even spread between them.
That keeps the first camera as the origin, so poses stay comparable between steps. On an RTX 3090,
a step takes about 1–2 s, and 96 photos take about 20 s and 13 GB.

**Tilt the camera a little down so it sees the floor.** Without a visible floor the answer has a
`note` instead of a map.

## Running it

```bash
uv venv -p 3.12 .venv
uv pip install -p .venv/bin/python torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
uv pip install -p .venv/bin/python -e '.[model,test]'
.venv/bin/python -m pytest
VLLM_API_KEY=secret .venv/bin/python -m mapper.server --model /path/to/VGGT-1B --port 8030
```

For VGGT-Ω, which needs torch 2.6 or newer, use a second venv and point `--model` at a `.pt`
checkpoint from [facebook/VGGT-Omega](https://huggingface.co/facebook/VGGT-Omega) (gated):

```bash
uv venv -p 3.12 .venv-omega
uv pip install -p .venv-omega/bin/python -e '.[omega,test]'
VLLM_API_KEY=secret .venv-omega/bin/python -m mapper.server --model /path/to/vggt_omega_1b_512.pt --port 8031
```

Use `--image-resolution 416` with `vggt_omega_1b_416_reproduce.pt`.

The server answers `/health`, `/v1/models`, `/metrics` and `/v1/chat/completions` (also with
`stream: true`), which is what the synapse supervisor and client check.

On HAICLUSTER, [`launchers/visual-geometry.slurm`](launchers/visual-geometry.slurm) is the job.
It uses one GPU on haicluster1 and port 8030, reads its key from
`synapse-external/backend-key-8030`, and registers with synapse through `supervisor.py`, like the
other models. The code is checked out in `/p/haicluster/llama/FastChat/depth-map-through-vllm-router`,
next to the other models, and the fleet runs a copy of the launcher from `blablador-fleet/launchers/`.
[`launchers/visual-geometry-omega.slurm`](launchers/visual-geometry-omega.slurm) is the VGGT-Ω job:
one GPU on haicluster2 (haicluster1 is full), port 8031, `.venv-omega`, and its own sessions folder.

## Licence

The code here is MIT. The VGGT-1B and VGGT-Ω weights are released by Meta for non-commercial research use only.

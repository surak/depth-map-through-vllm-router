"""Turn a VGGT reconstruction into something a small car can navigate by.

VGGT gives camera poses and a point cloud in the frame of the first camera (OpenCV axes: x right,
y down, z forward) and with an unknown scale. We assume the camera is mounted roughly level at a
known height, find the floor, use the height to get centimetres, and flatten everything into a
top-down grid: floor, obstacles and unknown. Frontiers are free cells next to unknown ones, the
places worth driving to next.
"""

from __future__ import annotations

import io
import math
from collections import deque
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

FREE, OBSTACLE, UNKNOWN = 0, 1, 2


@dataclass
class Reconstruction:
    extrinsic: np.ndarray  # (S, 3, 4) world-to-camera
    points: np.ndarray  # (S, H, W, 3) world points
    conf: np.ndarray  # (S, H, W) confidence
    colors: np.ndarray  # (S, H, W, 3) RGB in 0..1


def camera_poses(extrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Camera centres (S, 3) and forward directions (S, 3) in world coordinates."""
    rot, trans = extrinsic[:, :, :3], extrinsic[:, :, 3]
    centres = -np.einsum("sji,sj->si", rot, trans)
    forward = rot[:, 2, :]  # third row of R = camera z axis in world
    return centres, forward


def confident_points(rec: Reconstruction, keep: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """The most confident fraction `keep` of the points, with their colours."""
    pts = rec.points.reshape(-1, 3)
    cols = rec.colors.reshape(-1, 3)
    conf = rec.conf.reshape(-1)
    ok = np.isfinite(pts).all(axis=1)
    pts, cols, conf = pts[ok], cols[ok], conf[ok]
    if len(conf):
        mask = conf >= np.quantile(conf, 1 - keep)
        pts, cols = pts[mask], cols[mask]
    return pts, cols


def floor_height(points: np.ndarray, camera_y: float) -> float:
    """World y of the floor: the densest height band below the cameras."""
    below = points[points[:, 1] > camera_y]
    if len(below) < 50:
        raise ValueError("cannot see the floor yet; tilt the camera a little down")
    span = np.quantile(below[:, 1], 0.98) - camera_y
    hist, edges = np.histogram(below[:, 1], bins=60, range=(camera_y + 0.15 * span, camera_y + span))
    i = int(np.argmax(hist))
    return float((edges[i] + edges[i + 1]) / 2)


@dataclass
class Map:
    grid: np.ndarray  # (rows, cols) of FREE / OBSTACLE / UNKNOWN, row 0 = largest z
    origin: tuple[float, float]  # (x, z) in cm of the centre of cell (row 0, col 0)
    cell_cm: float
    path_cm: np.ndarray  # (S, 2) camera positions x, z in cm
    heading: float  # yaw of the last camera, radians, 0 = along +z, positive towards +x
    frontiers: list[dict]
    obstacle_ahead_cm: float | None
    scale_cm: float

    def cell_centre(self, row: int, col: int) -> tuple[float, float]:
        return self.origin[0] + col * self.cell_cm, self.origin[1] - row * self.cell_cm


def relative_angle(heading: float, dx: float, dz: float) -> float:
    """Degrees to turn from `heading` to face (dx, dz); positive means left, like the car."""
    rel = math.atan2(dx, dz) - heading
    rel = (rel + math.pi) % (2 * math.pi) - math.pi
    return -math.degrees(rel)


def build_map(rec: Reconstruction, camera_height_cm: float, cell_cm: float = 5.0,
              min_frontier_cells: int = 3) -> Map:
    centres, forward = camera_poses(rec.extrinsic)
    pts, _ = confident_points(rec)
    cam_y = float(np.median(centres[:, 1]))
    floor_y = floor_height(pts, cam_y)
    scale = camera_height_cm / (floor_y - cam_y)  # cm per world unit

    above_floor = (floor_y - pts[:, 1]) * scale
    xz = pts[:, [0, 2]] * scale
    path = centres[:, [0, 2]] * scale
    is_floor = np.abs(above_floor) < 3
    is_obstacle = (above_floor > 4) & (above_floor < camera_height_cm + 15)

    lo = np.minimum(xz.min(axis=0), path.min(axis=0)) - 2 * cell_cm
    hi = np.maximum(xz.max(axis=0), path.max(axis=0)) + 2 * cell_cm
    cols = int(math.ceil((hi[0] - lo[0]) / cell_cm)) + 1
    rows = int(math.ceil((hi[1] - lo[1]) / cell_cm)) + 1
    if rows * cols > 4_000_000:
        raise ValueError("map too large; the scale estimate is probably wrong")

    def cells(p):
        c = np.clip(((p[:, 0] - lo[0]) / cell_cm).round().astype(int), 0, cols - 1)
        r = np.clip(((hi[1] - p[:, 1]) / cell_cm).round().astype(int), 0, rows - 1)
        return r, c

    floor_count = np.zeros((rows, cols), int)
    obstacle_count = np.zeros((rows, cols), int)
    np.add.at(floor_count, cells(xz[is_floor]), 1)
    np.add.at(obstacle_count, cells(xz[is_obstacle]), 1)
    grid = np.full((rows, cols), UNKNOWN, np.int8)
    grid[floor_count > 0] = FREE
    pr, pc = cells(path)
    grid[pr, pc] = FREE  # the car has been there
    grid[obstacle_count >= 3] = OBSTACLE

    fwd = forward[-1]
    heading = math.atan2(fwd[0], fwd[2])
    here = path[-1]
    m = Map(grid, (float(lo[0]), float(hi[1])), cell_cm, path, heading, [], None, float(scale))
    m.frontiers = find_frontiers(m, here, min_frontier_cells)
    m.obstacle_ahead_cm = obstacle_ahead(m, here)
    return m


def find_frontiers(m: Map, here: np.ndarray, min_cells: int, sector_deg: float = 30.0) -> list[dict]:
    """Groups of touching frontier cells, split into sectors so one long edge gives several goals."""
    g = m.grid
    rows, cols = g.shape
    unknown = np.pad(g == UNKNOWN, 1)
    near_unknown = unknown[:-2, 1:-1] | unknown[2:, 1:-1] | unknown[1:-1, :-2] | unknown[1:-1, 2:]
    frontier = (g == FREE) & near_unknown
    sector = np.zeros(g.shape, int)
    for r, c in zip(*np.nonzero(frontier)):
        x, z = m.cell_centre(r, c)
        sector[r, c] = int((relative_angle(m.heading, x - here[0], z - here[1]) + 180) // sector_deg)
    seen = np.zeros_like(frontier)
    found = []
    for r0, c0 in zip(*np.nonzero(frontier)):
        if seen[r0, c0]:
            continue
        group, todo = [], deque([(r0, c0)])
        seen[r0, c0] = True
        while todo:
            r, c = todo.popleft()
            group.append((r, c))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < rows and 0 <= cc < cols and frontier[rr, cc] and not seen[rr, cc] \
                            and sector[rr, cc] == sector[r0, c0]:
                        seen[rr, cc] = True
                        todo.append((rr, cc))
        if len(group) < min_cells:
            continue
        x, z = np.mean([m.cell_centre(r, c) for r, c in group], axis=0)
        dx, dz = x - here[0], z - here[1]
        found.append({
            "angle": round(relative_angle(m.heading, dx, dz), 1),
            "dist_cm": round(float(math.hypot(dx, dz)), 1),
            "size_cells": len(group),
        })
    found.sort(key=lambda f: f["dist_cm"])
    return found


def obstacle_ahead(m: Map, here: np.ndarray, half_angle: float = 15.0) -> float | None:
    """Distance to the nearest obstacle cell within ±half_angle degrees of the heading."""
    best = None
    for r, c in zip(*np.nonzero(m.grid == OBSTACLE)):
        x, z = m.cell_centre(r, c)
        dx, dz = x - here[0], z - here[1]
        if abs(relative_angle(m.heading, dx, dz)) <= half_angle:
            d = math.hypot(dx, dz)
            best = d if best is None else min(best, d)
    return None if best is None else round(float(best), 1)


def render_png(m: Map, target_px: int = 400) -> bytes:
    """Top-down map: white floor, black obstacles, grey unknown, orange frontier, blue path."""
    palette = np.array([[255, 255, 255], [20, 20, 20], [150, 150, 150]], np.uint8)
    img = palette[m.grid]
    g = m.grid
    unknown = np.pad(g == UNKNOWN, 1)
    near = unknown[:-2, 1:-1] | unknown[2:, 1:-1] | unknown[1:-1, :-2] | unknown[1:-1, 2:]
    img[(g == FREE) & near] = (255, 140, 0)
    k = max(1, target_px // max(g.shape))
    pic = Image.fromarray(img).resize((g.shape[1] * k, g.shape[0] * k), Image.NEAREST)
    draw = ImageDraw.Draw(pic)

    def px(x, z):
        return ((x - m.origin[0]) / m.cell_cm * k + k / 2, (m.origin[1] - z) / m.cell_cm * k + k / 2)

    pts = [px(x, z) for x, z in m.path_cm]
    if len(pts) > 1:
        draw.line(pts, fill=(30, 90, 255), width=2)
    x, y = pts[-1]
    r = max(3, k * 2)
    draw.ellipse((x - r, y - r, x + r, y + r), fill=(220, 0, 0))
    draw.line((x, y, x + 3 * r * math.sin(m.heading), y - 3 * r * math.cos(m.heading)), fill=(220, 0, 0), width=2)
    out = io.BytesIO()
    pic.save(out, "PNG")
    return out.getvalue()


def write_ply(points: np.ndarray, colors: np.ndarray, max_points: int = 2_000_000) -> bytes:
    """Binary little-endian PLY with RGB colours."""
    if len(points) > max_points:
        pick = np.random.default_rng(0).choice(len(points), max_points, replace=False)
        points, colors = points[pick], colors[pick]
    rec = np.empty(len(points), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                       ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    rec["x"], rec["y"], rec["z"] = points.T
    rgb = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
    rec["red"], rec["green"], rec["blue"] = rgb.T
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(rec)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    return header.encode() + rec.tobytes()

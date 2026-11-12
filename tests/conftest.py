import numpy as np
import pytest

from mapper.mapping import Reconstruction


def synthetic_room(cameras: int = 2) -> Reconstruction:
    """Cameras at the origin looking along +z, floor 1 unit below, a wall 3 units ahead."""
    xs, zs = np.meshgrid(np.linspace(-2, 2, 80), np.linspace(0.3, 3, 60))
    floor = np.stack([xs, np.ones_like(xs), zs], -1).reshape(-1, 3)
    wx, wy = np.meshgrid(np.linspace(-0.5, 0.5, 40), np.linspace(0.0, 0.6, 20))
    wall = np.stack([wx, wy, np.full_like(wx, 3.0)], -1).reshape(-1, 3)
    pts = np.concatenate([floor, wall]).astype(np.float32)
    n = len(pts)
    extrinsic = np.tile(np.hstack([np.eye(3), np.zeros((3, 1))]), (cameras, 1, 1)).astype(np.float32)
    return Reconstruction(
        extrinsic=extrinsic,
        points=np.tile(pts, (cameras, 1)).reshape(cameras, 1, n, 3),
        conf=np.ones((cameras, 1, n), np.float32),
        colors=np.full((cameras, 1, n, 3), 0.5, np.float32),
    )


class FakeRunner:
    def __init__(self):
        self.calls = []

    def __call__(self, paths):
        self.calls.append(list(paths))
        return synthetic_room(len(paths))


@pytest.fixture
def room():
    return synthetic_room()

import math

import numpy as np

from mapper import mapping


def test_scale_from_camera_height(room):
    m = mapping.build_map(room, camera_height_cm=10)
    assert math.isclose(m.scale_cm, 10, rel_tol=0.1)


def test_wall_ahead_is_an_obstacle(room):
    m = mapping.build_map(room, camera_height_cm=10)
    assert m.obstacle_ahead_cm is not None
    assert 25 <= m.obstacle_ahead_cm <= 35


def test_frontiers_on_both_sides(room):
    m = mapping.build_map(room, camera_height_cm=10)
    angles = [f["angle"] for f in m.frontiers]
    assert any(a > 20 for a in angles) and any(a < -20 for a in angles)


def test_relative_angle_left_is_positive():
    assert mapping.relative_angle(0, -1, 1) == 45  # ahead and to the left
    assert mapping.relative_angle(0, 1, 1) == -45
    assert abs(mapping.relative_angle(math.pi / 2, 0, 1)) == 90


def test_camera_pose_of_moved_camera():
    rot = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], float)  # camera turned to face +x
    centre = np.array([1.0, 0, 2.0])
    ext = np.hstack([rot, (-rot @ centre)[:, None]])[None]
    c, f = mapping.camera_poses(ext)
    assert np.allclose(c[0], centre)
    assert np.allclose(f[0], [1, 0, 0])


def test_png_and_ply(room):
    m = mapping.build_map(room, camera_height_cm=10)
    assert mapping.render_png(m).startswith(b"\x89PNG")
    ply = mapping.write_ply(np.zeros((5, 3)), np.ones((5, 3)))
    assert ply.startswith(b"ply\n") and b"element vertex 5" in ply
    assert len(ply.split(b"end_header\n", 1)[1]) == 5 * 15

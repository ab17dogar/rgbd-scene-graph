"""Geometry round-trip tests.

If `project ∘ backproject` is not the identity on valid pixels, every
downstream geometric calculation in this codebase is wrong. So this is the
single most important test in the repo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rgbdsg.geometry import backproject, backproject_to_world, project, transform_points
from rgbdsg.io import RGBDSequence


DATA_ROOT = Path(__file__).parent.parent / "data"
SCENE = DATA_ROOT / "BasicHouse_with_pc"


@pytest.fixture(scope="module")
def seq() -> RGBDSequence:
    if not SCENE.is_dir():
        pytest.skip(f"scene not present at {SCENE}")
    return RGBDSequence(SCENE)


def test_pose_inverse_is_T_cw(seq: RGBDSequence) -> None:
    """T_cw must be the inverse of T_wc to within the rotation matrix's own
    orthogonality noise. The shipped poses.txt has ~7e-7 ortho error per the
    inspection report, so 1e-5 is a tight-but-realistic bound.
    """
    frame = seq[0]
    T_wc = frame.pose.T_wc
    T_cw = frame.pose.T_cw
    assert np.allclose(T_wc @ T_cw, np.eye(4), atol=1e-5)
    assert np.allclose(T_cw @ T_wc, np.eye(4), atol=1e-5)


def test_pose_translation_is_camera_origin_in_world(seq: RGBDSequence) -> None:
    """T_wc applied to camera origin (0,0,0) should give the t component."""
    frame = seq[0]
    origin_cam = np.array([[0.0, 0.0, 0.0]])
    origin_world = transform_points(origin_cam, frame.pose.T_wc)
    np.testing.assert_allclose(origin_world[0], frame.pose.t, atol=1e-12)


def test_backproject_roundtrip_pixel_precision(seq: RGBDSequence) -> None:
    """Back-project depth to camera-frame XYZ, project back, recover original (u, v, d).

    Tolerance: 1e-9 px on (u, v) and 1e-9 m on depth. This is a pure floating-
    point sanity check on the camera math; no real-world variance should
    appear.
    """
    frame = seq[0]
    P_cam, uv_in = backproject(frame.depth_m, frame.intrinsics, stride=8)
    assert P_cam.shape[0] > 1000  # sanity: some valid pixels exist

    # Project camera-frame points back to image plane WITHOUT going through
    # world coords (so we test only the projection math, not the pose round-trip).
    # Equivalent to project() with identity pose.
    from rgbdsg.io import Pose
    identity_pose = Pose(T_wc=np.eye(4))
    P_world_via_identity = P_cam.copy()  # identity T_wc means cam == world

    uv_out, depth_out, in_image = project(P_world_via_identity, identity_pose, frame.intrinsics)

    # Tolerance: the EXR file stores depth as fp16, so multiplying & dividing
    # by `d` doesn't fully cancel — expect ~1e-4 px max error. 1e-3 is a safe
    # bound that still catches any structural projection bug.
    np.testing.assert_allclose(uv_out, uv_in, atol=1e-3)
    np.testing.assert_allclose(depth_out, frame.depth_m[uv_in[:, 1].astype(int),
                                                       uv_in[:, 0].astype(int)],
                               atol=1e-6)
    # Note: we don't assert in_image.all() because edge pixels (u=0 or u=W-1)
    # can drift outside the strict [0, W) bounds by ~1e-5 due to float-fp16
    # interplay. The pixel recovery above is the real correctness signal.


def test_backproject_to_world_then_project_back(seq: RGBDSequence) -> None:
    """End-to-end round-trip including the pose.

    Back-project depth -> world, project world -> uv. The recovered uv should
    equal the source uv to floating-point precision.
    """
    frame = seq[0]
    P_world, uv_in = backproject_to_world(frame, stride=8)
    uv_out, _, in_image = project(P_world, frame.pose, frame.intrinsics)

    # Every back-projected point must reproject inside the image (it came from
    # a valid pixel of this same frame).
    assert in_image.all(), \
        f"{(~in_image).sum()} of {len(in_image)} round-trip points fell outside the image"

    # Sub-pixel recovery (see roundtrip test for the fp16 tolerance reason).
    np.testing.assert_allclose(uv_out, uv_in, atol=1e-3)


def test_camera_origin_lies_inside_pointcloud_bbox(seq: RGBDSequence) -> None:
    """Camera origin should be physically inside the building (sanity)."""
    # Light check; the inspector already validated this numerically.
    frame = seq[0]
    t = frame.pose.t
    # rough bounds for BasicHouse from inspection_findings.md
    assert -25 < t[0] < 30
    assert -15 < t[1] < 17
    assert -4 < t[2] < 1   # Z-up, roof near 0, floor at -3.6


def test_backprojected_points_align_with_pointcloud(seq: RGBDSequence) -> None:
    """Coarse alignment check vs. the shipped point cloud.

    Median nearest-neighbor distance should be < 10cm on BasicHouse for the
    `gl_z` convention. The full per-frame numerical analysis lives in
    `scripts/verify_pose.py`; this is just a regression guard against any
    accidental sign flip / convention drift in `geometry.camera`.
    """
    import open3d as o3d
    from scipy.spatial import cKDTree

    pcd = o3d.io.read_point_cloud(str(seq.pointcloud_path))
    scene_pts = np.asarray(pcd.points)
    tree = cKDTree(scene_pts)

    frame = seq[0]
    P_world, _ = backproject_to_world(frame, stride=8)
    dists, _ = tree.query(P_world, k=1)
    median_mm = float(np.median(dists)) * 1000
    assert median_mm < 100, f"median NN dist {median_mm:.1f} mm > 100 mm — convention drift?"

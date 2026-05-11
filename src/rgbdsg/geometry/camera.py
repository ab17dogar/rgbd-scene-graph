"""Camera projection and back-projection in the verified gl_z convention."""

from __future__ import annotations

import numpy as np

from rgbdsg.io import Frame, Intrinsics, Pose


def backproject(
    depth_m: np.ndarray,
    intrinsics: Intrinsics,
    valid_mask: np.ndarray | None = None,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project a depth image into camera-frame XYZ points."""
    H, W = depth_m.shape
    if intrinsics.width != W or intrinsics.height != H:
        raise ValueError(
            f"depth shape {(H, W)} does not match intrinsics "
            f"{(intrinsics.height, intrinsics.width)}"
        )

    if valid_mask is None:
        valid_mask = (
            np.isfinite(depth_m) & (depth_m > 0)
            & (depth_m < intrinsics.saturation_threshold_m)
        )

    if stride > 1:
        stride_mask = np.zeros_like(valid_mask)
        stride_mask[::stride, ::stride] = True
        valid_mask = valid_mask & stride_mask

    vs, us = np.where(valid_mask)
    d = depth_m[vs, us].astype(np.float64)

    # Normalised image plane coordinates (meters per meter of depth).
    x = (us.astype(np.float64) - intrinsics.cx) / intrinsics.fx
    y = (vs.astype(np.float64) - intrinsics.cy) / intrinsics.fy

    P_cam = np.stack([x * d, -y * d, -d], axis=1)
    uv = np.stack([us.astype(np.float64), vs.astype(np.float64)], axis=1)
    return P_cam, uv


def backproject_to_world(
    frame: Frame,
    valid_mask: np.ndarray | None = None,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project depth and transform to world coordinates."""
    P_cam, uv = backproject(frame.depth_m, frame.intrinsics, valid_mask, stride)
    P_world = transform_points(P_cam, frame.pose.T_wc)
    return P_world, uv


def project(
    P_world: np.ndarray,
    pose: Pose,
    intrinsics: Intrinsics,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world-frame points to pixel coordinates."""
    # Transform world -> camera (gl convention; -Z is forward).
    P_h = np.concatenate([P_world, np.ones((P_world.shape[0], 1))], axis=1)
    P_cam = (pose.T_cw @ P_h.T).T[:, :3]

    depth = -P_cam[:, 2]

    safe_depth = np.where(np.abs(depth) > 1e-9, depth, 1e-9)
    u = (P_cam[:, 0] / safe_depth) * intrinsics.fx + intrinsics.cx
    v = (-P_cam[:, 1] / safe_depth) * intrinsics.fy + intrinsics.cy
    uv = np.stack([u, v], axis=1)

    in_image = (
        (depth > intrinsics.near_m) & (depth < intrinsics.far_m)
        & (u >= 0) & (u < intrinsics.width)
        & (v >= 0) & (v < intrinsics.height)
    )
    return uv, depth, in_image


def transform_points(P: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to an (N, 3) point array, returning (N, 3)."""
    if P.size == 0:
        return P.copy()
    P_h = np.concatenate([P, np.ones((P.shape[0], 1), dtype=P.dtype)], axis=1)
    return (T @ P_h.T).T[:, :3]


def look_direction_world(pose: Pose) -> np.ndarray:
    """Unit vector in world coords pointing forward from the camera (-Z_cam)."""
    return pose.R @ np.array([0.0, 0.0, -1.0])


def up_direction_world(pose: Pose) -> np.ndarray:
    """Unit vector in world coords pointing up from the camera (+Y_cam)."""
    return pose.R @ np.array([0.0, 1.0, 0.0])

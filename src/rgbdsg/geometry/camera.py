"""Camera projection and back-projection in the verified gl_z convention.

The convention used throughout this codebase is documented in detail in
```

  * Camera frame: OpenGL/Blender. +X right, +Y up, **-Z forward** (camera
    looks down -Z in its own frame).
  * Depth values: planar Z-distance (the camera-frame Z component of the
    surface point), NOT Euclidean ray length. The scalar `d` returned by
    EXR is the unsigned magnitude — the world point sits at camera-frame
    Z = -d.
  * Pose: `T_wc` (world-from-camera). Translation column is the camera
    origin in world coords.

These conventions were established against ground truth in
`scripts/verify_pose.py` (median NN distance to scene.ply: 41 mm with this
formula vs. 292 mm with OpenCV-style +Z-forward).
"""

from __future__ import annotations

import numpy as np

from rgbdsg.io import Frame, Intrinsics, Pose


# ---------- back-projection -------------------------------------------------

def backproject(
    depth_m: np.ndarray,
    intrinsics: Intrinsics,
    valid_mask: np.ndarray | None = None,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project a depth image into camera-frame XYZ points.

    Args:
        depth_m: HxW float array of planar Z-distance in meters.
        intrinsics: pinhole intrinsics.
        valid_mask: optional HxW bool mask; pixels where False are dropped.
            Default: drop pixels where depth is non-finite, <=0, or above the
            saturation threshold (sky / past-far rays).
        stride: pixel subsampling stride. 1 = every pixel, 4 = every 4th
            pixel in each dim (16x fewer points). Useful for fast viz.

    Returns:
        P_cam: (N, 3) float64 camera-frame points (gl_z convention).
        uv:    (N, 2) float64 pixel coordinates of each returned point. Used
            downstream to attach RGB / detection labels to specific points.
    """
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
        # Build a stride mask and AND with validity. Doing it this way (rather
        # than slicing the arrays first) keeps the (u, v) coordinates correct.
        stride_mask = np.zeros_like(valid_mask)
        stride_mask[::stride, ::stride] = True
        valid_mask = valid_mask & stride_mask

    vs, us = np.where(valid_mask)
    d = depth_m[vs, us].astype(np.float64)

    # Normalised image plane coordinates (meters per meter of depth).
    x = (us.astype(np.float64) - intrinsics.cx) / intrinsics.fx
    y = (vs.astype(np.float64) - intrinsics.cy) / intrinsics.fy

    # gl_z: image v-axis points DOWN, camera +Y points UP, so flip y;
    # camera looks down -Z so the world point is at Z_cam = -d.
    P_cam = np.stack([x * d, -y * d, -d], axis=1)
    uv = np.stack([us.astype(np.float64), vs.astype(np.float64)], axis=1)
    return P_cam, uv


def backproject_to_world(
    frame: Frame,
    valid_mask: np.ndarray | None = None,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project depth and transform to world coordinates.

    Returns:
        P_world: (N, 3) world-frame points.
        uv:      (N, 2) the source pixel coordinates.
    """
    P_cam, uv = backproject(frame.depth_m, frame.intrinsics, valid_mask, stride)
    P_world = transform_points(P_cam, frame.pose.T_wc)
    return P_world, uv


# ---------- projection (world -> pixel) -------------------------------------

def project(
    P_world: np.ndarray,
    pose: Pose,
    intrinsics: Intrinsics,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world-frame points to pixel coordinates.

    Args:
        P_world: (N, 3) world-frame points.
        pose: T_wc for this frame.
        intrinsics: pinhole intrinsics.

    Returns:
        uv:        (N, 2) pixel coordinates (float64; unrounded).
        depth:     (N,)   planar Z-distance in front of camera (positive for
                          points that the camera can see; negative for points
                          behind it).
        in_image:  (N,)   bool mask: True iff the projected pixel is inside
                          [0, W) × [0, H) AND the point is in front of the
                          camera AND between near and far.
    """
    # Transform world -> camera (gl convention; -Z is forward).
    P_h = np.concatenate([P_world, np.ones((P_world.shape[0], 1))], axis=1)
    P_cam = (pose.T_cw @ P_h.T).T[:, :3]

    # Camera-frame Z is negative for points the camera sees. Define depth as
    # the positive distance in front of the camera so we can compare against
    # `near_m`/`far_m` directly.
    depth = -P_cam[:, 2]

    # Image plane: u = fx * (X / -Z) + cx,  v = fy * (-Y / -Z) + cy
    # (negative Z and Y flip cancel out in the standard pinhole formula.)
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


# ---------- low-level helpers -----------------------------------------------

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

"""Typed loaders for the HiWi-challenge RGB-D bundles.

Each scene directory has the same layout (`data/<scene>/`):
    rgb/000000.png ...               RGB frames
    depth_exr/000000.exr ...         linear-depth EXR (fp16, meters; planar Z)
    depth_png16/000000.png ...       same depth scaled to uint16 (we ignore)
    pose/poses.txt                   Nx16 row-major 4x4 T_wc
    frames.json                      per-frame quaternion + translation + ts
    camera_info.json                 intrinsics + near/far
    pointcloud/scene.ply             architectural ground-truth point cloud
    _ifcgeom_scene.obj               IFC geometry baked to OBJ
    _ifcgeom_scene.labels.json       per-mesh-group ifc_class + name

This module is the only place that knows about that layout. Everything
downstream consumes typed objects (`Frame`, `Intrinsics`, `Pose`).
"""

from __future__ import annotations

import os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from PIL import Image


# ---------- typed wrappers --------------------------------------------------

@dataclass(frozen=True)
class Intrinsics:
    """Pinhole camera intrinsics. Width/height in pixels, focal in pixels."""
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    near_m: float
    far_m: float
    fps: float

    @property
    def K(self) -> np.ndarray:
        """3x3 intrinsics matrix."""
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]], dtype=np.float64)

    @property
    def saturation_threshold_m(self) -> float:
        """Above this depth, treat the pixel as 'no hit' (sky / past far plane).

        We use 0.95 * far rather than far itself because Blender's float depth
        encodes 'no hit' at the fp16 saturation value (65504) which is well
        above far_m, but real geometry can also approach far_m from below for
        long-range views. 0.95 is a safe gap.
        """
        return 0.95 * self.far_m


@dataclass(frozen=True)
class Pose:
    """4x4 world-from-camera transform.

    Convention: T_wc — applying it to a camera-frame point yields a world-
    frame point. Camera-frame axes are OpenGL/Blender (+Y up, -Z forward).
    See `docs/coordinates.md`.
    """
    T_wc: np.ndarray  # shape (4, 4), float64

    def __post_init__(self) -> None:
        # Validate shape and bottom row at construction so downstream code can
        # trust it. Numerical tolerance is generous because file roundtrip
        # introduces ~1e-5 noise in the bottom row of some legacy formats.
        assert self.T_wc.shape == (4, 4), f"expected 4x4, got {self.T_wc.shape}"
        bot = self.T_wc[3]
        assert np.allclose(bot, [0, 0, 0, 1], atol=1e-4), \
            f"pose bottom row not [0,0,0,1]: {bot}"

    @property
    def R(self) -> np.ndarray:
        return self.T_wc[:3, :3]

    @property
    def t(self) -> np.ndarray:
        """Camera origin in world coordinates."""
        return self.T_wc[:3, 3]

    @cached_property
    def T_cw(self) -> np.ndarray:
        """4x4 camera-from-world transform (inverse of T_wc)."""
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.R.T
        T[:3, 3] = -self.R.T @ self.t
        return T


@dataclass(frozen=True)
class Frame:
    """One frame of the RGB-D sequence."""
    index: int
    timestamp_s: float
    rgb: np.ndarray              # HxWx3 uint8, sRGB
    depth_m: np.ndarray          # HxW float32, linear meters (planar Z)
    pose: Pose
    intrinsics: Intrinsics

    @property
    def valid_depth_mask(self) -> np.ndarray:
        """Boolean HxW mask: True where depth is finite, positive, < far.

        Used to suppress 'no hit' pixels (sky / past-far rays) before any
        downstream geometry. See `docs/inspection_findings.md` §4.
        """
        d = self.depth_m
        return np.isfinite(d) & (d > 0) & (d < self.intrinsics.saturation_threshold_m)


# ---------- file readers (low level) ----------------------------------------

def _read_rgb(path: Path) -> np.ndarray:
    """Load an sRGB image as HxWx3 uint8 (NOT BGR; PIL gives RGB by default)."""
    return np.array(Image.open(path).convert("RGB"))


def _read_depth_exr(path: Path) -> np.ndarray:
    """Read a single-channel float EXR, returning HxW float32 in meters."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
    if img is None:
        raise RuntimeError(f"cv2 returned None for {path} (OpenEXR plugin?)")
    if img.ndim == 3:
        img = img[..., 0]
    return img.astype(np.float32)


def _load_intrinsics(scene_dir: Path) -> Intrinsics:
    info = json.loads((scene_dir / "camera_info.json").read_text())
    K = info["intrinsics"]
    # tolerate symmetric pinhole only — assert no skew or distortion
    assert K[0][1] == 0.0, f"non-zero skew {K[0][1]}"
    assert all(abs(d) < 1e-9 for d in info.get("distortion", [0])), \
        "non-zero distortion not supported"
    return Intrinsics(
        width=info["width"], height=info["height"],
        fx=K[0][0], fy=K[1][1], cx=K[0][2], cy=K[1][2],
        near_m=info.get("near_m", 0.05), far_m=info.get("far_m", 100.0),
        fps=info.get("fps", 10.0),
    )


def _load_poses(scene_dir: Path) -> np.ndarray:
    """Return Nx4x4 stack of T_wc matrices (preferring poses.txt)."""
    flat = np.loadtxt(scene_dir / "pose" / "poses.txt", dtype=np.float64)
    return flat.reshape(-1, 4, 4)


def _load_frames_meta(scene_dir: Path) -> list[dict]:
    return json.loads((scene_dir / "frames.json").read_text())


# ---------- dataset ---------------------------------------------------------

class RGBDSequence:
    """Lazy-loading sequence of `Frame`s from a single scene directory.

    Frames are NOT loaded until indexed. Iteration is cheap; random access
    is O(1) plus the cost of decoding one RGB + one EXR.
    """

    def __init__(self, scene_dir: Path | str) -> None:
        self.scene_dir = Path(scene_dir).resolve()
        if not self.scene_dir.is_dir():
            raise FileNotFoundError(self.scene_dir)
        self.intrinsics = _load_intrinsics(self.scene_dir)
        self._poses = _load_poses(self.scene_dir)
        self._meta = _load_frames_meta(self.scene_dir)
        if len(self._meta) != self._poses.shape[0]:
            raise ValueError(
                f"frames.json has {len(self._meta)} entries but poses.txt has "
                f"{self._poses.shape[0]} rows — data corruption?"
            )
        self._n = self._poses.shape[0]

    def __len__(self) -> int:
        return self._n

    def __iter__(self) -> Iterator[Frame]:
        for i in range(self._n):
            yield self[i]

    def __getitem__(self, idx: int) -> Frame:
        if idx < 0:
            idx += self._n
        if not 0 <= idx < self._n:
            raise IndexError(idx)
        rgb = _read_rgb(self.scene_dir / "rgb" / f"{idx:06d}.png")
        depth = _read_depth_exr(self.scene_dir / "depth_exr" / f"{idx:06d}.exr")
        if rgb.shape[:2] != depth.shape:
            raise ValueError(
                f"frame {idx}: rgb {rgb.shape[:2]} != depth {depth.shape}"
            )
        return Frame(
            index=idx,
            timestamp_s=float(self._meta[idx]["timestamp_s"]),
            rgb=rgb,
            depth_m=depth,
            pose=Pose(self._poses[idx]),
            intrinsics=self.intrinsics,
        )

    @property
    def name(self) -> str:
        return self.scene_dir.name

    @property
    def pointcloud_path(self) -> Path:
        return self.scene_dir / "pointcloud" / "scene.ply"

    @property
    def ifc_obj_path(self) -> Path:
        return self.scene_dir / "_ifcgeom_scene.obj"

    @property
    def ifc_labels_path(self) -> Path:
        return self.scene_dir / "_ifcgeom_scene.labels.json"

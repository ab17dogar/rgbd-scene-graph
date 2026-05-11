"""Verify the camera/pose/depth conventions against the shipped point cloud.

The challenge data was rendered in Blender, so two ambiguities matter:

  1. Camera-frame axes: OpenCV (+X right, +Y down, +Z forward) vs.
     OpenGL/Blender (+X right, +Y up, -Z forward).
  2. Depth meaning: planar Z-distance (the camera-frame Z component of the
     world point) vs. Euclidean ray length.

The pose matrix in `pose/poses.txt` is `T_wc` (camera-to-world). To pick the
correct convention, we back-project a frame's depth under each candidate, push
to world coords, and measure how close the resulting cloud sits to the shipped
`pointcloud/scene.ply`. The right convention should give millimeter-to-
centimeter agreement with the architectural mesh that was rendered.

Run:
    python scripts/verify_pose.py --scene data/BasicHouse_with_pc \
        --frames 0 80 159

Outputs (per frame):
    media/pose_verify/<scene>/frame<idx>_<convention>.ply
    media/pose_verify/<scene>/_score.json
"""

from __future__ import annotations

import os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import argparse
import json
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from scipy.spatial import cKDTree


# ---------- IO ---------------------------------------------------------------

def load_intrinsics(scene_dir: Path) -> tuple[np.ndarray, dict]:
    info = json.loads((scene_dir / "camera_info.json").read_text())
    K = np.array(info["intrinsics"], dtype=np.float64)
    return K, info


def load_pose(scene_dir: Path, frame_idx: int) -> np.ndarray:
    """Return the 4x4 T_wc matrix for the given frame."""
    poses = np.loadtxt(scene_dir / "pose" / "poses.txt").reshape(-1, 4, 4)
    return poses[frame_idx]


def load_depth(scene_dir: Path, frame_idx: int) -> np.ndarray:
    path = scene_dir / "depth_exr" / f"{frame_idx:06d}.exr"
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
    if img is None:
        raise RuntimeError(f"failed to read {path}")
    if img.ndim == 3:
        img = img[..., 0]
    return img.astype(np.float32)


def load_scene_points(scene_dir: Path) -> np.ndarray:
    """Read xyz from scene.ply (binary little-endian or ASCII)."""
    ply_path = scene_dir / "pointcloud" / "scene.ply"
    with open(ply_path, "rb") as f:
        header = b""
        while b"end_header" not in header:
            header += f.readline()
        header_str = header.decode("ascii", errors="replace")

        n_vertex = next(int(line.split()[-1])
                        for line in header_str.splitlines()
                        if line.startswith("element vertex"))
        is_binary = "format binary_little_endian" in header_str

        props = [ln for ln in header_str.splitlines() if ln.startswith("property")]
        if is_binary:
            stride = 0
            xyz_off = {}
            for p in props:
                t = p.split()
                size = {"float": 4, "double": 8, "uchar": 1, "uint8": 1,
                        "ushort": 2, "uint16": 2, "int": 4}[t[1]]
                if t[2] in ("x", "y", "z"):
                    xyz_off[t[2]] = (stride, size)
                stride += size
            buf = np.frombuffer(f.read(stride * n_vertex), dtype=np.uint8) \
                    .reshape(n_vertex, stride)
            xyz = np.empty((n_vertex, 3), dtype=np.float32)
            for i, axis in enumerate(("x", "y", "z")):
                off, _ = xyz_off[axis]
                xyz[:, i] = np.frombuffer(buf[:, off:off + 4].tobytes(), dtype="<f4")
        else:
            xyz = np.loadtxt(f, max_rows=n_vertex, usecols=(0, 1, 2),
                             dtype=np.float32)
    return xyz


# ---------- back-projection conventions -------------------------------------
#
# All four candidates take depth+intrinsics and produce camera-frame XYZ. The
# "world" step (T_wc @ [X, Y, Z, 1]) is shared so we factor it out.
#
# Notation:
#   d = depth value at pixel (u, v)
#   x = (u - cx) / fx,  y = (v - cy) / fy   (normalized image coords)
#   r = sqrt(x² + y² + 1)                    (ray length per unit Z)


def backproject_cv_z(uv: np.ndarray, d: np.ndarray, K: np.ndarray) -> np.ndarray:
    """OpenCV convention (+Y down, +Z forward), depth = planar Z distance."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (uv[:, 0] - cx) / fx
    y = (uv[:, 1] - cy) / fy
    return np.stack([x * d, y * d, d], axis=1)


def backproject_cv_ray(uv: np.ndarray, d: np.ndarray, K: np.ndarray) -> np.ndarray:
    """OpenCV convention, depth = Euclidean ray length."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (uv[:, 0] - cx) / fx
    y = (uv[:, 1] - cy) / fy
    r = np.sqrt(x * x + y * y + 1.0)
    z = d / r
    return np.stack([x * z, y * z, z], axis=1)


def backproject_gl_z(uv: np.ndarray, d: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Blender/OpenGL convention (+Y up, -Z forward), depth = planar Z."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (uv[:, 0] - cx) / fx
    y = (uv[:, 1] - cy) / fy
    # Camera looks down -Z; world point is in front of camera, so Z_cam = -d.
    # Y axis is also flipped relative to image v.
    return np.stack([x * d, -y * d, -d], axis=1)


def backproject_gl_ray(uv: np.ndarray, d: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Blender/OpenGL convention, depth = Euclidean ray length."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (uv[:, 0] - cx) / fx
    y = (uv[:, 1] - cy) / fy
    r = np.sqrt(x * x + y * y + 1.0)
    z = d / r
    return np.stack([x * z, -y * z, -z], axis=1)


CONVENTIONS: dict[str, Callable] = {
    "cv_z": backproject_cv_z,
    "cv_ray": backproject_cv_ray,
    "gl_z": backproject_gl_z,
    "gl_ray": backproject_gl_ray,
}


# ---------- driver ----------------------------------------------------------

def transform_to_world(P_cam: np.ndarray, T_wc: np.ndarray) -> np.ndarray:
    """Apply 4x4 T_wc to Nx3 camera-frame points; return Nx3 world points."""
    P_h = np.concatenate([P_cam, np.ones((P_cam.shape[0], 1))], axis=1)  # Nx4
    return (T_wc @ P_h.T).T[:, :3]


def write_ply(path: Path, points: np.ndarray, color: tuple[int, int, int]) -> None:
    """Minimal ASCII PLY writer (xyz + uchar rgb)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = points.shape[0]
    r, g, b = color
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n"
                f"element vertex {n}\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                "end_header\n")
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]} {r} {g} {b}\n")


def write_ply_binary(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    """Binary PLY writer for combined cloud (much faster than ASCII)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = points.shape[0]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    pts = points.astype("<f4")
    cols = colors.astype(np.uint8)
    rec = np.empty(n, dtype=[("xyz", "<f4", 3), ("rgb", "u1", 3)])
    rec["xyz"] = pts
    rec["rgb"] = cols
    with open(path, "wb") as f:
        f.write(header)
        f.write(rec.tobytes())


def verify_one_frame(
    scene_dir: Path,
    frame_idx: int,
    K: np.ndarray,
    info: dict,
    scene_points: np.ndarray,
    scene_tree: cKDTree,
    out_dir: Path,
    stride: int = 4,
) -> dict:
    """Score all 4 conventions on one frame; write per-convention PLYs."""
    depth = load_depth(scene_dir, frame_idx)
    T_wc = load_pose(scene_dir, frame_idx)

    H, W = depth.shape
    far_m = float(info["far_m"])

    # Subsample pixels.
    vs, us = np.mgrid[0:H:stride, 0:W:stride]
    uv = np.stack([us.ravel(), vs.ravel()], axis=1).astype(np.float64)
    d = depth[vs, us].ravel().astype(np.float64)

    # Drop saturated / invalid pixels (no-hit rays).
    mask = (d > 0) & (d < 0.95 * far_m) & np.isfinite(d)
    uv = uv[mask]
    d = d[mask]

    cam_origin = T_wc[:3, 3]

    results = {}
    for name, fn in CONVENTIONS.items():
        P_cam = fn(uv, d, K)
        P_world = transform_to_world(P_cam, T_wc)

        # Score: median nearest-neighbor distance to the shipped scene cloud.
        # 1cm-scale = pose+depth+convention all aligned; >>1m = wrong.
        dists, _ = scene_tree.query(P_world, k=1)
        results[name] = {
            "n_points": int(P_world.shape[0]),
            "median_nn_dist_m": float(np.median(dists)),
            "p95_nn_dist_m": float(np.percentile(dists, 95)),
            "mean_nn_dist_m": float(np.mean(dists)),
            "frac_within_5cm": float((dists < 0.05).mean()),
            "frac_within_20cm": float((dists < 0.20).mean()),
            "world_extent": [
                P_world.min(axis=0).tolist(),
                P_world.max(axis=0).tolist(),
            ],
            "distance_camera_to_centroid_m":
                float(np.linalg.norm(P_world.mean(axis=0) - cam_origin)),
        }

        # Save back-projected cloud (red) merged with scene cloud (gray) for
        # visual inspection in MeshLab / CloudCompare / Open3D Viewer.
        merged_pts = np.concatenate([scene_points, P_world.astype(np.float32)])
        merged_cols = np.concatenate([
            np.full((scene_points.shape[0], 3), 160, dtype=np.uint8),
            np.tile(np.array([220, 30, 30], dtype=np.uint8),
                    (P_world.shape[0], 1)),
        ])
        out_path = out_dir / f"frame{frame_idx:06d}_{name}.ply"
        write_ply_binary(out_path, merged_pts, merged_cols)

    return {
        "frame_idx": frame_idx,
        "camera_origin_world": cam_origin.tolist(),
        "n_pixels_used": int(d.size),
        "by_convention": results,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scene", type=Path, required=True,
                   help="path to e.g. data/BasicHouse_with_pc")
    p.add_argument("--frames", type=int, nargs="+", default=[0],
                   help="frame indices to verify")
    p.add_argument("--stride", type=int, default=4,
                   help="pixel subsampling stride")
    p.add_argument("--out_dir", type=Path,
                   default=Path("./media/pose_verify"))
    args = p.parse_args()

    K, info = load_intrinsics(args.scene)
    print(f"loaded intrinsics: fx={K[0,0]:.2f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}")

    scene_points = load_scene_points(args.scene)
    print(f"loaded scene cloud: {scene_points.shape[0]:,} points")
    scene_tree = cKDTree(scene_points)

    out_dir = args.out_dir / args.scene.name
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for frame_idx in args.frames:
        print(f"\n--- frame {frame_idx} ---")
        r = verify_one_frame(args.scene, frame_idx, K, info,
                             scene_points, scene_tree, out_dir, args.stride)
        all_results.append(r)
        for name, stats in r["by_convention"].items():
            print(f"  {name:8s}  median={stats['median_nn_dist_m']*1000:8.1f}mm  "
                  f"p95={stats['p95_nn_dist_m']*1000:8.1f}mm  "
                  f"<5cm={stats['frac_within_5cm']*100:5.1f}%")

    # Pick the winner: smallest median distance averaged across frames.
    convention_scores = {name: 0.0 for name in CONVENTIONS}
    for r in all_results:
        for name, stats in r["by_convention"].items():
            convention_scores[name] += stats["median_nn_dist_m"]
    winner = min(convention_scores, key=convention_scores.get)

    print(f"\nWINNER (lowest mean-of-medians across frames): {winner}")
    print("convention scores (sum of median NN dist, m):")
    for name, score in sorted(convention_scores.items(), key=lambda x: x[1]):
        marker = "  <- winner" if name == winner else ""
        print(f"  {name:8s}  {score:.4f}{marker}")

    summary = {
        "scene": args.scene.name,
        "winner": winner,
        "convention_scores_summed_median_m": convention_scores,
        "frames": all_results,
    }
    (out_dir / "_score.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_dir / '_score.json'}")
    print(f"Per-frame merged PLYs in {out_dir}/")


if __name__ == "__main__":
    main()

"""Inspect a HiWi-challenge data bundle and report convention findings.

Goal: surface every silent assumption that could break the pipeline before any
code is written that depends on it. Specifically:

  * Depth encoding (EXR float meters? PNG16 mm? scaling factor?)
  * Pose convention (4x4 matrix in poses.txt vs quaternion in frames.json;
    world-from-camera vs camera-from-world; axis order)
  * Intrinsics (pinhole? distortion?)
  * IFC labels (which IFC classes are present? Is IfcSpace there?)
  * Point-cloud bounds (do they coincide with pose extent?)

Usage:
    python scripts/inspect_data.py --data_dir ./data
"""

from __future__ import annotations

# OpenCV's EXR reader is gated behind an env var on some builds; set it before
# importing cv2 so first-party code never has to think about this.
import os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


# ---------- pose helpers ----------------------------------------------------

def quat_xyzw_to_R(q: np.ndarray) -> np.ndarray:
    """Convert a unit quaternion (x, y, z, w) to a 3x3 rotation matrix.

    Hamilton convention. Matches the formula used by SciPy / Eigen / Blender.
    """
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array([
        [1 - (yy + zz),     xy - wz,         xz + wy],
        [xy + wz,           1 - (xx + zz),   yz - wx],
        [xz - wy,           yz + wx,         1 - (xx + yy)],
    ])


def load_poses_txt(path: Path) -> np.ndarray:
    """Load an Nx4x4 stack of 4x4 row-major pose matrices from a text file."""
    flat = np.loadtxt(path)
    assert flat.ndim == 2 and flat.shape[1] == 16, f"unexpected pose shape {flat.shape}"
    return flat.reshape(-1, 4, 4)


def load_frames_json(path: Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return json.load(f)


# ---------- depth helpers ---------------------------------------------------

def read_depth_exr(path: Path) -> np.ndarray:
    """Read a single-channel EXR depth file as float32. Returns NaN-free array."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
    if img is None:
        raise RuntimeError(f"cv2 failed to read EXR {path}; OpenEXR plugin missing?")
    if img.ndim == 3:
        img = img[..., 0]
    return img.astype(np.float32)


def read_depth_png16(path: Path) -> np.ndarray:
    """Read a 16-bit PNG depth file as raw uint16 values (no scaling applied)."""
    arr = np.array(Image.open(path))
    if arr.dtype != np.uint16:
        # PIL sometimes upcasts; force the underlying integer dtype back.
        arr = arr.astype(np.uint16)
    return arr


# ---------- inspection report builders --------------------------------------

def report_intrinsics(scene_dir: Path) -> dict[str, Any]:
    info = json.loads((scene_dir / "camera_info.json").read_text())
    K = np.array(info["intrinsics"], dtype=float)
    return {
        "width": info["width"],
        "height": info["height"],
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
        "skew": float(K[0, 1]),
        "distortion_max_abs": float(np.max(np.abs(info.get("distortion", [0.0])))),
        "near_m": info.get("near_m"),
        "far_m": info.get("far_m"),
        "fps": info.get("fps"),
    }


def _depth_one_frame(exr_path: Path, png_path: Path, far_m: float) -> dict[str, Any]:
    """Single-frame depth statistics.

    Saturation handling: in this dataset, "no geometry hit" (e.g. sky, far
    holes) is encoded as the high-end saturation value, NOT as zero — both
    encodings are dense. Specifically:
        EXR (fp16):   ~65504.0 (the fp16 max)
        PNG16:        65535    (uint16 max)
    We treat anything above 0.99 * far_m as saturated/invalid for stats and
    for the EXR<->PNG scale fit.
    """
    exr = read_depth_exr(exr_path)
    png = read_depth_png16(png_path)

    sat_threshold_m = 0.99 * far_m
    valid_exr = np.isfinite(exr) & (exr > 0) & (exr < sat_threshold_m)
    saturated_exr = np.isfinite(exr) & (exr >= sat_threshold_m)

    # Linear fit png = scale * depth_m on UNSATURATED pixels only.
    scale, resid = float("nan"), float("nan")
    if valid_exr.sum() > 1000:
        e = exr[valid_exr].astype(np.float64)
        p = png[valid_exr].astype(np.float64)
        scale = float((e * p).sum() / max((e * e).sum(), 1e-12))
        resid = float(np.median(np.abs(p - scale * e)))

    return {
        "exr_min_valid_m": float(exr[valid_exr].min()) if valid_exr.any() else None,
        "exr_max_valid_m": float(exr[valid_exr].max()) if valid_exr.any() else None,
        "exr_median_valid_m": float(np.median(exr[valid_exr])) if valid_exr.any() else None,
        "pct_saturated": float(100.0 * saturated_exr.mean()),
        "png_min_nz": int(png[png > 0].min()) if (png > 0).any() else None,
        "png_max": int(png.max()),
        "estimated_png_per_meter": scale,
        "png_vs_exr_residual_units": resid,
    }


def report_depth(scene_dir: Path, frame_indices: list[int]) -> dict[str, Any]:
    info = json.loads((scene_dir / "camera_info.json").read_text())
    far_m = float(info["far_m"])
    theoretical_scale = 65535.0 / far_m  # PNG_max / far_m, the natural encoding

    per_frame = {}
    for idx in frame_indices:
        exr_path = scene_dir / "depth_exr" / f"{idx:06d}.exr"
        png_path = scene_dir / "depth_png16" / f"{idx:06d}.png"
        if exr_path.exists() and png_path.exists():
            per_frame[f"frame_{idx:06d}"] = _depth_one_frame(exr_path, png_path, far_m)

    # Aggregate scale estimate across frames (median is robust to a frame
    # that's mostly sky / saturated).
    scales = [v["estimated_png_per_meter"] for v in per_frame.values()
              if v["estimated_png_per_meter"] == v["estimated_png_per_meter"]]  # !nan
    median_scale = float(np.median(scales)) if scales else float("nan")

    return {
        "far_m": far_m,
        "theoretical_png_per_meter": theoretical_scale,
        "estimated_median_png_per_meter": median_scale,
        "scale_matches_theory": (
            abs(median_scale - theoretical_scale) / theoretical_scale < 0.02
            if scales else False
        ),
        "per_frame": per_frame,
    }


def report_poses(scene_dir: Path) -> dict[str, Any]:
    poses = load_poses_txt(scene_dir / "pose" / "poses.txt")  # (N,4,4)
    frames = load_frames_json(scene_dir / "frames.json")

    # bottom-row sanity
    bottom = poses[:, 3, :]
    bottom_ok = bool(np.allclose(bottom, np.array([0, 0, 0, 1.0]), atol=1e-5))

    # rotation orthogonality
    R = poses[:, :3, :3]
    RtR = np.einsum("nij,nik->njk", R, R)
    eye_err = float(np.max(np.abs(RtR - np.eye(3))))

    # determinant -> +1 means proper rotation
    dets = np.linalg.det(R)

    # cross-check: does poses.txt translation == frames.json t_xyz_m?
    t_mat = poses[:, :3, 3]
    t_json = np.array([f["t_xyz_m"] for f in frames])
    t_match = float(np.max(np.abs(t_mat - t_json)))

    # cross-check: does poses.txt R == quat_xyzw_to_R(frames.json q_xyzw)?
    R_json = np.stack([quat_xyzw_to_R(np.array(f["q_xyzw"])) for f in frames])
    R_match = float(np.max(np.abs(R - R_json)))

    return {
        "n_poses_txt": int(poses.shape[0]),
        "n_frames_json": len(frames),
        "bottom_row_is_0001": bottom_ok,
        "rotation_orthogonality_err": eye_err,
        "rotation_det_min": float(dets.min()),
        "rotation_det_max": float(dets.max()),
        "translation_xyz_min_m": t_mat.min(axis=0).tolist(),
        "translation_xyz_max_m": t_mat.max(axis=0).tolist(),
        "translation_xyz_span_m": (t_mat.max(axis=0) - t_mat.min(axis=0)).tolist(),
        "vertical_axis_guess": ["x", "y", "z"][int(np.argmin(t_mat.max(axis=0) - t_mat.min(axis=0)))],
        "txt_vs_json_t_max_abs_diff": t_match,
        "txt_vs_json_R_max_abs_diff": R_match,
        # interpretation: the translation IS the camera origin in world coords,
        # which means poses.txt is world-from-camera (the matrix maps points
        # from the camera frame *into* the world).
        "interpretation": "world-from-camera (T_wc): camera origin = pose[:,:3,3]",
    }


def report_ifc_labels(scene_dir: Path) -> dict[str, Any]:
    labels = json.loads((scene_dir / "_ifcgeom_scene.labels.json").read_text())
    classes = Counter(v["ifc_class"] for v in labels.values())

    # explicitly check the classes the challenge cares about
    must_have = ["IfcSpace", "IfcDoor", "IfcBuildingStorey",
                 "IfcWall", "IfcWallStandardCase", "IfcSlab", "IfcWindow"]
    presence = {c: int(classes.get(c, 0)) for c in must_have}

    return {
        "n_entities": len(labels),
        "n_distinct_classes": len(classes),
        "presence_check": presence,
        "all_classes_sorted_by_count": classes.most_common(),
    }


def report_pointcloud(scene_dir: Path) -> dict[str, Any]:
    """Read scene.ply with a minimal parser to avoid Open3D's native init cost
    when we only need bounds + count."""
    ply_path = scene_dir / "pointcloud" / "scene.ply"
    with open(ply_path, "rb") as f:
        header_bytes = b""
        while b"end_header" not in header_bytes:
            line = f.readline()
            if not line:
                break
            header_bytes += line
        header = header_bytes.decode("ascii", errors="replace")

        # parse vertex count
        n_vertex = None
        for line in header.splitlines():
            if line.startswith("element vertex"):
                n_vertex = int(line.split()[-1])
                break
        if n_vertex is None:
            return {"error": "could not find vertex count in PLY header"}

        # determine if binary or ascii
        is_binary = "format binary_little_endian" in header

        # we only need x,y,z bounds — assume those are the first three float
        # properties (true for Open3D-written and Blender-written PLYs).
        if is_binary:
            # Read fixed dtype: assume float32 xyz, ignore everything else.
            # Compute stride from header: count `property` lines until 'end_header'.
            props = [ln for ln in header.splitlines() if ln.startswith("property")]
            # rough stride assuming floats and uchars only
            stride = 0
            xyz_offsets = []
            for i, p in enumerate(props):
                tokens = p.split()
                ty = tokens[1]
                size = {"float": 4, "float32": 4, "double": 8,
                        "uchar": 1, "uint8": 1,
                        "ushort": 2, "uint16": 2,
                        "int": 4, "int32": 4}.get(ty, None)
                if size is None:
                    return {"error": f"unsupported PLY property type {ty}"}
                if tokens[2] in ("x", "y", "z"):
                    xyz_offsets.append((tokens[2], stride, ty, size))
                stride += size
            buf = f.read(stride * n_vertex)
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(n_vertex, stride)
            xyz = np.empty((n_vertex, 3), dtype=np.float32)
            for axis_idx, axis_name in enumerate(("x", "y", "z")):
                off = next(off for n, off, _, _ in xyz_offsets if n == axis_name)
                xyz[:, axis_idx] = np.frombuffer(arr[:, off:off + 4].tobytes(), dtype="<f4")
        else:
            # ASCII fallback: assume 'x y z ...' as first three columns.
            xyz = np.loadtxt(f, max_rows=n_vertex, usecols=(0, 1, 2), dtype=np.float32)

    return {
        "ply_path": str(ply_path.name),
        "n_points": int(xyz.shape[0]),
        "min_xyz": xyz.min(axis=0).tolist(),
        "max_xyz": xyz.max(axis=0).tolist(),
        "size_xyz": (xyz.max(axis=0) - xyz.min(axis=0)).tolist(),
        "centroid_xyz": xyz.mean(axis=0).tolist(),
    }


def report_pose_in_pointcloud(poses_report: dict, pc_report: dict) -> dict[str, Any]:
    """Sanity check: is the camera trajectory inside the point cloud bbox?"""
    pmin = np.array(poses_report["translation_xyz_min_m"])
    pmax = np.array(poses_report["translation_xyz_max_m"])
    cmin = np.array(pc_report["min_xyz"])
    cmax = np.array(pc_report["max_xyz"])
    inside_min = (pmin >= cmin - 0.5).all()
    inside_max = (pmax <= cmax + 0.5).all()
    return {
        "trajectory_inside_pointcloud_bbox": bool(inside_min and inside_max),
        "axis_overshoot_min_m": (cmin - pmin).tolist(),  # negative => trajectory inside
        "axis_overshoot_max_m": (pmax - cmax).tolist(),
    }


# ---------- driver ----------------------------------------------------------

def inspect_scene(scene_dir: Path) -> dict[str, Any]:
    print(f"\n{'=' * 70}\nSCENE: {scene_dir.name}\n{'=' * 70}")

    intr = report_intrinsics(scene_dir)
    print("\n[intrinsics]")
    for k, v in intr.items():
        print(f"  {k}: {v}")

    n_frames = len(load_frames_json(scene_dir / "frames.json"))
    sample_indices = sorted({0, n_frames // 4, n_frames // 2, 3 * n_frames // 4, n_frames - 1})
    depth = report_depth(scene_dir, frame_indices=sample_indices)
    print(f"\n[depth — sampled frames {sample_indices}]")
    print(f"  far_m (camera_info): {depth['far_m']}")
    print(f"  theoretical png/m (= 65535/far_m): {depth['theoretical_png_per_meter']:.4f}")
    print(f"  estimated median png/m (across frames): {depth['estimated_median_png_per_meter']:.4f}")
    print(f"  scale matches theory (within 2%): {depth['scale_matches_theory']}")
    for fname, fstat in depth["per_frame"].items():
        print(f"  {fname}: pct_saturated={fstat['pct_saturated']:.1f}%, "
              f"valid_range=[{fstat['exr_min_valid_m']}, {fstat['exr_max_valid_m']}] m, "
              f"scale={fstat['estimated_png_per_meter']:.2f}")

    poses = report_poses(scene_dir)
    print("\n[poses]")
    for k, v in poses.items():
        print(f"  {k}: {v}")

    ifc = report_ifc_labels(scene_dir)
    print("\n[ifc labels]")
    print(f"  n_entities: {ifc['n_entities']}")
    print(f"  n_distinct_classes: {ifc['n_distinct_classes']}")
    print(f"  presence_check: {ifc['presence_check']}")
    if ifc["presence_check"]["IfcSpace"] == 0:
        print("  !!! IfcSpace MISSING — Task B fallback required (BEV/wall reconstruction)")

    pc = report_pointcloud(scene_dir)
    print("\n[pointcloud]")
    for k, v in pc.items():
        print(f"  {k}: {v}")

    align = report_pose_in_pointcloud(poses, pc)
    print("\n[trajectory ⊂ pointcloud bbox?]")
    for k, v in align.items():
        print(f"  {k}: {v}")

    return {
        "scene": scene_dir.name,
        "intrinsics": intr,
        "depth_frame0": depth,
        "poses": poses,
        "ifc_labels": ifc,
        "pointcloud": pc,
        "trajectory_in_pc": align,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=Path, default=Path("./data"))
    p.add_argument("--out", type=Path, default=Path("./docs/inspection_report.json"))
    args = p.parse_args()

    scene_dirs = sorted(d for d in args.data_dir.iterdir() if d.is_dir())
    if not scene_dirs:
        raise SystemExit(f"No scene subdirs under {args.data_dir}")

    report = {"scenes": [inspect_scene(d) for d in scene_dirs]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()

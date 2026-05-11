# Coordinate System and Pose Conventions

This document codifies the coordinate-system findings used throughout the
pipeline. All numerical results were produced by `scripts/verify_pose.py`.

> **TL;DR** — back-projection uses the **OpenGL/Blender camera convention**
> (`+Y` up, `−Z` forward) with **planar Z-distance** as the depth meaning.
> Pose matrices in `pose/poses.txt` are **`T_wc` (world-from-camera)** and
> require no inversion before applying to camera-frame points. The world is
> **Z-up** with the building's roof near `Z = 0` and the floor at negative `Z`.

---

## 1. The two ambiguities the data leaves open

Because the scenes were rendered in Blender 4.0.2 (per `run_blender_log.json`)
and the depth/pose files don't carry a convention tag, two questions are
unresolved by inspection alone:

1. **Camera-frame axes** — two industry standards exist:

   | Convention | +X | +Y | +Z (forward) |
   |---|---|---|---|
   | OpenCV (computer vision) | right | down | into scene |
   | OpenGL / Blender / robotics REP-103 cameras | right | up | out of scene |

   Same world point, same image, but the camera-frame coordinates that
   produce that image differ by a sign on Y and Z.

2. **Depth meaning** — Blender exposes two depth channels:

   - **Z-distance** (the default `Depth` pass in EEVEE/Cycles): the camera-
     frame Z component of the surface point. Constant for a fronto-parallel
     plane regardless of pixel location.
   - **Ray length** (Euclidean distance from camera origin to surface):
     longer for off-center pixels.

   The standard pinhole back-projection assumes Z-distance.

Naïvely pick the wrong convention and back-projected points either land on a
mirror image of the building or get pushed away from the camera by a factor
that grows toward the image corners.

## 2. How we resolved it

Brute-force, four candidates, scored against ground truth:

```
back-project depth(u, v)  →  P_cam (in one of 4 candidate frames)
                          →  P_world  via  P_world = T_wc · [P_cam; 1]
                          →  median nearest-neighbour distance to scene.ply
```

`scripts/verify_pose.py` runs all four conventions on three frames per scene
and picks the lowest median distance to the shipped point cloud (which is a
clean sample of the architectural mesh, voxelised at 3 cm).

### Result — BasicHouse (3 frames @ stride 4)

| Convention | median NN | p95 NN | within 5 cm | within 20 cm |
|---|---:|---:|---:|---:|
| **`gl_z`  (OpenGL +Y up, −Z fwd ; planar Z)** | **41 mm** | **84 mm** | **66 %** | **>99 %** |
| `gl_ray` (OpenGL ; ray length) | 125 mm | 614 mm | 7 % | — |
| `cv_ray` (OpenCV ; ray length) | 243 mm | 1796 mm | 6 % | — |
| `cv_z`  (OpenCV ; planar Z) | 292 mm | 2054 mm | 3 % | — |

Best representative numbers; full per-frame breakdown in
`docs/media/pose_verify/BasicHouse_with_pc/_score.json`.

### Result — synagoge (3 frames @ stride 4)

| Convention | median NN | p95 NN |
|---|---:|---:|
| **`gl_z`** | **83 - 90 mm** | **154 - 184 mm** |
| `cv_z` | 95 - 284 mm | 1.2 - 2.6 m |
| `gl_ray` | 158 - 237 mm | 266 - 615 mm |
| `cv_ray` | 149 - 268 mm | 1.2 - 2.3 m |

The synagoge's higher absolute medians are an artefact of point-cloud density,
not misalignment: synagoge bbox volume ≈ 84 × 60 × 16 m ≈ 80 000 m³ holds
~247 k points (≈ 3 pts/m³), vs BasicHouse's 5 600 m³ for ~240 k points
(≈ 43 pts/m³ — ~14 × denser). Even a perfect back-projection lands further from
the nearest sampled point because samples are sparser. The decisive evidence
is the *gap* between `gl_z`'s p95 (~17 cm) and the alternatives' p95 (>1 m
for the OpenCV variants), which a sparsity argument cannot explain.

## 3. The verified convention, end to end

```
                          world frame            camera frame (gl_z)
                       (Z-up, Z=0 ≈ roof)        (+Y up, −Z forward)
                              │                       │
       depth EXR ─────────────┘                       │
       (planar Z, fp16 m,                             │
        sat = 65504)                                  │
                                                     ▼
       intrinsics K  ──►  P_cam = ( (u−cx)·d/fx,  −(v−cy)·d/fy,  −d )
                                                     │
       T_wc 4×4  ───────►  P_world = T_wc · [P_cam; 1]
       (poses.txt /                                  │
        frames.json quat xyzw)                       ▼
                                                Aligns to scene.ply
                                                within 4-5 cm median.
```

In code (`src/geometry/`):

```python
def backproject(depth_m: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Pixel grid + Blender depth -> camera-frame XYZ (gl_z convention)."""
    H, W = depth_m.shape
    vs, us = np.mgrid[0:H, 0:W]
    x = (us - K[0, 2]) / K[0, 0]
    y = (vs - K[1, 2]) / K[1, 1]
    return np.stack([x * depth_m, -y * depth_m, -depth_m], axis=-1)
```

To project world → pixel, invert: `T_cw = inv(T_wc)`, then for any world
point `P_w`:

```python
P_c = T_cw @ [P_w; 1]                          # camera frame, gl convention
u = (P_c[0] / -P_c[2]) * fx + cx                # divide by -Z (camera looks -Z)
v = (-P_c[1] / -P_c[2]) * fy + cy
# valid pixel iff -P_c[2] > near_m and 0 <= u < W and 0 <= v < H
```

## 4. Rejected alternatives (with concrete reason)

| Alternative | Why I rejected it |
|---|---|
| `cv_z` (OpenCV, +Z forward, +Y down) | Median NN ≈ 0.3 m, p95 ≈ 2 m on BasicHouse — back-projected cloud is mirror-reflected through the camera frame. |
| `cv_ray` (OpenCV with ray length) | Same mirror issue plus a stretching error toward image corners; p95 ≈ 1.8 m. |
| `gl_ray` (Blender axes but Euclidean depth) | The radial inflation away from the principal point shifts off-axis pixels by ~5-15 cm too far; p95 ≈ 0.6 m. |
| Inverting the pose (using `T_cw` to map P_cam to world) | Translation column of `poses.txt` matches the in-scene trajectory bounds (camera origin in world); the matrix is already `T_wc`, no inversion. |
| Quaternion handedness flip | `quat_xyzw_to_R(frames.json) ≈ poses.txt[:3,:3]` to 4 × 10⁻⁷ — there is no handedness disagreement to fix. |

## 5. World-frame quirks worth remembering

- **Z is vertical, but its direction is data-dependent.** This dataset's
  `IfcSlab` named "Floor:Bjälklag" sits at world `z ∈ [0, 0.30]` while the
  "Basic Roof" slab sits at `z ∈ [-3.62, -2.25]` — i.e. the *floor* is at
  the *high* Z and the *roof* is at the *low* Z, with "up in real-world"
  mapping to `-Z` in this world frame. This is the inverse of the more
  common Z-up convention.

  Don't hardcode "floor = min Z" or "ceiling = max Z" — use the `IfcSlab`
  bbox values to identify each horizontal level. The BEV-rooms module
  (`src/rgbdsg/ifc/rooms_bev.py`) defines a "room" as the interval between
  any two adjacent slabs, which is orientation-agnostic.
- **Y is the long horizontal axis** in BasicHouse; X is the long axis in
  synagoge. No safe assumption here — use the actual bbox dimensions.
- **OBJ source claims `mesh_up_axis: "y"`** in `render_summary.json`. This
  refers to the OBJ file's native orientation; Blender's `wm.obj_import`
  converts it to Z-up on import and the rendered RGB-D + poses are all
  Z-up. The OBJ file itself, if loaded by anything other than Blender's
  importer, will need to be rotated 90° around X.

## 6. Camera-frame orientation as a sanity-check

For frame 0 of BasicHouse, the rotation matrix `T_wc[:3,:3]` decomposes such
that the camera's −Z (forward) axis points along `T_wc[:3, :3] @ [0, 0, −1]`,
which lands roughly horizontally inside the building — i.e. the camera looks
*forward* at human eye level, not up at the ceiling. This is the final smell
test before trusting the convention.

## 7. Files referenced from this doc

- `scripts/verify_pose.py` — the four-convention scoring driver.
- `docs/inspection_report.json` — raw output of `inspect_data.py`.
- `docs/media/pose_verify/<scene>/_score.json` — per-frame, per-convention
  numerical results.
- `docs/media/pose_verify/<scene>/frame*_<conv>.ply` — merged scene + back-
  projected cloud, openable in MeshLab, CloudCompare, or Open3D Viewer for
  visual inspection. The winning convention's PLYs show the back-projected
  red cloud sitting flush against the gray architectural mesh; the losing
  conventions show clear separation.

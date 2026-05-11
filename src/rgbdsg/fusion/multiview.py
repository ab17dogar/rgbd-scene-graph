"""Multi-view fusion: 2D masks across frames -> 3D object instances.

Pipeline flow this module sits in:

    SAM 2 video propagation
            │  per (frame, obj_id) -> 2D bool mask
            ▼
    [this module]                       per obj_id ->
            │  for each frame: lift masked depth pixels to world via
            │  the verified gl_z back-projection in `rgbdsg.geometry`,
            │  accumulate, filter outliers, compute centroid + bbox.
            ▼
    Scene graph construction in `rgbdsg.graph`

Why aggregate across many frames?
=================================
A single back-projected mask gives a 2D-thick slab of points (one ray per
pixel) — fine for a coarse centroid but noisy in shape. Pooling masks across
say 30-50 frames where the same object is visible from different angles
yields a much better estimate of:

  * the object's 3D extent (front/back, left/right become observable)
  * the centroid (pose noise averages out)
  * the bbox (less sensitive to a single bad frame's depth saturation)

This matters because the scene-graph edges (proximity, nearest-neighbour) are
defined on these centroids and bboxes; a 30 cm centroid error from
single-frame fusion can flip a "near" edge to a "far" edge.

Outlier handling
================
Every mask leaks a few pixels onto the wrong surface (mask boundary, motion
blur, sky/far-pane saturation). Without outlier rejection the bbox blows up.
We apply two filters:

  1. Per-pixel: drop pixels where the depth is invalid (NaN / non-finite /
     beyond saturation threshold). This is essentially free.
  2. Per-cluster: statistical filter — drop points whose distance to the
     cluster centroid exceeds a multiple of the std. Default 2.5σ removes
     ~1 % of points and reliably trims mask-edge leakage.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from rgbdsg.detection import Detection, FrameSegmentation
from rgbdsg.geometry import backproject_to_world
from rgbdsg.io import RGBDSequence


# ---------- output type -----------------------------------------------------

@dataclass
class ObjectInstance:
    """One 3D object aggregated across a sequence — a node of the scene graph.

    Coordinate frame is world (Z-up).
    """
    obj_id: int
    label: str                                    # majority-vote across frames
    label_distribution: dict[str, int] = field(repr=False)  # diagnostic
    points: np.ndarray = field(repr=False)        # (N, 3) world points
    centroid: np.ndarray = field(repr=False)      # (3,)
    bbox_min: np.ndarray = field(repr=False)      # (3,)
    bbox_max: np.ndarray = field(repr=False)      # (3,)
    n_observations: int                           # number of frames it was seen in
    avg_score: float                              # mean GroundingDINO confidence
    n_points: int

    @property
    def bbox_size(self) -> np.ndarray:
        return self.bbox_max - self.bbox_min

    @property
    def floor_z(self) -> float:
        return float(self.bbox_min[2])

    @property
    def height(self) -> float:
        return float(self.bbox_max[2] - self.bbox_min[2])


# ---------- main entry point ------------------------------------------------

def fuse_object_masks(
    seq: RGBDSequence,
    segmentations: list[FrameSegmentation],
    seed_detections: dict[int, Detection],
    frame_label_history: dict[tuple[int, int], str] | None = None,
    *,
    min_mask_pixels: int = 100,
    min_total_points: int = 200,
    outlier_std: float = 2.5,
    max_points_per_object: int = 50_000,
) -> list[ObjectInstance]:
    """Aggregate SAM 2 masks into 3D object instances.

    Args:
        seq: the source RGB-D sequence.
        segmentations: SAM 2 video output, one entry per frame, in order.
        seed_detections: GroundingDINO output that seeded each obj_id, keyed
            by obj_id. Used for the initial label and the average score.
        frame_label_history: optional override providing a label per
            (frame_idx, obj_id). If omitted, every frame inherits the seed
            detection's label. Useful when GDINO is rerun on multiple
            keyframes and an object's label drifts (e.g. "chair" in one
            frame, "armchair" in another).
        min_mask_pixels: per-frame: ignore masks smaller than this. Tiny
            masks are usually noise or partial occlusion edges.
        min_total_points: per-object: drop objects whose accumulated cloud is
            smaller than this after fusion + outlier filter.
        outlier_std: σ-threshold for the statistical outlier filter.
        max_points_per_object: cap on accumulated points per object — random
            subsample beyond this. Prevents memory blow-up on very long
            sequences with the same object always visible.

    Returns:
        List of ObjectInstance, sorted by obj_id ascending.
    """
    # Step 1: per-object accumulation. We keep a list of per-frame point
    # batches keyed by obj_id and concatenate at the end (faster than vstack
    # in a loop).
    accum_points: dict[int, list[np.ndarray]] = defaultdict(list)
    obs_count: dict[int, int] = defaultdict(int)
    label_votes: dict[int, Counter] = defaultdict(Counter)

    if len(segmentations) != len(seq):
        # SAM 2 video propagation can return fewer frames than the input
        # sequence under memory pressure (especially with the fp32 patch).
        # We carry on with the partial coverage and emit a warning rather
        # than crash; objects merely have fewer observations.
        print(f"  [warn] segmentations cover {len(segmentations)} of "
              f"{len(seq)} frames — partial coverage, continuing.")

    for fs in segmentations:
        if fs.frame_idx >= len(seq):
            continue
        if not fs.obj_ids:
            continue
        # Load the frame ONCE per iteration (avoids the per-mask cost of
        # decoding the depth EXR multiple times if many objects share a frame).
        frame = seq[fs.frame_idx]
        valid = frame.valid_depth_mask  # HxW

        for obj_id, mask in zip(fs.obj_ids, fs.masks):
            if mask.dtype != bool:
                mask = mask.astype(bool)
            mask_and_valid = mask & valid
            n_pix = int(mask_and_valid.sum())
            if n_pix < min_mask_pixels:
                continue

            P_world, _ = backproject_to_world(
                frame, valid_mask=mask_and_valid, stride=1,
            )
            accum_points[obj_id].append(P_world.astype(np.float32))
            obs_count[obj_id] += 1

            # label vote: prefer the per-frame override, else the seed label
            if frame_label_history is not None:
                lbl = frame_label_history.get((fs.frame_idx, obj_id))
                if lbl is None and obj_id in seed_detections:
                    lbl = seed_detections[obj_id].label
            elif obj_id in seed_detections:
                lbl = seed_detections[obj_id].label
            else:
                lbl = "<unknown>"
            label_votes[obj_id][lbl] += 1

    # Step 2: per-object filter + summary.
    out: list[ObjectInstance] = []
    rng = np.random.default_rng(0)

    for obj_id in sorted(accum_points):
        pts = np.concatenate(accum_points[obj_id], axis=0)
        if pts.shape[0] < min_total_points:
            continue

        # Statistical outlier filter: drop points beyond `outlier_std` σ.
        centroid = pts.mean(axis=0)
        offsets = pts - centroid
        per_pt_dist = np.linalg.norm(offsets, axis=1)
        std = per_pt_dist.std()
        keep = per_pt_dist < (per_pt_dist.mean() + outlier_std * std + 1e-6)
        pts_filtered = pts[keep]
        if pts_filtered.shape[0] < min_total_points:
            continue

        # Cap point count for memory.
        if pts_filtered.shape[0] > max_points_per_object:
            sel = rng.choice(pts_filtered.shape[0], max_points_per_object, replace=False)
            pts_filtered = pts_filtered[sel]

        centroid = pts_filtered.mean(axis=0)
        bbox_min = pts_filtered.min(axis=0)
        bbox_max = pts_filtered.max(axis=0)

        # majority label
        votes = label_votes[obj_id]
        label = votes.most_common(1)[0][0] if votes else "<unknown>"

        avg_score = float(seed_detections[obj_id].score) if obj_id in seed_detections else 0.0

        out.append(ObjectInstance(
            obj_id=int(obj_id),
            label=label,
            label_distribution=dict(votes),
            points=pts_filtered,
            centroid=centroid.astype(np.float64),
            bbox_min=bbox_min.astype(np.float64),
            bbox_max=bbox_max.astype(np.float64),
            n_observations=int(obs_count[obj_id]),
            avg_score=avg_score,
            n_points=int(pts_filtered.shape[0]),
        ))

    return out


# ---------- duplicate-object merging ---------------------------------------

def _label_compatible(a: str, b: str) -> bool:
    """Two GroundingDINO labels refer to the same physical class when their
    word-sets overlap. GDINO sometimes returns multi-phrase labels like
    'chair sofa' when a box matches two prompt phrases; treating those as
    compatible with single-word 'chair' is necessary to dedupe correctly.
    """
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    return bool(wa & wb)


def _bbox_iou_3d(min_a, max_a, min_b, max_b) -> float:
    """Axis-aligned 3D bounding-box IoU."""
    inter_min = np.maximum(min_a, min_b)
    inter_max = np.minimum(max_a, max_b)
    inter_dims = np.maximum(0.0, inter_max - inter_min)
    inter_vol = float(np.prod(inter_dims))
    if inter_vol <= 0:
        return 0.0
    vol_a = float(np.prod(np.maximum(0.0, max_a - min_a)))
    vol_b = float(np.prod(np.maximum(0.0, max_b - min_b)))
    union = vol_a + vol_b - inter_vol
    return inter_vol / union if union > 0 else 0.0


def _pointset_iou_voxel(
    points_a: np.ndarray,
    points_b: np.ndarray,
    voxel_m: float = 0.10,
) -> float:
    """Approximate 3D point-set IoU via shared voxels.

    Quantise each point cloud to a regular voxel grid (default 10 cm),
    take the cardinality of voxel intersection over voxel union. This is
    the ConceptGraphs-style geometric overlap measure: two
    `ObjectInstance`s sampled from the same physical chair occupy almost
    the same voxels even if their centroids drift by tens of cm; two
    DIFFERENT chairs in a row don't share any voxels even if their
    centroids are close.

    Cheap (microseconds per pair) and discriminative — the right
    upgrade over centroid-distance for "two chairs at a table" cases
    where the centroids are close but the bodies don't overlap.
    """
    if points_a.shape[0] == 0 or points_b.shape[0] == 0:
        return 0.0
    # Use a shared origin so voxel grids align between point sets.
    origin = np.minimum(points_a.min(axis=0), points_b.min(axis=0))
    qa = np.floor((points_a - origin) / voxel_m).astype(np.int64)
    qb = np.floor((points_b - origin) / voxel_m).astype(np.int64)
    # Pack three int coordinates into one int64 for set operations.
    pa = (qa[:, 0] << 42) | ((qa[:, 1] & 0x1FFFFF) << 21) | (qa[:, 2] & 0x1FFFFF)
    pb = (qb[:, 0] << 42) | ((qb[:, 1] & 0x1FFFFF) << 21) | (qb[:, 2] & 0x1FFFFF)
    sa = set(pa.tolist())
    sb = set(pb.tolist())
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union > 0 else 0.0


def dedup_object_instances(
    objects: list[ObjectInstance],
    *,
    centroid_dist_m: float = 0.6,
    bbox_iou_threshold: float = 0.25,
    pointset_iou_threshold: float = 0.20,
    voxel_m: float = 0.10,
    require_label_match: bool = True,
) -> list[ObjectInstance]:
    """Merge `ObjectInstance`s that almost certainly refer to the same
    physical object.

    SAM 2 video propagation tracks each `obj_id` independently, so when the
    same chair gets a fresh GroundingDINO prompt on multiple keyframes it
    becomes multiple obj_ids whose post-fusion centroids and bboxes
    overlap. Without dedup the scene graph reports the same chair multiple
    times, inflating Task A's V_obj and corrupting the spatial-edge
    statistics.

    Merge rule (greedy, deterministic):
        For each pair (a, b) where a has more observations than b:
            if label_compatible AND
               (centroid_distance < centroid_dist_m OR
                bbox_IoU > bbox_iou_threshold OR
                pointset_voxel_IoU > pointset_iou_threshold):
                fold b into a (combine points, recompute centroid + bbox,
                                accumulate label votes).

    The voxel-based point-set IoU criterion (ConceptGraphs-style) catches
    cases where two `ObjectInstance`s overlap geometrically but their
    centroids and bbox extents differ enough that the simpler distance/
    bbox tests miss them — e.g. partial visibility on different keyframes
    producing offset centroids on the same physical chair.

    The dominant object's `obj_id` survives. Weighted by `n_observations`
    so the most-tracked instance "wins".
    """
    if not objects:
        return []

    # Order by descending n_observations; the dominant obj absorbs the rest.
    order = sorted(range(len(objects)), key=lambda i: -objects[i].n_observations)
    merged_into: dict[int, int] = {}     # absorbed_index -> survivor_index
    survivors: list[int] = []

    for i in order:
        if i in merged_into:
            continue
        a = objects[i]
        survived = i
        # Try to fold any later (smaller-or-equal observation count) object
        # into this one if it's compatible.
        for j in order:
            if j == i or j in merged_into or j in survivors:
                continue
            if objects[j].n_observations > a.n_observations:
                continue
            b = objects[j]
            if require_label_match and not _label_compatible(a.label, b.label):
                continue
            d = float(np.linalg.norm(a.centroid - b.centroid))
            iou = _bbox_iou_3d(a.bbox_min, a.bbox_max, b.bbox_min, b.bbox_max)
            # Only compute the (more expensive) point-set IoU if the cheaper
            # tests didn't already merge — saves work on N² scaling.
            if d < centroid_dist_m or iou > bbox_iou_threshold:
                merged_into[j] = survived
            elif pointset_iou_threshold > 0:
                pset = _pointset_iou_voxel(a.points, b.points, voxel_m=voxel_m)
                if pset > pointset_iou_threshold:
                    merged_into[j] = survived
        survivors.append(survived)

    # Build merged ObjectInstance for each survivor.
    out: list[ObjectInstance] = []
    rng = np.random.default_rng(0)
    for s in survivors:
        members = [s] + [j for j, v in merged_into.items() if v == s]
        if len(members) == 1:
            out.append(objects[s])
            continue

        # Concatenate points (sub-sample if total grows huge), accumulate
        # label votes weighted by n_observations.
        all_pts = np.concatenate([objects[m].points for m in members], axis=0)
        if all_pts.shape[0] > 50_000:
            sel = rng.choice(all_pts.shape[0], 50_000, replace=False)
            all_pts = all_pts[sel]

        votes: Counter = Counter()
        for m in members:
            for k, v in objects[m].label_distribution.items():
                votes[k] += v

        total_obs = sum(objects[m].n_observations for m in members)
        avg_score_weighted = (
            sum(objects[m].avg_score * objects[m].n_observations for m in members)
            / max(1, total_obs)
        )

        # Use the dominant obj_id (the one we kept).
        dom = objects[s]
        out.append(ObjectInstance(
            obj_id=dom.obj_id,
            label=votes.most_common(1)[0][0] if votes else dom.label,
            label_distribution=dict(votes),
            points=all_pts,
            centroid=all_pts.mean(axis=0),
            bbox_min=all_pts.min(axis=0),
            bbox_max=all_pts.max(axis=0),
            n_observations=total_obs,
            avg_score=float(avg_score_weighted),
            n_points=int(all_pts.shape[0]),
        ))

    out.sort(key=lambda o: o.obj_id)
    return out

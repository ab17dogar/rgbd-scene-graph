"""End-to-end RGB-D scene-graph generation driver.

Usage:
    python scripts/run_pipeline.py \\
        --scene data/BasicHouse_with_pc \\
        --device mps \\
        --keyframes 0 40 80 120 \\
        --out outputs/basichouse

Pipeline stages:
    1. Load RGB-D sequence (rgbdsg.io)
    2. Run Grounding DINO on a small set of keyframes (the prompt seeds new
       object identities; we only re-seed on a few frames because SAM 2
       handles propagation between keyframes for free).
    3. Initialise SAM 2 video predictor on the full clip; add box prompts
       for each detected object on its keyframe; propagate through the clip.
    4. Multi-view fuse: aggregate masked depth pixels across frames,
       back-project under the gl_z convention, filter outliers.
    5. Load IFC fixtures (doors, windows, walls, slabs) from OBJ + labels.
    6. Build a NetworkX MultiDiGraph with object/fixture nodes and
       proximity / nearest / above-below edges.
    7. Save graph as GraphML + a JSON node-link summary; print a summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np

from rgbdsg.detection import (
    Detection,
    GroundingDINO,
    OWLv2,
    SAM2VideoSegmenter,
    merge_detections_iou,
)
from rgbdsg.fusion import dedup_object_instances, fuse_object_masks
from rgbdsg.graph import GraphBuildConfig, build_graph, graph_summary
from rgbdsg.ifc import (
    extract_door_wall_relations,
    extract_ifc_storeys,
    find_ifc_path,
    infer_storeys,
    load_ifc_entities,
    rooms_to_graph_dicts,
    synthesize_rooms,
    synthesize_rooms_from_walls,
)
from rgbdsg.io import RGBDSequence
from rgbdsg.viz import (
    bev_plot,
    render_3d_html,
    render_3d_png,
    trajectory_xy_from_sequence,
)


DEFAULT_PROMPT = (
    "chair. table. sofa. bed. lamp. tv. monitor. refrigerator. "
    "stove. oven. sink. toilet. door. window. cabinet. counter. "
    "shelf. desk. mirror. plant. picture. bookcase. nightstand. dresser."
)

# IFC classes that have *physical* geometry useful as fixtures in the graph.
# We exclude property-set / relationship classes (their "geometry" is empty).
IFC_FIXTURE_CLASSES = [
    "IfcDoor", "IfcWindow", "IfcWallStandardCase", "IfcWall",
    "IfcSlab", "IfcStair", "IfcStairFlight", "IfcColumn",
    "IfcRoof", "IfcCovering", "IfcFurnishingElement",
    "IfcFlowTerminal", "IfcBuildingElementProxy",
]


def detect_keyframes(
    seq: RGBDSequence,
    keyframes: list[int],
    prompt: str,
    device: str,
    box_threshold: float,
    text_threshold: float,
    max_per_frame: int = 12,
    use_owlv2: bool = False,
    owlv2_threshold: float = 0.10,
    ensemble_iou: float = 0.5,
):
    """Run open-vocabulary detection on a list of keyframes.

    By default uses Grounding DINO. With `use_owlv2=True` we ALSO run
    OWLv2 on each keyframe and merge the two by 2D-bbox IoU — duplicate
    boxes (same physical object detected by both) are collapsed by score,
    unique boxes from either detector are kept. The ensemble materially
    improves recall on textureless / synthetic scenes where one detector
    often misses what the other finds.
    """
    print(f"[GDINO] loading model on {device}...")
    det = GroundingDINO(device=device)
    owl = None
    if use_owlv2:
        print(f"[OWLv2] loading model on {device}...")
        owl = OWLv2(device=device)

    print(f"[GDINO] running on {len(keyframes)} keyframes"
          f"{' + OWLv2 ensemble' if use_owlv2 else ''}...")
    out: list[tuple[int, list]] = []
    for kf in keyframes:
        if not (0 <= kf < len(seq)):
            print(f"  ! keyframe {kf} out of range [0, {len(seq)}); skipping")
            continue
        frame = seq[kf]
        results = det.detect(frame.rgb, prompt,
                             box_threshold=box_threshold,
                             text_threshold=text_threshold)
        if owl is not None:
            owl_dets = owl.detect(frame.rgb, prompt, threshold=owlv2_threshold)
            n_g, n_o = len(results), len(owl_dets)
            results = merge_detections_iou(
                results, owl_dets, iou_threshold=ensemble_iou,
            )
            print(f"  frame {kf:4d}: GDINO={n_g} OWLv2={n_o} "
                  f"merged={len(results)}")
        else:
            print(f"  frame {kf:4d}: {len(results)} detections")
        results.sort(key=lambda r: -r.score)
        results = results[:max_per_frame]
        out.append((kf, results))
    return out


def dedup_prompts_3d(
    keyframe_dets: list[tuple[int, list]],
    seq: RGBDSequence,
    cluster_dist_m: float = 0.5,
) -> list[tuple[int, list]]:
    """Across-keyframe prompt dedup using 3D back-projected box centres.

    When GDINO is run on many keyframes (e.g. via --gdino_every) the same
    physical object often produces a detection on every frame it appears
    in. Feeding all of those as separate SAM 2 prompts costs memory and
    creates duplicate tracks the post-fusion dedup has to clean up.

    Instead, we back-project each box's pixel centre using the frame's
    depth + pose into a world-space 3D point, group detections by
    (compatible-label) ∧ (3D distance < cluster_dist_m), and keep the
    highest-scoring detection per cluster. The dropped detections are
    redundant SAM 2 seeds — SAM 2's video predictor will track the kept
    object through every frame it appears in regardless.

    Returns the same shape as `keyframe_dets` (list of `(frame_idx,
    detections)`), with duplicates removed.
    """
    from rgbdsg.geometry import backproject

    # Each detection becomes a record with its 3D world centre + score.
    records: list[dict] = []
    for kf, dets in keyframe_dets:
        if not dets:
            continue
        frame = seq[kf]
        T_wc = frame.pose.T_wc
        for d in dets:
            x1, y1, x2, y2 = d.bbox
            cu = int(round(0.5 * (x1 + x2)))
            cv = int(round(0.5 * (y1 + y2)))
            cu = max(0, min(seq.intrinsics.width - 1, cu))
            cv = max(0, min(seq.intrinsics.height - 1, cv))
            depth = float(frame.depth_m[cv, cu])
            if not np.isfinite(depth) or depth <= 0 or depth >= 0.95 * 100.0:
                # Saturated / no-hit: skip this detection (we have no 3D
                # anchor). Still keep the original 2D prompt — fall through
                # and let it sit in its own cluster.
                world = None
            else:
                # gl_z back-projection of a single pixel. The bulk
                # `backproject(depth_image, intrinsics)` API is overkill
                # here — for one (cu, cv) we apply the formula inline.
                intr = seq.intrinsics
                xn = (cu - intr.cx) / intr.fx
                yn = (cv - intr.cy) / intr.fy
                p_cam = np.array([xn * depth, -yn * depth, -depth, 1.0])
                world = (T_wc @ p_cam)[:3]
            records.append({
                "kf": kf, "det": d, "world": world,
                "label_words": frozenset(d.label.lower().split()),
                "score": float(d.score),
            })

    # Greedy clustering: highest-score-first. Each new record either
    # joins a compatible-label cluster within `cluster_dist_m` or starts
    # its own.
    records.sort(key=lambda r: -r["score"])
    clusters: list[list[dict]] = []
    for r in records:
        joined = False
        for c in clusters:
            head = c[0]
            if not (r["label_words"] & head["label_words"]):
                continue
            if r["world"] is None or head["world"] is None:
                continue
            if float(np.linalg.norm(r["world"] - head["world"])) <= cluster_dist_m:
                c.append(r)
                joined = True
                break
        if not joined:
            clusters.append([r])

    # Keep only the highest-scoring detection from each cluster.
    keep: dict[int, list] = {}
    for c in clusters:
        head = c[0]
        keep.setdefault(head["kf"], []).append(head["det"])
    deduped = [(kf, dets) for kf, dets in keep.items()]
    deduped.sort(key=lambda x: x[0])
    return deduped


def assign_obj_ids(
    keyframe_dets: list[tuple[int, list]],
) -> tuple[dict[int, Detection], list[tuple[int, int, Detection]]]:
    """Map each keyframe detection to a globally-unique obj_id.

    Cross-keyframe association is handled by `dedup_prompts_3d` upstream
    when called; this function just assigns sequential obj_ids to the
    surviving detections.

    Returns:
        seed_detections: {obj_id: Detection} — used by the fusion module
            for label + score lookup.
        prompts: list of (frame_idx, obj_id, Detection) — to be fed to SAM 2.
    """
    seed: dict[int, object] = {}
    prompts: list = []
    next_id = 1
    for kf, dets in keyframe_dets:
        for d in dets:
            seed[next_id] = d
            prompts.append((kf, next_id, d))
            next_id += 1
    return seed, prompts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", type=Path, required=True,
                   help="path to e.g. data/BasicHouse_with_pc")
    p.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    p.add_argument("--keyframes", type=int, nargs="+", default=[0, 40, 80, 120],
                   help="frames to run Grounding DINO on")
    p.add_argument("--auto_keyframes", type=int, default=None,
                   help="if set, override --keyframes with this many evenly-"
                        "spaced indices across the sequence (recommended for "
                        "long or visually-sparse scenes)")
    p.add_argument("--gdino_every", type=int, default=None,
                   help="run Grounding DINO on every Nth frame (alternative "
                        "to --keyframes / --auto_keyframes). Detections are "
                        "deduped in 3D before being fed to SAM 2.")
    p.add_argument("--prompt_dedup_dist_m", type=float, default=0.5,
                   help="3D distance below which two GDINO box-centre "
                        "back-projections are merged as the same object")
    p.add_argument("--use_owlv2", action="store_true",
                   help="run OWLv2 alongside Grounding DINO and merge their "
                        "boxes by IoU. Higher recall on textureless scenes "
                        "at the cost of an extra ~1-2 GB / ~30 s loading.")
    p.add_argument("--owlv2_threshold", type=float, default=0.10,
                   help="OWLv2 confidence threshold (lower than GDINO's "
                        "because OWLv2 emits more lower-score boxes)")
    p.add_argument("--prompt", default=DEFAULT_PROMPT,
                   help="period-separated open-vocabulary phrases")
    p.add_argument("--box_threshold", type=float, default=0.30)
    p.add_argument("--text_threshold", type=float, default=0.25)
    p.add_argument("--max_per_frame", type=int, default=12,
                   help="cap detections per keyframe to bound SAM 2 cost")
    p.add_argument("--knn", type=int, default=3, help="K for `nearest` edges")
    p.add_argument("--near_radius_m", type=float, default=2.0)
    p.add_argument("--dedup_centroid_m", type=float, default=0.4,
                   help="centroid distance below which two ObjectInstances "
                        "are merged as the same physical object")
    p.add_argument("--dedup_iou", type=float, default=0.35,
                   help="3D bbox IoU above which two ObjectInstances are merged")
    p.add_argument("--out", type=Path, required=True,
                   help="output directory for graph + diagnostics")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # 1) load sequence
    seq = RGBDSequence(args.scene)
    print(f"[io] loaded {seq.name}: {len(seq)} frames @ "
          f"{seq.intrinsics.width}x{seq.intrinsics.height}")

    # 1a) auto-keyframes override (for long or visually-sparse scenes).
    if args.gdino_every is not None:
        step = max(1, int(args.gdino_every))
        args.keyframes = list(range(0, len(seq), step))
        if args.keyframes[-1] != len(seq) - 1:
            args.keyframes.append(len(seq) - 1)
        print(f"[keyframes] gdino_every={step} -> {len(args.keyframes)} "
              f"keyframes (every {step}th frame)")
    elif args.auto_keyframes is not None:
        n = len(seq)
        k = max(2, int(args.auto_keyframes))
        # evenly spaced including endpoints
        args.keyframes = [int(round(i * (n - 1) / (k - 1))) for i in range(k)]
        # de-dup (rounding can produce repeats on very short sequences)
        seen: set[int] = set()
        args.keyframes = [f for f in args.keyframes
                          if f not in seen and not seen.add(f)]
        print(f"[keyframes] auto-picked {len(args.keyframes)} evenly-spaced: "
              f"{args.keyframes}")

    # 1b) per-scene prompt override: if `data/<scene>/prompt.txt` exists,
    # load it as the GDINO prompt unless --prompt was given explicitly.
    prompt_file = args.scene / "prompt.txt"
    if prompt_file.is_file() and args.prompt == DEFAULT_PROMPT:
        loaded = prompt_file.read_text().strip()
        if loaded:
            args.prompt = loaded
            print(f"[prompt] loaded scene-specific prompt from {prompt_file}")
            print(f"[prompt]   {args.prompt[:120]}{'...' if len(args.prompt) > 120 else ''}")

    # 2) detection on keyframes (optionally with OWLv2 ensemble)
    kf_dets = detect_keyframes(
        seq, args.keyframes, args.prompt, args.device,
        args.box_threshold, args.text_threshold, args.max_per_frame,
        use_owlv2=args.use_owlv2,
        owlv2_threshold=args.owlv2_threshold,
    )
    n_raw = sum(len(d) for _, d in kf_dets)

    # 2b) 3D-aware cross-keyframe prompt dedup (Tier 2.2). Only run if we
    # have many keyframes — for ≤6 keyframes redundant detections are rare
    # and dedup adds latency without gain.
    if len(kf_dets) > 6 or args.gdino_every is not None:
        kf_dets = dedup_prompts_3d(
            kf_dets, seq, cluster_dist_m=args.prompt_dedup_dist_m,
        )
        n_after = sum(len(d) for _, d in kf_dets)
        print(f"[GDINO] 3D prompt dedup: {n_raw} raw -> {n_after} unique seeds "
              f"({n_raw - n_after} duplicates collapsed by 3D position)")

    seed_detections, prompts = assign_obj_ids(kf_dets)
    print(f"[GDINO] total {len(seed_detections)} unique obj_id seeds across "
          f"{len(kf_dets)} keyframes")

    # 3) SAM 2 video propagation
    print(f"[SAM 2] loading + initing video state...")
    seg = SAM2VideoSegmenter(device=args.device)
    seg.init_video(args.scene / "rgb")
    for frame_idx, obj_id, d in prompts:
        seg.add_box_prompt(frame_idx=frame_idx, obj_id=obj_id, box=d.bbox)
    print(f"[SAM 2] propagating through {seg.n_frames} frames...")
    segmentations = list(seg.propagate())
    print(f"[SAM 2] done: {len(segmentations)} per-frame results")

    # 4) multi-view fusion + dedup
    print(f"[fuse] aggregating masks into 3D ObjectInstances...")
    objects_raw = fuse_object_masks(seq, segmentations, seed_detections)
    print(f"[fuse] -> {len(objects_raw)} pre-dedup objects")
    objects = dedup_object_instances(
        objects_raw,
        centroid_dist_m=args.dedup_centroid_m,
        bbox_iou_threshold=args.dedup_iou,
    )
    print(f"[fuse] dedup merged {len(objects_raw) - len(objects)} duplicate "
          f"identities -> {len(objects)} unique objects")
    for o in objects:
        print(f"  obj_id={o.obj_id:2d}  {o.label:14s}  "
              f"frames={o.n_observations:3d}  pts={o.n_points:5d}  "
              f"centroid=({o.centroid[0]:6.2f},{o.centroid[1]:6.2f},{o.centroid[2]:6.2f})")

    # 5) IFC fixtures + all entities (for room synthesis)
    # When the source `.ifc` file is shipped under <scene>/<file>.ifc, the
    # IfcOpenShell path is used (canonical). Otherwise, OBJ + labels.json
    # is the fallback.
    ifc_path = find_ifc_path(args.scene)
    if ifc_path is not None:
        print(f"[ifc] canonical IfcOpenShell path: {ifc_path.name}")
    else:
        print(f"[ifc] no .ifc found; using OBJ + labels.json fallback")
    print(f"[ifc] loading fixtures...")
    fixtures = load_ifc_entities(args.scene, classes_filter=IFC_FIXTURE_CLASSES)
    all_entities = load_ifc_entities(args.scene)  # includes IfcSlab for storey detection
    print(f"[ifc] -> {len(fixtures)} fixtures (filtered), {len(all_entities)} total")

    # 5a) IFC topological relations (only meaningful with the .ifc file).
    door_wall_pairs: list[tuple[str, str]] = []
    if ifc_path is not None:
        try:
            door_wall_pairs = extract_door_wall_relations(ifc_path)
            print(f"[ifc] door↔wall fills_opening_in pairs: {len(door_wall_pairs)}")
        except Exception as e:
            print(f"[ifc] door-wall extraction skipped: {e}")

    # 5b) Pointcloud — needed both as a fallback for room synthesis AND for
    # the BEV visualisation. Load once, use for whichever needs it.
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(str(seq.pointcloud_path))
    pc_xyz = np.asarray(pcd.points)

    # Prefer wall-mesh rasterisation (clean polygons, sealed doorways);
    # fall back to pointcloud-occupancy if no wall meshes are present.
    print(f"[rooms] synthesising rooms from IfcWall + IfcDoor mesh rasterisation...")
    rooms = synthesize_rooms_from_walls(
        args.scene, all_entities,
        cell_m=0.05, wall_dilate_cells=2, door_dilate_cells=6,
        min_room_area_m2=2.0,
    )
    if not rooms:
        print(f"[rooms] no wall meshes; falling back to pointcloud BEV...")
        rooms = synthesize_rooms(pc_xyz, all_entities, cell_m=0.05, min_room_area_m2=2.0)
    print(f"[rooms] -> {len(rooms)} rooms (areas: "
          f"{[round(r.area_m2, 1) for r in rooms]})")

    # 5c) Storey hierarchy. Prefer the canonical IFC path (real IfcBuilding-
    # Storey entities with their elevations) over the slab-clustering
    # heuristic when the source IFC is available — it gives us exact storey
    # names and membership instead of a Z-cluster guess.
    if ifc_path is not None:
        try:
            storeys = extract_ifc_storeys(ifc_path)
            print(f"[storeys] canonical IFC storeys: {len(storeys)}: "
                  f"{[(s['storey_id'], s['name'], round(s['z_min'],2), round(s['z_max'],2)) for s in storeys]}")
        except Exception as e:
            print(f"[storeys] IFC storey extraction failed ({e}); "
                  f"falling back to slab Z-clustering")
            storeys = infer_storeys(pc_xyz, all_entities, cluster_tol_m=0.5)
    else:
        storeys = infer_storeys(pc_xyz, all_entities, cluster_tol_m=0.5)
        print(f"[storeys] inferred {len(storeys)} storey levels (fallback): "
              f"{[(s['storey_id'], round(s['z_min'],2), round(s['z_max'],2)) for s in storeys]}")

    # 6) graph build — Armeni 4-layer hierarchy with cameras
    print(f"[graph] constructing scene graph...")
    cam_records = [{
        "frame_idx": idx,
        "position": seq[idx].pose.t.tolist(),
        "rotation": seq[idx].pose.R.tolist(),
        "fov_deg": float(2 * np.degrees(np.arctan(
            seq.intrinsics.height / (2 * seq.intrinsics.fy)))),
        "modality": "RGB-D",
        "resolution": [int(seq.intrinsics.width), int(seq.intrinsics.height)],
    } for idx in args.keyframes]
    G = build_graph(
        objects=objects,
        ifc_entities=fixtures,
        rooms=rooms_to_graph_dicts(rooms),
        storeys=storeys,
        door_wall_pairs=door_wall_pairs,
        cameras=cam_records,
        building_name=seq.name,
        config=GraphBuildConfig(knn=args.knn, near_radius_m=args.near_radius_m),
    )
    summary = graph_summary(G)
    print(f"[graph] {json.dumps(summary, indent=2)}")

    # 7) persist
    graph_path = args.out / f"{seq.name}.graphml"
    nx.write_graphml(_jsonify_graph_attrs(G), graph_path)
    print(f"[out] wrote {graph_path}")

    # also write a node-link JSON (easier to grep) and a per-object summary
    nl = nx.node_link_data(G, edges="links")
    nl_clean = _make_json_safe(nl)
    (args.out / f"{seq.name}.json").write_text(json.dumps(nl_clean, indent=2))
    print(f"[out] wrote {args.out / f'{seq.name}.json'}")

    obj_summary = [{
        "obj_id": o.obj_id, "label": o.label,
        "n_observations": o.n_observations, "n_points": o.n_points,
        "centroid": o.centroid.tolist(),
        "bbox_min": o.bbox_min.tolist(),
        "bbox_max": o.bbox_max.tolist(),
        "label_distribution": o.label_distribution,
        "avg_score": o.avg_score,
    } for o in objects]
    (args.out / f"{seq.name}_objects.json").write_text(
        json.dumps({"summary": summary, "objects": obj_summary}, indent=2)
    )
    print(f"[out] wrote {args.out / f'{seq.name}_objects.json'}")

    # 8) Semantic 3D scene-graph viz (per Armeni 2019 / SceneGraphFusion 2021).
    # Rendered FIRST because the 3D graph is the canonical output — the BEV
    # below is just a top-down (X-Y) projection of the same NetworkX `G`,
    # superimposed on the BEV-occupancy footprint we used for room synthesis.
    print(f"[viz] rendering 3D semantic scene graph...")
    png_3d = args.out / f"{seq.name}_graph_3d.png"
    render_3d_png(
        G, png_3d,
        title=f"{seq.name} — 3D semantic scene graph",
        pointcloud_xyz=pc_xyz,
    )
    print(f"[out] wrote {png_3d}")
    html_3d = args.out / f"{seq.name}_graph_3d.html"
    render_3d_html(G, html_3d, title=f"{seq.name} — 3D scene graph")
    print(f"[out] wrote {html_3d} (interactive Plotly view)")

    # 9) BEV-occupancy scene-graph viz (top-down projection of the same G,
    # over the same BEV occupancy footprint that produced the room polygons).
    print(f"[viz] rendering BEV scene graph over occupancy footprint...")
    obj_edges = [(u.split(":")[1], v.split(":")[1])
                 for u, v, d in G.edges(data=True)
                 if d.get("relation") == "near" and u.startswith("obj:") and v.startswith("obj:")]
    obj_edges = [(int(a), int(b)) for a, b in obj_edges]
    fig_path = bev_plot(
        args.out / f"{seq.name}_bev.png",
        title=f"{seq.name} — scene graph (BEV)",
        pointcloud_xyz=pc_xyz,
        fixtures=fixtures,
        rooms=rooms,
        objects=objects,
        edges=obj_edges,
        trajectory_xy=trajectory_xy_from_sequence(seq),
    )
    print(f"[out] wrote {fig_path}")


# ---------- helpers ---------------------------------------------------------

def _jsonify_graph_attrs(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """GraphML can't store list/dict attributes directly. Stringify them."""
    H = G.copy()
    for _, data in H.nodes(data=True):
        for k, v in list(data.items()):
            if isinstance(v, (list, dict)):
                data[k] = json.dumps(v)
    for _, _, data in H.edges(data=True):
        for k, v in list(data.items()):
            if isinstance(v, (list, dict)):
                data[k] = json.dumps(v)
    return H


def _make_json_safe(obj):
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj


if __name__ == "__main__":
    main()

"""NetworkX scene-graph construction.

The graph G = (V, E) has FOUR node types, mirroring the canonical
Building → Storey → Room → Object hierarchy from Armeni et al. 2019:

    storey       — a horizontal building level (Z interval), derived from
                   IfcBuildingStorey or IfcSlab clusters
    room         — a BEV-synthesized room polygon, child of one storey
    ifc_fixture  — a structural element parsed from the IFC OBJ + labels
                   (IfcDoor, IfcWindow, IfcWall*, IfcSlab, ...)
    object       — a fused 3D detection (ObjectInstance from rgbdsg.fusion)

And edges:

    nearest         — undirected, every object to its K nearest neighbours
                      by centroid distance. Captures local layout.
    near            — undirected, every object pair within `near_radius_m`.
    above / below   — directed, vertically-aligned pairs (XY close, Z apart).
                      `below` is emitted as the symmetric counterpart of
                      `above` so consumers don't need to invert.
    next_to         — undirected, both XY-close (< 1.5 m) AND Z-close
                      (< 0.5 m). Implies "side-by-side at similar height".
    aligned_with    — undirected, sharing one centroid axis to within
                      `align_tol_m` (e.g. two chairs along a wall in a row).
    contains        — directed Storey → Room, Storey → Object/Fixture
                      (when the room/centroid Z falls in the storey
                      interval), Room → Object/Fixture (when the centroid
                      falls in the room polygon and Z range).
    connects        — undirected (emitted symmetrically) Room ↔ Room via
                      a shared IfcDoor portal, tagged with the door's GUID.

We use a `MultiDiGraph` so multiple relation types can coexist on the same
node pair. Each edge carries `relation` and `weight` (= distance, in metres).
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

from rgbdsg.fusion import ObjectInstance
from rgbdsg.ifc import IFCEntity


# Node-id prefixes — keeps node ids globally unique in one graph.
# These mirror the four-layer Armeni et al. 2019 scene-graph hierarchy:
#     building > storey > room > object   (+ ifc_fixture and camera siblings)
_OBJ_PREFIX = "obj"
_FIX_PREFIX = "ifc"
_ROOM_PREFIX = "room"
_STOREY_PREFIX = "storey"
_BUILDING_PREFIX = "building"
_CAMERA_PREFIX = "cam"


@dataclass
class GraphBuildConfig:
    """Tunables for scene-graph construction. All distances in metres."""
    knn: int = 3                    # K for the `nearest` edges
    near_radius_m: float = 2.0      # `near` edge cutoff
    vertical_align_xy_m: float = 0.5  # XY distance threshold for above/below
    vertical_min_dz_m: float = 0.5    # min Z-difference to call above/below
    next_to_xy_m: float = 1.5         # max XY distance for `next_to`
    next_to_dz_m: float = 0.5         # max Z difference for `next_to`
    align_tol_m: float = 0.3          # axis tolerance for `aligned_with`


def build_graph(
    objects: list[ObjectInstance],
    ifc_entities: list[IFCEntity] | None = None,
    rooms: list[dict] | None = None,
    storeys: list[dict] | None = None,
    door_wall_pairs: list[tuple[str, str]] | None = None,
    cameras: list[dict] | None = None,
    building_name: str | None = None,
    config: GraphBuildConfig | None = None,
) -> nx.MultiDiGraph:
    """Construct the scene graph.

    Args:
        objects: fused 3D detections (Task A nodes).
        ifc_entities: optional structural fixtures from IFC OBJ + labels.
            Recommend filtering to the geometric subset (`IfcDoor`,
            `IfcWindow`, `IfcWallStandardCase`, `IfcSlab`, ...) before
            passing in.
        rooms: optional list of dicts with keys
            `{room_id, polygon_xy, z_floor, z_ceiling}` describing a
            BEV-synthesised room. If provided, hierarchical contains-edges
            are added wherever an object/fixture centroid lies inside a
            room polygon and within its z range.
        storeys: optional list of dicts with keys
            `{storey_id, z_min, z_max, name}` describing horizontal building
            levels. If provided, Storey nodes are emitted at the top of the
            hierarchy and Storey → Room / Object / Fixture `contains` edges
            are added whenever a child's Z falls in the storey interval.
        door_wall_pairs: optional list of (door_guid, wall_guid) tuples
            extracted from `IfcRelFillsElement`/`IfcRelVoidsElement`. When
            present, a `fills_opening_in` edge is emitted from each door
            fixture to its host wall fixture. This is the canonical IFC
            portal relation that is more precise than our heuristic
            `connects` portal edges between rooms.
        config: tunables. Defaults are reasonable for indoor scenes.

    Returns:
        A `nx.MultiDiGraph` with node attributes documented per node type
        and edge attributes `relation`, `weight`.
    """
    cfg = config or GraphBuildConfig()
    G = nx.MultiDiGraph()

    # ---- nodes -----------------------------------------------------------
    # ---- Building root (Armeni layer 1) ---------------------------------
    building_node = f"{_BUILDING_PREFIX}:0"
    G.add_node(
        building_node,
        node_type="building",
        name=building_name or "building",
    )

    # ---- objects --------------------------------------------------------
    obj_nodes: list[str] = []
    obj_centroids: list[np.ndarray] = []
    for o in objects:
        node = f"{_OBJ_PREFIX}:{o.obj_id}"
        # Per-paper geometric attributes: AABB volume, max-extent length,
        # std-dev of points (mirrors SceneGraphFusion segment properties).
        bbox_size = np.maximum(0.0, o.bbox_max - o.bbox_min)
        volume_m3 = float(np.prod(bbox_size))
        max_length_m = float(bbox_size.max())
        std_xyz = (
            o.points.std(axis=0).tolist() if hasattr(o, "points") and
            o.points is not None and o.points.shape[0] > 1 else [0.0, 0.0, 0.0]
        )
        G.add_node(
            node,
            node_type="object",
            label=o.label,
            obj_id=o.obj_id,
            centroid=o.centroid.tolist(),
            bbox_min=o.bbox_min.tolist(),
            bbox_max=o.bbox_max.tolist(),
            bbox_size_m=bbox_size.tolist(),
            volume_m3=volume_m3,
            max_length_m=max_length_m,
            std_xyz=std_xyz,
            n_observations=o.n_observations,
            n_points=o.n_points,
            avg_score=o.avg_score,
        )
        obj_nodes.append(node)
        obj_centroids.append(o.centroid)

    fix_nodes: list[str] = []
    fix_centroids: list[np.ndarray] = []
    if ifc_entities:
        for e in ifc_entities:
            node = f"{_FIX_PREFIX}:{e.guid}"
            bsize = np.maximum(0.0, e.bbox_max - e.bbox_min)
            G.add_node(
                node,
                node_type="ifc_fixture",
                ifc_class=e.ifc_class,
                name=e.name,
                guid=e.guid,
                centroid=e.centroid.tolist(),
                bbox_min=e.bbox_min.tolist(),
                bbox_max=e.bbox_max.tolist(),
                bbox_size_m=bsize.tolist(),
                volume_m3=float(np.prod(bsize)),
                max_length_m=float(bsize.max()),
            )
            fix_nodes.append(node)
            fix_centroids.append(e.centroid)

    if rooms:
        for r in rooms:
            node = f"{_ROOM_PREFIX}:{r['room_id']}"
            G.add_node(
                node,
                node_type="room",
                room_id=r["room_id"],
                polygon_xy=r.get("polygon_xy"),
                z_floor=r.get("z_floor"),
                z_ceiling=r.get("z_ceiling"),
                area_m2=r.get("area_m2"),
            )

    storey_nodes: list[tuple[str, float, float]] = []
    if storeys:
        for s in storeys:
            node = f"{_STOREY_PREFIX}:{s['storey_id']}"
            G.add_node(
                node,
                node_type="storey",
                storey_id=s["storey_id"],
                name=s.get("name", f"Storey {s['storey_id']}"),
                z_min=float(s["z_min"]),
                z_max=float(s["z_max"]),
            )
            storey_nodes.append((node, float(s["z_min"]), float(s["z_max"])))
            # Building → Storey containment (Armeni hierarchy layer 1↔2).
            G.add_edge(building_node, node, relation="contains", weight=0.0)

    # ---- Camera nodes (Armeni hierarchy layer 4) -----------------------
    cam_nodes: list[tuple[str, np.ndarray]] = []   # [(node_id, world_pos)]
    if cameras:
        for c in cameras:
            node = f"{_CAMERA_PREFIX}:{c['frame_idx']:06d}"
            pos = np.asarray(c["position"], dtype=np.float64)
            G.add_node(
                node,
                node_type="camera",
                frame_idx=int(c["frame_idx"]),
                position=pos.tolist(),
                # Optional 3x3 rotation as a flat list — keep it serialisable.
                rotation_3x3=(np.asarray(c.get("rotation"),
                              dtype=np.float64).flatten().tolist()
                              if c.get("rotation") is not None else None),
                # Per Armeni: cameras have FOV / modality / resolution attrs.
                fov_deg=c.get("fov_deg"),
                modality=c.get("modality", "RGB-D"),
                resolution=c.get("resolution"),
            )
            cam_nodes.append((node, pos))

    # ---- spatial edges between objects ----------------------------------
    if len(obj_centroids) >= 2:
        cents = np.asarray(obj_centroids, dtype=np.float64)
        tree = cKDTree(cents)

        # K-NN for `nearest` edges. We add as bidirectional pairs so the
        # MultiDiGraph reflects symmetric proximity.
        K = min(cfg.knn + 1, len(cents))   # +1 because point's own NN is itself
        dists, idxs = tree.query(cents, k=K)
        for i, (d_row, j_row) in enumerate(zip(dists, idxs)):
            for d, j in zip(d_row[1:], j_row[1:]):  # skip self
                G.add_edge(obj_nodes[i], obj_nodes[j],
                           relation="nearest", weight=float(d))

        # `near` edges: every pair within near_radius_m. Use ball_query.
        pairs = tree.query_pairs(r=cfg.near_radius_m, output_type="ndarray")
        for i, j in pairs:
            d = float(np.linalg.norm(cents[i] - cents[j]))
            G.add_edge(obj_nodes[i], obj_nodes[j],
                       relation="near", weight=d)

        # above/below: vertically-aligned objects (XY close, Z apart). This is
        # cheap O(N²) for typical N < 100; skip the spatial index.
        # We emit BOTH directions explicitly: i->j with relation="above" and
        # j->i with relation="below" so traversal in either direction works
        # without inverting edge attributes.
        for i in range(len(cents)):
            for j in range(len(cents)):
                if i == j:
                    continue
                dxy = float(np.linalg.norm(cents[i, :2] - cents[j, :2]))
                dz = float(cents[i, 2] - cents[j, 2])
                if dxy < cfg.vertical_align_xy_m and dz > cfg.vertical_min_dz_m:
                    # i is above j (Z-up).
                    G.add_edge(obj_nodes[i], obj_nodes[j],
                               relation="above", weight=dz)
                    G.add_edge(obj_nodes[j], obj_nodes[i],
                               relation="below", weight=dz)

        # next_to: side-by-side at similar height (XY close + Z close).
        # Pure horizontal proximity — distinct from `above`/`below`.
        for i, j in pairs:
            dxy = float(np.linalg.norm(cents[i, :2] - cents[j, :2]))
            dz = abs(float(cents[i, 2] - cents[j, 2]))
            if dxy < cfg.next_to_xy_m and dz < cfg.next_to_dz_m:
                G.add_edge(obj_nodes[i], obj_nodes[j],
                           relation="next_to", weight=dxy)
                G.add_edge(obj_nodes[j], obj_nodes[i],
                           relation="next_to", weight=dxy)

        # aligned_with: objects share one centroid axis to within tolerance.
        # E.g. two chairs at the same X form an "aligned along Y" pair —
        # picks up "row of chairs" / "objects against a wall" patterns.
        for i in range(len(cents)):
            for j in range(i + 1, len(cents)):
                dx = abs(float(cents[i, 0] - cents[j, 0]))
                dy = abs(float(cents[i, 1] - cents[j, 1]))
                d3 = float(np.linalg.norm(cents[i] - cents[j]))
                if d3 < 0.05 or d3 > cfg.near_radius_m * 2.0:
                    continue   # too close (degenerate) or too far (noise)
                aligned_axis: str | None = None
                if dx < cfg.align_tol_m and dy > cfg.align_tol_m:
                    aligned_axis = "y"   # same X, varying Y => row along Y
                elif dy < cfg.align_tol_m and dx > cfg.align_tol_m:
                    aligned_axis = "x"   # same Y, varying X => row along X
                if aligned_axis:
                    G.add_edge(obj_nodes[i], obj_nodes[j],
                               relation="aligned_with",
                               axis=aligned_axis, weight=d3)
                    G.add_edge(obj_nodes[j], obj_nodes[i],
                               relation="aligned_with",
                               axis=aligned_axis, weight=d3)

    # ---- room → object / fixture containment -----------------------------
    if rooms:
        for r in rooms:
            poly_xy = r.get("polygon_xy")
            if poly_xy is None:
                continue
            z0, z1 = r.get("z_floor", -np.inf), r.get("z_ceiling", np.inf)
            poly = np.asarray(poly_xy, dtype=np.float64)
            if poly.ndim != 2 or poly.shape[1] != 2 or poly.shape[0] < 3:
                continue

            for node, cent in [
                *zip(obj_nodes, obj_centroids),
                *zip(fix_nodes, fix_centroids),
            ]:
                if not (z0 <= cent[2] <= z1):
                    continue
                if _point_in_polygon(cent[:2], poly):
                    G.add_edge(f"{_ROOM_PREFIX}:{r['room_id']}", node,
                               relation="contains", weight=0.0)

    # ---- door-portal `connects` edges between adjacent rooms -------------
    # For each IfcDoor fixture, find the two rooms whose polygons are
    # closest to the door centroid (in XY, with the door's Z falling in
    # the room's vertical interval). Emit a directed `connects` edge in
    # both directions so room→room traversal works for path-planning use
    # cases (the canonical robotics consumer of a 3D scene graph).
    if rooms and ifc_entities:
        doors = [e for e in ifc_entities if e.ifc_class == "IfcDoor"]
        if doors and len(rooms) >= 2:
            _add_door_portal_edges(G, doors, rooms)

    # ---- canonical `fills_opening_in` door↔wall edges --------------------
    # When the source IFC was available we extracted IfcRelFillsElement →
    # IfcRelVoidsElement chains, which give us an exact door-to-wall
    # mapping (the door fills an opening, the opening voids a wall). This
    # is more precise than a geometric inference: a door's host wall is a
    # *topological* IFC relation, not a guess from proximity.
    if door_wall_pairs and ifc_entities:
        # Map GUIDs to their canonical node IDs so we only add edges for
        # fixtures we actually emitted (some may have been filtered out).
        present = {e.guid for e in ifc_entities}
        for door_guid, wall_guid in door_wall_pairs:
            if door_guid in present and wall_guid in present:
                G.add_edge(f"{_FIX_PREFIX}:{door_guid}",
                           f"{_FIX_PREFIX}:{wall_guid}",
                           relation="fills_opening_in", weight=0.0)

    # ---- storey hierarchical edges --------------------------------------
    # Storey -> Room / Object / Fixture wherever the child's Z falls in the
    # storey interval. This implements the Storey layer of the canonical
    # Armeni-style scene graph.
    if storey_nodes:
        # Storey -> Room
        if rooms:
            for r in rooms:
                z_mid = 0.5 * (r.get("z_floor", 0) + r.get("z_ceiling", 0))
                s_node = _pick_storey(z_mid, storey_nodes)
                if s_node is not None:
                    G.add_edge(s_node, f"{_ROOM_PREFIX}:{r['room_id']}",
                               relation="contains", weight=0.0)
        # Storey -> Object
        for node, c in zip(obj_nodes, obj_centroids):
            s_node = _pick_storey(float(c[2]), storey_nodes)
            if s_node is not None:
                G.add_edge(s_node, node, relation="contains", weight=0.0)
        # Storey -> Fixture
        for node, c in zip(fix_nodes, fix_centroids):
            s_node = _pick_storey(float(c[2]), storey_nodes)
            if s_node is not None:
                G.add_edge(s_node, node, relation="contains", weight=0.0)
        # Storey -> Camera
        for node, pos in cam_nodes:
            s_node = _pick_storey(float(pos[2]), storey_nodes)
            if s_node is not None:
                G.add_edge(s_node, node, relation="contains", weight=0.0)
    elif cam_nodes:
        # No storeys provided — still link cameras under the building root.
        for node, _pos in cam_nodes:
            G.add_edge(building_node, node, relation="contains", weight=0.0)

    # ---- peer edges: same_storey / same_room (per Armeni 2019 §3) -------
    # Objects sharing a parent get an explicit "sibling" relation. Useful
    # for graph-level reasoning ("which other objects share this chair's
    # room?") without re-traversing the hierarchy each time.
    # NOTE: we deliberately exclude `ifc_fixture` from sibling sets to avoid
    # O(F²) edge explosion (e.g. 125 fixtures in a single storey = 7 750
    # sibling edges per storey, which dominates the graph without adding
    # actionable semantic information — the hierarchical `contains` chain
    # already establishes co-storey membership).
    if storey_nodes:
        _add_same_parent_edges(G, parent_relation="contains",
                               parent_node_type="storey",
                               child_node_type=("object", "room"),
                               edge_relation="same_storey")
    if rooms:
        _add_same_parent_edges(G, parent_relation="contains",
                               parent_node_type="room",
                               child_node_type=("object",),
                               edge_relation="same_room")

    return G


def _add_same_parent_edges(
    G: nx.MultiDiGraph,
    *,
    parent_relation: str,
    parent_node_type: str,
    child_node_type: tuple[str, ...],
    edge_relation: str,
) -> None:
    """For each parent of `parent_node_type`, link its children with `edge_relation`.

    Children are nodes of any of `child_node_type` connected to the parent
    via an outbound edge whose relation == `parent_relation`. We add a
    *single direction* sibling edge between each unordered pair to keep
    the count O(C²/2) per parent (where C = child count).
    """
    for parent, parent_data in list(G.nodes(data=True)):
        if parent_data.get("node_type") != parent_node_type:
            continue
        children = [
            v for _, v, d in G.out_edges(parent, data=True)
            if d.get("relation") == parent_relation
            and G.nodes[v].get("node_type") in child_node_type
        ]
        for i in range(len(children)):
            for j in range(i + 1, len(children)):
                G.add_edge(children[i], children[j],
                           relation=edge_relation, weight=0.0)


def _pick_storey(z: float, storey_nodes: list[tuple[str, float, float]]) -> str | None:
    """Return the storey node whose Z interval contains `z`, or None."""
    for node, zmin, zmax in storey_nodes:
        if zmin <= z < zmax:
            return node
    return None


def _add_door_portal_edges(
    G: nx.MultiDiGraph,
    doors: list[IFCEntity],
    rooms: list[dict],
    abut_max_dist_m: float = 1.5,
) -> None:
    """Connect rooms that share a door.

    For each IfcDoor, we measure the minimum distance from the door
    centroid to every room polygon (treating the polygon as a closed
    boundary). Rooms within `abut_max_dist_m` are candidates; if exactly
    two qualify, we emit `connects` edges between them with the door GUID
    as the edge attribute. If more than two qualify, we keep the two
    closest. If only one qualifies, no portal edge is added (the door
    leads outside).

    The Z gating is permissive (within 1 m of the door centroid) because
    door meshes span the full floor-to-lintel height, so the door's
    centroid Z is roughly mid-height of the storey.
    """
    # Pre-compute room polygon arrays + their z extents.
    poly_data = []
    for r in rooms:
        poly_xy = r.get("polygon_xy")
        if poly_xy is None:
            continue
        poly = np.asarray(poly_xy, dtype=np.float64)
        if poly.ndim != 2 or poly.shape[1] != 2 or poly.shape[0] < 3:
            continue
        poly_data.append({
            "node_id": f"{_ROOM_PREFIX}:{r['room_id']}",
            "poly": poly,
            "z_floor": r.get("z_floor", -np.inf),
            "z_ceiling": r.get("z_ceiling", np.inf),
            "centroid_xy": poly.mean(axis=0),
        })
    if len(poly_data) < 2:
        return

    for door in doors:
        d_xy = door.centroid[:2]
        d_z = float(door.centroid[2])

        # Score each room by min-distance from door to polygon edge,
        # filtered to rooms whose Z interval the door overlaps.
        candidates: list[tuple[float, dict]] = []
        for rd in poly_data:
            if not (rd["z_floor"] - 0.5 <= d_z <= rd["z_ceiling"] + 0.5):
                continue
            d_to_poly = _min_distance_point_to_polygon(d_xy, rd["poly"])
            if d_to_poly <= abut_max_dist_m:
                candidates.append((d_to_poly, rd))
        if len(candidates) < 2:
            continue
        candidates.sort(key=lambda x: x[0])
        room_a = candidates[0][1]
        room_b = candidates[1][1]
        # Emit symmetric `connects` edges, tagged with the door GUID.
        G.add_edge(room_a["node_id"], room_b["node_id"],
                   relation="connects",
                   via_door_guid=door.guid,
                   weight=float(candidates[0][0] + candidates[1][0]))
        G.add_edge(room_b["node_id"], room_a["node_id"],
                   relation="connects",
                   via_door_guid=door.guid,
                   weight=float(candidates[0][0] + candidates[1][0]))


def _min_distance_point_to_polygon(p: np.ndarray, poly: np.ndarray) -> float:
    """Min Euclidean distance from a 2D point to a closed polygon's edges."""
    p = np.asarray(p, dtype=np.float64)
    n = poly.shape[0]
    min_d = float("inf")
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        ab = b - a
        ab_len2 = float(ab @ ab)
        if ab_len2 < 1e-12:
            d = float(np.linalg.norm(p - a))
        else:
            t = max(0.0, min(1.0, float((p - a) @ ab) / ab_len2))
            proj = a + t * ab
            d = float(np.linalg.norm(p - proj))
        if d < min_d:
            min_d = d
    return min_d


# ---------- low-level point-in-polygon (ray casting) ------------------------

def _point_in_polygon(p: np.ndarray, poly: np.ndarray) -> bool:
    """Pure-numpy ray-casting test. `poly` is (N, 2)."""
    x, y = float(p[0]), float(p[1])
    n = poly.shape[0]
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        cond = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi
        )
        if cond:
            inside = not inside
        j = i
    return inside


# ---------- diagnostics -----------------------------------------------------

def graph_summary(G: nx.MultiDiGraph) -> dict:
    """Quick summary for logging / README."""
    nt: dict[str, int] = {}
    for _, data in G.nodes(data=True):
        nt[data.get("node_type", "?")] = nt.get(data.get("node_type", "?"), 0) + 1

    et: dict[str, int] = {}
    for _, _, data in G.edges(data=True):
        et[data.get("relation", "?")] = et.get(data.get("relation", "?"), 0) + 1

    return {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "node_types": nt,
        "edge_relations": et,
    }

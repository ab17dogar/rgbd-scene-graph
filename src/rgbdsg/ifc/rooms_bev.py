"""Synthesise rooms from BEV occupancy of the architectural point cloud."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from rgbdsg.ifc.entities import IFCEntity


@dataclass
class Room:
    """One synthesised room. World-coordinate polygon, Z-up."""
    room_id: int
    polygon_xy: np.ndarray   # (N, 2) world-coord polygon
    z_floor: float
    z_ceiling: float
    area_m2: float
    storey_index: int        # which IfcBuildingStorey this came from
    n_pixels: int            # grid pixels of free space (a quality signal)


def synthesize_rooms(
    pointcloud_xyz: np.ndarray,
    ifc_entities: list[IFCEntity],
    cell_m: float = 0.05,
    slab_buffer_m: float = 0.10,
    min_room_area_m2: float = 1.0,
    slab_intervals: list[tuple[float, float]] | None = None,
) -> list[Room]:
    """Build a list of rooms from the architectural pointcloud + IFC entities."""
    if pointcloud_xyz.shape[0] < 1000:
        return []

    intervals = (
        slab_intervals if slab_intervals is not None
        else _infer_slab_intervals(pointcloud_xyz, ifc_entities)
    )
    rooms: list[Room] = []
    next_id = 1
    for s_idx, (z_low, z_high) in enumerate(intervals):
        z_lo = z_low + slab_buffer_m
        z_hi = z_high - slab_buffer_m
        if z_hi - z_lo < 0.5:
            # Interval too thin to contain a room.
            continue
        slice_pts = pointcloud_xyz[
            (pointcloud_xyz[:, 2] >= z_lo) & (pointcloud_xyz[:, 2] < z_hi)
        ]
        if slice_pts.shape[0] < 200:
            continue
        rooms_this = _rooms_from_slice(
            slice_pts, cell_m, min_room_area_m2, z_lo, z_hi, s_idx,
            next_id_start=next_id,
        )
        rooms.extend(rooms_this)
        next_id += len(rooms_this)
    return rooms


def _infer_slab_intervals(
    pc: np.ndarray,
    ents: list[IFCEntity],
    cluster_tol_m: float = 0.5,
) -> list[tuple[float, float]]:
    """Find inter-slab (z_low, z_high) intervals that bracket room interiors."""
    slabs = [e for e in ents if e.ifc_class == "IfcSlab"]
    if slabs:
        # Collect every face-plane Z (top and bottom of every slab).
        z_values: list[float] = []
        for s in slabs:
            z_values.extend([float(s.bbox_min[2]), float(s.bbox_max[2])])
        # Cluster: sort, walk, merge anything within tolerance.
        z_values.sort()
        clusters: list[float] = [z_values[0]]
        for z in z_values[1:]:
            if abs(z - clusters[-1]) <= cluster_tol_m:
                # blend into the existing cluster (running mean)
                clusters[-1] = 0.5 * (clusters[-1] + z)
            else:
                clusters.append(z)
        return [(clusters[i], clusters[i + 1]) for i in range(len(clusters) - 1)]

    # Fallback: peak-find on the Z histogram.
    z_min, z_max = float(pc[:, 2].min()), float(pc[:, 2].max())
    nbins = max(20, int((z_max - z_min) / 0.05))
    hist, edges = np.histogram(pc[:, 2], bins=nbins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    cand_mask = hist > 0.10 * hist.max()
    floors: list[float] = []
    in_run = False
    for i, m in enumerate(cand_mask):
        if m and not in_run:
            run_start, in_run = i, True
        elif not m and in_run:
            sub = hist[run_start:i]
            floors.append(float(centers[run_start + int(np.argmax(sub))]))
            in_run = False
    if in_run:
        sub = hist[run_start:]
        floors.append(float(centers[run_start + int(np.argmax(sub))]))
    return [(floors[i], floors[i + 1]) for i in range(len(floors) - 1)]


def _rooms_from_slice(
    pts: np.ndarray,
    cell_m: float,
    min_area_m2: float,
    z_floor: float,
    z_ceiling: float,
    storey_index: int,
    next_id_start: int,
) -> list[Room]:
    """Convert one storey's BEV slice into a list of Room polygons."""
    # Rasterise to a binary occupancy grid.
    x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
    x_max, y_max = pts[:, 0].max(), pts[:, 1].max()
    # Pad bbox by a few cells so room exteriors include a free-space ring.
    pad = 5
    nx = int(np.ceil((x_max - x_min) / cell_m)) + 2 * pad
    ny = int(np.ceil((y_max - y_min) / cell_m)) + 2 * pad
    occ = np.zeros((ny, nx), dtype=bool)
    ix = ((pts[:, 0] - x_min) / cell_m).astype(int) + pad
    iy = ((pts[:, 1] - y_min) / cell_m).astype(int) + pad
    occ[iy, ix] = True

    # Morphology: close to seal hairline gaps, open to remove pepper.
    occ = ndi.binary_closing(occ, iterations=2)
    occ = ndi.binary_opening(occ, iterations=1)

    free = ~occ
    labels, n_components = ndi.label(free)
    if n_components == 0:
        return []

    # Drop the largest component (almost always the unbounded exterior).
    sizes = ndi.sum_labels(free, labels, index=range(1, n_components + 1))
    exterior_id = 1 + int(np.argmax(sizes))

    out: list[Room] = []
    next_id = next_id_start
    px_per_m2 = 1.0 / (cell_m ** 2)
    for cid in range(1, n_components + 1):
        if cid == exterior_id:
            continue
        npx = int(sizes[cid - 1])
        area_m2 = npx / px_per_m2
        if area_m2 < min_area_m2:
            continue
        polygon_xy = _polygon_from_label(
            labels == cid, x_min, y_min, cell_m, pad,
        )
        if polygon_xy is None or polygon_xy.shape[0] < 3:
            continue
        out.append(Room(
            room_id=next_id,
            polygon_xy=polygon_xy,
            z_floor=float(z_floor),
            z_ceiling=float(z_ceiling),
            area_m2=float(area_m2),
            storey_index=int(storey_index),
            n_pixels=npx,
        ))
        next_id += 1
    return out


def _polygon_from_label(
    mask: np.ndarray,
    x_min: float,
    y_min: float,
    cell_m: float,
    pad: int,
) -> np.ndarray | None:
    """Trace mask boundary, simplify, return Nx2 world-coord polygon."""
    try:
        from skimage import measure
        from shapely.geometry import Polygon
    except ImportError:
        return _bbox_polygon(mask, x_min, y_min, cell_m, pad)

    contours = measure.find_contours(mask.astype(float), level=0.5)
    if not contours:
        return _bbox_polygon(mask, x_min, y_min, cell_m, pad)

    # Pick the longest contour (the outer boundary).
    contour = max(contours, key=len)  # (M, 2) in (row, col) = (y_idx, x_idx)
    pts = np.column_stack([
        (contour[:, 1] - pad) * cell_m + x_min,
        (contour[:, 0] - pad) * cell_m + y_min,
    ])
    poly = Polygon(pts)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or poly.area < 1e-3:
        return None
    poly = poly.simplify(0.05, preserve_topology=True)
    if hasattr(poly, "geoms"):
        # MultiPolygon: keep the largest piece.
        poly = max(poly.geoms, key=lambda g: g.area)
    return np.asarray(poly.exterior.coords, dtype=np.float64)


def _bbox_polygon(
    mask: np.ndarray,
    x_min: float,
    y_min: float,
    cell_m: float,
    pad: int,
) -> np.ndarray:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return np.zeros((0, 2))
    a = (xs.min() - pad) * cell_m + x_min
    b = (ys.min() - pad) * cell_m + y_min
    c = (xs.max() + 1 - pad) * cell_m + x_min
    d = (ys.max() + 1 - pad) * cell_m + y_min
    return np.array([[a, b], [c, b], [c, d], [a, d], [a, b]])


def _rasterise_mesh_faces_xy(
    vertices: np.ndarray,
    faces: np.ndarray,
    grid: np.ndarray,
    x_min: float,
    y_min: float,
    cell_m: float,
    pad: int,
) -> None:
    """Rasterise mesh triangle faces onto a 2D grid (in-place)."""
    try:
        import cv2  # OpenCV's fillPoly is fast and integer-pixel safe.
    except ImportError:
        return  # graceful no-op; caller will see an empty rasterisation
    ny, nx = grid.shape
    if faces.shape[0] == 0:
        return
    tri_xy = vertices[faces.reshape(-1)][:, :2].reshape(-1, 3, 2)
    ix = ((tri_xy[:, :, 0] - x_min) / cell_m).astype(np.int32) + pad
    iy = ((tri_xy[:, :, 1] - y_min) / cell_m).astype(np.int32) + pad
    pts = np.stack([ix, iy], axis=-1)
    # Clip to grid bounds.
    pts[..., 0] = np.clip(pts[..., 0], 0, nx - 1)
    pts[..., 1] = np.clip(pts[..., 1], 0, ny - 1)
    # uint8 buffer because cv2 fillPoly wants integer pixels.
    buf = grid.view(np.uint8) if grid.dtype == np.uint8 else grid.astype(np.uint8)
    cv2.fillPoly(buf, pts, color=1)
    if buf is not grid:
        grid |= buf.astype(bool)


def synthesize_rooms_from_walls(
    scene_dir,
    ifc_entities: list[IFCEntity],
    cell_m: float = 0.05,
    wall_dilate_cells: int = 2,
    door_dilate_cells: int = 6,
    min_room_area_m2: float = 1.0,
) -> list[Room]:
    """Build rooms by rasterising IfcWall* meshes (walls) and IfcDoor (portal seal)."""
    from rgbdsg.ifc.from_obj_labels import load_entity_meshes

    wall_meshes = load_entity_meshes(
        scene_dir, classes_filter=["IfcWall", "IfcWallStandardCase"]
    )
    if not wall_meshes:
        return []
    door_meshes = load_entity_meshes(scene_dir, classes_filter=["IfcDoor"])

    # Reuse the same slab-interval logic as the pointcloud path.
    intervals = _infer_slab_intervals(np.zeros((1, 3)), ifc_entities)
    if not intervals:
        return []

    # Compute global bbox once so all storeys share the same grid extent.
    all_v = np.concatenate(
        [m["vertices"][:, :2] for m in wall_meshes.values()] +
        [m["vertices"][:, :2] for m in door_meshes.values()],
        axis=0,
    )
    x_min, y_min = all_v[:, 0].min(), all_v[:, 1].min()
    x_max, y_max = all_v[:, 0].max(), all_v[:, 1].max()
    pad = 5
    nx = int(np.ceil((x_max - x_min) / cell_m)) + 2 * pad
    ny = int(np.ceil((y_max - y_min) / cell_m)) + 2 * pad

    rooms: list[Room] = []
    next_id = 1
    for s_idx, (z_low, z_high) in enumerate(intervals):
        # Skip slab-thickness intervals (those bracket slab bodies, not rooms).
        if z_high - z_low < 0.5:
            continue

        wall_grid = np.zeros((ny, nx), dtype=np.uint8)
        any_walls = False
        for m in wall_meshes.values():
            v = m["vertices"]
            z_med = float(np.median(v[:, 2]))
            if not (z_low <= z_med < z_high):
                continue
            _rasterise_mesh_faces_xy(v, m["faces"], wall_grid,
                                     x_min, y_min, cell_m, pad)
            any_walls = True
        if not any_walls:
            continue
        wall_grid = wall_grid.astype(bool)
        if wall_dilate_cells > 0:
            wall_grid = ndi.binary_dilation(wall_grid, iterations=wall_dilate_cells)

        # Rasterise door faces with bigger dilation to seal doorway gaps.
        door_grid = np.zeros((ny, nx), dtype=np.uint8)
        for m in door_meshes.values():
            v = m["vertices"]
            z_med = float(np.median(v[:, 2]))
            if not (z_low <= z_med < z_high):
                continue
            _rasterise_mesh_faces_xy(v, m["faces"], door_grid,
                                     x_min, y_min, cell_m, pad)
        door_grid = door_grid.astype(bool)
        if door_dilate_cells > 0:
            door_grid = ndi.binary_dilation(door_grid, iterations=door_dilate_cells)

        occ = wall_grid | door_grid
        # Light closing for any remaining hairline gaps.
        occ = ndi.binary_closing(occ, iterations=1)

        # Free-space components → rooms.
        free = ~occ
        labels, n_components = ndi.label(free)
        if n_components == 0:
            continue
        sizes = ndi.sum_labels(free, labels, index=range(1, n_components + 1))
        exterior_id = 1 + int(np.argmax(sizes))

        px_per_m2 = 1.0 / (cell_m ** 2)
        for cid in range(1, n_components + 1):
            if cid == exterior_id:
                continue
            npx = int(sizes[cid - 1])
            area_m2 = npx / px_per_m2
            if area_m2 < min_room_area_m2:
                continue
            polygon_xy = _polygon_from_label(
                labels == cid, x_min, y_min, cell_m, pad,
            )
            if polygon_xy is None or polygon_xy.shape[0] < 3:
                continue
            rooms.append(Room(
                room_id=next_id,
                polygon_xy=polygon_xy,
                z_floor=float(z_low),
                z_ceiling=float(z_high),
                area_m2=float(area_m2),
                storey_index=int(s_idx),
                n_pixels=npx,
            ))
            next_id += 1
    return rooms


def _rooms_from_wall_doors(
    wall_xy: np.ndarray,
    door_xy: np.ndarray,
    *,
    cell_m: float,
    wall_dilate: int,
    door_dilate: int,
    min_room_area_m2: float,
    z_low: float,
    z_high: float,
    storey_index: int,
    next_id_start: int,
) -> list[Room]:
    """Rasterise wall + door XY into an occupancy grid and find rooms."""
    if wall_xy.shape[0] < 3:
        return []

    all_xy = np.concatenate([wall_xy, door_xy], axis=0) if door_xy.shape[0] else wall_xy
    x_min, y_min = all_xy[:, 0].min(), all_xy[:, 1].min()
    x_max, y_max = all_xy[:, 0].max(), all_xy[:, 1].max()
    pad = 5
    nx = int(np.ceil((x_max - x_min) / cell_m)) + 2 * pad
    ny = int(np.ceil((y_max - y_min) / cell_m)) + 2 * pad

    # 1) Walls.
    occ = np.zeros((ny, nx), dtype=bool)
    ix = ((wall_xy[:, 0] - x_min) / cell_m).astype(int) + pad
    iy = ((wall_xy[:, 1] - y_min) / cell_m).astype(int) + pad
    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    occ[iy[valid], ix[valid]] = True
    if wall_dilate > 0:
        occ = ndi.binary_dilation(occ, iterations=wall_dilate)

    # 2) Doors (separate map, then OR — bigger dilation seals the doorway gap).
    if door_xy.shape[0] > 0:
        door_grid = np.zeros_like(occ)
        ix = ((door_xy[:, 0] - x_min) / cell_m).astype(int) + pad
        iy = ((door_xy[:, 1] - y_min) / cell_m).astype(int) + pad
        valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        door_grid[iy[valid], ix[valid]] = True
        if door_dilate > 0:
            door_grid = ndi.binary_dilation(door_grid, iterations=door_dilate)
        occ |= door_grid

    # 3) Light closing for any remaining hairline gaps.
    occ = ndi.binary_closing(occ, iterations=1)

    # 4) Free-space components.
    free = ~occ
    labels, n_components = ndi.label(free)
    if n_components == 0:
        return []
    sizes = ndi.sum_labels(free, labels, index=range(1, n_components + 1))
    exterior_id = 1 + int(np.argmax(sizes))

    out: list[Room] = []
    next_id = next_id_start
    px_per_m2 = 1.0 / (cell_m ** 2)
    for cid in range(1, n_components + 1):
        if cid == exterior_id:
            continue
        npx = int(sizes[cid - 1])
        area_m2 = npx / px_per_m2
        if area_m2 < min_room_area_m2:
            continue
        polygon_xy = _polygon_from_label(
            labels == cid, x_min, y_min, cell_m, pad,
        )
        if polygon_xy is None or polygon_xy.shape[0] < 3:
            continue
        out.append(Room(
            room_id=next_id,
            polygon_xy=polygon_xy,
            z_floor=float(z_low),
            z_ceiling=float(z_high),
            area_m2=float(area_m2),
            storey_index=int(storey_index),
            n_pixels=npx,
        ))
        next_id += 1
    return out


def infer_storeys(
    pointcloud_xyz: np.ndarray,
    ifc_entities: list[IFCEntity],
    cluster_tol_m: float = 0.5,
) -> list[dict]:
    """Derive storey definitions (Z intervals) for the graph hierarchy."""
    intervals = _infer_slab_intervals(pointcloud_xyz, ifc_entities,
                                      cluster_tol_m=cluster_tol_m)
    storeys: list[dict] = []
    for i, (z_low, z_high) in enumerate(intervals):
        if z_high - z_low < 0.5:
            continue   # skip slab-body intervals
        storeys.append({
            "storey_id": i + 1,
            "z_min": float(z_low),
            "z_max": float(z_high),
            "name": f"Storey {i + 1}",
        })
    return storeys


def rooms_to_graph_dicts(rooms: list[Room]) -> list[dict]:
    """Convert Room dataclasses to the dict shape `graph.build_graph` expects."""
    return [{
        "room_id": r.room_id,
        "polygon_xy": r.polygon_xy.tolist(),
        "z_floor": r.z_floor,
        "z_ceiling": r.z_ceiling,
        "area_m2": r.area_m2,
        "storey_index": r.storey_index,
    } for r in rooms]

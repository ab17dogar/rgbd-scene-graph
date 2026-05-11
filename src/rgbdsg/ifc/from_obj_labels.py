"""Recover per-IFC-entity geometry from `_ifcgeom_scene.obj` + `_ifcgeom_scene.labels.json`."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rgbdsg.ifc.entities import IFCEntity


OBJ_TO_WORLD = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)


def obj_to_world(points: np.ndarray) -> np.ndarray:
    """Apply the OBJ→world 180°-X rotation to (N, 3) points. In-place safe."""
    return points * np.array([1.0, -1.0, -1.0], dtype=points.dtype)


@dataclass
class _ObjGroup:
    """One `o`-group in the OBJ — vertex indices into the global array."""
    name: str
    face_indices: np.ndarray  # (M, 3) int32, 0-based vertex indices


def _parse_obj_groups(obj_path: Path) -> tuple[np.ndarray, list[_ObjGroup]]:
    """Return (global_vertices Nx3, list of _ObjGroup)."""
    verts: list[list[float]] = []
    groups: list[_ObjGroup] = []
    current_name: str | None = None
    current_faces: list[list[int]] = []

    def _flush() -> None:
        if current_name is not None:
            groups.append(_ObjGroup(
                name=current_name,
                face_indices=np.asarray(current_faces, dtype=np.int32)
                if current_faces else np.zeros((0, 3), dtype=np.int32),
            ))

    with open(obj_path, "r") as f:
        for line in f:
            line = line.lstrip()
            if not line or line[0] == "#":
                continue
            tag = line[0]
            if tag == "v" and (len(line) > 1 and line[1] == " "):
                # `v x y z` — vertex
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif tag == "f" and (len(line) > 1 and line[1] == " "):
                # `f i j k` (or `f i/u/n j/u/n k/u/n`); we take the first slash field.
                parts = line.split()
                if len(parts) != 4:
                    raise ValueError(
                        f"Non-triangular face in {obj_path}: {line.rstrip()!r}. "
                        "We only support triangle meshes."
                    )
                tri = [int(p.split("/", 1)[0]) - 1 for p in parts[1:]]
                current_faces.append(tri)
            elif tag == "o" and (len(line) > 1 and line[1] == " "):
                _flush()
                current_name = line[2:].strip()
                current_faces = []
            # silently ignore vn, vt, mtllib, usemtl, s, g (none affect geometry)

    _flush()
    return np.asarray(verts, dtype=np.float64), groups


def load_ifc_entities(
    scene_dir: Path | str,
    classes_filter: list[str] | None = None,
) -> list[IFCEntity]:
    """Load all IFC entities (with geometry) for one scene."""
    scene_dir = Path(scene_dir)
    obj_path = scene_dir / "_ifcgeom_scene.obj"
    lbl_path = scene_dir / "_ifcgeom_scene.labels.json"

    labels = json.loads(lbl_path.read_text())
    verts_obj, groups = _parse_obj_groups(obj_path)
    verts_world = obj_to_world(verts_obj)

    out: list[IFCEntity] = []
    for grp in groups:
        meta = labels.get(grp.name)
        if meta is None:
            print(f"  [warn] OBJ group {grp.name} has no labels entry; skipping")
            continue
        ifc_class = meta["ifc_class"]
        if classes_filter is not None and ifc_class not in classes_filter:
            continue

        # vertex indices touched by this group's faces
        if grp.face_indices.size == 0:
            print(f"  [warn] OBJ group {grp.name} has no faces; skipping")
            continue
        used = np.unique(grp.face_indices)
        v = verts_world[used]
        bbox_min = v.min(axis=0)
        bbox_max = v.max(axis=0)
        centroid = v.mean(axis=0)

        out.append(IFCEntity(
            guid=grp.name,
            ifc_class=ifc_class,
            name=meta.get("name", ""),
            centroid=centroid,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            n_vertices=int(used.size),
            n_faces=int(grp.face_indices.shape[0]),
        ))

    out.sort(key=lambda e: e.guid)
    return out


def load_entity_meshes(
    scene_dir: Path | str,
    classes_filter: list[str] | None = None,
) -> dict[str, dict]:
    """Like `load_ifc_entities` but also returns mesh data per entity."""
    scene_dir = Path(scene_dir)
    labels = json.loads((scene_dir / "_ifcgeom_scene.labels.json").read_text())
    verts_obj, groups = _parse_obj_groups(scene_dir / "_ifcgeom_scene.obj")
    verts_world = obj_to_world(verts_obj)

    out: dict[str, dict] = {}
    for grp in groups:
        meta = labels.get(grp.name)
        if meta is None or grp.face_indices.size == 0:
            continue
        if classes_filter is not None and meta["ifc_class"] not in classes_filter:
            continue
        used, inverse = np.unique(grp.face_indices, return_inverse=True)
        out[grp.name] = {
            "ifc_class": meta["ifc_class"],
            "name": meta.get("name", ""),
            "vertices": verts_world[used].copy(),
            "faces": inverse.reshape(-1, 3).astype(np.int32),
        }
    return out


def class_summary(entities: list[IFCEntity]) -> dict[str, int]:
    """Quick distribution of IFC classes (for the README / diagnostics)."""
    counts: dict[str, int] = defaultdict(int)
    for e in entities:
        counts[e.ifc_class] += 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

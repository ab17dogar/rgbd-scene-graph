"""Canonical IfcOpenShell-based IFC extraction (Task B primary path)."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from rgbdsg.ifc.entities import IFCEntity


OBJ_TO_WORLD = np.array([1.0, -1.0, -1.0])


GEOMETRIC_CLASSES = (
    "IfcSpace", "IfcDoor", "IfcWindow",
    "IfcWallStandardCase",
    "IfcSlab", "IfcRoof", "IfcStair", "IfcStairFlight",
    "IfcColumn", "IfcCovering", "IfcRailing",
    "IfcFurnishingElement", "IfcFlowTerminal",
    "IfcBuildingElementProxy",
)


_SI_PREFIX_TO_METRE = {
    None: 1.0,
    "EXA": 1e18, "PETA": 1e15, "TERA": 1e12, "GIGA": 1e9, "MEGA": 1e6,
    "KILO": 1e3, "HECTO": 1e2, "DECA": 1e1,
    "DECI": 1e-1, "CENTI": 1e-2, "MILLI": 1e-3, "MICRO": 1e-6,
    "NANO": 1e-9, "PICO": 1e-12, "FEMTO": 1e-15, "ATTO": 1e-18,
}


def _length_unit_to_m(model) -> float:
    """Return how many metres are in one length unit of this IFC's project."""
    try:
        proj = model.by_type("IfcProject")[0]
    except IndexError:
        return 1.0
    units = getattr(proj.UnitsInContext, "Units", []) or []
    for u in units:
        if u.is_a("IfcSIUnit") and u.UnitType == "LENGTHUNIT":
            return _SI_PREFIX_TO_METRE.get(u.Prefix, 1.0)
        if u.is_a("IfcConversionBasedUnit") and u.UnitType == "LENGTHUNIT":
            # Imperial / inch: not expected for these scenes; degrade safely.
            return 1.0
    return 1.0


def find_ifc_path(scene_dir: str | Path) -> Path | None:
    """Return the first `.ifc` file under `scene_dir`, or None if none exist."""
    scene_dir = Path(scene_dir)
    for p in sorted(scene_dir.glob("*.ifc")):
        return p
    return None


def _open_model(ifc_path: str | Path):
    try:
        import ifcopenshell  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "ifcopenshell is required. `pip install ifcopenshell`."
        ) from e
    import ifcopenshell
    ifc_path = Path(ifc_path)
    if not ifc_path.is_file():
        raise FileNotFoundError(ifc_path)
    return ifcopenshell.open(str(ifc_path))


def extract_ifc_entities(
    ifc_path: str | Path,
    classes: tuple[str, ...] = GEOMETRIC_CLASSES,
    apply_world_transform: bool = True,
) -> list[IFCEntity]:
    """Walk an IFC file and produce IFCEntity records for the requested classes."""
    import ifcopenshell.geom

    model = _open_model(ifc_path)

    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)

    out: list[IFCEntity] = []
    for cls in classes:
        try:
            elements = model.by_type(cls)
        except RuntimeError:
            # Class not in this schema (e.g. IfcFurniture in IFC2x3). Skip.
            continue
        for elem in elements:
            try:
                shape = ifcopenshell.geom.create_shape(settings, elem)
            except Exception as exc:
                # Empty/abstract entity (e.g. IfcSpace with no boundary).
                print(f"  [warn] {cls} {elem.GlobalId}: no geometry ({exc})",
                      file=sys.stderr)
                continue
            verts = np.asarray(shape.geometry.verts, dtype=np.float64) \
                      .reshape(-1, 3)
            faces = np.asarray(shape.geometry.faces, dtype=np.int32) \
                      .reshape(-1, 3)
            if verts.size == 0:
                continue
            if apply_world_transform:
                verts = verts * OBJ_TO_WORLD
            out.append(IFCEntity(
                guid=elem.GlobalId,
                ifc_class=cls,
                name=getattr(elem, "Name", "") or "",
                centroid=verts.mean(axis=0),
                bbox_min=verts.min(axis=0),
                bbox_max=verts.max(axis=0),
                n_vertices=int(verts.shape[0]),
                n_faces=int(faces.shape[0]),
            ))
    return out


def extract_ifc_storeys(ifc_path: str | Path) -> list[dict]:
    """Extract `IfcBuildingStorey` records with their world-frame Z extents."""
    import ifcopenshell.geom

    model = _open_model(ifc_path)
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)

    storeys = model.by_type("IfcBuildingStorey")
    rels = model.by_type("IfcRelContainedInSpatialStructure")

    # Bucket elements by storey GUID
    elements_by_storey: dict[str, list] = {s.GlobalId: [] for s in storeys}
    for r in rels:
        struct = r.RelatingStructure
        if struct.is_a("IfcBuildingStorey"):
            elements_by_storey.setdefault(struct.GlobalId, []) \
                              .extend(r.RelatedElements)

    unit_m = _length_unit_to_m(model)

    out = []
    for idx, storey in enumerate(storeys):
        elem_list = elements_by_storey.get(storey.GlobalId, [])
        zs: list[float] = []
        for el in elem_list:
            try:
                shape = ifcopenshell.geom.create_shape(settings, el)
            except Exception:
                continue
            v = np.asarray(shape.geometry.verts, dtype=np.float64) \
                  .reshape(-1, 3)
            if v.size == 0:
                continue
            v = v * OBJ_TO_WORLD
            zs.extend([float(v[:, 2].min()), float(v[:, 2].max())])

        if zs:
            z_min, z_max = float(min(zs)), float(max(zs))
        else:
            elev_raw = float(getattr(storey, "Elevation", 0.0) or 0.0)
            elev_m = elev_raw * unit_m
            z_world = elev_m * OBJ_TO_WORLD[2]
            z_min, z_max = z_world - 0.5, z_world + 0.5

        out.append({
            "storey_id": idx,
            "guid": storey.GlobalId,
            "name": getattr(storey, "Name", None) or f"Storey {idx}",
            "elevation_m": float(getattr(storey, "Elevation", 0.0) or 0.0) * unit_m,
            "z_min": z_min,
            "z_max": z_max,
            "n_elements": len(elem_list),
        })
    out.sort(key=lambda s: s["elevation_m"])
    for new_idx, s in enumerate(out):
        s["storey_id"] = new_idx
    return out


def extract_door_wall_relations(ifc_path: str | Path) -> list[tuple[str, str]]:
    """Find canonical (door_guid, wall_guid) pairs via `IfcRelFillsElement`."""
    model = _open_model(ifc_path)

    # opening_guid -> [wall_guid1, ...]
    opening_to_walls: dict[str, list[str]] = {}
    for r in model.by_type("IfcRelVoidsElement"):
        opening = r.RelatedOpeningElement
        wall = r.RelatingBuildingElement
        if opening is None or wall is None:
            continue
        opening_to_walls.setdefault(opening.GlobalId, []).append(wall.GlobalId)

    pairs: list[tuple[str, str]] = []
    for r in model.by_type("IfcRelFillsElement"):
        elem = r.RelatedBuildingElement
        opening = r.RelatingOpeningElement
        if elem is None or opening is None:
            continue
        if not elem.is_a("IfcDoor"):
            continue
        for wall_guid in opening_to_walls.get(opening.GlobalId, []):
            pairs.append((elem.GlobalId, wall_guid))
    return pairs


def summary(entities: list[IFCEntity]) -> dict[str, int]:
    """Per-class entity count, sorted descending."""
    c: Counter = Counter(e.ifc_class for e in entities)
    return dict(sorted(c.items(), key=lambda kv: -kv[1]))


def cross_check_against_labels(
    ifc_entities: list[IFCEntity],
    labels_path: Path | str,
) -> dict[str, Any]:
    """Compare an IfcOpenShell extraction against `_ifcgeom_scene.labels.json`."""
    import json
    labels = json.loads(Path(labels_path).read_text())
    obj_counts = Counter(v["ifc_class"] for v in labels.values())
    ifc_counts = Counter(e.ifc_class for e in ifc_entities)

    classes = sorted(set(obj_counts) | set(ifc_counts))
    diff = []
    for c in classes:
        a, b = obj_counts.get(c, 0), ifc_counts.get(c, 0)
        if a or b:
            diff.append({"ifc_class": c, "labels_json": a, "ifcopenshell": b,
                         "match": a == b})
    return {"by_class": diff,
            "total_match": all(row["match"] for row in diff)}


def _main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ifc_path", help="path to .ifc file")
    p.add_argument("--cross_check", type=Path,
                   help="path to a `_ifcgeom_scene.labels.json` to compare against")
    p.add_argument("--storeys", action="store_true",
                   help="also extract storey hierarchy")
    p.add_argument("--door_walls", action="store_true",
                   help="also extract door↔wall fills_opening relations")
    args = p.parse_args()

    ents = extract_ifc_entities(args.ifc_path)
    print(f"\nExtracted {len(ents)} geometric entities from {args.ifc_path}:")
    for k, v in summary(ents).items():
        print(f"  {k:30s} {v:5d}")

    if args.storeys:
        ss = extract_ifc_storeys(args.ifc_path)
        print(f"\nStoreys: {len(ss)}")
        for s in ss:
            print(f"  [{s['storey_id']}] {s['name']!s:30s} "
                  f"elevation={s['elevation_m']:7.3f} m  "
                  f"z=[{s['z_min']:7.3f}, {s['z_max']:7.3f}]  "
                  f"n_elements={s['n_elements']}")

    if args.door_walls:
        pairs = extract_door_wall_relations(args.ifc_path)
        print(f"\nDoor → Wall (fills_opening_in) relations: {len(pairs)}")
        for d, w in pairs[:10]:
            print(f"  IfcDoor {d}  ->  Wall {w}")
        if len(pairs) > 10:
            print(f"  ... +{len(pairs) - 10} more")

    if args.cross_check:
        print(f"\nCross-checking against {args.cross_check}...")
        cmp = cross_check_against_labels(ents, args.cross_check)
        for row in cmp["by_class"]:
            mark = "+" if row["match"] else "x"
            print(f"  {mark} {row['ifc_class']:30s} "
                  f"labels.json={row['labels_json']:5d}   "
                  f"IfcOpenShell={row['ifcopenshell']:5d}")
        print(f"\n  total_match: {cmp['total_match']}")


if __name__ == "__main__":
    _main()

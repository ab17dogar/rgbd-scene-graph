"""IFC-loader regression tests.

These guard the OBJ→world coordinate transform: a sign flip on Y or Z would
silently put every entity in the wrong place, then no scene-graph error
message would point back here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rgbdsg.ifc import IFCEntity, class_summary, load_ifc_entities

DATA_ROOT = Path(__file__).parent.parent / "data"
SCENES = ["BasicHouse_with_pc", "synagoge_with_pc"]


@pytest.fixture(params=SCENES, scope="module")
def scene_dir(request) -> Path:
    p = DATA_ROOT / request.param
    if not p.is_dir():
        pytest.skip(f"scene not present at {p}")
    return p


def test_loader_returns_nonempty(scene_dir: Path) -> None:
    ents = load_ifc_entities(scene_dir)
    assert len(ents) > 50, f"expected many entities, got {len(ents)}"
    assert all(isinstance(e, IFCEntity) for e in ents)


def test_no_ifcspace_present(scene_dir: Path) -> None:
    """Documented in inspection_findings.md §1.2 — the data lacks IfcSpace.
    This test is a regression guard for the methodology pivot rationale.
    """
    ents = load_ifc_entities(scene_dir)
    classes = {e.ifc_class for e in ents}
    assert "IfcSpace" not in classes, \
        "IfcSpace appeared! Update the BEV-rooms rationale."


def test_door_count_matches_inspection(scene_dir: Path) -> None:
    """Inspection step counted 8 IfcDoor in BasicHouse, 5 in synagoge."""
    ents = load_ifc_entities(scene_dir, classes_filter=["IfcDoor"])
    expected = {"BasicHouse_with_pc": 8, "synagoge_with_pc": 5}[scene_dir.name]
    assert len(ents) == expected, f"door count drift: got {len(ents)}, expected {expected}"


def test_entities_lie_within_pointcloud_bbox(scene_dir: Path) -> None:
    """The OBJ→world rotation is empirical; if we ever break it, every entity
    will end up outside the architectural pointcloud bbox. This is the
    cheapest possible alarm for that mistake.
    """
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(str(scene_dir / "pointcloud" / "scene.ply"))
    pc_pts = np.asarray(pcd.points)
    pc_min, pc_max = pc_pts.min(axis=0), pc_pts.max(axis=0)
    # 0.5 m margin to absorb numerical edge effects
    margin = 0.5

    ents = load_ifc_entities(scene_dir)
    for e in ents:
        assert (e.bbox_min >= pc_min - margin).all(), \
            f"{e.guid} bbox_min {e.bbox_min} < pointcloud min {pc_min} — coordinate drift?"
        assert (e.bbox_max <= pc_max + margin).all(), \
            f"{e.guid} bbox_max {e.bbox_max} > pointcloud max {pc_max} — coordinate drift?"


def test_class_summary_smoke(scene_dir: Path) -> None:
    ents = load_ifc_entities(scene_dir)
    s = class_summary(ents)
    # class summary should be sorted descending by count
    counts = list(s.values())
    assert counts == sorted(counts, reverse=True)
    # totals should add up
    assert sum(counts) == len(ents)

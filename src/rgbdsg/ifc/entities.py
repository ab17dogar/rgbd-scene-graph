"""IFC-entity dataclass used throughout the pipeline.

An IFC entity is one architectural element (wall, door, slab, ...) with:
  - a stable identifier (the IFC `GlobalId`, present in labels.json),
  - a class (e.g. `IfcWallStandardCase`),
  - a human-readable name,
  - a 3D mesh in WORLD coordinates (Z-up, after applying the OBJ→world
    rotation).

The pipeline only needs spatial summaries (centroid, bbox), but we keep an
optional handle to the mesh for visualisation. Heavy mesh data is opt-in
because we have ~150 entities per scene and we don't want the default API to
load 100s of MB into memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class IFCEntity:
    """Spatial summary of one IFC element in WORLD coords (Z-up)."""

    guid: str
    ifc_class: str
    name: str
    centroid: np.ndarray = field(repr=False)        # (3,) float64
    bbox_min: np.ndarray = field(repr=False)        # (3,) float64
    bbox_max: np.ndarray = field(repr=False)        # (3,) float64
    n_vertices: int
    n_faces: int

    @property
    def bbox_size(self) -> np.ndarray:
        """Axis-aligned bounding box dimensions: (sx, sy, sz)."""
        return self.bbox_max - self.bbox_min

    @property
    def floor_z(self) -> float:
        """Lowest Z coordinate — useful for storey assignment."""
        return float(self.bbox_min[2])

    @property
    def height(self) -> float:
        """Vertical extent of the entity's bbox."""
        return float(self.bbox_max[2] - self.bbox_min[2])

    @property
    def footprint_xy(self) -> tuple[np.ndarray, np.ndarray]:
        """2D BEV bbox (min_xy, max_xy) for room-assignment logic."""
        return self.bbox_min[:2], self.bbox_max[:2]

    def __repr__(self) -> str:
        c = self.centroid
        return (f"IFCEntity({self.ifc_class}, {self.guid[:8]}…, "
                f"name={self.name!r}, centroid=({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}))")

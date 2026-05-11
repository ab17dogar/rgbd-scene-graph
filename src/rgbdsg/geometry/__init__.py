"""Geometry primitives: projection, back-projection, transforms.

All functions assume the gl_z convention verified in `docs/coordinates.md`.
"""
from rgbdsg.geometry.camera import (
    backproject,
    backproject_to_world,
    look_direction_world,
    project,
    transform_points,
    up_direction_world,
)

__all__ = [
    "backproject",
    "backproject_to_world",
    "look_direction_world",
    "project",
    "transform_points",
    "up_direction_world",
]

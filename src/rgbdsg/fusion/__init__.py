"""Multi-view 2D-mask fusion to 3D object instances."""
from rgbdsg.fusion.multiview import (
    ObjectInstance,
    dedup_object_instances,
    fuse_object_masks,
)

__all__ = ["ObjectInstance", "dedup_object_instances", "fuse_object_masks"]

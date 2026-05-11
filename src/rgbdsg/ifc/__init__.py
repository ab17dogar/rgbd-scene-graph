"""IFC integration: two paths, automatically routed."""
from pathlib import Path

from rgbdsg.ifc.entities import IFCEntity
from rgbdsg.ifc.from_obj_labels import (
    OBJ_TO_WORLD,
    class_summary,
    load_entity_meshes,
)
from rgbdsg.ifc.from_obj_labels import (
    load_ifc_entities as _load_from_obj_labels,
)
from rgbdsg.ifc.from_obj_labels import obj_to_world
from rgbdsg.ifc.from_ifc_file import (
    extract_door_wall_relations,
    extract_ifc_entities,
    extract_ifc_storeys,
    find_ifc_path,
)
from rgbdsg.ifc.rooms_bev import (
    Room,
    infer_storeys,
    rooms_to_graph_dicts,
    synthesize_rooms,
    synthesize_rooms_from_walls,
)


def load_ifc_entities(
    scene_dir,
    classes_filter=None,
    *,
    prefer_ifc: bool = True,
):
    """Load IFC fixtures, preferring `.ifc` over OBJ+labels when both exist."""
    scene_dir = Path(scene_dir)
    ifc_path = find_ifc_path(scene_dir) if prefer_ifc else None
    if ifc_path is not None:
        ents = extract_ifc_entities(ifc_path)
        if classes_filter is not None:
            allowed = set(classes_filter)
            ents = [e for e in ents if e.ifc_class in allowed]
        return ents
    # Fallback path
    return _load_from_obj_labels(scene_dir, classes_filter=classes_filter)


__all__ = [
    "IFCEntity",
    "OBJ_TO_WORLD",
    "Room",
    "class_summary",
    "extract_door_wall_relations",
    "extract_ifc_entities",
    "extract_ifc_storeys",
    "find_ifc_path",
    "infer_storeys",
    "load_entity_meshes",
    "load_ifc_entities",
    "obj_to_world",
    "rooms_to_graph_dicts",
    "synthesize_rooms",
    "synthesize_rooms_from_walls",
]


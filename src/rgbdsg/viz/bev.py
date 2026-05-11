"""Top-down (BEV) visualisation of the scene graph and its inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")  # headless backend; this module never opens windows

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Polygon
from matplotlib.collections import LineCollection, PatchCollection

from rgbdsg.fusion import ObjectInstance
from rgbdsg.ifc import IFCEntity, Room
from rgbdsg.io import RGBDSequence


# Class -> color, deterministic by hashing class name into a colormap
_CLASS_COLOR_CACHE: dict[str, tuple[float, float, float]] = {}


def _class_color(name: str) -> tuple[float, float, float]:
    if name not in _CLASS_COLOR_CACHE:
        h = abs(hash(name)) % 20
        _CLASS_COLOR_CACHE[name] = plt.cm.tab20(h / 20)[:3]
    return _CLASS_COLOR_CACHE[name]


def bev_plot(
    out_path: Path | str,
    *,
    title: str = "",
    pointcloud_xyz: np.ndarray | None = None,
    fixtures: list[IFCEntity] | None = None,
    rooms: list[Room] | None = None,
    objects: list[ObjectInstance] | None = None,
    edges: list[tuple[int, int]] | None = None,
    trajectory_xy: np.ndarray | None = None,
    fixture_classes_to_color: tuple[str, ...] = (
        "IfcDoor", "IfcWindow", "IfcWallStandardCase", "IfcWall",
        "IfcSlab", "IfcColumn", "IfcStair", "IfcStairFlight",
    ),
    show_pc_density: bool = True,
    figsize: tuple[float, float] = (14, 10),
    dpi: int = 140,
) -> Path:
    """Render a single top-down BEV figure with whichever layers are passed."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")

    legend_handles: list[matplotlib.patches.Patch] = []

    # 1. point cloud — drawn first so everything else sits on top
    if pointcloud_xyz is not None and pointcloud_xyz.shape[0] > 0:
        # If huge, randomly subsample to keep the PNG light.
        if pointcloud_xyz.shape[0] > 50000:
            sel = np.random.default_rng(0).choice(
                pointcloud_xyz.shape[0], 50000, replace=False
            )
            pc = pointcloud_xyz[sel]
        else:
            pc = pointcloud_xyz
        ax.scatter(pc[:, 0], pc[:, 1], s=0.3, c="lightgray",
                   alpha=0.5, rasterized=True, zorder=1)

    # 2. rooms — semi-transparent fills under everything else but pc
    if rooms:
        for r in rooms:
            poly = np.asarray(r.polygon_xy)
            if poly.shape[0] < 3:
                continue
            patch = Polygon(poly, closed=True, facecolor=(0.4, 0.7, 0.4),
                            edgecolor=(0.2, 0.5, 0.2), alpha=0.25,
                            linewidth=1.0, zorder=2)
            ax.add_patch(patch)
            cx = float(poly[:, 0].mean())
            cy = float(poly[:, 1].mean())
            ax.text(cx, cy, f"R{r.room_id}",
                    ha="center", va="center", fontsize=8,
                    color=(0.1, 0.4, 0.1), zorder=4,
                    bbox=dict(facecolor="white", alpha=0.6, pad=1, edgecolor="none"))

    # 3. fixtures — bbox rectangles colored by class
    if fixtures:
        seen_classes: set[str] = set()
        for f in fixtures:
            if f.ifc_class not in fixture_classes_to_color:
                continue
            x0, y0 = float(f.bbox_min[0]), float(f.bbox_min[1])
            w = float(f.bbox_max[0] - f.bbox_min[0])
            h = float(f.bbox_max[1] - f.bbox_min[1])
            if w < 0.02 or h < 0.02:
                # Treat as a point — small circle instead of a near-zero rect.
                ax.scatter([f.centroid[0]], [f.centroid[1]],
                           s=20, c=[_class_color(f.ifc_class)],
                           marker="x", zorder=3)
                continue
            ec = _class_color(f.ifc_class)
            patch = Rectangle((x0, y0), w, h, linewidth=1.2,
                              edgecolor=ec, facecolor="none", zorder=3)
            ax.add_patch(patch)
            if f.ifc_class not in seen_classes:
                seen_classes.add(f.ifc_class)
                legend_handles.append(
                    matplotlib.patches.Patch(facecolor="none", edgecolor=ec,
                                             label=f.ifc_class)
                )

    # 4. trajectory
    if trajectory_xy is not None and trajectory_xy.shape[0] >= 2:
        ax.plot(trajectory_xy[:, 0], trajectory_xy[:, 1],
                "-", color=(0.85, 0.4, 0.1), linewidth=1.6,
                alpha=0.8, zorder=5, label="camera trajectory")
        ax.scatter([trajectory_xy[0, 0]], [trajectory_xy[0, 1]],
                   marker="o", s=40, c="green", zorder=6, label="trajectory start")
        ax.scatter([trajectory_xy[-1, 0]], [trajectory_xy[-1, 1]],
                   marker="s", s=40, c="darkred", zorder=6, label="trajectory end")

    # 5. graph edges between objects
    if objects and edges:
        cents = {o.obj_id: o.centroid[:2] for o in objects}
        seg = []
        for a, b in edges:
            if a in cents and b in cents:
                seg.append([cents[a], cents[b]])
        if seg:
            lc = LineCollection(seg, colors=(0.3, 0.3, 0.8, 0.4),
                                linewidths=0.6, zorder=6)
            ax.add_collection(lc)

    # 6. detected objects — circles + labels
    if objects:
        for o in objects:
            x, y = float(o.centroid[0]), float(o.centroid[1])
            color = _class_color(o.label)
            ax.scatter([x], [y], s=70, c=[color], edgecolor="black",
                       linewidth=0.8, zorder=7)
            ax.text(x + 0.15, y + 0.15, f"{o.label} #{o.obj_id}",
                    fontsize=7, color="black", zorder=8,
                    bbox=dict(facecolor="white", alpha=0.7, pad=1, edgecolor="none"))

    # cosmetics
    if title:
        ax.set_title(title)
    ax.set_xlabel("world X (m)")
    ax.set_ylabel("world Y (m)")
    ax.grid(True, alpha=0.2)
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right", fontsize=8,
                  framealpha=0.9)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def trajectory_xy_from_sequence(seq: RGBDSequence) -> np.ndarray:
    """Camera origin world-XY for every frame, ready for `bev_plot`."""
    return np.array([f.pose.t[:2] for f in seq])

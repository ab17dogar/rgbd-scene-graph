"""3D scene-graph renderer (Open3D + matplotlib).

Generates two artefacts side-by-side with the existing top-down BEV PNG:

  1. `<scene>_graph_3d.png`  — a static 3D snapshot of the scene graph in
     world coordinates: objects as labelled spheres, IFC fixtures as
     transparent boxes, rooms as floor polygons extruded slightly,
     camera trajectory as a polyline, and graph edges drawn as colour-
     coded line segments between centroids.

  2. `<scene>_graph_3d.html`  — an interactive Plotly scatter3d / mesh3d
     view of the same content (rotatable, zoomable, hoverable).

The renderer mirrors the layered structure proposed in
*"3D Scene Graph: A structure for unified semantics, 3D space, and camera"*
(Armeni et al. 2019) — building / storey / room / object / camera — and
visualises edges per *SceneGraphFusion* (Wu et al. 2021): each edge gets
a fixed colour by its `relation` so the rendered image is legible.
"""

from __future__ import annotations

import html
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


# Edge colour table — keep contrast strong so the static PNG is legible.
EDGE_COLOR = {
    "nearest":          "#1f77b4",   # blue
    "near":             "#aec7e8",   # light blue
    "above":            "#ff7f0e",   # orange
    "below":            "#ffbb78",   # light orange
    "next_to":          "#2ca02c",   # green
    "aligned_with":     "#98df8a",   # light green
    "contains":         "#7f7f7f",   # grey
    "connects":         "#d62728",   # red (door portals)
    "fills_opening_in": "#ff0000",   # bright red (canonical door↔wall)
    "same_storey":      "#bcbd22",   # olive
    "same_room":        "#dbdb8d",   # light olive
}

# Node visualisation hints by node type.
NODE_STYLE = {
    "object":      {"color": "#e377c2", "size": 0.18},   # pink sphere ø36 cm
    "ifc_fixture": {"color": "#9467bd", "size": 0.05},   # purple, smaller
    "room":        {"color": "#17becf", "alpha": 0.20},  # cyan polygon
    "storey":      {"color": "#8c564b", "alpha": 0.05},  # brown thin slab
    "building":    {"color": "#000000", "size": 0.40},   # black diamond
    "camera":      {"color": "#bcbd22", "size": 0.07},   # olive small
}


def _node_centroid(G: nx.MultiDiGraph, n: str) -> np.ndarray | None:
    """Pull a 3D point for any node type, returning None if not positioned."""
    d = G.nodes[n]
    nt = d.get("node_type")
    if nt == "object" or nt == "ifc_fixture":
        c = d.get("centroid")
        return np.asarray(c, dtype=np.float64) if c else None
    if nt == "room":
        poly = d.get("polygon_xy")
        if poly is None:
            return None
        poly_arr = np.asarray(poly, dtype=np.float64)
        if poly_arr.size == 0:
            return None
        z_floor = d.get("z_floor", 0.0)
        z_ceiling = d.get("z_ceiling", z_floor + 2.0)
        return np.array([poly_arr[:, 0].mean(),
                         poly_arr[:, 1].mean(),
                         0.5 * (z_floor + z_ceiling)])
    if nt == "storey":
        # Centred at the mid-Z of the storey; X/Y inferred from any child.
        zmid = 0.5 * (d.get("z_min", 0.0) + d.get("z_max", 0.0))
        # Try to use a child's XY for a more meaningful centroid.
        for _, child, ed in G.out_edges(n, data=True):
            if ed.get("relation") == "contains":
                cc = _node_centroid(G, child)
                if cc is not None:
                    return np.array([cc[0], cc[1], zmid])
        return np.array([0.0, 0.0, zmid])
    if nt == "camera":
        p = d.get("position")
        return np.asarray(p, dtype=np.float64) if p else None
    if nt == "building":
        # Anchor at the average of the storeys.
        zs = [G.nodes[s].get("z_min", 0) for s in G.successors(n)
              if G.nodes[s].get("node_type") == "storey"]
        return np.array([0.0, 0.0, float(np.mean(zs))]) if zs else None
    return None


# ---------- static Matplotlib 3-D PNG ----------------------------------------

def render_3d_png(
    G: nx.MultiDiGraph,
    out_path: Path | str,
    *,
    pointcloud_xyz: np.ndarray | None = None,
    title: str = "3D scene graph",
    show_relations: tuple[str, ...] = (
        "fills_opening_in", "connects", "contains",
        "nearest", "next_to", "above",
    ),
    figsize: tuple[float, float] = (12.0, 10.0),
    dpi: int = 140,
) -> None:
    """Render the scene graph as a 3-D matplotlib PNG.

    `show_relations` is a whitelist — only edges whose `relation` is in
    this set are drawn (otherwise the figure becomes a hairball on dense
    graphs). The `same_storey` / `near` / `aligned_with` / `same_room`
    edges are valuable in the data file but visually noisy; they're off
    by default. Pass them explicitly if you want them.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)
    ax.set_xlabel("world X (m)")
    ax.set_ylabel("world Y (m)")
    ax.set_zlabel("world Z (m)")

    # Optional pointcloud backdrop (heavily downsampled).
    if pointcloud_xyz is not None and pointcloud_xyz.shape[0] > 0:
        pc = pointcloud_xyz
        if pc.shape[0] > 20_000:
            sel = np.random.default_rng(0).choice(pc.shape[0], 20_000, replace=False)
            pc = pc[sel]
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2],
                   s=0.4, c="lightgray", alpha=0.18, depthshade=False)

    # Per-node positions.
    pos: dict[str, np.ndarray] = {}
    for n in G.nodes:
        p = _node_centroid(G, n)
        if p is not None:
            pos[n] = p

    # Layer 3: rooms as filled polygons at floor height.
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    room_polys = []
    for n, d in G.nodes(data=True):
        if d.get("node_type") != "room":
            continue
        poly = d.get("polygon_xy")
        if poly is None:
            continue
        poly_arr = np.asarray(poly, dtype=np.float64)
        if poly_arr.shape[0] < 3:
            continue
        z = float(d.get("z_floor", 0.0))
        verts3 = [(p[0], p[1], z) for p in poly_arr]
        room_polys.append(verts3)
    if room_polys:
        ax.add_collection3d(Poly3DCollection(
            room_polys,
            facecolors=NODE_STYLE["room"]["color"],
            alpha=NODE_STYLE["room"]["alpha"],
            edgecolor=NODE_STYLE["room"]["color"], linewidth=1.2,
        ))

    # Layer 4: IFC fixtures as small diamonds (cheap; full mesh would dominate).
    fix_xyz = np.asarray([pos[n] for n, d in G.nodes(data=True)
                          if d.get("node_type") == "ifc_fixture" and n in pos])
    if fix_xyz.size:
        ax.scatter(fix_xyz[:, 0], fix_xyz[:, 1], fix_xyz[:, 2],
                   s=18, c=NODE_STYLE["ifc_fixture"]["color"],
                   marker="s", alpha=0.55, depthshade=True,
                   label=f"IFC fixtures ({len(fix_xyz)})")

    # Layer 5: detected objects as labelled big spheres.
    obj_xyz = []
    obj_labels = []
    for n, d in G.nodes(data=True):
        if d.get("node_type") != "object" or n not in pos:
            continue
        obj_xyz.append(pos[n])
        obj_labels.append((pos[n], d.get("label", n)))
    if obj_xyz:
        oxyz = np.asarray(obj_xyz)
        ax.scatter(oxyz[:, 0], oxyz[:, 1], oxyz[:, 2],
                   s=160, c=NODE_STYLE["object"]["color"], alpha=0.95,
                   edgecolor="black", linewidth=0.8, depthshade=True,
                   label=f"detected objects ({len(oxyz)})")
        for p, lab in obj_labels:
            ax.text(p[0], p[1], p[2] + 0.25, lab, fontsize=8, color="black")

    # Layer 4: cameras as small olive points connected by polyline.
    cam_xyz = np.asarray([pos[n] for n, d in G.nodes(data=True)
                          if d.get("node_type") == "camera" and n in pos])
    if cam_xyz.size:
        order = sorted(
            ((G.nodes[n]["frame_idx"], pos[n])
             for n, d in G.nodes(data=True)
             if d.get("node_type") == "camera" and n in pos),
            key=lambda kv: kv[0],
        )
        traj = np.asarray([p for _, p in order])
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                "-", c=NODE_STYLE["camera"]["color"],
                linewidth=1.2, alpha=0.7,
                label=f"camera trajectory ({len(traj)} keyframes)")

    # Edges per relation, drawn as line segments between centroids.
    for rel in show_relations:
        segs = []
        for u, v, d in G.edges(data=True):
            if d.get("relation") != rel:
                continue
            if u not in pos or v not in pos:
                continue
            segs.append((pos[u], pos[v]))
        if not segs:
            continue
        col = EDGE_COLOR.get(rel, "#999999")
        # Draw all segments in a single Line3DCollection for performance.
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
        lc = Line3DCollection(
            segs, colors=col,
            linewidths=(2.0 if rel in ("fills_opening_in", "connects") else
                       0.9),
            alpha=(0.85 if rel in ("fills_opening_in", "connects") else 0.45),
            label=f"{rel} ({len(segs)})",
        )
        ax.add_collection3d(lc)

    # Legend with edge colours + node markers.
    handles, labels = ax.get_legend_handles_labels()
    # Add edge colour swatches manually.
    from matplotlib.lines import Line2D
    for rel in show_relations:
        n_edges = sum(1 for _, _, d in G.edges(data=True)
                       if d.get("relation") == rel)
        if n_edges == 0:
            continue
        handles.append(Line2D([0], [0], color=EDGE_COLOR.get(rel, "#999"),
                              linewidth=2.0))
        labels.append(f"edge: {rel} ({n_edges})")
    ax.legend(handles, labels, loc="upper left", fontsize=7,
              framealpha=0.85, ncol=1, bbox_to_anchor=(1.02, 1.0))

    # Decent default viewing angle (a bit elevated, looking from -X).
    ax.view_init(elev=22, azim=-55)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------- interactive Plotly HTML ------------------------------------------

def render_3d_html(
    G: nx.MultiDiGraph,
    out_path: Path | str,
    *,
    title: str = "3D scene graph",
    show_relations: tuple[str, ...] = (
        "fills_opening_in", "connects", "contains",
        "nearest", "next_to", "above",
    ),
) -> None:
    """Render the scene graph as a single-file interactive Plotly HTML.

    Plotly is an optional dep — if it isn't installed the file is written
    as a placeholder telling the user how to enable it. This keeps the
    pipeline runnable without forcing a heavy dependency for users who
    only want the static PNG and the GraphML data.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import plotly.graph_objects as go
    except ImportError:
        out_path.write_text(
            "<!doctype html><html><body><h2>plotly not installed</h2>"
            "<p>Run <code>uv pip install plotly</code> and re-run the "
            "pipeline to generate this interactive view. The static "
            "<code>_graph_3d.png</code> and <code>.graphml</code> files "
            "next to this one already contain the same graph data.</p>"
            "</body></html>"
        )
        return

    pos: dict[str, np.ndarray] = {}
    for n in G.nodes:
        p = _node_centroid(G, n)
        if p is not None:
            pos[n] = p

    # Bucket nodes by type.
    by_type: dict[str, list[tuple[str, np.ndarray, dict]]] = defaultdict(list)
    for n, d in G.nodes(data=True):
        if n in pos:
            by_type[d.get("node_type", "?")].append((n, pos[n], d))

    fig = go.Figure()

    # One scatter per node type with a stable marker style.
    for nt, recs in by_type.items():
        xs = [p[0] for _, p, _ in recs]
        ys = [p[1] for _, p, _ in recs]
        zs = [p[2] for _, p, _ in recs]
        labels = [_node_hover_text(n, d) for n, _, d in recs]
        style = NODE_STYLE.get(nt, {"color": "#444444"})
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="markers+text" if nt == "object" else "markers",
            marker=dict(size=10 if nt == "object" else
                        (6 if nt == "ifc_fixture" else 8),
                        color=style["color"]),
            text=[d.get("label", "") for _, _, d in recs] if nt == "object"
                 else None,
            textposition="top center",
            hovertext=labels,
            hoverinfo="text",
            name=f"{nt} ({len(recs)})",
        ))

    # Edges per relation.
    for rel in show_relations:
        xs, ys, zs = [], [], []
        for u, v, d in G.edges(data=True):
            if d.get("relation") != rel:
                continue
            if u not in pos or v not in pos:
                continue
            xs.extend([pos[u][0], pos[v][0], None])
            ys.extend([pos[u][1], pos[v][1], None])
            zs.extend([pos[u][2], pos[v][2], None])
        if not xs:
            continue
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines",
            line=dict(color=EDGE_COLOR.get(rel, "#999"),
                      width=4 if rel in ("fills_opening_in", "connects") else 2),
            name=f"edge: {rel}",
            hoverinfo="skip",
        ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="world X (m)",
            yaxis_title="world Y (m)",
            zaxis_title="world Z (m)",
            aspectmode="data",
        ),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn")


def _node_hover_text(node: str, d: dict) -> str:
    """Compact HTML-safe hover popup."""
    nt = d.get("node_type", "?")
    parts = [f"<b>{html.escape(node)}</b>", f"type: {nt}"]
    if "label" in d:
        parts.append(f"label: {d['label']}")
    if "ifc_class" in d:
        parts.append(f"class: {d['ifc_class']}")
    if "name" in d and d.get("name"):
        parts.append(f"name: {d['name']}")
    if "centroid" in d:
        c = d["centroid"]
        parts.append(f"centroid: ({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f})")
    if "volume_m3" in d:
        parts.append(f"volume: {d['volume_m3']:.3f} m³")
    if "max_length_m" in d:
        parts.append(f"max length: {d['max_length_m']:.2f} m")
    if "area_m2" in d:
        parts.append(f"area: {d['area_m2']:.1f} m²")
    return "<br>".join(parts)

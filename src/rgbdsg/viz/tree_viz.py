"""Hierarchical tree visualization of the scene graph."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def render_tree_graph_png(
    G: nx.MultiDiGraph,
    out_path: Path | str,
    *,
    title: str = "Hierarchical Scene Graph",
    figsize: tuple[float, float] = (20.0, 14.0),
    dpi: int = 140,
) -> None:
    """Render the scene graph as a 2D hierarchical tree diagram (excluding fixtures)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Filter nodes to keep meaningful ones
    keep_types = {"building", "storey", "room", "object", "camera"}
    nodes_to_keep = [n for n, d in G.nodes(data=True) if d.get("node_type") in keep_types]
    T = G.subgraph(nodes_to_keep).copy()

    # 2. Assign layers for multipartite layout
    for n, d in T.nodes(data=True):
        nt = d.get("node_type")
        if nt == "building":
            layer = 0
        elif nt == "storey":
            layer = 1
        elif nt == "room":
            layer = 2
        elif nt == "object":
            layer = 3
        elif nt == "camera":
            layer = 4
        else:
            layer = 5
        T.nodes[n]["layer"] = layer

    # 3. Compute layout
    pos = nx.multipartite_layout(T, subset_key="layer", align='horizontal')

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_title(title, fontsize=18, fontweight='bold')

    # 4. Draw nodes
    node_colors = {
        "building": "#2c3e50",
        "storey": "#8e44ad",
        "room": "#2980b9",
        "object": "#27ae60",
        "camera": "#f39c12",
    }
    
    for nt, color in node_colors.items():
        nlist = [n for n, d in T.nodes(data=True) if d.get("node_type") == nt]
        if not nlist:
            continue
        nx.draw_networkx_nodes(T, pos, nodelist=nlist, node_color=color, 
                               node_size=900 if nt in ("object", "camera") else 700, 
                               alpha=0.9, ax=ax, edgecolors="white", linewidths=1.5)

    # 5. Draw edges
    edge_styles = {
        "contains": {"color": "#bdc3c7", "style": "solid", "alpha": 0.4},
        "connects": {"color": "#e74c3c", "style": "dashed", "alpha": 0.6},
        "nearest":  {"color": "#3498db", "style": "dotted", "alpha": 0.5},
        "near":     {"color": "#3498db", "style": "dotted", "alpha": 0.3},
        "next_to":  {"color": "#f1c40f", "style": "dashdot", "alpha": 0.5},
        "aligned_with": {"color": "#9b59b6", "style": "solid", "alpha": 0.3},
    }

    all_edges = list(T.edges(data=True))
    
    for rel, style in edge_styles.items():
        elist = [(u, v) for u, v, d in all_edges if d.get("relation") == rel]
        if not elist:
            continue
        nx.draw_networkx_edges(T, pos, edgelist=elist, edge_color=style["color"], 
                               style=style["style"], alpha=style["alpha"], 
                               arrows=True, arrowsize=15, ax=ax)

    # 6. Draw labels for nodes
    node_labels = {}
    for n, d in T.nodes(data=True):
        nt = d.get("node_type")
        if nt == "object":
            node_labels[n] = d.get("label", n)
        elif nt == "camera":
            node_labels[n] = f"Cam {d.get('frame_idx', '')}"
        elif nt == "room":
            node_labels[n] = f"Room {d.get('room_id', '')}"
        elif nt == "storey":
            node_labels[n] = d.get("name", n)
        elif nt == "building":
            node_labels[n] = d.get("name", "Building")

    nx.draw_networkx_labels(T, pos, labels=node_labels, font_size=8, 
                            font_weight='bold', font_color='white', ax=ax)

    # 7. Draw ALL edge labels from the data
    edge_labels = {}
    for u, v, d in all_edges:
        rel = d.get("relation", "unknown")
        # Include weight (distance) for spatial relationships if present
        weight = d.get("weight")
        if weight is not None and rel in ["nearest", "near", "next_to"]:
            label = f"{rel}\n({weight:.2f}m)"
        else:
            label = rel
        edge_labels[(u, v)] = label

    nx.draw_networkx_edge_labels(T, pos, edge_labels=edge_labels, font_size=6, 
                                 alpha=0.8, rotate=False, label_pos=0.6, ax=ax,
                                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, pad=0.5))

    # 8. Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=s["color"], linestyle=s["style"], label=rel)
        for rel, s in edge_styles.items() if any(d.get("relation") == rel for _, _, d in all_edges)
    ]
    if legend_elements:
        ax.legend(handles=legend_elements, loc='lower right', title="Relationship Types", fontsize=8)

    ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

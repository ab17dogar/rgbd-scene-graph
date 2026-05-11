"""Lightweight matplotlib visualisations of the pipeline outputs.

`bev_plot` produces the top-down hero image used in the README; the higher-
level driver `scripts/render_bev.py` wires it to a full pipeline output.
"""
from rgbdsg.viz.bev import bev_plot, trajectory_xy_from_sequence
from rgbdsg.viz.scene_graph_3d import render_3d_html, render_3d_png

__all__ = [
    "bev_plot",
    "render_3d_html",
    "render_3d_png",
    "trajectory_xy_from_sequence",
]

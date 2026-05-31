---
title: "RGB-D Semantic Scene Graphs"
summary: "An end-to-end pipeline turning egocentric RGB-D into 4-layer 3D semantic scene graphs, fusing open-vocabulary 2D foundation models with BIM/IFC priors."
tags: ["ML", "CV"]
stack: ["Python", "PyTorch", "Grounding DINO", "SAM 2.1", "Open3D", "Docker"]
order: 2
---

## Overview

A pipeline that transforms egocentric RGB-D sequences into hierarchical 3D
semantic scene graphs by pairing open-vocabulary perception with building priors.

## Perception & 3D fusion

- Open-vocabulary 2D detection (Grounding DINO, Swin-Base) with stateful video
  mask propagation (SAM 2.1 Hiera-Large)
- Back-projects masked depth to world coordinates; deduplicates detections across
  views via centroid + bbox + voxel-IoU merging

## BIM/IFC integration

- Extracts canonical structural entities (walls, doors, slabs, multi-storey
  hierarchies) from IFC models with IfcOpenShell
- Synthesises room polygons via point-cloud BEV occupancy morphology

## Scene graph & reproducibility

- 4-layer hierarchical graph (Building → Storey → Room → Object) with
  spatial-relationship edges built over a 3D KD-tree
- Containerised with Docker + Make; exports GraphML / node-link JSON plus 3D,
  BEV, and hierarchical-tree visualisations

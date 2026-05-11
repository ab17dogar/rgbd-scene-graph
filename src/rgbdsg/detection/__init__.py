"""Open-vocabulary 2D detection + segmentation.

    GroundingDINO       — open-vocab box detection from a text prompt.
    Detection           — one detected box (xyxy) with score + matched phrase.
    SAM2VideoSegmenter  — multi-frame mask propagation from box prompts.
    FrameSegmentation   — per-frame masks for one or more tracked objects.
"""
from rgbdsg.detection.grounding_dino import Detection, GroundingDINO
from rgbdsg.detection.owlv2 import OWLv2, merge_detections_iou
from rgbdsg.detection.sam2_video import FrameSegmentation, SAM2VideoSegmenter

__all__ = [
    "Detection",
    "FrameSegmentation",
    "GroundingDINO",
    "OWLv2",
    "SAM2VideoSegmenter",
    "merge_detections_iou",
]

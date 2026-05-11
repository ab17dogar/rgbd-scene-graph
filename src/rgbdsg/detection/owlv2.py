"""OWLv2 open-vocabulary detector wrapper, complementary to Grounding DINO."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from rgbdsg.detection.grounding_dino import Detection, _resolve_device

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

DEFAULT_OWLV2_MODEL_ID = "google/owlv2-base-patch16-ensemble"


def _normalize_owlv2_queries(prompt: str) -> list[str]:
    """OWLv2 wants a list of phrases, not the GDINO period-joined string."""
    raw = prompt.replace("\n", ".").replace(",", ".")
    parts = [p.strip() for p in raw.split(".") if p.strip()]
    return parts


class OWLv2:
    """Lazy-loaded OWLv2 inference wrapper."""

    def __init__(
        self,
        device: str | torch.device = "auto",
        model_id: str = DEFAULT_OWLV2_MODEL_ID,
        cache_dir: str | Path | None = "weights/hf_cache",
    ) -> None:
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.device = _resolve_device(device)
        self.model_id = model_id
        self.cache_dir = str(cache_dir) if cache_dir else None

        self.processor = Owlv2Processor.from_pretrained(
            model_id, cache_dir=self.cache_dir
        )
        self.model = Owlv2ForObjectDetection.from_pretrained(
            model_id, cache_dir=self.cache_dir,
        ).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def detect(
        self,
        image: np.ndarray,
        prompt: str,
        threshold: float = 0.10,
    ) -> list[Detection]:
        """Run OWLv2 detection on one RGB image with the given prompt."""
        from PIL import Image

        if image.dtype != np.uint8:
            raise TypeError(f"expected uint8 RGB image, got {image.dtype}")
        H, W = image.shape[:2]

        queries = _normalize_owlv2_queries(prompt)
        if not queries:
            return []

        pil = Image.fromarray(image)
        inputs = self.processor(
            images=pil, text=[queries], return_tensors="pt",
        ).to(self.device)
        outputs = self.model(**inputs)

        # OWLv2 ships its own post-processor (Pascal-VOC-style boxes).
        target_sizes = torch.tensor([[H, W]], device=self.device)
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=threshold,
            text_labels=[queries],
        )[0]

        out: list[Detection] = []
        boxes = results["boxes"].detach().cpu().numpy()
        scores = results["scores"].detach().cpu().numpy()
        labels = results["text_labels"]
        for box, score, label in zip(boxes, scores, labels):
            out.append(Detection(
                bbox=np.asarray(box, dtype=np.float32),
                score=float(score),
                label=str(label),
            ))
        return out


def merge_detections_iou(
    a: list[Detection],
    b: list[Detection],
    iou_threshold: float = 0.5,
) -> list[Detection]:
    """Merge two detector outputs by 2D-bbox IoU."""
    if not a:
        return list(b)
    if not b:
        return list(a)

    out: list[Detection] = []
    used_b: set[int] = set()
    for da in a:
        best_j, best_iou = -1, 0.0
        for j, db in enumerate(b):
            if j in used_b:
                continue
            iou = _bbox_iou_xyxy(da.bbox, db.bbox)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou > iou_threshold and best_j >= 0:
            used_b.add(best_j)
            chosen = da if da.score >= b[best_j].score else b[best_j]
            out.append(chosen)
        else:
            out.append(da)
    for j, db in enumerate(b):
        if j not in used_b:
            out.append(db)
    return out


def _bbox_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    bb = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = aa + bb - inter
    return float(inter / union) if union > 0 else 0.0

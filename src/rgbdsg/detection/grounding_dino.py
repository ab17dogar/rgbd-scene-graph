"""Open-vocabulary 2D object detection via Grounding DINO.

We use HuggingFace's `transformers` integration of Grounding DINO rather than
the official IDEA-Research repo because:

    1. The HF wrapper avoids building a CUDA C++ extension (unnecessary on
       MPS / Apple Silicon, which is our target hardware).
    2. The same model weights and architecture, with a stable API.
    3. `from_pretrained()` handles caching and download automatically.

Model: `IDEA-Research/grounding-dino-base` (Swin-Base backbone, 1.6 GB).
Smaller alternative: `IDEA-Research/grounding-dino-tiny` (660 MB) — swap by
changing one constant.

Reference: Liu et al., "Grounding DINO: Marrying DINO with Grounded Pre-
Training for Open-Set Object Detection" (2023).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# Allow MPS fallbacks for ops that aren't yet implemented natively.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

DEFAULT_MODEL_ID = "IDEA-Research/grounding-dino-base"


@dataclass
class Detection:
    """One open-vocab detection in image coordinates.

    bbox: (x_min, y_min, x_max, y_max) in pixels (xyxy format).
    score: confidence in [0, 1].
    label: the matched text phrase (e.g. "chair").
    """
    bbox: np.ndarray  # shape (4,) float32
    score: float
    label: str

    @property
    def cx(self) -> float:
        return float((self.bbox[0] + self.bbox[2]) / 2.0)

    @property
    def cy(self) -> float:
        return float((self.bbox[1] + self.bbox[3]) / 2.0)

    @property
    def width(self) -> float:
        return float(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])


class GroundingDINO:
    """Lazy-loaded Grounding DINO inference wrapper.

    Heavy initialisation (model + processor) happens in `__init__`, but the
    expensive forward pass is in `detect()`.

    Usage:
        det = GroundingDINO(device="mps")
        results = det.detect(rgb_image, prompt="chair. table. lamp.",
                             box_threshold=0.35, text_threshold=0.25)
    """

    def __init__(
        self,
        device: str | torch.device = "auto",
        model_id: str = DEFAULT_MODEL_ID,
        cache_dir: str | Path | None = "weights/hf_cache",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.device = _resolve_device(device)
        self.model_id = model_id
        self.cache_dir = str(cache_dir) if cache_dir else None

        self.processor = AutoProcessor.from_pretrained(
            model_id, cache_dir=self.cache_dir
        )
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id, cache_dir=self.cache_dir, torch_dtype=dtype,
        ).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def detect(
        self,
        image: np.ndarray,
        prompt: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> list[Detection]:
        """Run open-vocab detection on a single RGB image.

        Args:
            image: HxWx3 uint8 RGB array.
            prompt: period-separated phrases, e.g. "chair. table. lamp.".
                Grounding DINO expects this exact format; the trailing period
                is non-optional. We normalize lazily inside this function.
            box_threshold: minimum predicted box confidence in [0, 1].
            text_threshold: minimum text-image alignment score in [0, 1].

        Returns:
            List of Detection objects, one per accepted box. Empty list if
            nothing scores above threshold.
        """
        if image.dtype != np.uint8:
            raise TypeError(f"expected uint8 RGB image, got {image.dtype}")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"expected HxWx3 image, got shape {image.shape}")

        prompt = _normalize_prompt(prompt)
        H, W = image.shape[:2]

        inputs = self.processor(
            images=image, text=prompt, return_tensors="pt"
        ).to(self.device)

        outputs = self.model(**inputs)

        # post_process_grounded_object_detection expects target_sizes=(H, W).
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[(H, W)],
        )[0]

        # Different transformers versions return labels under different keys
        # ("labels" or "text_labels"); handle both.
        labels = results.get("text_labels", results.get("labels", []))
        return [
            Detection(
                bbox=box.detach().cpu().numpy().astype(np.float32),
                score=float(score),
                label=str(label) if not isinstance(label, str) else label,
            )
            for box, score, label in zip(results["boxes"], results["scores"], labels)
        ]


# ---------- helpers ---------------------------------------------------------

def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def _normalize_prompt(prompt: str) -> str:
    """Ensure a Grounding DINO prompt is period-separated and lowercased.

    Grounding DINO is sensitive to formatting: phrases must be separated by
    `. ` and the whole string must end in a period. Casing matters less but
    lowercase is the convention in the original paper.
    """
    p = prompt.strip().lower()
    # Normalize separators: replace newlines and commas with periods.
    for sep in ("\n", ","):
        p = p.replace(sep, ".")
    # Collapse multiple periods, ensure exactly one trailing period.
    while ".." in p:
        p = p.replace("..", ".")
    if not p.endswith("."):
        p += "."
    return p

"""SAM 2.1 video-mode segmentation wrapper."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# MPS does not implement a few SAM 2 ops; fall back to CPU for those silently.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

SAM2_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEFAULT_CKPT = Path("weights/sam2.1_hiera_large.pt")


# Module-level patch flag so we don't repeatedly rewrite the methods.
_BF16_PATCH_APPLIED = False


def _patch_sam2_bf16_storage_to_fp32() -> None:
    """Replace SAM 2's bfloat16 memory-bank storage with fp32 in-place."""
    global _BF16_PATCH_APPLIED
    if _BF16_PATCH_APPLIED:
        return

    import inspect
    import textwrap

    import sam2.sam2_video_predictor as svp

    cls = svp.SAM2VideoPredictor
    for method_name in ("_run_single_frame_inference", "_run_memory_encoder"):
        original = getattr(cls, method_name)
        src = textwrap.dedent(inspect.getsource(original))
        if "torch.bfloat16" not in src:
            continue
        patched = src.replace("torch.bfloat16", "torch.float32")
        ns: dict = {}
        exec(compile(patched, f"<sam2-patch:{method_name}>", "exec"),
             svp.__dict__, ns)
        setattr(cls, method_name, ns[method_name])

    _BF16_PATCH_APPLIED = True


@dataclass
class FrameSegmentation:
    """Per-frame segmentation result for one or more tracked objects."""
    frame_idx: int
    obj_ids: list[int] = field(default_factory=list)
    # masks[i] is a HxW bool array for obj_ids[i]
    masks: list[np.ndarray] = field(default_factory=list)


def _ensure_jpg_cache(rgb_dir: Path, quality: int = 95) -> Path:
    """Mirror PNG frames into a JPEG cache directory for SAM 2."""
    rgb_dir = Path(rgb_dir)
    pngs = sorted(rgb_dir.glob("*.png"))
    if not pngs:
        raise FileNotFoundError(f"no PNG frames in {rgb_dir}")

    cache = rgb_dir.parent / "_sam2_jpg_cache"
    existing = sorted(cache.glob("*.jpg")) if cache.is_dir() else []
    if cache.is_dir() and len(existing) == len(pngs):
        return cache

    cache.mkdir(exist_ok=True)
    for png in pngs:
        jpg = cache / (png.stem + ".jpg")
        if jpg.exists():
            continue
        Image.open(png).convert("RGB").save(jpg, quality=quality)
    return cache


class SAM2VideoSegmenter:
    """Lazy wrapper around SAM 2's video predictor."""

    def __init__(
        self,
        device: str | torch.device = "auto",
        model_cfg: str = SAM2_MODEL_CFG,
        ckpt_path: str | Path = DEFAULT_CKPT,
        allow_mps: bool = True,
    ) -> None:
        """Build the SAM 2.1 video predictor."""
        from sam2.build_sam import build_sam2_video_predictor

        requested = self._resolve_device(device)
        if requested.type == "mps" and not allow_mps:
            self.device = torch.device("cpu")
        else:
            self.device = requested

        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"SAM 2 checkpoint not at {ckpt_path}. Run the weight-download "
                "step first (see scripts/download_weights.py)."
            )

        _patch_sam2_bf16_storage_to_fp32()

        self.predictor = build_sam2_video_predictor(
            model_cfg, str(ckpt_path), device=self.device,
        )
        self._state = None
        self._height = None
        self._width = None
        self._n_frames = None
        self._prompts: list[tuple[int, int, np.ndarray]] = []

    @staticmethod
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

    def init_video(
        self,
        rgb_dir: str | Path,
        offload_video_to_cpu: bool = True,
        offload_state_to_cpu: bool = True,
    ) -> None:
        """Initialise SAM 2's per-clip state from a directory of RGB frames."""
        rgb_dir = Path(rgb_dir)
        jpg_dir = _ensure_jpg_cache(rgb_dir)
        self._state = self.predictor.init_state(
            video_path=str(jpg_dir),
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu,
        )
        self._n_frames = len(sorted(jpg_dir.glob("*.jpg")))
        # Get image dimensions from the first frame.
        with Image.open(next(iter(jpg_dir.glob("*.jpg")))) as im:
            self._width, self._height = im.size

    def add_box_prompt(
        self,
        frame_idx: int,
        obj_id: int,
        box: np.ndarray | tuple[float, float, float, float],
    ) -> None:
        """Provide a 2D bounding-box prompt for one object on one frame."""
        self._require_init()
        box_arr = np.asarray(box, dtype=np.float32).reshape(4)
        self.predictor.add_new_points_or_box(
            inference_state=self._state,
            frame_idx=int(frame_idx),
            obj_id=int(obj_id),
            box=box_arr,
        )
        self._prompts.append((int(frame_idx), int(obj_id), box_arr))

    def _reset_for_reverse_sweep(self) -> None:
        """Clear SAM 2's per-clip state and replay every recorded prompt."""
        self._require_init()
        self.predictor.reset_state(self._state)
        for frame_idx, obj_id, box_arr in self._prompts:
            self.predictor.add_new_points_or_box(
                inference_state=self._state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                box=box_arr,
            )

    def propagate(self) -> list[FrameSegmentation]:
        """Run forward + reverse propagation; return one segmentation per frame."""
        self._require_init()
        n = self._n_frames or 0

        forward = self._collect_sweep(reverse=False)

        self._reset_for_reverse_sweep()
        reverse = self._collect_sweep(reverse=True)

        merged: dict[int, dict[int, np.ndarray]] = {}
        for sweep in (forward, reverse):
            for frame_idx, by_obj in sweep.items():
                for obj_id, mask in by_obj.items():
                    if obj_id not in merged.setdefault(frame_idx, {}):
                        merged[frame_idx][obj_id] = mask
                    else:
                        # Larger mask wins (more pixels = more SAM-2 confidence).
                        if int(mask.sum()) > int(merged[frame_idx][obj_id].sum()):
                            merged[frame_idx][obj_id] = mask

        out: list[FrameSegmentation] = []
        for frame_idx in range(n):
            by_obj = merged.get(frame_idx, {})
            out.append(FrameSegmentation(
                frame_idx=frame_idx,
                obj_ids=list(by_obj.keys()),
                masks=list(by_obj.values()),
            ))
        return out

    def _collect_sweep(self, reverse: bool) -> dict[int, dict[int, np.ndarray]]:
        """One propagation pass; returns {frame_idx: {obj_id: mask}}."""
        out: dict[int, dict[int, np.ndarray]] = {}
        try:
            for out_frame_idx, out_obj_ids, out_mask_logits in \
                    self.predictor.propagate_in_video(self._state, reverse=reverse):
                fi = int(out_frame_idx)
                for obj_id, logit in zip(out_obj_ids, out_mask_logits):
                    mask = (logit[0] > 0.0).detach().cpu().numpy()
                    out.setdefault(fi, {})[int(obj_id)] = mask
        except Exception as e:
            print(f"  [warn] {'reverse' if reverse else 'forward'} sweep "
                  f"stopped early: {type(e).__name__}: {e}")
        return out

    def reset(self) -> None:
        """Discard the current per-clip state and any added prompts."""
        if self._state is not None:
            self.predictor.reset_state(self._state)

    def _require_init(self) -> None:
        if self._state is None:
            raise RuntimeError("call init_video(...) before adding prompts.")

    @property
    def n_frames(self) -> int | None:
        return self._n_frames

    @property
    def image_size(self) -> tuple[int, int] | None:
        """(width, height) of the source frames, or None before init."""
        if self._width is None:
            return None
        return self._width, self._height

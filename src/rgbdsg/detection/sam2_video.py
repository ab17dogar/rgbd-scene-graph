"""SAM 2.1 video-mode segmentation wrapper.

Why video mode (not image mode)?
================================
SAM 2 has two operating modes:

  * Image mode — per-frame mask prediction; the user (or another model)
    must associate the same physical object across frames separately.
  * Video mode — propagates a mask through a clip using SAM 2's internal
    memory bank. The same `obj_id` follows a single object across frames.

For an RGB-D scene-graph pipeline, multi-view association of detections is
*the* hardest sub-problem. Most pipelines do it with handcrafted Hungarian
matching on 3D centroids + appearance descriptors and accumulate errors. By
running SAM 2 in video mode and feeding it Grounding DINO's boxes as object
prompts on a small set of keyframes, we get cross-frame object identity
*for free* — exactly the principle SAM 2 was designed around.

Frame-format constraint
=======================
SAM 2's video frame loader only accepts JPEG files (see
`sam2.utils.misc.load_video_frames_from_jpg_images`). Our data ships PNG, so
we transparently convert and cache JPEGs in `<scene>/_sam2_jpg_cache/` on
first use. The cache is reused on subsequent runs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# MPS does not implement a few SAM 2 ops; fall back to CPU for those silently.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# We hard-code the model config name for sam2.1_hiera_large here. SAM 2 ships
# multiple model sizes; if we ever swap, this is the only line that changes.
SAM2_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEFAULT_CKPT = Path("weights/sam2.1_hiera_large.pt")


# Module-level patch flag so we don't repeatedly rewrite the methods.
_BF16_PATCH_APPLIED = False


def _patch_sam2_bf16_storage_to_fp32() -> None:
    """Replace SAM 2's bfloat16 memory-bank storage with fp32 in-place.

    SAM 2's `_run_single_frame_inference` and `_run_memory_encoder` contain
    the literal:

        maskmem_features = maskmem_features.to(torch.bfloat16)

    which is a storage compaction step. On CUDA the downstream matmul auto-
    promotes between bf16 and fp32; on CPU and MPS it does not, and the
    multi-object propagation pipeline crashes. We rewrite both methods to
    use fp32 instead of bf16. This is idempotent (guarded by a module flag)
    so importing the wrapper twice is safe.
    """
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
        # Recompile inside the original module's globals so all referenced
        # names (`torch`, helper functions, etc.) still resolve.
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


# ---------- JPEG cache utility ----------------------------------------------

def _ensure_jpg_cache(rgb_dir: Path, quality: int = 95) -> Path:
    """Mirror PNG frames into a JPEG cache directory for SAM 2.

    Returns the cache directory path. Idempotent — if the cache already has
    one JPEG per source PNG, no re-encoding happens.
    """
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


# ---------- main wrapper ----------------------------------------------------

class SAM2VideoSegmenter:
    """Lazy wrapper around SAM 2's video predictor.

    Lifecycle:
        seg = SAM2VideoSegmenter(device="mps")
        seg.init_video("data/BasicHouse_with_pc/rgb")
        seg.add_box_prompt(frame_idx=0, obj_id=1, box=(x1, y1, x2, y2))
        seg.add_box_prompt(frame_idx=0, obj_id=2, box=(x1, y1, x2, y2))
        for fs in seg.propagate():
            # fs.frame_idx, fs.obj_ids, fs.masks
            ...

    Multiple `add_box_prompt` calls on the same `obj_id` refine the prompt for
    that object on the given frame. To start a new object, increment `obj_id`.
    """

    def __init__(
        self,
        device: str | torch.device = "auto",
        model_cfg: str = SAM2_MODEL_CFG,
        ckpt_path: str | Path = DEFAULT_CKPT,
        allow_mps: bool = True,
    ) -> None:
        """Build the SAM 2.1 video predictor.

        Args:
            device: "auto", "cuda", "mps", or "cpu".
            allow_mps: keep MPS when requested (default). The MPS fp32 patch
                below makes multi-object propagation work; on CUDA bf16 is
                fine and we leave it alone.

        The bfloat16 patch
        ------------------
        SAM 2's `_run_single_frame_inference` and `_run_memory_encoder` cast
        memory-bank features to bf16 (for storage compactness) while the
        model weights remain fp32. On CUDA the matmul kernels auto-promote;
        on CPU and MPS they assert with a dtype mismatch. We patch the two
        methods to keep features as fp32 — the cost is doubling the memory
        bank size, which is negligible on Apple Silicon's unified 18-36 GB.
        """
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
        """Initialise SAM 2's per-clip state from a directory of RGB frames.

        PNG frames are auto-converted to JPEG (cached) because SAM 2's loader
        is JPEG-only. The cache lives at `<scene>/_sam2_jpg_cache/`.

        Args:
            rgb_dir: directory of source PNG frames.
            offload_video_to_cpu: keep the loaded frame tensors on CPU and
                stream them to the compute device per-frame. SAM 2's
                official flag for long-sequence memory pressure. With it
                enabled, a 383-frame synagoge clip uses ~3 GB instead of
                the ~12 GB that would otherwise force partial propagation.
            offload_state_to_cpu: same idea but for the inference state's
                memory bank. Together with the above, this keeps the
                compute-device working set bounded by the per-frame
                processing cost rather than the full clip length.
        """
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
        """Provide a 2D bounding-box prompt for one object on one frame.

        Args:
            frame_idx: which frame to anchor the prompt to.
            obj_id: arbitrary positive integer identifying the object across
                frames. Two prompts with the same obj_id are interpreted as
                refinements of the same physical object.
            box: (x_min, y_min, x_max, y_max) in pixels.
        """
        self._require_init()
        box_arr = np.asarray(box, dtype=np.float32).reshape(4)
        self.predictor.add_new_points_or_box(
            inference_state=self._state,
            frame_idx=int(frame_idx),
            obj_id=int(obj_id),
            box=box_arr,
        )
        # Record so we can replay between forward + reverse sweeps after
        # `reset_state` clears the SAM 2 memory bank.
        self._prompts.append((int(frame_idx), int(obj_id), box_arr))

    def _reset_for_reverse_sweep(self) -> None:
        """Clear SAM 2's per-clip state and replay every recorded prompt.

        Called between the forward and reverse propagation sweeps so the
        reverse sweep doesn't inherit the forward sweep's memory bank.
        Without this isolation, the forward run's masks would seed the
        reverse run, biasing the merge.
        """
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
        """Run forward + reverse propagation; return one segmentation per frame.

        SAM 2's `propagate_in_video` runs a single sweep starting from a
        chosen frame index. Calling it once forward from the smallest prompt
        frame covers most cases, but for full robustness we explicitly run
        forward AND reverse and merge the results.

        Critically, the reverse sweep runs against an **isolated** state —
        we snapshot the prompt set, call `reset_state()` to clear the
        memory bank that the forward sweep populated, replay every prompt
        on the fresh state, and only then run the reverse sweep. Without
        this, the reverse-direction masks would carry forward-direction
        memory-bank context and bias toward whatever masks the forward
        sweep produced (which is exactly what we're trying to cross-check).

        Per-frame results from the two sweeps are merged by keeping the
        larger mask per (frame_idx, obj_id) — SAM 2 reports more confident
        boundaries with larger pixel counts on textured objects.
        """
        self._require_init()
        n = self._n_frames or 0

        # Snapshot prompts before either sweep, so we can replay them after
        # the reset between sweeps. (`add_box_prompt` recorded them into
        # self._prompts; the SAM 2 state itself stores them internally too.)
        forward = self._collect_sweep(reverse=False)

        # Reset SAM 2's per-clip state (clears memory bank but keeps the
        # video frame loader). Replay prompts so the reverse sweep starts
        # from a clean state with the same anchors.
        self._reset_for_reverse_sweep()
        reverse = self._collect_sweep(reverse=True)

        # Merge: for each frame, prefer non-empty masks; if both sweeps have
        # one for the same obj_id, keep the larger mask area (more votes).
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
            # Out-of-memory or any other propagation failure: keep whatever
            # we got. Caller's `propagate()` merges with the other direction.
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

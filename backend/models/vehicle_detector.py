"""
GPU-OPTIMIZED VEHICLE DETECTOR
================================
YOLOv11m — supports both PyTorch (.pt) and TensorRT (.engine) backends.

TensorRT engines exported with a fixed max_batch_size (e.g. 8) CANNOT
accept batch=1 during Ultralytics warmup — the engine hard-asserts the
input shape matches max_batch_size.  We detect this at load time and
expose two paths:

  self.is_trt      — True when a .engine file is loaded
  self.trt_batch   — The max_batch_size the engine was compiled with
  self.detect()    — Single-frame inference (always works, pads for TRT)
  self.detect_batch() — Multi-frame inference; for TRT always sends
                        exactly trt_batch frames (padding the tail)

video_pipeline uses detect_batch() so YOLO runs once per N frames
instead of once per frame.
"""

import numpy as np
import torch
from ultralytics import YOLO
from config.settings import settings


class VehicleDetector:
    VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}
    PERSON_CLASS    = "person"
    COCO_IDS        = [0, 2, 3, 5, 7]   # person, car, motorcycle, bus, truck

    def __init__(self, model_path: str = settings.VEHICLE_MODEL_PATH):
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"
        self.conf      = float(settings.VEHICLE_CONF_THRESHOLD)
        self.model     = YOLO(model_path)
        self.is_trt    = str(model_path).endswith(".engine")
        self.trt_batch = self._probe_trt_batch() if self.is_trt else 1
        self.half      = self.is_trt   # TRT engines are already quantised

        print(
            f"[VehicleDetector] {self.device} | TRT={self.is_trt} "
            f"| trt_batch={self.trt_batch} | FP16={self.half}"
        )

    # ── Public API ────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> list:
        """Single-frame inference. Always returns a list of detection dicts."""
        if self.is_trt and self.trt_batch > 1:
            # TRT engine needs exactly trt_batch frames — pad with the same frame
            return self.detect_batch([frame] * self.trt_batch)[0]
        return self._run_predict([frame])[0]

    def detect_batch(self, frames: list) -> list:
        """
        Multi-frame inference.  Returns a list (one entry per input frame)
        of detection lists.

        For TRT engines the input length MUST equal trt_batch exactly.
        video_pipeline guarantees this via tail-padding in video_reader.
        If called with a different length anyway, we pad/trim to be safe.
        """
        if self.is_trt:
            frames = self._pad_to_trt_batch(frames)
        return self._run_predict(frames)

    # ── Internals ─────────────────────────────────────────────────────────

    def _run_predict(self, frames: list) -> list:
        """Run model.predict on a list of frames, return per-frame det lists."""
        results = self.model.predict(
            source=frames,
            conf=self.conf,
            device=self.device,
            classes=self.COCO_IDS,
            imgsz=640,
            half=self.half,
            verbose=False,
        )
        per_frame = []
        for r in results:
            dets = []
            if r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy.squeeze().cpu().numpy())
                    cls_id = int(box.cls.item())
                    dets.append({
                        "bbox":       [x1, y1, x2, y2],
                        "confidence": float(box.conf.item()),
                        "class":      self.model.names[cls_id],
                    })
            per_frame.append(dets)
        return per_frame

    def _pad_to_trt_batch(self, frames: list) -> list:
        """Pad (or trim) frame list to exactly trt_batch using last frame repeat."""
        n = self.trt_batch
        if len(frames) == n:
            return frames
        if len(frames) > n:
            return frames[:n]
        # pad
        last = frames[-1] if frames else np.zeros((640, 640, 3), dtype=np.uint8)
        return frames + [last] * (n - len(frames))

    def _probe_trt_batch(self) -> int:
        """
        Read the max batch size from the loaded TRT engine without running
        a forward pass (which would crash at the wrong batch size).
        Falls back to 8 if introspection fails.
        """
        try:
            # Ultralytics wraps TRT in autobackend; the engine is at .predictor
            # or directly accessible via the underlying backend after the model
            # is loaded.  Safest path: read from the compiled engine binding.
            backend = self.model.model   # nn.Module or TRT backend wrapper
            # Ultralytics TRT backend stores max_batch as an attribute
            if hasattr(backend, "batch"):
                return int(backend.batch)
            # Fallback: peek at the engine directly
            if hasattr(backend, "engine"):
                profile_idx = 0
                binding_idx = 0  # first input binding
                shape = backend.engine.get_profile_shape(profile_idx, binding_idx)
                # shape is (min, opt, max) — we want max[0]
                return int(shape[2][0])
        except Exception as e:
            print(f"[VehicleDetector] TRT batch probe failed ({e}), assuming 8")
        return 8

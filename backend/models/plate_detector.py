"""
GPU-OPTIMIZED PLATE DETECTOR
==============================
YOLOv10s — supports both PyTorch (.pt) and TensorRT (.engine) backends.
Same TRT batch-size handling as VehicleDetector.
"""

import numpy as np
import torch
from ultralytics import YOLO
from config.settings import settings


class PlateDetector:
    def __init__(self, model_path: str = settings.PLATE_MODEL_PATH):
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"
        self.model     = YOLO(model_path)
        self.is_trt    = str(model_path).endswith(".engine")
        self.trt_batch = self._probe_trt_batch() if self.is_trt else 1
        self.half      = self.is_trt

        print(
            f"[PlateDetector] {self.device} | TRT={self.is_trt} "
            f"| trt_batch={self.trt_batch} | FP16={self.half}"
        )

    # ── Public API ────────────────────────────────────────────────────────

    def predict(self, image: np.ndarray) -> list:
        """Single-frame inference. Returns list of [x1,y1,x2,y2] boxes."""
        if self.is_trt and self.trt_batch > 1:
            return self.predict_batch([image] * self.trt_batch)[0]
        return self._run_predict([image])[0]

    def predict_batch(self, frames: list) -> list:
        """
        Multi-frame inference. Returns list (one entry per frame) of box lists.
        For TRT engines input length must equal trt_batch; we pad/trim if needed.
        """
        if self.is_trt:
            frames = self._pad_to_trt_batch(frames)
        return self._run_predict(frames)

    # ── Internals ─────────────────────────────────────────────────────────

    def _run_predict(self, frames: list) -> list:
        results = self.model.predict(
            source=frames,
            conf=0.4,
            imgsz=768,
            half=self.half,
            verbose=False,
            device=self.device,
        )
        per_frame = []
        for r in results:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                per_frame.append([])
            else:
                per_frame.append([
                    list(map(int, box.xyxy[0].cpu().numpy()))
                    for box in boxes
                ])
        return per_frame

    def _pad_to_trt_batch(self, frames: list) -> list:
        n = self.trt_batch
        if len(frames) == n:
            return frames
        if len(frames) > n:
            return frames[:n]
        last = frames[-1] if frames else np.zeros((768, 768, 3), dtype=np.uint8)
        return frames + [last] * (n - len(frames))

    def _probe_trt_batch(self) -> int:
        try:
            backend = self.model.model
            if hasattr(backend, "batch"):
                return int(backend.batch)
            if hasattr(backend, "engine"):
                shape = backend.engine.get_profile_shape(0, 0)
                return int(shape[2][0])
        except Exception as e:
            print(f"[PlateDetector] TRT batch probe failed ({e}), assuming 8")
        return 8

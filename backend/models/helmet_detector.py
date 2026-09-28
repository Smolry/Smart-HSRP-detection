"""
GPU-OPTIMIZED HELMET DETECTOR
================================
YOLOv8s — FP16 on GPU. Low conf=0.25, temporal fusion decides.

TensorRT engines exported with a fixed max_batch_size CANNOT accept
batch=1 during Ultralytics warmup.  We detect this at load time and
pad single-image calls to trt_batch so the engine always gets the
right input shape.
"""
import torch
import numpy as np
from ultralytics import YOLO
from config.settings import settings


class HelmetDetector:
    def __init__(self, model_path: str = settings.HELMET_MODEL_PATH):
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"
        self.model     = YOLO(model_path)
        self.is_trt    = str(model_path).endswith(".engine")
        self.trt_batch = self._probe_trt_batch() if self.is_trt else 1
        self.half      = self.is_trt

        print(
            f"[HelmetDetector] {self.device} | TRT={self.is_trt} "
            f"| trt_batch={self.trt_batch} | classes={self.model.names}"
        )

    # ── Public API ────────────────────────────────────────────────────────

    def predict(self, image: np.ndarray) -> dict:
        """Single-image inference. Always returns a result dict."""
        if image is None or image.size == 0:
            return self._empty()
        return self.predict_batch([image])[0]

    def predict_batch(self, images: list) -> list:
        """
        Multi-image inference on a list of head crops.
        Returns a list of result dicts (one per input image).
        For TRT engines pads to trt_batch; caller should only use
        the first len(images) results.
        """
        if not images:
            return []

        n_real = len(images)

        if self.is_trt and self.trt_batch > 1:
            images = self._pad_to_trt_batch(images)

        results = self.model(
            images,
            conf=0.25,
            half=self.half,
            verbose=False,
            device=self.device,
        )

        out = []
        for r in results[:n_real]:   # discard padded slots
            out.append(self._parse_result(r))
        return out

    # ── Internals ─────────────────────────────────────────────────────────

    def _parse_result(self, r) -> dict:
        boxes = r.boxes
        helmet_conf, no_helmet_conf           = 0.0, 0.0
        best_helmet_bbox, best_no_helmet_bbox = None, None
        detections = []

        if boxes is not None:
            for box in boxes:
                cls_id   = int(box.cls.cpu())
                conf     = float(box.conf.cpu())
                raw_name = self.model.names.get(cls_id, "").strip().lower()
                x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])

                if raw_name == "helmet":
                    label = "HELMET"
                    if conf > helmet_conf:
                        helmet_conf, best_helmet_bbox = conf, [x1, y1, x2, y2]
                elif raw_name in ("no-helmet", "no_helmet"):
                    label = "NO_HELMET"
                    if conf > no_helmet_conf:
                        no_helmet_conf, best_no_helmet_bbox = conf, [x1, y1, x2, y2]
                else:
                    label = "UNKNOWN"
                detections.append({
                    "class": label,
                    "confidence": round(conf, 4),
                    "bbox": [x1, y1, x2, y2],
                })

        if no_helmet_conf > helmet_conf and no_helmet_conf >= 0.4:
            status, final_conf, bbox = "NO_HELMET", no_helmet_conf, best_no_helmet_bbox
        elif helmet_conf >= 0.4:
            status, final_conf, bbox = "HELMET", helmet_conf, best_helmet_bbox
        else:
            status, final_conf, bbox = "UNCERTAIN", max(helmet_conf, no_helmet_conf), None

        return {
            "status":     status,
            "confidence": round(final_conf, 4),
            "bbox":       bbox,
            "detections": detections,
        }

    def _pad_to_trt_batch(self, images: list) -> list:
        n = self.trt_batch
        if len(images) >= n:
            return images[:n]
        last = images[-1]
        return images + [last] * (n - len(images))

    def _probe_trt_batch(self) -> int:
        try:
            backend = self.model.model
            if hasattr(backend, "batch"):
                return int(backend.batch)
            if hasattr(backend, "engine"):
                shape = backend.engine.get_profile_shape(0, 0)
                return int(shape[2][0])
        except Exception as e:
            print(f"[HelmetDetector] TRT batch probe failed ({e}), assuming 8")
        return 8

    def _empty(self) -> dict:
        return {"status": "UNCERTAIN", "confidence": 0.0, "bbox": None, "detections": []}

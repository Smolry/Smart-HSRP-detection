"""
TENSORRT-OPTIMIZED HELMET DETECTOR
=====================================
Uses TensorRT engine (.engine) if available, falls back to YOLO PyTorch.
Engine is auto-exported on first run and cached at weights/helmet.engine

Export once manually:
    yolo export model=weights/helmet.pt format=engine device=0 batch=8
"""
import os
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from config.settings import settings


def _trt_engine_path(pt_path: str) -> str:
    return str(Path(pt_path).with_suffix(".engine"))


def _export_to_trt(pt_path: str, batch: int = 8) -> str:
    engine_path = _trt_engine_path(pt_path)
    print(f"[HelmetDetector] Exporting TensorRT engine → {engine_path} (batch={batch})")
    m = YOLO(pt_path)
    m.export(format="engine", device=0, batch=batch, imgsz=640, half=True, simplify=True)
    return engine_path


class HelmetDetector:
    def __init__(self, model_path: str = settings.HELMET_MODEL_PATH):
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"
        self.using_trt = False

        if self.device == "cuda":
            engine_path = _trt_engine_path(model_path)
            if not os.path.exists(engine_path):
                try:
                    engine_path = _export_to_trt(model_path)
                except Exception as e:
                    print(f"[HelmetDetector] TRT export failed ({e}), falling back to PyTorch")
                    engine_path = None

            if engine_path and os.path.exists(engine_path):
                self.model = YOLO(engine_path)
                self.using_trt = True
                print(f"[HelmetDetector] TensorRT engine loaded | FP16 | classes={self.model.names}")
            else:
                self.model = YOLO(model_path)
                print(f"[HelmetDetector] PyTorch fallback | classes={self.model.names}")
        else:
            self.model = YOLO(model_path)
            print(f"[HelmetDetector] CPU mode (no TRT) | classes={self.model.names}")

    def predict(self, image: np.ndarray) -> dict:
        if image is None or image.size == 0:
            return self._empty()

        results = self.model(image, conf=0.25, half=self.using_trt, verbose=False, device=self.device)
        return self._parse(results[0])

    def predict_batch(self, images: list) -> list:
        """Batch helmet detection — faster with TRT on multiple crops."""
        valid = [(i, img) for i, img in enumerate(images) if img is not None and img.size > 0]
        if not valid:
            return [self._empty() for _ in images]

        indices, imgs = zip(*valid)
        results = self.model(
            list(imgs), conf=0.25, half=self.using_trt,
            verbose=False, device=self.device, stream=False,
        )
        parsed = {idx: self._parse(r) for idx, r in zip(indices, results)}
        return [parsed.get(i, self._empty()) for i in range(len(images))]

    def _parse(self, result) -> dict:
        boxes = result.boxes
        helmet_conf, no_helmet_conf             = 0.0, 0.0
        best_helmet_bbox, best_no_helmet_bbox   = None, None
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
                detections.append({"class": label, "confidence": round(conf, 4), "bbox": [x1, y1, x2, y2]})

        if no_helmet_conf > helmet_conf and no_helmet_conf >= 0.4:
            status, final_conf, bbox = "NO_HELMET", no_helmet_conf, best_no_helmet_bbox
        elif helmet_conf >= 0.4:
            status, final_conf, bbox = "HELMET", helmet_conf, best_helmet_bbox
        else:
            status, final_conf, bbox = "UNCERTAIN", max(helmet_conf, no_helmet_conf), None

        return {"status": status, "confidence": round(final_conf, 4), "bbox": bbox, "detections": detections}

    def _empty(self):
        return {"status": "UNCERTAIN", "confidence": 0.0, "bbox": None, "detections": []}

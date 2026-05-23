"""
TENSORRT-OPTIMIZED VEHICLE DETECTOR
=====================================
Uses TensorRT engine (.engine) if available, falls back to YOLO PyTorch.
Engine is auto-exported on first run and cached at weights/vehicle-person.engine

Export once manually (or let it auto-export on startup):
    yolo export model=weights/vehicle-person.pt format=engine device=0 batch=8 imgsz=640
"""
import os
import torch
from pathlib import Path
from ultralytics import YOLO
from config.settings import settings


def _trt_engine_path(pt_path: str) -> str:
    return str(Path(pt_path).with_suffix(".engine"))


def _export_to_trt(pt_path: str, batch: int = 8) -> str:
    engine_path = _trt_engine_path(pt_path)
    print(f"[VehicleDetector] Exporting TensorRT engine → {engine_path} (batch={batch})")
    m = YOLO(pt_path)
    m.export(format="engine", device=0, batch=batch, imgsz=640, half=True, simplify=True)
    return engine_path


class VehicleDetector:
    VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}
    PERSON_CLASS    = "person"
    COCO_IDS        = [0, 2, 3, 5, 7]  # person, car, motorcycle, bus, truck

    def __init__(self, model_path: str = settings.VEHICLE_MODEL_PATH):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.conf   = float(settings.VEHICLE_CONF_THRESHOLD)
        self.using_trt = False

        if self.device == "cuda":
            engine_path = _trt_engine_path(model_path)
            if not os.path.exists(engine_path):
                try:
                    engine_path = _export_to_trt(model_path)
                except Exception as e:
                    print(f"[VehicleDetector] TRT export failed ({e}), falling back to PyTorch")
                    engine_path = None

            if engine_path and os.path.exists(engine_path):
                self.model = YOLO(engine_path)
                self.using_trt = True
                print(f"[VehicleDetector] TensorRT engine loaded | batch=8 | FP16")
            else:
                self.model = YOLO(model_path)
                print(f"[VehicleDetector] PyTorch fallback | cuda | FP32")
        else:
            self.model = YOLO(model_path)
            print(f"[VehicleDetector] CPU mode (no TRT)")

    def detect(self, frame):
        """Single-frame detection. For batch use detect_batch()."""
        results = self.model.predict(
            source=frame, conf=self.conf, device=self.device,
            classes=self.COCO_IDS, imgsz=640, half=self.using_trt, verbose=False,
        )
        return self._parse(results)

    def detect_batch(self, frames: list) -> list:
        """Batch inference — significantly faster with TRT engine."""
        if not frames:
            return []
        results = self.model.predict(
            source=frames, conf=self.conf, device=self.device,
            classes=self.COCO_IDS, imgsz=640, half=self.using_trt, verbose=False,
            stream=False,
        )
        return [self._parse_single(r) for r in results]

    def _parse(self, results):
        out = []
        for r in results:
            out.extend(self._parse_single(r))
        return out

    def _parse_single(self, r):
        out = []
        if r.boxes is None:
            return out
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy.squeeze().cpu().numpy())
            cls_id = int(box.cls.item())
            out.append({
                "bbox":       [x1, y1, x2, y2],
                "confidence": float(box.conf.item()),
                "class":      self.model.names[cls_id],
            })
        return out

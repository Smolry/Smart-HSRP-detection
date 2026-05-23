"""
TENSORRT-OPTIMIZED PLATE DETECTOR
=====================================
Uses TensorRT engine (.engine) if available, falls back to YOLO PyTorch.
Engine is auto-exported on first run and cached at weights/plate_detector.engine

Export once manually:
    yolo export model=weights/plate_detector.pt format=engine device=0 batch=8 imgsz=768
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
    print(f"[PlateDetector] Exporting TensorRT engine → {engine_path} (batch={batch})")
    m = YOLO(pt_path)
    # imgsz=768 to keep small-plate accuracy
    m.export(format="engine", device=0, batch=batch, imgsz=768, half=True, simplify=True)
    return engine_path


class PlateDetector:
    def __init__(self, model_path: str = settings.PLATE_MODEL_PATH):
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"
        self.using_trt = False

        if self.device == "cuda":
            engine_path = _trt_engine_path(model_path)
            if not os.path.exists(engine_path):
                try:
                    engine_path = _export_to_trt(model_path)
                except Exception as e:
                    print(f"[PlateDetector] TRT export failed ({e}), falling back to PyTorch")
                    engine_path = None

            if engine_path and os.path.exists(engine_path):
                self.model = YOLO(engine_path)
                self.using_trt = True
                print(f"[PlateDetector] TensorRT engine loaded | imgsz=768 | FP16")
            else:
                self.model = YOLO(model_path)
                print(f"[PlateDetector] PyTorch fallback | cuda")
        else:
            self.model = YOLO(model_path)
            print(f"[PlateDetector] CPU mode (no TRT)")

    def predict(self, image) -> list:
        results = self.model(
            image, conf=0.4, imgsz=768, half=self.using_trt,
            verbose=False, device=self.device,
        )
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []
        return [list(map(int, box.xyxy[0].cpu().numpy())) for box in boxes]

    def predict_batch(self, images: list) -> list:
        """Batch plate detection — returns list of bbox lists per image."""
        if not images:
            return []
        results = self.model(
            images, conf=0.4, imgsz=768, half=self.using_trt,
            verbose=False, device=self.device, stream=False,
        )
        out = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                out.append([])
            else:
                out.append([list(map(int, box.xyxy[0].cpu().numpy())) for box in r.boxes])
        return out

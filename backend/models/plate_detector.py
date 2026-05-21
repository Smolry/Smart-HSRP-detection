"""
GPU-OPTIMIZED PLATE DETECTOR
==============================
YOLOv10s — FP16 on GPU, larger imgsz for small plates.
"""
import torch
from ultralytics import YOLO
from config.settings import settings


class PlateDetector:
    def __init__(self, model_path: str = settings.PLATE_MODEL_PATH):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.half   = False
        self.model  = YOLO(model_path)
        if self.half:
            self.model.model.half()
        print(f"[PlateDetector] {self.device} | FP16={self.half}")

    def predict(self, image):
        results = self.model(
            image, conf=0.4, imgsz=768, half=self.half,
            verbose=False, device=self.device,
        )
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []
        return [list(map(int, box.xyxy[0].cpu().numpy())) for box in boxes]

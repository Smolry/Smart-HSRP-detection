"""
GPU-OPTIMIZED VEHICLE DETECTOR
================================
YOLOv11m with FP16 on GPU, persistent model, no reload per frame.
"""
import torch
from ultralytics import YOLO
from config.settings import settings


class VehicleDetector:
    VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}
    PERSON_CLASS    = "person"
    COCO_IDS        = [0, 2, 3, 5, 7]  # person, car, motorcycle, bus, truck

    def __init__(self, model_path: str = settings.VEHICLE_MODEL_PATH):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.half   = False
        self.conf   = float(settings.VEHICLE_CONF_THRESHOLD)
        self.model  = YOLO(model_path)
        if self.half:
            self.model.model.half()
        print(f"[VehicleDetector] {self.device} | FP16={self.half}")

    def detect(self, frame):
        results = self.model.predict(
            source=frame, conf=self.conf, device=self.device,
            classes=self.COCO_IDS, imgsz=640, half=self.half, verbose=False,
        )
        out = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy.squeeze().cpu().numpy())
                cls_id = int(box.cls.item())
                out.append({
                    "bbox":       [x1, y1, x2, y2],
                    "confidence": float(box.conf.item()),
                    "class":      self.model.names[cls_id],
                })
        return out

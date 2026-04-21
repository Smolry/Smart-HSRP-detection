"""
GPU-OPTIMIZED HELMET DETECTOR
================================
YOLOv8s — FP16 on GPU. Low conf=0.25, temporal fusion decides.
"""
import torch
import numpy as np
from ultralytics import YOLO
from config.settings import settings


class HelmetDetector:
    def __init__(self, model_path: str = settings.HELMET_MODEL_PATH):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.half   = self.device == "cuda"
        self.model  = YOLO(model_path)
        if self.half:
            self.model.model.half()
        print(f"[HelmetDetector] {self.device} | FP16={self.half} | classes={self.model.names}")

    def predict(self, image: np.ndarray) -> dict:
        if image is None or image.size == 0:
            return self._empty()

        results = self.model(image, conf=0.25, half=self.half, verbose=False, device=self.device)
        boxes   = results[0].boxes

        helmet_conf, no_helmet_conf         = 0.0, 0.0
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

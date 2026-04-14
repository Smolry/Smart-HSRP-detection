import torch
import numpy as np
from ultralytics import YOLO
from config.settings import settings


class HelmetDetector:
    def __init__(self, model_path: str = settings.HELMET_MODEL_PATH):
        """
        YOLOv8 Helmet Detection Model

        Expected class mapping from weights:
            0 -> helmet
            1 -> no-helmet
        """
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self._load_model()

        if self.model:
            print(f"HelmetDetector: Model loaded on {self.device}")
            print(f"Class mapping: {self.model.names}")

    def _load_model(self):
        try:
            return YOLO(self.model_path)
        except Exception as e:
            print(f"Error loading helmet model: {e}")
            return None

    def predict(self, image: np.ndarray) -> dict:
        if image is None:
            raise ValueError("Invalid image passed to HelmetDetector")

        if self.model is None:
            raise RuntimeError("Helmet detection model not loaded")

        # 🔥 Lower confidence slightly to allow temporal fusion to decide
        results = self.model(image, conf=0.25, verbose=False)
        boxes = results[0].boxes

        helmet_conf = 0.0
        no_helmet_conf = 0.0
        best_helmet_bbox = None
        best_no_helmet_bbox = None
        detections = []

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls.cpu())
                conf = float(box.conf.cpu())

                raw_name = self.model.names.get(cls_id, "")
                cls_name = raw_name.strip().lower()

                x1, y1, x2, y2 = map(
                    int, box.xyxy.cpu().numpy()[0]
                )

                # ✅ Direct mapping (new model)
                if cls_name == "helmet":
                    interpreted_class = "HELMET"

                    if conf > helmet_conf:
                        helmet_conf = conf
                        best_helmet_bbox = [x1, y1, x2, y2]

                elif cls_name in ["no-helmet", "no_helmet"]:
                    interpreted_class = "NO_HELMET"

                    if conf > no_helmet_conf:
                        no_helmet_conf = conf
                        best_no_helmet_bbox = [x1, y1, x2, y2]

                else:
                    interpreted_class = "UNKNOWN"

                detections.append({
                    "class": interpreted_class,
                    "confidence": round(conf, 4),
                    "bbox": [x1, y1, x2, y2]
                })

        # --------------------------------------------------
        # FINAL DECISION (Simple + Safe)
        # --------------------------------------------------

        if no_helmet_conf > helmet_conf and no_helmet_conf >= 0.4:
            status = "NO_HELMET"
            final_conf = no_helmet_conf
            bbox = best_no_helmet_bbox

        elif helmet_conf >= 0.4:
            status = "HELMET"
            final_conf = helmet_conf
            bbox = best_helmet_bbox

        else:
            status = "UNCERTAIN"
            final_conf = max(helmet_conf, no_helmet_conf)
            bbox = None

        return {
            "status": status,
            "confidence": round(final_conf, 4),
            "bbox": bbox,
            "detections": detections,
            "count": len(detections),
            "device_used": self.device
        }

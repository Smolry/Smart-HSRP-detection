import torch
import numpy as np
from ultralytics import YOLO
from config.settings import settings

class PlateDetector:
    def __init__(self, model_path: str = settings.PLATE_MODEL_PATH):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path).to(self.device)

    def predict(self, image: np.ndarray):
        results = self.model(image, conf=0.4, imgsz=768 , verbose=False, device=self.device)
        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            return []

        bboxes = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            bboxes.append([x1, y1, x2, y2])

        return bboxes

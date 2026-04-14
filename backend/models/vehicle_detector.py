from ultralytics import YOLO
import torch
from config.settings import settings

class VehicleDetector:
    """
    Detects road objects using a single YOLO inference.
    Includes persons and vehicles.
    """

    VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}
    PERSON_CLASS = "person"

    def __init__(self, model_path: str = settings.VEHICLE_MODEL_PATH):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.conf_threshold = float(settings.VEHICLE_CONF_THRESHOLD)
        self.model = self._load_model(model_path)

        if self.model:
            print(f"VehicleDetector: YOLO loaded on {self.device}")

    def _load_model(self, model_path: str):
        try:
            return YOLO(model_path)
        except Exception as e:
            print(f"VehicleDetector Error loading model: {e}")
            return None

    def detect(self, frame):
        """
        Returns all relevant detections in the frame.
        """

        if self.model is None:
            return []

        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            device=self.device,
            classes=[0, 2, 3, 5, 7],  # person + vehicles
            imgsz=640,
            verbose=False
        )

        detections = []

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy.squeeze().cpu().numpy()
                )

                cls_id = int(box.cls.item())
                cls_name = self.model.names[cls_id]

                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(box.conf.item()),
                    "class": cls_name
                })

        return detections

    # 👇 convenience helpers (important)
    def split_detections(self, detections):
        vehicles = []
        persons = []

        for d in detections:
            if d["class"] in self.VEHICLE_CLASSES:
                vehicles.append(d)
            elif d["class"] == self.PERSON_CLASS:
                persons.append(d)

        return vehicles, persons

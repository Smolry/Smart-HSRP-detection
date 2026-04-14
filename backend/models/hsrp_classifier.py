import torch
import cv2
import numpy as np
from config.settings import settings


class HSRPClassifier:
    """
    EfficientNet-based binary classifier.
    Output convention:
        sigmoid(logit) → P(non_hsrp)
        0 = HSRP
        1 = Non-HSRP
    """

    def __init__(self, model_path: str = settings.HSRP_MODEL_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.decision_threshold = float(settings.HSRP_CONF_THRESHOLD)
        self.model = self._load_model(model_path)

        if self.model:
            print(f"HSRPClassifier loaded on {self.device}")

    def _load_model(self, model_path):
        try:
            model = torch.jit.load(model_path, map_location=self.device)
            model.eval()
            return model
        except Exception as e:
            print(f"[HSRPClassifier] Model load failed: {e}")
            return None

    # -------------------------------------------------
    # Preprocessing (ImageNet-compatible)
    # -------------------------------------------------
    def preprocess(self, img: np.ndarray) -> torch.Tensor:
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std

        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0)

        return img.to(self.device)

    # -------------------------------------------------
    # Prediction
    # -------------------------------------------------
    def predict(self, plate_image: np.ndarray) -> dict:
        if plate_image is None or plate_image.size == 0:
            return self._empty_result()

        if self.model is None:
            return self._empty_result()

        input_tensor = self.preprocess(plate_image)

        with torch.no_grad():
            logit = self.model(input_tensor).squeeze()
            prob_non_hsrp = torch.sigmoid(logit).item()

        prob_hsrp = 1.0 - prob_non_hsrp

        # Decision logic (clear + symmetric)
        if prob_non_hsrp >= self.decision_threshold:
            label = "non_hsrp"
            confidence = prob_non_hsrp
        else:
            label = "hsrp"
            confidence = prob_hsrp

        return {
            "label": label,
            "confidence": round(float(confidence), 4),
            "prob_non_hsrp": round(float(prob_non_hsrp), 4),
            "prob_hsrp": round(float(prob_hsrp), 4),
            "device_used": str(self.device)
        }

    # -------------------------------------------------
    # Safe fallback
    # -------------------------------------------------
    def _empty_result(self):
        return {
            "label": None,
            "confidence": 0.0,
            "prob_non_hsrp": 0.0,
            "prob_hsrp": 0.0,
            "device_used": str(self.device)
        }

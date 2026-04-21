"""
GPU-OPTIMIZED HSRP CLASSIFIER
================================
EfficientNet-B0 TorchScript — FP16 on GPU.
Normalisation tensors pinned on GPU to avoid per-frame CPU-GPU copy.
"""
import torch
import cv2
import numpy as np
from config.settings import settings


class HSRPClassifier:
    """sigmoid(logit) → P(non_hsrp). 0=HSRP, 1=Non-HSRP."""

    _MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    _STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __init__(self, model_path: str = settings.HSRP_MODEL_PATH):
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = float(settings.HSRP_CONF_THRESHOLD)
        self.model     = self._load(model_path)
        dtype          = torch.float16 if self.device.type == "cuda" else torch.float32
        self._mean     = self._MEAN.to(self.device, dtype=dtype)
        self._std      = self._STD.to(self.device, dtype=dtype)
        print(f"[HSRPClassifier] {self.device}")

    def _load(self, path):
        try:
            m = torch.jit.load(path, map_location=self.device)
            m.eval()
            if self.device.type == "cuda":
                m = m.half()
            return m
        except Exception as e:
            print(f"[HSRPClassifier] load failed: {e}")
            return None

    def preprocess(self, img: np.ndarray) -> torch.Tensor:
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t   = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        t   = t.to(self.device)
        if self.device.type == "cuda":
            t = t.half()
        return (t - self._mean) / self._std

    def predict(self, plate_image: np.ndarray) -> dict:
        if plate_image is None or plate_image.size == 0 or self.model is None:
            return self._empty()
        x = self.preprocess(plate_image)
        with torch.no_grad():
            prob_non_hsrp = torch.sigmoid(self.model(x).squeeze()).float().item()
        prob_hsrp = 1.0 - prob_non_hsrp
        if prob_non_hsrp >= self.threshold:
            label, conf = "non_hsrp", prob_non_hsrp
        else:
            label, conf = "hsrp", prob_hsrp
        return {
            "label":         label,
            "confidence":    round(conf, 4),
            "prob_non_hsrp": round(prob_non_hsrp, 4),
            "prob_hsrp":     round(prob_hsrp, 4),
        }

    def _empty(self):
        return {"label": None, "confidence": 0.0, "prob_non_hsrp": 0.0, "prob_hsrp": 0.0}

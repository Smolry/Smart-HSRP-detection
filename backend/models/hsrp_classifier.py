"""
TENSORRT-OPTIMIZED HSRP CLASSIFIER
=====================================
EfficientNet-B0 TorchScript → TensorRT via torch2trt.
Falls back to original TorchScript on CPU or if torch2trt is not installed.

Export once manually:
    python -c "
    from backend.models.hsrp_classifier import HSRPClassifier
    c = HSRPClassifier()
    c.export_trt()
    "

Requires: pip install torch2trt
"""
import os
import torch
import cv2
import numpy as np
from pathlib import Path
from config.settings import settings


def _trt_path(pt_path: str) -> str:
    return str(Path(pt_path).with_stem(Path(pt_path).stem + "_trt").with_suffix(".pth"))


class HSRPClassifier:
    """sigmoid(logit) → P(non_hsrp). 0=HSRP, 1=Non-HSRP."""

    _MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    _STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __init__(self, model_path: str = settings.HSRP_MODEL_PATH):
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = float(settings.HSRP_CONF_THRESHOLD)
        self.using_trt = False
        dtype          = torch.float16 if self.device.type == "cuda" else torch.float32

        self.model = self._load(model_path)

        self._mean = self._MEAN.to(self.device, dtype=dtype)
        self._std  = self._STD.to(self.device, dtype=dtype)
        print(f"[HSRPClassifier] {self.device} | TRT={self.using_trt}")

    def _load(self, path: str):
        """Try TRT first, then TorchScript."""
        trt_path = _trt_path(path)

        # Try loading pre-exported TRT model
        if self.device.type == "cuda" and os.path.exists(trt_path):
            try:
                from torch2trt import TRTModule
                m = TRTModule()
                m.load_state_dict(torch.load(trt_path))
                m.eval()
                self.using_trt = True
                print(f"[HSRPClassifier] TRT module loaded from {trt_path}")
                return m
            except Exception as e:
                print(f"[HSRPClassifier] TRT load failed ({e}), trying auto-export...")

        # Try auto-exporting to TRT
        if self.device.type == "cuda" and os.path.exists(path):
            try:
                base_model = self._load_torchscript(path)
                if base_model is not None:
                    trt_model = self._export_trt(base_model, trt_path)
                    if trt_model is not None:
                        return trt_model
            except Exception as e:
                print(f"[HSRPClassifier] TRT auto-export failed ({e})")

        # Fallback to TorchScript
        return self._load_torchscript(path)

    def _load_torchscript(self, path: str):
        try:
            m = torch.jit.load(path, map_location=self.device)
            m.eval()
            if self.device.type == "cuda":
                m = m.half()
            return m
        except Exception as e:
            print(f"[HSRPClassifier] TorchScript load failed: {e}")
            return None

    def _export_trt(self, base_model, trt_path: str):
        """Export TorchScript model to TRT and save."""
        try:
            from torch2trt import torch2trt
            print(f"[HSRPClassifier] Exporting TRT → {trt_path}")
            dummy = torch.randn(1, 3, 224, 224).to(self.device).half()
            trt_model = torch2trt(
                base_model, [dummy],
                fp16_mode=True,
                max_batch_size=16,
                use_onnx=True,
            )
            torch.save(trt_model.state_dict(), trt_path)
            self.using_trt = True
            print(f"[HSRPClassifier] TRT export complete")
            return trt_model
        except Exception as e:
            print(f"[HSRPClassifier] torch2trt export failed: {e}")
            return None

    def export_trt(self):
        """Public method to manually trigger TRT export."""
        path = settings.HSRP_MODEL_PATH
        base = self._load_torchscript(path)
        if base:
            self._export_trt(base, _trt_path(path))

    def preprocess(self, img: np.ndarray) -> torch.Tensor:
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t   = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        t   = t.to(self.device)
        if self.device.type == "cuda":
            t = t.half()
        return (t - self._mean) / self._std

    def preprocess_batch(self, imgs: list) -> torch.Tensor:
        tensors = []
        for img in imgs:
            img = cv2.resize(img, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            t   = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            tensors.append(t)
        batch = torch.stack(tensors).to(self.device)
        if self.device.type == "cuda":
            batch = batch.half()
        return (batch - self._mean) / self._std

    def predict(self, plate_image: np.ndarray) -> dict:
        if plate_image is None or plate_image.size == 0 or self.model is None:
            return self._empty()
        x = self.preprocess(plate_image)
        with torch.no_grad():
            prob_non_hsrp = torch.sigmoid(self.model(x).squeeze()).float().item()
        return self._result(prob_non_hsrp)

    def predict_batch(self, plate_images: list) -> list:
        """Batch HSRP classification."""
        valid = [(i, img) for i, img in enumerate(plate_images)
                 if img is not None and img.size > 0]
        if not valid or self.model is None:
            return [self._empty() for _ in plate_images]

        indices, imgs = zip(*valid)
        x = self.preprocess_batch(list(imgs))
        with torch.no_grad():
            probs = torch.sigmoid(self.model(x).squeeze(1)).float().tolist()
        if isinstance(probs, float):
            probs = [probs]

        parsed = {idx: self._result(p) for idx, p in zip(indices, probs)}
        return [parsed.get(i, self._empty()) for i in range(len(plate_images))]

    def _result(self, prob_non_hsrp: float) -> dict:
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

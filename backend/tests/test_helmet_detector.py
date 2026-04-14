import cv2
import numpy as np
from pathlib import Path
from backend.core.model_registry import helmet_detector

BASE_DIR = Path(__file__).resolve().parents[2]
IMAGE_PATH = BASE_DIR / "test_images" / "test3.jpg"

def main():
    img = cv2.imread(str(IMAGE_PATH))
    if img is None:
        raise RuntimeError(f"Image not found: {IMAGE_PATH}")

    # --------------------------------------------------
    # 🔍 DEBUG: Inspect Raw Image Before Inference
    # --------------------------------------------------
    print("\nINPUT IMAGE DEBUG")
    print("=" * 50)
    print("Shape        :", img.shape)
    print("Dtype        :", img.dtype)
    print("Min pixel    :", np.min(img))
    print("Max pixel    :", np.max(img))

    # Optional: test manual resize (uncomment to compare)
    img = cv2.resize(img, (640, 640))
    # print("\nAfter Manual Resize:")
    # print("Shape        :", img.shape)

    # --------------------------------------------------
    # Run Inference
    # --------------------------------------------------
    res = helmet_detector.predict(img)

    print("\nHELMET DETECTOR OUTPUT")
    print("=" * 50)
    for k, v in res.items():
        print(f"{k:20}: {v}")

if __name__ == "__main__":
    main()

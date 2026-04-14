import cv2
from pathlib import Path
from backend.core.model_registry import hsrp_model

BASE_DIR = Path(__file__).resolve().parents[2]
PLATE_PATH = BASE_DIR / "test_images" / "day_hsrp_0.jpg"

def main():
    img = cv2.imread(str(PLATE_PATH))
    if img is None:
        raise RuntimeError(f"Image not found: {PLATE_PATH}")

    res = hsrp_model.predict(img)

    print("\nHSRP CLASSIFIER OUTPUT")
    print("=" * 50)
    for k, v in res.items():
        print(f"{k:20}: {v}")

if __name__ == "__main__":
    main()

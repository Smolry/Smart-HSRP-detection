import cv2
from pathlib import Path
from backend.core.pipeline import run_pipeline

BASE_DIR = Path(__file__).resolve().parents[2]
IMAGE_PATH = BASE_DIR / "test_images" / "test0.jpg"

def main():
    img = cv2.imread(str(IMAGE_PATH))   # <-- CRITICAL FIX

    if img is None:
        raise RuntimeError(f"Failed to load image: {IMAGE_PATH}")

    print("Running pipeline...\n")
    result = run_pipeline(img)

    print("PIPELINE OUTPUT")
    print("=" * 50)
    for k, v in result.items():
        print(f"{k:20}: {v}")

if __name__ == "__main__":
    main()

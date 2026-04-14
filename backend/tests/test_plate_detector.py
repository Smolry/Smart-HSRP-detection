import cv2
from pathlib import Path
from backend.core.model_registry import plate_model

BASE_DIR = Path(__file__).resolve().parents[2]
IMAGE_PATH = BASE_DIR / "test_images" / "test0.jpg"


def main():
    img = cv2.imread(str(IMAGE_PATH))
    if img is None:
        raise RuntimeError("Image not loaded")

    plates = plate_model.predict(img)

    print("\nRAW plate_model.predict OUTPUT")
    print("=" * 60)

    print("Type of output:", type(plates))
    print("Length:", len(plates) if plates is not None else "None")

    if not plates:
        print("⚠️ No plates detected")
        return

    print("\nFirst element:")
    print("Value:", plates[0])
    print("Type:", type(plates[0]))

    # Deep inspection
    if hasattr(plates[0], "__len__"):
        print("Length of first element:", len(plates[0]))
        for i, v in enumerate(plates[0]):
            print(f"  [{i}] value={v}, type={type(v)}")

    print("\nFULL OUTPUT:")
    for i, p in enumerate(plates):
        print(f"[{i}] {p}  (type={type(p)})")


if __name__ == "__main__":
    main()

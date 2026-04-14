import cv2
from pathlib import Path
from ultralytics import YOLO
from config.settings import settings


BASE_DIR = Path(__file__).resolve().parents[2]
IMAGE_PATH = BASE_DIR / "test_images" / "test14.jpg"


def main():
    print("\nLOADING MODEL WEIGHTS")
    print("=" * 60)

    model_path = settings.HELMET_MODEL_PATH
    print("Model path:", model_path)

    model = YOLO(model_path)

    print("\nModel loaded successfully.")
    print("Class mapping from weights:")
    print(model.names)

    print("\nLOADING TEST IMAGE")
    print("=" * 60)

    img = cv2.imread(str(IMAGE_PATH))
    img = cv2.resize(img, (640, 640))
    if img is None:
        raise RuntimeError(f"Image not found: {IMAGE_PATH}")

    print("Image shape:", img.shape)

    print("\nRUNNING RAW INFERENCE (NO POST-PROCESSING)")
    print("=" * 60)

    results = model(img, conf=0.01, iou=0.5, verbose=False)
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        print("❌ No detections found")
        return

    print(f"Total detections: {len(boxes)}\n")

    class_conf_map = {}

    for i, box in enumerate(boxes):
        cls_id = int(box.cls.cpu())
        conf = float(box.conf.cpu())
        raw_name = model.names.get(cls_id, "UNKNOWN")

        print(f"[Detection {i+1}]")
        print("  Raw class name :", raw_name)
        print("  Confidence     :", round(conf, 4))
        print("  Class ID       :", cls_id)
        print("-" * 40)

        if raw_name not in class_conf_map:
            class_conf_map[raw_name] = []

        class_conf_map[raw_name].append(conf)

    print("\nCONFIDENCE SUMMARY")
    print("=" * 60)

    for cls_name, confs in class_conf_map.items():
        print(f"{cls_name}")
        print("  Count     :", len(confs))
        print("  Max conf  :", round(max(confs), 4))
        print("  Avg conf  :", round(sum(confs) / len(confs), 4))
        print("-" * 40)


if __name__ == "__main__":
    main()

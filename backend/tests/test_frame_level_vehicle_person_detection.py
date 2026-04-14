import json
import cv2
from pathlib import Path

from backend.services.video_reader import read_video
from backend.models.vehicle_detector import VehicleDetector

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
VIDEO_PATH = Path("test_videos/test_multiple.mp4")
OUTPUT_DIR = Path("video_person_vehicle_debug")
MAX_FRAMES = 150


def save_crop(img, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def test_vehicle_and_person_detection():
    # -------------------------------------------------
    # FAIL FAST
    # -------------------------------------------------
    assert VIDEO_PATH.exists(), f"❌ Video not found: {VIDEO_PATH.resolve()}"
    print(f"🎥 Processing video: {VIDEO_PATH.resolve()}")

    detector = VehicleDetector()

    dump = {
        "video_path": str(VIDEO_PATH),
        "frames": []
    }

    frame_count = 0
    total_persons = 0
    total_vehicles = 0

    for frame_id, frame in read_video(str(VIDEO_PATH)):
        frame_count += 1
        if MAX_FRAMES and frame_count > MAX_FRAMES:
            break

        detections = detector.detect(frame) 
        vehicles, persons = detector.split_detections(detections)

        total_persons += len(persons)
        total_vehicles += len(vehicles)

        frame_entry = {
            "frame_id": frame_id,
            "vehicles": [],
            "persons": []
        }

        # -----------------------------
        # Vehicles
        # -----------------------------
        for idx, v in enumerate(vehicles):
            x1, y1, x2, y2 = v["bbox"]
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            crop_path = (
                OUTPUT_DIR
                / "crops"
                / "vehicles"
                / f"frame_{frame_id:06d}_veh_{idx:02d}.jpg"
            )

            save_crop(crop, crop_path)

            frame_entry["vehicles"].append({
                "bbox": v["bbox"],
                "confidence": v["confidence"],
                "class": v["class"],
                "crop_path": str(crop_path)
            })

        # -----------------------------
        # Persons
        # -----------------------------
        for idx, p in enumerate(persons):
            x1, y1, x2, y2 = p["bbox"]
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            crop_path = (
                OUTPUT_DIR
                / "crops"
                / "persons"
                / f"frame_{frame_id:06d}_person_{idx:02d}.jpg"
            )

            save_crop(crop, crop_path)

            frame_entry["persons"].append({
                "bbox": p["bbox"],
                "confidence": p["confidence"],
                "class": p["class"],
                "crop_path": str(crop_path)
            })

        dump["frames"].append(frame_entry)

        if frame_count % 10 == 0:
            print(f"⏳ Processed frame {frame_count}")

    # -------------------------------------------------
    # SAVE JSON
    # -------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "vehicle_person_detection_dump.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dump, f, indent=2)

    # -------------------------------------------------
    # ASSERTIONS (important)
    # -------------------------------------------------
    assert frame_count > 0, "❌ No frames processed"
    assert total_vehicles > 0, "❌ No vehicles detected"
    assert total_persons > 0, "❌ No persons detected"

    print("\n✅ VEHICLE + PERSON DETECTION TEST COMPLETE")
    print(f"Frames processed : {frame_count}")
    print(f"Vehicles found   : {total_vehicles}")
    print(f"Persons found    : {total_persons}")
    print(f"JSON dump        : {json_path.resolve()}")
    print(f"Crops folder     : {(OUTPUT_DIR / 'crops').resolve()}")

import json
import cv2
from pathlib import Path

from backend.services.video_reader import read_video
from backend.models.vehicle_detector import VehicleDetector
from backend.services.vehicle_tracker import VehicleTracker
from backend.services.cropper import crop_rois
from backend.core.pipeline import run_pipeline


VIDEO_PATH = Path("test_videos/test_multiple.mp4")
OUTPUT_DIR = Path("video_debug_output")
MAX_FRAMES = 150  # set limit if needed


def save_crop(img, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def test_video_json_and_bbox_dump():
    # -------------------------------------------------
    # FAIL FAST CHECKS (IMPORTANT)
    # -------------------------------------------------
    assert VIDEO_PATH.exists(), f"❌ Video not found: {VIDEO_PATH.resolve()}"
    print(f"🎥 Processing video: {VIDEO_PATH.resolve()}")

    vehicle_detector = VehicleDetector()
    vehicle_tracker = VehicleTracker()

    dump = {
        "video_path": str(VIDEO_PATH),
        "frames": []
    }

    frame_counter = 0
    total_detections = 0
    total_tracks = 0
    total_pipeline_runs = 0

    for frame_id, frame in read_video(str(VIDEO_PATH)):
        frame_counter += 1
        if MAX_FRAMES and frame_counter > MAX_FRAMES:
            break

        if frame_counter % 10 == 0:
            print(f"⏳ Processed frame {frame_counter}")

        frame_entry = {
            "frame_id": frame_id,
            "vehicles": []
        }

        # -----------------------------
        # 1. Vehicle Detection
        # -----------------------------
        detections = vehicle_detector.detect(frame)

        if detections:
            total_detections += len(detections)
        else:
            dump["frames"].append(frame_entry)
            continue

        # -----------------------------
        # 2. Vehicle Tracking
        # -----------------------------
        tracks = vehicle_tracker.update(detections, frame)
        if not tracks:
            dump["frames"].append(frame_entry)
            continue

        total_tracks += len(tracks)

        # -----------------------------
        # 3. Per-track processing
        # -----------------------------
        for track in tracks:
            track_id = track["track_id"]
            bbox = track["bbox"]  # xywh

            x, y, w, h = bbox
            if w <= 0 or h <= 0:
                continue

            # -----------------------------
            # 3a. Vehicle crop
            # -----------------------------
            vehicle_crops = crop_rois(frame, [bbox])
            if not vehicle_crops:
                continue

            vehicle_crop = vehicle_crops[0]

            vehicle_crop_path = (
                OUTPUT_DIR
                / "crops"
                / "vehicles"
                / f"frame_{frame_id:06d}_track_{track_id}_vehicle.jpg"
            )


            save_crop(vehicle_crop, vehicle_crop_path)

            # -----------------------------
            # 3b. Run image pipeline
            # -----------------------------
            pipeline_output = run_pipeline(
                vehicle_crop,
                image_path=f"{VIDEO_PATH}:frame_{frame_id}:track_{track_id}",
                force_save=False
            )

            total_pipeline_runs += 1

            vehicle_entry = {
                "track_id": track_id,
                "vehicle_bbox": {
                    "format": "xywh",
                    "values": bbox
                },
                "vehicle_crop_path": str(vehicle_crop_path),
                "helmet": pipeline_output["event"],
                "plates": []
            }

            # -----------------------------
            # 3c. Plate crops
            # -----------------------------
            for idx, plate in enumerate(pipeline_output.get("plates", [])):
                plate_bbox = plate["bbox"]  # xyxy

                px1, py1, px2, py2 = plate_bbox
                plate_crop = vehicle_crop[py1:py2, px1:px2]

                if plate_crop.size == 0:
                    continue

                plate_crop_path = (
                    OUTPUT_DIR
                    / "crops"
                    / "plates"
                    / f"frame_{frame_id:06d}_track_{track_id}_plate_{idx:02d}.jpg"
                )


                save_crop(plate_crop, plate_crop_path)

                vehicle_entry["plates"].append({
                    "bbox": {
                        "format": "xyxy",
                        "values": plate_bbox
                    },
                    "plate_crop_path": str(plate_crop_path),
                    "hsrp": {
                        "is_hsrp": plate["is_hsrp"],
                        "confidence": plate["hsrp_confidence"],
                        "violation": plate["hsrp_violation"]
                    },
                    "ocr": {
                        "text": plate.get("ocr_text"),
                        "confidence": plate.get("ocr_confidence")
                    }
                })

            frame_entry["vehicles"].append(vehicle_entry)

        dump["frames"].append(frame_entry)

    # -------------------------------------------------
    # POST-RUN VALIDATION (NO SILENT SUCCESS)
    # -------------------------------------------------
    assert frame_counter > 0, (
        f"❌ Video opened but no frames were read: {VIDEO_PATH}"
    )

    assert total_detections > 0, (
        "❌ No vehicles detected in entire video"
    )

    assert total_pipeline_runs > 0, (
        "❌ Image pipeline never ran on any vehicle"
    )

    # -----------------------------
    # Save JSON
    # -----------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "video_debug_dump.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dump, f, indent=2)

    print("\n✅ FULL VIDEO DEBUG EXPORT COMPLETE")
    print(f"Frames processed      : {frame_counter}")
    print(f"Total detections      : {total_detections}")
    print(f"Total tracks          : {total_tracks}")
    print(f"Pipeline executions   : {total_pipeline_runs}")
    print(f"JSON dump             : {json_path.resolve()}")
    print(f"Crops folder          : {(OUTPUT_DIR / 'crops').resolve()}")

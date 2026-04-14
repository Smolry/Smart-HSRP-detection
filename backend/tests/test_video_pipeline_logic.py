import numpy as np
import pytest
from backend.core.video_pipeline import process_video

@pytest.fixture
def mock_frame():
    """Generates a blank RGB frame for testing."""
    return np.zeros((480, 640, 3), dtype=np.uint8)

def test_video_pipeline_emits_single_event(monkeypatch, mock_frame):
    # -----------------------
    # 1. Mock Video Reader (15 Frames)
    # -----------------------
    def fake_read_video(path):
        for i in range(15):
            yield i, mock_frame

    monkeypatch.setattr("backend.core.video_pipeline.read_video", fake_read_video)

    # -----------------------
    # 2. Mock FPS Controller (Always Process)
    # -----------------------
    # Prevents frames from being skipped due to clock-time logic
    monkeypatch.setattr(
        "backend.core.video_pipeline.TimeBasedFPSController.should_process",
        lambda self: True
    )

    # -----------------------
    # 3. Mock Vehicle Detector
    # -----------------------
    monkeypatch.setattr(
        "backend.core.video_pipeline.VehicleDetector.detect",
        lambda self, frame: [{"bbox": [50, 50, 100, 100]}]
    )

    # -----------------------
    # 4. Mock Vehicle Tracker
    # -----------------------
    monkeypatch.setattr(
        "backend.core.video_pipeline.VehicleTracker.update",
        lambda self, detections, frame: [{
            "track_id": 1,
            "bbox": [50, 50, 100, 100]
        }]
    )

    # -----------------------
    # 5. Mock Image Pipeline (Raw AI Output)
    # -----------------------
    def fake_run_pipeline(image, image_path=None, force_save=False):
        return {
            "vehicle_type": "motorcycle",
            "event": {
                "helmet_detected": False, # Simulate a violation
                "helmet_confidence": 0.95
            },
            "plates": [{
                "ocr_text": "MH12AB1234",
                "ocr_confidence": 0.92,
                "hsrp_violation": False,
                "bbox": [10, 10, 50, 20]
            }]
        }

    monkeypatch.setattr("backend.core.video_pipeline.run_pipeline", fake_run_pipeline)

    # -----------------------
    # 6. Mock Track Refiner (FORCED SUCCESS)
    # -----------------------
    # This ensures the refiner returns data immediately instead of waiting for 10 frames
    def mock_refine(self, track_id):
        return {
            "track_id": track_id,
            "vehicle_type": "motorcycle",
            "plate": {"text": "MH12AB1234", "confidence": 0.92},
            "helmet_violation": True,
            "hsrp_violation": False,
            "frames": {"seen": 1, "first": 0, "last": 0}
        }

    monkeypatch.setattr("backend.core.video_pipeline.TrackRefiner.refine", mock_refine)

    # -----------------------
    # 7. Mock Event Emitter
    # -----------------------
    def mock_emit(self, track_id, refined):
        return {
            "track_id": track_id,
            "plate": refined["plate"]["text"],
            "violations": ["NO_HELMET"] if refined["helmet_violation"] else []
        }

    monkeypatch.setattr("backend.core.video_pipeline.EventEmitter.emit", mock_emit)

    # -----------------------
    # 8. Run Pipeline Execution
    # -----------------------
    output = process_video("dummy_path.mp4", target_fps=10)

    print("\nPIPELINE OUTPUT:")
    print(output)

    # -----------------------
    # 9. Assertions
    # -----------------------
    # Verify exactly one final event is produced (optimization logic works)
    assert len(output["final_events"]) == 1
    
    event = output["final_events"][0]
    assert event["track_id"] == 1
    assert event["plate"] == "MH12AB1234"
    assert "NO_HELMET" in event["violations"]
    
    

# applied Simulation Testing for Know 

import numpy as np
import pytest
import cv2
from backend.services.vehicle_tracker import VehicleTracker

def test_vehicle_tracker_basic():
    print("\n=== VehicleTracker TEST ===")
    
    # Initialize tracker with n_init=1 to confirm tracks as quickly as possible
    tracker = VehicleTracker(max_age=5, n_init=1)
    
    # --- FRAME 1: INITIALIZATION ---
    frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
    detections1 = [
        {"bbox": [100, 50, 200, 150], "confidence": 0.9, "class": "car"},
        {"bbox": [300, 100, 400, 200], "confidence": 0.8, "class": "bus"},
    ]
    
    # Fill bounding boxes with random noise so the Re-ID embedder has features to extract
    for det in detections1:
        x1, y1, x2, y2 = det["bbox"]
        frame1[y1:y2, x1:x2] = np.random.randint(0, 255, (y2-y1, x2-x1, 3), dtype=np.uint8)
    
    # First update: DeepSORT creates the tracks but keeps them "Tentative"
    _ = tracker.update(detections1, frame1)
    
    # --- FRAME 2: CONFIRMATION ---
    # Objects move slightly
    frame2 = frame1.copy()
    detections2 = [
        {"bbox": [105, 55, 205, 155], "confidence": 0.9, "class": "car"},
        {"bbox": [305, 105, 405, 205], "confidence": 0.8, "class": "bus"},
    ]
    
    # Update again: DeepSORT now "confirms" the tracks because they appeared in > n_init frames
    tracks2 = tracker.update(detections2, frame2)
    print("Frame 2 Confirmed tracks:", tracks2)
    
    # --- ASSERTIONS ---
    assert len(tracks2) == 2, f"Expected 2 confirmed tracks, got {len(tracks2)}"
    
    for t in tracks2:
        assert "track_id" in t
        assert "bbox" in t
        assert "class" in t
        assert isinstance(t["track_id"], (int, str))
        assert len(t["bbox"]) == 4

    # --- FRAME 3: CONSISTENCY CHECK ---
    # Move objects again to verify ID persistence
    detections3 = [
        {"bbox": [110, 60, 210, 160], "confidence": 0.9, "class": "car"},
        {"bbox": [310, 110, 410, 210], "confidence": 0.8, "class": "bus"},
    ]
    tracks3 = tracker.update(detections3, frame2)
    
    # Map IDs to classes for easier comparison
    id_map_f2 = {t["class"]: t["track_id"] for t in tracks2}
    id_map_f3 = {t["class"]: t["track_id"] for t in tracks3}
    
    assert id_map_f2["car"] == id_map_f3["car"], "Car ID changed across frames!"
    assert id_map_f2["bus"] == id_map_f3["bus"], "Bus ID changed across frames!"
    print(f"Verified Persistence: Car ID {id_map_f2['car']}, Bus ID {id_map_f2['bus']}")


def test_vehicle_tracker_empty_and_invalid():
    """Verify that the tracker handles empty lists and invalid geometry without crashing."""
    tracker = VehicleTracker()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Test 1: Empty detections
    tracks = tracker.update([], frame)
    assert tracks == [], "Expected empty list for zero detections"

    # Test 2: Invalid bbox (width is 0)
    invalid_w = [{"bbox": [10, 10, 10, 20], "confidence": 0.9, "class": "car"}]
    tracks = tracker.update(invalid_w, frame)
    assert tracks == [], "Expected no tracks for invalid zero-width bbox"

    # Test 3: Invalid bbox (height is 0)
    invalid_h = [{"bbox": [10, 10, 20, 10], "confidence": 0.9, "class": "car"}]
    tracks = tracker.update(invalid_h, frame)
    assert tracks == [], "Expected no tracks for invalid zero-height bbox"

    print("Edge case tests passed.")

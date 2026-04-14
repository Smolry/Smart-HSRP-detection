# TO RUn these tests, use:
# pytest -s backend/tests/test_vehicle_detector.py
import pytest
import numpy as np
import cv2
from pathlib import Path
from backend.models.vehicle_detector import VehicleDetector
from config.settings import settings

BASE_DIR = Path(__file__).resolve().parents[2]
IMG_PATH = BASE_DIR / "test_images" / "test1.jpg"

@pytest.fixture
def detector():
    """Initialize the detector using settings."""
    return VehicleDetector(model_path=settings.VEHICLE_MODEL_PATH)

def test_initialization(detector):
    """Verify detector loads and assigns a device."""
    assert detector.model is not None
    assert detector.device in ["cuda", "cpu"]

def test_detect_real_image(detector):
    """Test detection on an actual image file."""
    #  Load the image using OpenCV (standard for 2026 workflows)
    frame = cv2.imread(str(IMG_PATH))
    assert frame is not None, "Failed to load test image"

    #  Run detection
    results = detector.detect(frame)
    # Print the raw output for inspection
    print("\n=== VehicleDetector Output ===")
    for det in results:
        print(det)
    print("=== End of Output ===\n")
    
    #  Assertions
    assert isinstance(results, list)
    # Even if 0 vehicles are found, the structure must be a list
    for det in results:
        assert all(key in det for key in ["bbox", "confidence", "class"])
        assert len(det["bbox"]) == 4


def test_detect_output_format(detector):
    """Ensure the output is a list of dictionaries with correct keys."""
    blank_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    results = detector.detect(blank_frame)
    
    assert isinstance(results, list)
    for det in results:
        assert all(key in det for key in ["bbox", "confidence", "class"])
        assert len(det["bbox"]) == 4

def test_invalid_model_path():
    """Verify handling of non-existent model files."""
    detector = VehicleDetector(model_path="invalid_path.pt")
    assert detector.model is None
    assert detector.detect(np.zeros((640, 640, 3))) == []

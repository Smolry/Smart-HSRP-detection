import torch
import cv2
from pathlib import Path
from torchvision import transforms
from PIL import Image

# -------------------------------
# Paths
# -------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "weights" / "hsrp_cls.pt"
IMAGE_PATH = BASE_DIR / "test_images" / "day_hsrp_0.jpg"

# -------------------------------
# Device
# -------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# Preprocessing (MUST match training)
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def main():
    print(f"Loading model from: {MODEL_PATH}")
    print(f"Using device: {DEVICE}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    # -------------------------------
    # Load TorchScript model
    # -------------------------------
    model = torch.jit.load(str(MODEL_PATH), map_location=DEVICE)
    model.eval()

    # -------------------------------
    # Load & preprocess image
    # -------------------------------
    image_bgr = cv2.imread(str(IMAGE_PATH))
    if image_bgr is None:
        raise RuntimeError(f"Failed to read image: {IMAGE_PATH}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image_rgb)

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # -------------------------------
    # Inference
    # -------------------------------
    with torch.no_grad():
        logits = model(input_tensor)
        prob = torch.sigmoid(logits).item()

    prediction = "HSRP" if prob >= 0.6 else "NON_HSRP"

    # -------------------------------
    # Output
    # -------------------------------
    print("\nHSRP CLASSIFIER (WEIGHTS ONLY) OUTPUT")
    print("=" * 50)
    print(f"Prediction         : {prediction}")
    print(f"HSRP Probability   : {prob:.4f}")
    print(f"Threshold          : 0.60")
    print(f"Model File         : {MODEL_PATH.name}")
    print(f"Test Image         : {IMAGE_PATH.name}")

if __name__ == "__main__":
    main()

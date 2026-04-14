import cv2
from pathlib import Path
from paddleocr import PaddleOCR

# ---------- CONFIG ----------
BASE_DIR = Path(__file__).resolve().parents[2]
IMAGE_PATH = BASE_DIR / "test_images" / "day_hsrp_0.jpg"  # <-- put your plate crop here
# ----------------------------

def main():
    # Load image
    img = cv2.imread(str(IMAGE_PATH))

    if img is None:
        print("❌ Failed to load image:", IMAGE_PATH)
        return

    print("Image shape:", img.shape)

    # Optional but recommended preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Initialize OCR
    ocr = PaddleOCR(
        lang="en",
        use_angle_cls=True
    )

    # Run OCR
    results = ocr.predict(img)

    print("\nRAW OCR OUTPUT")
    print("====================================")
    print(results)

    # Parse results
    if not results or results[0] is None:
        print("\n❌ No text detected")
        return

    result = results[0]
    texts = result.get("rec_texts", [])
    scores = result.get("rec_scores", [])
    
    print("\nPARSED TEXT")
    print("====================================")
    
    for t, s in zip(texts, scores):
        print(f"Text: {t} | Confidence: {s:.4f}")



if __name__ == "__main__":
    main()

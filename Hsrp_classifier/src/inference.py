import argparse

import torch
import torch.nn as nn
from PIL import Image

from .model import build_model
from .preprocessing import eval_transforms


def predict(image_path, checkpoint, device=None, threshold=0.5):
    """
    Predict HSRP/non-HSRP using the trained EfficientNet-B0 checkpoint.

    ImageFolder mapping from the training notebook:
        hsrp -> 0
        non-hsrp -> 1

    Therefore sigmoid(logit) is the probability of class 1 (NON_HSRP).
    """
    device = torch.device(
        device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model = model.to(device)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    tensor = eval_transforms()(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logit = model(tensor)
        class1_prob = torch.sigmoid(logit).item()

    # Faithful to the ImageFolder class mapping:
    # class 0 = HSRP, class 1 = NON_HSRP.
    predicted = "NON_HSRP" if class1_prob > threshold else "HSRP"

    return {
        "prediction": predicted,
        "hsrp_probability": 1.0 - class1_prob,
        "non_hsrp_probability": class1_prob,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    result = predict(
        args.image,
        args.checkpoint,
        threshold=args.threshold,
    )

    print(f"Prediction        : {result['prediction']}")
    print(f"HSRP Probability  : {result['hsrp_probability']:.4f}")
    print(f"NON-HSRP Prob.    : {result['non_hsrp_probability']:.4f}")


if __name__ == "__main__":
    main()

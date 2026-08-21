import argparse
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from .model import build_model
from .preprocessing import eval_transforms


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.sigmoid(outputs).squeeze(1)

        preds = (probs > 0.5).int()

        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())
        y_prob.extend(probs.cpu().numpy())

    return y_true, y_pred, y_prob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ImageFolder(
        Path(args.data_dir) / "test",
        transform=eval_transforms(),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)

    y_true, y_pred, _ = collect_predictions(model, loader, device)

    print("Class mapping:", dataset.class_to_idx)
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall   :", recall_score(y_true, y_pred))
    print("F1-score :", f1_score(y_true, y_pred))
    print("\nClassification Report:\n")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["HSRP", "NON_HSRP"],
        )
    )
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    main()

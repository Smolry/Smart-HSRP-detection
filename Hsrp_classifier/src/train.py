import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from .model import build_model, freeze_backbone, unfreeze_last_feature_blocks
from .preprocessing import train_transforms, eval_transforms


def make_loaders(data_dir, batch_size=32, num_workers=2):
    data_dir = Path(data_dir)

    train_dataset = ImageFolder(data_dir / "train", transform=train_transforms())
    val_dataset = ImageFolder(data_dir / "val", transform=eval_transforms())

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_dataset, val_dataset, train_loader, val_loader


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item()

    return running_loss / len(loader)


def run_phase(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    criterion,
    device,
    epochs,
    output_path,
    label,
):
    best_val_loss = float("inf")

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss = validate_one_epoch(
            model, val_loader, criterion, device
        )
        scheduler.step(val_loss)

        print(
            f"{label} Epoch [{epoch + 1}/{epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_path)

    return best_val_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--phase1-epochs", type=int, default=5)
    parser.add_argument("--phase2-epochs", type=int, default=20)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_dataset, val_dataset, train_loader, val_loader = make_loaders(
        args.data_dir, args.batch_size, args.num_workers
    )

    print("Class mapping:", train_dataset.class_to_idx)

    model = build_model(pretrained=True)
    freeze_backbone(model)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()

    # Phase 1: train only the replacement classifier head.
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.1
    )

    phase1_path = output_dir / "efficientnet_b0_phase1.pth"
    run_phase(
        model, train_loader, val_loader, optimizer, scheduler, criterion,
        device, args.phase1_epochs, phase1_path, "[Phase 1]"
    )

    model.load_state_dict(torch.load(phase1_path, map_location=device))

    # Phase 2: fine-tune the last two feature blocks plus classifier.
    unfreeze_last_feature_blocks(model, n_blocks=2)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.1
    )

    phase2_path = output_dir / "efficientnet_b0_finetuned.pth"
    run_phase(
        model, train_loader, val_loader, optimizer, scheduler, criterion,
        device, args.phase2_epochs, phase2_path, "[Phase 2]"
    )

    print(f"Saved final checkpoint: {phase2_path}")


if __name__ == "__main__":
    main()

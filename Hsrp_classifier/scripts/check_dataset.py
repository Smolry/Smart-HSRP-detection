import argparse
from pathlib import Path

from torchvision.datasets import ImageFolder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()

    root = Path(args.data_dir)

    for split in ("train", "val", "test"):
        path = root / split
        dataset = ImageFolder(path)
        print(f"{split}: {len(dataset)} images")
        print(f"  classes: {dataset.class_to_idx}")
        for cls, idx in dataset.class_to_idx.items():
            count = sum(1 for _, label in dataset.samples if label == idx)
            print(f"  {cls}: {count}")


if __name__ == "__main__":
    main()

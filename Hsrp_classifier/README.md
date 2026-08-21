# HSRP Classification

HSRP classification component of the Smart-HSRP project.

This component classifies cropped license-plate images into two
classes:

- `hsrp`
- `non-hsrp`

## Development history

`notebooks/exploratory_training.ipynb` preserves the original
exploratory Colab training notebook.

The implementation under `src/` is a cleaned, reusable version of
the training, evaluation, and inference pipeline derived from that
notebook.

## Dataset layout

The training code expects an `ImageFolder` layout:

```text
data.v0/
├── train/
│   ├── hsrp/
│   └── non-hsrp/
├── val/
│   ├── hsrp/
│   └── non-hsrp/
└── test/
    ├── hsrp/
    └── non-hsrp/
```

The original notebook reports the class mapping:

```text
hsrp      -> 0
non-hsrp  -> 1
```

## Preprocessing

Training images are:

1. resized to `224 × 224`;
2. augmented with `ColorJitter(brightness=0.2, contrast=0.2)`;
3. randomly rotated by up to `5` degrees;
4. converted to tensors; and
5. normalized using ImageNet mean/std:

```text
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

Validation, test, and inference use the same resize and normalization
without the training-only augmentations.

## Model

The classifier is EfficientNet-B0 initialized with ImageNet
weights. Its original classifier is replaced with a single-output
linear layer.

The notebook initially freezes the EfficientNet feature extractor.
After phase 1, the last two feature blocks are unfrozen for fine-tuning.

## Training

The exploratory notebook uses:

- Batch size: 32
- Phase 1: 5 epochs
- Phase 1 learning rate: `1e-4`
- Phase 2: 20 epochs
- Phase 2 learning rate: `1e-5`
- Optimizer: Adam
- Loss: BCEWithLogitsLoss
- Scheduler: ReduceLROnPlateau
- Scheduler patience: 3
- Scheduler factor: 0.1
- DataLoader workers: 2

### Train

From the repository root:

```bash
python -m src.train \
  --data-dir /path/to/data.v0 \
  --output-dir artifacts
```

The training script produces:

```text
artifacts/
├── efficientnet_b0_phase1.pth
└── efficientnet_b0_finetuned.pth
```

The phase 1 checkpoint is the best checkpoint by validation loss
during phase 1. The final checkpoint is the best checkpoint by
validation loss during phase 2.

## Evaluation

```bash
python -m src.evaluate \
  --data-dir /path/to/data.v0 \
  --checkpoint artifacts/efficientnet_b0_finetuned.pth
```

The evaluation reports:

- Accuracy
- Precision
- Recall
- F1 score
- Classification report
- Confusion matrix

The exploratory notebook reported the following test results on
439 samples:

| Metric | Score |
|---|---:|
| Accuracy | 0.8747 |
| Precision | 0.8792 |
| Recall | 0.8585 |
| F1 | 0.8687 |

Per-class results reported by the notebook:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| HSRP | 0.87 | 0.89 | 0.88 | 227 |
| NON_HSRP | 0.88 | 0.86 | 0.87 | 212 |

These are results recorded in the exploratory notebook and should
not be interpreted as a guarantee of performance on new data.

## Inference

```bash
python -m src.inference \
  --image path/to/plate.jpg \
  --checkpoint artifacts/efficientnet_b0_finetuned.pth
```

The model uses a `0.5` decision threshold by default.

### Important note about the original notebook

The exploratory notebook maps:

```text
hsrp      -> 0
non-hsrp  -> 1
```

and trains with `BCEWithLogitsLoss`. Therefore the sigmoid output
corresponds to class `1`, i.e. `non-hsrp`.

The original notebook's inference cell labels `prob > 0.5` as `HSRP`,
which appears inconsistent with its `ImageFolder` class mapping.

The cleaned `src/inference.py` preserves the class mapping and
therefore interprets sigmoid output as the probability of
`NON_HSRP`.

This discrepancy is intentionally documented rather than silently
preserving the apparent label inversion.

## Relationship to the full Smart-HSRP pipeline

The classifier operates on plate crops rather than full vehicle
images. Plate detection/cropping is a separate preprocessing stage.

Conceptually:

```text
Full image
    ↓
Plate detector
    ↓
Plate crop
    ↓
224 × 224 classification preprocessing
    ↓
EfficientNet-B0
    ↓
HSRP / NON_HSRP
```

## Exploratory notebook

The original development notebook is retained at:

```text
notebooks/exploratory_training.ipynb
```

It contains the original Colab workflow, training experiments,
evaluation, and inference code.

## License

MIT

# Smart-HSRP — Model Archives

This branch contains archival copies of model weights used during the
development, testing, and deployment of the **Smart-HSRP** project.

The purpose of this branch is to preserve the exact model artifacts
used by the project for **reproducibility, deployment recovery, and
long-term reference**.

This branch is separate from the `main` branch so that large model
artifacts do not form part of the primary application/deployment
source tree.

---

## Purpose of This Branch

The `model-archives` branch is intended to:

- Preserve model weights used by Smart-HSRP.
- Maintain a record of the specific model artifacts used by the
  project.
- Support reproducibility of experiments and deployments.
- Provide a backup in case the original local/server copies are lost.
- Document the provenance, licensing, and attribution of each model.

This branch is **not** the canonical source code of Smart-HSRP.

The `main` branch contains the application, deployment files,
configuration templates, tests, documentation, and other source
files required to develop and operate the project.

---

## Important: Model Ownership and Attribution

Not all models stored in this branch were developed by the
Smart-HSRP project.

The presence of a model weight file in this branch **does not imply
that the Smart-HSRP project or its contributors created, trained, or
own that model**.

Each model is classified below as either:

1. **Project-developed** — trained or developed specifically as part of
   Smart-HSRP; or
2. **Third-party** — obtained from an external author, repository,
   dataset, or pretrained model and used as a component of Smart-HSRP.

Third-party models remain subject to their original licenses and
attribution requirements.

Where redistribution of a third-party model is not permitted by its
license or original terms, the model should not be stored in this
branch. Instead, this README or the relevant model documentation
should reference the original source.

---

# Archived Models

| Model | Type | Usage | Source / Attribution |
|---|---|---|---|
| HSRP Classifier | Project-developed | HSRP / non-HSRP classification | Smart-HSRP project |
| Helmet Detector | Third-party / project fine-tuning* | Helmet detection | See model-specific documentation |
| Vehicle / Person Detector | Third-party pretrained | Vehicle and person detection | Ultralytics YOLO |
| License Plate Detector | Third-party pretrained | License-plate detection | Guardian-22 YOLOv10s repository |
| Other models | See individual directory | See individual directory | See individual directory |

\* The helmet model should be described according to its actual
provenance. If the base model or dataset originated from another
author, that source must remain credited even if the model was
fine-tuned or otherwise modified for Smart-HSRP.

---

# Directory Structure

```text
model-archives/
│
├── README.md
│
├── hsrp/
│   └── hsrp_classifier.<extension>
│
├── helmet/
│   └── helmet_detector.<extension>
│
├── vehicle/
│   └── vehicle_person_detector.<extension>
│
└── license-plate/
    └── plate_detector.<extension>

# Smart-HSRP — Reproducible Environment

This directory documents the **known-working AWS EC2 ML environment** used by the Smart-HSRP deployment.

It is intentionally separated from the application source. The project lives under `~/Smart-HSRP-detection`, while the known-good ML runtime lives under `/opt/pytorch`.

## Known-good core

| Component | Version / Location |
|---|---|
| Project | `~/Smart-HSRP-detection` |
| ML Python | `/opt/pytorch/bin/python3` |
| Python | 3.13.13 |
| PyTorch | 2.10.0+cu130 |
| TorchVision | 0.25.0+cu130 |
| TorchAudio | 2.10.0+cu130 |
| TorchData | 0.11.0 |
| Ultralytics | 8.4.53 |
| CUDA build | 13.0 |
| NVIDIA driver | 580.126.16 |
| GPU | NVIDIA Tesla T4, 15,360 MiB |
| TensorRT | 10.16.1 |
| TensorRT CLI | `trtexec` |
| Project venv | Not part of the known-good ML architecture |
| Ephemeral storage | `/opt/dlami/nvme` |

The project and ML runtime are separate layers of the deployment architecture. [Environment record](#source-record)

## Important reproducibility rule

Do **not** assume that `pip install -r requirements.txt` recreates this environment. The working deployment depends on Python, PyTorch/CUDA, NVIDIA drivers, TensorRT, native libraries, Ultralytics, ONNX/OpenCV and model artifacts in addition to ordinary Python packages.

The correct long-term strategy is:

```text
GitHub source
    +
environment manifest
    +
Python package snapshot
    +
system package snapshot
    +
GPU/driver snapshot
    +
TensorRT version
    +
model/archive references
```

## Files

```text
environment/
├── README.md
├── ENVIRONMENT.md
├── requirements-ml.txt
├── bootstrap-ec2.sh
├── verify-environment.sh
└── snapshot-environment.sh
```

## Verify an existing machine

```bash
./environment/verify-environment.sh
```

This checks Python, PyTorch, CUDA, NVIDIA GPU, TensorRT, Ultralytics, the project directory and disk space.

## Rebuild the Python ML layer

The intended interpreter is:

```text
/opt/pytorch/bin/python3
```

A conservative bootstrap is provided:

```bash
sudo ./environment/bootstrap-ec2.sh
```

The bootstrap deliberately does **not** install or replace the NVIDIA driver or TensorRT automatically. Those host-level components must match the selected AWS GPU/AMI and should be installed using the appropriate official procedure.

## Exact deployment recovery

For the closest reproduction of the deployment environment, use:

1. the same or compatible GPU instance type;
2. a compatible NVIDIA driver;
3. CUDA 13-compatible runtime;
4. TensorRT 10.16.1;
5. Python 3.13.13;
6. the `/opt/pytorch` package snapshot;
7. the Smart-HSRP Git repository;
8. required model artifacts;
9. non-secret configuration;
10. production secrets from the secure deployment store.

## Python boundary

Always use the ML interpreter explicitly:

```bash
/opt/pytorch/bin/python3
```

and install packages with:

```bash
/opt/pytorch/bin/python3 -m pip install ...
```

Do not assume `python`, `python3`, or `pip` refer to `/opt/pytorch`.

## PYTHONPATH

Do not globally add `/opt/pytorch/lib/python3.13/site-packages` to `PYTHONPATH` unless there is a specific reason. Prefer the actual interpreter. An uncontrolled `PYTHONPATH` previously caused package mixing, including a bcrypt conflict.

## Storage

`/opt/dlami/nvme` is ephemeral and must not be the only copy of source, models, configuration or generated artifacts. Keep important artifacts in GitHub, Hugging Face, S3 or another persistent store as appropriate.

## Verification before changes

```bash
/opt/pytorch/bin/python3 -c "import torch; print(torch.__version__)"
nvidia-smi
trtexec --version 2>&1 | head -5
df -h /
```

If these are healthy, take a snapshot before modifying the environment.

## Environment snapshots

```bash
./environment/snapshot-environment.sh
```

Store the generated snapshot on persistent storage. Do not rely on ephemeral NVMe for the only copy.

## Security

Never commit `.env`, passwords, secret keys, API keys, certificates, SSH private keys or production credentials. This documentation describes the environment, not secret values.

## Source record

This directory was normalized from the project's recorded EC2 environment document. The original record explicitly distinguishes the project from `/opt/pytorch`, records Python 3.13.13 and PyTorch 2.10.0+cu130, and warns that the environment contains dependencies not represented by a normal project requirements file.

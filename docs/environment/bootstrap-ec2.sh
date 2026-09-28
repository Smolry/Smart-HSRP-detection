#!/usr/bin/env bash
set -euo pipefail
PY="/opt/pytorch/bin/python3"
if [ "$(id -u)" -ne 0 ]; then echo "Run: sudo ./environment/bootstrap-ec2.sh"; exit 1; fi
if [ ! -x "$PY" ]; then echo "ERROR: $PY does not exist."; exit 1; fi
"$PY" --version
nvidia-smi || true
"$PY" -m pip install --upgrade pip setuptools wheel
"$PY" -m pip install \
  "torch==2.10.0+cu130" \
  "torchvision==0.25.0+cu130" \
  "torchaudio==2.10.0+cu130" \
  "torchdata==0.11.0" \
  "ultralytics==8.4.53"
echo "TensorRT is a host/native dependency. Known-good version: 10.16.1"
echo "Verify with: trtexec --version"
echo "Run ./environment/verify-environment.sh after bootstrap."

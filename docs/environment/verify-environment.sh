#!/usr/bin/env bash
set -u
PY="/opt/pytorch/bin/python3"
PROJECT="${PROJECT_DIR:-$HOME/Smart-HSRP-detection}"
fail=0

echo "=== Smart-HSRP Environment Verification ==="

if [ ! -x "$PY" ]; then echo "FAIL: $PY not found"; exit 1; fi

echo "=== Python ==="; "$PY" --version

echo "Executable:"; "$PY" -c 'import sys; print(sys.executable)'

echo "=== PyTorch ==="; "$PY" -c 'import torch; print(torch.__version__); print("CUDA available:", torch.cuda.is_available()); print("CUDA version:", torch.version.cuda)' || fail=1
echo "=== TorchVision ==="; "$PY" -c 'import torchvision; print(torchvision.__version__)' || fail=1
echo "=== TorchAudio ==="; "$PY" -c 'import torchaudio; print(torchaudio.__version__)' || fail=1
echo "=== Ultralytics ==="; "$PY" -c 'import ultralytics; print(ultralytics.__version__)' || fail=1
echo "=== NVIDIA ==="; nvidia-smi || fail=1
echo "=== TensorRT CLI ==="; command -v trtexec && trtexec --version 2>&1 | head -5 || fail=1
echo "=== TensorRT Python (ML interpreter) ==="; "$PY" -c 'import tensorrt as trt; print(trt.__version__)' || echo "WARNING: TensorRT Python is not visible to /opt/pytorch/bin/python3"
echo "=== Project ==="; [ -d "$PROJECT" ] && echo "$PROJECT" || echo "WARNING: project not found: $PROJECT"
echo "=== Disk ==="; df -h /
echo "=== PYTHONPATH ==="; printf '%s\n' "${PYTHONPATH:-<unset>}"

if [ "$fail" -eq 0 ]; then echo "Environment verification completed."; else echo "Environment has failed checks."; exit 1; fi

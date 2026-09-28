#!/usr/bin/env bash
set -euo pipefail
OUT="${1:-$HOME/environment-backup}"
PY="/opt/pytorch/bin/python3"
mkdir -p "$OUT"
"$PY" --version > "$OUT/python-version.txt"
"$PY" -m pip freeze > "$OUT/pytorch-pip-freeze.txt"
"$PY" -m pip list > "$OUT/pytorch-pip-list.txt"
python3 -m pip freeze > "$OUT/system-python-environment-freeze.txt" 2>&1 || true
dpkg-query -W -f='${binary:Package}\t${Version}\n' > "$OUT/dpkg-packages.txt"
nvidia-smi > "$OUT/nvidia-smi.txt" 2>&1 || true
trtexec --version > "$OUT/trtexec-version.txt" 2>&1 || true
"$PY" -c 'import torch; print("torch:", torch.__version__); print("cuda:", torch.version.cuda); print("cuda_available:", torch.cuda.is_available())' > "$OUT/pytorch-cuda.txt"
"$PY" -c 'import torchvision; print(torchvision.__version__)' > "$OUT/torchvision-version.txt" 2>&1 || true
"$PY" -c 'import ultralytics; print(ultralytics.__version__)' > "$OUT/ultralytics-version.txt" 2>&1 || true
echo "Snapshot written to $OUT"

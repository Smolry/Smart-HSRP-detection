# Smart-HSRP Environment Manifest

## Reference machine

```text
GPU: NVIDIA Tesla T4
VRAM: 15360 MiB
NVIDIA driver: 580.126.16
NVIDIA-reported CUDA: 13.0

ML interpreter: /opt/pytorch/bin/python3
Python: 3.13.13

PyTorch: 2.10.0+cu130
TorchVision: 0.25.0+cu130
TorchAudio: 2.10.0+cu130
TorchData: 0.11.0
Ultralytics: 8.4.53

TensorRT: 10.16.1
TensorRT CLI: trtexec
```

## Filesystem layout

```text
/home/ubuntu/Smart-HSRP-detection
    └── application source, models, configuration

/opt/pytorch
    ├── bin/python3
    └── lib/python3.13/site-packages
        ├── torch
        ├── torchvision
        ├── torchaudio
        ├── torchdata
        └── other ML packages

/usr
    ├── system Python
    └── TensorRT/system runtime components

/opt/dlami/nvme
    └── ephemeral NVMe storage
```

## Python installation policy

Identify the interpreter first:

```bash
which python
which python3
which pip
python --version
python3 --version
```

Use the ML environment explicitly:

```bash
/opt/pytorch/bin/python3 -m pip ...
```

Do not recreate a project venv merely because `~/Smart-HSRP-detection/venv` is absent. The known-good deployment did not rely on a project-local venv.

## PyTorch

```bash
/opt/pytorch/bin/python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```

Expected core result:

```text
2.10.0+cu130
True
13.0
```

## TorchVision / TorchAudio

```bash
/opt/pytorch/bin/python3 -c "import torchvision; print(torchvision.__version__)"
/opt/pytorch/bin/python3 -c "import torchaudio; print(torchaudio.__version__)"
```

Expected:

```text
0.25.0+cu130
2.10.0+cu130
```

## Ultralytics

```bash
/opt/pytorch/bin/python3 -c "import ultralytics; print(ultralytics.__version__)"
yolo version
```

Expected package version:

```text
8.4.53
```

## TensorRT

```bash
which trtexec
trtexec --version 2>&1 | head -5
```

Expected TensorRT version:

```text
10.16.1
```

The observed TensorRT Python installation was associated with system Python 3.12, while the main ML interpreter is Python 3.13. A working `trtexec` does not automatically mean `import tensorrt` works in every interpreter.

## Model conversion

### YOLO → TensorRT

```bash
yolo export model=weights/vehicle-person.pt format=engine device=0 batch=8 imgsz=640
```

Required: model weights, PyTorch, CUDA, Ultralytics and TensorRT.

### ONNX → TensorRT

```bash
trtexec \
  --onnx=weights/hsrp_cls.onnx \
  --saveEngine=weights/hsrp_cls.engine \
  --fp16 \
  --verbose
```

For dynamic inputs, inspect the ONNX input name first:

```bash
/opt/pytorch/bin/python3 -c "import onnx; m=onnx.load('weights/hsrp_cls.onnx'); print([x.name for x in m.graph.input])"
```

Then use the appropriate `--shapes` option.

### Test an engine

```bash
trtexec --loadEngine=weights/hsrp_cls.engine
```

## Storage

Reference deployment storage:

```text
Root filesystem: approximately 29G
Ephemeral NVMe: approximately 115G
```

Check:

```bash
df -h
```

Never keep the only copy of important source/model/configuration on `/opt/dlami/nvme`.

## Snapshot commands

```bash
mkdir -p ~/environment-backup
/opt/pytorch/bin/python3 --version > ~/environment-backup/python-version.txt
/opt/pytorch/bin/python3 -m pip freeze > ~/environment-backup/pytorch-pip-freeze.txt
/opt/pytorch/bin/python3 -m pip list > ~/environment-backup/pytorch-pip-list.txt
dpkg-query -W -f='${binary:Package}\t${Version}\n' > ~/environment-backup/dpkg-packages.txt
nvidia-smi > ~/environment-backup/nvidia-smi.txt
trtexec --version > ~/environment-backup/trtexec-version.txt 2>&1 || true
```

## Recovery order

If something breaks:

1. `which python`, `which pip`
2. `python --version`
3. inspect `PYTHONPATH`
4. verify `nvidia-smi`
5. verify PyTorch/CUDA
6. verify TensorRT
7. inspect package locations
8. inspect disk space
9. only then reinstall a specific broken component

Do not immediately reinstall the entire ML stack.

## Do not delete

Do not casually remove:

```text
/opt/pytorch
/usr/lib/python3.12/dist-packages
NVIDIA/CUDA libraries
system TensorRT libraries
```

Use package managers for system software.

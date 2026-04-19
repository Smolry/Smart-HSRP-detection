"""
LOCAL STORAGE UTILITY
======================
Replaces S3 with EC2 local disk storage.
All paths are anchored to the project root (two levels up from this file),
so they work regardless of what directory uvicorn is launched from.
"""

import os
import shutil
from pathlib import Path

# Anchor to the project root — robust regardless of working directory
_THIS_FILE   = Path(__file__).resolve()          # .../backend/utils/local_storage.py
_PROJECT_ROOT = _THIS_FILE.parent.parent.parent   # .../project/

# Respect STORAGE_DIR env var, but resolve it relative to project root
_STORAGE_DIR = os.getenv("STORAGE_DIR", "static")
STORAGE_BASE = (_PROJECT_ROOT / _STORAGE_DIR).resolve()
INPUTS_DIR   = STORAGE_BASE / "inputs"
OUTPUTS_DIR  = STORAGE_BASE / "outputs"


def _ensure_dirs():
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def save_input_video(src_path: str, job_id: str) -> str:
    """Copy uploaded video into <storage>/inputs/<job_id>.mp4"""
    _ensure_dirs()
    dest = INPUTS_DIR / f"{job_id}.mp4"
    shutil.copy2(src_path, dest)
    return str(dest)


def get_output_video_path(job_id: str) -> str:
    """Absolute path where the pipeline should write the annotated video."""
    _ensure_dirs()
    return str(OUTPUTS_DIR / f"{job_id}.mp4")


def output_video_exists(job_id: str) -> bool:
    """Check using the same absolute path as get_output_video_path."""
    return Path(get_output_video_path(job_id)).exists()


def get_video_url(job_id: str, base_url: str = "") -> str:
    """URL the frontend uses — served by FastAPI StaticFiles at /static/**"""
    return f"{base_url}/static/outputs/{job_id}.mp4"


def cleanup_input(job_id: str):
    """Delete the raw input after processing to free disk space."""
    path = INPUTS_DIR / f"{job_id}.mp4"
    if path.exists():
        path.unlink()

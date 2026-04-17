"""
LOCAL STORAGE UTILITY
======================
Replaces S3 with EC2 local disk storage.
Videos are served directly via FastAPI's FileResponse / StaticFiles.
"""

import os
import shutil
from pathlib import Path
from typing import Optional

# Base directory for all stored files (relative to project root)
STORAGE_BASE = Path(os.getenv("STORAGE_DIR", "static"))
INPUTS_DIR  = STORAGE_BASE / "inputs"
OUTPUTS_DIR = STORAGE_BASE / "outputs"

def _ensure_dirs():
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def save_input_video(src_path: str, job_id: str) -> str:
    """
    Copy uploaded video into static/inputs/<job_id>.mp4
    Returns the local file path (used internally).
    """
    _ensure_dirs()
    dest = INPUTS_DIR / f"{job_id}.mp4"
    shutil.copy2(src_path, dest)
    return str(dest)


def get_output_video_path(job_id: str) -> str:
    """Return the path where the pipeline should write the annotated video."""
    _ensure_dirs()
    return str(OUTPUTS_DIR / f"{job_id}.mp4")


def output_video_exists(job_id: str) -> bool:
    return (OUTPUTS_DIR / f"{job_id}.mp4").exists()


def get_video_url(job_id: str, base_url: str = "") -> str:
    """
    Build the URL the frontend uses to stream/download the output video.
    FastAPI serves /static/** so the URL is /static/outputs/<job_id>.mp4
    """
    return f"{base_url}/static/outputs/{job_id}.mp4"


def cleanup_input(job_id: str):
    """Delete the raw input video to free disk space after processing."""
    path = INPUTS_DIR / f"{job_id}.mp4"
    if path.exists():
        path.unlink()

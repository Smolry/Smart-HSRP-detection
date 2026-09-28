"""
LOCAL STORAGE UTILITY
======================
All paths anchored to project root via __file__ — works regardless
of which directory uvicorn is launched from.

FIX: cleanup_input() is now a no-op so input videos are NEVER deleted
after processing. They remain in static/inputs/ and are listed by the
Available Videos library on the frontend.
"""
import os
import shutil
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_STORAGE_DIR  = os.getenv("STORAGE_DIR", "static")
STORAGE_BASE  = (_PROJECT_ROOT / _STORAGE_DIR).resolve()
INPUTS_DIR    = STORAGE_BASE / "inputs"
OUTPUTS_DIR   = STORAGE_BASE / "outputs"


def _ensure_dirs():
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def save_input_video(src_path: str, job_id: str) -> str:
    _ensure_dirs()
    dest = INPUTS_DIR / f"{job_id}.mp4"
    shutil.copy2(src_path, dest)
    return str(dest)


def get_output_video_path(job_id: str) -> str:
    _ensure_dirs()
    return str(OUTPUTS_DIR / f"{job_id}.mp4")


def output_video_exists(job_id: str) -> bool:
    return Path(get_output_video_path(job_id)).exists()


def get_video_url(job_id: str, base_url: str = "") -> str:
    return f"{base_url}/static/outputs/{job_id}.mp4"


def get_input_video_url(job_id: str, base_url: str = "") -> str:
    return f"{base_url}/static/inputs/{job_id}.mp4"


def cleanup_input(job_id: str):
    """
    Intentionally disabled — input videos are kept in static/inputs/
    so they can be re-processed from the Available Videos library
    without re-uploading.
    """
    pass  # DO NOT DELETE — kept for Available Videos library

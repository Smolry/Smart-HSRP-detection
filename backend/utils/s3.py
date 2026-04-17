"""
S3 MODULE — REPLACED BY LOCAL STORAGE
This file is kept as a compatibility shim.
All actual logic lives in backend/utils/local_storage.py
"""
from backend.utils.local_storage import get_output_video_path

def upload_file_to_s3(file_path: str, folder: str = "uploads", job_id=None) -> str:
    raise NotImplementedError(
        "S3 has been removed. Use backend.utils.local_storage instead."
    )

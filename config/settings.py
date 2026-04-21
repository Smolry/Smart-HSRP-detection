from typing import ClassVar
from pydantic_settings import BaseSettings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    PROJECT_NAME: str = "Smart HSRP"
    VERSION:      str = "1.0.0"

    # Model paths
    HELMET_MODEL_PATH:  str
    PLATE_MODEL_PATH:   str
    HSRP_MODEL_PATH:    str
    VEHICLE_MODEL_PATH: str

    # Security
    SECRET_KEY:                  str
    ALGORITHM:                   str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Thresholds
    HELMET_CONF_THRESHOLD:  float
    HSRP_CONF_THRESHOLD:    float = 0.5
    OCR_CONF_THRESHOLD:     float
    VEHICLE_CONF_THRESHOLD: float
    RIDER_IOU_THRESHOLD:    float
    PLATE_IOU_THRESHOLD:    float
    HEAD_CROP_RATIO:        float
    HELMET_NO_THRESHOLD:    float
    HELMET_YES_THRESHOLD:   float
    FRAME_SKIP:             int

    # Database
    DATABASE_URL: str

    # Local storage
    STORAGE_DIR: str = "static"

    # Redis
    REDIS_HOST: str
    REDIS_PORT: int

    # API
    HOST:  str  = "0.0.0.0"
    PORT:  int  = 8000
    DEBUG: bool = False

    BASE_DIR:  ClassVar[Path] = BASE_DIR
    DB_CONFIG: ClassVar[dict] = {}

    model_config = {
        "env_file":       ".env",
        "case_sensitive": False,
        "extra":          "ignore",   # ignore unknown env vars (e.g. CORS_ORIGINS)
    }


settings = Settings()

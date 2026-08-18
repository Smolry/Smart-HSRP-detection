# Utility functions
# bcrypt 5.x dropped the passlib-compatible API that passlib 1.7 used internally.
# Instead of passlib we call bcrypt directly — works with any bcrypt >= 3.x.
from datetime import datetime, timedelta
import os
import bcrypt
from config.settings import settings
from jose import jwt


def hash_password(password: str) -> str:
    """Hash a plaintext password. Returns a bcrypt hash string."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against a stored bcrypt hash."""
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8"),
        )
    except Exception:
        return False


def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, settings.ALGORITHM)

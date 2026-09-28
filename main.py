"""
SMART HSRP MONITORING SYSTEM — GPU-Optimized
=============================================
Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
    (keep workers=1 — GPU models are not fork-safe)
"""

from contextlib import asynccontextmanager
from pathlib import Path
import torch

from jose import jwt
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr

from config.settings import settings
from backend.api.routes import router as detection_router, init_redis, close_redis
from backend.db.database import get_db
from backend.utils.access_func import create_access_token, hash_password, verify_password

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    await init_redis()
    yield
    await close_redis()


app = FastAPI(
    title="Smart HSRP Monitoring API",
    version="3.0.0-gpu",
    lifespan=lifespan,
)

# Static file serving
for d in ["static/outputs", "static/inputs", "state"]:
    Path(d).mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS — explicit list (wildcard + credentials is invalid per spec)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
    ],
    allow_origin_regex=r"http://\d+\.\d+\.\d+\.\d+.*",  # any EC2 IP
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(detection_router, prefix="/api")


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "3.0.0-gpu",
        "gpu": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


@app.get("/")
async def root():
    return {"message": "Smart HSRP Monitoring API v3 (GPU)"}


# ── Auth ──────────────────────────────────────────────────────────────────────

class UserSignup(BaseModel):
    email:    EmailStr
    password: str
    role:     str = "user"


class UserLogin(BaseModel):
    email:    EmailStr
    password: str


class Token(BaseModel):
    access_token: str
    token_type:   str
    user_id:      int
    email:        str
    role:         str


@app.post("/signup", response_model=Token)
def signup(user: UserSignup, conn=Depends(get_db)):
    if user.role not in ("admin", "user"):
        raise HTTPException(400, "Invalid role")
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE email = %s", (user.email,))
    if cursor.fetchone():
        raise HTTPException(400, "Email already registered")
    hashed = hash_password(user.password)
    cursor.execute(
        "INSERT INTO users (email, password_hash, role) VALUES (%s, %s, %s) RETURNING id",
        (user.email, hashed, user.role),
    )
    user_id = cursor.fetchone()["id"]
    conn.commit()
    token = create_access_token({"sub": user.email, "role": user.role, "user_id": user_id})
    return Token(access_token=token, token_type="bearer", user_id=user_id, email=user.email, role=user.role)


@app.post("/login", response_model=Token)
def login(user: UserLogin, conn=Depends(get_db)):
    cursor = conn.cursor()
    cursor.execute("SELECT id, email, password_hash, role FROM users WHERE email = %s", (user.email,))
    db_user = cursor.fetchone()
    if not db_user or not verify_password(user.password, db_user["password_hash"]):
        raise HTTPException(401, "Invalid credentials")
    token = create_access_token({"sub": db_user["email"], "role": db_user["role"], "user_id": db_user["id"]})
    return Token(access_token=token, token_type="bearer", user_id=db_user["id"],
                 email=db_user["email"], role=db_user["role"])


@app.get("/verify-token")
def verify_token(token: str):
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        email   = payload.get("sub")
        if not email:
            raise HTTPException(401, "Invalid token")
        return {"valid": True, "email": email, "role": payload.get("role"), "user_id": payload.get("user_id")}
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Token expired")
    except jwt.JWTError:
        raise HTTPException(401, "Invalid token")


@app.get("/users")
def get_all_users(authorization: str = Header(None), conn=Depends(get_db)):
    raw_token = authorization.replace("Bearer ", "") if authorization else ""
    try:
        payload = jwt.decode(raw_token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        if payload.get("role") != "admin":
            raise HTTPException(403, "Admin access required")
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Token expired")
    except jwt.JWTError:
        raise HTTPException(401, "Invalid token")
    cursor = conn.cursor()
    cursor.execute("SELECT id, email, role, created_at FROM users ORDER BY created_at DESC")
    return {"users": cursor.fetchall()}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False,          # reload=True breaks GPU model sharing
        workers=1,             # GPU models are not fork-safe
        log_level="info",
    )

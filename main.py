# main.py
"""
Vetra AI - Secure FastAPI single-file server (brain + API system)
- Production-minded: hashed api keys, JWT admin, rate-limits, audit logs, metrics
- NOT self-modifying. Use secure CI/CD for updates (notes at bottom).
"""

import os
import time
import secrets
import hashlib
import json
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

from fastapi import FastAPI, Request, HTTPException, status, Depends
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel, Field
from authlib.integrations.starlette_client import OAuth
import bcrypt
import asyncpg
import aioredis
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import jwt  # PyJWT
import uvicorn

# ----------------------
# CONFIG (use env vars)
# ----------------------
DATABASE_URL = os.getenv("DATABASE_URL")  # required in production (postgres)
REDIS_URL = os.getenv("REDIS_URL", None)  # optional (for caching/rate)
APP_SECRET_KEY = os.getenv("APP_SECRET_KEY", secrets.token_urlsafe(32))
BASE_URL = os.getenv("BASE_URL", "")  # e.g. https://vortek-ai.onrender.com
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()] or ["*"]
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
Vetra_MASTER_KEY = os.getenv("Vetra_MASTER_KEY", secrets.token_urlsafe(32))
Vetra_ADMIN_JWT_SECRET = os.getenv("Vetra_ADMIN_JWT_SECRET", secrets.token_urlsafe(32))
ADMIN_JWT_EXPIRE_MINUTES = int(os.getenv("ADMIN_JWT_EXPIRE_MINUTES", "15"))

# Rate-limits (example)
DEFAULT_RATE = os.getenv("DEFAULT_RATE", "30/minute")

# ----------------------
# Logging & metrics
# ----------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("vetra")

REQUESTS = Counter("vetra_requests_total", "Total API requests", ["endpoint"])
LATENCY = Histogram("vetra_request_latency_seconds", "Request latency seconds", ["endpoint"])

# ----------------------
# FastAPI init
# ----------------------
app = FastAPI(title="Vetra AI", version="1.0")
app.add_middleware(SessionMiddleware, secret_key=APP_SECRET_KEY)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Rate limiter
limiter = Limiter(key_func=get_remote_address, default_limits=[DEFAULT_RATE])
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

# OAuth (optional)
oauth = OAuth()
if GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET:
    oauth.register(
        name="google",
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )

# ----------------------
# Database + Redis Pool (async)
# ----------------------
@app.on_event("startup")
async def startup():
    # Postgres pool
    if not DATABASE_URL:
        logger.warning("DATABASE_URL not set — using in-memory fallback (NOT for production).")
        app.state.db = None
    else:
        app.state.db = await asyncpg.create_pool(DATABASE_URL)
        # create tables if missing
        async with app.state.db.acquire() as conn:
            await conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                email TEXT UNIQUE,
                name TEXT,
                created_at BIGINT
            );
            """)
            await conn.execute("""
            CREATE TABLE IF NOT EXISTS api_keys (
                id SERIAL PRIMARY KEY,
                user_email TEXT,
                key_hash TEXT,
                created_at BIGINT,
                expires_at BIGINT,
                max_uses INTEGER DEFAULT 1000,
                uses INTEGER DEFAULT 0,
                revoked BOOLEAN DEFAULT FALSE,
                viewed BOOLEAN DEFAULT FALSE,
                bound_ip TEXT
            );
            """)
            await conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id SERIAL PRIMARY KEY,
                api_key_id INTEGER,
                user_email TEXT,
                path TEXT,
                meta JSONB,
                ts BIGINT
            );
            """)
    # Redis pool
    if REDIS_URL:
        app.state.redis = await aioredis.from_url(REDIS_URL)
    else:
        app.state.redis = None
    logger.info("Startup complete")

@app.on_event("shutdown")
async def shutdown():
    if getattr(app.state, "db", None):
        await app.state.db.close()
    if getattr(app.state, "redis", None):
        await app.state.redis.close()
    logger.info("Shutdown complete")

# ----------------------
# Models (requests/responses)
# ----------------------
class RegisterModel(BaseModel):
    email: str = Field(..., example="user@example.com")
    name: Optional[str] = None

class AskModel(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=5000)

class CreateKeyResponse(BaseModel):
    api_key: str
    note: str

# ----------------------
# Helper functions (security & keys)
# ----------------------
def generate_plain_api_key() -> str:
    return "vetra_" + secrets.token_urlsafe(28)

def hash_api_key(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()

def verify_api_key(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode(), hashed.encode())
    except Exception:
        return False

async def store_api_key(email: str, plain: str, expires_days: int = 30, max_uses: int = 1000):
    hashed = hash_api_key(plain)
    now = int(time.time())
    expires = now + 60*60*24*expires_days
    if app.state.db:
        await app.state.db.execute(
            "INSERT INTO api_keys (user_email, key_hash, created_at, expires_at, max_uses, viewed) VALUES ($1,$2,$3,$4,$5,$6)",
            email, hashed, now, expires, max_uses, True
        )
    else:
        # fall-back in-memory
        keys = app.state.__dict__.setdefault("inmem_keys", {})
        keys[plain] = {"email": email, "hash": hashed, "expires": expires, "max_uses": max_uses, "uses": 0, "revoked": False}

async def find_key_row_by_plain(plain: str):
    if app.state.db:
        rows = await app.state.db.fetch("SELECT id,user_email,key_hash,expires_at,max_uses,uses,revoked,bound_ip FROM api_keys WHERE revoked=false")
        for r in rows:
            if verify_api_key(plain, r["key_hash"]):
                return dict(r)
        return None
    else:
        keys = app.state.__dict__.get("inmem_keys", {})
        v = keys.get(plain)
        if not v: return None
        return {"id": None, "user_email": v["email"], "key_hash": v["hash"], "expires_at": v["expires"], "max_uses": v["max_uses"], "uses": v["uses"], "revoked": v["revoked"], "bound_ip": None}

async def increment_key_use(key_id):
    if app.state.db and key_id:
        await app.state.db.execute("UPDATE api_keys SET uses = uses + 1 WHERE id=$1", key_id)

async def record_audit(api_key_id, user_email, path, meta=None):
    if app.state.db:
        await app.state.db.execute("INSERT INTO audit_logs (api_key_id,user_email,path,meta,ts) VALUES ($1,$2,$3,$4,$5)",
                                   api_key_id, user_email, path, json.dumps(meta or {}), int(time.time()))
    else:
        logs = app.state.__dict__.setdefault("inmem_audit", [])
        logs.append({"api_key_id":api_key_id, "user_email":user_email, "path":path, "meta":meta or {}, "ts":int(time.time())})

# ----------------------
# Admin JWT helpers
# ----------------------
def admin_create_token(admin_name: str) -> str:
    payload = {"sub": admin_name, "iat": int(time.time()), "exp": int(time.time()) + ADMIN_JWT_EXPIRE_MINUTES*60}
    return jwt.encode(payload, VORTEK_ADMIN_JWT_SECRET, algorithm="HS256")

def admin_verify_token(token: str) -> Dict[str, Any]:
    try:
        return jwt.decode(token, VETRA_ADMIN_JWT_SECRET, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Admin token expired")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid admin token")

# ----------------------
# Simple brain core (starter)
# ----------------------
# This is a simple and safe "brain" to handle prompts. Replace or extend with your model.
class SimpleBrain:
    def __init__(self):
        self.short_memory = []  # list of (ts, prompt, response)
        self.max_memory = 200

    def respond(self, prompt: str, user_email: str) -> str:
        # very simple: if prompt asks about itself, respond with system info
        p = prompt.strip().lower()
        if "who are you" in p or "what is your name" in p:
            return f"I am Vetra AI — your personal assistant. (user={user_email})"
        if p.endswith("?"):
            # echo with small transformation
            return f"Vetra AI says: I understood your question '{prompt}'. Here's a short reply."
        # fallback: short creative reply
        return f"Vetra AI processed: {prompt}"

    def remember(self, prompt: str, response: str):
        self.short_memory.append({"ts": int(time.time()), "prompt": prompt, "response": response})
        if len(self.short_memory) > self.max_memory:
            self.short_memory.pop(0)

brain = SimpleBrain()

# ----------------------
# Routes: health + metrics
# ----------------------
@app.get("/health")
async def health():
    return {"status": "ok", "time": int(time.time())}

@app.get("/metrics")
async def metrics():
    content = generate_latest()
    return JSONResponse(content=content, media_type=CONTENT_TYPE_LATEST)

# ----------------------
# User registration & key creation flow (session-based)
# ----------------------
@app.post("/register", tags=["auth"])
@limiter.limit("5/minute")
async def register(payload: RegisterModel, request: Request):
    # Minimal validation
    email = payload.email.strip().lower()
    name = payload.name or ""
    now = int(time.time())
    # store user in DB
    if app.state.db:
        await app.state.db.execute("INSERT INTO users (email,name,created_at) VALUES ($1,$2,$3) ON CONFLICT (email) DO NOTHING",
                                   email, name, now)
    else:
        users = app.state.__dict__.setdefault("inmem_users", {})
        users[email] = {"name": name, "created_at": now}
    # create session
    request.session["user_email"] = email
    return {"message": "Registered (session active). Use /create_key to get your one-time API key."}

@app.post("/create_key", response_model=CreateKeyResponse, tags=["auth"])
@limiter.limit("3/minute")
async def create_key(request: Request):
    email = request.session.get("user_email")
    if not email:
        raise HTTPException(status_code=401, detail="Not logged in (session missing). Call /register first.")
    # limit keys per user
    if app.state.db:
        cnt = await app.state.db.fetchval("SELECT COUNT(*) FROM api_keys WHERE user_email=$1 AND revoked=false", email)
    else:
        inmem = app.state.__dict__.get("inmem_keys", {})
        cnt = sum(1 for v in inmem.values() if v["email"] == email and not v["revoked"])
    if cnt and cnt >= 5:
        raise HTTPException(status_code=403, detail="Max active keys reached")
    plain = generate_plain_api_key()
    await store_api_key(email, plain)
    # return one-time plain key
    return CreateKeyResponse(api_key=plain, note="Save this API key now. It will not be shown again.")

@app.get("/list_keys", tags=["auth"])
async def list_keys(request: Request):
    email = request.session.get("user_email")
    if not email:
        raise HTTPException(status_code=401, detail="Not logged in")
    if app.state.db:
        rows = await app.state.db.fetch("SELECT id,created_at,expires_at,uses,max_uses,revoked,viewed FROM api_keys WHERE user_email=$1", email)
        return {"keys": [dict(r) for r in rows]}
    else:
        inmem = app.state.__dict__.get("inmem_keys", {})
        return {"keys": [{**v, "plain_key": k} for k, v in inmem.items() if v["email"] == email]}

@app.post("/revoke_key", tags=["auth"])
async def revoke_key(request: Request):
    email = request.session.get("user_email")
    if not email:
        raise HTTPException(status_code=401, detail="Not logged in")
    data = await request.json()
    key_id = data.get("key_id")
    if not key_id:
        raise HTTPException(status_code=400, detail="key_id required")
    # careful update
    if app.state.db:
        await app.state.db.execute("UPDATE api_keys SET revoked=true WHERE id=$1 AND user_email=$2", int(key_id), email)
    else:
        # find inmem by plain and revoke
        inmem = app.state.__dict__.get("inmem_keys", {})
        for k, v in list(inmem.items()):
            if v["email"] == email and v.get("id") == int(key_id):
                v["revoked"] = True
    return {"status": "revoked"}

# ----------------------
# API: /ask endpoint (protected by API key)
# ----------------------
@app.post("/ask", tags=["api"])
@limiter.limit("60/minute")
async def ask(request: Request, data: AskModel):
    REQUESTS.labels(endpoint="/ask").inc()
    start = time.time()
    api_key = request.headers.get("Authorization")
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing API key in Authorization header")
    row = await find_key_row_by_plain(api_key)
    if not row:
        raise HTTPException(status_code=403, detail="Invalid or revoked API key")
    # checks
    now = int(time.time())
    if row.get("expires_at") and now > row["expires_at"]:
        raise HTTPException(status_code=403, detail="API key expired")
    if row.get("uses") is not None and row["uses"] >= row.get("max_uses", 0):
        raise HTTPException(status_code=403, detail="Usage limit reached")
    # optional bound IP check
    if row.get("bound_ip"):
        if request.client and request.client.host != row["bound_ip"]:
            raise HTTPException(status_code=403, detail="IP not allowed for this key")
    # increment
    await increment_key_use(row.get("id"))
    # audit
    await record_audit(row.get("id"), row.get("user_email"), "/ask", {"prompt_len": len(data.prompt)})
    # CALL BRAIN
    safe_prompt = data.prompt.strip()[:5000]  # basic sanitization/limit
    response_text = brain.respond(safe_prompt, row.get("user_email"))
    brain.remember(safe_prompt, response_text)
    LATENCY.labels(endpoint="/ask").observe(time.time() - start)
    return {"answer": response_text, "user": row.get("user_email"), "timestamp": datetime.utcnow().isoformat()}

# ----------------------
# Admin endpoints (require JWT)
# ----------------------
def require_admin(request: Request):
    auth = request.headers.get("Authorization")
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Admin token missing")
    token = auth.split(" ", 1)[1]
    return admin_verify_token(token)

@app.post("/admin/token", tags=["admin"])
def get_admin_token(body: Dict[str, str]):
    # local check: require VETRA_MASTER_KEY to create admin token
    key = body.get("master_key")
    if not key or key != VETRA_MASTER_KEY:
        raise HTTPException(status_code=403, detail="Invalid master key")
    token = admin_create_token("admin")
    return {"token": token, "expires_min": ADMIN_JWT_EXPIRE_MINUTES}

@app.get("/admin/overview", tags=["admin"])
async def admin_overview(request: Request):
    require_admin(request)
    # show safe overview (no plain keys)
    total_users = 0
    total_keys = 0
    if app.state.db:
        total_users = await app.state.db.fetchval("SELECT COUNT(*) FROM users")
        total_keys = await app.state.db.fetchval("SELECT COUNT(*) FROM api_keys")
    else:
        total_users = len(app.state.__dict__.get("inmem_users", {}))
        total_keys = len(app.state.__dict__.get("inmem_keys", {}))
    return {"users": total_users, "api_keys": total_keys, "status": "ok"}

# ----------------------
# Security recommendations endpoint (read-only)
# ----------------------
@app.get("/security/health", tags=["admin"])
def security_health():
    return {
        "hints": [
            "Use managed Postgres (RDS) with encryption at rest",
            "Use managed Redis for rate-limits",
            "Store secrets in KMS / Vault, not env in plain text",
            "Use Cloudflare or similar WAF in front of API",
            "Use CI/CD with signed releases for updates (no self-modifying code)"
        ]
    }

# ----------------------
# Run (use uvicorn externally in Render)
# ----------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)

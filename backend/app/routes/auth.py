# routes_auth.py
from fastapi import APIRouter, HTTPException, Header
from firebase_admin import auth as firebase_auth
from typing import Dict, Any, Optional
import time, uuid

router = APIRouter()

# In-memory for demo; switch to Redis in prod.
HANDOFF_TTL = 120  # seconds
_handoff_store: Dict[str, Dict[str, Any]] = {}  # { code: { uid, exp } }

def _now() -> float:
    return time.time()

def _gc_handoffs():
    now = _now()
    expired = [k for k, v in _handoff_store.items() if v["exp"] <= now]
    for k in expired:
        _handoff_store.pop(k, None)

def _verify_bearer(auth_header: Optional[str]) -> str:
    if not auth_header:
        raise HTTPException(401, detail="Missing Authorization")
    try:
        scheme, token = auth_header.split(" ", 1)
    except ValueError:
        raise HTTPException(401, detail="Invalid Authorization header")
    if scheme.lower() != "bearer":
        raise HTTPException(401, detail="Expected Bearer <idToken>")
    try:
        decoded = firebase_auth.verify_id_token(token, check_revoked=True)
        return decoded["uid"]
    except Exception as e:
        raise HTTPException(401, detail=f"Invalid Firebase ID token: {e}")

@router.post("/handoff/start")
async def handoff_start(Authorization: Optional[str] = Header(None)):
    """RN calls this with Authorization: Bearer <Firebase ID token>."""
    _gc_handoffs()
    uid = _verify_bearer(Authorization)

    code = uuid.uuid4().hex
    _handoff_store[code] = {"uid": uid, "exp": _now() + HANDOFF_TTL}

    # (Optional) add IP/rate-limit protections here.
    return {"code": code, "expiresIn": HANDOFF_TTL}

@router.post("/handoff/consume")
async def handoff_consume(code: str):
    """Unity calls with the one-time 'code' to get a Firebase Custom Token."""
    _gc_handoffs()
    entry = _handoff_store.pop(code, None)
    if not entry or entry["exp"] <= _now():
        raise HTTPException(400, detail="Code invalid or expired")

    uid = entry["uid"]
    try:
        custom_token = firebase_auth.create_custom_token(uid)
        # firebase_admin returns bytes in some envs → normalize to str
        try:
            custom_token = custom_token.decode()
        except Exception:
            pass
    except Exception as e:
        raise HTTPException(500, detail=f"Failed to create custom token: {e}")

    return {"customToken": custom_token, "uid": uid}


import random
import string

def _create_custom_token(uid: str) -> str:
    """Helper to create a simple 8-character token using uid prefix + random numbers."""
    try:
        # Take first 4 chars of uid (or pad with 'x' if uid is too short)
        uid_prefix = (uid + 'xxxx')[:4]
        
        # Generate 4 random numbers
        random_numbers = ''.join(random.choices(string.digits, k=4))
        
        # Combine to create 8-char token
        token = f"{uid_prefix}{random_numbers}"
        
        return token
    except Exception as e:
        raise HTTPException(500, detail=f"Failed to create custom token: {e}")



async def require_firebase_user(Authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    """
    Accepts: Authorization: Bearer <Firebase ID token>
    Returns: {"uid": <uid>, "claims": <decoded_token>}
    """
    if not Authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    try:
        scheme, token = Authorization.split(" ", 1)
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid Authorization header")

    if scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Expected Bearer <idToken>")

    try:
        decoded = firebase_auth.verify_id_token(token, check_revoked=True)
        return {"uid": decoded["uid"], "claims": decoded}
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid Firebase ID token: {e}")
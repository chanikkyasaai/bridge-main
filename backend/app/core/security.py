from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
import uuid
import hashlib
from app.core.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Token types
TOKEN_TYPE_ACCESS = "access"
TOKEN_TYPE_REFRESH = "refresh"

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token with short expiry"""
    # Only include safe fields in the token
    safe_fields = {"sub", "phone", "device_id"}
    to_encode = {key: value for key, value in data.items() if key in safe_fields}
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": TOKEN_TYPE_ACCESS
    })
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

def create_refresh_token(data: dict) -> str:
    """Create JWT refresh token with long expiry"""
    # Only include safe fields in the token
    safe_fields = {"sub", "phone", "device_id"}
    to_encode = {key: value for key, value in data.items() if key in safe_fields}
    
    expire = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": TOKEN_TYPE_REFRESH,
        "jti": str(uuid.uuid4())  # Unique token ID for revocation
    })
    encoded_jwt = jwt.encode(to_encode, settings.REFRESH_SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

def create_session_token(phone: str, device_id: str, user_id: str) -> str:
    """Create a session token for WebSocket and session management"""
    session_data = {
        "user_phone": phone,
        "user_id": user_id,
        "device_id": device_id,
        "session_id": str(uuid.uuid4()),
        "created_at": datetime.utcnow().isoformat(),
        "type": "session",
        "exp": (datetime.utcnow() + timedelta(minutes=settings.SESSION_EXPIRE_MINUTES)).timestamp()
    }
    return jwt.encode(session_data, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

def verify_access_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify and decode access token"""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        if payload.get("type") != TOKEN_TYPE_ACCESS:
            return None
        return payload
    except JWTError:
        return None

def verify_refresh_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify and decode refresh token"""
    try:
        payload = jwt.decode(token, settings.REFRESH_SECRET_KEY, algorithms=[settings.ALGORITHM])
        if payload.get("type") != TOKEN_TYPE_REFRESH:
            return None
        return payload
    except JWTError:
        return None

def verify_session_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify and decode session token"""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        if payload.get("type") != "session":
            return None
        return payload
    except JWTError:
        return None

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def hash_mpin(mpin: str) -> str:
    """Hash MPIN using SHA256"""
    return hashlib.sha256(mpin.encode()).hexdigest()

def verify_mpin(plain_mpin: str, hashed_mpin: str) -> bool:
    """Verify MPIN"""
    return hash_mpin(plain_mpin) == hashed_mpin

# Legacy function for backward compatibility
def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """Legacy token verification - tries access token first, then session token"""
    # Try access token first
    payload = verify_access_token(token)
    if payload:
        return payload
    
    # Fallback to session token
    return verify_session_token(token)

def extract_session_info(token: str) -> Optional[Dict[str, Any]]:
    """Extract session information from token"""
    payload = verify_session_token(token)
    if payload:
        return {
            "user_phone": payload.get("user_phone"),
            "user_id": payload.get("user_id"),
            "session_id": payload.get("session_id"),
            "device_id": payload.get("device_id")
        }
    return None

def get_token_payload(token: str) -> Optional[Dict[str, Any]]:
    """Get token payload without verification (for debugging)"""
    try:
        # Decode without verification to see token content
        payload = jwt.decode(token, options={"verify_signature": False})
        return payload
    except JWTError:
        return None
        return None

def extract_session_info(token: str) -> Optional[Dict[str, Any]]:
    """Extract session information from JWT token"""
    payload = verify_token(token)
    if payload and "session_id" in payload:
        return {
            "session_id": payload["session_id"],
            "user_email": payload["user_email"],
            "device_id": payload["device_id"],
            "created_at": payload["created_at"]
        }
    return None

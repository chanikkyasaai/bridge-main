from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
import uuid
import hashlib
from app.core.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

def create_session_token(user_email: str, device_id: str) -> str:
    """Create a unique session token"""
    session_data = {
        "user_email": user_email,
        "device_id": device_id,
        "session_id": str(uuid.uuid4()),
        "created_at": datetime.utcnow().isoformat(),
        "exp": (datetime.utcnow() + timedelta(minutes=settings.SESSION_EXPIRE_MINUTES)).timestamp()
    }
    return jwt.encode(session_data, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

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

def verify_token(token: str) -> Optional[Dict[str, Any]]:
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload
    except JWTError:
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

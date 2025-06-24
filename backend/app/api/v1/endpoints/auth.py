from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from datetime import timedelta
from typing import Optional
from app.core.security import (
    create_session_token, verify_password, get_password_hash, 
    verify_token, hash_mpin, verify_mpin, extract_session_info
)
from app.core.session_manager import session_manager
from app.core.config import settings

router = APIRouter()
security = HTTPBearer()

class UserRegister(BaseModel):
    email: EmailStr
    password: str
    mpin: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str
    device_id: str

class MPINVerification(BaseModel):
    mpin: str

class SessionToken(BaseModel):
    session_token: str
    token_type: str
    expires_in: int
    session_id: str

class MPINChallenge(BaseModel):
    session_id: str
    mpin: str

# Mock user database - replace with actual database
mock_users = {}

def validate_mpin(mpin: str) -> bool:
    """Validate MPIN format"""
    return len(mpin) == settings.MPIN_LENGTH and mpin.isdigit()

@router.post("/register", response_model=dict)
async def register(user_data: UserRegister):
    """
    Register a new user with email, password, and MPIN
    """
    # Validate MPIN
    if not validate_mpin(user_data.mpin):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"MPIN must be exactly {settings.MPIN_LENGTH} digits"
        )
    
    # Check if user already exists
    if user_data.email in mock_users:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Hash password and MPIN
    hashed_password = get_password_hash(user_data.password)
    hashed_mpin = hash_mpin(user_data.mpin)
    
    # Store user
    mock_users[user_data.email] = {
        "email": user_data.email,
        "hashed_password": hashed_password,
        "hashed_mpin": hashed_mpin,
        "is_active": True,
        "failed_mpin_attempts": 0,
        "mpin_locked_until": None
    }
    
    return {
        "message": "User registered successfully",
        "email": user_data.email,
        "next_step": "Use login endpoint with email, password, and device_id"
    }

@router.post("/login", response_model=SessionToken)
async def login(user_data: UserLogin):
    """
    Login with email and password, create a session
    """
    # Verify user credentials
    user = mock_users.get(user_data.email)
    if not user or not verify_password(user_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )
    
    if not user["is_active"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Account is deactivated"
        )
    
    # Create session
    session_id = session_manager.create_session(user_data.email, user_data.device_id)
    
    # Create session token
    session_token = create_session_token(user_data.email, user_data.device_id)
    
    return SessionToken(
        session_token=session_token,
        token_type="bearer",
        expires_in=settings.SESSION_EXPIRE_MINUTES * 60,
        session_id=session_id
    )

@router.post("/verify-mpin")
async def verify_mpin_endpoint(
    mpin_data: MPINVerification,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Verify MPIN for the current session
    """
    # Extract session info from token
    session_info = extract_session_info(credentials.credentials)
    if not session_info:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid session token"
        )
    
    user_email = session_info["user_email"]
    session_id = session_info["session_id"]
    
    # Get user data
    user = mock_users.get(user_email)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Get session
    session = session_manager.get_session(session_id)
    if not session or session.is_blocked:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Session is blocked or invalid"
        )
    
    # Verify MPIN
    if verify_mpin(mpin_data.mpin, user["hashed_mpin"]):
        # Reset MPIN attempts on successful verification
        user["failed_mpin_attempts"] = 0
        session.mpin_attempts = 0
        
        # Log successful MPIN verification
        session.add_behavioral_data("mpin_verified", {
            "session_id": session_id,
            "success": True,
            "timestamp": session.last_activity.isoformat()
        })
        
        return {
            "message": "MPIN verified successfully",
            "session_id": session_id,
            "status": "verified"
        }
    else:
        # Increment failed attempts
        user["failed_mpin_attempts"] += 1
        session.mpin_attempts += 1
        
        # Log failed MPIN attempt
        session.add_behavioral_data("mpin_failed", {
            "session_id": session_id,
            "attempts": session.mpin_attempts,
            "timestamp": session.last_activity.isoformat()
        })
        
        # Block account after max attempts
        if user["failed_mpin_attempts"] >= settings.MAX_MPIN_ATTEMPTS:
            session.block_session("Too many failed MPIN attempts")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account locked due to too many failed MPIN attempts"
            )
        
        remaining_attempts = settings.MAX_MPIN_ATTEMPTS - user["failed_mpin_attempts"]
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid MPIN. {remaining_attempts} attempts remaining"
        )

@router.post("/mpin-challenge")
async def mpin_challenge(
    challenge_data: MPINChallenge,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Handle MPIN challenge when suspicious behavior is detected
    """
    # Extract session info
    session_info = extract_session_info(credentials.credentials)
    if not session_info or session_info["session_id"] != challenge_data.session_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid session or session mismatch"
        )
    
    # Verify MPIN using the existing endpoint logic
    mpin_verification = MPINVerification(mpin=challenge_data.mpin)
    result = await verify_mpin_endpoint(mpin_verification, credentials)
    
    # If MPIN is verified, reduce risk score
    session = session_manager.get_session(challenge_data.session_id)
    if session:
        session.update_risk_score(max(0.0, session.risk_score - 0.3))  # Reduce risk
        
        session.add_behavioral_data("mpin_challenge_passed", {
            "session_id": challenge_data.session_id,
            "previous_risk_score": session.risk_score + 0.3,
            "new_risk_score": session.risk_score
        })
    
    return {
        **result,
        "risk_score_updated": True,
        "new_risk_score": session.risk_score if session else 0.0
    }

@router.get("/session-status")
async def get_session_status(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Get current session status and risk information
    """
    session_info = extract_session_info(credentials.credentials)
    if not session_info:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid session token"
        )
    
    session = session_manager.get_session(session_info["session_id"])
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    return session.get_session_stats()

@router.post("/logout")
async def logout(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Logout and terminate current session
    """
    session_info = extract_session_info(credentials.credentials)
    if session_info:
        session_manager.terminate_session(session_info["session_id"])
    
    return {"message": "Logged out successfully"}

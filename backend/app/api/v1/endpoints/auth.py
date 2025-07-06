from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from datetime import timedelta, datetime
from typing import Optional
from app.core.security import (
    create_access_token, create_refresh_token, create_session_token, 
    verify_password, get_password_hash, verify_access_token, 
    verify_refresh_token, extract_session_info, hash_mpin, verify_mpin
)
from app.core.session_manager import session_manager
from app.core.config import settings
from app.core.supabase_client import supabase_client
from app.core.token_manager import token_manager

router = APIRouter()
security = HTTPBearer()

class UserRegister(BaseModel):
    phone: str
    password: str
    mpin: str

class UserLogin(BaseModel):
    phone: str
    password: str
    device_id: str

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str
    expires_in: int

class SessionResponse(BaseModel):
    message: str
    user_id: str
    status: str
    session_id: str
    session_token: str
    behavioral_logging: str
    phone: str

class RefreshTokenRequest(BaseModel):
    refresh_token: str

class MPINVerification(BaseModel):
    mpin: str

class MPINChallenge(BaseModel):
    session_id: str
    mpin: str

class MPINLogin(BaseModel):
    phone: str
    mpin: str
    device_id: str

class FullAuthResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str
    expires_in: int
    session_id: str
    session_token: str
    behavioral_logging: str
    message: str

# User storage - this will be replaced by Supabase integration
mock_users = {}

def validate_phone(phone: str) -> bool:
    """Validate phone number format"""
    return len(phone) >= 10 and phone.isdigit()

def validate_mpin(mpin: str) -> bool:
    """Validate MPIN format"""
    return len(mpin) == settings.MPIN_LENGTH and mpin.isdigit()

@router.post("/register", response_model=dict)
async def register(user_data: UserRegister):
    import logging
    """
    Register a new user with phone, password, and MPIN
    """
    # Validate phone number
    if not validate_phone(user_data.phone):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid phone number format"
        )
    logging.info(user_data.phone)
    # Validate MPIN
    if not validate_mpin(user_data.mpin):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"MPIN must be exactly {settings.MPIN_LENGTH} digits"
        )
    logging.info(user_data.mpin)
    try:
        # Check if user already exists
        existing_user = await supabase_client.get_user_by_phone(user_data.phone)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Phone number already registered"
            )
        logging.info(existing_user)
        # Hash password and MPIN
        hashed_password = get_password_hash(user_data.password)
        hashed_mpin = hash_mpin(user_data.mpin)
        
        # Create user in Supabase
        user = await supabase_client.create_user(
            user_data.phone,
            hashed_password,
            hashed_mpin
        )
        
        return {
            "message": "User registered successfully",
            "user_id": user['id'],
            "phone": user_data.phone,
            "next_step": "Use login endpoint with phone, password, and device_id"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}"
        )

@router.post("/login", response_model=TokenResponse)
async def login(user_data: UserLogin):
    """
    Login with phone and password, return access + refresh tokens
    """
    try:
        # Verify user credentials
        user = await supabase_client.get_user_by_phone(user_data.phone)
        if not user or not verify_password(user_data.password, user["password_hash"]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect phone number or password"
            )
        
        user_id = user['id']
        
        # Create tokens
        access_token_data = {
            "sub": user_id,
            "phone": user_data.phone,
            "device_id": user_data.device_id
        }
        
        refresh_token_data = {
            "sub": user_id,
            "phone": user_data.phone,
            "device_id": user_data.device_id
        }
        
        # Generate tokens
        access_token = create_access_token(access_token_data)
        refresh_token = create_refresh_token(refresh_token_data)
        
        # Extract JTI from refresh token for storage
        from jose import jwt
        refresh_payload = jwt.decode(refresh_token, settings.SECRET_KEY, options={"verify_signature": False})
        jti = refresh_payload.get("jti")
        expires_at = datetime.fromtimestamp(refresh_payload.get("exp"))
        
        # Store refresh token
        token_manager.store_refresh_token(jti, user_id, user_data.device_id, expires_at)
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login failed: {str(e)}"
        )

@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(token_request: RefreshTokenRequest):
    """
    Refresh access token using refresh token
    """
    try:
        # Verify refresh token
        payload = verify_refresh_token(token_request.refresh_token)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        jti = payload.get("jti")
        user_id = payload.get("sub")
        phone = payload.get("phone")
        device_id = payload.get("device_id")
        
        # Check if token is still valid (not revoked)
        if not token_manager.is_token_valid(jti):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Refresh token has been revoked"
            )
        
        # Create new access token
        access_token_data = {
            "sub": user_id,
            "phone": phone,
            "device_id": device_id
        }
        
        new_access_token = create_access_token(access_token_data)
        
        # Optionally rotate refresh token (recommended for security)
        new_refresh_token_data = {
            "sub": user_id,
            "phone": phone,
            "device_id": device_id
        }
        
        new_refresh_token = create_refresh_token(new_refresh_token_data)
        
        # Extract new JTI and store new refresh token
        from jose import jwt
        new_refresh_payload = jwt.decode(new_refresh_token, settings.SECRET_KEY, options={"verify_signature": False})
        new_jti = new_refresh_payload.get("jti")
        new_expires_at = datetime.fromtimestamp(new_refresh_payload.get("exp"))
        
        # Revoke old refresh token and store new one
        token_manager.revoke_token(jti)
        token_manager.store_refresh_token(new_jti, user_id, device_id, new_expires_at)
        
        return TokenResponse(
            access_token=new_access_token,
            refresh_token=new_refresh_token,
            token_type="bearer",
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Token refresh failed: {str(e)}"
        )

# Dependency to verify access token
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Dependency to get current user from access token"""
    payload = verify_access_token(credentials.credentials)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid access token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return {
        "user_id": payload.get("sub"),
        "phone": payload.get("phone"),
        "device_id": payload.get("device_id")
    }

@router.post("/verify-mpin", response_model=SessionResponse)
async def verify_mpin_endpoint(
    mpin_data: MPINVerification,
    current_user: dict = Depends(get_current_user)
):
    """
    Verify MPIN for the current user (requires valid access token)
    """
    try:
        user_id = current_user["user_id"]
        phone = current_user["phone"]
        
        # Get user data from Supabase
        user = await supabase_client.get_user_by_phone(phone)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Verify MPIN
        if verify_mpin(mpin_data.mpin, user["mpin_hash"]):
            # Create behavioral logging session first
            session_id = await session_manager.create_session(
                user_id,
                phone,
                current_user["device_id"],
                None  # Pass None for session_token, we'll update it after creation
            )
            # Create session token with the actual session_id
            session_token = create_session_token(
                phone, current_user["device_id"], user_id, session_id)

            # Update the session with the token
            session = session_manager.get_session(session_id)
            if session:
                session.session_token = session_token
            
            if session:
                # Reset MPIN attempts on successful verification
                session.mpin_attempts = 0
                
                # Log successful MPIN verification
                session.add_behavioral_data("mpin_verified", {
                    "session_id": session.session_id,
                    "success": True,
                    "timestamp": session.last_activity.isoformat()
                })
                
                # Create security event
                if session.supabase_session_id:
                    await supabase_client.create_security_event(
                        session.supabase_session_id,
                        1,  # Level 1
                        "continue",
                        "MPIN verified successfully",
                        "mpin_verification",
                        1.0
                    )
            
            return SessionResponse(
                message="MPIN verified successfully",
                user_id=user_id,
                phone=phone,
                status="verified",
                session_id=session_id,
                session_token=session_token,
                behavioral_logging="started"
            )
        else:
            # MPIN verification failed
            # Since sessions are only created after successful MPIN verification,
            # we need to track failed attempts differently
            
            # For now, we'll implement a simple counter based on user_id
            # In a production system, this should be stored in a database
            # with proper expiration and cleanup
            
            # Check if user is currently locked out
            # This is a simplified implementation - in production, you'd want
            # to store lockout information in the database
            
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid MPIN"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"MPIN verification failed: {str(e)}"
        )

@router.post("/logout")
async def logout(current_user: dict = Depends(get_current_user)):
    """
    Logout user and revoke all tokens
    """
    try:
        user_id = current_user["user_id"]
        device_id = current_user.get("device_id")
        
        # Revoke all refresh tokens for the user
        if device_id:
            # Revoke tokens for specific device
            token_manager.revoke_device_tokens(user_id, device_id)
        else:
            # Revoke all tokens for user
            token_manager.revoke_user_tokens(user_id)
        
        # End session if exists
        user_sessions = session_manager.get_user_sessions(user_id)
        for session in user_sessions:
            if not device_id or session.device_id == device_id:
                session.end_session()
        
        return {
            "message": "Logged out successfully",
            "user_id": user_id
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Logout failed: {str(e)}"
        )

@router.get("/session-status")
async def get_session_status(current_user: dict = Depends(get_current_user)):
    """
    Get current session status and information
    """
    try:
        user_id = current_user["user_id"]
        device_id = current_user.get("device_id")
        
        # Get user sessions
        user_sessions = session_manager.get_user_sessions(user_id)
        active_session = None
        
        for session in user_sessions:
            if session.device_id == device_id and session.is_active:
                active_session = session
                break
        
        if not active_session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No active session found"
            )
        
        # Get active tokens count
        active_tokens = token_manager.get_user_active_tokens(user_id)
        device_tokens = [token for token in active_tokens if token["device_id"] == device_id]
        
        return {
            "session_id": active_session.session_id,
            "user_id": user_id,
            "device_id": device_id,
            "is_active": active_session.is_active,
            "created_at": active_session.created_at.isoformat(),
            "last_activity": active_session.last_activity.isoformat(),
            "risk_score": active_session.risk_score,
            "mpin_attempts": active_session.mpin_attempts,
            "is_locked": getattr(active_session, 'is_locked', False),
            "active_tokens_count": len(device_tokens),
            "total_user_tokens": len(active_tokens)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get session status: {str(e)}"
        )

@router.get("/tokens/active")
async def get_active_tokens(current_user: dict = Depends(get_current_user)):
    """
    Get all active tokens for the current user
    """
    try:
        user_id = current_user["user_id"]
        
        # Get user's active tokens
        active_tokens = token_manager.get_user_active_tokens(user_id)
        
        # Format response
        formatted_tokens = []
        for token in active_tokens:
            formatted_tokens.append({
                "device_id": token["device_id"],
                "created_at": token["created_at"].isoformat(),
                "expires_at": token["expires_at"].isoformat(),
                "is_current": token["device_id"] == current_user.get("device_id")
            })
        
        return {
            "user_id": user_id,
            "active_tokens": formatted_tokens,
            "total_count": len(formatted_tokens)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get active tokens: {str(e)}"
        )

@router.post("/tokens/revoke-device")
async def revoke_device_tokens(
    device_data: dict,
    current_user: dict = Depends(get_current_user)
):
    """
    Revoke all tokens for a specific device
    """
    try:
        user_id = current_user["user_id"]
        target_device_id = device_data.get("device_id")
        
        if not target_device_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="device_id is required"
            )
        
        # Revoke tokens for the specified device
        token_manager.revoke_device_tokens(user_id, target_device_id)
        
        return {
            "message": f"All tokens for device {target_device_id} have been revoked",
            "user_id": user_id,
            "device_id": target_device_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to revoke device tokens: {str(e)}"
        )

@router.post("/mpin-login", response_model=FullAuthResponse)
async def mpin_login(user_data: MPINLogin):
    """
    MPIN-only login for returning users. 
    Creates both authentication tokens AND behavioral session in one step.
    Perfect for users who open the app and just enter MPIN.
    """
    try:
        # Verify user exists and MPIN is correct
        user = await supabase_client.get_user_by_phone(user_data.phone)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )

        # Verify MPIN
        if not verify_mpin(user_data.mpin, user["mpin_hash"]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid MPIN"
            )

        user_id = user['id']

        # Create authentication tokens
        access_token_data = {
            "sub": user_id,
            "phone": user_data.phone,
            "device_id": user_data.device_id
        }

        refresh_token_data = {
            "sub": user_id,
            "phone": user_data.phone,
            "device_id": user_data.device_id
        }

        # Generate authentication tokens
        access_token = create_access_token(access_token_data)
        refresh_token = create_refresh_token(refresh_token_data)

        # Store refresh token
        from jose import jwt
        refresh_payload = jwt.decode(refresh_token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        jti = refresh_payload.get("jti")
        expires_at = datetime.fromtimestamp(refresh_payload.get("exp"))
        token_manager.store_refresh_token(
            jti, user_id, user_data.device_id, expires_at)

        # Create behavioral logging session immediately
        # First create session to get session_id, then create token with that session_id
        session_id = await session_manager.create_session(
            user_id,
            user_data.phone,
            user_data.device_id,
            None  # Pass None for session_token, we'll update it after creation
        )

        # Create session token with the actual session_id
        session_token = create_session_token(
            user_data.phone, user_data.device_id, user_id, session_id)

        # Update the session with the token
        
        # Get the created session for behavioral logging
        session = session_manager.get_session(session_id)
        if session:
            session.session_token = session_token
        
        if session:
            # Log MPIN verification success
            session.add_behavioral_data("mpin_verified", {
                "session_id": session.session_id,
                "success": True,
                "timestamp": session.last_activity.isoformat(),
                "login_type": "mpin_only"
            })
            
            # Create security event
            if session.supabase_session_id:
                await supabase_client.create_security_event(
                    session.supabase_session_id,
                    1,  # Level 1
                    "continue",
                    "MPIN-only login successful",
                    "mpin_login",
                    1.0
                )
        
        return FullAuthResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            session_id=session_id,
            session_token=session_token,
            behavioral_logging="started",
            message="MPIN login successful - behavioral logging started"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"MPIN login failed: {str(e)}"
        )
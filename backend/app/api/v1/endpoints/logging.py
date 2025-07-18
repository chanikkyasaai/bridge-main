from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from datetime import datetime
from app.api.v1.endpoints.auth import get_current_user
from app.core.session_manager import session_manager
from app.core.security import extract_session_info
from app.core.supabase_client import supabase_client
import logging

router = APIRouter()
security = HTTPBearer()

logging = logging.getLogger(__name__)

class StartSessionRequest(BaseModel):
    phone: str
    device_id: str
    context: Optional[Dict[str, Any]] = {}
    mpin: str


class BehaviorDataRequest(BaseModel):
    session_id: str
    event_type: str
    data: Dict[str, Any]


class EndSessionRequest(BaseModel):
    session_id: str
    final_decision: Optional[str] = "normal"
    session_token: str


class SessionResponse(BaseModel):
    session_id: str
    session_token: str
    supabase_session_id: Optional[str]
    message: str
    status: str


class AppCloseRequest(BaseModel):
    session_id: str
    session_token: str
    reason: str = "app_closed"  # app_closed, user_logout, app_background, etc.


class AppStateRequest(BaseModel):
    session_id: str
    state: str  # "background", "foreground", "minimized", "restored"
    details: Optional[Dict[str, Any]] = None


try:
    from app.ml_hooks import end_session_hook
    ML_INTEGRATION_AVAILABLE = True

    print("Imported ML-Engine integration in logging")
except ImportError:
    print("ML-Engine integration not available in logging")

    async def end_session_hook(*args, **kwargs):
        return None
    ML_INTEGRATION_AVAILABLE = False

@router.post("/start-session", response_model=SessionResponse)
async def start_session(request: StartSessionRequest, current_user: dict = Depends(get_current_user)):
    """
    Start a new behavioral logging session
    """
    try:
        user_id = current_user["user_id"]
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not authenticated"
            )
        # Create session token (you may want to integrate with your auth system)
        from app.core.security import create_session_token
        session_token = create_session_token(request.phone, request.device_id, user_id)
        
        # Create session
        session_id = await session_manager.create_session(
            user_id=user_id,
            phone=request.phone,
            device_id=request.device_id,
            session_token=session_token,
        )
        
        session = session_manager.get_session(session_id)
        
        return SessionResponse(
            session_id=session_id,
            session_token=session_token,
            supabase_session_id=session.supabase_session_id if session else None,
            message="Session started successfully",
            status="active"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start session: {str(e)}"
        )

@router.post("/behavior-data")
async def log_behavior_data(
    request: BehaviorDataRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Accept behavioral JSON data during an active session
    Note: Data is stored in memory and only uploaded to Supabase when session ends
    """
    try:
        # Verify session token
        session_info = extract_session_info(credentials.credentials)
        if not session_info:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid session token"
            )
        
        # Get session
        session = session_manager.get_session(request.session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        
        if session.is_blocked:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Session is blocked"
            )
        
        # Add behavioral data to session buffer (kept in memory)
        session.add_behavioral_data(request.event_type, request.data)
        
        return {
            "message": "Behavioral data logged successfully",
            "session_id": request.session_id,
            "event_type": request.event_type,
            "total_events": len(session.behavioral_buffer),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to log behavioral data: {str(e)}"
        )

@router.post("/end-session")
async def end_session(
    request: EndSessionRequest,
):
    """
    End session and upload all behavioral data to Supabase Storage
    """
    try:
        # Verify session token
        session_info = extract_session_info(request.session_token)
        if not session_info:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid session token"
            )
        
        # Get session
        session = session_manager.get_session(request.session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        
        # Terminate session (this will save behavioral data to Supabase)
        final_decision = request.final_decision if request.final_decision is not None else "normal"
        success = await session_manager.terminate_session(
            request.session_id, 
            final_decision
        )
        
        # --- ML Engine Integration: Call end_session_hook if available ---
        try:
            if ML_INTEGRATION_AVAILABLE:
                await end_session_hook(request.session_id, final_decision)
        except Exception as e:
            import logging
            logging.warning(f"ML end_session_hook failed: {e}")
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to terminate session"
            )
        
        return {
            "message": "Session ended successfully",
            "session_id": request.session_id,
            "final_decision": request.final_decision,
            "behavioral_data_saved": True,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to end session: {str(e)}"
        )

@router.get("/session/{session_id}/status")
async def get_session_status(
    session_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Get current session status and behavioral data summary
    """
    try:
        # Verify session token
        session_info = extract_session_info(credentials.credentials)
        if not session_info:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid session token"
            )
        
        # Get session
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        
        return {
            **session.get_session_stats(),
            "behavioral_data_summary": {
                "total_events": len(session.behavioral_buffer),
                "event_types": list(set(bd.event_type for bd in session.behavioral_buffer)),
                "last_event": session.behavioral_buffer[-1].timestamp if session.behavioral_buffer else None
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get session status: {str(e)}"
        )

@router.get("/session/{session_id}/logs")
async def get_session_logs(
    session_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Get behavioral logs for a completed session from Supabase Storage
    """
    try:
        # Verify session token
        session_info = extract_session_info(credentials.credentials)
        if not session_info:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid session token"
            )
        
        # Get session info from Supabase
        session_data = await supabase_client.get_session(session_id)
        if not session_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found in database"
            )
        
        if not session_data.get('log_file_url'):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No behavioral logs found for this session"
            )
        
        # Download logs from Supabase Storage
        logs = await supabase_client.download_behavioral_log(session_data['log_file_url'])
        
        return {
            "session_id": session_id,
            "logs": logs,
            "file_path": session_data['log_file_url']
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get session logs: {str(e)}"
        )

@router.post("/app-close")
async def handle_app_close(
    request: AppCloseRequest,
):
    """
    Handle explicit app closure by terminating the session and saving behavioral data
    """
    try:
        # Verify session token
        session_info = extract_session_info(request.session_token)
        if not session_info:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid session token"
            )
        # Verify session_id matches
        if session_info.get("session_id") != request.session_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Session ID mismatch"
            )
        # Get session
        session = session_manager.get_session(request.session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        # Handle app closure using lifecycle management
        success = await session_manager.handle_app_lifecycle_event(
            request.session_id,
            request.reason,
            {
                "explicit_close": True,
                "session_duration": (datetime.utcnow() - session.created_at).total_seconds()
            }
        )
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to terminate session on app close"
            )
        
        return {
            "message": "App closure handled successfully",
            "session_id": request.session_id,
            "reason": request.reason,
            "behavioral_data_saved": True,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to handle app closure: {str(e)}"
        )

@router.post("/app-state")
async def handle_app_state_change(
    request: AppStateRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Handle app state changes (background, foreground, etc.)
    """
    try:
        # Verify session token
        session_info = extract_session_info(credentials.credentials)
        if not session_info:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid session token"
            )
        
        # Verify session_id matches
        if session_info.get("session_id") != request.session_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Session ID mismatch"
            )
        
        # Handle state change
        success = await session_manager.handle_app_lifecycle_event(
            request.session_id,
            f"app_{request.state}",
            request.details or {}
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found or state change failed"
            )
        
        return {
            "message": f"App state changed to {request.state}",
            "session_id": request.session_id,
            "state": request.state,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to handle app state change: {str(e)}"
        )

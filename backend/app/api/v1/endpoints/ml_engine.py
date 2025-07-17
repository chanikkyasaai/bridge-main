"""
ML Engine endpoints for backend API
Provides admin and monitoring capabilities for ML integration
"""

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from app.core.security import verify_access_token
from app.ml_hooks import (
    get_ml_engine_status,
    get_ml_health,
    get_ml_statistics,
    feedback_hook,
    get_active_ml_sessions,
    cleanup_ml_session,
    is_ml_session_active
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic models
class MLEngineStatus(BaseModel):
    ml_engine_available: bool
    active_sessions: int
    session_list: List[str]
    health: Dict[str, Any]
    statistics: Dict[str, Any]
    last_check: str

class FeedbackRequest(BaseModel):
    user_id: str
    session_id: str
    decision_id: str
    was_correct: bool
    feedback_source: str = "manual"

class SessionCleanupRequest(BaseModel):
    session_id: str
    force: bool = False

@router.get("/status", response_model=MLEngineStatus)
async def get_ml_status(token_data: dict = Depends(verify_access_token)):
    """
    Get comprehensive ML Engine status
    Requires authentication
    """
    try:
        status = await get_ml_engine_status()
        return MLEngineStatus(**status)
    except Exception as e:
        logger.error(f"Failed to get ML status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get ML Engine status: {str(e)}"
        )

@router.get("/health")
async def get_ml_health_endpoint():
    """
    Get ML Engine health (public endpoint for monitoring)
    """
    try:
        health = await get_ml_health()
        return {
            "status": health.get("status", "unknown"),
            "timestamp": datetime.utcnow().isoformat(),
            "components": health.get("components", {}),
            "available": health.get("status") == "healthy"
        }
    except Exception as e:
        logger.error(f"Failed to get ML health: {e}")
        return {
            "status": "error",
            "timestamp": datetime.utcnow().isoformat(),
            "available": False,
            "error": str(e)
        }

@router.get("/statistics")
async def get_ml_statistics_endpoint(token_data: dict = Depends(verify_access_token)):
    """
    Get detailed ML Engine statistics
    Requires authentication
    """
    try:
        stats = await get_ml_statistics()
        return {
            "status": "success",
            "statistics": stats.get("statistics", {}),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get ML statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get ML statistics: {str(e)}"
        )

@router.post("/feedback")
async def submit_feedback(
    request: FeedbackRequest,
    token_data: dict = Depends(verify_access_token)
):
    """
    Submit feedback for ML model improvement
    Requires authentication
    """
    try:
        # Verify user has permission to submit feedback for this session
        token_user_id = token_data.get("user_id")
        if token_user_id != request.user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cannot submit feedback for other users"
            )
        
        result = await feedback_hook(
            user_id=request.user_id,
            session_id=request.session_id,
            decision_id=request.decision_id,
            was_correct=request.was_correct,
            feedback_source=request.feedback_source
        )
        
        if result.get("status") == "success":
            return {
                "status": "success",
                "message": "Feedback submitted successfully",
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to submit feedback: {result.get('message')}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Feedback submission failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit feedback: {str(e)}"
        )

@router.get("/sessions")
async def get_active_sessions(token_data: dict = Depends(verify_access_token)):
    """
    Get list of active ML sessions
    Requires authentication
    """
    try:
        sessions = get_active_ml_sessions()
        
        # Add session details
        session_details = []
        for session_id in sessions:
            session_details.append({
                "session_id": session_id,
                "is_active": is_ml_session_active(session_id),
                "status": "active"
            })
        
        return {
            "status": "success",
            "total_sessions": len(sessions),
            "sessions": session_details,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get active sessions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get active sessions: {str(e)}"
        )

@router.post("/sessions/cleanup")
async def cleanup_session(
    request: SessionCleanupRequest,
    token_data: dict = Depends(verify_access_token)
):
    """
    Cleanup ML session (admin function)
    Requires authentication
    """
    try:
        session_id = request.session_id
        
        # Check if session exists
        if not is_ml_session_active(session_id):
            return {
                "status": "info",
                "message": f"Session {session_id} is not active",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Force cleanup
        cleanup_ml_session(session_id)
        
        return {
            "status": "success",
            "message": f"Session {session_id} cleaned up successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Session cleanup failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cleanup session: {str(e)}"
        )

@router.get("/metrics")
async def get_ml_metrics():
    """
    Get ML Engine metrics for monitoring (public endpoint)
    """
    try:
        health = await get_ml_health()
        active_sessions = len(get_active_ml_sessions())
        
        return {
            "ml_engine_healthy": health.get("status") == "healthy",
            "active_sessions_count": active_sessions,
            "components": health.get("components", {}),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get ML metrics: {e}")
        return {
            "ml_engine_healthy": False,
            "active_sessions_count": 0,
            "components": {},
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

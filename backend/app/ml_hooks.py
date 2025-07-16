"""
ML Hooks for Backend Integration
Provides clean interface for ML Engine integration
"""

from typing import Dict, Any, List, Optional
import asyncio
import logging
from datetime import datetime

# Import ML Engine client
from .ml_engine_client import (
    start_ml_session,
    end_ml_session,
    submit_ml_feedback,
    behavioral_event_hook as _behavioral_event_hook,
    ml_engine_client
)

logger = logging.getLogger(__name__)

# Session tracking for ML integration
active_ml_sessions = set()

async def behavioral_event_hook(user_id: str, session_id: str, events: List[Dict]) -> Optional[Dict]:
    """
    Main behavioral event processing hook
    Called from WebSocket when behavioral events are received
    """
    try:
        # Ensure ML session is active
        if session_id not in active_ml_sessions:
            logger.warning(f"Behavioral event received for inactive ML session: {session_id}")
            # Try to start the session
            await start_session_hook(user_id, session_id)
        
        # Process events through ML Engine
        result = await _behavioral_event_hook(user_id, session_id, events)
        
        if result and result.get("status") == "success":
            # Check if action is needed based on ML decision
            decision = result.get("decision")
            confidence = result.get("confidence", 0.0)
            
            if decision in ["block", "challenge"] and confidence > 0.7:
                logger.warning(f"High-risk behavior detected in session {session_id}: "
                             f"decision={decision}, confidence={confidence:.2f}")
                
                # You can add additional logic here:
                # - Send alerts
                # - Update session risk status
                # - Trigger additional authentication steps
        
        return result
        
    except Exception as e:
        logger.error(f"Behavioral event hook failed: {e}")
        return None

async def start_session_hook(user_id: str, session_id: str, device_info: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Hook called when a user session starts
    Initializes ML Engine session tracking
    """
    try:
        logger.info(f"Starting ML session for {session_id}")
        
        # Start ML session
        result = await start_ml_session(user_id, session_id, device_info)
        
        if result.get("status") == "success":
            active_ml_sessions.add(session_id)
            logger.info(f"ML session started successfully: {session_id}")
        else:
            logger.error(f"Failed to start ML session: {result.get('message')}")
        
        return result
        
    except Exception as e:
        logger.error(f"Session start hook failed: {e}")
        return {"status": "error", "message": str(e)}

async def end_session_hook(session_id: str, reason: str = "completed") -> Dict[str, Any]:
    """
    Hook called when a user session ends
    Cleans up ML Engine session tracking
    """
    try:
        logger.info(f"Ending ML session for {session_id}")
        
        # End ML session
        result = await end_ml_session(session_id, reason)
        
        # Remove from active sessions
        active_ml_sessions.discard(session_id)
        
        if result.get("status") == "success":
            logger.info(f"ML session ended successfully: {session_id}")
        else:
            logger.error(f"Failed to end ML session: {result.get('message')}")
        
        return result
        
    except Exception as e:
        logger.error(f"Session end hook failed: {e}")
        return {"status": "error", "message": str(e)}

async def feedback_hook(user_id: str, session_id: str, decision_id: str, 
                       was_correct: bool, feedback_source: str = "system") -> Dict[str, Any]:
    """
    Hook for submitting feedback to ML Engine
    """
    try:
        logger.info(f"Submitting feedback for decision {decision_id}: correct={was_correct}")
        
        result = await submit_ml_feedback(user_id, session_id, decision_id, 
                                        was_correct, feedback_source)
        
        if result.get("status") == "success":
            logger.info(f"Feedback submitted successfully for {decision_id}")
        else:
            logger.error(f"Failed to submit feedback: {result.get('message')}")
        
        return result
        
    except Exception as e:
        logger.error(f"Feedback hook failed: {e}")
        return {"status": "error", "message": str(e)}

async def get_ml_health() -> Dict[str, Any]:
    """Get ML Engine health status"""
    try:
        async with ml_engine_client as client:
            return await client.health_check()
    except Exception as e:
        logger.error(f"ML health check failed: {e}")
        return {"status": "unavailable", "error": str(e)}

async def get_ml_statistics() -> Dict[str, Any]:
    """Get ML Engine statistics"""
    try:
        async with ml_engine_client as client:
            return await client.get_statistics()
    except Exception as e:
        logger.error(f"ML statistics failed: {e}")
        return {"status": "unavailable", "error": str(e)}

# Background task for ML Engine monitoring
async def ml_engine_monitor():
    """Background monitoring of ML Engine health"""
    while True:
        try:
            health = await get_ml_health()
            if health.get("status") != "healthy":
                logger.warning(f"ML Engine health check: {health.get('status')}")
            
            # Monitor active sessions
            if len(active_ml_sessions) > 0:
                logger.info(f"Active ML sessions: {len(active_ml_sessions)}")
            
            await asyncio.sleep(60)  # Check every minute
            
        except Exception as e:
            logger.error(f"ML Engine monitoring failed: {e}")
            await asyncio.sleep(60)

# Utility functions for session management
def is_ml_session_active(session_id: str) -> bool:
    """Check if ML session is active"""
    return session_id in active_ml_sessions

def get_active_ml_sessions() -> List[str]:
    """Get list of active ML sessions"""
    return list(active_ml_sessions)

def cleanup_ml_session(session_id: str) -> None:
    """Force cleanup of ML session (emergency)"""
    active_ml_sessions.discard(session_id)
    logger.info(f"Force cleanup of ML session: {session_id}")

# ML Engine status for health checks
async def get_ml_engine_status() -> Dict[str, Any]:
    """Comprehensive ML Engine status"""
    try:
        health = await get_ml_health()
        stats = await get_ml_statistics()
        
        return {
            "ml_engine_available": health.get("status") == "healthy",
            "active_sessions": len(active_ml_sessions),
            "session_list": list(active_ml_sessions),
            "health": health,
            "statistics": stats.get("statistics", {}),
            "last_check": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"ML Engine status check failed: {e}")
        return {
            "ml_engine_available": False,
            "active_sessions": len(active_ml_sessions),
            "error": str(e),
            "last_check": datetime.utcnow().isoformat()
        }

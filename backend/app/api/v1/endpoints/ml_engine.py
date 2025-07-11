"""
ML-Engine API endpoints for backend integration
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, Optional
import logging

# ML-Engine Integration
try:
    from ...ml_hooks import get_ml_engine_stats, get_session_ml_status
    ML_INTEGRATION_AVAILABLE = True
except ImportError:
    async def get_ml_engine_stats():
        return {"ml_enabled": False, "error": "ML integration not available"}
    async def get_session_ml_status(session_id: str):
        return None
    ML_INTEGRATION_AVAILABLE = False

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/ml-engine/status")
async def get_ml_engine_status():
    """Get ML-Engine status and statistics"""
    try:
        stats = await get_ml_engine_stats()
        return {
            "status": "success",
            "ml_integration_available": ML_INTEGRATION_AVAILABLE,
            "data": stats
        }
    except Exception as e:
        logger.error(f"Error getting ML-Engine status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get ML-Engine status")

@router.get("/ml-engine/session/{session_id}")
async def get_ml_session_info(session_id: str):
    """Get ML-Engine information for a specific session"""
    try:
        if not ML_INTEGRATION_AVAILABLE:
            raise HTTPException(status_code=503, detail="ML integration not available")
        
        ml_info = await get_session_ml_status(session_id)
        if ml_info is None:
            raise HTTPException(status_code=404, detail="Session not found in ML-Engine")
        
        return {
            "status": "success",
            "session_id": session_id,
            "data": ml_info
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting ML session info: {e}")
        raise HTTPException(status_code=500, detail="Failed to get ML session information")

@router.post("/ml-alerts")
async def handle_ml_alert(alert_data: Dict[str, Any]):
    """Handle ML-Engine alerts"""
    try:
        # Process ML alert (forward to appropriate handlers)
        logger.info(f"Received ML alert: {alert_data}")
        
        # Here you could:
        # - Send notifications to administrators
        # - Log to security systems
        # - Trigger additional security measures
        # - Update user risk profiles
        
        return {"status": "success", "message": "Alert processed"}
    except Exception as e:
        logger.error(f"Error processing ML alert: {e}")
        raise HTTPException(status_code=500, detail="Failed to process ML alert")

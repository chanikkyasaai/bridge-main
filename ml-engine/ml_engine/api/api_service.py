"""
BRIDGE ML-Engine Standalone API Service

This service runs the ML-engine as a separate API service that can be hosted
independently and connected to the backend via HTTP REST API calls.

Features:
- REST API endpoints for all ML operations
- Session lifecycle management
- Real-time behavioral processing
- Health monitoring and statistics
- Authentication and security
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
import sys
from pathlib import Path

# Add current directory to Python path
sys.path.append(str(Path(__file__).parent))

from core_ml_engine import ml_engine, BehavioralEvent, SessionContext, AuthenticationResponse
from session_integration import ml_session_integrator
from ml_engine.config import CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("🚀 Starting BRIDGE ML-Engine API Service...")
    
    try:
        # Initialize ML-Engine
        if not await ml_engine.initialize():
            logger.error("❌ Failed to initialize ML-Engine")
            raise RuntimeError("ML-Engine initialization failed")
        
        # Initialize session integrator
        if not await ml_session_integrator.initialize():
            logger.error("❌ Failed to initialize session integrator")
            raise RuntimeError("Session integrator initialization failed")
        
        logger.info("✅ ML-Engine API Service started successfully")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down ML-Engine API Service...")
    try:
        await ml_session_integrator.shutdown()
        await ml_engine.shutdown()
        logger.info("✅ ML-Engine API Service shutdown completed")
    except Exception as e:
        logger.error(f"❌ Shutdown error: {e}")

# Create FastAPI app
app = FastAPI(
    title="BRIDGE ML-Engine API",
    description="Industry-Grade Behavioral Authentication ML-Engine",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
from pydantic import BaseModel
from typing import Union

class SessionStartRequest(BaseModel):
    session_id: str
    user_id: str
    device_id: str
    phone: str
    device_type: str = "mobile"
    device_model: str = "unknown"
    os_version: str = "unknown"
    app_version: str = "unknown"
    network_type: str = "unknown"
    location_data: Optional[Dict[str, Any]] = None
    is_known_device: bool = False
    is_trusted_location: bool = False

class BehavioralEventRequest(BaseModel):
    session_id: str
    user_id: str
    device_id: str
    timestamp: str
    event_type: str
    features: Dict[str, float]
    raw_metadata: Dict[str, Any] = {}

class SessionEndRequest(BaseModel):
    session_id: str
    user_id: str
    final_decision: str = "normal"
    session_duration: float = 0
    total_events: int = 0

# API Endpoints

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        stats = await ml_engine.get_engine_stats()
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "ml_engine_initialized": stats.get("is_initialized", False),
            "ml_engine_running": stats.get("is_running", False)
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@app.get("/stats")
async def get_ml_stats():
    """Get comprehensive ML-Engine statistics"""
    try:
        stats = await ml_engine.get_engine_stats()
        return {
            "status": "success",
            "data": stats,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting ML stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get ML statistics")

@app.post("/session/start")
async def start_ml_session(request: SessionStartRequest):
    """Start ML-Engine session"""
    try:
        # Convert request to session context
        context = SessionContext(
            session_id=request.session_id,
            user_id=request.user_id,
            device_id=request.device_id,
            session_start_time=datetime.utcnow(),
            last_activity=datetime.utcnow(),
            session_duration_minutes=0.0,
            device_type=request.device_type,
            device_model=request.device_model,
            os_version=request.os_version,
            app_version=request.app_version,
            network_type=request.network_type,
            location_data=request.location_data,
            time_of_day=_get_time_of_day(),
            usage_pattern="standard",
            interaction_frequency=0.0,
            typical_session_duration=0.0,
            is_known_device=request.is_known_device,
            is_trusted_location=request.is_trusted_location
        )
        
        success = await ml_engine.start_session(request.session_id, request.user_id, context)
        
        if success:
            return {
                "status": "success",
                "message": f"ML session started for {request.session_id}",
                "session_id": request.session_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to start ML session")
            
    except Exception as e:
        logger.error(f"Error starting ML session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start ML session: {str(e)}")

@app.post("/session/process-event")
async def process_behavioral_event(request: BehavioralEventRequest):
    """Process behavioral event through ML-Engine"""
    try:
        # Convert request to behavioral event
        event = BehavioralEvent(
            timestamp=datetime.fromisoformat(request.timestamp.replace('Z', '')),
            event_type=request.event_type,
            features=request.features,
            session_id=request.session_id,
            user_id=request.user_id,
            device_id=request.device_id,
            raw_metadata=request.raw_metadata
        )
        
        # Process through ML-Engine
        response = await ml_engine.process_behavioral_event(event)
        
        if response:
            # Convert response to dictionary
            response_dict = _convert_response_to_dict(response)
            return {
                "status": "success",
                "data": response_dict,
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            return {
                "status": "no_response",
                "message": "No ML response generated",
                "timestamp": datetime.utcnow().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Error processing behavioral event: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process behavioral event: {str(e)}")

@app.post("/session/end")
async def end_ml_session(request: SessionEndRequest):
    """End ML-Engine session"""
    try:
        success = await ml_engine.end_session(request.session_id, request.final_decision)
        
        if success:
            return {
                "status": "success",
                "message": f"ML session ended for {request.session_id}",
                "session_id": request.session_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            raise HTTPException(status_code=404, detail="Session not found")
            
    except Exception as e:
        logger.error(f"Error ending ML session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to end ML session: {str(e)}")

@app.get("/session/{session_id}/status")
async def get_session_status(session_id: str):
    """Get ML-Engine session status"""
    try:
        status = await ml_engine.get_session_status(session_id)
        
        if status:
            return {
                "status": "success",
                "data": status,
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            raise HTTPException(status_code=404, detail="Session not found")
            
    except Exception as e:
        logger.error(f"Error getting session status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get session status: {str(e)}")

@app.post("/alerts")
async def handle_ml_alert(alert_data: Dict[str, Any]):
    """Handle ML-Engine alerts"""
    try:
        logger.info(f"Received ML alert: {alert_data}")
        
        # Process alert (you can extend this to send notifications, etc.)
        return {
            "status": "success",
            "message": "Alert processed",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error processing ML alert: {e}")
        raise HTTPException(status_code=500, detail="Failed to process alert")

# Utility functions

def _get_time_of_day() -> str:
    """Get current time of day classification"""
    hour = datetime.now().hour
    if 6 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 21:
        return "evening"
    else:
        return "night"

def _convert_response_to_dict(response: AuthenticationResponse) -> Dict[str, Any]:
    """Convert AuthenticationResponse to dictionary"""
    try:
        return {
            "session_id": response.session_id,
            "user_id": response.user_id,
            "request_id": response.request_id,
            "decision": response.decision.value,
            "risk_level": response.risk_level.value,
            "risk_score": float(response.risk_score),
            "confidence": float(response.confidence),
            "total_processing_time_ms": float(response.total_processing_time_ms),
            "timestamp": response.timestamp.isoformat(),
            "next_verification_delay": response.next_verification_delay,
            
            "layer1_result": {
                "similarity_score": float(response.layer1_result.similarity_score),
                "confidence_level": response.layer1_result.confidence_level,
                "matched_profile_mode": response.layer1_result.matched_profile_mode,
                "decision": response.layer1_result.decision,
                "processing_time_ms": float(response.layer1_result.processing_time_ms)
            },
            
            "layer2_result": {
                "overall_confidence": float(response.layer2_result.overall_confidence),
                "decision": response.layer2_result.decision,
                "processing_time_ms": float(response.layer2_result.processing_time_ms)
            } if response.layer2_result else None,
            
            "drift_result": {
                "drift_detected": response.drift_result.drift_detected,
                "drift_magnitude": float(response.drift_result.drift_magnitude),
                "drift_type": response.drift_result.drift_type
            } if response.drift_result else None,
            
            "explanation": response.explanation,
            
            "stage_timings": {
                stage.value: float(timing) for stage, timing in response.stage_timings.items()
            }
        }
    except Exception as e:
        logger.error(f"Error converting response to dict: {e}")
        return {}

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='BRIDGE ML-Engine API Service')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8001, help='Port to bind to')
    parser.add_argument('--log-level', default='INFO', help='Log level')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload for development')
    
    args = parser.parse_args()
    
    # Update logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    
    logger.info(f"🚀 Starting BRIDGE ML-Engine API Service on {args.host}:{args.port}")
    
    uvicorn.run(
        "ml_engine_api_service:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
        reload=args.reload
    )

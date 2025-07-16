"""
BRIDGE ML-Engine API Service
FastAPI service for ML-Engine that runs independently from the backend
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
import asyncio
import uvicorn
import logging
from datetime import datetime
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlengine.core.industry_engine import IndustryGradeMLEngine, BehavioralVector, BehavioralEvent, SessionContext
from mlengine.core.industry_engine import RiskLevel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global ML Engine instance
ml_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application lifespan events"""
    global ml_engine
    
    # Startup
    logger.info("🚀 Starting BRIDGE ML-Engine Service...")
    
    try:
        ml_engine = IndustryGradeMLEngine()
        await ml_engine.initialize()
        logger.info("✅ ML-Engine initialized successfully")
        
        # Perform health check
        health = ml_engine.get_performance_stats()
        logger.info(f"📊 ML-Engine Health: {health}")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize ML-Engine: {e}")
        raise e
    
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down ML-Engine Service...")
    if ml_engine:
        try:
            await ml_engine.shutdown()
            logger.info("✅ ML-Engine shutdown complete")
        except Exception as e:
            logger.error(f"❌ Error during ML-Engine shutdown: {e}")

# Pydantic models for API
class BehavioralVectorRequest(BaseModel):
    vector: List[float]
    timestamp: str
    confidence: float
    source: str

class BehavioralEventRequest(BaseModel):
    event_type: str
    timestamp: str
    features: Dict[str, Any]
    session_id: str
    user_id: str

class SessionStartRequest(BaseModel):
    session_id: str
    user_id: str
    phone: str
    device_id: str
    context: Dict[str, Any]

class SessionVerifyRequest(BaseModel):
    session_id: str
    user_id: str
    vectors: List[BehavioralVectorRequest]
    events: List[BehavioralEventRequest]
    context: Dict[str, Any]

class SessionEndRequest(BaseModel):
    session_id: str
    user_id: str

# Global ML-Engine instance
ml_engine: Optional[IndustryGradeMLEngine] = None

app = FastAPI(
    title="BRIDGE ML-Engine API Service",
    description="Banking-grade behavioral verification ML service",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for backend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],  # Backend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not ml_engine:
        raise HTTPException(status_code=503, detail="ML-Engine not initialized")
    
    try:
        health = ml_engine.get_performance_stats()
        is_healthy = ml_engine.is_initialized
        
        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "ml_engine": health
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

@app.post("/sessions/start")
async def start_session(request: SessionStartRequest):
    """Start a new behavioral analysis session"""
    if not ml_engine:
        raise HTTPException(status_code=503, detail="ML-Engine not initialized")
    
    try:
        # Create session context from request
        session_context = SessionContext(
            session_id=request.session_id,
            user_id=request.user_id,
            device_id=request.device_id,
            session_start_time=datetime.now(),
            last_activity=datetime.now(),
            session_duration_minutes=0.0,
            
            # Extract device/environment info from context
            device_type=request.context.get("device_type", "unknown"),
            device_model=request.context.get("device_model", "unknown"),
            os_version=request.context.get("os_version", "unknown"),
            app_version=request.context.get("app_version", "unknown"),
            network_type=request.context.get("network_type", "unknown"),
            location_data=request.context.get("location_data"),
            
            # Behavioral context
            time_of_day=request.context.get("time_of_day", "unknown"),
            usage_pattern=request.context.get("usage_pattern", "normal"),
            interaction_frequency=request.context.get("interaction_frequency", 1.0),
            typical_session_duration=request.context.get("typical_session_duration", 30.0),
            
            # Security context
            is_known_device=request.context.get("is_known_device", False),
            is_trusted_location=request.context.get("is_trusted_location", False),
            recent_security_events=request.context.get("recent_security_events", []),
            current_risk_level=RiskLevel.LOW,  # <-- add this
            account_age_days=request.context.get("account_age_days", 0),  # <-- add this
            transaction_history_risk=request.context.get("transaction_history_risk", 0.0),  # <-- add this
            current_transaction_context=request.context.get("current_transaction_context", {})  # <-- add this
        )
        
        # Track session in ML-Engine
        ml_engine.active_sessions[request.session_id] = session_context
        
        logger.info(f"🎯 Started session {request.session_id} for user {request.user_id}")
        return {
            "status": "success",
            "session_id": request.session_id,
            "message": "Session started successfully",
            "session_context": {
                "user_id": session_context.user_id,
                "device_id": session_context.device_id,
                "session_start_time": session_context.session_start_time.isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Error starting session {request.session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start session: {str(e)}")

@app.post("/sessions/verify")
async def verify_session(request: SessionVerifyRequest):
    """Perform behavioral verification for ongoing session"""
    if not ml_engine:
        raise HTTPException(status_code=503, detail="ML-Engine not initialized")
    
    try:
        # Get session context
        session_context = ml_engine.active_sessions.get(request.session_id)
        if not session_context:
            raise HTTPException(status_code=404, detail=f"Session {request.session_id} not found")
        
        # Update session activity
        session_context.last_activity = datetime.now()
        session_context.session_duration_minutes = (
            session_context.last_activity - session_context.session_start_time
        ).total_seconds() / 60.0
        
        # Convert request data to ML-Engine format
        events = []
        for e in request.events:
            events.append(BehavioralEvent(
                timestamp=datetime.fromisoformat(e.timestamp),
                event_type=e.event_type,
                features=e.features,
                session_id=e.session_id,
                user_id=e.user_id,
                device_id=session_context.device_id,
                raw_metadata={}
            ))
        
        # Process events through ML-Engine
        result = await ml_engine.process_session_events(
            events=events,
            session_context=session_context,
            require_explanation=True
        )
        
        logger.info(f"🔍 Verified session {request.session_id}: {result.decision}")
        
        return {
            "status": "success",
            "session_id": request.session_id,
            "verification_result": {
                "decision": result.decision.value,
                "confidence": result.confidence,
                "risk_score": result.risk_score,
                "explanation": result.explanation,
                "processing_time_ms": result.total_processing_time_ms,
                "layer_results": {
                    "layer1": result.layer1_result.__dict__ if result.layer1_result else None,
                    "layer2": result.layer2_result.__dict__ if result.layer2_result else None
                }
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Error verifying session {request.session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")

@app.post("/sessions/end")
async def end_session(request: SessionEndRequest):
    """End behavioral analysis session"""
    if not ml_engine:
        raise HTTPException(status_code=503, detail="ML-Engine not initialized")
    
    try:
        # Get session context before removing
        session_context = ml_engine.active_sessions.get(request.session_id)
        
        if session_context:
            # Calculate final session duration
            end_time = datetime.now()
            session_duration = (end_time - session_context.session_start_time).total_seconds() / 60.0
            
            session_summary = {
                "session_id": request.session_id,
                "user_id": request.user_id,
                "session_start_time": session_context.session_start_time.isoformat(),
                "session_end_time": end_time.isoformat(),
                "session_duration_minutes": session_duration,
                "device_id": session_context.device_id
            }
            
            # Remove from active sessions
            del ml_engine.active_sessions[request.session_id]
        else:
            session_summary = {
                "session_id": request.session_id,
                "user_id": request.user_id,
                "message": "Session was not found in active sessions"
            }
        
        logger.info(f"🏁 Ended session {request.session_id} for user {request.user_id}")
        
        return {
            "status": "success",
            "session_id": request.session_id,
            "message": "Session ended successfully",
            "session_summary": session_summary
        }
        
    except Exception as e:
        logger.error(f"❌ Error ending session {request.session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to end session: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get ML-Engine statistics"""
    if not ml_engine:
        raise HTTPException(status_code=503, detail="ML-Engine not initialized")
    
    try:
        stats = ml_engine.get_performance_stats()
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "stats": stats
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

if __name__ == "__main__":
    # Configuration from environment
    host = os.getenv("ML_ENGINE_HOST", "0.0.0.0")
    port = int(os.getenv("ML_ENGINE_PORT", "8001"))
    workers = int(os.getenv("ML_ENGINE_WORKERS", "1"))
    
    logger.info(f"🚀 Starting ML-Engine API Service on {host}:{port}")
    
    uvicorn.run(
        "ml_engine_api_service:app",
        host=host,
        port=port,
        workers=workers,
        reload=False,
        access_log=True,
        log_level="info"
    )

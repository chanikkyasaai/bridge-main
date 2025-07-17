"""
FastAPI application for Behavioral Authentication ML Engine
Runs independently from the main backend on port 8001
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import asyncio
import logging
from datetime import datetime
import sys
import os
from contextlib import asynccontextmanager
import numpy as np

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import ML Engine components
from src.core.vector_store import HDF5VectorStore
from src.core.session_manager import SessionManager
from src.data.behavioral_processor import BehavioralProcessor
from src.layers.faiss_layer import FAISSLayer
from src.layers.adaptive_layer import AdaptiveLayer
from src.data.models import BehavioralFeatures, BehavioralVector, AuthenticationDecision
from src.config.settings import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global ML Engine components
settings = get_settings()
vector_store = None
session_manager = None
behavioral_processor = None
faiss_layer = None
adaptive_layer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    global vector_store, session_manager, behavioral_processor, faiss_layer, adaptive_layer
    
    # Startup
    try:
        logger.info("Initializing ML Engine components...")
        
        # Initialize core components in correct order
        vector_store = HDF5VectorStore()
        session_manager = SessionManager(vector_store)  # Pass vector_store
        behavioral_processor = BehavioralProcessor()
        
        # Initialize ML layers
        faiss_layer = FAISSLayer(vector_store)
        adaptive_layer = AdaptiveLayer(vector_store)
        
        logger.info("ML Engine initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize ML Engine: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down ML Engine...")

app = FastAPI(
    title="Behavioral Authentication ML Engine",
    description="Machine Learning engine for real-time behavioral authentication",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],  # Backend server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API
class BehavioralEvent(BaseModel):
    event_type: str
    timestamp: str
    data: Dict[str, Any]

class SessionStartRequest(BaseModel):
    user_id: str
    session_id: str
    device_info: Optional[Dict[str, Any]] = None

class SessionEndRequest(BaseModel):
    session_id: str
    reason: str = "completed"

class BehavioralAnalysisRequest(BaseModel):
    user_id: str
    session_id: str
    events: List[BehavioralEvent]

class AuthenticationFeedback(BaseModel):
    user_id: str
    session_id: str
    decision_id: str
    was_correct: bool
    feedback_source: str = "system"

class MLEngineStatus(BaseModel):
    status: str
    components: Dict[str, bool]
    statistics: Dict[str, Any]

@app.get("/", response_model=MLEngineStatus)
async def health_check():
    """Health check endpoint"""
    try:
        components_status = {
            "vector_store": vector_store is not None,
            "session_manager": session_manager is not None,
            "behavioral_processor": behavioral_processor is not None,
            "faiss_layer": faiss_layer is not None,
            "adaptive_layer": adaptive_layer is not None
        }
        
        # Get statistics
        stats = {}
        if faiss_layer:
            stats["faiss"] = await faiss_layer.get_layer_statistics()
        if adaptive_layer:
            stats["adaptive"] = await adaptive_layer.get_layer_statistics()
        if session_manager:
            stats["sessions"] = await session_manager.get_session_statistics()
        
        return MLEngineStatus(
            status="healthy" if all(components_status.values()) else "degraded",
            components=components_status,
            statistics=stats
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/session/start")
async def start_session(request: SessionStartRequest):
    """Start a new behavioral analysis session"""
    try:
        logger.info(f"Starting session {request.session_id} for user {request.user_id}")
        
        # Create session context
        session_id = await session_manager.create_session(
            user_id=request.user_id,
            device_id=request.device_info.get("device_id") if request.device_info else None,
            user_agent=request.device_info.get("user_agent") if request.device_info else None,
            ip_address=request.device_info.get("ip_address")
        )
        
        return {
            "status": "success",
            "session_id": session_id,
            "user_id": request.user_id,
            "session_phase": "learning",  # Default phase
            "message": "Session started successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to start session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/session/end")
async def end_session(request: SessionEndRequest):
    """End a behavioral analysis session"""
    try:
        logger.info(f"Ending session {request.session_id}")
        
        # Get session statistics before termination
        session_context = session_manager.get_session_context(request.session_id)
        if session_context:
            stats = {
                "total_events": len(session_context.behavioral_events),
                "risk_scores": session_context.risk_scores,
                "session_duration": (datetime.utcnow() - session_context.created_at).total_seconds()
            }
        else:
            stats = {}
        
        # Terminate session
        await session_manager.terminate_session(request.session_id, request.reason)
        
        return {
            "status": "success",
            "session_id": request.session_id,
            "session_statistics": stats,
            "message": "Session ended successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to end session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze")
async def analyze_behavior(request: BehavioralAnalysisRequest):
    """Analyze behavioral data and make authentication decision"""
    try:
        logger.info(f"Analyzing behavior for session {request.session_id}")
        
        # Convert events to BehavioralFeatures format
        # For now, create a simple behavioral features object from events
        from src.data.models import BehavioralFeatures
        
        # Extract basic features from events (simplified for demo)
        now = datetime.utcnow()

        # --- Aggregate features from events ---
        # Typing features
        typing_speeds = []
        keystroke_intervals = []
        typing_rhythm_variances = []
        backspace_counts = []
        typing_pressures = []
        typing_areas = []
        for e in request.events:
            if e.event_type == "typing_pattern":
                data = e.data
                if "typing_speed" in data:
                    typing_speeds.append(float(data["typing_speed"]))
                if "keystroke_dynamics" in data:
                    keystroke_intervals.extend([float(x) for x in data["keystroke_dynamics"]])
                if "average_delay" in data:
                    typing_rhythm_variances.append(float(data["average_delay"]))
                if "delete_count" in data and "keystroke_count" in data and data["keystroke_count"]:
                    backspace_counts.append(float(data["delete_count"]) / float(data["keystroke_count"]))
                if "touch_pressure" in data:
                    typing_pressures.append(float(data["touch_pressure"]))
                if "touch_area" in data:
                    typing_areas.append(float(data["touch_area"]))

        # Touch features
        touch_pressures = []
        touch_durations = []
        touch_areas = []
        touch_coordinates = []
        for e in request.events:
            if e.event_type == "touch_down":
                data = e.data
                if "pressure" in data:
                    touch_pressures.append(float(data["pressure"]))
                if "coordinates" in data and isinstance(data["coordinates"], list) and len(data["coordinates"]) == 2:
                    touch_coordinates.append({"x": float(data["coordinates"][0]), "y": float(data["coordinates"][1])})
            if e.event_type == "touch_up":
                data = e.data
                if "touch_duration_ms" in data:
                    touch_durations.append(float(data["touch_duration_ms"]))
                if "coordinates" in data and isinstance(data["coordinates"], list) and len(data["coordinates"]) == 2:
                    touch_coordinates.append({"x": float(data["coordinates"][0]), "y": float(data["coordinates"][1])})
            # Optionally, add area if present
                if "touch_area" in data:
                    touch_areas.append(float(data["touch_area"]))

        # Navigation features
        navigation_patterns = []
        screen_time_distribution = {}
        for e in request.events:
            if e.event_type == "navigation_pattern":
                data = e.data
                if "route" in data:
                    navigation_patterns.append(str(data["route"]))
                if "duration" in data and "route" in data:
                    screen = str(data["route"])
                    screen_time_distribution[screen] = screen_time_distribution.get(screen, 0.0) + float(data["duration"])

        # Contextual features
        device_orientation = "portrait"
        for e in request.events:
            if e.event_type == "orientation_change":
                data = e.data
                if "orientation" in data:
                    device_orientation = str(data["orientation"])
        time_of_day = now.hour
        day_of_week = now.weekday()
        app_version = "1.0.0"

        # Calculate aggregate values or use defaults
        typing_speed = float(np.mean(typing_speeds)) if typing_speeds else 0.0
        typing_rhythm_variance = float(np.var(keystroke_intervals)) if keystroke_intervals else 0.0
        backspace_frequency = float(np.mean(backspace_counts)) if backspace_counts else 0.0
        typing_pressure = typing_pressures or [0.0]
        touch_pressure = touch_pressures or [0.0]
        touch_duration = touch_durations or [0.0]
        touch_area = typing_areas + touch_areas or [0.0]
        swipe_velocity = [0.0]  # Not tracked in frontend, set to default
        touch_coordinates = touch_coordinates or [{"x": 0.0, "y": 0.0}]
        navigation_patterns = navigation_patterns or ["unknown"]
        screen_time_distribution = screen_time_distribution or {"unknown": 0.0}
        interaction_frequency = float(len(request.events)) / 300.0  # crude default
        session_duration = 300.0  # Could be improved if session start/end events are present

        behavioral_features_data = BehavioralFeatures(
            typing_speed=typing_speed,
            keystroke_intervals=keystroke_intervals or [0.0],
            typing_rhythm_variance=typing_rhythm_variance,
            backspace_frequency=backspace_frequency,
            typing_pressure=typing_pressure,
            touch_pressure=touch_pressure,
            touch_duration=touch_duration,
            touch_area=touch_area,
            swipe_velocity=swipe_velocity,
            touch_coordinates=touch_coordinates,
            navigation_patterns=navigation_patterns,
            screen_time_distribution=screen_time_distribution,
            interaction_frequency=interaction_frequency,
            session_duration=session_duration,
            device_orientation=device_orientation,
            time_of_day=time_of_day,
            day_of_week=day_of_week,
            app_version=app_version
        )
        
        # Process behavioral events
        behavioral_vector = await behavioral_processor.process_behavioral_data(
            behavioral_data=behavioral_features_data,
            user_id=request.user_id,
            session_id=request.session_id
        )
        
        if not behavioral_vector:
            return {
                "status": "insufficient_data",
                "decision": "learn",
                "confidence": 0.0,
                "message": "Insufficient behavioral data for analysis"
            }
        
        # Create behavioral vector for storage
        behavioral_vector_obj = BehavioralVector(
            user_id=request.user_id,
            session_id=request.session_id,
            vector=behavioral_vector.vector,
            confidence_score=0.8,  # Default confidence
            feature_source=behavioral_features_data  # Pass the BehavioralFeatures object
        )
        
        # Store vector
        await vector_store.store_vector(request.user_id, behavioral_vector_obj)
        
        # Get user profile
        user_profile = await vector_store.get_user_profile(request.user_id)
        
        # Make authentication decision using FAISS layer
        auth_result_tuple = await faiss_layer.make_authentication_decision(
            user_id=request.user_id,
            query_vector=behavioral_vector_obj,
            user_profile=user_profile
        )
        
        # Unpack the tuple result
        decision, risk_level, confidence, similarity_score, risk_factors = auth_result_tuple
        
        # Adaptive threshold adjustment
        adaptive_threshold = await adaptive_layer.get_adaptive_threshold(
            user_id=request.user_id,
            base_threshold=settings.similarity_threshold
        )
        
        decision_id = f"decision_{request.session_id}_{datetime.utcnow().isoformat()}"
        
        return {
            "status": "success",
            "decision_id": decision_id,
            "decision": decision.value if hasattr(decision, 'value') else str(decision),
            "confidence": confidence,
            "similarity_score": similarity_score,
            "threshold_used": adaptive_threshold,
            "session_phase": user_profile.current_phase.value,
            "risk_factors": risk_factors or [],
            "processing_time_ms": 50  # Default processing time
        }
        
    except Exception as e:
        logger.error(f"Behavioral analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def submit_feedback(request: AuthenticationFeedback):
    """Submit feedback for model improvement"""
    try:
        logger.info(f"Receiving feedback for decision {request.decision_id}")
        
        # Get the behavioral vector and decision for this feedback
        session_context = session_manager.get_session_context(request.session_id)
        if not session_context:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Find the corresponding behavioral vector (simplified - in production, store decision-vector mapping)
        user_vectors = await vector_store.get_user_vectors(request.user_id, limit=1)
        if user_vectors:
            behavioral_vector = user_vectors[0]
            
            # Submit to adaptive layer
            await adaptive_layer.learn_from_authentication(
                user_id=request.user_id,
                behavioral_vector=behavioral_vector,
                decision=AuthenticationDecision.ALLOW if request.was_correct else AuthenticationDecision.BLOCK,
                was_correct=request.was_correct,
                confidence=0.8,  # Default confidence
                context={"feedback_source": request.feedback_source}
            )
        
        return {
            "status": "success",
            "message": "Feedback submitted successfully"
        }
        
    except Exception as e:
        logger.error(f"Feedback submission failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/statistics")
async def get_statistics():
    """Get ML Engine statistics"""
    try:
        stats = {
            "faiss_layer": await faiss_layer.get_layer_statistics() if faiss_layer else {},
            "adaptive_layer": await adaptive_layer.get_layer_statistics() if adaptive_layer else {},
            "session_manager": await session_manager.get_session_statistics() if session_manager else {},
            "vector_store": await vector_store.get_storage_stats() if vector_store else {}
        }
        
        return {
            "status": "success",
            "statistics": stats,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "ml_engine_api_service:app",
        host="127.0.0.1",
        port=8001,
        reload=True,
        log_level="info"
    )

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
import math
import json
from datetime import datetime
import sys
import os
from contextlib import asynccontextmanager
import numpy as np

# Add the current directory to the Python path to enable src imports
sys.path.append(os.path.dirname(__file__))

def make_json_safe(obj):
    """Make any object JSON-serializable by replacing NaN/infinity values"""
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(item) for item in obj]
    elif isinstance(obj, float):
        if math.isnan(obj):
            return 0.0
        elif math.isinf(obj):
            return 1.0 if obj > 0 else -1.0
        else:
            return obj
    else:
        return obj

# Import ML Engine components
from src.core.vector_store import HDF5VectorStore
from src.core.session_manager import SessionManager
from src.core.learning_system import Phase1LearningSystem
from src.core.continuous_analysis import Phase2ContinuousAnalysis
from src.core.ml_database import ml_db
from src.data.behavioral_processor import BehavioralProcessor
from src.layers.faiss_layer import FAISSLayer
from src.layers.adaptive_layer import AdaptiveLayer
from src.core.enhanced_faiss_engine import EnhancedFAISSEngine
from src.core.enhanced_behavioral_processor import EnhancedBehavioralProcessor
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
learning_system = None
continuous_analysis = None
enhanced_faiss_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    global vector_store, session_manager, behavioral_processor, faiss_layer, adaptive_layer, learning_system, continuous_analysis, enhanced_faiss_engine
    
    # Startup
    try:
        logger.info("Initializing ML Engine components...")
        
        # Test database connectivity first
        db_health = await ml_db.health_check()
        if not db_health:
            logger.error("Database connectivity failed - continuing with local storage only")
        else:
            logger.info("Database connectivity confirmed")
        
        # Initialize core components in correct order
        vector_store = HDF5VectorStore()
        session_manager = SessionManager(vector_store)  # Pass vector_store
        behavioral_processor = BehavioralProcessor()
        
        # Initialize Enhanced FAISS Engine
        enhanced_faiss_engine = EnhancedFAISSEngine(vector_dimension=90)
        await enhanced_faiss_engine.initialize()
        
        # Initialize ML layers (legacy support)
        faiss_layer = FAISSLayer(vector_store)
        adaptive_layer = AdaptiveLayer(vector_store)
        
        # Initialize Phase 1 Learning System
        learning_system = Phase1LearningSystem(vector_store, behavioral_processor)
        
        # Initialize Phase 2 Continuous Analysis
        continuous_analysis = Phase2ContinuousAnalysis(vector_store, faiss_layer, adaptive_layer)
        
        logger.info("ML Engine initialized successfully with Enhanced FAISS Engine & Phase 1 & 2 systems")
        
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

class MobileBehavioralDataRequest(BaseModel):
    user_id: str
    session_id: str
    logs: List[Dict[str, Any]]  # Raw behavioral logs from mobile

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
    """Health check endpoint with Enhanced FAISS Engine & Phase 1 & 2 systems"""
    try:
        components_status = {
            "vector_store": vector_store is not None,
            "session_manager": session_manager is not None,
            "behavioral_processor": behavioral_processor is not None,
            "faiss_layer": faiss_layer is not None,
            "adaptive_layer": adaptive_layer is not None,
            "learning_system": learning_system is not None,
            "continuous_analysis": continuous_analysis is not None,
            "enhanced_faiss_engine": enhanced_faiss_engine is not None,
            "database": await ml_db.health_check()
        }
        
        # Get comprehensive statistics
        stats = {}
        if faiss_layer:
            stats["faiss"] = await faiss_layer.get_layer_statistics()
        if adaptive_layer:
            stats["adaptive"] = await adaptive_layer.get_layer_statistics()
        if session_manager:
            stats["sessions"] = await session_manager.get_session_statistics()
        if enhanced_faiss_engine:
            stats["enhanced_faiss"] = await enhanced_faiss_engine.get_layer_statistics()
        if learning_system:
            stats["learning"] = await learning_system.get_learning_statistics()
        if continuous_analysis:
            stats["analysis"] = await continuous_analysis.get_analysis_statistics()
        
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
    """Start a new behavioral analysis session with Phase 1 learning integration"""
    try:
        logger.info(f"Starting session {request.session_id} for user {request.user_id}")
        
        # Phase 1: Handle learning system for new session
        learning_phase, session_guidance = await learning_system.handle_new_session(
            request.user_id, request.session_id
        )
        
        # Create session context
        session_id = await session_manager.create_session(
            user_id=request.user_id,
            device_id=request.device_info.get("device_id") if request.device_info else None,
            user_agent=request.device_info.get("user_agent") if request.device_info else None,
            ip_address=request.device_info.get("ip_address")
        )
        
        # Get user profile for session context
        user_profile = await ml_db.get_user_profile(request.user_id)
        
        response = {
            "status": "success",
            "session_id": session_id,
            "user_id": request.user_id,
            "learning_phase": learning_phase.value,
            "session_guidance": session_guidance,
            "message": "Session started successfully with learning system integration"
        }
        
        # Add user profile info if available
        if user_profile:
            response["user_profile"] = {
                "current_phase": user_profile["current_phase"],
                "session_count": user_profile["current_session_count"],
                "total_sessions": user_profile["total_sessions"],
                "risk_score": user_profile["risk_score"]
            }
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to start session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/session/end")
async def end_session(request: SessionEndRequest):
    """End a behavioral analysis session with cumulative vector update"""
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
            
            # CRITICAL: Update cumulative vector at session end
            if session_context.behavioral_events:
                try:
                    # Process all behavioral events from this session into a final session vector
                    if hasattr(request, 'user_id') and request.user_id:
                        user_id = request.user_id
                    else:
                        # Extract user_id from session context
                        user_id = session_context.user_id
                    
                    # Get the latest session vector for this session from database
                    session_vectors_result = ml_db.supabase.table('enhanced_behavioral_vectors')\
                        .select('*')\
                        .eq('session_id', request.session_id)\
                        .eq('vector_type', 'session')\
                        .order('created_at', desc=True)\
                        .limit(1)\
                        .execute()
                    
                    if session_vectors_result.data:
                        session_vector_data = session_vectors_result.data[0]['vector_data']
                        session_vector = np.array(session_vector_data)
                        
                        # Update cumulative vector using enhanced FAISS engine
                        if enhanced_faiss_engine:
                            await enhanced_faiss_engine._update_cumulative_vector(
                                user_id, 
                                session_vector, 
                                "allow"  # Assume successful session completion
                            )
                            
                            # Check if user should transition learning phases
                            await enhanced_faiss_engine._check_learning_phase_transition(user_id)
                            
                            logger.info(f"Updated cumulative vector for user {user_id} at session end")
                        
                except Exception as e:
                    logger.warning(f"Failed to update cumulative vector at session end: {e}")
        else:
            stats = {}
        
        # Terminate session
        await session_manager.terminate_session(request.session_id, request.reason)
        
        return {
            "status": "success",
            "session_id": request.session_id,
            "session_statistics": stats,
            "message": "Session ended successfully with cumulative update"
        }
        
    except Exception as e:
        logger.error(f"Failed to end session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-mobile")
async def analyze_mobile_behavioral_data(request: MobileBehavioralDataRequest):
    """
    Enhanced endpoint for processing mobile behavioral data
    Uses Enhanced FAISS Engine with proper vector embeddings
    """
    try:
        logger.info(f"Processing mobile behavioral data for session {request.session_id}")
        
        # Ensure session exists in SessionManager
        try:
            session_context = session_manager.get_session_context(request.session_id)
            if not session_context:
                # Create session in SessionManager with the specific session_id
                created_session_id = await session_manager.create_session(
                    user_id=request.user_id,
                    session_id=request.session_id
                )
                logger.info(f"Created SessionManager session {created_session_id} for mobile data analysis")
        except Exception as e:
            logger.warning(f"SessionManager session creation failed: {e} - continuing with analysis")
        
        # Get user profile for learning phase determination
        user_profile = await ml_db.get_user_profile(request.user_id)
        learning_phase = user_profile.get('current_phase', 'learning') if user_profile else 'learning'
        
        # Process using Enhanced FAISS Engine with mobile data format
        analysis_result = await enhanced_faiss_engine.process_mobile_behavioral_data(
            user_id=request.user_id,
            session_id=request.session_id,
            behavioral_data={
                "user_id": request.user_id,
                "session_id": request.session_id,
                "logs": request.logs
            }
        )
        
        # Create response
        response = {
            "status": "success",
            "decision": analysis_result.decision,
            "confidence": analysis_result.confidence,
            "risk_score": 1.0 - analysis_result.similarity_score,  # Convert similarity to risk
            "risk_level": analysis_result.risk_level,
            "similarity_score": analysis_result.similarity_score,
            "analysis_type": "enhanced_faiss",
            "processing_time_ms": 0,  # Will be calculated
            "risk_factors": analysis_result.risk_factors,
            "similar_vectors": analysis_result.similar_vectors,
            "vector_id": analysis_result.vector_id,
            "learning_phase": learning_phase,
            "message": f"Enhanced FAISS analysis - {analysis_result.decision} decision"
        }
        
        # Add vector details if available (for debugging)
        if hasattr(analysis_result, 'vector_stats') and analysis_result.vector_stats:
            response["vector_stats"] = analysis_result.vector_stats
        
        if hasattr(analysis_result, 'session_vector') and analysis_result.session_vector:
            # Include only first 10 values of vector for debugging (to avoid huge responses)
            response["vector_sample"] = analysis_result.session_vector[:10]
        
        # Store session behavioral event
        try:
            await session_manager.add_behavioral_event(
                request.session_id,
                {
                    "logs": request.logs,
                    "analysis_result": response,
                    "timestamp": datetime.utcnow().isoformat(),
                    "analysis_engine": "enhanced_faiss"
                }
            )
        except Exception as e:
            logger.warning(f"Failed to store session event: {e}")
        
        # Make response JSON-safe
        safe_response = make_json_safe(response)
        return safe_response
        
    except Exception as e:
        logger.error(f"Failed to analyze mobile behavioral data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze")
async def analyze_behavior(request: BehavioralAnalysisRequest):
    """Analyze behavioral data using Phase 1 Learning + Phase 2 Continuous Analysis"""
    try:
        logger.info(f"Analyzing behavior for session {request.session_id}")
        
        # Ensure session exists in both database and SessionManager
        from src.core.ml_database import ml_db
        
        # 1. Ensure session exists in database
        try:
            db_session_id = await ml_db.create_session(
                user_id=request.user_id,
                session_name=request.session_id,
                device_info="ML Analysis Session"
            )
            if db_session_id:
                logger.info(f"Created database session {db_session_id} for ML session {request.session_id}")
            else:
                logger.warning(f"Session {request.session_id} may already exist in database")
        except Exception as e:
            logger.warning(f"Session creation/check failed: {e} - continuing with analysis")
        
        # 2. Ensure session exists in SessionManager
        try:
            session_context = session_manager.get_session_context(request.session_id)
            if not session_context:
                # Create session in SessionManager with the specific session_id
                created_session_id = await session_manager.create_session(
                    user_id=request.user_id,
                    session_id=request.session_id,  # Use the specific session_id from request
                    device_id=getattr(request, 'device_id', None),
                    ip_address=getattr(request, 'ip_address', None),
                    user_agent=getattr(request, 'user_agent', None)
                )
                logger.info(f"Created SessionManager session {created_session_id} for session {request.session_id}")
            else:
                logger.debug(f"Session {request.session_id} already exists in SessionManager")
        except Exception as e:
            logger.warning(f"SessionManager session creation failed: {e} - continuing with analysis")
        
        # Convert events to BehavioralFeatures format
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
        
        # Process behavioral events to create vector
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
        
        # Get user profile to determine current phase
        user_profile = await ml_db.get_user_profile(request.user_id)
        if not user_profile:
            # Create user profile if it doesn't exist
            user_profile = await ml_db.create_user_profile(request.user_id)
        
        # Convert to UserProfile object for analysis
        from src.data.models import UserProfile, LearningPhase
        user_profile_obj = UserProfile(
            user_id=request.user_id,
            current_phase=LearningPhase(user_profile['current_phase']),
            behavioral_vectors=[],  # Will be loaded as needed
            risk_score=user_profile.get('risk_score', 0.0),
            confidence_score=0.5,  # Default
            last_updated=datetime.utcnow(),
            creation_date=datetime.fromisoformat(user_profile['created_at'].replace('Z', '+00:00'))
        )
        
        # Determine analysis approach based on user phase
        current_phase = user_profile['current_phase']
        
        if current_phase in ['cold_start', 'learning']:
            # Phase 1: Learning System Processing
            logger.info(f"Using Phase 1 Learning System for user {request.user_id}")
            
            # Process vector through learning system
            learning_result = await learning_system.process_behavioral_vector(
                request.user_id, request.session_id, behavioral_vector
            )
            
            # Evaluate learning progress
            progress_report = await learning_system.evaluate_learning_progress(request.user_id)
            
            response = {
                "status": "success",
                "decision": "learn",
                "confidence": learning_result.get('phase_confidence', 0.5),
                "risk_score": 0.1,  # Low risk during learning
                "risk_level": "low",
                "analysis_type": "phase1_learning",
                "learning_result": learning_result,
                "progress_report": progress_report,
                "message": f"Learning phase - {learning_result.get('vectors_collected', 0)} vectors collected"
            }
            
        else:
            # Phase 2: Continuous Analysis System
            logger.info(f"Using Phase 2 Continuous Analysis for user {request.user_id}")
            
            # Run full continuous analysis
            analysis_result = await continuous_analysis.analyze_behavioral_vector(
                request.user_id, request.session_id, behavioral_vector, user_profile_obj
            )
            
            response = {
                "status": "success",
                "decision": analysis_result.decision.value,
                "confidence": analysis_result.confidence,
                "risk_score": analysis_result.risk_score,
                "risk_level": analysis_result.risk_level.value,
                "analysis_type": "phase2_continuous",
                "analysis_level": analysis_result.analysis_level.value,
                "processing_time_ms": analysis_result.processing_time_ms,
                "similarity_scores": analysis_result.similarity_scores,
                "risk_factors": analysis_result.risk_factors,
                "layer_decisions": analysis_result.layer_decisions,
                "message": f"Continuous analysis - {analysis_result.decision.value} decision"
            }
            
            # Add drift analysis if present
            if analysis_result.drift_analysis:
                response["drift_analysis"] = {
                    "drift_type": analysis_result.drift_analysis.drift_type.value,
                    "severity": analysis_result.drift_analysis.severity,
                    "confidence": analysis_result.drift_analysis.confidence,
                    "risk_assessment": analysis_result.drift_analysis.risk_assessment.value,
                    "recommended_action": analysis_result.drift_analysis.recommended_action
                }
        
        # Store session behavioral event
        try:
            await session_manager.add_behavioral_event(
                request.session_id,
                {
                    "events": [event.dict() for event in request.events],
                    "analysis_result": response,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
        except Exception as e:
            logger.warning(f"Failed to store session event: {e}")
        
        # Make response JSON-safe before returning
        safe_response = make_json_safe(response)
        return safe_response
        
    except Exception as e:
        logger.error(f"Failed to analyze behavior: {e}")
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
    """Get comprehensive ML Engine statistics"""
    try:
        stats = {
            "faiss_layer": await faiss_layer.get_layer_statistics() if faiss_layer else {},
            "adaptive_layer": await adaptive_layer.get_layer_statistics() if adaptive_layer else {},
            "session_manager": await session_manager.get_session_statistics() if session_manager else {},
            "vector_store": await vector_store.get_storage_stats() if vector_store else {},
            "learning_system": await learning_system.get_learning_statistics() if learning_system else {},
            "continuous_analysis": await continuous_analysis.get_analysis_statistics() if continuous_analysis else {},
            "database": await ml_db.get_database_stats()
        }
        
        return {
            "status": "success",
            "statistics": stats,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user/{user_id}/learning-progress")
async def get_user_learning_progress(user_id: str):
    """Get detailed learning progress for a specific user"""
    try:
        if not learning_system:
            raise HTTPException(status_code=503, detail="Learning system not available")
        
        # Get learning progress
        progress_report = await learning_system.evaluate_learning_progress(user_id)
        
        # Get user profile from database
        user_profile = await ml_db.get_user_profile(user_id)
        
        # Get recent authentication decisions
        recent_decisions = await ml_db.get_recent_decisions(user_id, limit=10)
        
        response = {
            "status": "success",
            "user_id": user_id,
            "progress_report": progress_report,
            "user_profile": user_profile,
            "recent_decisions": recent_decisions,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Make response JSON-safe before returning
        safe_response = make_json_safe(response)
        return safe_response
        
    except Exception as e:
        logger.error(f"Failed to get learning progress for {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/user/{user_id}/adapt-baseline")
async def adapt_user_baseline(user_id: str):
    """Manually trigger baseline adaptation for a user"""
    try:
        if not continuous_analysis:
            raise HTTPException(status_code=503, detail="Continuous analysis system not available")
        
        # Adapt user baseline
        success = await continuous_analysis.adapt_user_baseline(user_id)
        
        if success:
            return {
                "status": "success",
                "message": f"Baseline adapted successfully for user {user_id}",
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            return {
                "status": "failed",
                "message": f"Failed to adapt baseline for user {user_id} - insufficient data",
                "timestamp": datetime.utcnow().isoformat()
            }
        
    except Exception as e:
        logger.error(f"Failed to adapt baseline for {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health/database")
async def check_database_health():
    """Check database connectivity and health"""
    try:
        db_health = await ml_db.health_check()
        db_stats = await ml_db.get_database_stats()
        
        return {
            "status": "healthy" if db_health else "unhealthy",
            "connectivity": db_health,
            "statistics": db_stats,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            "status": "error",
            "connectivity": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

if __name__ == "__main__":
    uvicorn.run(
        "ml_engine_api_service:app",
        host="127.0.0.1",
        port=8001,
        reload=True,
        log_level="info"
    )

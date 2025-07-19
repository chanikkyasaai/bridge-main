"""
ML Engine API Server for Behavioral Authentication
Handles continuous authentication using FAISS vector similarity
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import uvicorn
import logging
import asyncio
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from learning_manager import LearningManager
from auth_manager import AuthenticationManager, ContinuousAuthenticator
from feature_extractor import FeatureExtractor
from database import DatabaseManager
from bot_detector import BotDetector
from faiss.vector_store import FAISSVectorStore, VectorStorageManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ML Engine API with FAISS", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
db_manager = DatabaseManager()
feature_extractor = FeatureExtractor(vector_dimensions=48)
bot_detector = BotDetector()

# Initialize FAISS components
faiss_store = FAISSVectorStore(vector_dim=48, storage_path="./faiss_data")
vector_storage_manager = VectorStorageManager(faiss_store, db_manager)

# Initialize managers with FAISS integration
learning_manager = LearningManager(db_manager, feature_extractor, vector_storage_manager, bot_detector)
auth_manager = AuthenticationManager(db_manager, feature_extractor, vector_storage_manager, faiss_store, bot_detector)
continuous_authenticator = ContinuousAuthenticator(learning_manager, auth_manager, db_manager)

# Active sessions tracking
active_sessions = {}

class SessionStart(BaseModel):
    user_id: str
    session_id: str
    device_info: Optional[Dict] = None

class SessionEnd(BaseModel):
    session_id: str
    reason: str = "completed"

class BehavioralData(BaseModel):
    user_id: str
    session_id: str
    logs: List[Dict[str, Any]]

class FeedbackData(BaseModel):
    user_id: str
    session_id: str
    decision_id: str
    was_correct: bool
    feedback_source: str = "system"

@app.get("/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "database": "connected",
            "feature_extractor": "ready",
            "learning_manager": "ready",
            "auth_manager": "ready"
        },
        "statistics": {
            "active_sessions": len(active_sessions),
            "total_users": await db_manager.get_total_users(),
            "total_sessions": await db_manager.get_total_sessions()
        }
    }

@app.post("/session/start")
async def start_session(data: SessionStart):
    """Start ML analysis session"""
    try:
        logger.info(f"Starting session {data.session_id} for user {data.user_id}")
        
        # Get user session count
        session_count = await db_manager.get_user_session_count(data.user_id)
        
        # Determine phase
        phase = "learning" if session_count < 6 else "authentication"
        
        # Initialize session state
        active_sessions[data.session_id] = {
            "user_id": data.user_id,
            "session_id": data.session_id,
            "phase": phase,
            "started_at": datetime.utcnow(),
            "device_info": data.device_info or {},
            "events_buffer": [],
            "vectors": [],
            "last_analysis": None
        }
        
        logger.info(f"Session {data.session_id} started in {phase} phase")
        
        return {
            "status": "success",
            "session_id": data.session_id,
            "phase": phase,
            "message": f"Session started in {phase} mode"
        }
        
    except Exception as e:
        logger.error(f"Failed to start session {data.session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/session/end")
async def end_session(data: SessionEnd):
    """End ML analysis session"""
    try:
        logger.info(f"Ending session {data.session_id}")
        
        if data.session_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session_data = active_sessions[data.session_id]
        
        # Use continuous authenticator for session end processing
        end_result = await continuous_authenticator.end_session_processing(session_data)
        
        # Remove from active sessions
        del active_sessions[data.session_id]
        
        logger.info(f"Session {data.session_id} ended successfully")
        
        return {
            "status": "success",
            "message": "Session ended successfully",
            "end_result": end_result
        }
        
    except Exception as e:
        logger.error(f"Failed to end session {data.session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-mobile")
async def analyze_behavior(data: BehavioralData, background_tasks: BackgroundTasks):
    """Analyze behavioral data from mobile app using continuous authenticator"""
    try:
        logger.info(f"Analyzing {len(data.logs)} events for session {data.session_id}")
        
        # Ensure session exists
        if data.session_id not in active_sessions:
            session_count = await db_manager.get_user_session_count(data.user_id)
            phase = "learning" if session_count < 6 else "authentication"
            
            active_sessions[data.session_id] = {
                "user_id": data.user_id,
                "session_id": data.session_id,
                "phase": phase,
                "started_at": datetime.utcnow(),
                "events_buffer": [],
                "vectors": [],
                "last_analysis": datetime.utcnow()
            }
        
        session_data = active_sessions[data.session_id]
        
        # Use continuous authenticator for processing
        result = await continuous_authenticator.process_continuous_authentication(
            data.user_id, data.session_id, data.logs, session_data
        )
        
        # Update last analysis time
        session_data["last_analysis"] = datetime.utcnow()
        
        logger.info(f"Analysis result for {data.session_id}: {result['decision']} (confidence: {result.get('confidence', 0):.3f})")
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to analyze behavior for session {data.session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def submit_feedback(data: FeedbackData):
    """Submit feedback for model improvement"""
    try:
        logger.info(f"Received feedback for session {data.session_id}")
        
        # Store feedback in database
        await db_manager.store_feedback(
            data.user_id,
            data.session_id,
            data.decision_id,
            data.was_correct,
            data.feedback_source
        )
        
        return {
            "status": "success",
            "message": "Feedback stored successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to store feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/statistics")
async def get_statistics():
    """Get ML Engine statistics"""
    try:
        stats = await db_manager.get_system_statistics()
        
        return {
            "status": "success",
            "statistics": {
                **stats,
                "active_sessions": len(active_sessions),
                "active_sessions_list": list(active_sessions.keys())
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user/{user_id}/profile")
async def get_user_profile(user_id: str):
    """Get user behavioral profile"""
    try:
        profile = await db_manager.get_user_profile(user_id)
        
        return {
            "status": "success",
            "profile": profile
        }
        
    except Exception as e:
        logger.error(f"Failed to get user profile for {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)

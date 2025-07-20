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
import httpx

# Load environment variables
load_dotenv()

from learning_manager import LearningManager
from auth_manager import AuthenticationManager, ContinuousAuthenticator
from feature_extractor import FeatureExtractor
from database import DatabaseManager
from bot_detector import BotDetector
from faiss.vector_store import FAISSVectorStore, VectorStorageManager
from gnn_escalation import detect_anomaly_with_user_adaptation, CachingManager, SupabaseStorageManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ML Engine API with FAISS", version="1.0.0", description="Behavioral authentication with FAISS, user-specific threshold, and Level 2 GNN-based escalation for anomaly detection.")

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

# Initialize GNN escalation components with caching
gnn_caching_manager = CachingManager(max_cache_size=2000)
gnn_storage_manager = SupabaseStorageManager(gnn_caching_manager)

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

class GNNAnomalyRequest(BaseModel):
    user_id: str
    session_data: Dict[str, Any]

class BehavioralDriftTracker:
    """Monitors device, location, and network drift for a session."""
    def __init__(self):
        self.session_states = {}  # session_id -> last device_info

    def process_device_info(self, session_id, user_id, event):
        device_data = event.get('data', {})
        timestamp = event.get('timestamp')
        state = self.session_states.get(session_id)
        log_prefix = f"[DriftTracker][Session {session_id}][User {user_id}]"

        # Extract relevant fields
        device_id = device_data.get('device_id')
        device_model = device_data.get('device_model')
        os_version = device_data.get('os_version')
        ip_address = device_data.get('ip_address')
        location = device_data.get('location_data', {})
        latitude = location.get('latitude')
        longitude = location.get('longitude')

        # If this is the first device_info, just store it
        if not state:
            self.session_states[session_id] = {
                'device_id': device_id,
                'device_model': device_model,
                'os_version': os_version,
                'ip_address': ip_address,
                'latitude': latitude,
                'longitude': longitude,
                'timestamp': timestamp
            }
            logger.info(f"{log_prefix} Initial device_info received.")
            return None

        # Check for device change
        if device_id and device_id != state['device_id']:
            logger.warning(f"{log_prefix} Device ID changed mid-session! Blocking session.")
            return 'block'
        if device_model and device_model != state['device_model']:
            logger.warning(f"{log_prefix} Device model changed mid-session! Blocking session.")
            return 'block'
        if os_version and os_version != state['os_version']:
            logger.warning(f"{log_prefix} OS version changed mid-session! Blocking session.")
            return 'block'

        # Check for impossible travel
        if latitude is not None and longitude is not None and state['latitude'] is not None and state['longitude'] is not None:
            from geopy.distance import geodesic
            try:
                prev_coords = (state['latitude'], state['longitude'])
                curr_coords = (latitude, longitude)
                dist_km = geodesic(prev_coords, curr_coords).km
                # Calculate time difference in seconds
                from dateutil.parser import parse as parse_dt
                t1 = parse_dt(state['timestamp'])
                t2 = parse_dt(timestamp)
                time_diff = abs((t2 - t1).total_seconds())
                # If distance > 100km and time < 5min, block
                if dist_km > 100 and time_diff < 300:
                    logger.warning(f"{log_prefix} Impossible travel detected ({dist_km:.1f}km in {time_diff:.1f}s)! Blocking session.")
                    return 'block'
            except Exception as e:
                logger.error(f"{log_prefix} Error in travel check: {e}")

        # Check for IP change
        if ip_address and ip_address != state['ip_address']:
            logger.warning(f"{log_prefix} IP address changed from {state['ip_address']} to {ip_address}. Forcing re-authentication.")
            self.session_states[session_id]['ip_address'] = ip_address
            return 'reauth'

        # Update state
        self.session_states[session_id].update({
            'device_id': device_id,
            'device_model': device_model,
            'os_version': os_version,
            'ip_address': ip_address,
            'latitude': latitude,
            'longitude': longitude,
            'timestamp': timestamp
        })
        return None

# Instantiate the drift tracker
drift_tracker = BehavioralDriftTracker()

class AuditExplainabilityEngine:
    """Logs similarity scores, anomaly weights, and decision reasoning for security audit."""
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.bucket = "behavior-logs"
        self.folder = "logs/security-logs/"
        # Buffer logs in memory per session
        self.session_logs = {}  # session_id -> list of log lines
        import os
        self.backend_url = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

    def buffer_log(self, session_id, user_id, log_data: dict):
        from datetime import datetime
        lines = [f"Timestamp: {datetime.utcnow().isoformat()}"]
        if 'event' in log_data:
            lines.append(f"Event: {log_data['event']}")
        for k, v in log_data.items():
            if k != 'event':
                lines.append(f"{k}: {v}")
        log_text = "\n".join(lines) + "\n---\n"
        if session_id not in self.session_logs:
            self.session_logs[session_id] = []
        self.session_logs[session_id].append(log_text)

    async def upload_log(self, session_id, user_id):
        # Upload the buffered log to Supabase at session end
        await self.db_manager.initialize()
        supabase = self.db_manager.supabase
        file_path = self.folder + f"{session_id}.txt"
        log_text = "".join(self.session_logs.get(session_id, []))
        try:
            supabase.storage.from_(self.bucket).upload(file_path, log_text.encode('utf-8'))
        except Exception as e:
            # Try to remove and re-upload if file exists
            try:
                supabase.storage.from_(self.bucket).remove(file_path)
                supabase.storage.from_(self.bucket).upload(file_path, log_text.encode('utf-8'))
            except Exception as e2:
                logger.error(f"[AuditExplainabilityEngine] Failed to upload log after remove: {e2}")
        # Clean up buffer
        if session_id in self.session_logs:
            del self.session_logs[session_id]

    async def send_security_event(self, session_id, user_id, event_type, details):
        url = f"{self.backend_url}/api/v1/ws/security-event"
        payload = {
            "session_id": session_id,
            "user_id": user_id,
            "event_type": event_type,
            "details": details
        }
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
        except Exception as e:
            logger.error(f"[AuditExplainabilityEngine] Failed to send security event: {e}")

# Instantiate the audit engine
audit_engine = AuditExplainabilityEngine(db_manager)

# Test the upload directly at startup
if __name__ == "__main__":
    import asyncio
    async def test_audit_log():
        test_session = "test_audit_log_manual"
        test_user = "manual_test_user"
        log_data = {"event": "manual_test", "message": "This is a direct test of the audit log upload."}
        audit_engine.buffer_log(test_session, test_user, log_data)
        await audit_engine.upload_log(test_session, test_user)
        print(f"Manual audit log test for session {test_session} complete.")
    asyncio.run(test_audit_log())

@app.get("/")
async def health_check():
    """Health check endpoint"""
    try:
        # Initialize GNN storage manager if not already done
        await gnn_storage_manager.initialize()
        
        # Get cache statistics
        cache_stats = gnn_caching_manager.get_cache_stats()
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "database": "connected",
                "feature_extractor": "ready",
                "learning_manager": "ready",
                "auth_manager": "ready",
                "gnn_escalation": "ready",
                "supabase_storage": "connected"
            },
            "statistics": {
                "active_sessions": len(active_sessions),
                "total_users": await db_manager.get_total_users(),
                "total_sessions": await db_manager.get_total_sessions()
            },
            "gnn_cache_stats": cache_stats
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "degraded",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
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
        
        # Audit log for session start
        audit_engine.buffer_log(data.session_id, data.user_id, {
            "event": "session_start",
            "phase": phase,
            "status": "started"
        })
        
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
        
        # Audit log for session end
        await audit_engine.upload_log(data.session_id, session_data["user_id"])
        
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
    """Analyze behavioral data from mobile app using continuous authenticator.\n\nIf similarity < user-specific threshold, escalates to Level 2 (GNN-based anomaly detection)."""
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

        # Separate device_info events for drift tracking
        device_info_events = [e for e in data.logs if e.get('event_type') == 'device_info']
        behavioral_events = [e for e in data.logs if e.get('event_type') != 'device_info']

        # Process device_info events with drift tracker
        for event in device_info_events:
            drift_action = drift_tracker.process_device_info(data.session_id, data.user_id, event)
            if drift_action == 'block':
                logger.error(f"[DriftTracker] Blocking session {data.session_id} due to drift event.")
                # Audit log for block
                await audit_engine.send_security_event(data.session_id, data.user_id, "drift_detected", {"reason": "behavioral_drift_detected", "message": "Session blocked due to device/location drift."})
                return {
                    "status": "blocked",
                    "decision": "block",
                    "reason": "behavioral_drift_detected",
                    "message": "Session blocked due to device/location drift.",
                    "timestamp": datetime.utcnow().isoformat()
                }
            elif drift_action == 'reauth':
                logger.warning(f"[DriftTracker] Forcing re-authentication for session {data.session_id} due to IP/network change.")
                await audit_engine.send_security_event(data.session_id, data.user_id, "ip_change_detected", {"reason": "ip_change_detected", "message": "Re-authentication required due to network change."})
                return {
                    "status": "reauth_required",
                    "decision": "reauth",
                    "reason": "ip_change_detected",
                    "message": "Re-authentication required due to network change.",
                    "timestamp": datetime.utcnow().isoformat()
                }

        # Only send behavioral events to the main pipeline
        result = await continuous_authenticator.process_continuous_authentication(
            data.user_id, data.session_id, behavioral_events, session_data
        )
        
        # Update last analysis time
        session_data["last_analysis"] = datetime.utcnow()
        
        logger.info(f"Analysis result for {data.session_id}: {result['decision']} (confidence: {result.get('confidence', 0):.3f})")
        # Audit log for every decision
        audit_engine.buffer_log(data.session_id, data.user_id, result)
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to analyze behavior for session {data.session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/gnn/anomaly-detection")
async def gnn_anomaly_detection(data: GNNAnomalyRequest):
    """Direct GNN-based anomaly detection with user adaptation using Supabase historical data"""
    try:
        logger.info(f"🚀 Starting GNN anomaly detection for user {data.user_id}")
        
        # Ensure storage manager is initialized
        await gnn_storage_manager.initialize()
        
        # Perform anomaly detection with caching
        result = await detect_anomaly_with_user_adaptation(
            current_session_json=data.session_data,
            user_id=data.user_id,
            storage_manager=gnn_storage_manager,
            caching_manager=gnn_caching_manager
        )
        
        logger.info(f"✅ GNN anomaly detection completed for user {data.user_id}")
        
        return {
            "status": "success",
            "gnn_result": result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ GNN anomaly detection failed for user {data.user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/gnn/cache-stats")
async def get_gnn_cache_stats():
    """Get GNN caching statistics"""
    try:
        cache_stats = gnn_caching_manager.get_cache_stats()
        return {
            "status": "success",
            "cache_stats": cache_stats,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/gnn/clear-cache")
async def clear_gnn_cache():
    """Clear all GNN caches"""
    try:
        gnn_caching_manager.clear_cache()
        logger.info("🧹 GNN cache cleared")
        return {
            "status": "success",
            "message": "GNN cache cleared successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
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
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")

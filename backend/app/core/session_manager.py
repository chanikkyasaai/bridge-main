import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import deque
import aiofiles
import os
from app.core.config import settings

class BehavioralData:
    """Structure to hold behavioral data points"""
    def __init__(self, event_type: str, data: Dict[str, Any]):
        self.timestamp = datetime.utcnow().isoformat()
        self.event_type = event_type
        self.data = data
        self.session_id = data.get('session_id')

class UserSession:
    """Manages individual user session and behavioral data"""
    def __init__(self, session_id: str, user_email: str, device_id: str):
        self.session_id = session_id
        self.user_email = user_email
        self.device_id = device_id
        self.created_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        self.is_active = True
        self.is_blocked = False
        self.risk_score = 0.0
        self.mpin_attempts = 0
        self.behavioral_buffer = deque(maxlen=settings.BEHAVIOR_BUFFER_SIZE)
        self.websocket_connection = None
        
    def add_behavioral_data(self, event_type: str, data: Dict[str, Any]):
        """Add behavioral data to the session buffer"""
        behavior_data = BehavioralData(event_type, data)
        self.behavioral_buffer.append(behavior_data)
        self.last_activity = datetime.utcnow()
        
        # Save to file for ML model processing
        asyncio.create_task(self._save_to_buffer_file(behavior_data))
    
    async def _save_to_buffer_file(self, behavior_data: BehavioralData):
        """Save behavioral data to session-specific buffer file"""
        buffer_dir = "session_buffers"
        os.makedirs(buffer_dir, exist_ok=True)
        
        file_path = os.path.join(buffer_dir, f"{self.session_id}.jsonl")
        
        data_entry = {
            "timestamp": behavior_data.timestamp,
            "event_type": behavior_data.event_type,
            "data": behavior_data.data,
            "session_id": self.session_id,
            "user_email": self.user_email
        }
        
        async with aiofiles.open(file_path, mode='a') as f:
            await f.write(json.dumps(data_entry) + '\n')
    
    def update_risk_score(self, new_score: float):
        """Update risk score and handle security actions"""
        self.risk_score = new_score
        
        if new_score >= settings.HIGH_RISK_THRESHOLD:
            self.block_session("High risk behavior detected")
        elif new_score >= settings.SUSPICIOUS_THRESHOLD:
            self.request_mpin_verification()
    
    def block_session(self, reason: str):
        """Block the session due to suspicious activity"""
        self.is_blocked = True
        self.is_active = False
        print(f"Session {self.session_id} blocked: {reason}")
        
        # Notify client via WebSocket if connected
        if self.websocket_connection:
            asyncio.create_task(self._notify_client({
                "type": "session_blocked",
                "reason": reason,
                "timestamp": datetime.utcnow().isoformat()
            }))
    
    def request_mpin_verification(self):
        """Request MPIN verification from user"""
        print(f"Requesting MPIN verification for session {self.session_id}")
        
        # Notify client via WebSocket if connected
        if self.websocket_connection:
            asyncio.create_task(self._notify_client({
                "type": "mpin_required",
                "message": "Please verify your MPIN to continue",
                "timestamp": datetime.utcnow().isoformat()
            }))
    
    async def _notify_client(self, message: Dict[str, Any]):
        """Send notification to client via WebSocket"""
        if self.websocket_connection:
            try:
                await self.websocket_connection.send_text(json.dumps(message))
            except Exception as e:
                print(f"Failed to send WebSocket message: {e}")
    
    def is_expired(self) -> bool:
        """Check if session has expired"""
        expiry_time = self.created_at + timedelta(minutes=settings.SESSION_EXPIRE_MINUTES)
        return datetime.utcnow() > expiry_time
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        return {
            "session_id": self.session_id,
            "user_email": self.user_email,
            "device_id": self.device_id,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "is_active": self.is_active,
            "is_blocked": self.is_blocked,
            "risk_score": self.risk_score,
            "behavioral_events_count": len(self.behavioral_buffer),
            "mpin_attempts": self.mpin_attempts
        }

class SessionManager:
    """Global session manager"""
    def __init__(self):
        self.active_sessions: Dict[str, UserSession] = {}
        self.user_sessions: Dict[str, List[str]] = {}  # user_email -> [session_ids]
        
    def create_session(self, user_email: str, device_id: str) -> str:
        """Create a new user session"""
        session_id = str(uuid.uuid4())
        session = UserSession(session_id, user_email, device_id)
        
        self.active_sessions[session_id] = session
        
        if user_email not in self.user_sessions:
            self.user_sessions[user_email] = []
        self.user_sessions[user_email].append(session_id)
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[UserSession]:
        """Get session by ID"""
        return self.active_sessions.get(session_id)
    
    def get_user_sessions(self, user_email: str) -> List[UserSession]:
        """Get all active sessions for a user"""
        session_ids = self.user_sessions.get(user_email, [])
        return [self.active_sessions[sid] for sid in session_ids if sid in self.active_sessions]
    
    def terminate_session(self, session_id: str) -> bool:
        """Terminate a specific session"""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session.is_active = False
            
            # Remove from user sessions
            if session.user_email in self.user_sessions:
                if session_id in self.user_sessions[session.user_email]:
                    self.user_sessions[session.user_email].remove(session_id)
            
            del self.active_sessions[session_id]
            return True
        return False
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        expired_sessions = []
        for session_id, session in self.active_sessions.items():
            if session.is_expired():
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.terminate_session(session_id)
        
        print(f"Cleaned up {len(expired_sessions)} expired sessions")
    
    def get_all_sessions_stats(self) -> Dict[str, Any]:
        """Get statistics for all sessions"""
        return {
            "total_active_sessions": len(self.active_sessions),
            "total_users": len(self.user_sessions),
            "sessions": [session.get_session_stats() for session in self.active_sessions.values()]
        }

# Global session manager instance
session_manager = SessionManager()

# Background task to cleanup expired sessions
async def cleanup_sessions_task():
    """Background task to periodically clean up expired sessions"""
    while True:
        await asyncio.sleep(settings.SESSION_CLEANUP_INTERVAL)
        session_manager.cleanup_expired_sessions()

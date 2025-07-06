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
    def __init__(self, session_id: str, user_id: str, phone: str, device_id: str, supabase_session_id: str = None):
        self.session_id = session_id  # Local session ID
        self.supabase_session_id = supabase_session_id  # Supabase database session ID
        self.user_id = user_id  # Supabase user ID
        self.phone = phone
        self.device_id = device_id
        self.created_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        self.is_active = True
        self.is_blocked = False
        self.risk_score = 0.0
        self.mpin_attempts = 0
        self.behavioral_buffer = []  # Store all behavioral data in memory during session
        self.websocket_connection = None
        self.ended_at = None
        self.session_token = None  # JWT session token for this session
        
    def add_behavioral_data(self, event_type: str, data: Dict[str, Any]):
        """Add behavioral data to the session buffer (kept in memory during session)"""
        behavior_data = BehavioralData(event_type, data)
        self.behavioral_buffer.append(behavior_data)
        self.last_activity = datetime.utcnow()
        
        # No longer save to file immediately - keep in memory until session ends
    
    async def save_behavioral_data_to_supabase(self):
        """Save all behavioral data to Supabase Storage when session ends"""
        from app.core.supabase_client import supabase_client
        
        try:
            # Convert behavioral buffer to JSON-serializable format
            behavioral_logs = []
            for behavior_data in self.behavioral_buffer:
                log_entry = {
                    "timestamp": behavior_data.timestamp,
                    "event_type": behavior_data.event_type,
                    "data": behavior_data.data
                }
                behavioral_logs.append(log_entry)
            
            # Upload to Supabase Storage
            if behavioral_logs:
                file_path = await supabase_client.upload_behavioral_log(
                    self.user_id, 
                    self.session_id, 
                    behavioral_logs
                )
                
                # Update session record with log file URL
                if self.supabase_session_id:
                    await supabase_client.update_session(
                        self.supabase_session_id,
                        {"log_file_url": file_path, "anomaly_score": self.risk_score}
                    )
                
                return file_path
            
        except Exception as e:
            print(f"Failed to save behavioral data to Supabase: {e}")
            return None
    
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
    
    def end_session(self):
        """End the session and prepare for data persistence"""
        self.is_active = False
        self.ended_at = datetime.utcnow()
        
        # Log session completion
        print(f"Session {self.session_id} ended. Total behavioral events: {len(self.behavioral_buffer)}")
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        return {
            "session_id": self.session_id,
            "supabase_session_id": self.supabase_session_id,
            "user_id": self.user_id,
            "phone": self.phone,
            "device_id": self.device_id,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "is_active": self.is_active,
            "is_blocked": self.is_blocked,
            "risk_score": self.risk_score,
            "behavioral_events_count": len(self.behavioral_buffer),
            "mpin_attempts": self.mpin_attempts
        }
    
    def get_behavioral_summary(self) -> Dict[str, Any]:
        """Get summary of behavioral data collected during session"""
        if not self.behavioral_buffer:
            return {
                "total_events": 0,
                "event_types": [],
                "duration_minutes": 0,
                "first_event": None,
                "last_event": None
            }
        
        event_types = list(set(bd.event_type for bd in self.behavioral_buffer))
        duration = (self.last_activity - self.created_at).total_seconds() / 60
        
        return {
            "total_events": len(self.behavioral_buffer),
            "event_types": event_types,
            "duration_minutes": round(duration, 2),
            "first_event": self.behavioral_buffer[0].timestamp if self.behavioral_buffer else None,
            "last_event": self.behavioral_buffer[-1].timestamp if self.behavioral_buffer else None,
            "risk_score": self.risk_score
        }
    
    async def validate_and_save_behavioral_data(self):
        """Validate behavioral data before saving to permanent storage"""
        if not self.behavioral_buffer:
            print(f"No behavioral data to save for session {self.session_id}")
            return None
        
        # Validate data integrity
        valid_events = []
        for behavior_data in self.behavioral_buffer:
            if self._validate_behavioral_event(behavior_data):
                valid_events.append(behavior_data)
            else:
                print(f"Invalid behavioral event skipped: {behavior_data.event_type}")
        
        if not valid_events:
            print(f"No valid behavioral data to save for session {self.session_id}")
            return None
        
        # Save validated data
        try:
            return await self.save_behavioral_data_to_supabase()
        except Exception as e:
            print(f"Failed to save behavioral data: {e}")
            # Fallback: save to local file
            return await self._save_to_local_backup()
    
    def _validate_behavioral_event(self, behavior_data: 'BehavioralData') -> bool:
        """Validate individual behavioral event"""
        try:
            # Check required fields
            if not behavior_data.event_type or not behavior_data.timestamp:
                return False
            
            # Validate timestamp format
            datetime.fromisoformat(behavior_data.timestamp.replace('Z', ''))
            
            # Validate data is not empty
            if not behavior_data.data or not isinstance(behavior_data.data, dict):
                return False
            
            return True
        except Exception:
            return False
    
    async def _save_to_local_backup(self) -> str:
        """Backup behavioral data to local file if Supabase fails"""
        try:
            import os
            import json
            
            backup_dir = "session_backups"
            os.makedirs(backup_dir, exist_ok=True)
            
            backup_file = f"{backup_dir}/session_{self.session_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
            
            behavioral_logs = []
            for behavior_data in self.behavioral_buffer:
                log_entry = {
                    "timestamp": behavior_data.timestamp,
                    "event_type": behavior_data.event_type,
                    "data": behavior_data.data
                }
                behavioral_logs.append(log_entry)
            
            session_backup = {
                "session_id": self.session_id,
                "user_id": self.user_id,
                "phone": self.phone,
                "device_id": self.device_id,
                "created_at": self.created_at.isoformat(),
                "ended_at": getattr(self, 'ended_at', datetime.utcnow()).isoformat(),
                "risk_score": self.risk_score,
                "behavioral_data": behavioral_logs
            }
            
            with open(backup_file, 'w') as f:
                json.dump(session_backup, f, indent=2)
            
            print(f"Session data backed up to: {backup_file}")
            return backup_file
            
        except Exception as e:
            print(f"Failed to create local backup: {e}")
            return None

class SessionManager:
    """Global session manager"""
    def __init__(self):
        self.active_sessions: Dict[str, UserSession] = {}
        self.user_sessions: Dict[str, List[str]] = {}  # user_id -> [session_ids]
        
    async def create_session(self, user_id: str, phone: str, device_id: str, session_token: str = None) -> str:
        """Create a new user session with Supabase integration"""
        from app.core.supabase_client import supabase_client
        
        session_id = str(uuid.uuid4())
        
        # Create session in Supabase database
        try:
            supabase_session = await supabase_client.create_session(
                user_id, device_id, session_token
            )
            supabase_session_id = supabase_session['id'] if supabase_session else None
        except Exception as e:
            print(f"Failed to create session in Supabase: {e}")
            supabase_session_id = None
        
        # Create local session object
        session = UserSession(session_id, user_id, phone, device_id, supabase_session_id)
        session.session_token = session_token  # Set the session token

        self.active_sessions[session_id] = session
        
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = []
        self.user_sessions[user_id].append(session_id)
        
        print(f"Created session: {session_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[UserSession]:
        """Get session by ID"""
        
        print(f"Getting session: {session_id}")
        return self.active_sessions.get(session_id)
    
    def get_user_sessions(self, user_id: str) -> List[UserSession]:
        """Get all active sessions for a user"""
        session_ids = self.user_sessions.get(user_id, [])
        return [self.active_sessions[sid] for sid in session_ids if sid in self.active_sessions]
    
    async def terminate_session(self, session_id: str, final_decision: str = "normal") -> bool:
        """Terminate a specific session and save behavioral data to Supabase"""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            
            # End the session
            session.end_session()
            
            # Get behavioral data summary before saving
            summary = session.get_behavioral_summary()
            print(f"Terminating session {session_id}: {summary}")
            
            # Validate and save behavioral data to permanent storage
            log_file_path = await session.validate_and_save_behavioral_data()
            
            # Mark session as ended in Supabase
            if session.supabase_session_id:
                try:
                    from app.core.supabase_client import supabase_client
                    await supabase_client.end_session(
                        session.supabase_session_id, 
                        final_decision, 
                        log_file_path or "",
                        session.risk_score
                    )
                except Exception as e:
                    print(f"Failed to end session in Supabase: {e}")
            
            # Remove from user sessions
            if session.user_id in self.user_sessions:
                if session_id in self.user_sessions[session.user_id]:
                    self.user_sessions[session.user_id].remove(session_id)
            
            # Clean up session from memory
            del self.active_sessions[session_id]
            
            print(f"Session {session_id} terminated successfully. Log file: {log_file_path}")
            return True
        return False
    
    async def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        expired_sessions = []
        for session_id, session in self.active_sessions.items():
            if session.is_expired():
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            await self.terminate_session(session_id, "expired")
        
        print(f"Cleaned up {len(expired_sessions)} expired sessions")
    
    def get_all_sessions_stats(self) -> Dict[str, Any]:
        """Get statistics for all sessions"""
        return {
            "total_active_sessions": len(self.active_sessions),
            "total_users": len(self.user_sessions),
            "sessions": [session.get_session_stats() for session in self.active_sessions.values()]
        }

    async def handle_app_lifecycle_event(self, session_id: str, event_type: str, details: Dict[str, Any] = None):
        """
        Handle various app lifecycle events
        
        Args:
            session_id: The session identifier
            event_type: Type of lifecycle event (app_close, app_background, app_foreground, user_logout, etc.)
            details: Additional details about the event
        """
        session = self.get_session(session_id)
        if not session:
            return False

        # Add lifecycle event to behavioral data
        lifecycle_data = {
            "session_id": session_id,
            "event_type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            "session_duration": (datetime.utcnow() - session.created_at).total_seconds()
        }

        if details:
            lifecycle_data.update(details)

        session.add_behavioral_data("app_lifecycle", lifecycle_data)

        # Handle different lifecycle events
        if event_type in ["app_close", "user_logout", "force_close"]:
            # Terminate session immediately
            await self.terminate_session(session_id, event_type)
            return True
        elif event_type == "app_background":
            # Mark session as backgrounded but keep it active
            session.add_behavioral_data("session_backgrounded", lifecycle_data)
            print(f"Session {session_id} moved to background")
            return True
        elif event_type == "app_foreground":
            # Session resumed from background
            session.add_behavioral_data("session_resumed", lifecycle_data)
            print(f"Session {session_id} resumed from background")
            return True
        elif event_type == "websocket_disconnect":
            # WebSocket disconnected but app may still be active
            session.websocket_connection = None
            session.add_behavioral_data(
                "websocket_disconnected", lifecycle_data)
            print(f"WebSocket disconnected for session {session_id}")
            return True

        return False

    def get_session_by_user_and_device(self, user_id: str, device_id: str) -> Optional[UserSession]:
        """Get active session for a specific user and device combination"""
        user_sessions = self.get_user_sessions(user_id)
        for session in user_sessions:
            if session.device_id == device_id and session.is_active:
                return session
        return None

# Global session manager instance
session_manager = SessionManager()

# Background task to cleanup expired sessions
async def cleanup_sessions_task():
    """Background task to periodically clean up expired sessions"""
    while True:
        await asyncio.sleep(settings.SESSION_CLEANUP_INTERVAL)
        await session_manager.cleanup_expired_sessions()

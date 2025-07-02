import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from supabase import create_client, Client
from app.core.config import settings

class SupabaseClient:
    """Supabase client for database and storage operations"""
    
    def __init__(self):
        self.supabase: Client = create_client(
            settings.SUPABASE_URL,
            settings.SUPABASE_SERVICE_KEY
        )
    
    # User Management
    async def create_user(self, phone: str, password_hash: str, mpin_hash: str) -> Dict[str, Any]:
        """Create a new user in the database"""
        try:
            result = self.supabase.table('users').insert({
                'phone': phone,
                'password_hash': password_hash,
                'mpin_hash': mpin_hash
            }).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            raise Exception(f"Failed to create user: {str(e)}")
    
    async def get_user_by_phone(self, phone: str) -> Optional[Dict[str, Any]]:
        """Get user by phone number"""
        try:
            result = self.supabase.table('users').select('*').eq('phone', phone).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            raise Exception(f"Failed to get user: {str(e)}")
    
    # Session Management
    async def create_session(self, user_id: str, device_info: str, session_token: str) -> Dict[str, Any]:
        """Create a new session in the database"""
        try:
            result = self.supabase.table('sessions').insert({
                'user_id': user_id,
                'device_info': device_info,
                'session_token': session_token,
                'is_escalated': False,
                'anomaly_score': 0.0
            }).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            raise Exception(f"Failed to create session: {str(e)}")
    
    async def update_session(self, session_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update session data"""
        try:
            result = self.supabase.table('sessions').update(updates).eq('id', session_id).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            raise Exception(f"Failed to update session: {str(e)}")
    
    async def end_session(self, session_id: str, final_decision: str, log_file_url: str) -> Dict[str, Any]:
        """Mark session as ended and save log file URL"""
        try:
            updates = {
                'ended_at': datetime.utcnow().isoformat(),
                'final_decision': final_decision,
                'log_file_url': log_file_url
            }
            result = self.supabase.table('sessions').update(updates).eq('id', session_id).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            raise Exception(f"Failed to end session: {str(e)}")
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session by ID"""
        try:
            result = self.supabase.table('sessions').select('*').eq('id', session_id).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            raise Exception(f"Failed to get session: {str(e)}")
    
    # Security Events
    async def create_security_event(self, session_id: str, level: int, decision: str, 
                                   reason: str, model_used: str, match_score: float) -> Dict[str, Any]:
        """Create a security event record"""
        try:
            result = self.supabase.table('security_events').insert({
                'session_id': session_id,
                'level': level,
                'decision': decision,
                'reason': reason,
                'model_used': model_used,
                'match_score': match_score
            }).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            raise Exception(f"Failed to create security event: {str(e)}")
    
    # Storage Operations
    async def upload_behavioral_log(self, user_id: str, session_id: str, 
                                   behavioral_data: List[Dict[str, Any]]) -> str:
        """Upload behavioral log JSON to Supabase Storage"""
        try:
            # Prepare log data
            log_data = {
                "user_id": user_id,
                "session_id": session_id,
                "logs": behavioral_data,
                "uploaded_at": datetime.utcnow().isoformat(),
                "total_events": len(behavioral_data)
            }
            
            # Convert to JSON string
            json_data = json.dumps(log_data, indent=2, default=str)
            
            # Create file path
            file_path = f"logs/{user_id}/{session_id}.json"
            
            # Upload to Supabase Storage
            result = self.supabase.storage.from_(settings.SUPABASE_STORAGE_BUCKET).upload(
                file_path,
                json_data.encode('utf-8'),
                file_options={
                    "content-type": "application/json",
                    "x-upsert": "true"  # Overwrite if exists
                }
            )
            
            if result:
                # Return the public URL or file path
                return file_path
            else:
                raise Exception("Upload failed")
                
        except Exception as e:
            raise Exception(f"Failed to upload behavioral log: {str(e)}")
    
    async def download_behavioral_log(self, file_path: str) -> Dict[str, Any]:
        """Download behavioral log from Supabase Storage"""
        try:
            result = self.supabase.storage.from_(settings.SUPABASE_STORAGE_BUCKET).download(file_path)
            if result:
                return json.loads(result.decode('utf-8'))
            else:
                raise Exception("Download failed")
        except Exception as e:
            raise Exception(f"Failed to download behavioral log: {str(e)}")
    
    async def get_public_url(self, file_path: str) -> str:
        """Get public URL for a file in storage"""
        try:
            result = self.supabase.storage.from_(settings.SUPABASE_STORAGE_BUCKET).get_public_url(file_path)
            return result['publicURL'] if result else None
        except Exception as e:
            raise Exception(f"Failed to get public URL: {str(e)}")
    
    # Analytics and Reporting
    async def get_user_sessions(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent sessions for a user"""
        try:
            result = self.supabase.table('sessions')\
                .select('*')\
                .eq('user_id', user_id)\
                .order('started_at', desc=True)\
                .limit(limit)\
                .execute()
            return result.data
        except Exception as e:
            raise Exception(f"Failed to get user sessions: {str(e)}")
    
    async def get_security_events_for_session(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all security events for a session"""
        try:
            result = self.supabase.table('security_events')\
                .select('*')\
                .eq('session_id', session_id)\
                .order('timestamp', desc=True)\
                .execute()
            return result.data
        except Exception as e:
            raise Exception(f"Failed to get security events: {str(e)}")

# Global Supabase client instance
supabase_client = SupabaseClient()

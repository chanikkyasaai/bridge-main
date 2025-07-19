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
                'phone_number': phone,  # Fixed: use phone_number to match schema
                'password_hash': password_hash,
                'mpin_hash': mpin_hash,
                'sessions_count': 0  # Initialize sessions_count as per schema
            }).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            raise Exception(f"Failed to create user: {str(e)}")
    
    async def get_user_by_phone(self, phone: str) -> Optional[Dict[str, Any]]:
        """Get user by phone number"""
        try:
            result = self.supabase.table('users').select('*').eq('phone_number', phone).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            raise Exception(f"Failed to get user: {str(e)}")
    
    async def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID"""
        try:
            result = self.supabase.table('users').select('*').eq('id', user_id).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            raise Exception(f"Failed to get user: {str(e)}")
    
    async def increment_user_session_count(self, user_id: str) -> int:
        """Increment user session count and return new count"""
        try:
            # Get current count
            user = await self.get_user_by_id(user_id)
            if not user:
                raise Exception("User not found")
            
            new_count = (user.get('sessions_count', 0) or 0) + 1
            
            # Update count
            result = self.supabase.table('users').update({
                'sessions_count': new_count
            }).eq('id', user_id).execute()
            
            return new_count
        except Exception as e:
            raise Exception(f"Failed to increment session count: {str(e)}")
    
    async def get_user_session_count(self, user_id: str) -> int:
        """Get user's current session count"""
        try:
            user = await self.get_user_by_id(user_id)
            if not user:
                raise Exception("User not found")
            return user.get('sessions_count', 0) or 0
        except Exception as e:
            raise Exception(f"Failed to get session count: {str(e)}")

    async def mark_session_ended(self, session_id: str) -> Dict[str, Any]:
        """Mark session as ended"""
        try:
            result = self.supabase.table('sessions').update({
                'ended_at': datetime.utcnow().isoformat()
            }).eq('id', session_id).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            raise Exception(f"Failed to mark session ended: {str(e)}")
    
    # Session Management
    async def create_session(self, user_id: str, session_token: str, device_info: Optional[str] = None) -> Dict[str, Any]:
        """Create a new session in the database"""
        try:
            result = self.supabase.table('sessions').insert({
                'user_id': user_id,
                'session_token': session_token
            }).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            raise Exception(f"Failed to create session: {str(e)}")
    
    # Session Vectors Management
    async def store_session_vector(self, session_id: str, vector: List[float]) -> Dict[str, Any]:
        """Store session vector in database"""
        try:
            result = self.supabase.table('session_vectors').insert({
                'session_id': session_id,
                'vector': vector
            }).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            raise Exception(f"Failed to store session vector: {str(e)}")
    
    async def get_user_session_vectors(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get session vectors for a user"""
        try:
            result = self.supabase.table('session_vectors').select(
                'id, session_id, vector, created_at, sessions!inner(user_id)'
            ).eq('sessions.user_id', user_id).order('created_at', desc=True).limit(limit).execute()
            
            return result.data if result.data else []
        except Exception as e:
            raise Exception(f"Failed to get session vectors: {str(e)}")
    
    # User Clusters Management
    async def store_user_clusters(self, user_id: str, clusters: List[Dict[str, Any]]) -> bool:
        """Store user behavioral clusters"""
        try:
            # Delete existing clusters
            self.supabase.table('user_clusters').delete().eq('user_id', user_id).execute()
            
            # Insert new clusters
            cluster_data = []
            for cluster in clusters:
                cluster_data.append({
                    'user_id': user_id,
                    'cluster_label': cluster['label'],
                    'centroid': cluster['centroid']
                })
            
            if cluster_data:
                result = self.supabase.table('user_clusters').insert(cluster_data).execute()
                return True
            return False
        except Exception as e:
            raise Exception(f"Failed to store user clusters: {str(e)}")
    
    async def get_user_clusters(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user behavioral clusters"""
        try:
            result = self.supabase.table('user_clusters').select('*').eq('user_id', user_id).order('cluster_label').execute()
            return result.data if result.data else []
        except Exception as e:
            raise Exception(f"Failed to get user clusters: {str(e)}")
    
    async def find_nearest_cluster(self, user_id: str, vector: List[float]) -> Optional[Dict[str, Any]]:
        """Find nearest cluster for a vector (simplified - would need proper similarity calculation)"""
        try:
            clusters = await self.get_user_clusters(user_id)
            if not clusters:
                return None
            
            # For now, return first cluster - in production, calculate actual similarity
            return clusters[0] if clusters else None
        except Exception as e:
            raise Exception(f"Failed to find nearest cluster: {str(e)}")
    
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

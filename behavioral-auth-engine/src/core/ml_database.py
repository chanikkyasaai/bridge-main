"""
Supabase Database Client for ML Engine
Handles all database operations for behavioral authentication data
"""

import os
import uuid
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import asyncio
import logging
from supabase import create_client, Client
from src.config.settings import get_settings

logger = logging.getLogger(__name__)

class MLSupabaseClient:
    """Supabase client specifically for ML Engine operations"""
    
    def __init__(self):
        self.settings = get_settings()
        
        # Initialize Supabase client
        try:
            self.supabase: Client = create_client(
                self.settings.supabase_url,
                self.settings.supabase_service_key
            )
            logger.info("ML Engine Supabase client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {e}")
            raise
    
    # ============================================================================
    # USER PROFILE MANAGEMENT
    # ============================================================================
    
    async def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user behavioral profile from database"""
        try:
            result = self.supabase.table('user_profiles').select('*').eq('user_id', user_id).execute()
            if result.data:
                return result.data[0]
            return None
        except Exception as e:
            logger.error(f"Failed to get user profile for {user_id}: {e}")
            return None
    
    async def create_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Create new user behavioral profile"""
        try:
            # First, ensure the user exists in the users table
            await self._ensure_user_exists(user_id)
            
            profile_data = {
                'user_id': user_id,
                'current_session_count': 0,
                'total_sessions': 0,
                'current_phase': 'learning',
                'risk_score': 0.0,
                'last_activity': datetime.utcnow().isoformat(),
                'behavioral_model_version': 1
            }
            
            result = self.supabase.table('user_profiles').insert(profile_data).execute()
            if result.data:
                logger.info(f"Created user profile for {user_id}")
                return result.data[0]
            return None
        except Exception as e:
            logger.error(f"Failed to create user profile for {user_id}: {e}")
            return None
    
    async def _ensure_user_exists(self, user_id: str):
        """Ensure user exists in users table"""
        try:
            # Check if user exists
            result = self.supabase.table('users').select('id').eq('id', user_id).execute()
            if not result.data:
                # Create user if not exists - using required fields based on actual schema
                user_data = {
                    'id': user_id,
                    'phone': f"test{user_id[:8]}",  # Generate test phone from user ID
                    'password_hash': 'test_hash',
                    'mpin_hash': 'test_mpin_hash'
                }
                result = self.supabase.table('users').insert(user_data).execute()
                if result.data:
                    logger.info(f"Created user record for {user_id}")
                else:
                    logger.error(f"Failed to create user record for {user_id}: {result}")
        except Exception as e:
            logger.error(f"Failed to ensure user exists for {user_id}: {e}")
            # If users table doesn't exist or has different schema, continue anyway
            pass
    
    async def update_user_profile(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """Update user behavioral profile"""
        try:
            updates['updated_at'] = datetime.utcnow().isoformat()
            
            result = self.supabase.table('user_profiles').update(updates).eq('user_id', user_id).execute()
            return len(result.data) > 0
        except Exception as e:
            logger.error(f"Failed to update user profile for {user_id}: {e}")
            return False
    
    async def increment_session_count(self, user_id: str) -> bool:
        """Increment session count and update phase if needed"""
        try:
            # Get current profile
            profile = await self.get_user_profile(user_id)
            if not profile:
                # Create profile if doesn't exist
                profile = await self.create_user_profile(user_id)
                if not profile:
                    return False
            
            new_session_count = profile['current_session_count'] + 1
            new_total_sessions = profile['total_sessions'] + 1
            
            # Determine phase based on session count
            if new_session_count <= 5:  # First 5 sessions - learning phase
                new_phase = 'learning'
            elif new_session_count <= 15:  # Next 10 sessions - gradual risk
                new_phase = 'gradual_risk'  
            else:  # After 15 sessions - full authentication
                new_phase = 'full_auth'
            
            updates = {
                'current_session_count': new_session_count,
                'total_sessions': new_total_sessions,
                'current_phase': new_phase,
                'last_activity': datetime.utcnow().isoformat()
            }
            
            return await self.update_user_profile(user_id, updates)
        except Exception as e:
            logger.error(f"Failed to increment session count for {user_id}: {e}")
            return False
    
    async def create_session(self, user_id: str, session_name: str = None, device_info: str = None) -> Optional[str]:
        """Create a new session record and return the session UUID"""
        try:
            await self._ensure_user_exists(user_id)
            
            # Check if session already exists by session_token (our session name)
            if session_name:
                try:
                    existing_result = self.supabase.table('sessions')\
                        .select('id')\
                        .eq('session_token', session_name)\
                        .eq('user_id', user_id)\
                        .execute()
                    
                    if existing_result.data:
                        session_id = existing_result.data[0]['id']
                        logger.info(f"Found existing session {session_id} for {session_name}")
                        return session_id
                except Exception as e:
                    logger.warning(f"Error checking existing session: {e}")
            
            session_record = {
                'user_id': user_id,
                'device_info': device_info or 'ML Engine Session',
                'session_token': session_name,  # Store the human-readable name here
                'started_at': datetime.utcnow().isoformat()
            }
            
            result = self.supabase.table('sessions').insert(session_record).execute()
            if result.data:
                session_id = result.data[0]['id']
                logger.info(f"Created session {session_id} for user {user_id}")
                return session_id
            return None
        except Exception as e:
            logger.error(f"Failed to create session for {user_id}: {e}")
            return None
    
    async def _ensure_session_exists(self, user_id: str, session_id: str) -> Optional[str]:
        """Ensure session exists in database, create if needed, return actual session UUID"""
        try:
            # First, try to find existing session by UUID (direct match)
            try:
                direct_result = self.supabase.table('sessions')\
                    .select('id')\
                    .eq('id', session_id)\
                    .eq('user_id', user_id)\
                    .execute()
                
                if direct_result.data:
                    logger.debug(f"Found session by direct UUID match: {session_id}")
                    return session_id
            except Exception as e:
                logger.debug(f"Direct UUID lookup failed: {e}")
            
            # If not found by UUID, try by session_token
            try:
                token_result = self.supabase.table('sessions')\
                    .select('id')\
                    .eq('session_token', session_id)\
                    .eq('user_id', user_id)\
                    .execute()
                
                if token_result.data:
                    actual_session_id = token_result.data[0]['id']
                    logger.debug(f"Found session by token: {session_id} -> {actual_session_id}")
                    return actual_session_id
            except Exception as e:
                logger.debug(f"Token lookup failed: {e}")
            
            # Session doesn't exist, create it
            logger.info(f"Session {session_id} not found, creating new session")
            return await self.create_session(
                user_id=user_id,
                session_name=session_id,
                device_info="Auto-created for ML operations"
            )
            
        except Exception as e:
            logger.error(f"Failed to ensure session exists for {session_id}: {e}")
            return None
    
    # ============================================================================
    # BEHAVIORAL VECTOR STORAGE
    # ============================================================================
    
    async def store_behavioral_vector(self, user_id: str, session_id: str, 
                                    vector_data: List[float], confidence_score: float,
                                    feature_source: str) -> Optional[str]:
        """Store behavioral vector in database"""
        try:
            # Ensure session exists - create if not exists
            actual_session_id = await self._ensure_session_exists(user_id, session_id)
            if not actual_session_id:
                logger.error(f"Failed to ensure session exists for {session_id}")
                return None
            
            vector_record = {
                'user_id': user_id,
                'session_id': actual_session_id,
                'vector_data': vector_data,  # PostgreSQL array
                'confidence_score': confidence_score,
                'feature_source': feature_source
            }
            
            result = self.supabase.table('behavioral_vectors').insert(vector_record).execute()
            if result.data:
                vector_id = result.data[0]['id']
                logger.info(f"Stored behavioral vector {vector_id} for user {user_id}")
                return vector_id
            return None
        except Exception as e:
            logger.error(f"Failed to store behavioral vector for {user_id}: {e}")
            return None
    
    async def get_user_vectors(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get behavioral vectors for a user"""
        try:
            query = self.supabase.table('behavioral_vectors')\
                .select('*')\
                .eq('user_id', user_id)\
                .order('created_at', desc=True)
            
            if limit:
                query = query.limit(limit)
            
            result = query.execute()
            return result.data if result.data else []
        except Exception as e:
            logger.error(f"Failed to get vectors for user {user_id}: {e}")
            return []
    
    async def get_user_vectors_by_phase(self, user_id: str, phase: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get behavioral vectors for a specific learning phase"""
        try:
            # Get vectors from sessions where user was in specific phase
            # For now, we'll get recent vectors and filter by creation time
            # In production, you might want to add phase info to vectors table
            vectors = await self.get_user_vectors(user_id, limit)
            return vectors  # Simplified for now
        except Exception as e:
            logger.error(f"Failed to get vectors by phase for {user_id}: {e}")
            return []
    
    # ============================================================================
    # AUTHENTICATION DECISIONS
    # ============================================================================
    
    async def store_authentication_decision(self, user_id: str, session_id: str,
                                          decision: str, confidence: float,
                                          similarity_score: Optional[float] = None,
                                          layer_used: str = 'faiss',
                                          risk_factors: List[str] = None,
                                          threshold_used: Optional[float] = None,
                                          processing_time_ms: Optional[int] = None) -> Optional[str]:
        """Store authentication decision in database"""
        try:
            # Ensure session exists - create if not exists
            actual_session_id = await self._ensure_session_exists(user_id, session_id)
            if not actual_session_id:
                logger.error(f"Failed to ensure session exists for {session_id}")
                return None
            
            decision_record = {
                'user_id': user_id,
                'session_id': actual_session_id,
                'decision': decision,
                'confidence': confidence,
                'similarity_score': similarity_score,
                'layer_used': layer_used,
                'risk_factors': risk_factors or [],
                'threshold_used': threshold_used,
                'processing_time_ms': processing_time_ms
            }
            
            result = self.supabase.table('authentication_decisions').insert(decision_record).execute()
            if result.data:
                decision_id = result.data[0]['id']
                logger.info(f"Stored decision {decision_id} for user {user_id}: {decision}")
                return decision_id
            return None
        except Exception as e:
            logger.error(f"Failed to store authentication decision for {user_id}: {e}")
            return None
    
    async def get_recent_decisions(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent authentication decisions for a user"""
        try:
            result = self.supabase.table('authentication_decisions')\
                .select('*')\
                .eq('user_id', user_id)\
                .order('created_at', desc=True)\
                .limit(limit)\
                .execute()
            
            return result.data if result.data else []
        except Exception as e:
            logger.error(f"Failed to get recent decisions for {user_id}: {e}")
            return []
    
    # ============================================================================
    # BEHAVIORAL FEEDBACK
    # ============================================================================
    
    async def store_behavioral_feedback(self, user_id: str, session_id: str,
                                      decision_id: str, was_correct: bool,
                                      feedback_source: str = 'system',
                                      corrective_action: Optional[str] = None) -> Optional[str]:
        """Store behavioral feedback for model improvement"""
        try:
            feedback_record = {
                'user_id': user_id,
                'session_id': session_id,
                'decision_id': decision_id,
                'was_correct': was_correct,
                'feedback_source': feedback_source,
                'corrective_action': corrective_action
            }
            
            result = self.supabase.table('behavioral_feedback').insert(feedback_record).execute()
            if result.data:
                feedback_id = result.data[0]['id']
                logger.info(f"Stored feedback {feedback_id} for decision {decision_id}")
                return feedback_id
            return None
        except Exception as e:
            logger.error(f"Failed to store behavioral feedback: {e}")
            return None
    
    # ============================================================================
    # SESSION BEHAVIORAL SUMMARY
    # ============================================================================
    
    async def create_session_summary(self, session_id: str, total_events: int,
                                   unique_event_types: List[str],
                                   session_duration_seconds: int,
                                   total_vectors_generated: int,
                                   average_confidence: float,
                                   anomaly_indicators: List[str] = None,
                                   final_risk_assessment: str = 'low') -> Optional[str]:
        """Create session behavioral summary"""
        try:
            summary_record = {
                'session_id': session_id,
                'total_events': total_events,
                'unique_event_types': unique_event_types or [],
                'session_duration_seconds': session_duration_seconds,
                'total_vectors_generated': total_vectors_generated,
                'average_confidence': average_confidence,
                'anomaly_indicators': anomaly_indicators or [],
                'final_risk_assessment': final_risk_assessment
            }
            
            result = self.supabase.table('session_behavioral_summary').insert(summary_record).execute()
            if result.data:
                summary_id = result.data[0]['id']
                logger.info(f"Created session summary {summary_id} for session {session_id}")
                return summary_id
            return None
        except Exception as e:
            logger.error(f"Failed to create session summary for {session_id}: {e}")
            return None
    
    # ============================================================================
    # UTILITY METHODS
    # ============================================================================
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            stats = {}
            
            # Count records in each table
            tables = ['user_profiles', 'behavioral_vectors', 'authentication_decisions', 
                     'behavioral_feedback', 'session_behavioral_summary']
            
            for table in tables:
                try:
                    result = self.supabase.table(table).select('id').execute()
                    stats[f"{table}_count"] = len(result.data) if result.data else 0
                except:
                    stats[f"{table}_count"] = 0
            
            return stats
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {}
    
    async def health_check(self) -> bool:
        """Check database connectivity"""
        try:
            result = self.supabase.table('user_profiles').select('id').limit(1).execute()
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False

# Global instance
ml_db = MLSupabaseClient()
    
"""
Database Manager for ML Engine
Handles all database operations for behavioral authentication using Supabase
"""

import os
import logging
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from supabase import create_client, Client

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages database connections and operations for ML Engine using Supabase"""
    
    def __init__(self):
        # Get Supabase configuration from environment
        self.supabase_url = os.getenv('SUPABASE_URL', 'https://zuyoowgeytuqfysomovy.supabase.co')
        self.supabase_key = os.getenv('SUPABASE_SERVICE_KEY', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inp1eW9vd2dleXR1cWZ5c29tb3Z5Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1MTQyNzMwOSwiZXhwIjoyMDY3MDAzMzA5fQ.bpom1qKQCQ3Bz_XhNy9jsFQF1KlJcZoxIzRAXFqbfpE')
        self.supabase = None
        
    async def initialize(self):
        """Initialize Supabase client"""
        try:
            self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
            logger.info("Database connection initialized successfully with Supabase")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {e}")
            raise
    
    async def close(self):
        """Close database connection"""
        # Supabase client doesn't need explicit closing
        logger.info("Database connection closed")
    
    async def get_user_session_count(self, user_id: str) -> int:
        """Get user's current session count"""
        try:
            if not self.supabase:
                await self.initialize()
            
            result = self.supabase.table('users').select('sessions_count').eq('id', user_id).execute()
            if result.data and len(result.data) > 0:
                count = result.data[0].get('sessions_count', 0) or 0
                logger.info(f"User {user_id} session count: {count}")
                return count
            
            logger.warning(f"User {user_id} not found, returning 0 sessions")
            return 0
        except Exception as e:
            logger.error(f"Failed to get user session count for {user_id}: {e}")
            return 0
    
    async def increment_user_session_count(self, user_id: str) -> int:
        """Increment user's session count and return new count"""
        try:
            if not self.supabase:
                await self.initialize()
            
            # Get current count
            current_count = await self.get_user_session_count(user_id)
            new_count = current_count + 1
            
            # Update count
            result = self.supabase.table('users').update({
                'sessions_count': new_count
            }).eq('id', user_id).execute()
            
            if result.data:
                logger.info(f"Incremented session count for {user_id}: {current_count} → {new_count}")
                return new_count
            else:
                logger.error(f"Failed to update session count for {user_id}")
                return current_count
        except Exception as e:
            logger.error(f"Failed to increment session count for {user_id}: {e}")
            return 0
    
    async def store_session_vector(self, session_id: str, vector: np.ndarray) -> bool:
        """Store session vector in database"""
        try:
            if not self.supabase:
                await self.initialize()
            
            # Convert numpy array to list for PostgreSQL FLOAT8[] array
            vector_list = vector.tolist() if hasattr(vector, 'tolist') else list(vector)
            
            data = {
                'session_id': session_id,
                'vector': vector_list
            }
            
            result = self.supabase.table('session_vectors').insert(data).execute()
            
            if result.data:
                logger.info(f"Stored session vector for {session_id} (dimensions: {len(vector_list)})")
                return True
            else:
                logger.error(f"Failed to store session vector for {session_id}")
                return False
        except Exception as e:
            logger.error(f"Failed to store session vector for {session_id}: {e}")
            return False
    
    async def get_user_session_vectors(self, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get user's session vectors"""
        try:
            if not self.supabase:
                await self.initialize()
            
            # Join with sessions table to get user_id
            result = self.supabase.table('session_vectors').select(
                'session_id, vector, created_at'
            ).limit(limit).execute()
            
            vectors = []
            if result.data:
                for row in result.data:
                    vectors.append({
                        "session_id": row['session_id'],
                        "vector": np.array(row['vector']),
                        "created_at": row['created_at']
                    })
                
                logger.info(f"Retrieved {len(vectors)} session vectors")
            
            return vectors
        except Exception as e:
            logger.error(f"Failed to get session vectors for {user_id}: {e}")
            return []
    
    async def get_latest_user_vectors(self, user_id: str, limit: int = 6) -> List[np.ndarray]:
        """Get latest user vectors for clustering"""
        try:
            vectors_data = await self.get_user_session_vectors(user_id, limit)
            
            if vectors_data:
                vectors = [data['vector'] for data in vectors_data]
                logger.info(f"Retrieved {len(vectors)} latest vectors for {user_id}")
                return [np.array(v) for v in vectors]
            
            return []
        except Exception as e:
            logger.error(f"Failed to get latest vectors for {user_id}: {e}")
            return []
    
    async def store_user_clusters(self, user_id: str, clusters: List[Tuple[int, np.ndarray]]) -> bool:
        """Store user cluster centroids"""
        try:
            if not self.supabase:
                await self.initialize()
            
            # Delete existing clusters for this user
            self.supabase.table('user_clusters').delete().eq('user_id', user_id).execute()
            
            # Insert new clusters
            for cluster_label, centroid in clusters:
                centroid_list = centroid.tolist() if hasattr(centroid, 'tolist') else list(centroid)
                
                data = {
                    'user_id': user_id,
                    'cluster_label': cluster_label,
                    'centroid': centroid_list
                }
                
                result = self.supabase.table('user_clusters').insert(data).execute()
                
                if not result.data:
                    logger.error(f"Failed to store cluster {cluster_label} for {user_id}")
                    return False
            
            logger.info(f"Stored {len(clusters)} clusters for {user_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to store user clusters for {user_id}: {e}")
            return False
    
    async def get_user_clusters(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user's cluster centroids"""
        try:
            if not self.supabase:
                await self.initialize()
            
            result = self.supabase.table('user_clusters').select('*').eq('user_id', user_id).execute()
            
            clusters = []
            if result.data:
                for row in result.data:
                    clusters.append({
                        "cluster_label": row['cluster_label'],
                        "centroid": np.array(row['centroid']),
                        "created_at": row['created_at']
                    })
                
                logger.info(f"Retrieved {len(clusters)} clusters for {user_id}")
            
            return clusters
        except Exception as e:
            logger.error(f"Failed to get user clusters for {user_id}: {e}")
            return []
    
    async def find_nearest_cluster(self, user_id: str, vector: np.ndarray) -> Optional[Dict[str, Any]]:
        """Find the nearest cluster centroid for a vector"""
        try:
            clusters = await self.get_user_clusters(user_id)
            
            if not clusters:
                logger.warning(f"No clusters found for {user_id}")
                return None
            
            best_cluster = None
            best_similarity = -1.0
            
            # Normalize input vector
            vector_norm = vector / (np.linalg.norm(vector) + 1e-8)
            
            for cluster in clusters:
                centroid = cluster['centroid']
                centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-8)
                
                # Calculate cosine similarity
                similarity = np.dot(vector_norm, centroid_norm)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_cluster = {
                        **cluster,
                        'similarity': similarity,
                        'distance': 1.0 - similarity
                    }
            
            if best_cluster:
                logger.info(f"Found nearest cluster for {user_id}: similarity={best_similarity:.4f}")
            
            return best_cluster
        except Exception as e:
            logger.error(f"Failed to find nearest cluster for {user_id}: {e}")
            return None
    
    async def update_cluster_with_vector(self, cluster_id: str, new_vector: np.ndarray, alpha: float = 0.1) -> bool:
        """Update cluster centroid with new vector using exponential moving average"""
        try:
            if not self.supabase:
                await self.initialize()
            
            # Get current cluster
            result = self.supabase.table('user_clusters').select('*').eq('id', cluster_id).execute()
            
            if not result.data:
                logger.error(f"Cluster {cluster_id} not found")
                return False
            
            cluster = result.data[0]
            old_centroid = np.array(cluster['centroid'])
            
            # Update centroid using exponential moving average
            new_centroid = (1 - alpha) * old_centroid + alpha * new_vector
            new_centroid_list = new_centroid.tolist()
            
            # Update in database (no updated_at column in actual schema)
            update_result = self.supabase.table('user_clusters').update({
                'centroid': new_centroid_list
            }).eq('id', cluster_id).execute()
            
            if update_result.data:
                logger.info(f"Updated cluster {cluster_id} with new vector")
                return True
            else:
                logger.error(f"Failed to update cluster {cluster_id}")
                return False
        except Exception as e:
            logger.error(f"Failed to update cluster {cluster_id}: {e}")
            return False
    
    async def mark_session_ended(self, session_id: str) -> bool:
        """Mark session as ended"""
        try:
            if not self.supabase:
                await self.initialize()
            
            result = self.supabase.table('sessions').update({
                'ended_at': datetime.utcnow().isoformat()
            }).eq('id', session_id).execute()
            
            if result.data:
                logger.info(f"Marked session {session_id} as ended")
                return True
            else:
                logger.warning(f"Session {session_id} not found or already ended")
                return False
        except Exception as e:
            logger.error(f"Failed to mark session {session_id} as ended: {e}")
            return False
    
    async def get_total_users(self) -> int:
        """Get total number of users"""
        try:
            if not self.supabase:
                await self.initialize()
            
            result = self.supabase.table('users').select('id', count='exact').execute()
            return result.count or 0
        except Exception as e:
            logger.error(f"Failed to get total users: {e}")
            return 0
    
    async def get_total_sessions(self) -> int:
        """Get total number of sessions"""
        try:
            if not self.supabase:
                await self.initialize()
            
            result = self.supabase.table('sessions').select('id', count='exact').execute()
            return result.count or 0
        except Exception as e:
            logger.error(f"Failed to get total sessions: {e}")
            return 0
    
    async def get_system_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        try:
            total_users = await self.get_total_users()
            total_sessions = await self.get_total_sessions()
            
            return {
                'total_users': total_users,
                'total_sessions': total_sessions,
                'database_status': 'connected'
            }
        except Exception as e:
            logger.error(f"Failed to get system statistics: {e}")
            return {
                'total_users': 0,
                'total_sessions': 0,
                'database_status': 'error'
            }

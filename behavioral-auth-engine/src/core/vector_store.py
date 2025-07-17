"""
Vector storage interface and HDF5 implementation for behavioral data.
"""

import os
import h5py
import numpy as np
from typing import List, Optional, Dict, Any
from datetime import datetime
from abc import ABC, abstractmethod
import threading
from contextlib import contextmanager

from src.data.models import BehavioralVector, UserProfile
from src.config.settings import get_settings, get_vector_storage_path
from src.config.ml_config import get_ml_config
from src.utils.constants import TOTAL_VECTOR_DIM


class VectorStoreInterface(ABC):
    """Abstract interface for vector storage implementations."""
    
    @abstractmethod
    async def store_vector(self, user_id: str, vector: BehavioralVector) -> bool:
        """Store a behavioral vector for a user."""
        pass
    
    @abstractmethod
    async def get_user_vectors(self, user_id: str, limit: Optional[int] = None) -> List[BehavioralVector]:
        """Retrieve vectors for a specific user."""
        pass
    
    @abstractmethod
    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Retrieve user profile with aggregated data."""
        pass
    
    @abstractmethod
    async def update_user_profile(self, profile: UserProfile) -> bool:
        """Update or create user profile."""
        pass
    
    @abstractmethod
    async def delete_user_data(self, user_id: str) -> bool:
        """Delete all data for a user."""
        pass
    
    @abstractmethod
    async def get_similar_vectors(self, vector: List[float], user_id: str, top_k: int = 10) -> List[tuple]:
        """Find similar vectors for a user."""
        pass


class HDF5VectorStore(VectorStoreInterface):
    """HDF5-based vector storage implementation."""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path or get_vector_storage_path()
        self.settings = get_settings()
        self.ml_config = get_ml_config()
        self._locks = {}  # Per-user file locks
        self._global_lock = threading.RLock()
        
        # Ensure storage directory exists
        os.makedirs(self.storage_path, exist_ok=True)
    
    def _get_user_file_path(self, user_id: str) -> str:
        """Get the HDF5 file path for a specific user."""
        return os.path.join(self.storage_path, f"user_{user_id}.h5")
    
    def _get_user_lock(self, user_id: str) -> threading.RLock:
        """Get or create a lock for a specific user."""
        with self._global_lock:
            if user_id not in self._locks:
                self._locks[user_id] = threading.RLock()
            return self._locks[user_id]
    
    @contextmanager
    def _open_user_file(self, user_id: str, mode: str = 'r'):
        """Context manager for opening user HDF5 files with proper locking."""
        file_path = self._get_user_file_path(user_id)
        user_lock = self._get_user_lock(user_id)
        
        with user_lock:
            try:
                file_handle = h5py.File(file_path, mode)
                yield file_handle
            finally:
                if 'file_handle' in locals():
                    file_handle.close()
    
    async def store_vector(self, user_id: str, vector: BehavioralVector) -> bool:
        """Store a behavioral vector for a user."""
        try:
            with self._open_user_file(user_id, 'a') as f:
                # Create groups if they don't exist
                if 'session_vectors' not in f:
                    vectors_group = f.create_group('session_vectors')
                else:
                    vectors_group = f['session_vectors']
                
                if 'metadata' not in f:
                    metadata_group = f.create_group('metadata')
                else:
                    metadata_group = f['metadata']
                
                # Create dataset name with timestamp
                timestamp_str = vector.timestamp.strftime("%Y%m%d_%H%M%S_%f")
                dataset_name = f"{vector.session_id}_{timestamp_str}"
                
                # Store vector data
                vector_data = np.array(vector.vector, dtype=np.float32)
                vectors_group.create_dataset(
                    dataset_name, 
                    data=vector_data,
                    compression='gzip',
                    compression_opts=self.ml_config.vector_storage.compression_level
                )
                
                # Store metadata
                metadata_group.create_dataset(
                    f"{dataset_name}_meta",
                    data=np.string_(vector.model_dump_json())
                )
                
                # Update user statistics
                self._update_user_stats(f, vector)
                
            return True
            
        except Exception as e:
            # Log error here
            print(f"Error storing vector for user {user_id}: {e}")
            return False
    
    def _update_user_stats(self, file_handle: h5py.File, vector: BehavioralVector) -> None:
        """Update user statistics in the HDF5 file."""
        stats_group_name = 'user_stats'
        
        if stats_group_name not in file_handle:
            stats_group = file_handle.create_group(stats_group_name)
            # Initialize stats
            stats_group.attrs['session_count'] = 1
            stats_group.attrs['last_updated'] = vector.timestamp.isoformat()
            stats_group.attrs['user_id'] = vector.user_id
        else:
            stats_group = file_handle[stats_group_name]
            stats_group.attrs['session_count'] = stats_group.attrs.get('session_count', 0) + 1
            stats_group.attrs['last_updated'] = vector.timestamp.isoformat()
    
    async def get_user_vectors(self, user_id: str, limit: Optional[int] = None) -> List[BehavioralVector]:
        """Retrieve vectors for a specific user."""
        vectors = []
        file_path = self._get_user_file_path(user_id)
        
        if not os.path.exists(file_path):
            return vectors
        
        try:
            with self._open_user_file(user_id, 'r') as f:
                if 'session_vectors' not in f or 'metadata' not in f:
                    return vectors
                
                vectors_group = f['session_vectors']
                metadata_group = f['metadata']
                
                # Get all vector dataset names and sort by timestamp
                vector_names = list(vectors_group.keys())
                vector_names.sort(reverse=True)  # Most recent first
                
                if limit:
                    vector_names = vector_names[:limit]
                
                for vector_name in vector_names:
                    try:
                        # Load vector data
                        vector_data = vectors_group[vector_name][:]
                        
                        # Load metadata
                        meta_name = f"{vector_name}_meta"
                        if meta_name in metadata_group:
                            metadata_json = metadata_group[meta_name][()].decode('utf-8')
                            vector_obj = BehavioralVector.model_validate_json(metadata_json)
                            vectors.append(vector_obj)
                        
                    except Exception as e:
                        print(f"Error loading vector {vector_name}: {e}")
                        continue
                
        except Exception as e:
            print(f"Error retrieving vectors for user {user_id}: {e}")
        
        return vectors
    
    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Retrieve user profile with aggregated data."""
        file_path = self._get_user_file_path(user_id)
        
        if not os.path.exists(file_path):
            return None
        
        try:
            with self._open_user_file(user_id, 'r') as f:
                if 'user_stats' not in f:
                    return None
                
                stats_group = f['user_stats']
                
                # Create user profile from stored stats
                profile = UserProfile(
                    user_id=user_id,
                    session_count=stats_group.attrs.get('session_count', 0),
                    updated_at=datetime.fromisoformat(stats_group.attrs.get('last_updated', datetime.utcnow().isoformat()))
                )
                
                # Load stored profile attributes if they exist
                if 'aggregated_profile' in f:
                    profile_group = f['aggregated_profile']
                    
                    # Load stored attributes
                    if 'current_phase' in profile_group.attrs:
                        from src.data.models import SessionPhase
                        profile.current_phase = SessionPhase(profile_group.attrs['current_phase'])
                    
                    profile.risk_threshold = profile_group.attrs.get('risk_threshold', profile.risk_threshold)
                    profile.false_positive_rate = profile_group.attrs.get('false_positive_rate', 0.0)
                    profile.drift_score = profile_group.attrs.get('drift_score', 0.0)
                    
                    # Load baseline vector if available
                    if 'baseline_vector' in profile_group:
                        profile.baseline_vector = profile_group['baseline_vector'][:].tolist()
                    
                    # Load variance if available
                    if 'vector_variance' in profile_group:
                        profile.vector_variance = profile_group['vector_variance'][:].tolist()
                else:
                    # Update learning phase based on session count for new profiles
                    profile.update_phase()
                
                # Load recent vectors
                recent_vectors = await self.get_user_vectors(user_id, limit=50)
                profile.recent_vectors = recent_vectors
                
                # Calculate baseline vector if we have enough data and no stored baseline
                if len(recent_vectors) >= 5 and profile.baseline_vector is None:
                    vectors_array = np.array([v.vector for v in recent_vectors])
                    profile.baseline_vector = np.mean(vectors_array, axis=0).tolist()
                    profile.vector_variance = np.var(vectors_array, axis=0).tolist()
                
                return profile
                
        except Exception as e:
            print(f"Error retrieving profile for user {user_id}: {e}")
            return None
    
    async def update_user_profile(self, profile: UserProfile) -> bool:
        """Update or create user profile."""
        try:
            with self._open_user_file(profile.user_id, 'a') as f:
                # Store/update aggregated profile data
                if 'aggregated_profile' not in f:
                    profile_group = f.create_group('aggregated_profile')
                else:
                    profile_group = f['aggregated_profile']
                
                # Store baseline vector if available
                if profile.baseline_vector:
                    if 'baseline_vector' in profile_group:
                        del profile_group['baseline_vector']
                    profile_group.create_dataset(
                        'baseline_vector',
                        data=np.array(profile.baseline_vector, dtype=np.float32),
                        compression='gzip'
                    )
                
                # Store variance if available
                if profile.vector_variance:
                    if 'vector_variance' in profile_group:
                        del profile_group['vector_variance']
                    profile_group.create_dataset(
                        'vector_variance',
                        data=np.array(profile.vector_variance, dtype=np.float32),
                        compression='gzip'
                    )
                
                # Update attributes
                profile_group.attrs['current_phase'] = profile.current_phase.value
                profile_group.attrs['risk_threshold'] = profile.risk_threshold
                profile_group.attrs['false_positive_rate'] = profile.false_positive_rate
                profile_group.attrs['drift_score'] = profile.drift_score
                profile_group.attrs['updated_at'] = profile.updated_at.isoformat()
                
            return True
            
        except Exception as e:
            print(f"Error updating profile for user {profile.user_id}: {e}")
            return False
    
    async def delete_user_data(self, user_id: str) -> bool:
        """Delete all data for a user."""
        file_path = self._get_user_file_path(user_id)
        
        try:
            if os.path.exists(file_path):
                user_lock = self._get_user_lock(user_id)
                with user_lock:
                    os.remove(file_path)
                
                # Clean up lock
                with self._global_lock:
                    if user_id in self._locks:
                        del self._locks[user_id]
            
            return True
            
        except Exception as e:
            print(f"Error deleting data for user {user_id}: {e}")
            return False
    
    async def get_similar_vectors(self, vector: List[float], user_id: str, top_k: int = 10) -> List[tuple]:
        """Find similar vectors for a user using cosine similarity."""
        similar_vectors = []
        
        try:
            user_vectors = await self.get_user_vectors(user_id, limit=100)  # Get recent vectors
            
            if not user_vectors:
                return similar_vectors
            
            query_vector = np.array(vector)
            similarities = []
            
            for stored_vector in user_vectors:
                stored_array = np.array(stored_vector.vector)
                
                # Calculate cosine similarity
                similarity = np.dot(query_vector, stored_array) / (
                    np.linalg.norm(query_vector) * np.linalg.norm(stored_array)
                )
                
                similarities.append((stored_vector, float(similarity)))
            
            # Sort by similarity (highest first) and return top_k
            similarities.sort(key=lambda x: x[1], reverse=True)
            similar_vectors = similarities[:top_k]
            
        except Exception as e:
            print(f"Error finding similar vectors for user {user_id}: {e}")
        
        return similar_vectors
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        stats = {
            'total_users': 0,
            'total_size_mb': 0,
            'storage_path': self.storage_path
        }
        
        try:
            for filename in os.listdir(self.storage_path):
                if filename.endswith('.h5') and filename.startswith('user_'):
                    stats['total_users'] += 1
                    file_path = os.path.join(self.storage_path, filename)
                    stats['total_size_mb'] += os.path.getsize(file_path) / (1024 * 1024)
            
        except Exception as e:
            print(f"Error getting storage stats: {e}")
        
        return stats

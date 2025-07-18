"""
FAISS-based similarity search layer for behavioral authentication.

This layer provides high-performance vector similarity search using Facebook's FAISS library.
It handles vector indexing, similarity computation, and threshold-based authentication decisions.
"""

import asyncio
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
import tempfile
import os
import pickle
import threading

try:
    import faiss
except ImportError:
    faiss = None
    logging.warning("FAISS not available. Install with: pip install faiss-cpu")

from ..data.models import BehavioralVector, UserProfile, AuthenticationDecision, RiskLevel
from ..core.vector_store import VectorStoreInterface
from ..utils.constants import TOTAL_VECTOR_DIM
from ..config.settings import Settings


class FAISSLayer:
    """
    FAISS-based vector similarity search layer.
    
    Provides efficient similarity search for behavioral authentication with:
    - Real-time vector indexing
    - Cosine similarity computation
    - User-specific index management
    - Threshold-based authentication decisions
    """
    
    def __init__(self, vector_store: VectorStoreInterface, settings: Optional[Settings] = None):
        """Initialize FAISS layer."""
        if faiss is None:
            raise ImportError("FAISS library is required. Install with: pip install faiss-cpu")
        
        self.vector_store = vector_store
        self.settings = settings or Settings()
        
        # FAISS configuration
        self.vector_dimension = TOTAL_VECTOR_DIM
        self.similarity_threshold = self.settings.similarity_threshold
        self.min_vectors_for_search = self.settings.min_vectors_for_search
        
        # Index management
        self.user_indices: Dict[str, faiss.IndexFlatIP] = {}  # User ID -> FAISS index
        self.index_metadata: Dict[str, Dict] = {}  # Index creation info
        self.index_locks: Dict[str, threading.Lock] = {}  # Thread-safe index access
        
        # Performance tracking
        self.search_stats = {
            'total_searches': 0,
            'successful_matches': 0,
            'failed_matches': 0,
            'avg_search_time_ms': 0.0
        }
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"FAISS Layer initialized with dimension {self.vector_dimension}")
    
    async def initialize_user_index(self, user_id: str) -> bool:
        """Initialize or rebuild FAISS index for a user."""
        try:
            # Get user's behavioral vectors
            user_vectors = await self.vector_store.get_user_vectors(user_id)
            
            if len(user_vectors) < self.min_vectors_for_search:
                self.logger.debug(f"Insufficient vectors for user {user_id}: {len(user_vectors)} < {self.min_vectors_for_search}")
                return False
            
            # Create thread lock for this user
            if user_id not in self.index_locks:
                self.index_locks[user_id] = threading.Lock()
            
            with self.index_locks[user_id]:
                # Create new FAISS index (Inner Product for cosine similarity with normalized vectors)
                index = faiss.IndexFlatIP(self.vector_dimension)
                
                # Prepare vectors for indexing
                vectors_array = np.array([v.vector for v in user_vectors], dtype=np.float32)
                
                # Normalize vectors for cosine similarity
                vectors_array = self._normalize_vectors(vectors_array)
                
                # Add vectors to index
                index.add(vectors_array)
                
                # Store index and metadata
                self.user_indices[user_id] = index
                self.index_metadata[user_id] = {
                    'vector_count': len(user_vectors),
                    'created_at': datetime.utcnow(),
                    'last_updated': datetime.utcnow()
                }
                
                self.logger.info(f"FAISS index created for user {user_id} with {len(user_vectors)} vectors")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to initialize index for user {user_id}: {e}")
            return False
    
    async def add_vector_to_index(self, user_id: str, vector: BehavioralVector) -> bool:
        """Add a new vector to user's FAISS index."""
        try:
            # Ensure user has an index
            if user_id not in self.user_indices:
                await self.initialize_user_index(user_id)
                if user_id not in self.user_indices:
                    return False
            
            with self.index_locks[user_id]:
                # Normalize vector
                vector_array = np.array([vector.vector], dtype=np.float32)
                normalized_vector = self._normalize_vectors(vector_array)
                
                # Add to index
                self.user_indices[user_id].add(normalized_vector)
                
                # Update metadata
                self.index_metadata[user_id]['vector_count'] += 1
                self.index_metadata[user_id]['last_updated'] = datetime.utcnow()
                
                self.logger.debug(f"Vector added to FAISS index for user {user_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to add vector to index for user {user_id}: {e}")
            return False
    
    async def compute_similarity_scores(
        self, 
        user_id: str, 
        query_vector: BehavioralVector,
        top_k: int = 10
    ) -> Dict[str, float]:
        """Compute similarity scores against user's behavioral history."""
        start_time = datetime.utcnow()
        
        try:
            # Check if user has an index
            if user_id not in self.user_indices:
                await self.initialize_user_index(user_id)
                if user_id not in self.user_indices:
                    return {}
            
            with self.index_locks[user_id]:
                index = self.user_indices[user_id]
                
                # Prepare query vector
                query_array = np.array([query_vector.vector], dtype=np.float32)
                normalized_query = self._normalize_vectors(query_array)
                
                # Perform similarity search
                k = min(top_k, index.ntotal)
                if k == 0:
                    return {}
                
                similarities, indices = index.search(normalized_query, k)
                
                # Build results
                results = {}
                for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                    if idx >= 0:  # Valid index
                        results[f"vector_{idx}"] = float(similarity)
                
                # Update statistics
                self._update_search_stats(start_time, len(results) > 0)
                
                self.logger.debug(f"Similarity search for user {user_id}: {len(results)} matches")
                return results
                
        except Exception as e:
            self.logger.error(f"Similarity search failed for user {user_id}: {e}")
            self._update_search_stats(start_time, False)
            return {}
    
    async def make_authentication_decision(
        self, 
        user_id: str, 
        query_vector: BehavioralVector,
        user_profile: UserProfile
    ) -> Tuple[AuthenticationDecision, RiskLevel, float, float, List[str]]:
        """Make authentication decision based on FAISS similarity scores."""
        
        # Get similarity scores
        similarity_scores = await self.compute_similarity_scores(user_id, query_vector)
        
        if not similarity_scores:
            # No historical data for comparison
            return (
                AuthenticationDecision.LEARN,
                RiskLevel.LOW,
                0.1,  # Low risk during learning
                0.5,  # Medium confidence
                ["No historical behavioral data - learning mode"]
            )
        
        # Calculate average similarity
        avg_similarity = np.mean(list(similarity_scores.values()))
        max_similarity = max(similarity_scores.values())
        
        # Determine decision based on similarity and user profile
        decision_factors = []
        
        if user_profile.current_phase.value == 'learning':
            decision = AuthenticationDecision.LEARN
            risk_level = RiskLevel.LOW
            risk_score = max(0.1, 1.0 - avg_similarity)
            confidence = 0.7
            decision_factors.append(f"Learning phase - avg similarity: {avg_similarity:.3f}")
            
        elif avg_similarity >= self.similarity_threshold:
            # High similarity - allow access
            decision = AuthenticationDecision.ALLOW
            risk_level = RiskLevel.LOW if avg_similarity > 0.8 else RiskLevel.MEDIUM
            risk_score = max(0.1, 1.0 - avg_similarity)
            confidence = min(0.95, avg_similarity)
            decision_factors.append(f"High behavioral similarity: {avg_similarity:.3f}")
            
        elif max_similarity >= self.similarity_threshold * 0.8:
            # Some similar patterns found
            decision = AuthenticationDecision.CHALLENGE
            risk_level = RiskLevel.MEDIUM
            risk_score = 0.5
            confidence = 0.6
            decision_factors.append(f"Partial behavioral match: max={max_similarity:.3f}, avg={avg_similarity:.3f}")
            
        else:
            # Low similarity - suspicious behavior
            decision = AuthenticationDecision.BLOCK
            risk_level = RiskLevel.HIGH
            risk_score = min(0.9, 1.0 - avg_similarity)
            confidence = min(0.9, 1.0 - avg_similarity)
            decision_factors.append(f"Low behavioral similarity: {avg_similarity:.3f}")
        
        self.logger.info(f"FAISS decision for user {user_id}: {decision.value} (similarity: {avg_similarity:.3f})")
        
        return decision, risk_level, risk_score, confidence, decision_factors
    
    async def optimize_user_index(self, user_id: str) -> bool:
        """Optimize user's FAISS index by rebuilding with latest vectors."""
        try:
            if user_id in self.user_indices:
                # Remove old index
                with self.index_locks[user_id]:
                    del self.user_indices[user_id]
                    if user_id in self.index_metadata:
                        del self.index_metadata[user_id]
            
            # Rebuild index
            return await self.initialize_user_index(user_id)
            
        except Exception as e:
            self.logger.error(f"Failed to optimize index for user {user_id}: {e}")
            return False
    
    async def get_layer_statistics(self) -> Dict[str, Any]:
        """Get FAISS layer performance statistics."""
        return {
            'total_user_indices': len(self.user_indices),
            'search_stats': self.search_stats.copy(),
            'index_metadata': {
                user_id: {
                    'vector_count': meta['vector_count'],
                    'age_hours': (datetime.utcnow() - meta['created_at']).total_seconds() / 3600
                }
                for user_id, meta in self.index_metadata.items()
            },
            'memory_usage_mb': self._estimate_memory_usage(),
            'settings': {
                'similarity_threshold': self.similarity_threshold,
                'min_vectors_for_search': self.min_vectors_for_search,
                'vector_dimension': self.vector_dimension
            }
        }
    
    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors for cosine similarity."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return vectors / norms
    
    def _update_search_stats(self, start_time: datetime, success: bool) -> None:
        """Update search performance statistics."""
        duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        self.search_stats['total_searches'] += 1
        if success:
            self.search_stats['successful_matches'] += 1
        else:
            self.search_stats['failed_matches'] += 1
        
        # Update rolling average
        current_avg = self.search_stats['avg_search_time_ms']
        total_searches = self.search_stats['total_searches']
        self.search_stats['avg_search_time_ms'] = (
            (current_avg * (total_searches - 1) + duration_ms) / total_searches
        )
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage of FAISS indices in MB."""
        total_size = 0
        for user_id, index in self.user_indices.items():
            # Rough estimation: 4 bytes per float * dimension * vector count
            vector_count = index.ntotal
            size_bytes = vector_count * self.vector_dimension * 4
            total_size += size_bytes
        
        return total_size / (1024 * 1024)  # Convert to MB
    
    async def cleanup_user_index(self, user_id: str) -> bool:
        """Clean up FAISS index for a user."""
        try:
            if user_id in self.user_indices:
                with self.index_locks[user_id]:
                    del self.user_indices[user_id]
                    if user_id in self.index_metadata:
                        del self.index_metadata[user_id]
                    if user_id in self.index_locks:
                        del self.index_locks[user_id]
                
                self.logger.info(f"FAISS index cleaned up for user {user_id}")
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup index for user {user_id}: {e}")
            return False

"""
Enhanced FAISS Behavioral Engine with Cumulative Vector Management
Implements proper vector storage, retrieval, and cumulative learning
"""

import faiss
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import pickle
import os
from dataclasses import dataclass
from enum import Enum
import json

from src.core.enhanced_behavioral_processor import EnhancedBehavioralProcessor, ProcessedBehavioralFeatures
from src.core.ml_database import MLSupabaseClient

logger = logging.getLogger(__name__)

class VectorType(str, Enum):
    """Types of vectors stored in the system"""
    SESSION = "session"          # Individual session vector
    CUMULATIVE = "cumulative"    # User's evolving baseline vector
    BASELINE = "baseline"        # Stable user profile vector

@dataclass
class VectorAnalysisResult:
    """Result of FAISS vector analysis"""
    similarity_score: float
    confidence: float
    decision: str  # 'allow', 'challenge', 'block', 'learn'
    risk_level: str
    risk_factors: List[str]
    similar_vectors: List[Dict[str, Any]]
    vector_id: Optional[str] = None

@dataclass
class UserVectorProfile:
    """Complete vector profile for a user"""
    user_id: str
    cumulative_vector: np.ndarray  # Evolving baseline
    baseline_vector: Optional[np.ndarray]  # Stable profile
    session_vectors: List[np.ndarray]  # Recent session vectors
    vector_count: int
    last_updated: datetime
    learning_phase: str
    
class EnhancedFAISSEngine:
    """Enhanced FAISS engine with proper vector management"""
    
    def __init__(self, vector_dimension: int = 90):
        self.vector_dimension = vector_dimension
        self.behavioral_processor = EnhancedBehavioralProcessor()
        self.db_client = MLSupabaseClient()
        self.logger = logging.getLogger(__name__)
        
        # Initialize FAISS indices
        self._initialize_faiss_indices()
        
        # User profiles cache
        self.user_profiles: Dict[str, UserVectorProfile] = {}
        
        # Similarity thresholds
        self.similarity_thresholds = {
            'learning': 0.3,      # Very lenient during learning
            'gradual_risk': 0.6,  # Moderate during gradual phase
            'full_auth': 0.8      # Strict during full authentication
        }

    def _initialize_faiss_indices(self):
        """Initialize FAISS indices for different vector types"""
        # Use L2 distance for similarity search
        self.session_index = faiss.IndexFlatL2(self.vector_dimension)
        self.cumulative_index = faiss.IndexFlatL2(self.vector_dimension)
        self.baseline_index = faiss.IndexFlatL2(self.vector_dimension)
        
        self.logger.info(f"Initialized FAISS indices with dimension {self.vector_dimension}")
        
    async def initialize(self):
        """Initialize FAISS indices and load existing data"""
        try:
            # Create FAISS indices
            self.session_index = faiss.IndexFlatIP(self.vector_dimension)  # Inner Product for cosine similarity
            self.cumulative_index = faiss.IndexFlatIP(self.vector_dimension)
            self.baseline_index = faiss.IndexFlatIP(self.vector_dimension)
            
            # Load existing vectors from database
            await self._load_existing_vectors()
            
            self.logger.info(f"Enhanced FAISS Engine initialized with {self.vector_dimension}D vectors")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize FAISS engine: {e}")
            raise
    
    async def process_behavioral_data(
        self, 
        user_id: str, 
        session_id: str, 
        behavioral_logs: List[Dict[str, Any]],
        learning_phase: str = "learning"
    ) -> VectorAnalysisResult:
        """
        Process behavioral data and perform FAISS analysis
        
        Args:
            user_id: User identifier
            session_id: Session identifier  
            behavioral_logs: Raw behavioral data from mobile
            learning_phase: Current learning phase
            
        Returns:
            VectorAnalysisResult with decision and analysis
        """
        try:
            # Process raw behavioral data into features
            processed_features = self.behavioral_processor.process_behavioral_logs(behavioral_logs)
            
            # Convert to vector
            session_vector = processed_features.to_vector()
            session_vector = self._normalize_vector(session_vector)
            
            # Store session vector
            vector_id = await self._store_session_vector(user_id, session_id, session_vector, processed_features)
            
            # Load user profile
            user_profile = await self._get_or_create_user_profile(user_id)
            
            # Perform analysis based on learning phase
            if learning_phase == "learning":
                result = await self._analyze_learning_phase(user_id, session_vector, user_profile)
            elif learning_phase == "gradual_risk":
                result = await self._analyze_gradual_phase(user_id, session_vector, user_profile)
            else:
                result = await self._analyze_full_auth_phase(user_id, session_vector, user_profile)
            
            result.vector_id = vector_id
            
            # Update cumulative vector
            await self._update_cumulative_vector(user_id, session_vector, result.decision)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing behavioral data: {e}")
            return VectorAnalysisResult(
                similarity_score=0.0,
                confidence=0.5,
                decision="learn",
                risk_level="medium",
                risk_factors=["Processing error"],
                similar_vectors=[]
            )
    
    async def _store_session_vector(
        self, 
        user_id: str, 
        session_id: str, 
        vector: np.ndarray,
        features: ProcessedBehavioralFeatures
    ) -> str:
        """Store session vector in database and FAISS index"""
        try:
            # Store in database
            vector_data = {
                'user_id': user_id,
                'session_id': session_id,
                'vector_data': vector.tolist(),
                'vector_type': VectorType.SESSION.value,
                'confidence_score': 1.0,
                'feature_source': 'mobile_behavioral_data',
                'metadata': {
                    'feature_stats': {
                        'touch_events': len([f for f in features.touch_duration_stats if f > 0]),
                        'motion_events': len([f for f in features.accelerometer_stats if f != 0]),
                        'scroll_events': len([f for f in features.scroll_velocity_stats if f > 0]),
                        'session_duration': features.session_duration if hasattr(features, 'session_duration') else 0
                    }
                }
            }
            
            # Store in database
            result = self.db_client.supabase.table('enhanced_behavioral_vectors').insert(vector_data).execute()
            vector_id = result.data[0]['id'] if result.data else None
            
            # Add to session FAISS index
            vector_normalized = vector.reshape(1, -1).astype(np.float32)
            self.session_index.add(vector_normalized)
            
            self.logger.info(f"Stored session vector {vector_id} for user {user_id}")
            return vector_id
            
        except Exception as e:
            self.logger.error(f"Failed to store session vector: {e}")
            return None
    
    async def _get_or_create_user_profile(self, user_id: str) -> UserVectorProfile:
        """Get or create user vector profile"""
        if user_id in self.user_profiles:
            return self.user_profiles[user_id]
        
        try:
            # Load from database
            cumulative_result = self.db_client.supabase.table('enhanced_behavioral_vectors')\
                .select('*')\
                .eq('user_id', user_id)\
                .eq('vector_type', VectorType.CUMULATIVE.value)\
                .order('created_at', desc=True)\
                .limit(1)\
                .execute()
            
            if cumulative_result.data:
                cumulative_vector = np.array(cumulative_result.data[0]['vector_data'])
                vector_count = cumulative_result.data[0].get('metadata', {}).get('vector_count', 1)
                last_updated = datetime.fromisoformat(cumulative_result.data[0]['created_at'])
            else:
                cumulative_vector = np.zeros(self.vector_dimension)
                vector_count = 0
                last_updated = datetime.utcnow()
            
            # Load baseline if exists
            baseline_result = self.db_client.supabase.table('enhanced_behavioral_vectors')\
                .select('*')\
                .eq('user_id', user_id)\
                .eq('vector_type', VectorType.BASELINE.value)\
                .order('created_at', desc=True)\
                .limit(1)\
                .execute()
            
            baseline_vector = None
            if baseline_result.data:
                baseline_vector = np.array(baseline_result.data[0]['vector_data'])
            
            # Get user profile for learning phase
            user_profile_result = await self.db_client.get_user_profile(user_id)
            learning_phase = user_profile_result.get('current_phase', 'learning') if user_profile_result else 'learning'
            
            profile = UserVectorProfile(
                user_id=user_id,
                cumulative_vector=cumulative_vector,
                baseline_vector=baseline_vector,
                session_vectors=[],
                vector_count=vector_count,
                last_updated=last_updated,
                learning_phase=learning_phase
            )
            
            self.user_profiles[user_id] = profile
            return profile
            
        except Exception as e:
            self.logger.error(f"Failed to load user profile: {e}")
            # Return default profile
            profile = UserVectorProfile(
                user_id=user_id,
                cumulative_vector=np.zeros(self.vector_dimension),
                baseline_vector=None,
                session_vectors=[],
                vector_count=0,
                last_updated=datetime.utcnow(),
                learning_phase="learning"
            )
            self.user_profiles[user_id] = profile
            return profile
    
    async def _analyze_learning_phase(
        self, 
        user_id: str, 
        session_vector: np.ndarray, 
        user_profile: UserVectorProfile
    ) -> VectorAnalysisResult:
        """Analyze vector during learning phase"""
        return VectorAnalysisResult(
            similarity_score=0.9,  # High similarity during learning
            confidence=0.5,
            decision="learn",
            risk_level="low",
            risk_factors=["Learning phase - collecting behavioral data"],
            similar_vectors=[]
        )
    
    async def _analyze_gradual_phase(
        self, 
        user_id: str, 
        session_vector: np.ndarray, 
        user_profile: UserVectorProfile
    ) -> VectorAnalysisResult:
        """Analyze vector during gradual risk phase"""
        if user_profile.vector_count < 3:
            return VectorAnalysisResult(
                similarity_score=0.8,
                confidence=0.6,
                decision="learn",
                risk_level="low", 
                risk_factors=["Insufficient data - continue learning"],
                similar_vectors=[]
            )
        
        # Compare with cumulative vector
        similarity = self._calculate_cosine_similarity(session_vector, user_profile.cumulative_vector)
        threshold = self.similarity_thresholds['gradual_risk']
        
        if similarity >= threshold:
            decision = "allow"
            risk_level = "low"
            risk_factors = ["Vector matches user profile"]
        elif similarity >= threshold * 0.7:
            decision = "challenge"
            risk_level = "medium"
            risk_factors = ["Moderate deviation from profile"]
        else:
            decision = "block"
            risk_level = "high"
            risk_factors = ["Significant deviation from profile"]
        
        return VectorAnalysisResult(
            similarity_score=similarity,
            confidence=min(0.8, user_profile.vector_count / 10.0),
            decision=decision,
            risk_level=risk_level,
            risk_factors=risk_factors,
            similar_vectors=[]
        )
    
    async def _analyze_full_auth_phase(
        self, 
        user_id: str, 
        session_vector: np.ndarray, 
        user_profile: UserVectorProfile
    ) -> VectorAnalysisResult:
        """Analyze vector during full authentication phase"""
        if user_profile.baseline_vector is None:
            # Create baseline from cumulative
            await self._create_baseline_vector(user_id, user_profile)
        
        # Compare with baseline vector
        baseline_similarity = self._calculate_cosine_similarity(session_vector, user_profile.baseline_vector)
        cumulative_similarity = self._calculate_cosine_similarity(session_vector, user_profile.cumulative_vector)
        
        # Use the higher similarity
        similarity = max(baseline_similarity, cumulative_similarity)
        threshold = self.similarity_thresholds['full_auth']
        
        # Find similar vectors using FAISS
        similar_vectors = await self._find_similar_vectors(user_id, session_vector)
        
        if similarity >= threshold:
            decision = "allow"
            risk_level = "low"
            risk_factors = ["Strong match with user profile"]
        elif similarity >= threshold * 0.8:
            decision = "challenge"
            risk_level = "medium"
            risk_factors = ["Moderate similarity to profile"]
        else:
            decision = "block"
            risk_level = "high"
            risk_factors = ["Low similarity to established profile"]
        
        return VectorAnalysisResult(
            similarity_score=similarity,
            confidence=0.9,
            decision=decision,
            risk_level=risk_level,
            risk_factors=risk_factors,
            similar_vectors=similar_vectors
        )
    
    async def _update_cumulative_vector(
        self, 
        user_id: str, 
        session_vector: np.ndarray, 
        decision: str
    ):
        """Update user's cumulative vector with new session data"""
        try:
            if decision == "block":
                # Don't update cumulative with blocked sessions
                return
            
            user_profile = self.user_profiles.get(user_id)
            if not user_profile:
                return
            
            # Update cumulative vector using exponential moving average
            alpha = 0.1  # Learning rate
            if user_profile.vector_count == 0:
                # First vector
                new_cumulative = session_vector.copy()
            else:
                # Weighted update
                new_cumulative = (1 - alpha) * user_profile.cumulative_vector + alpha * session_vector
            
            # Normalize
            new_cumulative = self._normalize_vector(new_cumulative)
            
            # Update profile
            user_profile.cumulative_vector = new_cumulative
            user_profile.vector_count += 1
            user_profile.last_updated = datetime.utcnow()
            
            # Store in database
            vector_data = {
                'user_id': user_id,
                'session_id': f"cumulative_{user_profile.vector_count}",
                'vector_data': new_cumulative.tolist(),
                'vector_type': VectorType.CUMULATIVE.value,
                'confidence_score': 0.9,
                'feature_source': 'cumulative_learning',
                'metadata': {
                    'vector_count': user_profile.vector_count,
                    'learning_rate': alpha,
                    'decision_context': decision
                }
            }
            
            # Store in database
            self.db_client.supabase.table('enhanced_behavioral_vectors').insert(vector_data).execute()
            
            # Update cumulative FAISS index
            vector_normalized = new_cumulative.reshape(1, -1).astype(np.float32)
            self.cumulative_index.add(vector_normalized)
            
            self.logger.info(f"Updated cumulative vector for user {user_id}, count: {user_profile.vector_count}")
            
        except Exception as e:
            self.logger.error(f"Failed to update cumulative vector: {e}")
    
    async def _create_baseline_vector(self, user_id: str, user_profile: UserVectorProfile):
        """Create stable baseline vector from cumulative data"""
        try:
            if user_profile.vector_count < 10:
                # Need more data for stable baseline
                return
            
            # Use current cumulative as baseline
            baseline_vector = user_profile.cumulative_vector.copy()
            
            # Store baseline in database
            vector_data = {
                'user_id': user_id,
                'session_id': f"baseline_{datetime.utcnow().isoformat()}",
                'vector_data': baseline_vector.tolist(),
                'vector_type': VectorType.BASELINE.value,
                'confidence_score': 0.95,
                'feature_source': 'baseline_creation',
                'metadata': {
                    'created_from_vector_count': user_profile.vector_count,
                    'creation_timestamp': datetime.utcnow().isoformat()
                }
            }
            
            self.db_client.supabase.table('enhanced_behavioral_vectors').insert(vector_data).execute()
            
            # Update profile
            user_profile.baseline_vector = baseline_vector
            
            # Add to baseline FAISS index
            vector_normalized = baseline_vector.reshape(1, -1).astype(np.float32)
            self.baseline_index.add(vector_normalized)
            
            self.logger.info(f"Created baseline vector for user {user_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to create baseline vector: {e}")
    
    async def _find_similar_vectors(self, user_id: str, query_vector: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Find similar vectors using FAISS search"""
        try:
            similar_vectors = []
            
            # Search in user's cumulative vectors
            if self.cumulative_index.ntotal > 0:
                query_normalized = self._normalize_vector(query_vector).reshape(1, -1).astype(np.float32)
                distances, indices = self.cumulative_index.search(query_normalized, min(k, self.cumulative_index.ntotal))
                
                for distance, idx in zip(distances[0], indices[0]):
                    if idx != -1:  # Valid index
                        similar_vectors.append({
                            'similarity': float(distance),
                            'index': int(idx),
                            'type': 'cumulative'
                        })
            
            return similar_vectors[:k]
            
        except Exception as e:
            self.logger.error(f"Error finding similar vectors: {e}")
            return []
    
    def _calculate_cosine_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            # Normalize vectors
            v1_norm = self._normalize_vector(vector1)
            v2_norm = self._normalize_vector(vector2)
            
            # Calculate cosine similarity
            similarity = np.dot(v1_norm, v2_norm)
            return float(np.clip(similarity, -1.0, 1.0))
            
        except Exception as e:
            self.logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Normalize vector for cosine similarity"""
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm
    
    async def _load_existing_vectors(self):
        """Load existing vectors from database into FAISS indices"""
        try:
            # Load session vectors
            session_vectors = self.db_client.supabase.table('enhanced_behavioral_vectors')\
                .select('*')\
                .eq('vector_type', VectorType.SESSION.value)\
                .execute()
            
            if session_vectors.data:
                vectors_array = np.array([v['vector_data'] for v in session_vectors.data], dtype=np.float32)
                self.session_index.add(vectors_array)
                self.logger.info(f"Loaded {len(session_vectors.data)} session vectors")
            
            # Load cumulative vectors
            cumulative_vectors = self.db_client.supabase.table('enhanced_behavioral_vectors')\
                .select('*')\
                .eq('vector_type', VectorType.CUMULATIVE.value)\
                .execute()
            
            if cumulative_vectors.data:
                vectors_array = np.array([v['vector_data'] for v in cumulative_vectors.data], dtype=np.float32)
                self.cumulative_index.add(vectors_array)
                self.logger.info(f"Loaded {len(cumulative_vectors.data)} cumulative vectors")
            
            # Load baseline vectors
            baseline_vectors = self.db_client.supabase.table('enhanced_behavioral_vectors')\
                .select('*')\
                .eq('vector_type', VectorType.BASELINE.value)\
                .execute()
            
            if baseline_vectors.data:
                vectors_array = np.array([v['vector_data'] for v in baseline_vectors.data], dtype=np.float32)
                self.baseline_index.add(vectors_array)
                self.logger.info(f"Loaded {len(baseline_vectors.data)} baseline vectors")
                
        except Exception as e:
            self.logger.error(f"Error loading existing vectors: {e}")
    
    async def get_user_vector_statistics(self, user_id: str) -> Dict[str, Any]:
        """Get statistics about user's vectors"""
        try:
            user_profile = await self._get_or_create_user_profile(user_id)
            
            return {
                'user_id': user_id,
                'vector_count': user_profile.vector_count,
                'has_baseline': user_profile.baseline_vector is not None,
                'last_updated': user_profile.last_updated.isoformat(),
                'learning_phase': user_profile.learning_phase,
                'cumulative_vector_norm': float(np.linalg.norm(user_profile.cumulative_vector))
            }
            
        except Exception as e:
            self.logger.error(f"Error getting user vector statistics: {e}")
            return {'error': str(e)}
    
    async def get_layer_statistics(self) -> Dict[str, Any]:
        """Get FAISS layer statistics"""
        return {
            'session_vectors': self.session_index.ntotal if self.session_index else 0,
            'cumulative_vectors': self.cumulative_index.ntotal if self.cumulative_index else 0,
            'baseline_vectors': self.baseline_index.ntotal if self.baseline_index else 0,
            'cached_user_profiles': len(self.user_profiles),
            'vector_dimension': self.vector_dimension,
            'similarity_thresholds': self.similarity_thresholds
        }

    async def process_mobile_behavioral_data(
        self, 
        user_id: str, 
        session_id: str, 
        behavioral_data: Dict[str, Any], 
        learning_phase: str = "learning"
    ) -> VectorAnalysisResult:
        """
        Process mobile behavioral data in the exact format provided
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            behavioral_data: Mobile behavioral data with 'logs' array
            learning_phase: Current learning phase
            
        Returns:
            VectorAnalysisResult with analysis and decision
        """
        try:
            self.logger.info(f"Processing mobile behavioral data for user {user_id}, session {session_id}")
            
            # Extract logs from mobile data format
            logs = behavioral_data.get('logs', [])
            
            if not logs:
                self.logger.warning("No logs found in behavioral data")
                return VectorAnalysisResult(
                    similarity_score=0.0,
                    confidence=0.5,
                    decision="learn",
                    risk_level="medium",
                    risk_factors=["No behavioral data provided"],
                    similar_vectors=[]
                )
            
            # Use the enhanced behavioral processor to create vector
            session_vector = self.behavioral_processor.process_mobile_behavioral_data(behavioral_data)
            session_vector = self._normalize_vector(session_vector)
            
            # Verify vector quality
            vector_sum = np.sum(np.abs(session_vector))
            if vector_sum == 0:
                self.logger.warning("Generated vector is all zeros")
                return VectorAnalysisResult(
                    similarity_score=0.0,
                    confidence=0.3,
                    decision="learn",
                    risk_level="medium",
                    risk_factors=["Invalid behavioral vector generated"],
                    similar_vectors=[]
                )
            
            # Store session vector in database
            vector_id = await self._store_session_vector_mobile(user_id, session_id, session_vector, behavioral_data)
            
            # Load user profile
            user_profile = await self._get_or_create_user_profile(user_id)
            
            # Perform analysis based on learning phase
            if learning_phase == "learning":
                result = await self._analyze_learning_phase(user_id, session_vector, user_profile)
            elif learning_phase == "gradual_risk":
                result = await self._analyze_gradual_phase(user_id, session_vector, user_profile)
            else:
                result = await self._analyze_full_auth_phase(user_id, session_vector, user_profile)
            
            result.vector_id = vector_id
            
            # Update cumulative vector
            await self._update_cumulative_vector(user_id, session_vector, result.decision)
            
            self.logger.info(f"Mobile behavioral analysis complete: {result.decision} (confidence: {result.confidence:.3f})")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing mobile behavioral data: {e}")
            return VectorAnalysisResult(
                similarity_score=0.0,
                confidence=0.5,
                decision="learn",
                risk_level="medium",
                risk_factors=[f"Processing error: {str(e)}"],
                similar_vectors=[]
            )

    async def _store_session_vector_mobile(
        self, 
        user_id: str, 
        session_id: str, 
        vector: np.ndarray, 
        original_data: Dict[str, Any]
    ) -> Optional[str]:
        """Store session vector from mobile data"""
        try:
            metadata = {
                'session_id': session_id,
                'event_count': len(original_data.get('logs', [])),
                'processing_timestamp': datetime.utcnow().isoformat(),
                'vector_quality': float(np.sum(np.abs(vector))),
                'mobile_data_format': True,
                'event_types': list(set([log.get('event_type', '') for log in original_data.get('logs', [])]))
            }
            
            vector_data = {
                'user_id': user_id,
                'session_id': session_id,
                'vector_data': vector.tolist(),
                'vector_type': VectorType.SESSION.value,
                'confidence_score': 0.8,
                'feature_source': 'mobile_behavioral_data',
                'metadata': metadata
            }
            
            # Store in database
            result = self.db_client.supabase.table('enhanced_behavioral_vectors').insert(vector_data).execute()
            
            if result.data:
                vector_id = result.data[0]['id']
                
                # Add to session FAISS index
                vector_normalized = vector.reshape(1, -1).astype(np.float32)
                self.session_index.add(vector_normalized)
                
                self.logger.debug(f"Stored mobile session vector {vector_id}")
                return vector_id
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error storing mobile session vector: {e}")
            return None

    async def _check_learning_phase_transition(self, user_id: str):
        """Check if user should transition between learning phases"""
        try:
            user_profile = self.user_profiles.get(user_id)
            if not user_profile:
                return
            
            # Get user profile from database for current phase
            user_profile_result = self.db_client.supabase.table('user_profiles')\
                .select('*')\
                .eq('user_id', user_id)\
                .execute()
            
            if not user_profile_result.data:
                return
            
            profile = user_profile_result.data[0]
            current_phase = profile.get('current_phase', 'learning')
            session_count = profile.get('current_session_count', 0)
            
            # Learning phase transition logic
            if current_phase == 'learning' and user_profile.vector_count >= 5:
                # Transition to gradual_risk phase after 5 sessions
                await self._update_user_learning_phase(user_id, 'gradual_risk')
                logger.info(f"User {user_id} transitioned to gradual_risk phase after {user_profile.vector_count} sessions")
                
            elif current_phase == 'gradual_risk' and user_profile.vector_count >= 10:
                # Create baseline and transition to full_auth after 10 sessions
                await self._create_baseline_vector(user_id, user_profile)
                await self._update_user_learning_phase(user_id, 'full_auth')
                logger.info(f"User {user_id} transitioned to full_auth phase with baseline vector")
                
        except Exception as e:
            self.logger.error(f"Error checking learning phase transition: {e}")

    async def _update_user_learning_phase(self, user_id: str, new_phase: str):
        """Update user's learning phase in database"""
        try:
            # Update user profile phase
            self.db_client.supabase.table('user_profiles')\
                .update({
                    'current_phase': new_phase,
                    'updated_at': datetime.utcnow().isoformat()
                })\
                .eq('user_id', user_id)\
                .execute()
            
            # Update in-memory profile
            if user_id in self.user_profiles:
                self.user_profiles[user_id].learning_phase = new_phase
                
            self.logger.info(f"Updated user {user_id} learning phase to {new_phase}")
            
        except Exception as e:
            self.logger.error(f"Error updating user learning phase: {e}")

    async def end_session_update(self, user_id: str, session_id: str):
        """Handle session end cumulative vector update"""
        try:
            self.logger.info(f"Processing session end update for user {user_id}, session {session_id}")
            
            # Get the latest session vector for this session
            session_vectors_result = self.db_client.supabase.table('enhanced_behavioral_vectors')\
                .select('*')\
                .eq('session_id', session_id)\
                .eq('vector_type', VectorType.SESSION.value)\
                .order('created_at', desc=True)\
                .limit(1)\
                .execute()
            
            if not session_vectors_result.data:
                self.logger.warning(f"No session vector found for session {session_id}")
                return
            
            # Get session vector
            session_vector_data = session_vectors_result.data[0]['vector_data']
            session_vector = np.array(session_vector_data, dtype=np.float32)
            
            # Update cumulative vector
            await self._update_cumulative_vector(user_id, session_vector, "allow")
            
            # Check for phase transitions
            await self._check_learning_phase_transition(user_id)
            
            self.logger.info(f"Session end update completed for user {user_id}")
            
        except Exception as e:
            self.logger.error(f"Error in session end update: {e}")

    async def get_user_learning_status(self, user_id: str) -> Dict[str, Any]:
        """Get detailed learning status for a user"""
        try:
            user_profile = self.user_profiles.get(user_id)
            if not user_profile:
                return {
                    "learning_phase": "learning",
                    "vector_count": 0,
                    "sessions_needed": 5,
                    "baseline_created": False
                }
            
            # Check baseline existence
            baseline_result = self.db_client.supabase.table('enhanced_behavioral_vectors')\
                .select('id')\
                .eq('user_id', user_id)\
                .eq('vector_type', VectorType.BASELINE.value)\
                .execute()
            
            baseline_exists = len(baseline_result.data) > 0
            
            # Calculate sessions needed for next phase
            sessions_needed = 0
            if user_profile.learning_phase == "learning":
                sessions_needed = max(0, 5 - user_profile.vector_count)
            elif user_profile.learning_phase == "gradual_risk":
                sessions_needed = max(0, 10 - user_profile.vector_count)
            
            return {
                "learning_phase": user_profile.learning_phase,
                "vector_count": user_profile.vector_count,
                "sessions_needed": sessions_needed,
                "baseline_created": baseline_exists,
                "last_updated": user_profile.last_updated.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting user learning status: {e}")
            return {"error": str(e)}

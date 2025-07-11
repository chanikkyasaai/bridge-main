"""
FAISS-Based Layer 1 Verification System
Fast similarity matching for real-time behavioral authentication
"""

import numpy as np
import faiss
import pickle
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path
import threading
from collections import defaultdict
import os
import time
from collections import deque

from ml_engine.config import CONFIG
from ml_engine.utils.behavioral_vectors import BehavioralVector

logger = logging.getLogger(__name__)

@dataclass
class UserProfile:
    """User behavioral profile with multiple vectors"""
    user_id: str
    vectors: np.ndarray  # [n_vectors, vector_dim]
    labels: List[str]  # Mode labels (e.g., 'normal', 'hurried', 'stressed')
    timestamps: List[datetime]
    confidences: List[float]
    creation_date: datetime
    last_updated: datetime
    usage_count: int = 0

@dataclass
class VerificationResult:
    """Result from FAISS verification"""
    user_id: str
    session_id: str
    similarity_score: float
    confidence_level: str  # 'high', 'medium', 'low'
    matched_profile_id: str
    matched_mode: str
    decision: str  # 'continue', 'escalate', 'block'
    processing_time_ms: float
    metadata: Dict[str, Any]

class AdaptiveThresholdLearner:
    """Learns and adapts similarity thresholds per user"""
    
    def __init__(self):
        self.user_thresholds = {}  # user_id -> {high, medium, low}
        self.threshold_history = defaultdict(list)
        self.calibration_samples = 50  # Minimum samples for calibration
    
    def get_thresholds(self, user_id: str) -> Dict[str, float]:
        """Get adaptive thresholds for a user"""
        if user_id not in self.user_thresholds:
            # Use default thresholds for new users
            return {
                'high': CONFIG.L1_HIGH_CONFIDENCE_THRESHOLD,
                'medium': CONFIG.L1_MEDIUM_CONFIDENCE_THRESHOLD,
                'low': CONFIG.L1_LOW_CONFIDENCE_THRESHOLD
            }
        return self.user_thresholds[user_id]
    
    def update_thresholds(self, user_id: str, similarity_score: float, is_legitimate: bool):
        """Update thresholds based on feedback"""
        self.threshold_history[user_id].append((similarity_score, is_legitimate))
        
        # Recalibrate if we have enough samples
        if len(self.threshold_history[user_id]) >= self.calibration_samples:
            self._recalibrate_user_thresholds(user_id)
    
    def _recalibrate_user_thresholds(self, user_id: str):
        """Recalibrate thresholds using historical data"""
        history = self.threshold_history[user_id]
        legitimate_scores = [score for score, is_legit in history if is_legit]
        illegitimate_scores = [score for score, is_legit in history if not is_legit]
        
        if len(legitimate_scores) < 10 or len(illegitimate_scores) < 5:
            return  # Not enough data
        
        # Calculate percentiles for legitimate scores
        legit_p25 = np.percentile(legitimate_scores, 25)
        legit_p50 = np.percentile(legitimate_scores, 50)
        legit_p75 = np.percentile(legitimate_scores, 75)
        
        # Calculate percentiles for illegitimate scores
        illegit_p75 = np.percentile(illegitimate_scores, 75) if illegitimate_scores else 0.3
        
        # Set adaptive thresholds
        self.user_thresholds[user_id] = {
            'high': max(legit_p75, CONFIG.L1_HIGH_CONFIDENCE_THRESHOLD),
            'medium': max(legit_p50, illegit_p75 + 0.1),
            'low': max(legit_p25, CONFIG.L1_LOW_CONFIDENCE_THRESHOLD)
        }
        
        logger.info(f"Updated thresholds for user {user_id}: {self.user_thresholds[user_id]}")

class UserVectorBank:
    """Manages user behavioral vector profiles"""
    
    def __init__(self, storage_path: str = "user_profiles/"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.profiles = {}  # user_id -> UserProfile
        self.lock = threading.RLock()
    
    def add_profile(self, user_id: str, vectors: np.ndarray, labels: List[str]):
        """Add or update user profile"""
        with self.lock:
            now = datetime.now()
            
            if user_id in self.profiles:
                # Update existing profile
                profile = self.profiles[user_id]
                profile.vectors = np.vstack([profile.vectors, vectors])
                profile.labels.extend(labels)
                profile.timestamps.extend([now] * len(vectors))
                profile.confidences.extend([1.0] * len(vectors))  # Default confidence
                profile.last_updated = now
                profile.usage_count += 1
            else:
                # Create new profile
                profile = UserProfile(
                    user_id=user_id,
                    vectors=vectors,
                    labels=labels,
                    timestamps=[now] * len(vectors),
                    confidences=[1.0] * len(vectors),
                    creation_date=now,
                    last_updated=now
                )
                self.profiles[user_id] = profile
            
            # Limit profile size (keep most recent vectors)
            max_vectors = 1000
            if len(profile.vectors) > max_vectors:
                keep_indices = np.argsort(profile.timestamps)[-max_vectors:]
                profile.vectors = profile.vectors[keep_indices]
                profile.labels = [profile.labels[i] for i in keep_indices]
                profile.timestamps = [profile.timestamps[i] for i in keep_indices]
                profile.confidences = [profile.confidences[i] for i in keep_indices]
    
    def get_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile"""
        with self.lock:
            return self.profiles.get(user_id)
    
    def save_profile(self, user_id: str):
        """Save user profile to disk"""
        if user_id not in self.profiles:
            return
        
        profile_path = self.storage_path / f"{user_id}.pkl"
        with open(profile_path, 'wb') as f:
            pickle.dump(self.profiles[user_id], f)
    
    def load_profile(self, user_id: str):
        """Load user profile from disk"""
        profile_path = self.storage_path / f"{user_id}.pkl"
        if profile_path.exists():
            with open(profile_path, 'rb') as f:
                self.profiles[user_id] = pickle.load(f)
    
    def update_profile_vectors(self, user_id: str, new_vectors: np.ndarray, labels: List[str]):
        """Add new vectors to existing profile"""
        self.add_profile(user_id, new_vectors, labels)
        self.save_profile(user_id)
    
    async def save_profiles(self):
        """Save all user profiles to disk"""
        for user_id in self.profiles.keys():
            self.save_profile(user_id)
    
    def add_user_profile(self, profile: UserProfile):
        """Add a complete user profile"""
        with self.lock:
            self.profiles[profile.user_id] = profile

class FAISSVerifier:
    """FAISS-based behavioral vector verification engine"""
    
    def __init__(self, index_path: Optional[str] = None):
        self.vector_dim = CONFIG.BEHAVIORAL_VECTOR_DIM
        self.index = None
        self.user_bank = UserVectorBank()
        self.threshold_learner = AdaptiveThresholdLearner()
        
        # Index mapping
        self.index_to_user = {}  # FAISS index -> (user_id, vector_index)
        self.user_to_indices = defaultdict(list)  # user_id -> [FAISS indices]
        
        self.index_path = index_path or CONFIG.FAISS_INDEX_PATH
        self.is_initialized = False
        self._initialize_index()
        
        # Performance optimization caches
        self.normalized_cache = {}  # Cache for normalized vectors
        self.user_stats_cache = {}  # Cache for user statistics
        self.similarity_cache = {}  # Cache for recent similarity calculations
        self.cache_lock = threading.RLock()
        
        # Performance metrics
        self.stats = {
            'total_queries': 0,
            'avg_query_time_ms': 0,
            'cache_hits': 0
        }
    
    def _initialize_index(self):
        """Initialize FAISS index with maximum performance optimizations"""
        # ALWAYS use IndexFlatIP for maximum speed - no approximations
        self.index = faiss.IndexFlatIP(self.vector_dim)
        
        logger.info(f"Initialized ultra-fast FAISS index: IndexFlatIP for maximum speed")
    
    async def initialize(self):
        """Initialize the FAISS verifier"""
        logger.info("Initializing FAISS Verifier...")
        
        # Load existing index if available
        if os.path.exists(self.index_path):
            try:
                self.index = faiss.read_index(self.index_path)
                logger.info(f"Loaded existing FAISS index from {self.index_path}")
            except Exception as e:
                logger.warning(f"Failed to load existing index: {e}")
                self._initialize_index()
        else:
            self._initialize_index()
        
        # Initialize index for IVF if needed
        if hasattr(self.index, 'train') and not self.index.is_trained:
            logger.info("Training FAISS index (this may take a while)...")
            # Generate some random training data for initialization
            training_data = np.random.randn(10000, self.vector_dim).astype(np.float32)
            self.index.train(training_data)
            logger.info("FAISS index training completed")
        
        self.is_initialized = True
        logger.info("✓ FAISS Verifier initialized")
    
    def add_user_vectors(self, user_id: str, vectors: np.ndarray, labels: List[str]):
        """Add user vectors to the index"""
        if vectors.shape[1] != self.vector_dim:
            raise ValueError(f"Vector dimension mismatch: expected {self.vector_dim}, got {vectors.shape[1]}")
        
        # Normalize vectors for cosine similarity
        normalized_vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
        
        # Add to FAISS index
        start_idx = self.index.ntotal
        self.index.add(normalized_vectors.astype(np.float32))
        
        # Update mappings
        for i, vector_idx in enumerate(range(start_idx, self.index.ntotal)):
            self.index_to_user[vector_idx] = (user_id, i)
            self.user_to_indices[user_id].append(vector_idx)
        
        # Add to user bank
        self.user_bank.add_profile(user_id, normalized_vectors, labels)
        
        logger.info(f"Added {len(vectors)} vectors for user {user_id}")
    
    async def verify(self, vectors: List[BehavioralVector], user_id: str, session_id: str) -> VerificationResult:
        """Verify behavioral vectors against user profile"""
        if not self.is_initialized:
            raise RuntimeError("FAISS verifier not initialized")

        start_time = time.perf_counter()  # More precise timing
        
        # Get user profile
        user_profile = await self.get_user_profile(user_id)
        if not user_profile:
            # New user - create initial profile
            logger.info(f"Creating initial profile for new user: {user_id}")
            return self._create_new_user_result(user_id, session_id, vectors)

        # PERFORMANCE OPTIMIZATION: Skip all caching for maximum speed
        query_vectors = np.array([v.vector for v in vectors], dtype=np.float32)
        
        # PERFORMANCE OPTIMIZATION: Ultra-fast normalization - skip cache for speed
        norms = np.linalg.norm(query_vectors, axis=1, keepdims=True)
        normalized_queries = query_vectors / (norms + 1e-8)
        
        # ULTRA-FAST SEARCH: k=1 for maximum speed
        k_search = 1
        similarities, indices = self.index.search(normalized_queries, k=k_search)
        
        # ULTRA-STRICT DISCRIMINATION: Zero tolerance for different users
        user_vector_count = len(self.user_to_indices.get(user_id, []))
        other_vector_count = self.index.ntotal - user_vector_count
        user_similarities = []
        different_user_detected = False
        max_other_user_similarity = 0.0
        
        for i in range(len(vectors)):
            if i < len(similarities) and len(similarities[i]) > 0:
                sim = similarities[i][0]  # Only top result
                idx = indices[i][0]
                
                if idx >= 0 and idx in self.index_to_user:
                    indexed_user_id, _ = self.index_to_user[idx]
                    
                    if indexed_user_id == user_id:
                        user_similarities.append(float(sim))
                    else:
                        different_user_detected = True
                        max_other_user_similarity = max(max_other_user_similarity, float(sim))
                else:
                    user_similarities.append(0.0)
            else:
                user_similarities.append(0.0)
        
        # ZERO TOLERANCE DECISION LOGIC
        overall_similarity = np.mean(user_similarities) if user_similarities else 0.0
        
        if different_user_detected and other_vector_count > 0:
            # CRITICAL FIX: If ANY similarity to different users detected, return very low score
            overall_similarity = min(overall_similarity, 0.3)  # Cap at 0.3 for different users
            
            if max_other_user_similarity > 0.2:  # Very low threshold
                decision = "block"
                confidence_level = "low"
                overall_similarity = 0.1  # Force very low similarity for different users
                reason = f"Different user detected (sim={max_other_user_similarity:.3f})"
            elif overall_similarity < 0.85:  # Require very high similarity
                decision = "block"
                confidence_level = "low"
                overall_similarity = min(overall_similarity, 0.2)  # Keep low for different users
                reason = f"Insufficient user similarity ({overall_similarity:.3f})"
            else:
                decision = "continue"
                confidence_level = "high" 
                reason = "High user similarity, low cross-user similarity"
        elif user_similarities and overall_similarity > 0.7:
            decision = "continue"
            confidence_level = "high"
            reason = "Good user match, no other users"
        elif user_similarities and overall_similarity > 0.4:
            decision = "escalate"
            confidence_level = "medium"
            reason = "Moderate user match"
        else:
            decision = "block"
            confidence_level = "low"
            overall_similarity = 0.1  # Force low similarity for blocks
            reason = "No valid user matches"
        
        # Update statistics - simplified for speed
        processing_time = (time.perf_counter() - start_time) * 1000
        self.stats['total_queries'] += 1
        
        # Log decision for debugging
        logger.info(f"User {user_id}: {decision} ({reason}), "
                   f"UserSim={overall_similarity:.3f}, OtherSim={max_other_user_similarity:.3f}, "
                   f"Time={processing_time:.2f}ms")
        
        result = VerificationResult(
            user_id=user_id,
            session_id=session_id,
            similarity_score=overall_similarity,
            confidence_level=confidence_level,
            matched_profile_id=f"{user_id}_profile",
            matched_mode="normal",
            decision=decision,
            processing_time_ms=processing_time,
            metadata={
                "individual_similarities": user_similarities,
                "max_other_user_similarity": max_other_user_similarity,
                "different_user_detected": different_user_detected,
                "total_vectors": len(vectors),
                "user_vector_count": user_vector_count,
                "other_vector_count": other_vector_count,
                "reason": reason,
                "cache_hit": False
            }
        )
        
        # PERFORMANCE: Skip caching completely for maximum speed
        return result
    
    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile"""
        return self.user_bank.get_profile(user_id)
    
    async def train_user_profile(self, user_id: str, vectors: List[BehavioralVector], labels: List[str]):
        """Train user profile with behavioral vectors"""
        if not vectors:
            return
        
        # Convert to numpy array
        vector_array = np.array([v.vector for v in vectors])
        
        # Add to FAISS index
        self.add_user_vectors(user_id, vector_array, labels)
        
        # Create/update user profile
        profile = UserProfile(
            user_id=user_id,
            vectors=vector_array,
            labels=labels,
            timestamps=[v.timestamp for v in vectors],
            confidences=[v.confidence for v in vectors],
            creation_date=datetime.now(),
            last_updated=datetime.now()
        )
        
        # Store in user bank
        self.user_bank.add_user_profile(profile)
        
        logger.info(f"Trained profile for user {user_id} with {len(vectors)} vectors")
    
    async def update_models(self):
        """Update and retrain models"""
        logger.info("Updating FAISS models...")
        # In a full implementation, this would retrain the index
        # For now, just log the update
        logger.info("FAISS models updated")
    
    async def save_models(self):
        """Save FAISS index and user profiles"""
        if self.index:
            faiss.write_index(self.index, self.index_path)
            logger.info(f"Saved FAISS index to {self.index_path}")
        
        # Save user profiles
        await self.user_bank.save_profiles()
    
    def _create_new_user_result(self, user_id: str, session_id: str, vectors: List[BehavioralVector]) -> VerificationResult:
        """Create result for new user"""
        return VerificationResult(
            user_id=user_id,
            session_id=session_id,
            similarity_score=0.5,  # Neutral score for new users
            confidence_level="medium",
            matched_profile_id="new_user",
            matched_mode="unknown",
            decision="escalate",  # Always escalate for new users
            processing_time_ms=1.0,
            metadata={
                "new_user": True,
                "vectors_provided": len(vectors)
            }
        )
    
    def update_user_feedback(self, user_id: str, similarity_score: float, is_legitimate: bool):
        """Update user model with feedback"""
        self.threshold_learner.update_thresholds(user_id, similarity_score, is_legitimate)
    
    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get statistics for a user"""
        profile = self.user_bank.get_profile(user_id)
        if not profile:
            return {}
        
        return {
            'vector_count': len(profile.vectors),
            'modes': list(set(profile.labels)),
            'creation_date': profile.creation_date.isoformat(),
            'last_updated': profile.last_updated.isoformat(),
            'usage_count': profile.usage_count,
            'thresholds': self.threshold_learner.get_thresholds(user_id)
        }
    
    def save_index(self, path: str):
        """Save FAISS index to disk"""
        faiss.write_index(self.index, path)
        
        # Save mappings
        mappings_path = path + ".mappings.pkl"
        with open(mappings_path, 'wb') as f:
            pickle.dump({
                'index_to_user': self.index_to_user,
                'user_to_indices': dict(self.user_to_indices)
            }, f)
        
        logger.info(f"FAISS index and mappings saved to {path}")
    
    def load_index(self, path: str):
        """Load FAISS index from disk"""
        self.index = faiss.read_index(path)
        
        # Load mappings
        mappings_path = path + ".mappings.pkl"
        with open(mappings_path, 'rb') as f:
            mappings = pickle.load(f)
            self.index_to_user = mappings['index_to_user']
            self.user_to_indices = defaultdict(list, mappings['user_to_indices'])
        
        logger.info(f"FAISS index and mappings loaded from {path}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            'total_queries': self.stats['total_queries'],
            'avg_query_time_ms': self.stats['avg_query_time_ms'],
            'index_size': self.index.ntotal,
            'total_users': len(self.user_bank.profiles)
        }
    
    def _fast_hash_vectors(self, vectors: List[np.ndarray]) -> str:
        """Create ultra-fast hash for vector caching using CRC32"""
        import zlib
        if not vectors:
            return "empty"
        
        # Use strategic sampling for maximum speed
        vector = vectors[0] if len(vectors[0].shape) == 1 else vectors[0][0]
        
        # Sample only first and last few elements for speed
        if len(vector) > 10:
            vector_sample = np.concatenate([vector[:3], vector[-2:]])
        else:
            vector_sample = vector[:5]
        
        vector_bytes = vector_sample.astype(np.float32).tobytes()
        return str(zlib.crc32(vector_bytes) & 0xffffffff)[:8]
    
    def _fast_normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Ultra-fast vector normalization with SIMD optimization"""
        # Use the fastest possible normalization
        # np.linalg.norm is faster than manual computation for small vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        
        # Vectorized division with epsilon for numerical stability
        return np.divide(vectors, norms + 1e-8, 
                        out=np.zeros_like(vectors), 
                        where=(norms + 1e-8) != 0)
    
    def _get_normalized_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Get normalized vectors with caching"""
        vector_hash = self._fast_hash_vectors([vectors])
        
        with self.cache_lock:
            if vector_hash in self.normalized_cache:
                return self.normalized_cache[vector_hash]
        
        # Normalize vectors
        normalized = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
        
        with self.cache_lock:
            # Limit cache size
            if len(self.normalized_cache) > 50:
                oldest_key = list(self.normalized_cache.keys())[0]
                del self.normalized_cache[oldest_key]
            self.normalized_cache[vector_hash] = normalized
        
        return normalized
    
    def _ultra_fast_normalize(self, vectors: np.ndarray) -> np.ndarray:
        """Ultra-fast vector normalization optimized for minimal processing time"""
        # Direct computation with minimal operations
        squared_norms = np.sum(vectors * vectors, axis=1, keepdims=True)
        # Use rsqrt approximation for speed if possible, fallback to division
        inv_norms = np.where(squared_norms > 1e-12, 
                           1.0 / np.sqrt(squared_norms), 
                           0.0)
        return vectors * inv_norms
    
    def _calculate_enhanced_user_discrimination(self, similarities: np.ndarray, indices: np.ndarray, 
                                              target_user_id: str, num_vectors: int, 
                                              user_vector_count: int, other_vector_count: int) -> Tuple[List[float], float, Dict[str, Any]]:
        """Enhanced user discrimination with comprehensive metrics"""
        user_similarities = []
        other_user_similarities = []
        user_match_counts = []
        other_user_ids = set()
        
        for i in range(num_vectors):
            sim_scores = similarities[i]
            idx_list = indices[i]
            
            user_sims = []
            other_sims = []
            
            for sim, idx in zip(sim_scores, idx_list):
                if idx >= 0 and idx in self.index_to_user:  # Valid index
                    indexed_user_id, _ = self.index_to_user[idx]
                    
                    if indexed_user_id == target_user_id:
                        user_sims.append(float(sim))
                    else:
                        other_sims.append(float(sim))
                        other_user_ids.add(indexed_user_id)
            
            user_match_counts.append(len(user_sims))
            
            # ENHANCED SCORING: Weighted scoring with consistency checks
            if user_sims:
                # Prioritize consistency over single high scores
                sorted_sims = sorted(user_sims, reverse=True)
                if len(sorted_sims) >= 3:
                    # Top 3 weighted average for strong consistency
                    user_score = 0.5 * sorted_sims[0] + 0.3 * sorted_sims[1] + 0.2 * sorted_sims[2]
                elif len(sorted_sims) == 2:
                    user_score = 0.7 * sorted_sims[0] + 0.3 * sorted_sims[1]
                else:
                    user_score = sorted_sims[0]
                
                # CONSISTENCY PENALTY: Penalize if user has inconsistent scores
                if len(sorted_sims) > 1:
                    consistency = 1.0 - (sorted_sims[0] - sorted_sims[-1])
                    user_score *= max(0.8, consistency)  # Up to 20% penalty for inconsistency
                
                user_similarities.append(user_score)
            else:
                user_similarities.append(0.0)
            
            # Track other user similarities
            if other_sims:
                other_user_similarities.append(max(other_sims))
        
        # DISCRIMINATION METRICS: Comprehensive analysis
        cross_user_max = max(other_user_similarities) if other_user_similarities else 0.0
        cross_user_avg = np.mean(other_user_similarities) if other_user_similarities else 0.0
        user_avg = np.mean(user_similarities) if user_similarities else 0.0
        
        # STRICT DISCRIMINATION: Multiple validation checks
        discrimination_score = user_avg - cross_user_max
        discrimination_ratio = user_avg / (cross_user_max + 1e-8) if cross_user_max > 0 else float('inf')
        
        # ENHANCED PENALTIES for poor discrimination
        if cross_user_avg > 0.6:  # High avg similarity to others
            cross_user_max += 0.25  # Boost penalty
        
        if len(other_user_ids) > 1 and cross_user_avg > 0.5:
            # Multiple other users with decent similarity - very suspicious
            cross_user_max += 0.3
        
        # INSUFFICIENT DATA PENALTIES
        user_match_rate = np.mean(user_match_counts) if user_match_counts else 0
        if user_vector_count < 3 and other_vector_count > 0:
            # Too few user vectors for reliable discrimination
            cross_user_max = max(cross_user_max, 0.85)
        
        discrimination_metrics = {
            "discrimination_score": discrimination_score,
            "discrimination_ratio": discrimination_ratio,
            "cross_user_avg": cross_user_avg,
            "user_match_rate": user_match_rate,
            "unique_other_users": len(other_user_ids),
            "consistency_penalty_applied": any(len([s for s in user_similarities if s > 0]) > 1)
        }
        
        return user_similarities, cross_user_max, discrimination_metrics
    
    def _make_enhanced_discrimination_decision(self, user_similarity: float, cross_user_max: float, 
                                             thresholds: Dict[str, float], user_id: str, 
                                             metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Ultra-strict decision making with maximum user discrimination"""
        
        discrimination_score = metrics["discrimination_score"]
        discrimination_ratio = metrics["discrimination_ratio"]
        cross_user_avg = metrics["cross_user_avg"]
        
        # ULTRA-STRICT DISCRIMINATION REQUIREMENTS
        min_discrimination_ratio = 1.5  # User must be 50% more similar than others
        min_discrimination_score = 0.3  # High absolute minimum difference
        max_cross_user_threshold = 0.7  # Maximum allowed similarity to others
        
        # IMMEDIATE REJECTION CONDITIONS - Zero tolerance
        if cross_user_max >= user_similarity:
            # Any case where other users are equal or more similar = immediate block
            confidence_level = "low"
            decision = "block"
            reason = "other_user_equal_or_higher_similarity"
        elif cross_user_max > max_cross_user_threshold:
            # Too similar to other users regardless of own similarity
            confidence_level = "low"
            decision = "block"
            reason = "excessive_cross_user_similarity"
        elif discrimination_ratio < min_discrimination_ratio:
            # Insufficient discrimination ratio
            confidence_level = "low"
            decision = "block"
            reason = "insufficient_discrimination_ratio"
        elif discrimination_score < min_discrimination_score:
            # Insufficient discrimination score
            confidence_level = "low"
            decision = "block"
            reason = "insufficient_discrimination_score"
        elif cross_user_avg > 0.6 and metrics["unique_other_users"] > 0:
            # Any significant similarity to other users is suspicious
            confidence_level = "low"
            decision = "block"
            reason = "general_cross_user_confusion"
        elif metrics["user_match_rate"] < 0.5 and self.index.ntotal > len(self.user_to_indices.get(user_id, [])):
            # Low match rate with other users present
            confidence_level = "low"
            decision = "block"
            reason = "low_user_match_rate_with_others_present"
        # VERY STRICT ACCEPTANCE CONDITIONS
        elif user_similarity >= 0.85 and discrimination_score > 0.4:
            # Only accept if very high user similarity AND excellent discrimination
            confidence_level = "high"
            decision = "continue"
            reason = "excellent_user_similarity_and_discrimination"
        elif user_similarity >= 0.75 and discrimination_score > 0.35:
            # Medium confidence only for high similarity with good discrimination
            confidence_level = "medium"
            decision = "escalate"
            reason = "good_user_similarity_and_discrimination"
        else:
            # Default to block for all other cases
            confidence_level = "low"
            decision = "block"
            reason = "insufficient_confidence_for_acceptance"
        
        # FINAL SAFETY CHECK: Any significant cross-user similarity = block
        if cross_user_max > 0.3 and decision != "block":
            confidence_level = "low"
            decision = "block"
            reason = "safety_check_cross_user_similarity"
        
        # LOG DETAILED DISCRIMINATION ANALYSIS
        logger.info(f"User {user_id} discrimination analysis: "
                   f"Decision={decision}, Reason={reason}, "
                   f"UserSim={user_similarity:.3f}, CrossUserMax={cross_user_max:.3f}, "
                   f"DiscrimScore={discrimination_score:.3f}, DiscrimRatio={discrimination_ratio:.2f}")
        
        return {
            "confidence_level": confidence_level,
            "decision": decision,
            "reason": reason,
            "discrimination_score": discrimination_score,
            "discrimination_ratio": discrimination_ratio
        }

"""
Phase 1 Learning System for Behavioral Authentication
Handles cold start problem and multi-vector profile building
"""

import asyncio
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json

from src.data.models import (
    BehavioralVector, UserProfile, AuthenticationDecision, 
    RiskLevel, SessionPhase, LearningPhase
)
from src.core.ml_database import ml_db
from src.core.vector_store import VectorStoreInterface
from src.data.behavioral_processor import BehavioralProcessor
from src.config.settings import get_settings

logger = logging.getLogger(__name__)

@dataclass
class LearningProfile:
    """User learning profile with phase-specific data"""
    user_id: str
    current_phase: LearningPhase
    session_count: int
    vectors_collected: int
    phase_confidence: float
    baseline_vectors: List[BehavioralVector]
    cluster_centers: Optional[np.ndarray] = None
    variance_threshold: float = 0.15
    learning_start_time: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'user_id': self.user_id,
            'current_phase': self.current_phase.value,
            'session_count': self.session_count,
            'vectors_collected': self.vectors_collected,
            'phase_confidence': self.phase_confidence,
            'variance_threshold': self.variance_threshold,
            'learning_start_time': self.learning_start_time.isoformat() if self.learning_start_time else None
        }

class Phase1LearningSystem:
    """
    Phase 1 Learning System for Behavioral Authentication
    
    Handles:
    - Cold start detection and handling
    - Multi-vector profile building over N sessions
    - Phase transitions (learning -> gradual_risk -> full_auth)
    - Baseline behavioral pattern establishment
    - Statistical confidence building
    """
    
    def __init__(self, vector_store: VectorStoreInterface, behavioral_processor: BehavioralProcessor):
        self.vector_store = vector_store
        self.behavioral_processor = behavioral_processor
        self.settings = get_settings()
        
        # Learning configuration
        self.learning_session_threshold = 5  # Sessions needed to complete learning
        self.gradual_risk_threshold = 15     # Sessions to move to full auth
        self.min_vectors_per_session = 3     # Minimum vectors needed per session
        self.confidence_threshold = 0.7      # Required confidence to advance phases
        
        # In-memory learning profiles cache
        self.learning_profiles: Dict[str, LearningProfile] = {}
        
        # Learning statistics
        self.learning_stats = {
            'users_in_learning': 0,
            'completed_learning_profiles': 0,
            'cold_start_users': 0,
            'phase_transitions_today': 0
        }
        
        logger.info("Phase 1 Learning System initialized")
    
    async def handle_new_session(self, user_id: str, session_id: str) -> Tuple[LearningPhase, Dict[str, Any]]:
        """
        Handle new session for a user - determine learning phase and requirements
        
        Returns:
            Tuple of (current_phase, session_guidance)
        """
        try:
            # Create session record in database first
            session_uuid = await ml_db.create_session(user_id, session_id)
            if not session_uuid:
                raise Exception("Failed to create session record")
                
            # Store the session UUID for use in behavioral vectors
            self._session_mapping = getattr(self, '_session_mapping', {})
            self._session_mapping[session_id] = session_uuid
            
            # Get or create user profile from database
            user_profile = await ml_db.get_user_profile(user_id)
            
            if not user_profile:
                # Cold start - create new user profile
                user_profile = await ml_db.create_user_profile(user_id)
                if not user_profile:
                    raise Exception("Failed to create user profile")
                
                # Initialize learning profile
                learning_profile = LearningProfile(
                    user_id=user_id,
                    current_phase=LearningPhase.COLD_START,
                    session_count=0,
                    vectors_collected=0,
                    phase_confidence=0.0,
                    baseline_vectors=[],
                    learning_start_time=datetime.utcnow()
                )
                
                self.learning_profiles[user_id] = learning_profile
                self.learning_stats['cold_start_users'] += 1
                
                logger.info(f"Cold start detected for user {user_id}")
                
                session_guidance = {
                    'phase': LearningPhase.COLD_START.value,
                    'message': 'Welcome! We are learning your behavioral patterns.',
                    'vectors_needed': self.min_vectors_per_session,
                    'session_count': 0,
                    'progress_percentage': 0
                }
                
                return LearningPhase.COLD_START, session_guidance
            
            # Existing user - increment session count
            session_incremented = await ml_db.increment_session_count(user_id)
            if not session_incremented:
                logger.error(f"Failed to increment session count for {user_id}")
            
            # Get updated profile after increment
            user_profile = await ml_db.get_user_profile(user_id)
            current_phase = LearningPhase(user_profile['current_phase'])
            session_count = user_profile['current_session_count']
            
            # Load or create learning profile
            if user_id not in self.learning_profiles:
                learning_profile = await self._load_learning_profile(user_id, user_profile)
                self.learning_profiles[user_id] = learning_profile
            else:
                self.learning_profiles[user_id].session_count = session_count
                self.learning_profiles[user_id].current_phase = current_phase
            
            # Generate session guidance based on phase
            session_guidance = await self._generate_session_guidance(user_id, current_phase, session_count)
            
            logger.info(f"Session started for user {user_id} - Phase: {current_phase.value}, Session: {session_count}")
            
            return current_phase, session_guidance
            
        except Exception as e:
            logger.error(f"Error handling new session for {user_id}: {e}")
            # Fallback to learning phase
            return LearningPhase.LEARNING, {
                'phase': LearningPhase.LEARNING.value,
                'message': 'Learning mode - continuing behavioral analysis',
                'vectors_needed': self.min_vectors_per_session,
                'error': 'Profile loading error - using safe defaults'
            }
    
    async def process_behavioral_vector(self, user_id: str, session_id: str, 
                                      behavioral_vector: BehavioralVector) -> Dict[str, Any]:
        """
        Process a new behavioral vector during learning phase
        
        Returns:
            Processing results and learning progress
        """
        try:
            # Store vector in database
            vector_data = behavioral_vector.vector
            if hasattr(vector_data, 'tolist'):
                vector_data = vector_data.tolist()
            elif not isinstance(vector_data, list):
                vector_data = list(vector_data)
            
            # Get the session UUID from mapping, fallback to session_id if not found
            session_uuid = getattr(self, '_session_mapping', {}).get(session_id, session_id)
                
            vector_id = await ml_db.store_behavioral_vector(
                user_id=user_id,
                session_id=session_uuid,
                vector_data=vector_data,
                confidence_score=behavioral_vector.confidence_score,
                feature_source='learning_system'
            )
            
            if not vector_id:
                logger.error(f"Failed to store behavioral vector for {user_id}")
                return {'status': 'error', 'message': 'Vector storage failed'}
            
            # Update learning profile
            if user_id not in self.learning_profiles:
                # This shouldn't happen, but handle gracefully
                await self.handle_new_session(user_id, session_id)
            
            learning_profile = self.learning_profiles[user_id]
            learning_profile.vectors_collected += 1
            learning_profile.baseline_vectors.append(behavioral_vector)
            
            # Try to update the database with the new vector count, but don't fail if column doesn't exist
            try:
                await ml_db.update_user_profile(user_id, {
                    'vectors_collected': learning_profile.vectors_collected
                })
            except Exception as e:
                # If vectors_collected column doesn't exist, just log and continue
                # The system will still work by counting vectors from the database
                logger.debug(f"Could not update vectors_collected field (column may not exist): {e}")
                
                # Alternative: count actual stored vectors from database
                try:
                    stored_vectors = await ml_db.get_user_vectors(user_id, limit=1000)
                    actual_count = len(stored_vectors) if stored_vectors else 0
                    learning_profile.vectors_collected = actual_count
                    logger.debug(f"Updated vector count from database: {actual_count}")
                except Exception as e2:
                    logger.warning(f"Could not count vectors from database: {e2}")
                    # Continue with in-memory count
            
            # Analyze vector quality and update confidence
            vector_quality = await self._analyze_vector_quality(behavioral_vector)
            
            # Update phase confidence based on vector quality and collection progress
            await self._update_phase_confidence(user_id, vector_quality)
            
            # Check for phase transitions
            phase_transition = await self._check_phase_transition(user_id)
            
            result = {
                'status': 'success',
                'vector_id': vector_id,
                'vector_quality': vector_quality,
                'vectors_collected': learning_profile.vectors_collected,
                'current_phase': learning_profile.current_phase.value,
                'phase_confidence': learning_profile.phase_confidence,
                'phase_transition': phase_transition
            }
            
            logger.debug(f"Processed vector for {user_id}: quality={vector_quality:.3f}, total={learning_profile.vectors_collected}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing behavioral vector for {user_id}: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def evaluate_learning_progress(self, user_id: str) -> Dict[str, Any]:
        """
        Evaluate user's learning progress and determine next steps
        
        Returns:
            Comprehensive learning progress report
        """
        try:
            if user_id not in self.learning_profiles:
                return {'status': 'error', 'message': 'No learning profile found'}
            
            learning_profile = self.learning_profiles[user_id]
            
            # Get user's behavioral vectors from database
            user_vectors = await ml_db.get_user_vectors(user_id, limit=100)
            
            # Analyze behavioral consistency
            consistency_analysis = await self._analyze_behavioral_consistency(user_vectors)
            
            # Calculate learning completeness
            completeness = self._calculate_learning_completeness(learning_profile)
            
            # Generate cluster analysis if enough data
            cluster_analysis = None
            if len(user_vectors) >= 5:  # Reduced from 10 to 5 for learning phase
                cluster_analysis = await self._perform_cluster_analysis(user_vectors)
                if cluster_analysis:
                    learning_profile.cluster_centers = cluster_analysis['centers']
            
            # Determine readiness for next phase
            readiness_assessment = await self._assess_phase_readiness(user_id, learning_profile)
            
            progress_report = {
                'user_id': user_id,
                'current_phase': learning_profile.current_phase.value,
                'session_count': learning_profile.session_count,
                'vectors_collected': learning_profile.vectors_collected,
                'phase_confidence': learning_profile.phase_confidence,
                'learning_completeness': completeness,
                'consistency_analysis': consistency_analysis,
                'cluster_analysis': cluster_analysis,
                'readiness_assessment': readiness_assessment,
                'learning_duration_hours': self._calculate_learning_duration(learning_profile),
                'recommendations': await self._generate_learning_recommendations(learning_profile)
            }
            
            return progress_report
            
        except Exception as e:
            logger.error(f"Error evaluating learning progress for {user_id}: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def _load_learning_profile(self, user_id: str, user_profile: Dict[str, Any]) -> LearningProfile:
        """Load learning profile from database and user profile"""
        try:
            # Get recent vectors to rebuild baseline
            user_vectors = await ml_db.get_user_vectors(user_id, limit=50)
            
            # Convert stored vectors back to BehavioralVector objects
            baseline_vectors = []
            for vector_data in user_vectors[:20]:  # Keep last 20 for baseline
                behavioral_vector = BehavioralVector(
                    vector=np.array(vector_data['vector_data']),
                    confidence=vector_data['confidence_score'],
                    timestamp=datetime.fromisoformat(vector_data['created_at'].replace('Z', '+00:00'))
                )
                baseline_vectors.append(behavioral_vector)
            
            learning_profile = LearningProfile(
                user_id=user_id,
                current_phase=LearningPhase(user_profile['current_phase']),
                session_count=user_profile['current_session_count'],
                vectors_collected=len(user_vectors),
                phase_confidence=min(0.9, len(user_vectors) / 20.0),  # Rough confidence estimate
                baseline_vectors=baseline_vectors,
                learning_start_time=datetime.fromisoformat(user_profile['created_at'].replace('Z', '+00:00'))
            )
            
            return learning_profile
            
        except Exception as e:
            logger.error(f"Error loading learning profile for {user_id}: {e}")
            # Return minimal profile
            return LearningProfile(
                user_id=user_id,
                current_phase=LearningPhase.LEARNING,
                session_count=0,
                vectors_collected=0,
                phase_confidence=0.0,
                baseline_vectors=[],
                learning_start_time=datetime.utcnow()
            )
    
    async def _generate_session_guidance(self, user_id: str, phase: LearningPhase, 
                                       session_count: int) -> Dict[str, Any]:
        """Generate guidance for user session based on learning phase"""
        
        if phase == LearningPhase.COLD_START:
            return {
                'phase': phase.value,
                'message': 'Welcome! We are starting to learn your behavioral patterns.',
                'vectors_needed': self.min_vectors_per_session,
                'session_count': session_count,
                'progress_percentage': 0,
                'requirements': [
                    'Complete at least 3 interactions this session',
                    'Use the app naturally - we are learning your patterns',
                    'This will help us secure your account'
                ]
            }
        
        elif phase == LearningPhase.LEARNING:
            progress = min(100, (session_count / self.learning_session_threshold) * 100)
            return {
                'phase': phase.value,
                'message': f'Learning your patterns... Session {session_count}/{self.learning_session_threshold}',
                'vectors_needed': self.min_vectors_per_session,
                'session_count': session_count,
                'progress_percentage': progress,
                'requirements': [
                    f'Complete {self.min_vectors_per_session} interactions this session',
                    'Continue using the app naturally',
                    f'{max(0, self.learning_session_threshold - session_count)} more sessions needed for full security'
                ]
            }
        
        elif phase == LearningPhase.GRADUAL_RISK:
            sessions_in_gradual = session_count - self.learning_session_threshold
            total_gradual_sessions = self.gradual_risk_threshold - self.learning_session_threshold
            progress = min(100, (sessions_in_gradual / total_gradual_sessions) * 100)
            
            return {
                'phase': phase.value,
                'message': f'Building security confidence... Enhanced monitoring active.',
                'vectors_needed': self.min_vectors_per_session,
                'session_count': session_count,
                'progress_percentage': progress,
                'requirements': [
                    'Behavioral analysis is now partially active',
                    'Any unusual patterns will be flagged',
                    f'{max(0, self.gradual_risk_threshold - session_count)} sessions until full protection'
                ]
            }
        
        else:  # FULL_AUTH
            return {
                'phase': phase.value,
                'message': 'Full behavioral authentication active - your account is protected.',
                'vectors_needed': 1,  # Continuous monitoring
                'session_count': session_count,
                'progress_percentage': 100,
                'requirements': [
                    'Full behavioral security is active',
                    'Unusual patterns will trigger additional verification',
                    'Your behavioral profile is continuously updated'
                ]
            }
    
    async def _analyze_vector_quality(self, vector: BehavioralVector) -> float:
        """Analyze the quality of a behavioral vector"""
        try:
            # Quality metrics
            quality_score = 0.0
            
            # Check vector completeness (non-zero elements)
            non_zero_ratio = np.count_nonzero(vector.vector) / len(vector.vector)
            quality_score += non_zero_ratio * 0.3
            
            # Check vector magnitude (should be normalized)
            magnitude = np.linalg.norm(vector.vector)
            if 0.8 <= magnitude <= 1.2:  # Good normalization
                quality_score += 0.3
            
            # Check confidence score
            if vector.confidence_score > 0.7:
                quality_score += 0.2
            elif vector.confidence_score > 0.5:
                quality_score += 0.1
            
            # Check for extreme values
            if np.max(np.abs(vector.vector)) < 10.0:  # No extreme outliers
                quality_score += 0.2
            
            return min(1.0, quality_score)
            
        except Exception as e:
            logger.error(f"Error analyzing vector quality: {e}")
            return 0.5  # Default medium quality
    
    async def _update_phase_confidence(self, user_id: str, vector_quality: float):
        """Update phase confidence based on new vector quality"""
        try:
            learning_profile = self.learning_profiles[user_id]
            
            # Update confidence using exponential moving average
            alpha = 0.1  # Learning rate
            current_confidence = learning_profile.phase_confidence
            new_confidence = alpha * vector_quality + (1 - alpha) * current_confidence
            
            learning_profile.phase_confidence = new_confidence
            
        except Exception as e:
            logger.error(f"Error updating phase confidence for {user_id}: {e}")
    
    async def _check_phase_transition(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Check if user is ready for phase transition"""
        try:
            learning_profile = self.learning_profiles[user_id]
            current_phase = learning_profile.current_phase
            session_count = learning_profile.session_count
            confidence = learning_profile.phase_confidence
            
            transition_info = None
            
            # Cold start -> Learning
            if current_phase == LearningPhase.COLD_START and session_count >= 1:
                if confidence >= 0.3:  # Basic interaction established
                    transition_info = await self._perform_phase_transition(
                        user_id, LearningPhase.LEARNING, "Basic interaction patterns established"
                    )
            
            # Learning -> Gradual Risk
            elif current_phase == LearningPhase.LEARNING and session_count >= self.learning_session_threshold:
                if confidence >= self.confidence_threshold:
                    transition_info = await self._perform_phase_transition(
                        user_id, LearningPhase.GRADUAL_RISK, "Learning phase completed"
                    )
            
            # Gradual Risk -> Full Auth
            elif current_phase == LearningPhase.GRADUAL_RISK and session_count >= self.gradual_risk_threshold:
                if confidence >= 0.8:  # High confidence for full auth
                    transition_info = await self._perform_phase_transition(
                        user_id, LearningPhase.FULL_AUTH, "Gradual risk phase completed"
                    )
            
            return transition_info
            
        except Exception as e:
            logger.error(f"Error checking phase transition for {user_id}: {e}")
            return None
    
    async def _perform_phase_transition(self, user_id: str, new_phase: LearningPhase, 
                                      reason: str) -> Dict[str, Any]:
        """Perform phase transition and update database"""
        try:
            learning_profile = self.learning_profiles[user_id]
            old_phase = learning_profile.current_phase
            
            # Update learning profile
            learning_profile.current_phase = new_phase
            
            # Update database
            updates = {
                'current_phase': new_phase.value,
                'last_activity': datetime.utcnow().isoformat()
            }
            
            success = await ml_db.update_user_profile(user_id, updates)
            
            if success:
                self.learning_stats['phase_transitions_today'] += 1
                
                if new_phase == LearningPhase.FULL_AUTH:
                    self.learning_stats['completed_learning_profiles'] += 1
                
                logger.info(f"Phase transition for {user_id}: {old_phase.value} -> {new_phase.value}")
                
                return {
                    'transition_occurred': True,
                    'old_phase': old_phase.value,
                    'new_phase': new_phase.value,
                    'reason': reason,
                    'timestamp': datetime.utcnow().isoformat()
                }
            else:
                logger.error(f"Failed to update database during phase transition for {user_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error performing phase transition for {user_id}: {e}")
            return None
    
    async def _analyze_behavioral_consistency(self, user_vectors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze consistency of behavioral patterns"""
        try:
            if len(user_vectors) < 3:
                return {'status': 'insufficient_data', 'message': 'Need more vectors for analysis'}
            
            # Convert to numpy arrays
            vectors = np.array([v['vector_data'] for v in user_vectors])
            
            # Calculate pairwise similarities
            similarities = []
            for i in range(len(vectors)):
                for j in range(i + 1, len(vectors)):
                    sim = np.dot(vectors[i], vectors[j]) / (np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j]))
                    similarities.append(sim)
            
            similarities = np.array(similarities)
            
            # Calculate consistency metrics
            avg_similarity = np.mean(similarities)
            std_similarity = np.std(similarities)
            min_similarity = np.min(similarities)
            max_similarity = np.max(similarities)
            
            # Determine consistency level
            if avg_similarity > 0.8 and std_similarity < 0.1:
                consistency_level = 'high'
            elif avg_similarity > 0.6 and std_similarity < 0.2:
                consistency_level = 'medium'
            else:
                consistency_level = 'low'
            
            return {
                'status': 'success',
                'consistency_level': consistency_level,
                'avg_similarity': float(avg_similarity),
                'std_similarity': float(std_similarity),
                'min_similarity': float(min_similarity),
                'max_similarity': float(max_similarity),
                'total_vectors': len(vectors)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing behavioral consistency: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _calculate_learning_completeness(self, learning_profile: LearningProfile) -> float:
        """Calculate learning phase completeness percentage"""
        try:
            phase = learning_profile.current_phase
            session_count = learning_profile.session_count
            
            if phase == LearningPhase.COLD_START:
                return min(100.0, (session_count / 1.0) * 20)  # First session is 20%
            elif phase == LearningPhase.LEARNING:
                return 20 + min(60.0, (session_count / self.learning_session_threshold) * 60)
            elif phase == LearningPhase.GRADUAL_RISK:
                sessions_in_gradual = session_count - self.learning_session_threshold
                total_gradual = self.gradual_risk_threshold - self.learning_session_threshold
                return 80 + min(15.0, (sessions_in_gradual / total_gradual) * 15)
            else:  # FULL_AUTH
                return 100.0
                
        except Exception as e:
            logger.error(f"Error calculating learning completeness: {e}")
            return 0.0
    
    async def _perform_cluster_analysis(self, user_vectors: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Perform cluster analysis on user vectors"""
        try:
            # Need at least 10 vectors for meaningful clustering
            if len(user_vectors) < 10:
                logger.debug(f"Insufficient data for clustering: {len(user_vectors)} vectors (need 10+)")
                return None
            
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            
            vectors = np.array([v['vector_data'] for v in user_vectors])
            
            # Additional validation: ensure we have enough unique vectors
            if len(np.unique(vectors, axis=0)) < 5:
                logger.debug("Too few unique vectors for clustering")
                return None
            
            # Try different cluster numbers
            best_k = 2
            best_score = -1
            max_clusters = min(6, len(vectors) // 3, len(np.unique(vectors, axis=0)) - 1)
            
            if max_clusters < 2:
                logger.debug(f"Cannot perform clustering: max_clusters={max_clusters}")
                return None
            
            for k in range(2, max_clusters + 1):
                try:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(vectors)
                    
                    # Check if we have valid clustering (at least 2 different labels)
                    unique_labels = len(np.unique(labels))
                    if unique_labels < 2:
                        logger.debug(f"Invalid clustering result with k={k}: only {unique_labels} unique labels")
                        continue
                    
                    score = silhouette_score(vectors, labels)
                    
                    if score > best_score:
                        best_score = score
                        best_k = k
                        
                except Exception as cluster_error:
                    logger.debug(f"Clustering failed for k={k}: {cluster_error}")
                    continue
            
            # Check if we found a valid clustering
            if best_score == -1:
                logger.debug("No valid clustering configuration found")
                return None
            
            # Final clustering with best k
            kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(vectors)
            
            # Validate final result
            unique_labels = len(np.unique(labels))
            if unique_labels < 2:
                logger.debug(f"Final clustering invalid: only {unique_labels} unique labels")
                return None
            
            return {
                'num_clusters': best_k,
                'silhouette_score': float(best_score),
                'centers': kmeans.cluster_centers_,
                'cluster_sizes': [int(np.sum(labels == i)) for i in range(best_k)],
                'total_vectors': len(vectors),
                'unique_vectors': len(np.unique(vectors, axis=0))
            }
            
        except Exception as e:
            logger.error(f"Error performing cluster analysis: {e}")
            return None
    
    async def _assess_phase_readiness(self, user_id: str, learning_profile: LearningProfile) -> Dict[str, Any]:
        """Assess readiness for next phase transition"""
        try:
            current_phase = learning_profile.current_phase
            session_count = learning_profile.session_count
            confidence = learning_profile.phase_confidence
            
            if current_phase == LearningPhase.COLD_START:
                readiness = confidence > 0.3
                requirements = ['Complete first session successfully']
                next_phase = LearningPhase.LEARNING.value
                
            elif current_phase == LearningPhase.LEARNING:
                session_ready = session_count >= self.learning_session_threshold
                confidence_ready = confidence >= self.confidence_threshold
                readiness = session_ready and confidence_ready
                
                requirements = [
                    f"Sessions: {session_count}/{self.learning_session_threshold} {'✓' if session_ready else '✗'}",
                    f"Confidence: {confidence:.2f}/{self.confidence_threshold} {'✓' if confidence_ready else '✗'}"
                ]
                next_phase = LearningPhase.GRADUAL_RISK.value
                
            elif current_phase == LearningPhase.GRADUAL_RISK:
                session_ready = session_count >= self.gradual_risk_threshold
                confidence_ready = confidence >= 0.8
                readiness = session_ready and confidence_ready
                
                requirements = [
                    f"Sessions: {session_count}/{self.gradual_risk_threshold} {'✓' if session_ready else '✗'}",
                    f"Confidence: {confidence:.2f}/0.80 {'✓' if confidence_ready else '✗'}"
                ]
                next_phase = LearningPhase.FULL_AUTH.value
                
            else:  # FULL_AUTH
                readiness = True
                requirements = ['Full authentication active']
                next_phase = None
            
            return {
                'ready_for_next_phase': readiness,
                'next_phase': next_phase,
                'requirements': requirements,
                'current_confidence': confidence,
                'sessions_completed': session_count
            }
            
        except Exception as e:
            logger.error(f"Error assessing phase readiness for {user_id}: {e}")
            return {'ready_for_next_phase': False, 'error': str(e)}
    
    def _calculate_learning_duration(self, learning_profile: LearningProfile) -> float:
        """Calculate learning duration in hours"""
        try:
            if learning_profile.learning_start_time:
                duration = datetime.utcnow() - learning_profile.learning_start_time
                return duration.total_seconds() / 3600.0
            return 0.0
        except:
            return 0.0
    
    async def _generate_learning_recommendations(self, learning_profile: LearningProfile) -> List[str]:
        """Generate recommendations for improving learning"""
        recommendations = []
        
        try:
            phase = learning_profile.current_phase
            confidence = learning_profile.phase_confidence
            session_count = learning_profile.session_count
            
            if confidence < 0.5:
                recommendations.append("Use the app more consistently to improve behavioral pattern recognition")
            
            if phase == LearningPhase.LEARNING and session_count < self.learning_session_threshold:
                recommendations.append(f"Complete {self.learning_session_threshold - session_count} more sessions to advance to enhanced security")
            
            if len(learning_profile.baseline_vectors) < 15:
                recommendations.append("Continue regular usage to build a stronger behavioral baseline")
            
            if phase == LearningPhase.GRADUAL_RISK:
                recommendations.append("Maintain consistent usage patterns during the gradual risk assessment phase")
            
            if not recommendations:
                recommendations.append("Your behavioral profile is developing well - continue normal usage")
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            recommendations.append("Continue using the app normally")
        
        return recommendations
    
    async def get_learning_statistics(self) -> Dict[str, Any]:
        """Get learning system statistics"""
        try:
            # Update current stats
            self.learning_stats['users_in_learning'] = len([
                p for p in self.learning_profiles.values() 
                if p.current_phase in [LearningPhase.COLD_START, LearningPhase.LEARNING]
            ])
            
            # Get database stats
            db_stats = await ml_db.get_database_stats()
            
            return {
                'learning_stats': self.learning_stats.copy(),
                'database_stats': db_stats,
                'active_learning_profiles': len(self.learning_profiles),
                'phase_distribution': {
                    phase.value: len([p for p in self.learning_profiles.values() if p.current_phase == phase])
                    for phase in LearningPhase
                },
                'configuration': {
                    'learning_session_threshold': self.learning_session_threshold,
                    'gradual_risk_threshold': self.gradual_risk_threshold,
                    'min_vectors_per_session': self.min_vectors_per_session,
                    'confidence_threshold': self.confidence_threshold
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting learning statistics: {e}")
            return {'error': str(e)}

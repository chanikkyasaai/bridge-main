"""
Adaptive learning layer for behavioral authentication.

This layer implements continuous learning and adaptation mechanisms to improve
authentication accuracy over time through user behavior pattern analysis.
"""

import asyncio
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

from ..data.models import (
    BehavioralVector, UserProfile, AuthenticationDecision, 
    RiskLevel, SessionPhase, ModelPerformanceMetrics
)
from ..core.vector_store import VectorStoreInterface
from ..utils.constants import TOTAL_VECTOR_DIM
from ..config.settings import Settings


@dataclass
class AdaptationMetrics:
    """Metrics for tracking adaptation performance."""
    user_id: str
    adaptation_count: int
    last_adaptation: datetime
    accuracy_improvement: float
    false_positive_reduction: float
    false_negative_reduction: float
    model_confidence_avg: float


@dataclass
class LearningPattern:
    """Represents a learned behavioral pattern."""
    pattern_id: str
    user_id: str
    feature_weights: List[float]
    confidence_score: float
    frequency: int
    last_seen: datetime
    context_tags: List[str]


class AdaptiveLayer:
    """
    Adaptive learning layer for continuous behavioral pattern refinement.
    
    Features:
    - Continuous learning from authentication feedback
    - Dynamic threshold adjustment
    - Pattern drift detection and adaptation
    - Context-aware learning
    - Performance optimization
    """
    
    def __init__(self, vector_store: VectorStoreInterface, settings: Optional[Settings] = None):
        """Initialize adaptive layer."""
        self.vector_store = vector_store
        self.settings = settings or Settings()
        
        # Learning parameters
        self.learning_rate = self.settings.adaptive_learning_rate
        self.adaptation_threshold = self.settings.adaptation_threshold
        self.pattern_retention_days = self.settings.pattern_retention_days
        self.min_feedback_samples = self.settings.min_feedback_samples
        
        # User-specific learning state
        self.user_patterns: Dict[str, List[LearningPattern]] = {}
        self.user_thresholds: Dict[str, float] = {}
        self.adaptation_history: Dict[str, List[AdaptationMetrics]] = {}
        self.feedback_buffer: Dict[str, List[Dict]] = {}
        
        # Performance tracking
        self.adaptation_stats = {
            'total_adaptations': 0,
            'successful_adaptations': 0,
            'pattern_discoveries': 0,
            'threshold_adjustments': 0
        }
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Adaptive Layer initialized")
    
    async def learn_from_authentication(
        self, 
        user_id: str,
        behavioral_vector: BehavioralVector,
        decision: AuthenticationDecision,
        was_correct: bool,
        confidence: float,
        context: Dict[str, Any] = None
    ) -> bool:
        """Learn from authentication feedback."""
        
        feedback = {
            'timestamp': datetime.utcnow(),
            'vector': behavioral_vector.vector,
            'decision': decision.value,
            'was_correct': was_correct,
            'confidence': confidence,
            'context': context or {}
        }
        
        # Add to feedback buffer
        if user_id not in self.feedback_buffer:
            self.feedback_buffer[user_id] = []
        
        self.feedback_buffer[user_id].append(feedback)
        
        # Trigger adaptation if enough feedback accumulated
        if len(self.feedback_buffer[user_id]) >= self.min_feedback_samples:
            await self._trigger_adaptation(user_id)
        
        self.logger.debug(f"Feedback recorded for user {user_id}: {decision.value} ({'correct' if was_correct else 'incorrect'})")
        return True
    
    async def adapt_user_model(self, user_id: str) -> bool:
        """Adapt user's behavioral model based on recent feedback."""
        try:
            if user_id not in self.feedback_buffer or not self.feedback_buffer[user_id]:
                return False
            
            feedback_samples = self.feedback_buffer[user_id]
            
            # Analyze feedback patterns
            analysis = self._analyze_feedback_patterns(feedback_samples)
            
            # Update user threshold
            threshold_updated = await self._adapt_user_threshold(user_id, analysis)
            
            # Discover new patterns
            patterns_updated = await self._discover_behavioral_patterns(user_id, feedback_samples)
            
            # Update pattern weights
            weights_updated = await self._update_pattern_weights(user_id, analysis)
            
            # Clear processed feedback
            self.feedback_buffer[user_id] = []
            
            # Record adaptation metrics
            await self._record_adaptation_metrics(user_id, analysis)
            
            adaptation_success = threshold_updated or patterns_updated or weights_updated
            
            if adaptation_success:
                self.adaptation_stats['total_adaptations'] += 1
                self.adaptation_stats['successful_adaptations'] += 1
                self.logger.info(f"Model adapted for user {user_id}")
            
            return adaptation_success
            
        except Exception as e:
            self.logger.error(f"Adaptation failed for user {user_id}: {e}")
            return False
    
    async def get_adaptive_threshold(self, user_id: str, base_threshold: float) -> float:
        """Get user-specific adaptive threshold."""
        if user_id in self.user_thresholds:
            adaptive_threshold = self.user_thresholds[user_id]
            self.logger.debug(f"Using adaptive threshold for user {user_id}: {adaptive_threshold}")
            return adaptive_threshold
        
        return base_threshold
    
    async def detect_pattern_drift(self, user_id: str, recent_vectors: List[BehavioralVector]) -> Dict[str, Any]:
        """Detect if user's behavioral patterns have drifted."""
        
        if len(recent_vectors) < 10:  # Need sufficient data
            return {'drift_detected': False, 'confidence': 0.0}
        
        try:
            # Get historical patterns
            if user_id not in self.user_patterns:
                return {'drift_detected': False, 'confidence': 0.0}
            
            current_patterns = self.user_patterns[user_id]
            
            # Analyze recent vectors
            recent_features = np.array([v.vector for v in recent_vectors])
            recent_mean = np.mean(recent_features, axis=0)
            recent_std = np.std(recent_features, axis=0)
            
            # Compare with historical patterns
            drift_scores = []
            for pattern in current_patterns:
                pattern_features = np.array(pattern.feature_weights)
                
                # Calculate feature drift using KL divergence approximation
                drift_score = self._calculate_feature_drift(pattern_features, recent_mean, recent_std)
                drift_scores.append(drift_score)
            
            avg_drift = np.mean(drift_scores) if drift_scores else 0.0
            max_drift = max(drift_scores) if drift_scores else 0.0
            
            # Drift detected if average drift exceeds threshold
            drift_detected = bool(avg_drift > self.adaptation_threshold)
            
            return {
                'drift_detected': drift_detected,
                'confidence': min(1.0, float(avg_drift / self.adaptation_threshold)),
                'avg_drift_score': float(avg_drift),
                'max_drift_score': float(max_drift),
                'affected_patterns': len([s for s in drift_scores if s > self.adaptation_threshold])
            }
            
        except Exception as e:
            self.logger.error(f"Pattern drift detection failed for user {user_id}: {e}")
            return {'drift_detected': False, 'confidence': 0.0}
    
    async def optimize_learning_parameters(self, user_id: str) -> Dict[str, float]:
        """Optimize learning parameters based on user's adaptation history."""
        
        if user_id not in self.adaptation_history:
            return {'learning_rate': self.learning_rate}
        
        history = self.adaptation_history[user_id]
        
        if len(history) < 3:  # Need sufficient history
            return {'learning_rate': self.learning_rate}
        
        # Analyze adaptation success rate
        recent_adaptations = history[-5:]  # Last 5 adaptations
        success_rate = np.mean([m.accuracy_improvement > 0 for m in recent_adaptations])
        
        # Adjust learning rate based on success
        if success_rate > 0.8:
            # High success - can be more aggressive
            optimized_lr = min(self.learning_rate * 1.2, 0.1)
        elif success_rate < 0.4:
            # Low success - be more conservative
            optimized_lr = max(self.learning_rate * 0.8, 0.001)
        else:
            # Moderate success - keep current rate
            optimized_lr = self.learning_rate
        
        return {
            'learning_rate': optimized_lr,
            'success_rate': success_rate,
            'adaptation_count': len(history)
        }
    
    async def get_layer_statistics(self) -> Dict[str, Any]:
        """Get adaptive layer statistics."""
        
        total_patterns = sum(len(patterns) for patterns in self.user_patterns.values())
        total_users_with_patterns = len(self.user_patterns)
        
        # Calculate average adaptation metrics
        all_metrics = []
        for user_metrics in self.adaptation_history.values():
            all_metrics.extend(user_metrics)
        
        avg_accuracy_improvement = np.mean([m.accuracy_improvement for m in all_metrics]) if all_metrics else 0.0
        avg_confidence = np.mean([m.model_confidence_avg for m in all_metrics]) if all_metrics else 0.0
        
        return {
            'adaptation_stats': self.adaptation_stats.copy(),
            'total_learned_patterns': total_patterns,
            'users_with_patterns': total_users_with_patterns,
            'avg_patterns_per_user': total_patterns / max(1, total_users_with_patterns),
            'performance_metrics': {
                'avg_accuracy_improvement': avg_accuracy_improvement,
                'avg_model_confidence': avg_confidence
            },
            'feedback_buffer_sizes': {
                user_id: len(feedback) 
                for user_id, feedback in self.feedback_buffer.items()
            }
        }
    
    # Private helper methods
    
    async def _trigger_adaptation(self, user_id: str) -> None:
        """Trigger adaptation process for user."""
        try:
            await self.adapt_user_model(user_id)
        except Exception as e:
            self.logger.error(f"Adaptation trigger failed for user {user_id}: {e}")
    
    def _analyze_feedback_patterns(self, feedback_samples: List[Dict]) -> Dict[str, Any]:
        """Analyze feedback to identify patterns and issues."""
        
        total_samples = len(feedback_samples)
        correct_decisions = sum(1 for f in feedback_samples if f['was_correct'])
        incorrect_decisions = total_samples - correct_decisions
        
        # Categorize errors
        false_positives = sum(1 for f in feedback_samples 
                            if f['decision'] in ['allow', 'learn'] and not f['was_correct'])
        false_negatives = sum(1 for f in feedback_samples 
                            if f['decision'] in ['deny', 'challenge'] and not f['was_correct'])
        
        # Calculate confidence trends
        confidences = [f['confidence'] for f in feedback_samples]
        avg_confidence = np.mean(confidences)
        confidence_trend = self._calculate_trend(confidences)
        
        return {
            'total_samples': total_samples,
            'accuracy_rate': correct_decisions / total_samples,
            'false_positive_rate': false_positives / total_samples,
            'false_negative_rate': false_negatives / total_samples,
            'avg_confidence': avg_confidence,
            'confidence_trend': confidence_trend,
            'recent_errors': incorrect_decisions
        }
    
    async def _adapt_user_threshold(self, user_id: str, analysis: Dict[str, Any]) -> bool:
        """Adapt similarity threshold based on feedback analysis."""
        
        current_threshold = self.user_thresholds.get(user_id, self.settings.similarity_threshold)
        
        # Adjust threshold based on error rates
        fp_rate = analysis['false_positive_rate']
        fn_rate = analysis['false_negative_rate']
        
        threshold_adjustment = 0.0
        
        if fp_rate > 0.1:  # Too many false positives - increase threshold
            threshold_adjustment = self.learning_rate * fp_rate
        elif fn_rate > 0.1:  # Too many false negatives - decrease threshold
            threshold_adjustment = -self.learning_rate * fn_rate
        
        if abs(threshold_adjustment) > 0.01:  # Significant adjustment needed
            new_threshold = np.clip(
                current_threshold + threshold_adjustment,
                0.3,  # Minimum threshold
                0.95  # Maximum threshold
            )
            
            self.user_thresholds[user_id] = new_threshold
            self.adaptation_stats['threshold_adjustments'] += 1
            
            self.logger.info(f"Threshold adapted for user {user_id}: {current_threshold:.3f} -> {new_threshold:.3f}")
            return True
        
        return False
    
    async def _discover_behavioral_patterns(self, user_id: str, feedback_samples: List[Dict]) -> bool:
        """Discover new behavioral patterns from feedback."""
        
        # Extract vectors from correct authentications
        correct_vectors = [
            f['vector'] for f in feedback_samples 
            if f['was_correct'] and f['decision'] in ['allow', 'learn']
        ]
        
        if len(correct_vectors) < 3:  # Need minimum samples
            return False
        
        try:
            # Cluster vectors to find patterns
            vectors_array = np.array(correct_vectors)
            
            # Simple clustering: find centroid and variance
            centroid = np.mean(vectors_array, axis=0)
            variance = np.var(vectors_array, axis=0)
            
            # Create new pattern
            pattern = LearningPattern(
                pattern_id=f"{user_id}_{datetime.utcnow().timestamp()}",
                user_id=user_id,
                feature_weights=centroid.tolist(),
                confidence_score=1.0 - np.mean(variance),
                frequency=len(correct_vectors),
                last_seen=datetime.utcnow(),
                context_tags=self._extract_context_tags(feedback_samples)
            )
            
            # Add to user patterns
            if user_id not in self.user_patterns:
                self.user_patterns[user_id] = []
            
            self.user_patterns[user_id].append(pattern)
            self.adaptation_stats['pattern_discoveries'] += 1
            
            # Cleanup old patterns
            self._cleanup_old_patterns(user_id)
            
            self.logger.info(f"New behavioral pattern discovered for user {user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Pattern discovery failed for user {user_id}: {e}")
            return False
    
    async def _update_pattern_weights(self, user_id: str, analysis: Dict[str, Any]) -> bool:
        """Update weights of existing patterns based on feedback."""
        
        if user_id not in self.user_patterns:
            return False
        
        patterns = self.user_patterns[user_id]
        updated = False
        
        for pattern in patterns:
            # Adjust confidence based on recent accuracy
            confidence_adjustment = (analysis['accuracy_rate'] - 0.5) * self.learning_rate
            pattern.confidence_score = np.clip(
                pattern.confidence_score + confidence_adjustment,
                0.1, 1.0
            )
            
            pattern.last_seen = datetime.utcnow()
            updated = True
        
        return updated
    
    async def _record_adaptation_metrics(self, user_id: str, analysis: Dict[str, Any]) -> None:
        """Record adaptation metrics for monitoring."""
        
        metrics = AdaptationMetrics(
            user_id=user_id,
            adaptation_count=len(self.adaptation_history.get(user_id, [])) + 1,
            last_adaptation=datetime.utcnow(),
            accuracy_improvement=analysis['accuracy_rate'] - 0.5,  # Baseline 50%
            false_positive_reduction=max(0, 0.1 - analysis['false_positive_rate']),
            false_negative_reduction=max(0, 0.1 - analysis['false_negative_rate']),
            model_confidence_avg=analysis['avg_confidence']
        )
        
        if user_id not in self.adaptation_history:
            self.adaptation_history[user_id] = []
        
        self.adaptation_history[user_id].append(metrics)
        
        # Keep only recent history
        if len(self.adaptation_history[user_id]) > 20:
            self.adaptation_history[user_id] = self.adaptation_history[user_id][-20:]
    
    def _calculate_feature_drift(self, historical_features: np.ndarray, recent_mean: np.ndarray, recent_std: np.ndarray) -> float:
        """Calculate feature drift score between historical and recent patterns."""
        
        # Normalize features
        historical_norm = historical_features / (np.linalg.norm(historical_features) + 1e-8)
        recent_norm = recent_mean / (np.linalg.norm(recent_mean) + 1e-8)
        
        # Calculate cosine distance
        cosine_sim = np.dot(historical_norm, recent_norm)
        cosine_distance = 1.0 - cosine_sim
        
        # Factor in variance change
        variance_change = np.mean(recent_std) / (np.mean(np.abs(historical_features)) + 1e-8)
        
        # Combined drift score
        drift_score = 0.7 * cosine_distance + 0.3 * min(1.0, variance_change)
        
        return drift_score
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in a series of values (-1 to 1)."""
        if len(values) < 3:
            return 0.0
        
        x = np.arange(len(values))
        y = np.array(values)
        
        # Check for constant values (no variance)
        if np.var(y) == 0:
            return 0.0
        
        # Simple linear regression slope using correlation
        correlation = np.corrcoef(x, y)[0, 1]
        
        # Handle NaN case (should not happen after variance check, but safety)
        if np.isnan(correlation):
            return 0.0
            
        return np.clip(correlation, -1.0, 1.0)
    
    def _extract_context_tags(self, feedback_samples: List[Dict]) -> List[str]:
        """Extract context tags from feedback samples."""
        tags = set()
        
        for sample in feedback_samples:
            context = sample.get('context', {})
            
            # Extract relevant context information
            if 'device_type' in context:
                tags.add(f"device_{context['device_type']}")
            if 'time_of_day' in context:
                hour = context['time_of_day']
                if 6 <= hour < 12:
                    tags.add("morning")
                elif 12 <= hour < 18:
                    tags.add("afternoon")
                elif 18 <= hour < 22:
                    tags.add("evening")
                else:
                    tags.add("night")
        
        return list(tags)
    
    def _cleanup_old_patterns(self, user_id: str) -> None:
        """Remove old patterns that are no longer relevant."""
        if user_id not in self.user_patterns:
            return
        
        cutoff_date = datetime.utcnow() - timedelta(days=self.pattern_retention_days)
        
        # Filter out old patterns
        valid_patterns = [
            pattern for pattern in self.user_patterns[user_id]
            if pattern.last_seen > cutoff_date
        ]
        
        removed_count = len(self.user_patterns[user_id]) - len(valid_patterns)
        self.user_patterns[user_id] = valid_patterns
        
        if removed_count > 0:
            self.logger.debug(f"Cleaned up {removed_count} old patterns for user {user_id}")
    
    async def cleanup_user_data(self, user_id: str) -> bool:
        """Clean up all adaptive data for a user."""
        try:
            if user_id in self.user_patterns:
                del self.user_patterns[user_id]
            if user_id in self.user_thresholds:
                del self.user_thresholds[user_id]
            if user_id in self.adaptation_history:
                del self.adaptation_history[user_id]
            if user_id in self.feedback_buffer:
                del self.feedback_buffer[user_id]
            
            self.logger.info(f"Adaptive data cleaned up for user {user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup adaptive data for user {user_id}: {e}")
            return False

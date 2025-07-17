"""
Phase 2 Continuous Analysis System for Behavioral Authentication
Integrates FAISS + GNN/Transformer layers with behavioral drift detection
"""

import asyncio
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import warnings
warnings.filterwarnings('ignore')

from src.data.models import (
    BehavioralVector, UserProfile, AuthenticationDecision, 
    RiskLevel, SessionPhase
)
from src.core.ml_database import ml_db
from src.core.vector_store import VectorStoreInterface
from src.layers.faiss_layer import FAISSLayer
from src.layers.adaptive_layer import AdaptiveLayer
from src.data.behavioral_processor import BehavioralProcessor
from src.config.settings import get_settings

logger = logging.getLogger(__name__)

class AnalysisLevel(str, Enum):
    """Analysis complexity levels"""
    BASIC_FAISS = "basic_faiss"
    ENHANCED_FAISS = "enhanced_faiss"
    GNN_TRANSFORMER = "gnn_transformer"
    FULL_ENSEMBLE = "full_ensemble"

class DriftType(str, Enum):
    """Types of behavioral drift"""
    GRADUAL_DRIFT = "gradual_drift"
    SUDDEN_DRIFT = "sudden_drift"
    RECURRING_PATTERN = "recurring_pattern"
    ANOMALY = "anomaly"

@dataclass
class BehavioralDriftAlert:
    """Alert for detected behavioral drift"""
    user_id: str
    drift_type: DriftType
    severity: float  # 0.0 to 1.0
    confidence: float
    detection_timestamp: datetime
    baseline_deviation: float
    risk_assessment: RiskLevel
    recommended_action: str
    drift_metrics: Dict[str, float]

@dataclass
class AnalysisResult:
    """Result from continuous analysis system"""
    decision: AuthenticationDecision
    confidence: float
    risk_level: RiskLevel
    risk_score: float
    analysis_level: AnalysisLevel
    processing_time_ms: int
    similarity_scores: Dict[str, float]
    risk_factors: List[str]
    drift_analysis: Optional[BehavioralDriftAlert]
    layer_decisions: Dict[str, Dict[str, Any]]

class Phase2ContinuousAnalysis:
    """
    Phase 2 Continuous Analysis System
    
    Features:
    - Level 1: Enhanced FAISS with database integration
    - Level 2: GNN/Transformer behavioral modeling
    - Real-time decision making with risk assessment
    - Behavioral drift detection and adaptation
    - Multi-layer ensemble decisions
    - Statistical monitoring with ADWIN/DDM algorithms
    """
    
    def __init__(self, vector_store: VectorStoreInterface, faiss_layer: FAISSLayer, 
                 adaptive_layer: AdaptiveLayer):
        self.vector_store = vector_store
        self.faiss_layer = faiss_layer
        self.adaptive_layer = adaptive_layer
        self.settings = get_settings()
        
        # Analysis configuration
        self.similarity_threshold_high = 0.85
        self.similarity_threshold_medium = 0.70
        self.similarity_threshold_low = 0.50
        self.drift_detection_window = 20  # Vectors to analyze for drift
        self.drift_threshold = 0.15  # Threshold for drift detection
        
        # Behavioral drift detection
        self.user_baselines: Dict[str, np.ndarray] = {}  # User baseline patterns
        self.drift_histories: Dict[str, List[float]] = {}  # Drift history per user
        self.analysis_statistics = {
            'total_analyses': 0,
            'faiss_decisions': 0,
            'gnn_decisions': 0,
            'ensemble_decisions': 0,
            'drift_alerts_triggered': 0,
            'successful_authentications': 0,
            'blocked_attempts': 0
        }
        
        # Layer weights for ensemble decisions
        self.layer_weights = {
            'faiss': 0.4,
            'gnn_transformer': 0.6
        }
        
        logger.info("Phase 2 Continuous Analysis System initialized")
    
    async def analyze_behavioral_vector(self, user_id: str, session_id: str, 
                                      behavioral_vector: BehavioralVector,
                                      user_profile: UserProfile) -> AnalysisResult:
        """
        Main analysis function - orchestrates all layers and drift detection
        
        Returns:
            Comprehensive analysis result with decision and risk assessment
        """
        start_time = datetime.utcnow()
        
        try:
            # Store vector in database first
            vector_data = behavioral_vector.vector
            if hasattr(vector_data, 'tolist'):
                vector_data = vector_data.tolist()
            elif not isinstance(vector_data, list):
                vector_data = list(vector_data)
                
            vector_id = await ml_db.store_behavioral_vector(
                user_id=user_id,
                session_id=session_id,
                vector_data=vector_data,
                confidence_score=behavioral_vector.confidence_score,
                feature_source='continuous_analysis'
            )
            
            # Determine analysis level based on user profile and risk context
            analysis_level = self._determine_analysis_level(user_profile)
            
            # Layer 1: Enhanced FAISS Analysis
            faiss_result = await self._perform_faiss_analysis(
                user_id, behavioral_vector, user_profile
            )
            
            # Layer 2: GNN/Transformer Analysis (if needed)
            gnn_result = None
            if analysis_level in [AnalysisLevel.GNN_TRANSFORMER, AnalysisLevel.FULL_ENSEMBLE]:
                gnn_result = await self._perform_gnn_analysis(
                    user_id, behavioral_vector, user_profile
                )
            
            # Behavioral drift detection
            drift_analysis = await self._detect_behavioral_drift(
                user_id, behavioral_vector
            )
            
            # Ensemble decision making
            final_decision = await self._make_ensemble_decision(
                faiss_result, gnn_result, drift_analysis, user_profile
            )
            
            # Calculate processing time
            processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            # Store authentication decision in database
            await ml_db.store_authentication_decision(
                user_id=user_id,
                session_id=session_id,
                decision=final_decision.decision.value,
                confidence=final_decision.confidence,
                similarity_score=final_decision.similarity_scores.get('average', 0.0),
                layer_used=final_decision.analysis_level.value,
                risk_factors=final_decision.risk_factors,
                processing_time_ms=processing_time
            )
            
            # Update statistics
            self._update_analysis_statistics(final_decision)
            
            logger.info(f"Analysis complete for {user_id}: {final_decision.decision.value} "
                       f"(confidence: {final_decision.confidence:.3f}, time: {processing_time}ms)")
            
            return final_decision
            
        except Exception as e:
            logger.error(f"Error in behavioral analysis for {user_id}: {e}")
            
            # Return safe fallback decision
            return AnalysisResult(
                decision=AuthenticationDecision.CHALLENGE,
                confidence=0.3,
                risk_level=RiskLevel.MEDIUM,
                risk_score=0.6,
                analysis_level=AnalysisLevel.BASIC_FAISS,
                processing_time_ms=int((datetime.utcnow() - start_time).total_seconds() * 1000),
                similarity_scores={},
                risk_factors=[f"Analysis error: {str(e)}"],
                drift_analysis=None,
                layer_decisions={'error': {'message': str(e)}}
            )
    
    def _determine_analysis_level(self, user_profile: UserProfile) -> AnalysisLevel:
        """Determine appropriate analysis level based on user profile and context"""
        try:
            # Check user phase
            if user_profile.current_phase.value in ['learning', 'cold_start']:
                return AnalysisLevel.BASIC_FAISS
            elif user_profile.current_phase.value == 'gradual_risk':
                return AnalysisLevel.ENHANCED_FAISS
            else:  # full_auth
                # Use risk score to determine level
                if user_profile.risk_score > 0.7:
                    return AnalysisLevel.FULL_ENSEMBLE
                elif user_profile.risk_score > 0.4:
                    return AnalysisLevel.GNN_TRANSFORMER
                else:
                    return AnalysisLevel.ENHANCED_FAISS
                    
        except Exception as e:
            logger.error(f"Error determining analysis level: {e}")
            return AnalysisLevel.BASIC_FAISS
    
    async def _perform_faiss_analysis(self, user_id: str, behavioral_vector: BehavioralVector,
                                    user_profile: UserProfile) -> Dict[str, Any]:
        """Perform Level 1 FAISS analysis with database integration"""
        try:
            # Get FAISS decision
            decision, risk_level, risk_score, confidence, factors = await self.faiss_layer.make_authentication_decision(
                user_id, behavioral_vector, user_profile
            )
            
            # Get similarity scores
            similarity_scores = await self.faiss_layer.compute_similarity_scores(
                user_id, behavioral_vector, top_k=10
            )
            
            # Calculate enhanced metrics
            avg_similarity = np.mean(list(similarity_scores.values())) if similarity_scores else 0.0
            max_similarity = max(similarity_scores.values()) if similarity_scores else 0.0
            min_similarity = min(similarity_scores.values()) if similarity_scores else 0.0
            
            # Add vector to FAISS index for future comparisons
            await self.faiss_layer.add_vector_to_index(user_id, behavioral_vector)
            
            result = {
                'decision': decision,
                'risk_level': risk_level,
                'risk_score': risk_score,
                'confidence': confidence,
                'risk_factors': factors,
                'similarity_scores': {
                    'average': avg_similarity,
                    'maximum': max_similarity,
                    'minimum': min_similarity,
                    'individual': similarity_scores
                },
                'analysis_type': 'faiss_enhanced'
            }
            
            logger.debug(f"FAISS analysis for {user_id}: {decision.value} (avg_sim: {avg_similarity:.3f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in FAISS analysis for {user_id}: {e}")
            return {
                'decision': AuthenticationDecision.CHALLENGE,
                'risk_level': RiskLevel.MEDIUM,
                'risk_score': 0.6,
                'confidence': 0.3,
                'risk_factors': [f"FAISS analysis error: {str(e)}"],
                'similarity_scores': {},
                'analysis_type': 'faiss_error'
            }
    
    async def _perform_gnn_analysis(self, user_id: str, behavioral_vector: BehavioralVector,
                                  user_profile: UserProfile) -> Dict[str, Any]:
        """Perform Level 2 GNN/Transformer analysis"""
        try:
            # Use adaptive layer for advanced analysis
            analysis_result = await self.adaptive_layer.analyze_behavior(
                user_id, behavioral_vector, user_profile
            )
            
            # Enhanced GNN-style analysis simulation
            # In production, this would use actual GNN/Transformer models
            gnn_confidence = await self._simulate_gnn_confidence(user_id, behavioral_vector)
            transformer_score = await self._simulate_transformer_analysis(user_id, behavioral_vector)
            
            # Combine results
            combined_confidence = (analysis_result.confidence + gnn_confidence + transformer_score) / 3.0
            
            # Determine GNN decision based on combined analysis
            if combined_confidence > 0.8:
                gnn_decision = AuthenticationDecision.ALLOW
                gnn_risk = RiskLevel.LOW
            elif combined_confidence > 0.6:
                gnn_decision = AuthenticationDecision.CHALLENGE
                gnn_risk = RiskLevel.MEDIUM
            else:
                gnn_decision = AuthenticationDecision.BLOCK
                gnn_risk = RiskLevel.HIGH
            
            result = {
                'decision': gnn_decision,
                'risk_level': gnn_risk,
                'risk_score': 1.0 - combined_confidence,
                'confidence': combined_confidence,
                'risk_factors': [
                    f"GNN confidence: {gnn_confidence:.3f}",
                    f"Transformer score: {transformer_score:.3f}",
                    f"Adaptive layer: {analysis_result.confidence:.3f}"
                ],
                'gnn_metrics': {
                    'gnn_confidence': gnn_confidence,
                    'transformer_score': transformer_score,
                    'adaptive_confidence': analysis_result.confidence,
                    'combined_confidence': combined_confidence
                },
                'analysis_type': 'gnn_transformer'
            }
            
            logger.debug(f"GNN analysis for {user_id}: {gnn_decision.value} (confidence: {combined_confidence:.3f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in GNN analysis for {user_id}: {e}")
            return {
                'decision': AuthenticationDecision.CHALLENGE,
                'risk_level': RiskLevel.MEDIUM,
                'risk_score': 0.6,
                'confidence': 0.4,
                'risk_factors': [f"GNN analysis error: {str(e)}"],
                'gnn_metrics': {},
                'analysis_type': 'gnn_error'
            }
    
    async def _simulate_gnn_confidence(self, user_id: str, behavioral_vector: BehavioralVector) -> float:
        """Simulate GNN analysis confidence (placeholder for actual GNN implementation)"""
        try:
            # Get recent user vectors for graph-based analysis
            recent_vectors = await ml_db.get_user_vectors(user_id, limit=20)
            
            if len(recent_vectors) < 5:
                return 0.5  # Medium confidence for limited data
            
            # Simulate graph neural network analysis
            # This would use actual GNN models in production
            vector_similarities = []
            current_vector = behavioral_vector.vector
            
            for vector_data in recent_vectors[:10]:
                stored_vector = np.array(vector_data['vector_data'])
                similarity = np.dot(current_vector, stored_vector) / (
                    np.linalg.norm(current_vector) * np.linalg.norm(stored_vector)
                )
                vector_similarities.append(similarity)
            
            # Graph-based confidence simulation
            avg_similarity = np.mean(vector_similarities)
            consistency = 1.0 - np.std(vector_similarities)
            
            # Combine metrics
            gnn_confidence = (avg_similarity * 0.7 + consistency * 0.3)
            return max(0.0, min(1.0, gnn_confidence))
            
        except Exception as e:
            logger.error(f"Error simulating GNN confidence: {e}")
            return 0.5
    
    async def _simulate_transformer_analysis(self, user_id: str, behavioral_vector: BehavioralVector) -> float:
        """Simulate transformer-based sequence analysis"""
        try:
            # Get recent vectors for sequence analysis
            recent_vectors = await ml_db.get_user_vectors(user_id, limit=15)
            
            if len(recent_vectors) < 3:
                return 0.5
            
            # Simulate transformer attention mechanism
            current_vector = behavioral_vector.vector
            attention_scores = []
            
            for i, vector_data in enumerate(recent_vectors[:10]):
                stored_vector = np.array(vector_data['vector_data'])
                
                # Simulate attention score (recency weighted)
                recency_weight = np.exp(-i * 0.1)  # Decay with age
                similarity = np.dot(current_vector, stored_vector) / (
                    np.linalg.norm(current_vector) * np.linalg.norm(stored_vector)
                )
                attention_score = similarity * recency_weight
                attention_scores.append(attention_score)
            
            # Transformer-style aggregation
            transformer_score = np.mean(attention_scores) if attention_scores else 0.5
            return max(0.0, min(1.0, transformer_score))
            
        except Exception as e:
            logger.error(f"Error simulating transformer analysis: {e}")
            return 0.5
    
    async def _detect_behavioral_drift(self, user_id: str, 
                                     behavioral_vector: BehavioralVector) -> Optional[BehavioralDriftAlert]:
        """Detect behavioral drift using statistical monitoring"""
        try:
            # Get user's recent vectors for drift analysis
            recent_vectors = await ml_db.get_user_vectors(user_id, limit=self.drift_detection_window)
            
            if len(recent_vectors) < 10:
                return None  # Not enough data for drift detection
            
            # Convert to numpy arrays
            vectors = np.array([v['vector_data'] for v in recent_vectors])
            current_vector = behavioral_vector.vector
            
            # Update or create baseline
            if user_id not in self.user_baselines:
                # Create initial baseline from first 60% of vectors
                baseline_count = max(5, len(vectors) * 6 // 10)
                self.user_baselines[user_id] = np.mean(vectors[:baseline_count], axis=0)
            
            baseline = self.user_baselines[user_id]
            
            # Calculate drift metrics
            baseline_deviation = np.linalg.norm(current_vector - baseline)
            
            # Statistical drift detection using ADWIN-style algorithm
            drift_score = await self._calculate_drift_score(user_id, current_vector, vectors)
            
            # Detect drift types
            drift_type, severity = self._classify_drift(
                baseline_deviation, drift_score, vectors, current_vector
            )
            
            # Determine if drift is significant
            if severity > self.drift_threshold:
                # Update drift history
                if user_id not in self.drift_histories:
                    self.drift_histories[user_id] = []
                self.drift_histories[user_id].append(drift_score)
                
                # Keep only recent drift history
                if len(self.drift_histories[user_id]) > 50:
                    self.drift_histories[user_id] = self.drift_histories[user_id][-50:]
                
                # Determine risk level
                if severity > 0.7:
                    risk_level = RiskLevel.HIGH
                    recommended_action = "Block access and require additional verification"
                elif severity > 0.4:
                    risk_level = RiskLevel.MEDIUM
                    recommended_action = "Challenge with additional authentication"
                else:
                    risk_level = RiskLevel.LOW
                    recommended_action = "Monitor closely for continued drift"
                
                # Create drift alert
                drift_alert = BehavioralDriftAlert(
                    user_id=user_id,
                    drift_type=drift_type,
                    severity=severity,
                    confidence=min(0.9, severity + 0.2),
                    detection_timestamp=datetime.utcnow(),
                    baseline_deviation=baseline_deviation,
                    risk_assessment=risk_level,
                    recommended_action=recommended_action,
                    drift_metrics={
                        'drift_score': drift_score,
                        'baseline_deviation': baseline_deviation,
                        'vector_consistency': float(np.std(vectors[-5:], axis=0).mean()),
                        'recent_variance': float(np.var(vectors[-5:], axis=0).mean())
                    }
                )
                
                self.analysis_statistics['drift_alerts_triggered'] += 1
                
                logger.warning(f"Behavioral drift detected for {user_id}: {drift_type.value} "
                             f"(severity: {severity:.3f})")
                
                return drift_alert
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting behavioral drift for {user_id}: {e}")
            return None
    
    async def _calculate_drift_score(self, user_id: str, current_vector: np.ndarray, 
                                   recent_vectors: np.ndarray) -> float:
        """Calculate drift score using statistical methods"""
        try:
            # Calculate sliding window statistics
            window_size = min(10, len(recent_vectors) // 2)
            
            if len(recent_vectors) < window_size * 2:
                return 0.0
            
            # Compare recent window vs historical window
            recent_window = recent_vectors[:window_size]
            historical_window = recent_vectors[window_size:window_size*2]
            
            # Calculate distributions
            recent_mean = np.mean(recent_window, axis=0)
            historical_mean = np.mean(historical_window, axis=0)
            
            # Kolmogorov-Smirnov style test simulation
            distribution_difference = np.linalg.norm(recent_mean - historical_mean)
            
            # Current vector deviation from recent pattern
            current_deviation = np.linalg.norm(current_vector - recent_mean)
            
            # Combine metrics
            drift_score = (distribution_difference * 0.6 + current_deviation * 0.4)
            
            return min(1.0, drift_score)
            
        except Exception as e:
            logger.error(f"Error calculating drift score: {e}")
            return 0.0
    
    def _classify_drift(self, baseline_deviation: float, drift_score: float, 
                       vectors: np.ndarray, current_vector: np.ndarray) -> Tuple[DriftType, float]:
        """Classify type and severity of behavioral drift"""
        try:
            # Calculate recent variance and trend
            recent_variance = np.var(vectors[-5:], axis=0).mean() if len(vectors) >= 5 else 0.0
            
            # Calculate trend (are patterns changing gradually or suddenly?)
            if len(vectors) >= 10:
                first_half = vectors[:len(vectors)//2]
                second_half = vectors[len(vectors)//2:]
                trend_difference = np.linalg.norm(
                    np.mean(second_half, axis=0) - np.mean(first_half, axis=0)
                )
            else:
                trend_difference = 0.0
            
            # Classify drift type and calculate severity
            if drift_score > 0.6 and baseline_deviation > 0.8:
                # Sudden, significant change
                drift_type = DriftType.SUDDEN_DRIFT
                severity = min(1.0, (drift_score + baseline_deviation) / 2.0)
                
            elif trend_difference > 0.3 and recent_variance > 0.2:
                # Gradual change over time
                drift_type = DriftType.GRADUAL_DRIFT
                severity = min(0.8, trend_difference + recent_variance * 0.5)
                
            elif baseline_deviation > 1.0:
                # Single anomalous behavior
                drift_type = DriftType.ANOMALY
                severity = min(0.9, baseline_deviation * 0.8)
                
            else:
                # Possible recurring pattern
                drift_type = DriftType.RECURRING_PATTERN
                severity = min(0.6, drift_score * 0.8)
            
            return drift_type, severity
            
        except Exception as e:
            logger.error(f"Error classifying drift: {e}")
            return DriftType.ANOMALY, 0.5
    
    async def _make_ensemble_decision(self, faiss_result: Dict[str, Any], 
                                    gnn_result: Optional[Dict[str, Any]],
                                    drift_analysis: Optional[BehavioralDriftAlert],
                                    user_profile: UserProfile) -> AnalysisResult:
        """Make final ensemble decision combining all analysis layers"""
        try:
            layer_decisions = {'faiss': faiss_result}
            
            # Start with FAISS as base decision
            base_decision = faiss_result['decision']
            base_confidence = faiss_result['confidence']
            base_risk_score = faiss_result['risk_score']
            base_risk_level = faiss_result['risk_level']
            
            risk_factors = faiss_result['risk_factors'].copy()
            
            # Factor in GNN/Transformer results if available
            if gnn_result:
                layer_decisions['gnn_transformer'] = gnn_result
                
                # Weighted ensemble
                faiss_weight = self.layer_weights['faiss']
                gnn_weight = self.layer_weights['gnn_transformer']
                
                ensemble_confidence = (
                    base_confidence * faiss_weight + 
                    gnn_result['confidence'] * gnn_weight
                )
                
                ensemble_risk_score = (
                    base_risk_score * faiss_weight + 
                    gnn_result['risk_score'] * gnn_weight
                )
                
                # Consensus decision making
                if base_decision == gnn_result['decision']:
                    # Agreement between layers
                    final_decision = base_decision
                    final_confidence = min(0.95, ensemble_confidence + 0.1)  # Boost for agreement
                else:
                    # Disagreement - use more conservative approach
                    if base_decision == AuthenticationDecision.BLOCK or gnn_result['decision'] == AuthenticationDecision.BLOCK:
                        final_decision = AuthenticationDecision.BLOCK
                    elif base_decision == AuthenticationDecision.CHALLENGE or gnn_result['decision'] == AuthenticationDecision.CHALLENGE:
                        final_decision = AuthenticationDecision.CHALLENGE
                    else:
                        final_decision = AuthenticationDecision.ALLOW
                    
                    final_confidence = ensemble_confidence * 0.8  # Reduce confidence for disagreement
                    risk_factors.append(f"Layer disagreement: FAISS={base_decision.value}, GNN={gnn_result['decision'].value}")
                
                analysis_level = AnalysisLevel.FULL_ENSEMBLE
                base_risk_score = ensemble_risk_score
                
            else:
                final_decision = base_decision
                final_confidence = base_confidence
                analysis_level = AnalysisLevel.ENHANCED_FAISS
            
            # Factor in drift analysis
            if drift_analysis:
                layer_decisions['drift_detection'] = {
                    'drift_type': drift_analysis.drift_type.value,
                    'severity': drift_analysis.severity,
                    'confidence': drift_analysis.confidence,
                    'risk_assessment': drift_analysis.risk_assessment.value
                }
                
                # Adjust decision based on drift severity
                if drift_analysis.severity > 0.7:
                    final_decision = AuthenticationDecision.BLOCK
                    final_confidence = max(0.8, final_confidence)
                    risk_factors.append(f"High behavioral drift detected: {drift_analysis.drift_type.value}")
                    base_risk_level = RiskLevel.HIGH
                    
                elif drift_analysis.severity > 0.4:
                    if final_decision == AuthenticationDecision.ALLOW:
                        final_decision = AuthenticationDecision.CHALLENGE
                    final_confidence *= 0.8
                    risk_factors.append(f"Moderate behavioral drift: {drift_analysis.drift_type.value}")
                    base_risk_level = RiskLevel.MEDIUM
                    
                else:
                    risk_factors.append(f"Minor behavioral variation detected")
            
            # Final risk level adjustment
            if final_confidence > 0.8 and base_risk_score < 0.3:
                final_risk_level = RiskLevel.LOW
            elif final_confidence > 0.6 and base_risk_score < 0.6:
                final_risk_level = RiskLevel.MEDIUM
            else:
                final_risk_level = RiskLevel.HIGH
            
            # Build similarity scores summary
            similarity_scores = faiss_result.get('similarity_scores', {})
            if gnn_result and 'gnn_metrics' in gnn_result:
                similarity_scores.update(gnn_result['gnn_metrics'])
            
            return AnalysisResult(
                decision=final_decision,
                confidence=final_confidence,
                risk_level=final_risk_level,
                risk_score=base_risk_score,
                analysis_level=analysis_level,
                processing_time_ms=0,  # Will be set by caller
                similarity_scores=similarity_scores,
                risk_factors=risk_factors,
                drift_analysis=drift_analysis,
                layer_decisions=layer_decisions
            )
            
        except Exception as e:
            logger.error(f"Error making ensemble decision: {e}")
            
            # Fallback to safe decision
            return AnalysisResult(
                decision=AuthenticationDecision.CHALLENGE,
                confidence=0.3,
                risk_level=RiskLevel.MEDIUM,
                risk_score=0.7,
                analysis_level=AnalysisLevel.BASIC_FAISS,
                processing_time_ms=0,
                similarity_scores={},
                risk_factors=[f"Ensemble decision error: {str(e)}"],
                drift_analysis=None,
                layer_decisions={'error': {'message': str(e)}}
            )
    
    def _update_analysis_statistics(self, result: AnalysisResult):
        """Update analysis statistics"""
        try:
            self.analysis_statistics['total_analyses'] += 1
            
            if result.analysis_level == AnalysisLevel.BASIC_FAISS or result.analysis_level == AnalysisLevel.ENHANCED_FAISS:
                self.analysis_statistics['faiss_decisions'] += 1
            elif result.analysis_level == AnalysisLevel.GNN_TRANSFORMER:
                self.analysis_statistics['gnn_decisions'] += 1
            elif result.analysis_level == AnalysisLevel.FULL_ENSEMBLE:
                self.analysis_statistics['ensemble_decisions'] += 1
            
            if result.decision == AuthenticationDecision.ALLOW:
                self.analysis_statistics['successful_authentications'] += 1
            elif result.decision == AuthenticationDecision.BLOCK:
                self.analysis_statistics['blocked_attempts'] += 1
                
        except Exception as e:
            logger.error(f"Error updating analysis statistics: {e}")
    
    async def adapt_user_baseline(self, user_id: str) -> bool:
        """Adapt user baseline based on recent behavioral patterns"""
        try:
            # Get recent vectors
            recent_vectors = await ml_db.get_user_vectors(user_id, limit=30)
            
            if len(recent_vectors) < 15:
                return False
            
            # Calculate new baseline from recent stable patterns
            vectors = np.array([v['vector_data'] for v in recent_vectors])
            
            # Use recent stable period (middle 50% to avoid extremes)
            stable_start = len(vectors) // 4
            stable_end = 3 * len(vectors) // 4
            stable_vectors = vectors[stable_start:stable_end]
            
            # Update baseline
            new_baseline = np.mean(stable_vectors, axis=0)
            self.user_baselines[user_id] = new_baseline
            
            logger.info(f"Adapted baseline for user {user_id} using {len(stable_vectors)} stable vectors")
            return True
            
        except Exception as e:
            logger.error(f"Error adapting user baseline for {user_id}: {e}")
            return False
    
    async def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get comprehensive analysis statistics"""
        try:
            # Get database stats
            db_stats = await ml_db.get_database_stats()
            
            # Get FAISS layer stats
            faiss_stats = await self.faiss_layer.get_layer_statistics()
            
            # Calculate drift statistics
            drift_stats = {
                'users_with_baselines': len(self.user_baselines),
                'users_with_drift_history': len(self.drift_histories),
                'avg_drift_score': np.mean([
                    np.mean(history[-10:]) for history in self.drift_histories.values() 
                    if len(history) > 0
                ]) if self.drift_histories else 0.0
            }
            
            return {
                'analysis_statistics': self.analysis_statistics.copy(),
                'database_stats': db_stats,
                'faiss_stats': faiss_stats,
                'drift_statistics': drift_stats,
                'configuration': {
                    'similarity_thresholds': {
                        'high': self.similarity_threshold_high,
                        'medium': self.similarity_threshold_medium,
                        'low': self.similarity_threshold_low
                    },
                    'drift_detection': {
                        'window_size': self.drift_detection_window,
                        'threshold': self.drift_threshold
                    },
                    'layer_weights': self.layer_weights
                },
                'active_users': {
                    'with_baselines': len(self.user_baselines),
                    'with_drift_tracking': len(self.drift_histories)
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting analysis statistics: {e}")
            return {'error': str(e)}

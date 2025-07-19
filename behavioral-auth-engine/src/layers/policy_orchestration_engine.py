"""
Layer J: Policy Orchestration Engine
4-level risk decision engine that combines FAISS, GNN, and contextual risk indicators.
Central decision-making component for national-level behavioral authentication.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json

from src.data.models import (
    BehavioralVector, UserProfile, AuthenticationDecision, 
    RiskLevel, SessionPhase
)
from src.layers.faiss_layer import FAISSLayer
from src.layers.gnn_anomaly_detector import GNNAnomalyDetector, GNNAnomalyResult
from src.layers.session_graph_generator import SessionGraph
from src.layers.adaptive_layer import AdaptiveLayer

logger = logging.getLogger(__name__)

class PolicyLevel(str, Enum):
    """4-level policy framework"""
    LEVEL_1_BASIC = "level_1_basic"           # FAISS only
    LEVEL_2_ENHANCED = "level_2_enhanced"     # FAISS + Adaptive
    LEVEL_3_ADVANCED = "level_3_advanced"    # FAISS + GNN + Context
    LEVEL_4_MAXIMUM = "level_4_maximum"      # Full ensemble + Explainability

class RiskContext(str, Enum):
    """Contextual risk factors"""
    HIGH_VALUE_TRANSACTION = "high_value_transaction"
    NEW_BENEFICIARY = "new_beneficiary"
    UNUSUAL_TIME = "unusual_time"
    SUSPICIOUS_LOCATION = "suspicious_location"
    RAPID_TRANSACTIONS = "rapid_transactions"
    MULTIPLE_FAILURES = "multiple_failures"
    DEVICE_CHANGE = "device_change"
    VPN_DETECTED = "vpn_detected"

class PolicyDecisionReason(str, Enum):
    """Reasons for policy decisions"""
    FAISS_HIGH_SIMILARITY = "faiss_high_similarity"
    FAISS_LOW_SIMILARITY = "faiss_low_similarity"
    GNN_ANOMALY_DETECTED = "gnn_anomaly_detected"
    GNN_NORMAL_PATTERN = "gnn_normal_pattern"
    CONTEXTUAL_RISK_HIGH = "contextual_risk_high"
    ADAPTIVE_DRIFT_DETECTED = "adaptive_drift_detected"
    ENSEMBLE_CONSENSUS = "ensemble_consensus"
    ENSEMBLE_DISAGREEMENT = "ensemble_disagreement"
    POLICY_OVERRIDE = "policy_override"

@dataclass
class ContextualRiskFactors:
    """Contextual risk indicators beyond behavioral analysis"""
    transaction_amount: float = 0.0
    is_new_beneficiary: bool = False
    time_of_day_risk: float = 0.0  # 0.0-1.0
    location_risk: float = 0.0     # 0.0-1.0
    device_risk: float = 0.0       # 0.0-1.0
    transaction_frequency_risk: float = 0.0
    recent_failures: int = 0
    vpn_detected: bool = False
    
    def get_overall_risk(self) -> float:
        """Calculate overall contextual risk score"""
        risk_factors = [
            min(1.0, self.transaction_amount / 100000),  # Normalize amount
            0.3 if self.is_new_beneficiary else 0.0,
            self.time_of_day_risk,
            self.location_risk,
            self.device_risk,
            self.transaction_frequency_risk,
            min(1.0, self.recent_failures * 0.2),
            0.4 if self.vpn_detected else 0.0
        ]
        return min(1.0, sum(risk_factors) / len(risk_factors))

@dataclass
class PolicyDecisionResult:
    """Complete policy decision with full explainability"""
    final_decision: AuthenticationDecision
    final_risk_level: RiskLevel
    final_risk_score: float
    confidence: float
    policy_level_used: PolicyLevel
    
    # Layer-specific results
    faiss_result: Optional[Dict[str, Any]] = None
    gnn_result: Optional[GNNAnomalyResult] = None
    adaptive_result: Optional[Dict[str, Any]] = None
    contextual_risk: Optional[ContextualRiskFactors] = None
    
    # Decision reasoning
    primary_reasons: List[PolicyDecisionReason] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    layer_weights: Dict[str, float] = field(default_factory=dict)
    
    # Explainability
    explanation: str = ""
    recommendations: List[str] = field(default_factory=list)
    
    # Performance metrics
    processing_time_ms: float = 0.0
    layer_timings: Dict[str, float] = field(default_factory=dict)

class PolicyOrchestrationEngine:
    """
    Layer J: Policy Orchestration Engine
    
    4-Level Risk Decision Framework:
    - Level 1: Basic FAISS similarity matching
    - Level 2: Enhanced FAISS + Adaptive learning
    - Level 3: Advanced FAISS + GNN + Contextual analysis  
    - Level 4: Maximum security with full ensemble + explainability
    
    Combines behavioral analysis with contextual risk factors for
    comprehensive authentication decisions.
    """
    
    def __init__(
        self,
        faiss_layer: FAISSLayer,
        gnn_detector: GNNAnomalyDetector,
        adaptive_layer: AdaptiveLayer,
        config: Dict[str, Any] = None
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config or {}
        
        # Core analysis layers
        self.faiss_layer = faiss_layer
        self.gnn_detector = gnn_detector
        self.adaptive_layer = adaptive_layer
        
        # Policy configuration
        self.policy_thresholds = {
            PolicyLevel.LEVEL_1_BASIC: {
                'allow_threshold': 0.8,
                'challenge_threshold': 0.6,
                'block_threshold': 0.3
            },
            PolicyLevel.LEVEL_2_ENHANCED: {
                'allow_threshold': 0.75,
                'challenge_threshold': 0.55,
                'block_threshold': 0.35
            },
            PolicyLevel.LEVEL_3_ADVANCED: {
                'allow_threshold': 0.7,
                'challenge_threshold': 0.5,
                'block_threshold': 0.4
            },
            PolicyLevel.LEVEL_4_MAXIMUM: {
                'allow_threshold': 0.65,
                'challenge_threshold': 0.45,
                'block_threshold': 0.45
            }
        }
        
        # Layer weights for ensemble decisions
        self.layer_weights = {
            PolicyLevel.LEVEL_1_BASIC: {
                'faiss': 1.0
            },
            PolicyLevel.LEVEL_2_ENHANCED: {
                'faiss': 0.7,
                'adaptive': 0.3
            },
            PolicyLevel.LEVEL_3_ADVANCED: {
                'faiss': 0.4,
                'gnn': 0.4,
                'contextual': 0.2
            },
            PolicyLevel.LEVEL_4_MAXIMUM: {
                'faiss': 0.3,
                'gnn': 0.4,
                'adaptive': 0.2,
                'contextual': 0.1
            }
        }
        
        # Contextual risk policies
        self.contextual_policies = self._initialize_contextual_policies()
        
        # Performance tracking
        self.decision_history: List[PolicyDecisionResult] = []
        self.performance_metrics = {
            'total_decisions': 0,
            'level_usage': {level: 0 for level in PolicyLevel},
            'avg_processing_time': 0.0,
            'decision_distribution': {decision: 0 for decision in AuthenticationDecision}
        }
        
        self.logger.info("Policy Orchestration Engine (Layer J) initialized")
    
    async def make_policy_decision(
        self,
        user_id: str,
        session_id: str,
        behavioral_vector: BehavioralVector,
        session_graph: SessionGraph,
        user_profile: UserProfile,
        contextual_factors: ContextualRiskFactors,
        requested_policy_level: Optional[PolicyLevel] = None
    ) -> PolicyDecisionResult:
        """
        Make comprehensive authentication decision using 4-level policy framework
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            behavioral_vector: Current behavioral vector
            session_graph: Session behavioral graph
            user_profile: User's behavioral profile
            contextual_factors: Contextual risk factors
            requested_policy_level: Specific policy level to use (optional)
            
        Returns:
            PolicyDecisionResult with complete decision reasoning
        """
        start_time = datetime.now()
        layer_timings = {}
        
        try:
            # Determine appropriate policy level
            policy_level = requested_policy_level or self._determine_policy_level(
                user_profile, contextual_factors
            )
            
            self.logger.info(f"Using policy level: {policy_level.value} for user {user_id}")
            
            # Execute analysis based on policy level
            if policy_level == PolicyLevel.LEVEL_1_BASIC:
                result = await self._execute_level_1_basic(
                    user_id, behavioral_vector, user_profile, layer_timings
                )
            elif policy_level == PolicyLevel.LEVEL_2_ENHANCED:
                result = await self._execute_level_2_enhanced(
                    user_id, behavioral_vector, user_profile, layer_timings
                )
            elif policy_level == PolicyLevel.LEVEL_3_ADVANCED:
                result = await self._execute_level_3_advanced(
                    user_id, behavioral_vector, session_graph, user_profile, 
                    contextual_factors, layer_timings
                )
            else:  # LEVEL_4_MAXIMUM
                result = await self._execute_level_4_maximum(
                    user_id, session_id, behavioral_vector, session_graph, 
                    user_profile, contextual_factors, layer_timings
                )
            
            # Set policy level and timing information
            result.policy_level_used = policy_level
            result.layer_timings = layer_timings
            result.processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            # Update performance metrics
            self._update_performance_metrics(result)
            
            # Store decision history
            self.decision_history.append(result)
            if len(self.decision_history) > 1000:  # Keep last 1000 decisions
                self.decision_history.pop(0)
            
            self.logger.info(
                f"Policy decision for {user_id}: {result.final_decision.value} "
                f"(risk: {result.final_risk_score:.3f}, confidence: {result.confidence:.3f})"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in policy decision making: {e}")
            return self._create_fallback_decision(str(e))
    
    def _determine_policy_level(
        self, 
        user_profile: UserProfile, 
        contextual_factors: ContextualRiskFactors
    ) -> PolicyLevel:
        """Determine appropriate policy level based on user and context"""
        
        # Calculate risk indicators
        contextual_risk = contextual_factors.get_overall_risk()
        user_risk = user_profile.drift_score + user_profile.false_positive_rate
        
        # High-risk scenarios require maximum security
        if (contextual_risk > 0.7 or 
            user_risk > 0.8 or
            contextual_factors.transaction_amount > 50000 or
            contextual_factors.recent_failures > 3):
            return PolicyLevel.LEVEL_4_MAXIMUM
        
        # Moderate risk scenarios
        elif (contextual_risk > 0.4 or 
              user_risk > 0.5 or
              contextual_factors.transaction_amount > 10000):
            return PolicyLevel.LEVEL_3_ADVANCED
        
        # Users in learning phase or low risk
        elif user_profile.current_phase in [SessionPhase.LEARNING, SessionPhase.COLD_START]:
            return PolicyLevel.LEVEL_1_BASIC
        
        # Default to enhanced level
        else:
            return PolicyLevel.LEVEL_2_ENHANCED
    
    async def _execute_level_1_basic(
        self,
        user_id: str,
        behavioral_vector: BehavioralVector,
        user_profile: UserProfile,
        layer_timings: Dict[str, float]
    ) -> PolicyDecisionResult:
        """Execute Level 1 Basic Policy - FAISS only"""
        
        # FAISS analysis
        faiss_start = datetime.now()
        faiss_decision, faiss_risk_level, faiss_risk_score, faiss_confidence, faiss_factors = \
            await self.faiss_layer.make_authentication_decision(
                user_id, behavioral_vector, user_profile
            )
        layer_timings['faiss'] = (datetime.now() - faiss_start).total_seconds() * 1000
        
        faiss_result = {
            'decision': faiss_decision,
            'risk_level': faiss_risk_level,
            'risk_score': faiss_risk_score,
            'confidence': faiss_confidence,
            'factors': faiss_factors
        }
        
        # Apply Level 1 thresholds
        thresholds = self.policy_thresholds[PolicyLevel.LEVEL_1_BASIC]
        final_decision, final_risk_level = self._apply_thresholds(
            faiss_confidence, thresholds
        )
        
        return PolicyDecisionResult(
            final_decision=final_decision,
            final_risk_level=final_risk_level,
            final_risk_score=faiss_risk_score,
            confidence=faiss_confidence,
            faiss_result=faiss_result,
            primary_reasons=[
                PolicyDecisionReason.FAISS_HIGH_SIMILARITY if faiss_confidence > 0.7 
                else PolicyDecisionReason.FAISS_LOW_SIMILARITY
            ],
            risk_factors=faiss_factors,
            layer_weights={'faiss': 1.0},
            explanation=f"Basic FAISS analysis: {faiss_confidence:.3f} similarity"
        )
    
    async def _execute_level_2_enhanced(
        self,
        user_id: str,
        behavioral_vector: BehavioralVector,
        user_profile: UserProfile,
        layer_timings: Dict[str, float]
    ) -> PolicyDecisionResult:
        """Execute Level 2 Enhanced Policy - FAISS + Adaptive"""
        
        # FAISS analysis
        faiss_start = datetime.now()
        faiss_decision, faiss_risk_level, faiss_risk_score, faiss_confidence, faiss_factors = \
            await self.faiss_layer.make_authentication_decision(
                user_id, behavioral_vector, user_profile
            )
        layer_timings['faiss'] = (datetime.now() - faiss_start).total_seconds() * 1000
        
        # Adaptive analysis
        adaptive_start = datetime.now()
        adaptive_result = await self.adaptive_layer.analyze_behavioral_pattern(
            user_id, behavioral_vector, user_profile
        )
        layer_timings['adaptive'] = (datetime.now() - adaptive_start).total_seconds() * 1000
        
        # Ensemble decision
        weights = self.layer_weights[PolicyLevel.LEVEL_2_ENHANCED]
        ensemble_confidence = (
            faiss_confidence * weights['faiss'] + 
            adaptive_result.confidence * weights['adaptive']
        )
        
        ensemble_risk_score = (
            faiss_risk_score * weights['faiss'] + 
            (1.0 - adaptive_result.confidence) * weights['adaptive']
        )
        
        # Apply Level 2 thresholds
        thresholds = self.policy_thresholds[PolicyLevel.LEVEL_2_ENHANCED]
        final_decision, final_risk_level = self._apply_thresholds(
            ensemble_confidence, thresholds
        )
        
        # Combine risk factors
        combined_factors = faiss_factors + [
            f"Adaptive confidence: {adaptive_result.confidence:.3f}",
            f"Pattern drift: {adaptive_result.drift_detected}"
        ]
        
        primary_reasons = []
        if adaptive_result.drift_detected:
            primary_reasons.append(PolicyDecisionReason.ADAPTIVE_DRIFT_DETECTED)
        if faiss_confidence > 0.7:
            primary_reasons.append(PolicyDecisionReason.FAISS_HIGH_SIMILARITY)
        elif faiss_confidence < 0.4:
            primary_reasons.append(PolicyDecisionReason.FAISS_LOW_SIMILARITY)
        
        return PolicyDecisionResult(
            final_decision=final_decision,
            final_risk_level=final_risk_level,
            final_risk_score=ensemble_risk_score,
            confidence=ensemble_confidence,
            faiss_result={
                'decision': faiss_decision,
                'risk_level': faiss_risk_level,
                'risk_score': faiss_risk_score,
                'confidence': faiss_confidence,
                'factors': faiss_factors
            },
            adaptive_result=adaptive_result.__dict__,
            primary_reasons=primary_reasons,
            risk_factors=combined_factors,
            layer_weights=weights,
            explanation=f"Enhanced analysis: FAISS {faiss_confidence:.3f}, Adaptive {adaptive_result.confidence:.3f}"
        )
    
    async def _execute_level_3_advanced(
        self,
        user_id: str,
        behavioral_vector: BehavioralVector,
        session_graph: SessionGraph,
        user_profile: UserProfile,
        contextual_factors: ContextualRiskFactors,
        layer_timings: Dict[str, float]
    ) -> PolicyDecisionResult:
        """Execute Level 3 Advanced Policy - FAISS + GNN + Context"""
        
        # FAISS analysis
        faiss_start = datetime.now()
        faiss_decision, faiss_risk_level, faiss_risk_score, faiss_confidence, faiss_factors = \
            await self.faiss_layer.make_authentication_decision(
                user_id, behavioral_vector, user_profile
            )
        layer_timings['faiss'] = (datetime.now() - faiss_start).total_seconds() * 1000
        
        # GNN analysis
        gnn_start = datetime.now()
        gnn_result = self.gnn_detector.detect_anomalies(
            session_graph, user_profile
        )
        layer_timings['gnn'] = (datetime.now() - gnn_start).total_seconds() * 1000
        
        # Contextual analysis
        contextual_start = datetime.now()
        contextual_risk_score = contextual_factors.get_overall_risk()
        layer_timings['contextual'] = (datetime.now() - contextual_start).total_seconds() * 1000
        
        # Ensemble decision
        weights = self.layer_weights[PolicyLevel.LEVEL_3_ADVANCED]
        
        # Convert GNN anomaly score to confidence (inverted)
        gnn_confidence = 1.0 - gnn_result.anomaly_score
        
        ensemble_confidence = (
            faiss_confidence * weights['faiss'] + 
            gnn_confidence * weights['gnn'] + 
            (1.0 - contextual_risk_score) * weights['contextual']
        )
        
        ensemble_risk_score = (
            faiss_risk_score * weights['faiss'] + 
            gnn_result.anomaly_score * weights['gnn'] +
            contextual_risk_score * weights['contextual']
        )
        
        # Apply Level 3 thresholds
        thresholds = self.policy_thresholds[PolicyLevel.LEVEL_3_ADVANCED]
        final_decision, final_risk_level = self._apply_thresholds(
            ensemble_confidence, thresholds
        )
        
        # Override decisions for critical anomalies
        if (gnn_result.decision == AuthenticationDecision.BLOCK or
            any(anomaly.value in ['fraud_signature', 'automation_detected'] 
                for anomaly in gnn_result.anomaly_types)):
            final_decision = AuthenticationDecision.BLOCK
            final_risk_level = RiskLevel.HIGH
        
        # Combine risk factors
        combined_factors = faiss_factors + [
            f"GNN anomaly score: {gnn_result.anomaly_score:.3f}",
            f"Contextual risk: {contextual_risk_score:.3f}",
            f"Anomaly types: {[a.value for a in gnn_result.anomaly_types]}"
        ]
        
        primary_reasons = []
        if gnn_result.anomaly_score > 0.7:
            primary_reasons.append(PolicyDecisionReason.GNN_ANOMALY_DETECTED)
        if contextual_risk_score > 0.6:
            primary_reasons.append(PolicyDecisionReason.CONTEXTUAL_RISK_HIGH)
        if faiss_confidence > 0.7:
            primary_reasons.append(PolicyDecisionReason.FAISS_HIGH_SIMILARITY)
        elif faiss_confidence < 0.4:
            primary_reasons.append(PolicyDecisionReason.FAISS_LOW_SIMILARITY)
        
        return PolicyDecisionResult(
            final_decision=final_decision,
            final_risk_level=final_risk_level,
            final_risk_score=ensemble_risk_score,
            confidence=ensemble_confidence,
            faiss_result={
                'decision': faiss_decision,
                'risk_level': faiss_risk_level,
                'risk_score': faiss_risk_score,
                'confidence': faiss_confidence,
                'factors': faiss_factors
            },
            gnn_result=gnn_result,
            contextual_risk=contextual_factors,
            primary_reasons=primary_reasons,
            risk_factors=combined_factors,
            layer_weights=weights,
            explanation=(
                f"Advanced analysis: FAISS {faiss_confidence:.3f}, "
                f"GNN anomaly {gnn_result.anomaly_score:.3f}, "
                f"Context {contextual_risk_score:.3f}"
            )
        )
    
    async def _execute_level_4_maximum(
        self,
        user_id: str,
        session_id: str,
        behavioral_vector: BehavioralVector,
        session_graph: SessionGraph,
        user_profile: UserProfile,
        contextual_factors: ContextualRiskFactors,
        layer_timings: Dict[str, float]
    ) -> PolicyDecisionResult:
        """Execute Level 4 Maximum Policy - Full ensemble + explainability"""
        
        # All layer analyses
        faiss_start = datetime.now()
        faiss_decision, faiss_risk_level, faiss_risk_score, faiss_confidence, faiss_factors = \
            await self.faiss_layer.make_authentication_decision(
                user_id, behavioral_vector, user_profile
            )
        layer_timings['faiss'] = (datetime.now() - faiss_start).total_seconds() * 1000
        
        gnn_start = datetime.now()
        gnn_result = self.gnn_detector.detect_anomalies(
            session_graph, user_profile
        )
        layer_timings['gnn'] = (datetime.now() - gnn_start).total_seconds() * 1000
        
        adaptive_start = datetime.now()
        adaptive_result = await self.adaptive_layer.analyze_behavioral_pattern(
            user_id, behavioral_vector, user_profile
        )
        layer_timings['adaptive'] = (datetime.now() - adaptive_start).total_seconds() * 1000
        
        contextual_start = datetime.now()
        contextual_risk_score = contextual_factors.get_overall_risk()
        layer_timings['contextual'] = (datetime.now() - contextual_start).total_seconds() * 1000
        
        # Full ensemble decision
        weights = self.layer_weights[PolicyLevel.LEVEL_4_MAXIMUM]
        
        gnn_confidence = 1.0 - gnn_result.anomaly_score
        
        ensemble_confidence = (
            faiss_confidence * weights['faiss'] + 
            gnn_confidence * weights['gnn'] + 
            adaptive_result.confidence * weights['adaptive'] +
            (1.0 - contextual_risk_score) * weights['contextual']
        )
        
        ensemble_risk_score = (
            faiss_risk_score * weights['faiss'] + 
            gnn_result.anomaly_score * weights['gnn'] +
            (1.0 - adaptive_result.confidence) * weights['adaptive'] +
            contextual_risk_score * weights['contextual']
        )
        
        # Apply Level 4 thresholds (most restrictive)
        thresholds = self.policy_thresholds[PolicyLevel.LEVEL_4_MAXIMUM]
        final_decision, final_risk_level = self._apply_thresholds(
            ensemble_confidence, thresholds
        )
        
        # Critical anomaly overrides
        if (gnn_result.decision == AuthenticationDecision.BLOCK or
            adaptive_result.drift_detected and adaptive_result.confidence < 0.3):
            final_decision = AuthenticationDecision.BLOCK
            final_risk_level = RiskLevel.HIGH
        
        # Comprehensive risk factors
        combined_factors = faiss_factors + [
            f"GNN anomaly score: {gnn_result.anomaly_score:.3f}",
            f"Adaptive confidence: {adaptive_result.confidence:.3f}",
            f"Contextual risk: {contextual_risk_score:.3f}",
            f"Pattern drift: {adaptive_result.drift_detected}",
            f"Anomaly types: {[a.value for a in gnn_result.anomaly_types]}"
        ]
        
        # Primary decision reasons
        primary_reasons = []
        if gnn_result.anomaly_score > 0.6:
            primary_reasons.append(PolicyDecisionReason.GNN_ANOMALY_DETECTED)
        if adaptive_result.drift_detected:
            primary_reasons.append(PolicyDecisionReason.ADAPTIVE_DRIFT_DETECTED)
        if contextual_risk_score > 0.5:
            primary_reasons.append(PolicyDecisionReason.CONTEXTUAL_RISK_HIGH)
        if faiss_confidence > 0.7:
            primary_reasons.append(PolicyDecisionReason.FAISS_HIGH_SIMILARITY)
        elif faiss_confidence < 0.4:
            primary_reasons.append(PolicyDecisionReason.FAISS_LOW_SIMILARITY)
        
        # Check for layer consensus/disagreement
        layer_decisions = [faiss_decision, gnn_result.decision, adaptive_result.decision]
        if len(set(layer_decisions)) == 1:
            primary_reasons.append(PolicyDecisionReason.ENSEMBLE_CONSENSUS)
        else:
            primary_reasons.append(PolicyDecisionReason.ENSEMBLE_DISAGREEMENT)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            final_decision, gnn_result, adaptive_result, contextual_factors
        )
        
        return PolicyDecisionResult(
            final_decision=final_decision,
            final_risk_level=final_risk_level,
            final_risk_score=ensemble_risk_score,
            confidence=ensemble_confidence,
            faiss_result={
                'decision': faiss_decision,
                'risk_level': faiss_risk_level,
                'risk_score': faiss_risk_score,
                'confidence': faiss_confidence,
                'factors': faiss_factors
            },
            gnn_result=gnn_result,
            adaptive_result=adaptive_result.__dict__,
            contextual_risk=contextual_factors,
            primary_reasons=primary_reasons,
            risk_factors=combined_factors,
            layer_weights=weights,
            explanation=(
                f"Maximum security analysis: FAISS {faiss_confidence:.3f}, "
                f"GNN {gnn_result.anomaly_score:.3f}, "
                f"Adaptive {adaptive_result.confidence:.3f}, "
                f"Context {contextual_risk_score:.3f}"
            ),
            recommendations=recommendations
        )
    
    def _apply_thresholds(
        self, 
        confidence: float, 
        thresholds: Dict[str, float]
    ) -> Tuple[AuthenticationDecision, RiskLevel]:
        """Apply policy thresholds to determine final decision"""
        
        if confidence >= thresholds['allow_threshold']:
            return AuthenticationDecision.ALLOW, RiskLevel.LOW
        elif confidence >= thresholds['challenge_threshold']:
            return AuthenticationDecision.CHALLENGE, RiskLevel.MEDIUM
        else:
            return AuthenticationDecision.BLOCK, RiskLevel.HIGH
    
    def _generate_recommendations(
        self,
        decision: AuthenticationDecision,
        gnn_result: GNNAnomalyResult,
        adaptive_result: Any,
        contextual_factors: ContextualRiskFactors
    ) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        
        recommendations = []
        
        if decision == AuthenticationDecision.BLOCK:
            recommendations.append("Immediate security review required")
            if gnn_result.anomaly_score > 0.8:
                recommendations.append("Potential fraud pattern detected - escalate to security team")
            if contextual_factors.recent_failures > 2:
                recommendations.append("Consider temporary account suspension")
        
        elif decision == AuthenticationDecision.CHALLENGE:
            recommendations.append("Require additional authentication (MPIN/biometric)")
            if contextual_factors.is_new_beneficiary:
                recommendations.append("Verify beneficiary details before proceeding")
            if adaptive_result.drift_detected:
                recommendations.append("Monitor for continued behavioral changes")
        
        else:  # ALLOW
            if contextual_factors.transaction_amount > 25000:
                recommendations.append("Consider additional verification for high-value transaction")
            recommendations.append("Continue behavioral monitoring")
        
        return recommendations
    
    def _initialize_contextual_policies(self) -> Dict[str, Dict[str, Any]]:
        """Initialize contextual risk policies"""
        return {
            'high_value_transaction': {
                'threshold': 50000,
                'risk_multiplier': 1.5,
                'required_policy_level': PolicyLevel.LEVEL_4_MAXIMUM
            },
            'new_beneficiary': {
                'risk_addition': 0.3,
                'challenge_required': True
            },
            'unusual_time': {
                'night_hours_risk': 0.2,  # 11PM - 6AM
                'weekend_risk': 0.1
            },
            'rapid_transactions': {
                'max_per_hour': 5,
                'risk_per_excess': 0.2
            },
            'multiple_failures': {
                'max_allowed': 3,
                'risk_per_failure': 0.25
            }
        }
    
    def _update_performance_metrics(self, result: PolicyDecisionResult):
        """Update engine performance metrics"""
        self.performance_metrics['total_decisions'] += 1
        self.performance_metrics['level_usage'][result.policy_level_used] += 1
        self.performance_metrics['decision_distribution'][result.final_decision] += 1
        
        # Update average processing time
        total_time = self.performance_metrics['avg_processing_time'] * (self.performance_metrics['total_decisions'] - 1)
        total_time += result.processing_time_ms
        self.performance_metrics['avg_processing_time'] = total_time / self.performance_metrics['total_decisions']
    
    def _create_fallback_decision(self, error_reason: str) -> PolicyDecisionResult:
        """Create safe fallback decision for error cases"""
        return PolicyDecisionResult(
            final_decision=AuthenticationDecision.CHALLENGE,
            final_risk_level=RiskLevel.MEDIUM,
            final_risk_score=0.7,
            confidence=0.3,
            policy_level_used=PolicyLevel.LEVEL_1_BASIC,
            primary_reasons=[PolicyDecisionReason.POLICY_OVERRIDE],
            risk_factors=[f"Policy engine error: {error_reason}"],
            explanation="Fallback decision due to policy engine error"
        )
    
    def get_policy_statistics(self) -> Dict[str, Any]:
        """Get comprehensive policy engine statistics"""
        return {
            'performance_metrics': self.performance_metrics,
            'policy_thresholds': self.policy_thresholds,
            'layer_weights': self.layer_weights,
            'recent_decisions': len(self.decision_history),
            'contextual_policies': len(self.contextual_policies)
        }
    
    def adjust_policy_thresholds(
        self, 
        policy_level: PolicyLevel, 
        new_thresholds: Dict[str, float]
    ):
        """Dynamically adjust policy thresholds"""
        self.policy_thresholds[policy_level].update(new_thresholds)
        self.logger.info(f"Updated thresholds for {policy_level.value}: {new_thresholds}")
    
    def get_decision_history(self, limit: int = 100) -> List[PolicyDecisionResult]:
        """Get recent decision history for analysis"""
        return self.decision_history[-limit:]

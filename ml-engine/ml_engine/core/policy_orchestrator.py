"""
Policy Orchestration Engine
Risk-based session control and explainability logging for BRIDGE ML-Engine
"""

import numpy as np
import torch
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import json
from collections import defaultdict, deque

from ml_engine.config import CONFIG
from ml_engine.utils.behavioral_vectors import BehavioralVector
from faiss.verifier.layer1_verifier import VerificationResult as L1Result
from level2_adapter.layer2_verifier import Layer2Result
from drift_detection import DriftDetectionResult

logger = logging.getLogger(__name__)

class SessionAction(Enum):
    """Possible session actions"""
    CONTINUE = "continue"
    STEP_UP_AUTH = "step_up_auth"
    CHALLENGE = "challenge"
    TEMPORARY_BLOCK = "temporary_block"
    PERMANENT_BLOCK = "permanent_block"
    LOGOUT = "logout"
    MONITOR = "monitor"

class RiskLevel(Enum):
    """Risk levels for session control"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class PolicyDecision:
    """Comprehensive policy decision"""
    session_id: str
    user_id: str
    action: SessionAction
    risk_level: RiskLevel
    risk_score: float
    confidence: float
    reasoning: List[str]
    evidence: Dict[str, Any]
    timestamp: datetime
    expires_at: Optional[datetime]
    monitoring_level: str
    next_verification_delay: int  # seconds
    
@dataclass
class ExplainabilityReport:
    """Detailed explanation of authentication decision"""
    decision_id: str
    session_id: str
    user_id: str
    timestamp: datetime
    
    # Risk factors breakdown
    l1_similarity_score: float
    l1_confidence: str
    l2_transformer_score: float
    l2_gnn_anomaly_score: float
    drift_score: float
    context_score: float
    
    # Feature importance
    feature_contributions: Dict[str, float]
    top_risk_factors: List[Tuple[str, float]]
    protective_factors: List[Tuple[str, float]]
    
    # Model decisions
    faiss_decision: str
    transformer_decision: str
    gnn_decision: str
    drift_decision: str
    
    # Final reasoning
    final_decision: str
    decision_confidence: float
    human_readable_explanation: str

class SessionRiskTracker:
    """Tracks risk evolution for sessions"""
    
    def __init__(self, max_history_size: int = 100):
        self.sessions: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history_size))
        self.risk_trends: Dict[str, List[float]] = defaultdict(list)
        
    def add_risk_point(self, session_id: str, risk_score: float, timestamp: datetime):
        """Add risk score point for session"""
        self.sessions[session_id].append((risk_score, timestamp))
        self.risk_trends[session_id].append(risk_score)
        
    def get_risk_trend(self, session_id: str) -> str:
        """Get risk trend: increasing, decreasing, stable"""
        if session_id not in self.risk_trends or len(self.risk_trends[session_id]) < 3:
            return "insufficient_data"
            
        recent_scores = self.risk_trends[session_id][-5:]
        if len(recent_scores) < 3:
            return "insufficient_data"
            
        # Calculate trend
        x = np.arange(len(recent_scores))
        slope = np.polyfit(x, recent_scores, 1)[0]
        
        if slope > 0.05:
            return "increasing"
        elif slope < -0.05:
            return "decreasing"
        else:
            return "stable"
    
    def get_session_volatility(self, session_id: str) -> float:
        """Calculate risk volatility for session"""
        if session_id not in self.risk_trends or len(self.risk_trends[session_id]) < 3:
            return 0.0
            
        scores = np.array(self.risk_trends[session_id])
        return float(np.std(scores))

class PolicyOrchestrator:
    """Main policy orchestration engine"""
    
    def __init__(self):
        self.risk_tracker = SessionRiskTracker()
        self.policy_cache: Dict[str, PolicyDecision] = {}
        self.user_trust_scores: Dict[str, float] = defaultdict(lambda: 0.5)
        self.session_contexts: Dict[str, Dict[str, Any]] = {}
        
        # Policy rules configuration
        self.risk_thresholds = {
            RiskLevel.VERY_LOW: 0.1,
            RiskLevel.LOW: 0.3,
            RiskLevel.MEDIUM: 0.5,
            RiskLevel.HIGH: 0.7,
            RiskLevel.CRITICAL: 0.9
        }
        
        # Action mappings based on risk level
        self.risk_actions = {
            RiskLevel.VERY_LOW: SessionAction.CONTINUE,
            RiskLevel.LOW: SessionAction.CONTINUE,
            RiskLevel.MEDIUM: SessionAction.MONITOR,
            RiskLevel.HIGH: SessionAction.STEP_UP_AUTH,
            RiskLevel.CRITICAL: SessionAction.TEMPORARY_BLOCK
        }
        
        # Contextual modifiers
        self.context_modifiers = {
            "time_sensitive_operation": 1.2,
            "high_value_transaction": 1.5,
            "new_device": 1.3,
            "unusual_location": 1.4,
            "suspicious_network": 1.6,
            "off_hours": 1.1,
            "known_device": 0.8,
            "home_network": 0.7,
            "regular_usage_pattern": 0.9
        }

    def calculate_composite_risk_score(
        self,
        l1_result: L1Result,
        l2_result: Layer2Result,
        drift_result: DriftDetectionResult,
        context: Dict[str, Any]
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate composite risk score from all sources"""
        
        # Base scores from each layer
        scores = {
            "faiss_similarity": 1.0 - l1_result.similarity_score,  # Higher similarity = lower risk
            "transformer_confidence": 1.0 - l2_result.transformer_confidence,
            "gnn_anomaly": l2_result.gnn_anomaly_score,
            "drift_score": drift_result.drift_magnitude if drift_result.drift_detected else 0.0
        }
        
        # Context-based risk
        context_risk = self._calculate_context_risk(context)
        scores["context_score"] = context_risk
        
        # User trust adjustment
        user_id = l1_result.user_id
        trust_modifier = 1.0 - self.user_trust_scores[user_id]
        
        # Weighted composite score
        weighted_score = sum(
            scores[key] * CONFIG.RISK_WEIGHTS[key] 
            for key in scores.keys()
        )
        
        # Apply trust modifier
        final_score = min(1.0, weighted_score * trust_modifier)
        
        return final_score, scores

    def _calculate_context_risk(self, context: Dict[str, Any]) -> float:
        """Calculate risk based on contextual factors"""
        base_risk = 0.0
        
        # Time-based risk
        current_hour = datetime.now().hour
        if current_hour < 6 or current_hour > 22:
            base_risk += 0.1  # Off hours risk
            
        # Device and location risk
        if context.get("new_device", False):
            base_risk += 0.2
        if context.get("unusual_location", False):
            base_risk += 0.3
        if context.get("suspicious_network", False):
            base_risk += 0.4
            
        # Transaction context
        if context.get("high_value_transaction", False):
            base_risk += 0.1
        if context.get("time_sensitive_operation", False):
            base_risk += 0.1
            
        # Protective factors
        if context.get("known_device", False):
            base_risk -= 0.1
        if context.get("home_network", False):
            base_risk -= 0.1
        if context.get("regular_pattern", False):
            base_risk -= 0.1
            
        return max(0.0, min(1.0, base_risk))

    def determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level from score"""
        for level in [RiskLevel.CRITICAL, RiskLevel.HIGH, RiskLevel.MEDIUM, RiskLevel.LOW]:
            if risk_score >= self.risk_thresholds[level]:
                return level
        return RiskLevel.VERY_LOW

    def make_policy_decision(
        self,
        session_id: str,
        l1_result: L1Result,
        l2_result: Layer2Result,
        drift_result: DriftDetectionResult,
        context: Dict[str, Any]
    ) -> PolicyDecision:
        """Make comprehensive policy decision"""
        
        # Calculate composite risk
        risk_score, component_scores = self.calculate_composite_risk_score(
            l1_result, l2_result, drift_result, context
        )
        
        # Track risk evolution
        self.risk_tracker.add_risk_point(session_id, risk_score, datetime.now())
        
        # Determine risk level
        risk_level = self.determine_risk_level(risk_score)
        
        # Get base action
        action = self.risk_actions[risk_level]
        
        # Apply contextual modifications
        action, reasoning = self._apply_contextual_rules(
            action, risk_score, risk_level, session_id, context, component_scores
        )
        
        # Calculate confidence
        confidence = self._calculate_decision_confidence(component_scores, context)
        
        # Determine monitoring and verification settings
        monitoring_level, next_delay = self._determine_monitoring_strategy(
            risk_level, action, session_id
        )
        
        # Create decision
        decision = PolicyDecision(
            session_id=session_id,
            user_id=l1_result.user_id,
            action=action,
            risk_level=risk_level,
            risk_score=risk_score,
            confidence=confidence,
            reasoning=reasoning,
            evidence={
                "l1_result": asdict(l1_result),
                "l2_result": asdict(l2_result),
                "drift_result": asdict(drift_result),
                "component_scores": component_scores,
                "context": context
            },
            timestamp=datetime.now(),
            expires_at=datetime.now() + timedelta(minutes=30),
            monitoring_level=monitoring_level,
            next_verification_delay=next_delay
        )
        
        # Cache decision
        self.policy_cache[session_id] = decision
        
        # Update user trust
        self._update_user_trust(l1_result.user_id, risk_score, action)
        
        return decision

    def _apply_contextual_rules(
        self,
        base_action: SessionAction,
        risk_score: float,
        risk_level: RiskLevel,
        session_id: str,
        context: Dict[str, Any],
        component_scores: Dict[str, float]
    ) -> Tuple[SessionAction, List[str]]:
        """Apply contextual business rules"""
        
        action = base_action
        reasoning = [f"Base action: {base_action.value} for risk level: {risk_level.value}"]
        
        # High-value transaction protection
        if context.get("high_value_transaction", False) and risk_score > 0.3:
            if action == SessionAction.CONTINUE:
                action = SessionAction.STEP_UP_AUTH
                reasoning.append("Elevated to step-up auth for high-value transaction")
        
        # Fraud indicators
        if (component_scores.get("gnn_anomaly", 0) > 0.7 or 
            component_scores.get("drift_score", 0) > 0.5):
            if action in [SessionAction.CONTINUE, SessionAction.MONITOR]:
                action = SessionAction.CHALLENGE
                reasoning.append("Challenge required due to anomaly indicators")
        
        # Risk trend analysis
        trend = self.risk_tracker.get_risk_trend(session_id)
        if trend == "increasing" and risk_score > 0.5:
            if action == SessionAction.CONTINUE:
                action = SessionAction.MONITOR
                reasoning.append("Monitoring due to increasing risk trend")
        
        # Time-sensitive operations
        if context.get("time_sensitive_operation", False):
            if action == SessionAction.STEP_UP_AUTH and risk_score < 0.8:
                action = SessionAction.CHALLENGE  # Faster than full step-up
                reasoning.append("Quick challenge for time-sensitive operation")
        
        # Multiple risk factors
        high_risk_factors = sum(1 for score in component_scores.values() if score > 0.6)
        if high_risk_factors >= 3:
            if action in [SessionAction.CONTINUE, SessionAction.MONITOR]:
                action = SessionAction.STEP_UP_AUTH
                reasoning.append("Step-up auth due to multiple risk factors")
        
        # Session volatility check
        volatility = self.risk_tracker.get_session_volatility(session_id)
        if volatility > 0.3:  # High volatility
            if action == SessionAction.CONTINUE:
                action = SessionAction.MONITOR
                reasoning.append("Monitoring due to high risk volatility")
        
        return action, reasoning

    def _calculate_decision_confidence(
        self, 
        component_scores: Dict[str, float], 
        context: Dict[str, Any]
    ) -> float:
        """Calculate confidence in the decision"""
        
        # Base confidence from score consistency
        scores = list(component_scores.values())
        score_std = np.std(scores)
        consistency_confidence = 1.0 - min(1.0, score_std * 2)
        
        # Context confidence
        context_confidence = 0.8  # Base
        if context.get("known_device", False):
            context_confidence += 0.1
        if context.get("regular_pattern", False):
            context_confidence += 0.1
        if context.get("suspicious_indicators", 0) > 2:
            context_confidence -= 0.2
            
        # Combined confidence
        return min(1.0, (consistency_confidence + context_confidence) / 2)

    def _determine_monitoring_strategy(
        self, 
        risk_level: RiskLevel, 
        action: SessionAction, 
        session_id: str
    ) -> Tuple[str, int]:
        """Determine monitoring level and next verification delay"""
        
        if risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH]:
            monitoring_level = "intensive"
            next_delay = 30  # 30 seconds
        elif risk_level == RiskLevel.MEDIUM:
            monitoring_level = "moderate"
            next_delay = 60  # 1 minute
        else:
            monitoring_level = "standard"
            next_delay = 300  # 5 minutes
        
        # Adjust based on action
        if action in [SessionAction.TEMPORARY_BLOCK, SessionAction.PERMANENT_BLOCK]:
            monitoring_level = "blocked"
            next_delay = 0
        elif action == SessionAction.STEP_UP_AUTH:
            next_delay = min(30, next_delay)  # More frequent checks
            
        return monitoring_level, next_delay

    def _update_user_trust(self, user_id: str, risk_score: float, action: SessionAction):
        """Update user trust score based on behavior"""
        current_trust = self.user_trust_scores[user_id]
        
        # Trust adjustment based on risk and action
        if risk_score < 0.3 and action == SessionAction.CONTINUE:
            # Good behavior, increase trust slightly
            adjustment = 0.01
        elif risk_score > 0.7:
            # Risky behavior, decrease trust
            adjustment = -0.05
        elif action in [SessionAction.TEMPORARY_BLOCK, SessionAction.PERMANENT_BLOCK]:
            # Blocked action, significant trust decrease
            adjustment = -0.1
        else:
            # Neutral
            adjustment = 0.0
            
        # Apply adjustment with bounds
        self.user_trust_scores[user_id] = max(0.0, min(1.0, current_trust + adjustment))

    def generate_explainability_report(
        self,
        decision: PolicyDecision,
        l1_result: L1Result,
        l2_result: Layer2Result,
        drift_result: DriftDetectionResult
    ) -> ExplainabilityReport:
        """Generate detailed explainability report"""
        
        component_scores = decision.evidence["component_scores"]
        
        # Feature importance analysis
        feature_contributions = {
            "FAISS Similarity": component_scores["faiss_similarity"],
            "Transformer Confidence": component_scores["transformer_confidence"],
            "GNN Anomaly Detection": component_scores["gnn_anomaly"],
            "Behavioral Drift": component_scores["drift_score"],
            "Contextual Factors": component_scores["context_score"]
        }
        
        # Sort by impact
        sorted_features = sorted(
            feature_contributions.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        top_risk_factors = [(name, score) for name, score in sorted_features if score > 0.3]
        protective_factors = [(name, 1-score) for name, score in sorted_features if score < 0.2]
        
        # Generate human-readable explanation
        explanation = self._generate_human_explanation(decision, component_scores)
        
        return ExplainabilityReport(
            decision_id=f"{decision.session_id}_{decision.timestamp.isoformat()}",
            session_id=decision.session_id,
            user_id=decision.user_id,
            timestamp=decision.timestamp,
            l1_similarity_score=l1_result.similarity_score,
            l1_confidence=l1_result.confidence_level,
            l2_transformer_score=l2_result.transformer_confidence,
            l2_gnn_anomaly_score=l2_result.gnn_anomaly_score,
            drift_score=drift_result.drift_magnitude,
            context_score=component_scores["context_score"],
            feature_contributions=feature_contributions,
            top_risk_factors=top_risk_factors,
            protective_factors=protective_factors,
            faiss_decision=l1_result.decision,
            transformer_decision=l2_result.transformer_decision,
            gnn_decision=l2_result.gnn_decision,
            drift_decision="drift_detected" if drift_result.drift_detected else "stable",
            final_decision=decision.action.value,
            decision_confidence=decision.confidence,
            human_readable_explanation=explanation
        )

    def _generate_human_explanation(
        self, 
        decision: PolicyDecision, 
        component_scores: Dict[str, float]
    ) -> str:
        """Generate human-readable explanation"""
        
        risk_level = decision.risk_level.value
        action = decision.action.value
        
        explanation = f"Authentication decision: {action.upper()} (Risk Level: {risk_level.upper()})\n\n"
        
        # Risk factors
        high_factors = [name for name, score in component_scores.items() if score > 0.5]
        if high_factors:
            explanation += f"Primary risk factors: {', '.join(high_factors)}\n"
        
        # Reasoning
        explanation += "Reasoning:\n"
        for reason in decision.reasoning:
            explanation += f"• {reason}\n"
        
        # Confidence
        confidence_text = "high" if decision.confidence > 0.8 else "medium" if decision.confidence > 0.5 else "low"
        explanation += f"\nDecision confidence: {confidence_text} ({decision.confidence:.2f})"
        
        return explanation

    def get_session_policy_history(self, session_id: str) -> List[PolicyDecision]:
        """Get policy decision history for session"""
        # In production, this would query a database
        return [decision for decision in self.policy_cache.values() 
                if decision.session_id == session_id]

    def export_audit_log(self, user_id: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Export audit log for compliance"""
        decisions = [
            decision for decision in self.policy_cache.values()
            if (decision.user_id == user_id and 
                start_date <= decision.timestamp <= end_date)
        ]
        
        return {
            "user_id": user_id,
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "total_decisions": len(decisions),
            "decisions": [asdict(decision) for decision in decisions],
            "risk_summary": {
                "avg_risk_score": np.mean([d.risk_score for d in decisions]) if decisions else 0,
                "max_risk_score": max([d.risk_score for d in decisions]) if decisions else 0,
                "actions_taken": {action.value: sum(1 for d in decisions if d.action == action) 
                                 for action in SessionAction}
            }
        }

# Global policy orchestrator instance
policy_orchestrator = PolicyOrchestrator()

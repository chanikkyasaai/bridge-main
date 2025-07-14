"""
BRIDGE Industry-Grade ML-Engine Core
Banking-focused behavioral authentication with complete session lifecycle management
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor

from mlengine.config import CONFIG

logger = logging.getLogger(__name__)

class AuthenticationDecision(Enum):
    """Final authentication decisions"""
    ALLOW = "allow"                    # Continue session normally
    MONITOR = "monitor"                # Increased monitoring, no user impact
    CHALLENGE = "challenge"            # Soft challenge (e.g., additional question)
    STEP_UP_AUTH = "step_up_auth"     # Require biometric/PIN verification
    TEMPORARY_BLOCK = "temporary_block" # Block for X minutes
    PERMANENT_BLOCK = "permanent_block" # End session permanently
    REQUIRE_REAUTH = "require_reauth"  # Force complete re-authentication

class RiskLevel(Enum):
    """Risk assessment levels"""
    VERY_LOW = 0    # 0.0 - 0.2
    LOW = 1         # 0.2 - 0.4
    MEDIUM = 2      # 0.4 - 0.6
    HIGH = 3        # 0.6 - 0.8
    CRITICAL = 4    # 0.8 - 1.0

class ProcessingStage(Enum):
    """ML processing pipeline stages"""
    INPUT_VALIDATION = "input_validation"
    PREPROCESSING = "preprocessing"
    LAYER1_FAISS = "layer1_faiss"
    LAYER2_ADAPTIVE = "layer2_adaptive"
    DRIFT_DETECTION = "drift_detection"
    RISK_AGGREGATION = "risk_aggregation"
    POLICY_DECISION = "policy_decision"
    OUTPUT_GENERATION = "output_generation"

@dataclass
class BehavioralEvent:
    """Single behavioral event from client"""
    timestamp: datetime
    event_type: str
    features: Dict[str, float]
    session_id: str
    user_id: str
    device_id: str
    raw_metadata: Dict[str, Any]

@dataclass
class BehavioralVector:
    """Processed behavioral vector"""
    vector: np.ndarray
    confidence: float
    timestamp: datetime
    session_id: str
    user_id: str
    source_events: List[BehavioralEvent]
    processing_metadata: Dict[str, Any]

@dataclass
class SessionContext:
    """Complete session context for ML processing"""
    session_id: str
    user_id: str
    device_id: str
    session_start_time: datetime
    last_activity: datetime
    session_duration_minutes: float
    
    # Device & Environment
    device_type: str
    device_model: str
    os_version: str
    app_version: str
    network_type: str
    location_data: Optional[Dict[str, Any]]
    
    # Behavioral Context
    time_of_day: str
    usage_pattern: str
    interaction_frequency: float
    typical_session_duration: float
    
    # Security Context
    is_known_device: bool
    is_trusted_location: bool
    recent_security_events: List[Dict[str, Any]]
    current_risk_level: RiskLevel
    
    # Banking Context
    account_age_days: int
    transaction_history_risk: float
    current_transaction_context: Optional[Dict[str, Any]]

@dataclass
class Layer1Result:
    """Layer 1 FAISS verification result"""
    user_id: str
    session_id: str
    similarity_score: float
    confidence_level: str
    matched_profiles: List[Dict[str, Any]]
    decision: str
    processing_time_ms: float
    threshold_analysis: Dict[str, float]
    metadata: Dict[str, Any]

@dataclass
class Layer2Result:
    """Layer 2 adaptive verification result"""
    session_id: str
    user_id: str
    transformer_confidence: float
    gnn_anomaly_score: float
    contextual_adaptation_score: float
    session_graph_analysis: Dict[str, Any]
    temporal_pattern_score: float
    decision_confidence: float
    processing_time_ms: float
    detailed_analysis: Dict[str, Any]

@dataclass
class DriftAnalysisResult:
    """Behavioral drift analysis result"""
    user_id: str
    drift_detected: bool
    drift_type: str
    drift_magnitude: float
    drift_confidence: float
    affected_behavioral_features: List[str]
    adaptation_recommendation: str
    stability_score: float
    temporal_analysis: Dict[str, Any]

@dataclass
class RiskAssessmentResult:
    """Comprehensive risk assessment"""
    session_id: str
    user_id: str
    overall_risk_score: float
    risk_level: RiskLevel
    risk_components: Dict[str, float]
    confidence: float
    contributing_factors: List[Tuple[str, float]]
    mitigating_factors: List[Tuple[str, float]]
    recommendation: AuthenticationDecision
    explanation: str

@dataclass
class MLEngineResponse:
    """Complete ML engine response"""
    session_id: str
    user_id: str
    decision: AuthenticationDecision
    risk_assessment: RiskAssessmentResult
    layer1_result: Layer1Result
    layer2_result: Optional[Layer2Result]
    drift_result: Optional[DriftAnalysisResult]
    
    # Processing metrics
    total_processing_time_ms: float
    processing_stages: Dict[ProcessingStage, float]
    
    # Session management
    next_verification_interval_seconds: int
    monitoring_level: str
    session_actions: List[str]
    
    # Explainability
    explanation: Dict[str, Any]
    confidence: float
    timestamp: datetime

class IndustryGradeMLEngine:
    """
    Industry-grade ML Engine for Banking Behavioral Authentication
    
    This is the core engine that processes behavioral data through multiple
    layers and provides banking-compliant authentication decisions.
    """
    
    def __init__(self):
        # Initialize all components
        self.preprocessing_pipeline = None
        self.layer1_faiss_engine = None
        self.layer2_adaptive_engine = None
        self.drift_analysis_engine = None
        self.risk_assessment_engine = None
        self.policy_decision_engine = None
        
        # Session management
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.session_lock = threading.RLock()
        
        # Performance monitoring
        self.processing_metrics = defaultdict(list)
        self.performance_stats = {
            "total_requests": 0,
            "successful_authentications": 0,
            "blocked_sessions": 0,
            "average_processing_time": 0.0,
            "layer1_accuracy": 0.0,
            "layer2_accuracy": 0.0
        }
        
        # Thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=CONFIG.MAX_CONCURRENT_SESSIONS // 5)
        
        # Component status
        self.is_initialized = False
        
        logger.info("Industry-grade ML Engine initialized")

    async def initialize(self):
        """Initialize all ML engine components"""
        logger.info("🏦 Initializing Industry-Grade Banking ML Engine...")
        
        try:
            # Initialize components in order
            logger.info("📊 Initializing preprocessing pipeline...")
            await self._initialize_preprocessing()
            
            logger.info("🔍 Initializing Layer 1 FAISS Engine...")
            await self._initialize_layer1_faiss()
            
            logger.info("🧠 Initializing Layer 2 Adaptive Engine...")
            await self._initialize_layer2_adaptive()
            
            logger.info("📈 Initializing Drift Analysis Engine...")
            await self._initialize_drift_analysis()
            
            logger.info("⚖️ Initializing Risk Assessment Engine...")
            await self._initialize_risk_assessment()
            
            logger.info("🛡️ Initializing Policy Decision Engine...")
            await self._initialize_policy_engine()
            
            self.is_initialized = True
            logger.info("✅ Industry-Grade Banking ML Engine fully initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize ML Engine: {e}")
            raise

    async def _initialize_preprocessing(self):
        """Initialize preprocessing pipeline"""
        from mlengine.core.behavioral_preprocessing import BehavioralPreprocessingPipeline
        self.preprocessing_pipeline = BehavioralPreprocessingPipeline()
        await self.preprocessing_pipeline.initialize()

    async def _initialize_layer1_faiss(self):
        """Initialize Layer 1 FAISS engine"""
        from mlengine.adapters.faiss.verifier.layer1_verifier import FAISSVerifier
        self.layer1_faiss_engine = FAISSVerifier()
        await self.layer1_faiss_engine.initialize()

    async def _initialize_layer2_adaptive(self):
        """Initialize Layer 2 adaptive engine"""
        from mlengine.adapters.level2.layer2_verifier import Layer2Verifier
        self.layer2_adaptive_engine = Layer2Verifier()
        await self.layer2_adaptive_engine.initialize()

    async def _initialize_drift_analysis(self):
        """Initialize drift analysis engine"""
        from mlengine.core.drift_detection import BehavioralDriftDetector
        self.drift_analysis_engine = BehavioralDriftDetector()
        await self.drift_analysis_engine.initialize()

    async def _initialize_risk_assessment(self):
        """Initialize risk assessment engine"""
        from mlengine.adapters.level2.layer2_verifier import Layer2Verifier
        # Use Layer2Verifier for risk assessment as it contains risk scoring logic
        self.risk_assessment_engine = Layer2Verifier()
        await self.risk_assessment_engine.initialize()

    async def _initialize_policy_engine(self):
        """Initialize policy decision engine"""
        from mlengine.core.policy_orchestrator import PolicyOrchestrator
        self.policy_decision_engine = PolicyOrchestrator()
        await self.policy_decision_engine.initialize()

    async def process_session_events(
        self,
        events: List[BehavioralEvent],
        session_context: SessionContext,
        require_explanation: bool = False
    ) -> MLEngineResponse:
        """
        Main processing pipeline for session behavioral events
        
        This is the industry-standard processing flow:
        1. Input validation and preprocessing
        2. Layer 1: Fast FAISS-based similarity matching
        3. Layer 2: Adaptive context-aware analysis (if needed)
        4. Drift detection and adaptation
        5. Risk assessment and aggregation
        6. Policy decision making
        7. Response generation with explainability
        """
        
        if not self.is_initialized:
            raise RuntimeError("ML Engine not initialized")
        
        start_time = time.time()
        processing_stages = {}
        
        try:
            # Track session
            self._update_session_tracking(session_context)
            
            # Stage 1: Input Validation and Preprocessing
            stage_start = time.time()
            validated_events, vectors = await self._stage1_preprocessing(events, session_context)
            processing_stages[ProcessingStage.PREPROCESSING] = (time.time() - stage_start) * 1000
            
            # Stage 2: Layer 1 FAISS Processing
            stage_start = time.time()
            layer1_result = await self._stage2_layer1_faiss(vectors, session_context)
            processing_stages[ProcessingStage.LAYER1_FAISS] = (time.time() - stage_start) * 1000
            
            # Stage 3: Layer 2 Adaptive Processing (conditional)
            layer2_result = None
            if self._requires_layer2_processing(layer1_result, session_context):
                stage_start = time.time()
                layer2_result = await self._stage3_layer2_adaptive(vectors, validated_events, session_context)
                processing_stages[ProcessingStage.LAYER2_ADAPTIVE] = (time.time() - stage_start) * 1000
            
            # Stage 4: Drift Detection
            stage_start = time.time()
            drift_result = await self._stage4_drift_detection(vectors, session_context)
            processing_stages[ProcessingStage.DRIFT_DETECTION] = (time.time() - stage_start) * 1000
            
            # Stage 5: Risk Assessment
            stage_start = time.time()
            risk_assessment = await self._stage5_risk_assessment(
                layer1_result, layer2_result, drift_result, session_context
            )
            processing_stages[ProcessingStage.RISK_AGGREGATION] = (time.time() - stage_start) * 1000
            
            # Stage 6: Policy Decision
            stage_start = time.time()
            decision = await self._stage6_policy_decision(risk_assessment, session_context)
            processing_stages[ProcessingStage.POLICY_DECISION] = (time.time() - stage_start) * 1000
            
            # Stage 7: Response Generation
            stage_start = time.time()
            response = await self._stage7_generate_response(
                layer1_result, layer2_result, drift_result, risk_assessment, 
                decision, session_context, processing_stages, start_time, require_explanation
            )
            processing_stages[ProcessingStage.OUTPUT_GENERATION] = (time.time() - stage_start) * 1000
            
            # Update performance metrics
            self._update_performance_metrics(response, time.time() - start_time)
            
            logger.info(
                f"🏦 Banking ML Engine processed session {session_context.session_id}: "
                f"{response.decision.value} (risk: {response.risk_assessment.overall_risk_score:.3f}, "
                f"time: {response.total_processing_time_ms:.1f}ms)"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"❌ ML Engine processing error for session {session_context.session_id}: {e}")
            return await self._create_error_response(session_context, str(e), time.time() - start_time)

    async def _stage1_preprocessing(
        self, 
        events: List[BehavioralEvent], 
        context: SessionContext
    ) -> Tuple[List[BehavioralEvent], List[BehavioralVector]]:
        """Stage 1: Input validation and preprocessing"""
        
        # Validate input events
        validated_events = await self.preprocessing_pipeline.validate_events(events)
        
        # Extract and engineer features
        feature_vectors = await self.preprocessing_pipeline.extract_features(validated_events, context)
        
        # Convert to behavioral vectors
        vectors = await self.preprocessing_pipeline.create_behavioral_vectors(
            feature_vectors, validated_events, context
        )
        
        logger.debug(f"Preprocessing: {len(events)} events → {len(vectors)} vectors")
        return validated_events, vectors

    async def _stage2_layer1_faiss(
        self, 
        vectors: List[BehavioralVector], 
        context: SessionContext
    ) -> Layer1Result:
        """Stage 2: Layer 1 FAISS-based fast verification"""
        
        result = await self.layer1_faiss_engine.verify_vectors(vectors, context)
        
        logger.debug(
            f"Layer 1 FAISS: similarity={result.similarity_score:.3f}, "
            f"confidence={result.confidence_level}, decision={result.decision}"
        )
        
        return result

    def _requires_layer2_processing(self, layer1_result: Layer1Result, context: SessionContext) -> bool:
        """Determine if Layer 2 processing is required"""
        
        # Banking-specific Layer 2 triggers
        banking_triggers = [
            # Low/medium confidence from Layer 1
            layer1_result.confidence_level in ['medium', 'low'],
            
            # High-value transaction context
            context.current_transaction_context and 
            context.current_transaction_context.get('amount', 0) > 50000,
            
            # New or untrusted device
            not context.is_known_device,
            
            # Unusual time or location
            not context.is_trusted_location,
            
            # Recent security events
            len(context.recent_security_events) > 0,
            
            # Similarity score below banking threshold
            layer1_result.similarity_score < CONFIG.BANKING_L2_TRIGGER_THRESHOLD
        ]
        
        should_trigger = any(banking_triggers)
        
        if should_trigger:
            logger.debug(f"Layer 2 triggered for session {context.session_id}")
        
        return should_trigger

    async def _stage3_layer2_adaptive(
        self, 
        vectors: List[BehavioralVector],
        events: List[BehavioralEvent],
        context: SessionContext
    ) -> Layer2Result:
        """Stage 3: Layer 2 adaptive context-aware analysis"""
        
        result = await self.layer2_adaptive_engine.analyze_adaptive(vectors, events, context)
        
        logger.debug(
            f"Layer 2 Adaptive: transformer={result.transformer_confidence:.3f}, "
            f"gnn_anomaly={result.gnn_anomaly_score:.3f}, "
            f"contextual={result.contextual_adaptation_score:.3f}"
        )
        
        return result

    async def _stage4_drift_detection(
        self, 
        vectors: List[BehavioralVector], 
        context: SessionContext
    ) -> DriftAnalysisResult:
        """Stage 4: Behavioral drift detection and adaptation"""
        
        result = await self.drift_analysis_engine.analyze_drift(vectors, context)
        
        if result.drift_detected:
            logger.info(
                f"Behavioral drift detected for user {context.user_id}: "
                f"{result.drift_type} (magnitude: {result.drift_magnitude:.3f})"
            )
        
        return result

    async def _stage5_risk_assessment(
        self,
        layer1_result: Layer1Result,
        layer2_result: Optional[Layer2Result],
        drift_result: DriftAnalysisResult,
        context: SessionContext
    ) -> RiskAssessmentResult:
        """Stage 5: Comprehensive banking risk assessment"""
        
        result = await self.risk_assessment_engine.assess_risk(
            layer1_result, layer2_result, drift_result, context
        )
        
        logger.debug(
            f"Risk Assessment: overall={result.overall_risk_score:.3f}, "
            f"level={result.risk_level.name}, confidence={result.confidence:.3f}"
        )
        
        return result

    async def _stage6_policy_decision(
        self, 
        risk_assessment: RiskAssessmentResult, 
        context: SessionContext
    ) -> AuthenticationDecision:
        """Stage 6: Banking policy decision making"""
        
        decision = await self.policy_decision_engine.make_decision(risk_assessment, context)
        
        logger.debug(f"Policy Decision: {decision.value} for session {context.session_id}")
        
        return decision

    async def _stage7_generate_response(
        self,
        layer1_result: Layer1Result,
        layer2_result: Optional[Layer2Result],
        drift_result: DriftAnalysisResult,
        risk_assessment: RiskAssessmentResult,
        decision: AuthenticationDecision,
        context: SessionContext,
        processing_stages: Dict[ProcessingStage, float],
        start_time: float,
        require_explanation: bool
    ) -> MLEngineResponse:
        """Stage 7: Generate comprehensive response"""
        
        total_processing_time = (time.time() - start_time) * 1000
        
        # Generate explanation if required
        explanation = {}
        if require_explanation:
            explanation = await self._generate_explanation(
                layer1_result, layer2_result, drift_result, risk_assessment, decision, context
            )
        
        # Determine next verification interval and monitoring level
        next_interval, monitoring_level = self._calculate_session_parameters(risk_assessment, decision)
        
        # Generate session actions
        session_actions = self._generate_session_actions(decision, risk_assessment, context)
        
        response = MLEngineResponse(
            session_id=context.session_id,
            user_id=context.user_id,
            decision=decision,
            risk_assessment=risk_assessment,
            layer1_result=layer1_result,
            layer2_result=layer2_result,
            drift_result=drift_result,
            total_processing_time_ms=total_processing_time,
            processing_stages=processing_stages,
            next_verification_interval_seconds=next_interval,
            monitoring_level=monitoring_level,
            session_actions=session_actions,
            explanation=explanation,
            confidence=risk_assessment.confidence,
            timestamp=datetime.now()
        )
        
        return response

    def _update_session_tracking(self, context: SessionContext):
        """Update session tracking for the ML engine"""
        with self.session_lock:
            self.active_sessions[context.session_id] = {
                "user_id": context.user_id,
                "start_time": context.session_start_time,
                "last_activity": context.last_activity,
                "processing_count": self.active_sessions.get(context.session_id, {}).get("processing_count", 0) + 1,
                "current_risk_level": context.current_risk_level
            }

    def _update_performance_metrics(self, response: MLEngineResponse, processing_time: float):
        """Update performance metrics"""
        self.performance_stats["total_requests"] += 1
        
        if response.decision in [AuthenticationDecision.ALLOW, AuthenticationDecision.MONITOR]:
            self.performance_stats["successful_authentications"] += 1
        elif response.decision in [AuthenticationDecision.TEMPORARY_BLOCK, AuthenticationDecision.PERMANENT_BLOCK]:
            self.performance_stats["blocked_sessions"] += 1
        
        # Update average processing time
        total_requests = self.performance_stats["total_requests"]
        current_avg = self.performance_stats["average_processing_time"]
        self.performance_stats["average_processing_time"] = (
            (current_avg * (total_requests - 1) + processing_time * 1000) / total_requests
        )

    async def _generate_explanation(
        self,
        layer1_result: Layer1Result,
        layer2_result: Optional[Layer2Result],
        drift_result: DriftAnalysisResult,
        risk_assessment: RiskAssessmentResult,
        decision: AuthenticationDecision,
        context: SessionContext
    ) -> Dict[str, Any]:
        """Generate detailed explanation for the decision"""
        
        return {
            "decision": decision.value,
            "risk_score": risk_assessment.overall_risk_score,
            "risk_level": risk_assessment.risk_level.name,
            "primary_factors": [
                {"factor": factor, "impact": impact} 
                for factor, impact in risk_assessment.contributing_factors[:3]
            ],
            "mitigating_factors": [
                {"factor": factor, "impact": impact}
                for factor, impact in risk_assessment.mitigating_factors[:3]
            ],
            "layer1_analysis": {
                "similarity_score": layer1_result.similarity_score,
                "confidence": layer1_result.confidence_level,
                "matched_profiles": len(layer1_result.matched_profiles)
            },
            "layer2_analysis": {
                "transformer_confidence": layer2_result.transformer_confidence if layer2_result else None,
                "gnn_anomaly": layer2_result.gnn_anomaly_score if layer2_result else None,
                "contextual_score": layer2_result.contextual_adaptation_score if layer2_result else None
            } if layer2_result else None,
            "drift_analysis": {
                "drift_detected": drift_result.drift_detected,
                "drift_type": drift_result.drift_type,
                "magnitude": drift_result.drift_magnitude
            },
            "context_factors": {
                "device_trust": context.is_known_device,
                "location_trust": context.is_trusted_location,
                "session_duration": context.session_duration_minutes,
                "time_of_day": context.time_of_day
            },
            "human_readable": risk_assessment.explanation
        }

    def _calculate_session_parameters(
        self, 
        risk_assessment: RiskAssessmentResult, 
        decision: AuthenticationDecision
    ) -> Tuple[int, str]:
        """Calculate next verification interval and monitoring level"""
        
        # Banking-grade intervals based on risk and decision
        if decision == AuthenticationDecision.ALLOW:
            if risk_assessment.risk_level == RiskLevel.VERY_LOW:
                return 600, "standard"  # 10 minutes
            elif risk_assessment.risk_level == RiskLevel.LOW:
                return 300, "standard"  # 5 minutes
            else:
                return 180, "elevated"  # 3 minutes
                
        elif decision == AuthenticationDecision.MONITOR:
            return 120, "intensive"  # 2 minutes
            
        elif decision == AuthenticationDecision.CHALLENGE:
            return 60, "intensive"   # 1 minute
            
        else:  # STEP_UP_AUTH, BLOCKS, etc.
            return 30, "critical"    # 30 seconds

    def _generate_session_actions(
        self, 
        decision: AuthenticationDecision, 
        risk_assessment: RiskAssessmentResult, 
        context: SessionContext
    ) -> List[str]:
        """Generate session management actions"""
        
        actions = []
        
        # Standard actions based on decision
        if decision == AuthenticationDecision.MONITOR:
            actions.append("increase_monitoring")
            
        elif decision == AuthenticationDecision.CHALLENGE:
            actions.append("request_soft_challenge")
            
        elif decision == AuthenticationDecision.STEP_UP_AUTH:
            actions.append("request_biometric_verification")
            
        elif decision == AuthenticationDecision.TEMPORARY_BLOCK:
            actions.append("temporary_session_block")
            actions.append("notify_security_team")
            
        elif decision == AuthenticationDecision.PERMANENT_BLOCK:
            actions.append("end_session")
            actions.append("alert_fraud_team")
            actions.append("log_security_incident")
            
        # Risk-based additional actions
        if risk_assessment.overall_risk_score > 0.8:
            actions.append("flag_high_risk_session")
            
        if not context.is_known_device and risk_assessment.overall_risk_score > 0.6:
            actions.append("require_device_verification")
            
        # Banking-specific actions
        if context.current_transaction_context:
            transaction_amount = context.current_transaction_context.get('amount', 0)
            if transaction_amount > 100000 and risk_assessment.overall_risk_score > 0.4:
                actions.append("require_transaction_otp")
                
        return actions

    async def _create_error_response(
        self, 
        context: SessionContext, 
        error: str, 
        processing_time: float
    ) -> MLEngineResponse:
        """Create error response for failed processing"""
        
        # Create minimal risk assessment for error case
        error_risk_assessment = RiskAssessmentResult(
            session_id=context.session_id,
            user_id=context.user_id,
            overall_risk_score=0.7,  # Conservative high risk on error
            risk_level=RiskLevel.HIGH,
            risk_components={"error": 1.0},
            confidence=0.3,
            contributing_factors=[("processing_error", 1.0)],
            mitigating_factors=[],
            recommendation=AuthenticationDecision.STEP_UP_AUTH,
            explanation=f"Authentication failed due to processing error: {error}"
        )
        
        # Create minimal Layer 1 result
        error_layer1_result = Layer1Result(
            user_id=context.user_id,
            session_id=context.session_id,
            similarity_score=0.0,
            confidence_level="error",
            matched_profiles=[],
            decision="escalate",
            processing_time_ms=processing_time * 1000,
            threshold_analysis={},
            metadata={"error": error}
        )
        
        return MLEngineResponse(
            session_id=context.session_id,
            user_id=context.user_id,
            decision=AuthenticationDecision.STEP_UP_AUTH,
            risk_assessment=error_risk_assessment,
            layer1_result=error_layer1_result,
            layer2_result=None,
            drift_result=None,
            total_processing_time_ms=processing_time * 1000,
            processing_stages={},
            next_verification_interval_seconds=30,
            monitoring_level="critical",
            session_actions=["error_fallback", "require_manual_verification"],
            explanation={"error": error, "fallback_decision": True},
            confidence=0.3,
            timestamp=datetime.now()
        )

    async def shutdown(self):
        """Graceful shutdown of ML engine"""
        logger.info("🏦 Shutting down Industry-Grade Banking ML Engine...")
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        # Save models and state
        if self.layer1_faiss_engine:
            await self.layer1_faiss_engine.save_models()
        if self.layer2_adaptive_engine:
            await self.layer2_adaptive_engine.save_models()
        if self.drift_analysis_engine:
            await self.drift_analysis_engine.save_models()
        
        logger.info("✅ Industry-Grade Banking ML Engine shutdown complete")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        return {
            **self.performance_stats,
            "active_sessions": len(self.active_sessions),
            "component_status": {
                "preprocessing": self.preprocessing_pipeline is not None,
                "layer1_faiss": self.layer1_faiss_engine is not None,
                "layer2_adaptive": self.layer2_adaptive_engine is not None,
                "drift_analysis": self.drift_analysis_engine is not None,
                "risk_assessment": self.risk_assessment_engine is not None,
                "policy_decision": self.policy_decision_engine is not None
            }
        }

# Global industry-grade ML engine instance
banking_ml_engine = IndustryGradeMLEngine()

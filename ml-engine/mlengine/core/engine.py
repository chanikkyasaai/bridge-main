"""
BRIDGE Industry-Grade ML-Engine Core
Banking-focused behavioral authentication with complete session lifecycle management

This is the main orchestrator for the BRIDGE ML-Engine, designed specifically for 
banking applications with industry-grade standards, compliance, and security.

Pipeline Order (Strict):
1. Input Validation & Preprocessing
2. Layer 1: FAISS Fast Verification  
3. Layer 2: Adaptive Context Analysis
4. Drift Detection & Profile Adaptation
5. Risk Assessment & Aggregation
6. Policy Decision Engine
7. Response Generation & Logging

Session Lifecycle Integration:
- ML-Engine activates when backend session starts
- Continuous processing during session lifetime
- Proper cleanup when session ends
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor
import uuid
from pathlib import Path

from mlengine.config import CONFIG
from mlengine.scripts.banking_cold_start import banking_cold_start_handler, UserProfileStage, ThreatLevel
from mlengine.adapters.level2.layer2_verifier import Layer2Verifier

logger = logging.getLogger(__name__)

class AuthenticationDecision(Enum):
    """Final authentication decisions for banking applications"""
    ALLOW = "allow"                    # Continue session normally
    MONITOR = "monitor"                # Increased monitoring, no user impact
    CHALLENGE = "challenge"            # Soft challenge (security question)
    STEP_UP_AUTH = "step_up_auth"     # Require biometric/PIN verification
    TEMPORARY_BLOCK = "temporary_block" # Block for specified duration
    PERMANENT_BLOCK = "permanent_block" # End session permanently
    REQUIRE_REAUTH = "require_reauth"  # Force complete re-authentication

class RiskLevel(Enum):
    """Risk assessment levels with banking compliance thresholds"""
    VERY_LOW = "very_low"     # 0.0 - 0.15
    LOW = "low"               # 0.15 - 0.35
    MEDIUM = "medium"         # 0.35 - 0.60
    HIGH = "high"             # 0.60 - 0.80
    CRITICAL = "critical"     # 0.80 - 1.0

class ProcessingStage(Enum):
    """ML processing pipeline stages - ORDERED"""
    INPUT_VALIDATION = "input_validation"
    PREPROCESSING = "preprocessing"
    LAYER1_FAISS = "layer1_faiss"
    LAYER2_ADAPTIVE = "layer2_adaptive"
    DRIFT_DETECTION = "drift_detection"
    RISK_AGGREGATION = "risk_aggregation"
    POLICY_DECISION = "policy_decision"
    OUTPUT_GENERATION = "output_generation"

class SessionState(Enum):
    """ML-Engine session states"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    MONITORING = "monitoring"
    SUSPENDED = "suspended"
    TERMINATED = "terminated"
    ERROR = "error"

@dataclass
class BehavioralEvent:
    """Single behavioral event from client"""
    timestamp: datetime
    event_type: str              # touch, swipe, type, scroll, etc.
    features: Dict[str, float]   # pressure, velocity, duration, etc.
    session_id: str
    user_id: str
    device_id: str
    raw_metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "features": self.features,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "device_id": self.device_id,
            "raw_metadata": self.raw_metadata
        }

@dataclass
class BehavioralVector:
    """Processed behavioral vector ready for ML analysis"""
    vector: np.ndarray
    confidence: float
    timestamp: datetime
    session_id: str
    user_id: str
    source_events: List[BehavioralEvent]
    processing_metadata: Dict[str, Any]
    vector_type: str = "standard"  # standard, contextual, temporal
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id,
            "user_id": self.user_id,
            "vector_shape": self.vector.shape,
            "vector_type": self.vector_type,
            "source_event_count": len(self.source_events),
            "processing_metadata": self.processing_metadata
        }

@dataclass
class SessionContext:
    """Complete session context for ML processing"""
    session_id: str
    user_id: str
    device_id: str
    session_start_time: datetime
    last_activity: datetime
    session_duration_minutes: float
    # Device & Environment Context
    device_type: str
    device_model: str
    os_version: str
    app_version: str
    network_type: str
    # Behavioral Context
    time_of_day: str
    usage_pattern: str
    interaction_frequency: float
    typical_session_duration: float
    # Security Context
    is_known_device: bool
    is_trusted_location: bool
    # Fields with defaults below
    location_data: Optional[Dict[str, Any]] = None
    recent_security_events: List[Dict[str, Any]] = field(default_factory=list)
    risk_flags: List[str] = field(default_factory=list)

@dataclass
class Layer1Result:
    """FAISS Layer 1 verification result"""
    user_id: str
    session_id: str
    similarity_score: float
    confidence_level: str           # high, medium, low
    matched_profile_mode: str       # normal, hurried, stressed, etc.
    decision: str                   # continue, escalate, block
    processing_time_ms: float
    requires_layer2: bool
    metadata: Dict[str, Any]

@dataclass
class Layer2Result:
    """Adaptive Layer 2 verification result"""
    user_id: str
    session_id: str
    transformer_confidence: float
    gnn_anomaly_score: float
    context_adaptation_score: float
    overall_confidence: float
    decision: str
    processing_time_ms: float
    explanation_features: Dict[str, float]
    metadata: Dict[str, Any]

@dataclass
class DriftResult:
    """Drift detection result"""
    user_id: str
    session_id: str
    drift_detected: bool
    drift_magnitude: float
    drift_type: str                 # concept, data, behavioral
    adaptation_required: bool
    profile_update_recommendation: Dict[str, Any]
    confidence: float
    processing_time_ms: float

@dataclass
class RiskAssessment:
    """Comprehensive risk assessment"""
    user_id: str
    session_id: str
    overall_risk_score: float
    risk_level: RiskLevel
    component_scores: Dict[str, float]  # faiss, transformer, gnn, drift, context
    risk_factors: List[Tuple[str, float]]  # [(factor_name, weight), ...]
    confidence: float
    processing_time_ms: float

@dataclass
class PolicyDecision:
    """Final policy decision with banking compliance"""
    user_id: str
    session_id: str
    decision: AuthenticationDecision
    risk_assessment: RiskAssessment
    reasoning: str
    compliance_flags: List[str]
    next_verification_delay: int    # seconds
    session_constraints: Dict[str, Any]
    audit_trail: List[Dict[str, Any]]
    processing_time_ms: float

@dataclass
class AuthenticationResponse:
    """Final authentication response"""
    session_id: str
    user_id: str
    request_id: str
    decision: AuthenticationDecision
    risk_level: RiskLevel
    risk_score: float
    confidence: float
    
    # Processing Results
    layer1_result: Layer1Result
    layer2_result: Optional[Layer2Result]
    drift_result: Optional[DriftResult]
    policy_decision: PolicyDecision
    
    # Performance Metrics
    total_processing_time_ms: float
    stage_timings: Dict[ProcessingStage, float]
    
    # Explainability & Compliance
    explanation: Dict[str, Any]
    audit_trail: List[Dict[str, Any]]
    
    timestamp: datetime
    next_verification_delay: int

@dataclass
class SessionMLState:
    """ML-Engine state for active session"""
    session_id: str
    user_id: str
    state: SessionState
    created_at: datetime
    last_activity: datetime
    session_start_time: datetime  # <-- Add this line
    # Processing Buffers
    event_buffer: deque
    vector_buffer: deque
    # Session Statistics
    total_events_processed: int
    total_authentications: int
    average_risk_score: float
    last_risk_assessment: Optional[RiskAssessment]
    # Performance Tracking
    processing_times: deque  # Keep last 100 processing times
    error_count: int
    last_error: Optional[Dict[str, Any]]

class BankingMLEngine:
    """
    Industry-Grade ML-Engine for Banking Behavioral Authentication
    
    Features:
    - Strict ordered processing pipeline
    - Session lifecycle management
    - Real-time behavioral analysis
    - Banking compliance & audit trails
    - High performance & reliability
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or CONFIG
        self.logger = logging.getLogger(f"{__name__}.BankingMLEngine")
        
        # Core Components (initialized async)
        self.input_validator = None
        self.behavioral_preprocessor = None
        self.layer1_faiss_engine = None
        self.layer2_adaptive_engine = None
        self.drift_analysis_engine = None
        self.risk_assessment_engine = None
        self.policy_decision_engine = None
        
        # Session Management
        self.active_sessions: Dict[str, SessionMLState] = {}
        self.session_lock = threading.RLock()
        
        # Performance & Monitoring
        self.total_requests = 0
        self.total_processing_time = 0.0
        self.error_count = 0
        self.start_time = datetime.utcnow()
        self.performance_metrics = defaultdict(list)
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=self.config.get('max_workers', 8))
        self.background_tasks = []
        
        # State
        self.is_initialized = False
        self.is_running = False
        
        # Layer 2 Verifier
        self.layer2_verifier = None
        
    async def initialize(self) -> bool:
        """Initialize all ML components and verify system readiness"""
        try:
            self.logger.info("Initializing BRIDGE Banking ML-Engine...")
            
            # Initialize components in dependency order
            await self._init_input_validator()
            await self._init_behavioral_preprocessor()
            await self._init_layer1_faiss_engine()
            await self._init_layer2_adaptive_engine()
            await self._init_drift_analysis_engine()
            await self._init_risk_assessment_engine()
            await self._init_policy_decision_engine()
            
            # Start background monitoring tasks
            await self._start_background_tasks()
            
            self.is_initialized = True
            self.is_running = True
            self.logger.info("ML-Engine initialization completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"ML-Engine initialization failed: {e}")
            self.is_initialized = False
            return False
    
    async def start_session(self, session_id: str, user_id: str, context: SessionContext) -> bool:
        """Start ML processing for a new session"""
        try:
            with self.session_lock:
                if session_id in self.active_sessions:
                    self.logger.warning(f"Session {session_id} already exists")
                    return False
                
                # Create session ML state
                session_state = SessionMLState(
                    session_id=session_id,
                    user_id=user_id,
                    state=SessionState.INITIALIZING,
                    created_at=datetime.utcnow(),
                    last_activity=datetime.utcnow(),
                    session_start_time=context.session_start_time,  # <-- Set here
                    event_buffer=deque(maxlen=self.config.get('event_buffer_size', 1000)),
                    vector_buffer=deque(maxlen=self.config.get('vector_buffer_size', 100)),
                    total_events_processed=0,
                    total_authentications=0,
                    average_risk_score=0.0,
                    last_risk_assessment=None,
                    processing_times=deque(maxlen=100),
                    error_count=0,
                    last_error=None
                )
                
                self.active_sessions[session_id] = session_state
                
                # Initialize session-specific components
                await self._initialize_session_components(session_id, user_id, context)
                
                session_state.state = SessionState.ACTIVE
                self.logger.info(f"ML-Engine session started: {session_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to start ML session {session_id}: {e}")
            return False
    
    async def process_behavioral_event(self, event: BehavioralEvent) -> Optional[AuthenticationResponse]:
        """
        Process a single behavioral event through the complete ML pipeline
        
        STRICT PIPELINE ORDER:
        1. Input Validation & Preprocessing
        2. Layer 1: FAISS Fast Verification
        3. Layer 2: Adaptive Context Analysis (if needed)
        4. Drift Detection & Profile Adaptation
        5. Risk Assessment & Aggregation
        6. Policy Decision Engine
        7. Response Generation & Logging
        """
        request_id = str(uuid.uuid4())
        start_time = time.perf_counter()
        stage_timings = {}
        audit_trail = []  # Always define audit_trail as a list
        
        try:
            self.total_requests += 1
            session_state = self.active_sessions.get(event.session_id)
            
            if not session_state or session_state.state != SessionState.ACTIVE:
                self.logger.warning(f"No active session for event: {event.session_id}")
                return AuthenticationResponse(
                    session_id=event.session_id,
                    user_id=event.user_id,
                    request_id=request_id,
                    decision=AuthenticationDecision.MONITOR,
                    risk_level=RiskLevel.LOW,
                    risk_score=0.0,
                    confidence=0.0,
                    layer1_result=None,
                    layer2_result=None,
                    drift_result=None,
                    policy_decision=None,
                    total_processing_time_ms=(time.perf_counter() - start_time) * 1000,
                    stage_timings=stage_timings,
                    explanation={"error": "No active session"},
                    audit_trail=audit_trail,
                    timestamp=datetime.utcnow(),
                    next_verification_delay=30
                )
            
            # Update session activity
            session_state.last_activity = datetime.utcnow()
            session_state.total_events_processed += 1
            session_state.event_buffer.append(event)
            
            # STAGE 1: Input Validation & Preprocessing
            stage_start = time.perf_counter()
            processed_vector = await self._stage1_input_validation_preprocessing(event, session_state)
            if not processed_vector:
                return AuthenticationResponse(
                    session_id=event.session_id,
                    user_id=event.user_id,
                    request_id=request_id,
                    decision=AuthenticationDecision.MONITOR,
                    risk_level=RiskLevel.LOW,
                    risk_score=0.0,
                    confidence=0.0,
                    layer1_result=None,
                    layer2_result=None,
                    drift_result=None,
                    policy_decision=None,
                    total_processing_time_ms=(time.perf_counter() - start_time) * 1000,
                    stage_timings=stage_timings,
                    explanation={"error": "Input validation failed"},
                    audit_trail=audit_trail,
                    timestamp=datetime.utcnow(),
                    next_verification_delay=30
                )
            stage_timings[ProcessingStage.PREPROCESSING] = (time.perf_counter() - stage_start) * 1000
            audit_trail.append({"stage": "preprocessing", "status": "success", "timestamp": datetime.utcnow().isoformat()})
            
            # STAGE 2: Layer 1 FAISS Fast Verification
            stage_start = time.perf_counter()
            layer1_result = await self._stage2_layer1_faiss_verification(processed_vector, session_state)
            stage_timings[ProcessingStage.LAYER1_FAISS] = (time.perf_counter() - stage_start) * 1000
            audit_trail.append({"stage": "layer1_faiss", "decision": layer1_result.decision, "confidence": layer1_result.confidence_level})
            
            # STAGE 3: Layer 2 Adaptive Analysis (conditional)
            layer2_result = None
            if layer1_result.requires_layer2:
                stage_start = time.perf_counter()
                layer2_result = await self._stage3_layer2_adaptive_analysis(processed_vector, layer1_result, session_state, audit_trail)
                stage_timings[ProcessingStage.LAYER2_ADAPTIVE] = (time.perf_counter() - stage_start) * 1000
                audit_trail.append({"stage": "layer2_adaptive", "confidence": layer2_result.overall_confidence})
            
            # STAGE 4: Drift Detection & Profile Adaptation
            stage_start = time.perf_counter()
            drift_result = await self._stage4_drift_detection(processed_vector, session_state)
            stage_timings[ProcessingStage.DRIFT_DETECTION] = (time.perf_counter() - stage_start) * 1000
            audit_trail.append({"stage": "drift_detection", "drift_detected": drift_result.drift_detected})
            
            # STAGE 5: Risk Assessment & Aggregation
            stage_start = time.perf_counter()
            
            # Banking Cold Start & Early Threat Detection
            banking_decision = await self._banking_cold_start_assessment(event, session_state)
            
            risk_assessment = await self._stage5_risk_assessment(layer1_result, layer2_result, drift_result, session_state, banking_decision)
            stage_timings[ProcessingStage.RISK_AGGREGATION] = (time.perf_counter() - stage_start) * 1000
            
            # STAGE 6: Policy Decision Engine
            stage_start = time.perf_counter()
            policy_decision = await self._stage6_policy_decision(risk_assessment, session_state, banking_decision, audit_trail)
            stage_timings[ProcessingStage.POLICY_DECISION] = (time.perf_counter() - stage_start) * 1000
            
            # STAGE 7: Response Generation & Logging
            stage_start = time.perf_counter()
            response = await self._stage7_response_generation(
                request_id, layer1_result, layer2_result, drift_result, 
                policy_decision, stage_timings, audit_trail, start_time
            )
            stage_timings[ProcessingStage.OUTPUT_GENERATION] = (time.perf_counter() - stage_start) * 1000
            
            # Update session statistics
            total_time = (time.perf_counter() - start_time) * 1000
            session_state.processing_times.append(total_time)
            session_state.total_authentications += 1
            session_state.last_risk_assessment = risk_assessment
            session_state.average_risk_score = (
                (session_state.average_risk_score * (session_state.total_authentications - 1) + 
                 risk_assessment.overall_risk_score) / session_state.total_authentications
            )
            
            self.total_processing_time += total_time
            self.performance_metrics['processing_time'].append(total_time)
            
            return response
            
        except Exception as e:
            self.error_count += 1
            if event.session_id in self.active_sessions:
                self.active_sessions[event.session_id].error_count += 1
                self.active_sessions[event.session_id].last_error = {
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat(),
                    "request_id": request_id
                }
            
            self.logger.error(f"Error processing behavioral event {request_id}: {e}")
            # Always return a fallback AuthenticationResponse with audit_trail
            return AuthenticationResponse(
                session_id=event.session_id,
                user_id=event.user_id,
                request_id=request_id,
                decision=AuthenticationDecision.MONITOR,
                risk_level=RiskLevel.LOW,
                risk_score=0.0,
                confidence=0.0,
                layer1_result=None,
                layer2_result=None,
                drift_result=None,
                policy_decision=None,
                total_processing_time_ms=(time.perf_counter() - start_time) * 1000,
                stage_timings=stage_timings,
                explanation={"error": str(e)},
                audit_trail=audit_trail,
                timestamp=datetime.utcnow(),
                next_verification_delay=30
            )
    
    async def end_session(self, session_id: str, final_decision: str = "normal") -> bool:
        """End ML processing for a session and cleanup resources"""
        try:
            with self.session_lock:
                if session_id not in self.active_sessions:
                    self.logger.warning(f"Attempted to end non-existent session: {session_id}")
                    return False
                
                session_state = self.active_sessions[session_id]
                session_state.state = SessionState.TERMINATED
                
                # Perform final session analysis and cleanup
                await self._finalize_session_analysis(session_state, final_decision)
                
                # Cleanup session resources
                await self._cleanup_session_resources(session_id)
                
                # Remove from active sessions
                del self.active_sessions[session_id]
                
                self.logger.info(f"ML-Engine session ended: {session_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error ending ML session {session_id}: {e}")
            return False
    
    async def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of ML processing for a session"""
        session_state = self.active_sessions.get(session_id)
        if not session_state:
            return None
        
        return {
            "session_id": session_id,
            "user_id": session_state.user_id,
            "state": session_state.state.value,
            "created_at": session_state.created_at.isoformat(),
            "last_activity": session_state.last_activity.isoformat(),
            "total_events_processed": session_state.total_events_processed,
            "total_authentications": session_state.total_authentications,
            "average_risk_score": session_state.average_risk_score,
            "error_count": session_state.error_count,
            "avg_processing_time_ms": np.mean(list(session_state.processing_times)) if session_state.processing_times else 0
        }
    
    async def get_engine_stats(self) -> Dict[str, Any]:
        """Get comprehensive ML-Engine statistics"""
        uptime = (datetime.utcnow() - self.start_time).total_seconds() / 3600
        avg_processing_time = (self.total_processing_time / self.total_requests) if self.total_requests > 0 else 0
        
        return {
            "is_initialized": self.is_initialized,
            "is_running": self.is_running,
            "uptime_hours": round(uptime, 2),
            "total_requests": self.total_requests,
            "error_count": self.error_count,
            "error_rate": (self.error_count / self.total_requests) if self.total_requests > 0 else 0,
            "average_processing_time_ms": round(avg_processing_time, 2),
            "active_sessions": len(self.active_sessions),
            "recent_performance": {
                "last_100_requests_avg_ms": np.mean(self.performance_metrics['processing_time'][-100:]) if self.performance_metrics['processing_time'] else 0
            }
        }
    
    async def shutdown(self) -> bool:
        """Graceful shutdown of ML-Engine"""
        try:
            self.logger.info("Shutting down BRIDGE Banking ML-Engine...")
            
            self.is_running = False
            
            # End all active sessions
            session_ids = list(self.active_sessions.keys())
            for session_id in session_ids:
                await self.end_session(session_id, "shutdown")
            
            # Stop background tasks
            for task in self.background_tasks:
                task.cancel()
            
            # Shutdown components
            await self._shutdown_components()
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            self.logger.info("ML-Engine shutdown completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during ML-Engine shutdown: {e}")
            return False
    
    # Stage Implementation Methods (Placeholders for now - will be implemented with actual components)
    
    async def _stage1_input_validation_preprocessing(self, event: BehavioralEvent, session_state: SessionMLState) -> Optional[BehavioralVector]:
        """Stage 1: Validate input and preprocess behavioral data"""
        # TODO: Implement with actual behavioral preprocessing component
        # For now, return a mock vector
        vector = np.random.random(128)  # Mock 128-dimensional vector
        return BehavioralVector(
            vector=vector,
            confidence=0.9,
            timestamp=event.timestamp,
            session_id=event.session_id,
            user_id=event.user_id,
            source_events=[event],
            processing_metadata={"stage": "preprocessing", "mock": True}
        )
    
    async def _stage2_layer1_faiss_verification(self, vector: BehavioralVector, session_state: SessionMLState) -> Layer1Result:
        """Stage 2: FAISS-based fast similarity verification"""
        # TODO: Implement with actual FAISS component
        # For now, return a mock result
        return Layer1Result(
            user_id=vector.user_id,
            session_id=vector.session_id,
            similarity_score=0.85,
            confidence_level="high",
            matched_profile_mode="normal",
            decision="continue",
            processing_time_ms=8.5,
            requires_layer2=False,
            metadata={"mock": True}
        )
    
    async def _stage3_layer2_adaptive_analysis(self, vector: BehavioralVector, layer1_result: Layer1Result, session_state: SessionMLState, audit_trail: list = None) -> Layer2Result:
        """Stage 3: Adaptive context-aware analysis using Transformer/GNN"""
        if audit_trail is None:
            audit_trail = []
        try:
            N = 10
            if hasattr(session_state, 'vector_buffer') and session_state.vector_buffer:
                vectors = list(session_state.vector_buffer)[-N:]
            else:
                vectors = [vector]
            events = list(session_state.event_buffer)
            context = getattr(session_state, 'context', {})
            if not context and hasattr(session_state, 'user_id'):
                context = {
                    'user_id': session_state.user_id,
                    'session_id': session_state.session_id
                }
            result = await self.layer2_verifier.verify(
                vectors=vectors,
                events=events,
                context=context,
                session_id=session_state.session_id
            )
            return result
        except Exception as e:
            self.logger.error(f"Layer 2 adaptive analysis failed: {e}")
            audit_trail.append({"stage": "layer2_adaptive", "error": str(e)})
            return Layer2Result(
                user_id=vector.user_id,
                session_id=vector.session_id,
                transformer_confidence=0.5,
                gnn_anomaly_score=0.5,
                context_adaptation_score=0.5,
                overall_confidence=0.5,
                decision="escalate",
                processing_time_ms=0.0,
                explanation_features={"error": str(e)},
                metadata={"error": str(e)}
            )
    
    async def _stage4_drift_detection(self, vector: BehavioralVector, session_state: SessionMLState) -> DriftResult:
        """Stage 4: Detect behavioral drift and adaptation needs"""
        # TODO: Implement with actual drift detection component
        return DriftResult(
            user_id=vector.user_id,
            session_id=vector.session_id,
            drift_detected=False,
            drift_magnitude=0.05,
            drift_type="none",
            adaptation_required=False,
            profile_update_recommendation={},
            confidence=0.9,
            processing_time_ms=12.3
        )
    
    async def _banking_cold_start_assessment(self, event: BehavioralEvent, session_state: SessionMLState) -> Dict[str, Any]:
        """Banking-specific cold start and early threat detection"""
        try:
            # Convert event buffer to format expected by cold start handler
            session_events = []
            for buffered_event in session_state.event_buffer:
                session_events.append(buffered_event.to_dict())
            
            # Get banking security decision
            banking_decision = await banking_cold_start_handler.get_banking_security_decision(
                event.user_id, event.session_id, session_events
            )
            
            # Process session for learning (if in observation mode)
            if banking_decision.get('requires_profile_building', False):
                session_duration = (datetime.utcnow() - session_state.session_start_time).total_seconds() / 60.0
                await banking_cold_start_handler.process_session_learning(
                    event.user_id, event.session_id, session_events, session_duration
                )
            
            self.logger.info(f"Banking decision for {event.session_id}: {banking_decision['action']} - {banking_decision['reason']}")
            
            return banking_decision
            
        except Exception as e:
            self.logger.error(f"Banking cold start assessment failed: {e}")
            return {
                "action": "continue",
                "reason": "Banking assessment failed - defaulting to continue",
                "profile_stage": "unknown",
                "observation_mode": False,
                "threat_level": "none"
            }
    
    async def _stage5_risk_assessment(self, layer1_result: Layer1Result, layer2_result: Optional[Layer2Result], 
                                    drift_result: DriftResult, session_state: SessionMLState, 
                                    banking_decision: Dict[str, Any] = None) -> RiskAssessment:
        """Stage 5: Aggregate all risk signals into final assessment with banking logic"""
        
        # Base risk scores from ML components
        base_risk = 0.25
        component_scores = {
            "faiss_similarity": 0.15,
            "transformer_confidence": 0.22 if layer2_result else 0.0,
            "gnn_anomaly": 0.15 if layer2_result else 0.0,
            "drift_score": 0.05,
            "context_score": 0.10
        }
        
        # Integrate banking decision
        if banking_decision:
            threat_indicators = banking_decision.get('threat_indicators', {})
            
            # Add banking-specific risk components
            component_scores.update({
                "bot_detection": threat_indicators.get('bot_score', 0.0),
                "automation_detection": threat_indicators.get('automation_score', 0.0),
                "speed_anomaly": threat_indicators.get('speed_anomaly_score', 0.0),
                "pattern_anomaly": threat_indicators.get('pattern_anomaly_score', 0.0),
                "device_compromise": threat_indicators.get('device_compromise_score', 0.0)
            })
            
            # Calculate overall risk with banking factors
            banking_risk = max(
                threat_indicators.get('bot_score', 0.0),
                threat_indicators.get('automation_score', 0.0),
                threat_indicators.get('speed_anomaly_score', 0.0),
                threat_indicators.get('pattern_anomaly_score', 0.0),
                threat_indicators.get('device_compromise_score', 0.0)
            )
            
            # Combine base risk with banking risk
            overall_risk = max(base_risk, banking_risk)
            
            # Override risk level based on banking decision
            if banking_decision.get('threat_level') == 'critical':
                risk_level = RiskLevel.CRITICAL
                overall_risk = max(overall_risk, 0.9)
            elif banking_decision.get('threat_level') == 'high':
                risk_level = RiskLevel.HIGH
                overall_risk = max(overall_risk, 0.7)
            elif banking_decision.get('threat_level') == 'medium':
                risk_level = RiskLevel.MEDIUM
                overall_risk = max(overall_risk, 0.5)
            elif banking_decision.get('observation_mode'):
                risk_level = RiskLevel.LOW  # Observation mode - keep risk low
                overall_risk = min(overall_risk, 0.3)
            else:
                # Determine risk level from score
                if overall_risk >= 0.8:
                    risk_level = RiskLevel.CRITICAL
                elif overall_risk >= 0.6:
                    risk_level = RiskLevel.HIGH
                elif overall_risk >= 0.35:
                    risk_level = RiskLevel.MEDIUM
                elif overall_risk >= 0.15:
                    risk_level = RiskLevel.LOW
                else:
                    risk_level = RiskLevel.VERY_LOW
        else:
            overall_risk = base_risk
            risk_level = RiskLevel.LOW
        
        # Build risk factors list
        risk_factors = []
        for component, score in component_scores.items():
            if score > 0.1:  # Only include significant factors
                risk_factors.append((component.replace('_', ' ').title(), score))
        
        # Add banking-specific factors
        if banking_decision and banking_decision.get('threat_indicators', {}).get('specific_threats'):
            for threat in banking_decision['threat_indicators']['specific_threats']:
                risk_factors.append(("Banking Threat", threat))
        
        return RiskAssessment(
            user_id=layer1_result.user_id,
            session_id=layer1_result.session_id,
            overall_risk_score=overall_risk,
            risk_level=risk_level,
            component_scores=component_scores,
            risk_factors=risk_factors,
            confidence=0.87,
            processing_time_ms=5.8
        )
    
    async def _stage6_policy_decision(self, risk_assessment: RiskAssessment, session_state: SessionMLState, 
                                     banking_decision: Dict[str, Any] = None, audit_trail: list = None) -> PolicyDecision:
        """Stage 6: Make policy decision based on risk assessment and banking logic"""
        if audit_trail is None:
            audit_trail = []
        # Default values
        decision = AuthenticationDecision.ALLOW
        reasoning = "Normal behavioral patterns detected"
        next_delay = 300  # 5 minutes
        constraints = {}
        compliance_flags = []
        # Banking decision takes precedence
        if banking_decision:
            action = banking_decision.get('action', 'continue')
            if action == 'block':
                decision = AuthenticationDecision.PERMANENT_BLOCK
                reasoning = banking_decision.get('reason', 'Critical security threat detected')
                next_delay = 0
                compliance_flags.append("SECURITY_THREAT_DETECTED")
            elif action == 'step_up_auth':
                decision = AuthenticationDecision.STEP_UP_AUTH
                reasoning = banking_decision.get('reason', 'Additional verification required')
                next_delay = 0
            elif action == 'flag_and_continue':
                decision = AuthenticationDecision.MONITOR
                reasoning = banking_decision.get('reason', 'Flagged for review but continuing')
                constraints['increased_monitoring'] = True
                compliance_flags.append("FLAGGED_FOR_REVIEW")
            elif action == 'observe':
                decision = AuthenticationDecision.ALLOW
                reasoning = banking_decision.get('reason', 'Observation mode - learning user behavior')
                constraints['observation_mode'] = True
                constraints['session_count'] = banking_decision.get('session_count', 0)
                compliance_flags.append("OBSERVATION_MODE")
            else:  # continue
                decision = AuthenticationDecision.ALLOW
                reasoning = banking_decision.get('reason', 'Normal behavior - continue session')
        elif risk_assessment.risk_level == RiskLevel.CRITICAL:
            decision = AuthenticationDecision.PERMANENT_BLOCK
            reasoning = "Critical risk level - session terminated"
            next_delay = 0
        elif risk_assessment.risk_level == RiskLevel.HIGH:
            decision = AuthenticationDecision.STEP_UP_AUTH
            reasoning = "High risk level - additional verification required"
            next_delay = 0
        elif risk_assessment.risk_level == RiskLevel.MEDIUM:
            decision = AuthenticationDecision.CHALLENGE
            reasoning = "Medium risk level - soft challenge"
            next_delay = 60
        # Add compliance information
        if banking_decision:
            profile_stage = banking_decision.get('profile_stage', 'unknown')
            if profile_stage in ['cold_start', 'observation']:
                compliance_flags.append(f"PROFILE_STAGE_{profile_stage.upper()}")
        self.logger.info(f"Policy Decision: {decision}, Risk Level: {risk_assessment.risk_level}, Score: {risk_assessment.overall_risk_score}")
        return PolicyDecision(
            user_id=risk_assessment.user_id,
            session_id=risk_assessment.session_id,
            decision=decision,
            risk_assessment=risk_assessment,
            reasoning=reasoning,
            compliance_flags=compliance_flags,
            next_verification_delay=next_delay,
            session_constraints=constraints,
            audit_trail=audit_trail or [],
            processing_time_ms=2.1
        )
    
    async def _stage7_response_generation(self, request_id: str, layer1_result: Layer1Result, 
                                        layer2_result: Optional[Layer2Result], drift_result: DriftResult,
                                        policy_decision: PolicyDecision, stage_timings: Dict[ProcessingStage, float],
                                        audit_trail: List[Dict[str, Any]], start_time: float) -> AuthenticationResponse:
        """Stage 7: Generate final response with full explainability"""
        total_time = (time.perf_counter() - start_time) * 1000
        
        explanation = {
            "decision_confidence": policy_decision.risk_assessment.confidence,
            "top_risk_factors": policy_decision.risk_assessment.risk_factors[:5],
            "human_readable_explanation": f"Authentication decision: {policy_decision.decision.value.upper()} (Risk Level: {policy_decision.risk_assessment.risk_level.value.upper()}). {policy_decision.reasoning}"
        }
        
        return AuthenticationResponse(
            session_id=policy_decision.session_id,
            user_id=policy_decision.user_id,
            request_id=request_id,
            decision=policy_decision.decision,
            risk_level=policy_decision.risk_assessment.risk_level,
            risk_score=policy_decision.risk_assessment.overall_risk_score,
            confidence=policy_decision.risk_assessment.confidence,
            layer1_result=layer1_result,
            layer2_result=layer2_result,
            drift_result=drift_result,
            policy_decision=policy_decision,
            total_processing_time_ms=total_time,
            stage_timings=stage_timings,
            explanation=explanation,
            audit_trail=audit_trail,
            timestamp=datetime.utcnow(),
            next_verification_delay=policy_decision.next_verification_delay
        )
    
    # Component Initialization Methods (Placeholders)
    
    async def _init_input_validator(self):
        """Initialize input validation component"""
        # TODO: Implement actual input validator
        self.logger.info("Input validator initialized (mock)")
    
    async def _init_behavioral_preprocessor(self):
        """Initialize behavioral data preprocessor"""
        # TODO: Implement actual preprocessor
        self.logger.info("Behavioral preprocessor initialized (mock)")
    
    async def _init_layer1_faiss_engine(self):
        """Initialize FAISS Layer 1 engine"""
        # TODO: Implement actual FAISS engine initialization
        self.logger.info("Layer 1 FAISS engine initialized (mock)")
    
    async def _init_layer2_adaptive_engine(self):
        """Initialize Layer 2 adaptive engine"""
        self.layer2_verifier = Layer2Verifier()
        await self.layer2_verifier.initialize()
        self.logger.info("Layer 2 adaptive engine initialized (production)")
    
    async def _init_drift_analysis_engine(self):
        """Initialize drift detection engine"""
        # TODO: Implement actual drift detection initialization
        self.logger.info("Drift analysis engine initialized (mock)")
    
    async def _init_risk_assessment_engine(self):
        """Initialize risk assessment engine"""
        # TODO: Implement actual risk assessment initialization
        self.logger.info("Risk assessment engine initialized (mock)")
    
    async def _init_policy_decision_engine(self):
        """Initialize policy decision engine"""
        # TODO: Implement actual policy engine initialization
        self.logger.info("Policy decision engine initialized (mock)")
    
    async def _start_background_tasks(self):
        """Start background monitoring and maintenance tasks"""
        # TODO: Implement background tasks
        self.logger.info("Background tasks started")
    
    async def _initialize_session_components(self, session_id: str, user_id: str, context: SessionContext):
        """Initialize session-specific ML components"""
        # TODO: Load user profiles, initialize session-specific models
        pass
    
    async def _finalize_session_analysis(self, session_state: SessionMLState, final_decision: str):
        """Perform final analysis and save session insights"""
        # TODO: Generate session summary, update user profiles
        pass
    
    async def _cleanup_session_resources(self, session_id: str):
        """Cleanup session-specific resources"""
        # TODO: Clear caches, release memory
        pass
    
    async def _shutdown_components(self):
        """Shutdown all ML components"""
        # TODO: Properly shutdown all components
        pass

# Global ML-Engine instance
ml_engine = BankingMLEngine()

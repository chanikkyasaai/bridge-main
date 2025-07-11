"""
BRIDGE ML-Engine Main Orchestrator
Coordinates all ML components for real-time behavioral authentication
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import defaultdict, deque

from ml_engine.config import CONFIG
from ml_engine.utils.behavioral_vectors import (
    BehavioralEvent, BehavioralVector, BehavioralVectorProcessor, 
    EventBuffer, BehavioralEncoder
)
from faiss.verifier.layer1_verifier import FAISSVerifier, VerificationResult as L1Result
from level2_adapter.layer2_verifier import Layer2Verifier, Layer2Result
from drift_detection import (
    BehavioralDriftDetector, DriftDetectionResult, UserBehaviorProfile
)
from policy_orchestrator import (
    PolicyOrchestrator, PolicyDecision, ExplainabilityReport, 
    SessionAction, RiskLevel
)

logger = logging.getLogger(__name__)

@dataclass
class AuthenticationRequest:
    """Authentication request structure"""
    session_id: str
    user_id: str
    events: List[BehavioralEvent]
    context: Dict[str, Any]
    timestamp: datetime
    require_explanation: bool = False

@dataclass
class AuthenticationResponse:
    """Complete authentication response"""
    session_id: str
    user_id: str
    decision: SessionAction
    risk_level: RiskLevel
    risk_score: float
    confidence: float
    processing_time_ms: float
    
    # Component results
    l1_result: L1Result
    l2_result: Optional[L2Result]
    drift_result: Optional[DriftDetectionResult]
    policy_decision: PolicyDecision
    
    # Optional explainability
    explanation: Optional[ExplainabilityReport]
    
    timestamp: datetime
    next_verification_delay: int

@dataclass
class MLEngineStats:
    """ML Engine performance statistics"""
    total_requests: int
    successful_authentications: int
    blocked_sessions: int
    average_processing_time: float
    l1_accuracy: float
    l2_accuracy: float
    drift_detections: int
    uptime_hours: float

class SessionManager:
    """Manages active sessions and their states"""
    
    def __init__(self, max_sessions: int = 10000):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.max_sessions = max_sessions
        self.lock = threading.RLock()
        
    def create_session(self, session_id: str, user_id: str, context: Dict[str, Any]):
        """Create new session"""
        with self.lock:
            if len(self.sessions) >= self.max_sessions:
                # Remove oldest session
                oldest_session = min(
                    self.sessions.items(),
                    key=lambda x: x[1]['created_at']
                )[0]
                del self.sessions[oldest_session]
                
            self.sessions[session_id] = {
                'user_id': user_id,
                'created_at': datetime.now(),
                'last_activity': datetime.now(),
                'context': context,
                'risk_history': deque(maxlen=100),
                'event_count': 0,
                'authentication_count': 0
            }
    
    def update_session(self, session_id: str, risk_score: float, decision: SessionAction):
        """Update session with new authentication result"""
        with self.lock:
            if session_id in self.sessions:
                session = self.sessions[session_id]
                session['last_activity'] = datetime.now()
                session['risk_history'].append((risk_score, datetime.now()))
                session['authentication_count'] += 1
                session['last_decision'] = decision
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session info"""
        with self.lock:
            return self.sessions.get(session_id)
    
    def cleanup_expired_sessions(self, max_age_hours: int = 24):
        """Clean up expired sessions"""
        with self.lock:
            current_time = datetime.now()
            expired_sessions = [
                session_id for session_id, session_data in self.sessions.items()
                if (current_time - session_data['last_activity']).total_seconds() > max_age_hours * 3600
            ]
            
            for session_id in expired_sessions:
                del self.sessions[session_id]
                
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

class BRIDGEMLEngine:
    """Main BRIDGE ML Engine orchestrator"""
    
    def __init__(self):
        # Initialize components
        self.vector_processor = BehavioralVectorProcessor()
        self.l1_verifier = FAISSVerifier()
        self.l2_verifier = Layer2Verifier()
        self.drift_detector = BehavioralDriftDetector()
        self.policy_orchestrator = PolicyOrchestrator()
        
        # Session and performance management
        self.session_manager = SessionManager()
        self.stats = MLEngineStats(
            total_requests=0,
            successful_authentications=0,
            blocked_sessions=0,
            average_processing_time=0.0,
            l1_accuracy=0.0,
            l2_accuracy=0.0,
            drift_detections=0,
            uptime_hours=0.0
        )
        
        # Performance tracking
        self.processing_times = deque(maxlen=1000)
        self.start_time = datetime.now()
        
        # Thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=CONFIG.MAX_CONCURRENT_SESSIONS // 10)
        
        # Component readiness
        self.is_ready = False
        
        # Background tasks
        self._background_tasks = []
        
        logger.info("BRIDGE ML Engine initialized")

    async def initialize(self):
        """Initialize all ML components"""
        logger.info("Initializing BRIDGE ML Engine components...")
        
        try:
            # Initialize vector processor
            await self.vector_processor.initialize()
            logger.info("✓ Behavioral vector processor initialized")
            
            # Initialize FAISS verifier
            await self.l1_verifier.initialize()
            logger.info("✓ Layer 1 FAISS verifier initialized")
            
            # Initialize Layer 2 verifier
            await self.l2_verifier.initialize()
            logger.info("✓ Layer 2 transformer+GNN verifier initialized")
            
            # Initialize drift detector
            await self.drift_detector.initialize()
            logger.info("✓ Behavioral drift detector initialized")
            
            # Start background tasks
            self._start_background_tasks()
            
            self.is_ready = True
            logger.info("🚀 BRIDGE ML Engine fully initialized and ready")
            
        except Exception as e:
            logger.error(f"Failed to initialize ML Engine: {e}")
            raise

    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        
        # Session cleanup task
        async def session_cleanup_task():
            while True:
                try:
                    self.session_manager.cleanup_expired_sessions()
                    await asyncio.sleep(3600)  # Every hour
                except Exception as e:
                    logger.error(f"Session cleanup error: {e}")
                    await asyncio.sleep(60)
        
        # Model updates task
        async def model_update_task():
            while True:
                try:
                    await self.l1_verifier.update_models()
                    await self.l2_verifier.update_models()
                    await asyncio.sleep(3600)  # Every hour
                except Exception as e:
                    logger.error(f"Model update error: {e}")
                    await asyncio.sleep(300)
        
        # Statistics update task
        async def stats_update_task():
            while True:
                try:
                    self._update_statistics()
                    await asyncio.sleep(60)  # Every minute
                except Exception as e:
                    logger.error(f"Stats update error: {e}")
                    await asyncio.sleep(60)
        
        # Schedule background tasks
        self._background_tasks.extend([
            asyncio.create_task(session_cleanup_task()),
            asyncio.create_task(model_update_task()),
            asyncio.create_task(stats_update_task())
        ])

    async def authenticate(self, request: AuthenticationRequest) -> AuthenticationResponse:
        """Main authentication pipeline"""
        
        if not self.is_ready:
            raise RuntimeError("ML Engine not initialized")
        
        start_time = time.time()
        
        try:
            # Create or update session
            session = self.session_manager.get_session(request.session_id)
            if not session:
                self.session_manager.create_session(
                    request.session_id, 
                    request.user_id, 
                    request.context
                )
            
            # Update statistics
            self.stats.total_requests += 1
            
            # Process behavioral vectors
            vectors = await self._process_behavioral_vectors(request.events)
            
            # Layer 1: FAISS-based verification
            l1_result = await self._run_layer1_verification(
                vectors, request.user_id, request.session_id
            )
            
            # Determine if Layer 2 is needed
            l2_result = None
            if self._needs_layer2_verification(l1_result):
                l2_result = await self._run_layer2_verification(
                    vectors, request.events, request.context, request.session_id
                )
            
            # Drift detection
            drift_result = await self._run_drift_detection(
                vectors, request.user_id
            )
            
            # Policy decision
            policy_decision = self.policy_orchestrator.make_policy_decision(
                request.session_id, l1_result, l2_result or self._create_default_l2_result(),
                drift_result, request.context
            )
            
            # Generate explanation if requested
            explanation = None
            if request.require_explanation:
                explanation = self.policy_orchestrator.generate_explainability_report(
                    policy_decision, l1_result, l2_result or self._create_default_l2_result(), drift_result
                )
            
            # Update session
            self.session_manager.update_session(
                request.session_id, policy_decision.risk_score, policy_decision.action
            )
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            self.processing_times.append(processing_time)
            
            # Update success/block statistics
            if policy_decision.action in [SessionAction.CONTINUE, SessionAction.MONITOR]:
                self.stats.successful_authentications += 1
            elif policy_decision.action in [SessionAction.TEMPORARY_BLOCK, SessionAction.PERMANENT_BLOCK]:
                self.stats.blocked_sessions += 1
            
            # Create response
            response = AuthenticationResponse(
                session_id=request.session_id,
                user_id=request.user_id,
                decision=policy_decision.action,
                risk_level=policy_decision.risk_level,
                risk_score=policy_decision.risk_score,
                confidence=policy_decision.confidence,
                processing_time_ms=processing_time,
                l1_result=l1_result,
                l2_result=l2_result,
                drift_result=drift_result,
                policy_decision=policy_decision,
                explanation=explanation,
                timestamp=datetime.now(),
                next_verification_delay=policy_decision.next_verification_delay
            )
            
            logger.info(
                f"Authentication completed: {request.session_id} -> {policy_decision.action.value} "
                f"(risk: {policy_decision.risk_score:.3f}, time: {processing_time:.1f}ms)"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Authentication error for session {request.session_id}: {e}")
            
            # Return safe fallback
            return self._create_fallback_response(request, time.time() - start_time)

    async def _process_behavioral_vectors(self, events: List[BehavioralEvent]) -> List[BehavioralVector]:
        """Process behavioral events into vectors"""
        try:
            return await self.vector_processor.process_events(events)
        except Exception as e:
            logger.error(f"Vector processing error: {e}")
            raise

    async def _run_layer1_verification(
        self, 
        vectors: List[BehavioralVector], 
        user_id: str, 
        session_id: str
    ) -> L1Result:
        """Run Layer 1 FAISS verification"""
        try:
            return await self.l1_verifier.verify(vectors, user_id, session_id)
        except Exception as e:
            logger.error(f"Layer 1 verification error: {e}")
            raise

    def _needs_layer2_verification(self, l1_result: L1Result) -> bool:
        """Determine if Layer 2 verification is needed"""
        return (
            l1_result.confidence_level in ['medium', 'low'] or
            l1_result.similarity_score < CONFIG.L1_HIGH_CONFIDENCE_THRESHOLD
        )

    async def _run_layer2_verification(
        self, 
        vectors: List[BehavioralVector],
        events: List[BehavioralEvent],
        context: Dict[str, Any],
        session_id: str
    ) -> Layer2Result:
        """Run Layer 2 transformer+GNN verification"""
        try:
            return await self.l2_verifier.verify(vectors, events, context, session_id)
        except Exception as e:
            logger.error(f"Layer 2 verification error: {e}")
            raise

    async def _run_drift_detection(
        self, 
        vectors: List[BehavioralVector], 
        user_id: str
    ) -> DriftDetectionResult:
        """Run behavioral drift detection"""
        try:
            return await self.drift_detector.detect_drift(vectors, user_id)
        except Exception as e:
            logger.error(f"Drift detection error: {e}")
            # Return no-drift result on error
            return DriftDetectionResult(
                user_id=user_id,
                drift_detected=False,
                drift_type="none",
                drift_magnitude=0.0,
                confidence=0.0,
                affected_features=[],
                recommendation="none",
                timestamp=datetime.now(),
                metadata={"error": str(e)}
            )

    def _create_default_l2_result(self) -> Layer2Result:
        """Create default Layer 2 result when not run"""
        return Layer2Result(
            session_id="default",
            user_id="default",
            transformer_confidence=0.5,
            gnn_anomaly_score=0.0,
            contextual_score=0.5,
            session_graph_embedding=np.zeros(CONFIG.GRAPH_EMBEDDING_DIM),
            decision_factors={},
            transformer_decision="neutral",
            gnn_decision="normal",
            final_recommendation="continue",
            processing_time_ms=0.0,
            metadata={"default": True}
        )

    def _create_fallback_response(
        self, 
        request: AuthenticationRequest, 
        elapsed_time: float
    ) -> AuthenticationResponse:
        """Create safe fallback response on errors"""
        
        # Create minimal results
        l1_result = L1Result(
            user_id=request.user_id,
            session_id=request.session_id,
            similarity_score=0.3,  # Conservative
            confidence_level="low",
            matched_profile_id="fallback",
            matched_mode="unknown",
            decision="escalate",
            processing_time_ms=elapsed_time * 1000,
            metadata={"fallback": True}
        )
        
        policy_decision = PolicyDecision(
            session_id=request.session_id,
            user_id=request.user_id,
            action=SessionAction.STEP_UP_AUTH,  # Safe fallback
            risk_level=RiskLevel.MEDIUM,
            risk_score=0.6,
            confidence=0.3,
            reasoning=["Fallback due to processing error"],
            evidence={},
            timestamp=datetime.now(),
            expires_at=datetime.now() + timedelta(minutes=5),
            monitoring_level="intensive",
            next_verification_delay=30
        )
        
        return AuthenticationResponse(
            session_id=request.session_id,
            user_id=request.user_id,
            decision=SessionAction.STEP_UP_AUTH,
            risk_level=RiskLevel.MEDIUM,
            risk_score=0.6,
            confidence=0.3,
            processing_time_ms=elapsed_time * 1000,
            l1_result=l1_result,
            l2_result=None,
            drift_result=None,
            policy_decision=policy_decision,
            explanation=None,
            timestamp=datetime.now(),
            next_verification_delay=30
        )

    def _update_statistics(self):
        """Update engine statistics"""
        if self.processing_times:
            self.stats.average_processing_time = np.mean(list(self.processing_times))
        
        self.stats.uptime_hours = (datetime.now() - self.start_time).total_seconds() / 3600

    async def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive user profile"""
        
        # Get FAISS profile
        faiss_profile = await self.l1_verifier.get_user_profile(user_id)
        
        # Get drift profile
        drift_profile = await self.drift_detector.get_user_profile(user_id)
        
        # Get trust score
        trust_score = self.policy_orchestrator.user_trust_scores.get(user_id, 0.5)
        
        return {
            "user_id": user_id,
            "faiss_profile": asdict(faiss_profile) if faiss_profile else None,
            "drift_profile": asdict(drift_profile) if drift_profile else None,
            "trust_score": trust_score,
            "last_updated": datetime.now().isoformat()
        }

    async def train_user_model(
        self, 
        user_id: str, 
        training_vectors: List[BehavioralVector],
        labels: List[str]
    ):
        """Train user-specific models"""
        
        # Train FAISS profile
        await self.l1_verifier.train_user_profile(user_id, training_vectors, labels)
        
        # Update drift baseline
        await self.drift_detector.update_user_baseline(user_id, training_vectors)
        
        logger.info(f"User model trained for {user_id}")

    def get_engine_stats(self) -> MLEngineStats:
        """Get engine performance statistics"""
        self._update_statistics()
        return self.stats

    def get_active_sessions(self) -> Dict[str, Any]:
        """Get active session summary"""
        return {
            "total_sessions": len(self.session_manager.sessions),
            "max_sessions": self.session_manager.max_sessions,
            "sessions": {
                session_id: {
                    "user_id": data["user_id"],
                    "created_at": data["created_at"].isoformat(),
                    "last_activity": data["last_activity"].isoformat(),
                    "event_count": data["event_count"],
                    "authentication_count": data["authentication_count"]
                }
                for session_id, data in list(self.session_manager.sessions.items())[:10]  # First 10
            }
        }

    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check"""
        
        health = {
            "status": "healthy" if self.is_ready else "initializing",
            "timestamp": datetime.now().isoformat(),
            "uptime_hours": self.stats.uptime_hours,
            "components": {
                "vector_processor": "ready",
                "l1_verifier": "ready",
                "l2_verifier": "ready", 
                "drift_detector": "ready",
                "policy_orchestrator": "ready"
            },
            "performance": {
                "total_requests": self.stats.total_requests,
                "average_processing_time_ms": self.stats.average_processing_time,
                "active_sessions": len(self.session_manager.sessions)
            },
            "memory_usage": {
                "session_count": len(self.session_manager.sessions),
                "cache_sizes": {
                    "policy_cache": len(self.policy_orchestrator.policy_cache),
                    "processing_times": len(self.processing_times)
                }
            }
        }
        
        return health

    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down BRIDGE ML Engine...")
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        # Save models and profiles
        await self.l1_verifier.save_models()
        await self.l2_verifier.save_models()
        await self.drift_detector.save_profiles()
        
        logger.info("BRIDGE ML Engine shutdown complete")

# Global ML Engine instance
ml_engine = BRIDGEMLEngine()

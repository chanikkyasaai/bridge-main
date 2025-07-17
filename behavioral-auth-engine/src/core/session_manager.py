"""
Session lifecycle management for behavioral authentication.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any
from dataclasses import dataclass
from enum import Enum
import uuid
import time

from src.data.models import (
    UserProfile, BehavioralVector, SessionPhase, 
    AuthenticationRequest, AuthenticationResponse,
    AuthenticationDecision, RiskLevel
)
from src.core.vector_store import VectorStoreInterface
from src.config.settings import get_settings
from src.utils.constants import *

logger = logging.getLogger(__name__)


class SessionStatus(str, Enum):
    """Session status states."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    EXPIRED = "expired"
    TERMINATED = "terminated"


@dataclass
class SessionContext:
    """Context information for an active session."""
    user_id: str
    session_id: str
    created_at: datetime
    last_activity: datetime
    status: SessionStatus
    device_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    risk_scores: List[float] = None
    decision_history: List[AuthenticationDecision] = None
    
    def __post_init__(self):
        if self.risk_scores is None:
            self.risk_scores = []
        if self.decision_history is None:
            self.decision_history = []
    
    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = datetime.utcnow()
    
    def add_risk_score(self, score: float) -> None:
        """Add a risk score to the session history."""
        self.risk_scores.append(score)
        # Keep only last 50 scores for memory efficiency
        if len(self.risk_scores) > 50:
            self.risk_scores = self.risk_scores[-50:]
    
    def add_decision(self, decision: AuthenticationDecision) -> None:
        """Add an authentication decision to the session history."""
        self.decision_history.append(decision)
        # Keep only last 50 decisions for memory efficiency
        if len(self.decision_history) > 50:
            self.decision_history = self.decision_history[-50:]
    
    def is_expired(self, timeout_minutes: int = 30) -> bool:
        """Check if session has expired."""
        if self.status in [SessionStatus.EXPIRED, SessionStatus.TERMINATED]:
            return True
        
        timeout_threshold = self.last_activity + timedelta(minutes=timeout_minutes)
        return datetime.utcnow() > timeout_threshold
    
    def get_average_risk_score(self) -> float:
        """Get average risk score for the session."""
        if not self.risk_scores:
            return 0.0
        return sum(self.risk_scores) / len(self.risk_scores)


class SessionManager:
    """Manages user sessions and their lifecycle."""
    
    def __init__(self, vector_store: VectorStoreInterface):
        self.vector_store = vector_store
        self.settings = get_settings()
        self.active_sessions: Dict[str, SessionContext] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
    
    async def start(self) -> None:
        """Start the session manager and cleanup task."""
        self._cleanup_task = asyncio.create_task(self._cleanup_expired_sessions())
    
    async def stop(self) -> None:
        """Stop the session manager and cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
    
    async def create_session(
        self, 
        user_id: str, 
        device_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> str:
        """Create a new session for a user."""
        session_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        session_context = SessionContext(
            user_id=user_id,
            session_id=session_id,
            created_at=now,
            last_activity=now,
            status=SessionStatus.ACTIVE,
            device_id=device_id,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        async with self._lock:
            self.active_sessions[session_id] = session_context
        
        return session_id
    
    async def get_session(self, session_id: str) -> Optional[SessionContext]:
        """Get session context by session ID."""
        async with self._lock:
            session = self.active_sessions.get(session_id)
            if session and session.status == SessionStatus.ACTIVE and session.is_expired():
                session.status = SessionStatus.EXPIRED
                return session
            return session
    
    def get_session_context(self, session_id: str) -> Optional[SessionContext]:
        """Get session context by session ID (synchronous alias)."""
        # This is a synchronous alias for get_session for API compatibility
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If in async context, we need to handle this differently
                return self.active_sessions.get(session_id)
            else:
                return loop.run_until_complete(self.get_session(session_id))
        except:
            # Fallback to direct access
            return self.active_sessions.get(session_id)
    
    async def update_session_activity(self, session_id: str) -> bool:
        """Update session activity timestamp."""
        async with self._lock:
            session = self.active_sessions.get(session_id)
            if session and session.status == SessionStatus.ACTIVE:
                session.update_activity()
                return True
            return False
    
    async def terminate_session(self, session_id: str) -> bool:
        """Terminate a specific session."""
        async with self._lock:
            session = self.active_sessions.get(session_id)
            if session:
                session.status = SessionStatus.TERMINATED
                return True
            return False
    
    async def terminate_user_sessions(self, user_id: str) -> int:
        """Terminate all sessions for a user."""
        terminated_count = 0
        async with self._lock:
            for session in self.active_sessions.values():
                if session.user_id == user_id and session.status == SessionStatus.ACTIVE:
                    session.status = SessionStatus.TERMINATED
                    terminated_count += 1
        return terminated_count
    
    async def get_user_sessions(self, user_id: str) -> List[SessionContext]:
        """Get all active sessions for a user."""
        user_sessions = []
        async with self._lock:
            for session in self.active_sessions.values():
                if session.user_id == user_id and session.status == SessionStatus.ACTIVE:
                    user_sessions.append(session)
        return user_sessions
    
    async def process_authentication_request(
        self, 
        request: AuthenticationRequest
    ) -> AuthenticationResponse:
        """Process an authentication request within session context."""
        start_time = time.time()
        
        # Get or create session
        session = await self.get_session(request.session_id)
        if not session:
            # Create new session if none exists
            await self.create_session(
                request.user_id,
                request.device_id,
                request.ip_address,
                request.user_agent
            )
            session = await self.get_session(request.session_id)
        
        # Update session activity
        await self.update_session_activity(request.session_id)
        
        # Get user profile
        user_profile = await self.vector_store.get_user_profile(request.user_id)
        if not user_profile:
            # Create new user profile
            user_profile = UserProfile(user_id=request.user_id)
        
        # Determine session phase and make authentication decision
        decision, risk_level, risk_score, confidence, factors, similarity_scores = await self._make_authentication_decision(
            request, session, user_profile
        )
        
        # Update session history
        session.add_risk_score(risk_score)
        session.add_decision(decision)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        processing_time = max(processing_time, 0.1)  # Ensure minimum processing time
        
        return AuthenticationResponse(
            request_id=request.request_id,
            user_id=request.user_id,
            decision=decision,
            risk_level=risk_level,
            risk_score=risk_score,
            confidence=confidence,
            processing_time_ms=processing_time,
            decision_factors=factors,
            similarity_scores=similarity_scores,
            session_phase=user_profile.current_phase,
            session_count=user_profile.session_count
        )
    
    async def _make_authentication_decision(
        self, 
        request: AuthenticationRequest,
        session: SessionContext,
        user_profile: UserProfile
    ) -> tuple:
        """Make authentication decision based on session context and user profile."""
        
        # Initialize default values
        risk_score = 0.0
        confidence = 1.0
        decision_factors = []
        similarity_scores = {}
        
        # Determine session phase
        user_profile.update_phase()
        
        # Learning phase logic
        if user_profile.current_phase == SessionPhase.LEARNING:
            decision = AuthenticationDecision.LEARN
            risk_level = RiskLevel.LOW
            risk_score = 0.1  # Very low risk during learning
            decision_factors.append("Learning phase - collecting behavioral data")
            
        # Gradual risk phase logic
        elif user_profile.current_phase == SessionPhase.GRADUAL_RISK:
            # Basic pattern matching with lenient thresholds
            if user_profile.baseline_vector and len(user_profile.recent_vectors) >= 3:
                # Simple similarity check
                risk_score = await self._calculate_basic_risk_score(request, user_profile)
                confidence = min(0.8, user_profile.session_count / 10.0)  # Gradual confidence increase
                
                if risk_score < 0.3:
                    decision = AuthenticationDecision.ALLOW
                    risk_level = RiskLevel.LOW
                    decision_factors.append("Gradual risk phase - patterns match baseline")
                elif risk_score < 0.7:
                    decision = AuthenticationDecision.CHALLENGE
                    risk_level = RiskLevel.MEDIUM
                    decision_factors.append("Gradual risk phase - moderate deviation detected")
                else:
                    decision = AuthenticationDecision.BLOCK
                    risk_level = RiskLevel.HIGH
                    decision_factors.append("Gradual risk phase - significant deviation detected")
            else:
                decision = AuthenticationDecision.LEARN
                risk_level = RiskLevel.LOW
                risk_score = 0.2
                decision_factors.append("Gradual risk phase - insufficient data, continue learning")
        
        # Full authentication phase logic
        else:
            if user_profile.baseline_vector:
                risk_score = await self._calculate_advanced_risk_score(request, user_profile, session)
                confidence = min(0.95, user_profile.session_count / 20.0)  # High confidence
                
                threshold = user_profile.risk_threshold
                
                if risk_score < threshold * 0.5:
                    decision = AuthenticationDecision.ALLOW
                    risk_level = RiskLevel.LOW
                    decision_factors.append("Full auth - behavioral patterns match user profile")
                elif risk_score < threshold:
                    decision = AuthenticationDecision.CHALLENGE
                    risk_level = RiskLevel.MEDIUM
                    decision_factors.append("Full auth - moderate risk detected")
                else:
                    decision = AuthenticationDecision.BLOCK
                    risk_level = RiskLevel.HIGH
                    decision_factors.append("Full auth - high risk behavioral anomaly")
                    
                # Add similarity scores for transparency
                similarity_scores["baseline_similarity"] = 1.0 - risk_score
            else:
                decision = AuthenticationDecision.LEARN
                risk_level = RiskLevel.MEDIUM
                risk_score = 0.3
                decision_factors.append("Full auth - rebuilding user profile")
        
        # Session-specific factors
        if session:
            avg_session_risk = session.get_average_risk_score()
            if avg_session_risk > 0.7:
                decision_factors.append("Session shows consistently high risk")
                if decision == AuthenticationDecision.ALLOW:
                    decision = AuthenticationDecision.CHALLENGE
                    risk_level = RiskLevel.MEDIUM
        
        return decision, risk_level, risk_score, confidence, decision_factors, similarity_scores
    
    async def _calculate_basic_risk_score(
        self, 
        request: AuthenticationRequest, 
        user_profile: UserProfile
    ) -> float:
        """Calculate basic risk score for gradual risk phase."""
        # This is a simplified version - would be enhanced in production
        if not user_profile.baseline_vector or not user_profile.recent_vectors:
            return 0.5  # Medium risk if no baseline
        
        # Simple deviation calculation
        # In real implementation, this would extract features from request.behavioral_data
        # For now, we'll use a placeholder calculation
        return min(0.8, len(user_profile.recent_vectors) * 0.1)  # Placeholder
    
    async def _calculate_advanced_risk_score(
        self, 
        request: AuthenticationRequest, 
        user_profile: UserProfile, 
        session: SessionContext
    ) -> float:
        """Calculate advanced risk score for full authentication phase."""
        # This is a simplified version - would be enhanced with ML models in production
        base_risk = await self._calculate_basic_risk_score(request, user_profile)
        
        # Session-based adjustments
        if session:
            recent_decisions = session.decision_history[-5:]  # Last 5 decisions
            block_count = sum(1 for d in recent_decisions if d == AuthenticationDecision.BLOCK)
            
            if block_count >= 2:
                base_risk += 0.2  # Increase risk if recent blocks
        
        return min(1.0, base_risk)
    
    async def _cleanup_expired_sessions(self) -> None:
        """Background task to clean up expired sessions."""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                current_time = datetime.utcnow()
                expired_sessions = []
                
                async with self._lock:
                    for session_id, session in self.active_sessions.items():
                        if session.is_expired():
                            expired_sessions.append(session_id)
                    
                    # Remove expired sessions
                    for session_id in expired_sessions:
                        del self.active_sessions[session_id]
                
                if expired_sessions:
                    print(f"Cleaned up {len(expired_sessions)} expired sessions")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in session cleanup: {e}")
    
    async def add_behavioral_event(self, session_id: str, event_data: Dict[str, Any]) -> bool:
        """Add a behavioral event to the session history"""
        try:
            async with self._lock:
                if session_id in self.active_sessions:
                    session = self.active_sessions[session_id]
                    
                    # Add event to session history (you could extend SessionContext to include this)
                    # For now, just update last activity
                    session.last_activity = datetime.utcnow()
                    
                    # Log the event
                    logger.debug(f"Added behavioral event to session {session_id}")
                    return True
                else:
                    logger.warning(f"Session {session_id} not found for behavioral event")
                    return False
        except Exception as e:
            logger.error(f"Failed to add behavioral event to session {session_id}: {e}")
            return False
    
    async def get_session_statistics(self) -> Dict[str, Any]:
        """Get statistics about active sessions."""
        async with self._lock:
            total_sessions = len(self.active_sessions)
            active_count = sum(1 for s in self.active_sessions.values() if s.status == SessionStatus.ACTIVE)
            expired_count = sum(1 for s in self.active_sessions.values() if s.status == SessionStatus.EXPIRED)
            terminated_count = sum(1 for s in self.active_sessions.values() if s.status == SessionStatus.TERMINATED)
            
            # User distribution
            user_session_counts = {}
            for session in self.active_sessions.values():
                user_id = session.user_id
                user_session_counts[user_id] = user_session_counts.get(user_id, 0) + 1
            
            return {
                "total_sessions": total_sessions,
                "active_sessions": active_count,
                "expired_sessions": expired_count,
                "terminated_sessions": terminated_count,
                "unique_users": len(user_session_counts),
                "max_sessions_per_user": max(user_session_counts.values()) if user_session_counts else 0,
                "avg_sessions_per_user": sum(user_session_counts.values()) / len(user_session_counts) if user_session_counts else 0
            }

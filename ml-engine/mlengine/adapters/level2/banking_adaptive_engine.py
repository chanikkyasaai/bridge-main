"""
Banking Adaptive Engine - Layer 2 Context-Aware Verification
Simple wrapper for existing Layer2 functionality
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)

class BankingAdaptiveEngine:
    """Simple adaptive engine wrapper for banking behavioral verification"""
    
    def __init__(self):
        self.is_initialized = False
        
        # Performance stats
        self.stats = {
            "verifications_performed": 0,
            "average_verification_time_ms": 0.0,
            "decisions": {
                "continue": 0,
                "restrict": 0,
                "reauthenticate": 0,
                "block": 0
            }
        }
        
        logger.info("🧠 Banking Adaptive Engine initialized")
    
    async def initialize(self):
        """Initialize the adaptive engine"""
        try:
            # Simple initialization - in production this would initialize complex models
            self.is_initialized = True
            logger.info("✅ Banking Adaptive Engine ready")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Adaptive engine: {e}")
            raise
    
    async def verify_behavioral_context(self, 
                                      vectors: List[Any], 
                                      events: List[Any], 
                                      context: Dict[str, Any], 
                                      session_id: str,
                                      user_id: str) -> Dict[str, Any]:
        """Verify behavioral patterns with context awareness"""
        import time
        start_time = time.time()
        
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # Simple risk assessment based on context
            risk_score = self._calculate_simple_risk(context, len(events))
            
            # Make decision based on risk
            if risk_score < 0.3:
                decision = "continue"
            elif risk_score < 0.5:
                decision = "restrict"
            elif risk_score < 0.7:
                decision = "reauthenticate"
            else:
                decision = "block"
            
            # Update stats
            verification_time_ms = (time.time() - start_time) * 1000
            self.stats["verifications_performed"] += 1
            self.stats["average_verification_time_ms"] = (
                (self.stats["average_verification_time_ms"] * (self.stats["verifications_performed"] - 1) + verification_time_ms) /
                self.stats["verifications_performed"]
            )
            self.stats["decisions"][decision] += 1
            
            return {
                "session_id": session_id,
                "user_id": user_id,
                "decision": decision,
                "confidence": 1.0 - risk_score,
                "risk_score": risk_score,
                "gnn_anomaly": risk_score * 0.5,  # Simulated
                "context_score": 1.0 - risk_score * 0.8,
                "explanation": f"Risk assessment: {risk_score:.2f}, Decision: {decision}",
                "processing_time_ms": verification_time_ms,
                "metadata": {"simple_engine": True}
            }
            
        except Exception as e:
            logger.error(f"Error during adaptive verification: {e}")
            return {
                "session_id": session_id,
                "user_id": user_id,
                "decision": "block",
                "confidence": 0.0,
                "risk_score": 1.0,
                "explanation": f"Verification error: {e}",
                "processing_time_ms": (time.time() - start_time) * 1000,
                "error": True
            }
    
    def _calculate_simple_risk(self, context: Dict[str, Any], event_count: int) -> float:
        """Simple risk calculation"""
        risk = 0.2  # Base risk
        
        # Location risk
        location_risk = context.get("location_risk", 0.0)
        risk += location_risk * 0.3
        
        # Time of day risk
        time_of_day = context.get("time_of_day", "unknown")
        if time_of_day == "night":
            risk += 0.1
        
        # Device type risk
        device_type = context.get("device_type", "unknown")
        if device_type == "unknown":
            risk += 0.1
        
        # Event count risk (too few or too many events can be suspicious)
        if event_count < 3:
            risk += 0.1
        elif event_count > 100:
            risk += 0.05
        
        return min(1.0, max(0.0, risk))
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        return {
            "engine_type": "Banking Adaptive Engine",
            "initialized": self.is_initialized,
            "verifications_performed": self.stats["verifications_performed"],
            "average_verification_time_ms": round(self.stats["average_verification_time_ms"], 2),
            "decision_distribution": self.stats["decisions"]
        }
    
    async def clear_session(self, session_id: str):
        """Clear session data"""
        logger.debug(f"Cleared session {session_id}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        return {
            "status": "healthy" if self.is_initialized else "not_initialized",
            "last_check": datetime.now().isoformat()
        }

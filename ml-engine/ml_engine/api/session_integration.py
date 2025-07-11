"""
BRIDGE ML-Engine Session Lifecycle Integration

This module provides the integration layer between the backend session management
and the ML-engine, ensuring that ML processing is active for the entire session
lifecycle (from session start to session end).

Integration Points:
1. Session Start Hook -> ML-Engine Start Session
2. Behavioral Event Stream -> ML-Engine Processing
3. Session End Hook -> ML-Engine End Session
4. WebSocket Real-time Communication
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import aiohttp
from dataclasses import asdict

from core_ml_engine import (
    BankingMLEngine, ml_engine, BehavioralEvent, SessionContext, 
    AuthenticationResponse, AuthenticationDecision, RiskLevel
)

logger = logging.getLogger(__name__)

class MLEngineSessionIntegrator:
    """
    Integrates ML-Engine with backend session lifecycle management
    """
    
    def __init__(self, backend_host: str = "localhost", backend_port: int = 8000):
        self.backend_host = backend_host
        self.backend_port = backend_port
        self.ml_engine = ml_engine
        
        # Session callbacks
        self.session_start_callbacks: List[Callable] = []
        self.session_end_callbacks: List[Callable] = []
        self.authentication_callbacks: List[Callable] = []
        
        # WebSocket connections for real-time communication
        self.websocket_connections: Dict[str, Any] = {}
        
        self.logger = logging.getLogger(f"{__name__}.MLEngineSessionIntegrator")
    
    async def initialize(self) -> bool:
        """Initialize the ML-Engine and integration"""
        try:
            # Initialize ML-Engine
            if not await self.ml_engine.initialize():
                self.logger.error("Failed to initialize ML-Engine")
                return False
            
            # Register with backend session events
            await self._register_session_hooks()
            
            self.logger.info("ML-Engine Session Integration initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ML-Engine integration: {e}")
            return False
    
    async def handle_session_start(self, session_data: Dict[str, Any]) -> bool:
        """
        Handle session start event from backend
        
        Args:
            session_data: {
                "session_id": str,
                "user_id": str,
                "device_id": str,
                "phone": str,
                "device_type": str,
                "device_model": str,
                "os_version": str,
                "app_version": str,
                "network_type": str,
                "location_data": dict,
                "is_known_device": bool,
                "is_trusted_location": bool
            }
        """
        try:
            session_id = session_data["session_id"]
            user_id = session_data["user_id"]
            
            # Create session context for ML-Engine
            context = SessionContext(
                session_id=session_id,
                user_id=user_id,
                device_id=session_data.get("device_id", ""),
                session_start_time=datetime.utcnow(),
                last_activity=datetime.utcnow(),
                session_duration_minutes=0.0,
                device_type=session_data.get("device_type", "mobile"),
                device_model=session_data.get("device_model", "unknown"),
                os_version=session_data.get("os_version", "unknown"),
                app_version=session_data.get("app_version", "unknown"),
                network_type=session_data.get("network_type", "unknown"),
                location_data=session_data.get("location_data"),
                time_of_day=self._get_time_of_day(),
                usage_pattern="standard",  # TODO: Determine from user history
                interaction_frequency=0.0,
                typical_session_duration=0.0,  # TODO: Load from user profile
                is_known_device=session_data.get("is_known_device", False),
                is_trusted_location=session_data.get("is_trusted_location", False),
                recent_security_events=[],
                risk_flags=[]
            )
            
            # Start ML-Engine session
            success = await self.ml_engine.start_session(session_id, user_id, context)
            
            if success:
                self.logger.info(f"ML-Engine session started for {session_id}")
                
                # Run session start callbacks
                for callback in self.session_start_callbacks:
                    try:
                        await callback(session_id, user_id, context)
                    except Exception as e:
                        self.logger.error(f"Session start callback error: {e}")
                
                return True
            else:
                self.logger.error(f"Failed to start ML-Engine session for {session_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error handling session start: {e}")
            return False
    
    async def handle_behavioral_event(self, event_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Handle behavioral event from backend WebSocket
        
        Args:
            event_data: {
                "session_id": str,
                "user_id": str,
                "device_id": str,
                "timestamp": str,
                "event_type": str,
                "features": dict,
                "raw_metadata": dict
            }
        
        Returns:
            Authentication response dict or None
        """
        try:
            # Convert to BehavioralEvent
            event = BehavioralEvent(
                timestamp=datetime.fromisoformat(event_data["timestamp"].replace('Z', '')),
                event_type=event_data["event_type"],
                features=event_data["features"],
                session_id=event_data["session_id"],
                user_id=event_data["user_id"],
                device_id=event_data.get("device_id", ""),
                raw_metadata=event_data.get("raw_metadata", {})
            )
            
            # Process through ML-Engine
            response = await self.ml_engine.process_behavioral_event(event)
            
            if response:
                # Convert to dictionary for JSON serialization
                response_dict = await self._convert_response_to_dict(response)
                
                # Handle authentication decision
                await self._handle_authentication_decision(response)
                
                # Run authentication callbacks
                for callback in self.authentication_callbacks:
                    try:
                        await callback(response)
                    except Exception as e:
                        self.logger.error(f"Authentication callback error: {e}")
                
                return response_dict
            else:
                self.logger.warning(f"No response from ML-Engine for event in session {event_data['session_id']}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error handling behavioral event: {e}")
            return None
    
    async def handle_session_end(self, session_data: Dict[str, Any]) -> bool:
        """
        Handle session end event from backend
        
        Args:
            session_data: {
                "session_id": str,
                "user_id": str,
                "final_decision": str,
                "session_duration": float,
                "total_events": int
            }
        """
        try:
            session_id = session_data["session_id"]
            final_decision = session_data.get("final_decision", "normal")
            
            # End ML-Engine session
            success = await self.ml_engine.end_session(session_id, final_decision)
            
            if success:
                self.logger.info(f"ML-Engine session ended for {session_id}")
                
                # Run session end callbacks
                for callback in self.session_end_callbacks:
                    try:
                        await callback(session_id, session_data)
                    except Exception as e:
                        self.logger.error(f"Session end callback error: {e}")
                
                # Clean up WebSocket connections
                if session_id in self.websocket_connections:
                    del self.websocket_connections[session_id]
                
                return True
            else:
                self.logger.error(f"Failed to end ML-Engine session for {session_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error handling session end: {e}")
            return False
    
    async def get_session_ml_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get ML-Engine status for a specific session"""
        try:
            return await self.ml_engine.get_session_status(session_id)
        except Exception as e:
            self.logger.error(f"Error getting ML session status: {e}")
            return None
    
    async def get_ml_engine_stats(self) -> Dict[str, Any]:
        """Get comprehensive ML-Engine statistics"""
        try:
            return await self.ml_engine.get_engine_stats()
        except Exception as e:
            self.logger.error(f"Error getting ML-Engine stats: {e}")
            return {}
    
    async def send_realtime_alert(self, session_id: str, alert_data: Dict[str, Any]):
        """Send real-time alert to backend for session"""
        try:
            # Send to backend via HTTP
            async with aiohttp.ClientSession() as session:
                url = f"http://{self.backend_host}:{self.backend_port}/api/v1/ml-alerts"
                alert_payload = {
                    "session_id": session_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "alert_type": alert_data.get("type", "security"),
                    "severity": alert_data.get("severity", "medium"),
                    "message": alert_data.get("message", ""),
                    "details": alert_data.get("details", {})
                }
                
                async with session.post(url, json=alert_payload) as response:
                    if response.status == 200:
                        self.logger.info(f"Real-time alert sent for session {session_id}")
                    else:
                        self.logger.error(f"Failed to send alert: {response.status}")
                        
        except Exception as e:
            self.logger.error(f"Error sending real-time alert: {e}")
    
    def register_session_start_callback(self, callback: Callable):
        """Register callback for session start events"""
        self.session_start_callbacks.append(callback)
    
    def register_session_end_callback(self, callback: Callable):
        """Register callback for session end events"""
        self.session_end_callbacks.append(callback)
    
    def register_authentication_callback(self, callback: Callable):
        """Register callback for authentication events"""
        self.authentication_callbacks.append(callback)
    
    async def _register_session_hooks(self):
        """Register ML-Engine with backend session hooks"""
        # This would typically register HTTP endpoints or message queue consumers
        # For now, we'll assume the backend will call these methods directly
        self.logger.info("ML-Engine session hooks registered")
    
    async def _handle_authentication_decision(self, response: AuthenticationResponse):
        """Handle authentication decision and take appropriate actions"""
        try:
            session_id = response.session_id
            decision = response.decision
            
            # Handle different authentication decisions
            if decision == AuthenticationDecision.PERMANENT_BLOCK:
                await self.send_realtime_alert(session_id, {
                    "type": "security_block",
                    "severity": "critical",
                    "message": "Session permanently blocked due to high risk behavior",
                    "details": {
                        "risk_score": response.risk_score,
                        "risk_level": response.risk_level.value
                    }
                })
                
            elif decision == AuthenticationDecision.TEMPORARY_BLOCK:
                await self.send_realtime_alert(session_id, {
                    "type": "security_block",
                    "severity": "high",
                    "message": "Session temporarily blocked",
                    "details": {
                        "risk_score": response.risk_score,
                        "block_duration": response.next_verification_delay
                    }
                })
                
            elif decision == AuthenticationDecision.STEP_UP_AUTH:
                await self.send_realtime_alert(session_id, {
                    "type": "step_up_auth",
                    "severity": "medium",
                    "message": "Step-up authentication required",
                    "details": {
                        "risk_score": response.risk_score,
                        "auth_method": "biometric_or_pin"
                    }
                })
                
            elif decision == AuthenticationDecision.CHALLENGE:
                await self.send_realtime_alert(session_id, {
                    "type": "soft_challenge",
                    "severity": "low",
                    "message": "Additional verification requested",
                    "details": {
                        "risk_score": response.risk_score,
                        "challenge_type": "security_question"
                    }
                })
                
        except Exception as e:
            self.logger.error(f"Error handling authentication decision: {e}")
    
    async def _convert_response_to_dict(self, response: AuthenticationResponse) -> Dict[str, Any]:
        """Convert AuthenticationResponse to dictionary for JSON serialization"""
        try:
            # Convert numpy arrays and other non-serializable objects
            response_dict = {
                "session_id": response.session_id,
                "user_id": response.user_id,
                "request_id": response.request_id,
                "decision": response.decision.value,
                "risk_level": response.risk_level.value,
                "risk_score": float(response.risk_score),
                "confidence": float(response.confidence),
                "total_processing_time_ms": float(response.total_processing_time_ms),
                "timestamp": response.timestamp.isoformat(),
                "next_verification_delay": response.next_verification_delay,
                
                # Layer results
                "layer1_result": {
                    "similarity_score": float(response.layer1_result.similarity_score),
                    "confidence_level": response.layer1_result.confidence_level,
                    "matched_profile_mode": response.layer1_result.matched_profile_mode,
                    "decision": response.layer1_result.decision,
                    "processing_time_ms": float(response.layer1_result.processing_time_ms)
                },
                
                "layer2_result": {
                    "overall_confidence": float(response.layer2_result.overall_confidence),
                    "decision": response.layer2_result.decision,
                    "processing_time_ms": float(response.layer2_result.processing_time_ms)
                } if response.layer2_result else None,
                
                "drift_result": {
                    "drift_detected": response.drift_result.drift_detected,
                    "drift_magnitude": float(response.drift_result.drift_magnitude),
                    "drift_type": response.drift_result.drift_type
                } if response.drift_result else None,
                
                # Explainability
                "explanation": response.explanation,
                
                # Performance metrics
                "stage_timings": {
                    stage.value: float(timing) for stage, timing in response.stage_timings.items()
                }
            }
            
            return response_dict
            
        except Exception as e:
            self.logger.error(f"Error converting response to dict: {e}")
            return {}
    
    def _get_time_of_day(self) -> str:
        """Get current time of day classification"""
        hour = datetime.now().hour
        if 6 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        else:
            return "night"
    
    async def shutdown(self):
        """Shutdown ML-Engine integration"""
        try:
            self.logger.info("Shutting down ML-Engine integration...")
            
            # Shutdown ML-Engine
            await self.ml_engine.shutdown()
            
            # Close WebSocket connections
            for connection in self.websocket_connections.values():
                try:
                    await connection.close()
                except:
                    pass
            
            self.logger.info("ML-Engine integration shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during ML-Engine integration shutdown: {e}")

# Global integrator instance
ml_session_integrator = MLEngineSessionIntegrator()

# Convenience functions for backend integration
async def start_ml_session(session_data: Dict[str, Any]) -> bool:
    """Start ML-Engine session (called from backend)"""
    return await ml_session_integrator.handle_session_start(session_data)

async def process_behavioral_event(event_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Process behavioral event through ML-Engine (called from backend)"""
    return await ml_session_integrator.handle_behavioral_event(event_data)

async def end_ml_session(session_data: Dict[str, Any]) -> bool:
    """End ML-Engine session (called from backend)"""
    return await ml_session_integrator.handle_session_end(session_data)

async def get_ml_session_status(session_id: str) -> Optional[Dict[str, Any]]:
    """Get ML-Engine session status (called from backend)"""
    return await ml_session_integrator.get_session_ml_status(session_id)

async def get_ml_engine_statistics() -> Dict[str, Any]:
    """Get ML-Engine statistics (called from backend)"""
    return await ml_session_integrator.get_ml_engine_stats()

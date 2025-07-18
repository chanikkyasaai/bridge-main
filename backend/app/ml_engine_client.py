"""
ML Engine HTTP Client for Backend Integration
Handles communication between backend and ML engine
"""

import httpx
import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class MLEngineClient:
    """HTTP client for communicating with ML Engine API"""
    
    def __init__(self, ml_engine_url: str = "http://127.0.0.1:8001"):
        self.ml_engine_url = ml_engine_url
        self.client = None
        self.is_available = True
    
    async def __aenter__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.aclose()
    
    async def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Optional[Dict]:
        """Make HTTP request to ML Engine with error handling"""
        if not self.is_available:
            logger.warning("ML Engine is marked as unavailable")
            return None
        
        try:
            if not self.client:
                self.client = httpx.AsyncClient(timeout=30.0)
            
            url = f"{self.ml_engine_url}{endpoint}"
            
            if method.upper() == "GET":
                response = await self.client.get(url)
            elif method.upper() == "POST":
                response = await self.client.post(url, json=data)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except httpx.ConnectError:
            logger.error("Failed to connect to ML Engine - marking as unavailable")
            self.is_available = False
            return None
        except httpx.TimeoutException:
            logger.error("ML Engine request timed out")
            return None
        except httpx.HTTPStatusError as e:
            logger.error(f"ML Engine returned error {e.response.status_code}: {e.response.text}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error communicating with ML Engine: {e}")
            return None
    
    async def health_check(self) -> Dict[str, Any]:
        """Check ML Engine health status"""
        result = await self._make_request("GET", "/")
        if result:
            self.is_available = True
            return result
        return {"status": "unavailable", "components": {}, "statistics": {}}
    
    async def start_session(self, user_id: str, session_id: str, device_info: Optional[Dict] = None) -> Dict[str, Any]:
        """Start ML analysis session"""
        data = {
            "user_id": user_id,
            "session_id": session_id,
            "device_info": device_info or {}
        }
        
        print("Starting ML session... at client")
        
        result = await self._make_request("POST", "/session/start", data)
        if result:
            logger.info(f"ML session started: {session_id}")
            return result
        
        return {"status": "error", "message": "Failed to start ML session"}
    
    async def end_session(self, session_id: str, reason: str = "completed") -> Dict[str, Any]:
        """End ML analysis session"""
        data = {
            "session_id": session_id,
            "reason": reason
        }
        
        result = await self._make_request("POST", "/session/end", data)
        if result:
            logger.info(f"ML session ended: {session_id}")
            return result
        
        return {"status": "error", "message": "Failed to end ML session"}
    
    async def analyze_behavior(self, user_id: str, session_id: str, events: List[Dict]) -> Dict[str, Any]:
        """Analyze behavioral data"""
        data = {
            "user_id": user_id,
            "session_id": session_id,
            "events": events
        }
        
        result = await self._make_request("POST", "/analyze", data)
        if result:
            logger.info(f"Behavioral analysis completed for session {session_id}")
            return result
        
        # Fallback decision when ML engine is unavailable
        return {
            "status": "fallback",
            "decision": "allow",  # Default to allow when ML is unavailable
            "confidence": 0.5,
            "message": "ML Engine unavailable - using fallback decision"
        }
    
    async def submit_feedback(self, user_id: str, session_id: str, decision_id: str, 
                             was_correct: bool, feedback_source: str = "system") -> Dict[str, Any]:
        """Submit feedback for model improvement"""
        data = {
            "user_id": user_id,
            "session_id": session_id,
            "decision_id": decision_id,
            "was_correct": was_correct,
            "feedback_source": feedback_source
        }
        
        result = await self._make_request("POST", "/feedback", data)
        if result:
            logger.info(f"Feedback submitted for decision {decision_id}")
            return result
        
        return {"status": "error", "message": "Failed to submit feedback"}
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get ML Engine statistics"""
        result = await self._make_request("GET", "/statistics")
        if result:
            return result
        
        return {"status": "unavailable", "statistics": {}}

# Global ML Engine client instance
ml_engine_client = MLEngineClient()

# Behavioral event processing functions
async def behavioral_event_hook(user_id: str, session_id: str, events: List[Dict]) -> Optional[Dict]:
    """
    Process behavioral events through ML Engine
    Called from WebSocket handlers when behavioral data is received
    """
    try:
        async with ml_engine_client as client:
            result = await client.analyze_behavior(user_id, session_id, events)
            
            if result and result.get("status") == "success":
                # Log the ML decision for monitoring
                logger.info(f"ML Decision for {session_id}: {result.get('decision')} "
                           f"(confidence: {result.get('confidence'):.2f})")
                
                return result
            else:
                logger.warning(f"ML analysis failed for session {session_id}")
                return None
                
    except Exception as e:
        logger.error(f"Behavioral event hook failed: {e}")
        return None

async def start_ml_session(user_id: str, session_id: str, device_info: Optional[Dict] = None) -> Dict[str, Any]:
    """Start ML session when user session begins"""
    try:
        async with ml_engine_client as client:
            return await client.start_session(user_id, session_id, device_info)
    except Exception as e:
        logger.error(f"Failed to start ML session: {e}")
        return {"status": "error", "message": str(e)}

async def end_ml_session(session_id: str, reason: str = "completed") -> Dict[str, Any]:
    """End ML session when user session terminates"""
    try:
        async with ml_engine_client as client:
            return await client.end_session(session_id, reason)
    except Exception as e:
        logger.error(f"Failed to end ML session: {e}")
        return {"status": "error", "message": str(e)}

async def submit_ml_feedback(user_id: str, session_id: str, decision_id: str, 
                           was_correct: bool, feedback_source: str = "system") -> Dict[str, Any]:
    """Submit feedback to ML Engine"""
    try:
        async with ml_engine_client as client:
            return await client.submit_feedback(user_id, session_id, decision_id, 
                                              was_correct, feedback_source)
    except Exception as e:
        logger.error(f"Failed to submit ML feedback: {e}")
        return {"status": "error", "message": str(e)}

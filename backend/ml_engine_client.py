"""
ML-Engine HTTP Client for Backend Integration

This client handles HTTP communication between the backend and the separately hosted ML-Engine.
It provides all the necessary functions to interact with the ML-Engine API service.
"""

import aiohttp
import asyncio
import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime
import time

logger = logging.getLogger(__name__)

class MLEngineClient:
    """HTTP client for communicating with ML-Engine API service"""
    
    def __init__(self, ml_engine_url: str = "http://localhost:8001", timeout: int = 30):
        self.ml_engine_url = ml_engine_url.rstrip('/')
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.session = None
        self.is_available = False
        self.last_health_check = 0
        self.health_check_interval = 60  # Check every 60 seconds
        
    async def _get_session(self):
        """Get or create aiohttp session"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self.session
    
    async def _make_request(self, method: str, endpoint: str, data: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Make HTTP request to ML-Engine"""
        try:
            session = await self._get_session()
            url = f"{self.ml_engine_url}{endpoint}"
            
            if method.upper() == 'GET':
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.error(f"ML-Engine request failed: {response.status} - {await response.text()}")
                        return None
            
            elif method.upper() == 'POST':
                async with session.post(url, json=data) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.error(f"ML-Engine request failed: {response.status} - {await response.text()}")
                        return None
                        
        except asyncio.TimeoutError:
            logger.error(f"ML-Engine request timeout: {endpoint}")
            self.is_available = False
            return None
        except aiohttp.ClientConnectorError:
            logger.error(f"ML-Engine connection error: {endpoint}")
            self.is_available = False
            return None
        except Exception as e:
            logger.error(f"ML-Engine request error: {e}")
            self.is_available = False
            return None
    
    async def health_check(self) -> bool:
        """Check if ML-Engine is available"""
        current_time = time.time()
        
        # Only check if enough time has passed
        if current_time - self.last_health_check < self.health_check_interval:
            return self.is_available
        
        try:
            response = await self._make_request('GET', '/health')
            if response and response.get('status') == 'healthy':
                self.is_available = True
                logger.debug("ML-Engine health check: OK")
            else:
                self.is_available = False
                logger.warning("ML-Engine health check: FAILED")
                
            self.last_health_check = current_time
            return self.is_available
            
        except Exception as e:
            logger.error(f"ML-Engine health check error: {e}")
            self.is_available = False
            self.last_health_check = current_time
            return False
    
    async def start_session(self, session_id: str, user_id: str, phone: str, device_id: str, 
                           context: Dict[str, Any] = None) -> bool:
        """Start ML-Engine session"""
        # if not await self.health_check():
        #     logger.warning("ML-Engine not available for session start")
        #     return False
        
        try:
            logger.info(f"Starting ML-Engine session: {session_id}")
            data = {
                "session_id": session_id,
                "user_id": user_id,
                "device_id": device_id,
                "phone": phone,
                "device_type": context.get("device_type", "mobile") if context else "mobile",
                "device_model": context.get("device_model", "unknown") if context else "unknown",
                "os_version": context.get("os_version", "unknown") if context else "unknown",
                "app_version": context.get("app_version", "unknown") if context else "unknown",
                "network_type": context.get("network_type", "unknown") if context else "unknown",
                "location_data": context.get("location_data") if context else None,
                "is_known_device": context.get("is_known_device", False) if context else False,
                "is_trusted_location": context.get("is_trusted_location", False) if context else False
            }
            
            print(f"ML-Engine session data: {data} backend")
            
            response = await self._make_request('POST', '/session/start', data)
            if response and response.get('status') == 'success':
                logger.info(f"ML-Engine session started: {session_id}")
                return True
            else:
                logger.error(f"Failed to start ML-Engine session: {session_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error starting ML-Engine session: {e}")
            return False
    
    async def process_behavioral_event(self, session_id: str, user_id: str, device_id: str,
                                     event_type: str, event_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process behavioral event through ML-Engine"""
        if not self.is_available:
            # Quick health check
            if not await self.health_check():
                return None
        
        try:
            data = {
                "session_id": session_id,
                "user_id": user_id,
                "device_id": device_id,
                "timestamp": datetime.utcnow().isoformat(),
                "event_type": event_type,
                "features": event_data.get("features", {}),
                "raw_metadata": event_data.get("metadata", {})
            }
            
            response = await self._make_request('POST', '/session/process-event', data)
            if response and response.get('status') == 'success':
                return response.get('data')
            else:
                logger.warning(f"ML-Engine event processing failed for session: {session_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error processing behavioral event: {e}")
            return None
    
    async def end_session(self, session_id: str, user_id: str, final_decision: str = "normal",
                         session_stats: Dict[str, Any] = None) -> bool:
        """End ML-Engine session"""
        if not self.is_available:
            # Try anyway, in case ML-Engine came back online
            await self.health_check()
        
        try:
            data = {
                "session_id": session_id,
                "user_id": user_id,
                "final_decision": final_decision,
                "session_duration": session_stats.get("duration_minutes", 0) if session_stats else 0,
                "total_events": session_stats.get("total_events", 0) if session_stats else 0
            }
            
            response = await self._make_request('POST', '/session/end', data)
            if response and response.get('status') == 'success':
                logger.info(f"ML-Engine session ended: {session_id}")
                return True
            else:
                logger.error(f"Failed to end ML-Engine session: {session_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error ending ML-Engine session: {e}")
            return False
    
    async def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get ML-Engine session status"""
        if not self.is_available:
            return None
        
        try:
            response = await self._make_request('GET', f'/session/{session_id}/status')
            if response and response.get('status') == 'success':
                return response.get('data')
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error getting ML session status: {e}")
            return None
    
    async def get_ml_stats(self) -> Dict[str, Any]:
        """Get ML-Engine statistics"""
        if not await self.health_check():
            return {
                "ml_enabled": False,
                "ml_available": False,
                "error": "ML-Engine not available"
            }
        
        try:
            response = await self._make_request('GET', '/stats')
            if response and response.get('status') == 'success':
                stats = response.get('data', {})
                stats.update({
                    "ml_enabled": True,
                    "ml_available": True,
                    "ml_engine_url": self.ml_engine_url
                })
                return stats
            else:
                return {
                    "ml_enabled": False,
                    "ml_available": False,
                    "error": "Failed to get ML stats"
                }
                
        except Exception as e:
            logger.error(f"Error getting ML stats: {e}")
            return {
                "ml_enabled": False,
                "ml_available": False,
                "error": str(e)
            }
    
    async def send_alert(self, alert_data: Dict[str, Any]) -> bool:
        """Send alert to ML-Engine"""
        if not self.is_available:
            return False
        
        try:
            response = await self._make_request('POST', '/alerts', alert_data)
            return response and response.get('status') == 'success'
            
        except Exception as e:
            logger.error(f"Error sending ML alert: {e}")
            return False
    
    async def process_event(self, session_id: str, user_id: str, device_id: str,
                          event_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process behavioral event - wrapper for backward compatibility"""
        event_type = event_data.get("event_type", "unknown")
        return await self.process_behavioral_event(session_id, user_id, device_id, event_type, event_data)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get ML-Engine statistics - wrapper for backward compatibility"""
        return await self.get_ml_stats()
    
    async def close(self):
        """Close the HTTP session"""
        if self.session and not self.session.closed:
            await self.session.close()

# Global ML-Engine client instance
ml_client = None

def initialize_ml_client(ml_engine_url: str = "http://localhost:8001", timeout: int = 30) -> MLEngineClient:
    """Initialize the global ML-Engine client"""
    global ml_client
    ml_client = MLEngineClient(ml_engine_url, timeout)
    return ml_client

def get_ml_client() -> Optional[MLEngineClient]:
    """Get the global ML-Engine client"""
    return ml_client

# Convenience functions for backend integration
async def initialize_ml_integration(ml_engine_url: str = "http://localhost:8001") -> bool:
    """Initialize ML-Engine HTTP client integration"""
    try:
        client = initialize_ml_client(ml_engine_url)
        success = await client.health_check()
        if success:
            logger.info(f"✅ ML-Engine client initialized successfully: {ml_engine_url}")
        else:
            logger.warning(f"⚠️ ML-Engine client initialized but service not available: {ml_engine_url}")
        return True  # Return True even if service not available initially
    except Exception as e:
        logger.error(f"❌ Failed to initialize ML-Engine client: {e}")
        return False

async def session_created_hook(session_id: str, user_id: str, phone: str, 
                              device_id: str, context: Dict[str, Any] = None) -> bool:
    """Session creation hook using HTTP client"""
    client = get_ml_client()
    if client:
        return await client.start_session(session_id, user_id, phone, device_id, context)
    return True  # Don't fail if ML-Engine not available

async def behavioral_event_hook(session_id: str, user_id: str, device_id: str,
                               event_type: str, event_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Behavioral event hook using HTTP client"""
    client = get_ml_client()
    if client:
        return await client.process_behavioral_event(session_id, user_id, device_id, event_type, event_data)
    return None

async def session_ended_hook(session_id: str, user_id: str, final_decision: str = "normal",
                            session_stats: Dict[str, Any] = None) -> bool:
    """Session end hook using HTTP client"""
    client = get_ml_client()
    if client:
        return await client.end_session(session_id, user_id, final_decision, session_stats)
    return True

async def get_session_ml_status(session_id: str) -> Optional[Dict[str, Any]]:
    """Get ML status for session using HTTP client"""
    client = get_ml_client()
    if client:
        return await client.get_session_status(session_id)
    return None

async def get_ml_engine_stats() -> Dict[str, Any]:
    """Get ML-Engine statistics using HTTP client"""
    client = get_ml_client()
    if client:
        return await client.get_ml_stats()
    return {"ml_enabled": False, "error": "ML client not initialized"}

async def shutdown_ml_integration():
    """Shutdown ML integration"""
    client = get_ml_client()
    if client:
        await client.close()
        logger.info("✅ ML-Engine client closed")

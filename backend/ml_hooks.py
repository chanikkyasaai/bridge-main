"""
Backend Integration Hooks for ML-Engine Session Lifecycle

This module provides the backend hooks that integrate with the existing session 
management to ensure the ML-engine is active throughout the session lifecycle.

Integration with:
- app.core.session_manager
- app.api.v1.endpoints.websocket
- WebSocket behavioral data streaming

Uses HTTP client for ML-Engine communication when running as separate services.
"""

import asyncio
import logging
import json
import os
from typing import Dict, Any, Optional
from datetime import datetime

from ml_engine_client import MLEngineClient

logger = logging.getLogger(__name__)

class BackendMLHooks:
    """
    Integration hooks between backend session management and ML-Engine.
    
    This class manages the lifecycle connection between backend sessions
    and the ML-Engine, handling session start, behavioral event processing,
    and session end through HTTP API calls to the ML-Engine service.
    """
    
    def __init__(self, ml_engine_url: str = None):
        """Initialize ML hooks with HTTP client"""
        self.ml_engine_url = ml_engine_url or os.getenv('ML_ENGINE_URL', 'http://localhost:8001')
        self.ml_enabled = os.getenv('ML_ENGINE_ENABLED', 'true').lower() == 'true'
        self.client = MLEngineClient(self.ml_engine_url) if self.ml_enabled else None
        self._initialized = False
        logger.info(f"ML-Engine hooks initialized - URL: {self.ml_engine_url}, Enabled: {self.ml_enabled}")
    
    async def initialize(self) -> bool:
        """Initialize ML hooks and check ML-Engine availability"""
        if not self.ml_enabled:
            logger.info("ML-Engine integration disabled")
            self._initialized = True
            return True
        
        try:
            if self.client:
                is_available = await self.client.health_check()
                if is_available:
                    logger.info("✅ ML-Engine connection established")
                    self._initialized = True
                    return True
                else:
                    logger.warning("⚠️ ML-Engine not available, running without ML features")
                    self._initialized = True
                    return True  # Continue without ML features
            
        except Exception as e:
            logger.error(f"❌ ML-Engine initialization error: {e}")
            self._initialized = True
            return True  # Continue without ML features
    
    async def start_session(self, session_id: str, user_id: str, phone: str, 
                           device_id: str, context: Dict[str, Any] = None) -> bool:
        """
        Hook called when a new session starts.
        
        Args:
            session_id: Unique session identifier
            user_id: User identifier
            phone: User phone number
            device_id: Device identifier
            context: Additional session context data
        
        Returns:
            bool: True if successfully started ML session
        """
        if not self.ml_enabled or not self.client:
            return True
        
        try:
            success = await self.client.start_session(
                session_id=session_id,
                user_id=user_id,
                phone=phone,
                device_id=device_id,
                context=context or {}
            )
            
            if success:
                logger.info(f"✅ ML session started: {session_id}")
            else:
                logger.warning(f"⚠️ Failed to start ML session: {session_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Error starting ML session {session_id}: {e}")
            return False
    
    async def process_behavioral_event(self, session_id: str, user_id: str, 
                                     device_id: str, event_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Hook called when behavioral event data is received.
        
        Args:
            session_id: Session identifier
            user_id: User identifier
            device_id: Device identifier
            event_data: Behavioral event data
        
        Returns:
            Optional[Dict]: ML analysis response or None
        """
        if not self.ml_enabled or not self.client:
            return None
        
        try:
            response = await self.client.process_event(
                session_id=session_id,
                user_id=user_id,
                device_id=device_id,
                event_data=event_data
            )
            
            if response:
                logger.debug(f"✅ ML event processed: {session_id}")
                return response
            else:
                logger.debug(f"⚠️ No ML response for event: {session_id}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error processing ML event {session_id}: {e}")
            return None
    
    async def end_session(self, session_id: str, user_id: str, 
                         final_decision: str = "normal") -> bool:
        """
        Hook called when a session ends.
        
        Args:
            session_id: Session identifier
            user_id: User identifier
            final_decision: Final session decision
        
        Returns:
            bool: True if successfully ended ML session
        """
        if not self.ml_enabled or not self.client:
            return True
        
        try:
            success = await self.client.end_session(
                session_id=session_id,
                user_id=user_id,
                final_decision=final_decision
            )
            
            if success:
                logger.info(f"✅ ML session ended: {session_id}")
            else:
                logger.warning(f"⚠️ Failed to end ML session: {session_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Error ending ML session {session_id}: {e}")
            return False
    
    async def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get ML session status"""
        if not self.ml_enabled or not self.client:
            return None
        
        try:
            return await self.client.get_session_status(session_id)
        except Exception as e:
            logger.error(f"❌ Error getting ML session status {session_id}: {e}")
            return None
    
    async def get_ml_statistics(self) -> Dict[str, Any]:
        """Get ML-Engine statistics"""
        if not self.ml_enabled or not self.client:
            return {"ml_enabled": False, "status": "disabled"}
        
        try:
            stats = await self.client.get_stats()
            return stats or {"ml_enabled": True, "status": "unavailable"}
        except Exception as e:
            logger.error(f"❌ Error getting ML statistics: {e}")
            return {"ml_enabled": True, "status": "error", "error": str(e)}
    
    async def shutdown(self):
        """Shutdown ML hooks"""
        if self.client:
            try:
                await self.client.close()
                logger.info("✅ ML-Engine client closed")
            except Exception as e:
                logger.error(f"❌ Error closing ML client: {e}")

# Global ML hooks instance
ml_hooks = BackendMLHooks()

# Backward compatibility functions
async def initialize_ml_integration() -> bool:
    """Initialize ML integration"""
    return await ml_hooks.initialize()

async def hook_session_start(session_id: str, user_id: str, phone: str, 
                           device_id: str, context: Dict[str, Any] = None) -> bool:
    """Session start hook"""
    return await ml_hooks.start_session(session_id, user_id, phone, device_id, context)

async def hook_behavioral_event(session_id: str, user_id: str, device_id: str, 
                               event_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Behavioral event hook"""
    return await ml_hooks.process_behavioral_event(session_id, user_id, device_id, event_data)

async def hook_session_end(session_id: str, user_id: str, 
                          final_decision: str = "normal") -> bool:
    """Session end hook"""
    return await ml_hooks.end_session(session_id, user_id, final_decision)

async def get_ml_session_status(session_id: str) -> Optional[Dict[str, Any]]:
    """Get ML session status"""
    return await ml_hooks.get_session_status(session_id)

async def get_ml_engine_statistics() -> Dict[str, Any]:
    """Get ML engine statistics"""
    return await ml_hooks.get_ml_statistics()

async def shutdown_ml_integration():
    """Shutdown ML integration"""
    await ml_hooks.shutdown()

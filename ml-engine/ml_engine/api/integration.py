"""
Backend API Integration for BRIDGE ML-Engine
WebSocket and REST API interfaces for real-time behavioral streaming
"""

import asyncio
import websockets
import json
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import aiohttp
from fastapi import FastAPI, WebSocket, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from ml_engine import ml_engine, AuthenticationRequest, AuthenticationResponse
from ml_engine.utils.behavioral_vectors import BehavioralEvent
from ml_engine.config import CONFIG

logger = logging.getLogger(__name__)

# Pydantic models for API
class BehavioralEventAPI(BaseModel):
    timestamp: str
    event_type: str
    features: Dict[str, float]
    session_id: str
    user_id: str

class AuthenticationRequestAPI(BaseModel):
    session_id: str
    user_id: str
    events: List[BehavioralEventAPI]
    context: Dict[str, Any]
    require_explanation: bool = False

class AuthenticationResponseAPI(BaseModel):
    session_id: str
    user_id: str
    decision: str
    risk_level: str
    risk_score: float
    confidence: float
    processing_time_ms: float
    timestamp: str
    next_verification_delay: int
    explanation: Optional[Dict[str, Any]] = None

class StreamingEventAPI(BaseModel):
    session_id: str
    user_id: str
    event: BehavioralEventAPI
    context: Dict[str, Any]

class SessionControlAPI(BaseModel):
    session_id: str
    action: str  # 'continue', 'step_up_auth', 'challenge', 'block'
    reason: str
    timestamp: str

@dataclass
class WebSocketSession:
    """WebSocket session state"""
    session_id: str
    user_id: str
    websocket: WebSocket
    connected_at: datetime
    last_activity: datetime
    event_count: int = 0

class BehavioralStreamProcessor:
    """Processes real-time behavioral event streams"""
    
    def __init__(self):
        self.active_streams: Dict[str, WebSocketSession] = {}
        self.event_buffers: Dict[str, List[BehavioralEvent]] = {}
        self.buffer_size = 10  # Events before triggering authentication
        self.buffer_timeout = 2.0  # Seconds before timeout trigger
        
    async def handle_streaming_event(
        self, 
        websocket: WebSocket, 
        event_data: StreamingEventAPI
    ):
        """Handle single streaming behavioral event"""
        
        try:
            # Convert API event to internal format
            event = BehavioralEvent(
                timestamp=datetime.fromisoformat(event_data.event.timestamp),
                event_type=event_data.event.event_type,
                features=event_data.event.features,
                session_id=event_data.session_id,
                user_id=event_data.user_id
            )
            
            # Add to buffer
            if event_data.session_id not in self.event_buffers:
                self.event_buffers[event_data.session_id] = []
            
            self.event_buffers[event_data.session_id].append(event)
            
            # Update session activity
            if event_data.session_id in self.active_streams:
                session = self.active_streams[event_data.session_id]
                session.last_activity = datetime.now()
                session.event_count += 1
            
            # Check if buffer is ready for authentication
            buffer = self.event_buffers[event_data.session_id]
            if len(buffer) >= self.buffer_size:
                await self._trigger_authentication(event_data.session_id, event_data.context)
                
        except Exception as e:
            logger.error(f"Error handling streaming event: {e}")
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": f"Event processing error: {str(e)}"
            }))

    async def _trigger_authentication(self, session_id: str, context: Dict[str, Any]):
        """Trigger authentication with buffered events"""
        
        if session_id not in self.event_buffers:
            return
            
        events = self.event_buffers[session_id]
        if not events:
            return
            
        try:
            # Create authentication request
            request = AuthenticationRequest(
                session_id=session_id,
                user_id=events[0].user_id,
                events=events,
                context=context,
                timestamp=datetime.now(),
                require_explanation=context.get("require_explanation", False)
            )
            
            # Run authentication
            response = await ml_engine.authenticate(request)
            
            # Send result to WebSocket client
            if session_id in self.active_streams:
                websocket = self.active_streams[session_id].websocket
                await websocket.send_text(json.dumps({
                    "type": "authentication_result",
                    "session_id": session_id,
                    "decision": response.decision.value,
                    "risk_level": response.risk_level.value,
                    "risk_score": response.risk_score,
                    "confidence": response.confidence,
                    "next_verification_delay": response.next_verification_delay,
                    "timestamp": response.timestamp.isoformat()
                }))
            
            # Clear buffer (keep last few events for context)
            self.event_buffers[session_id] = events[-3:] if len(events) > 3 else []
            
        except Exception as e:
            logger.error(f"Authentication trigger error: {e}")

    async def register_websocket_session(
        self, 
        websocket: WebSocket, 
        session_id: str, 
        user_id: str
    ):
        """Register new WebSocket session"""
        
        ws_session = WebSocketSession(
            session_id=session_id,
            user_id=user_id,
            websocket=websocket,
            connected_at=datetime.now(),
            last_activity=datetime.now()
        )
        
        self.active_streams[session_id] = ws_session
        logger.info(f"WebSocket session registered: {session_id}")

    async def unregister_websocket_session(self, session_id: str):
        """Unregister WebSocket session"""
        
        if session_id in self.active_streams:
            del self.active_streams[session_id]
        if session_id in self.event_buffers:
            del self.event_buffers[session_id]
        
        logger.info(f"WebSocket session unregistered: {session_id}")

    def get_active_sessions(self) -> Dict[str, Any]:
        """Get active streaming sessions"""
        return {
            "total_sessions": len(self.active_streams),
            "sessions": {
                session_id: {
                    "user_id": session.user_id,
                    "connected_at": session.connected_at.isoformat(),
                    "last_activity": session.last_activity.isoformat(),
                    "event_count": session.event_count
                }
                for session_id, session in self.active_streams.items()
            }
        }

# Global streaming processor
stream_processor = BehavioralStreamProcessor()

# FastAPI application
app = FastAPI(
    title="BRIDGE ML-Engine API",
    description="Behavioral Risk Intelligence for Dynamic Guarded Entry - ML Engine API",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize ML Engine on startup"""
    await ml_engine.initialize()
    logger.info("BRIDGE ML-Engine API started")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    await ml_engine.shutdown()
    logger.info("BRIDGE ML-Engine API shutdown")

# REST API Endpoints

@app.post("/api/v1/authenticate", response_model=AuthenticationResponseAPI)
async def authenticate_batch(request: AuthenticationRequestAPI):
    """Batch authentication endpoint"""
    
    try:
        # Convert API events to internal format
        events = [
            BehavioralEvent(
                timestamp=datetime.fromisoformat(event.timestamp),
                event_type=event.event_type,
                features=event.features,
                session_id=event.session_id,
                user_id=event.user_id
            )
            for event in request.events
        ]
        
        # Create internal request
        auth_request = AuthenticationRequest(
            session_id=request.session_id,
            user_id=request.user_id,
            events=events,
            context=request.context,
            timestamp=datetime.now(),
            require_explanation=request.require_explanation
        )
        
        # Run authentication
        response = await ml_engine.authenticate(auth_request)
        
        # Convert to API response
        api_response = AuthenticationResponseAPI(
            session_id=response.session_id,
            user_id=response.user_id,
            decision=response.decision.value,
            risk_level=response.risk_level.value,
            risk_score=response.risk_score,
            confidence=response.confidence,
            processing_time_ms=response.processing_time_ms,
            timestamp=response.timestamp.isoformat(),
            next_verification_delay=response.next_verification_delay,
            explanation=asdict(response.explanation) if response.explanation else None
        )
        
        return api_response
        
    except Exception as e:
        logger.error(f"Authentication API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    return await ml_engine.health_check()

@app.get("/api/v1/stats")
async def get_stats():
    """Get engine statistics"""
    return {
        "ml_engine": asdict(ml_engine.get_engine_stats()),
        "active_sessions": ml_engine.get_active_sessions(),
        "streaming_sessions": stream_processor.get_active_sessions()
    }

@app.get("/api/v1/user/{user_id}/profile")
async def get_user_profile(user_id: str):
    """Get user behavioral profile"""
    return await ml_engine.get_user_profile(user_id)

@app.post("/api/v1/user/{user_id}/train")
async def train_user_model(
    user_id: str,
    training_data: Dict[str, Any],
    background_tasks: BackgroundTasks
):
    """Train user-specific model"""
    
    # Convert training data
    vectors = []  # Convert from training_data
    labels = training_data.get("labels", [])
    
    # Schedule training in background
    background_tasks.add_task(
        ml_engine.train_user_model,
        user_id,
        vectors,
        labels
    )
    
    return {"status": "training_scheduled", "user_id": user_id}

@app.post("/api/v1/session/{session_id}/control")
async def control_session(session_id: str, control: SessionControlAPI):
    """Manual session control"""
    
    # In a full implementation, this would send control commands
    # to the frontend application
    
    return {
        "status": "control_sent",
        "session_id": session_id,
        "action": control.action,
        "timestamp": datetime.now().isoformat()
    }

# WebSocket Endpoint

@app.websocket("/api/v1/stream/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str, user_id: str):
    """WebSocket endpoint for real-time behavioral streaming"""
    
    await websocket.accept()
    
    try:
        # Register session
        await stream_processor.register_websocket_session(websocket, session_id, user_id)
        
        # Send welcome message
        await websocket.send_text(json.dumps({
            "type": "connection_established",
            "session_id": session_id,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        }))
        
        # Handle incoming messages
        async for message in websocket.iter_text():
            try:
                data = json.loads(message)
                
                if data.get("type") == "behavioral_event":
                    # Process behavioral event
                    event_data = StreamingEventAPI(**data["data"])
                    await stream_processor.handle_streaming_event(websocket, event_data)
                    
                elif data.get("type") == "heartbeat":
                    # Respond to heartbeat
                    await websocket.send_text(json.dumps({
                        "type": "heartbeat_ack",
                        "timestamp": datetime.now().isoformat()
                    }))
                    
                else:
                    logger.warning(f"Unknown message type: {data.get('type')}")
                    
            except json.JSONDecodeError:
                logger.error("Invalid JSON received")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON format"
                }))
            except Exception as e:
                logger.error(f"WebSocket message error: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": str(e)
                }))
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Unregister session
        await stream_processor.unregister_websocket_session(session_id)

# Backend integration client

class BackendIntegrationClient:
    """Client for integrating with BRIDGE backend API"""
    
    def __init__(self, backend_url: str = "http://localhost:8000"):
        self.backend_url = backend_url
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def send_authentication_result(
        self, 
        response: AuthenticationResponse
    ) -> bool:
        """Send authentication result to backend"""
        
        try:
            data = {
                "session_id": response.session_id,
                "user_id": response.user_id,
                "decision": response.decision.value,
                "risk_level": response.risk_level.value,
                "risk_score": response.risk_score,
                "confidence": response.confidence,
                "timestamp": response.timestamp.isoformat(),
                "next_verification_delay": response.next_verification_delay
            }
            
            async with self.session.post(
                f"{self.backend_url}/api/v1/ml/authentication_result",
                json=data
            ) as resp:
                return resp.status == 200
                
        except Exception as e:
            logger.error(f"Failed to send authentication result: {e}")
            return False

    async def get_user_context(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """Get user context from backend"""
        
        try:
            async with self.session.get(
                f"{self.backend_url}/api/v1/users/{user_id}/context",
                params={"session_id": session_id}
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                return {}
                
        except Exception as e:
            logger.error(f"Failed to get user context: {e}")
            return {}

    async def notify_risk_event(
        self, 
        session_id: str, 
        user_id: str, 
        risk_level: str, 
        details: Dict[str, Any]
    ) -> bool:
        """Notify backend of high-risk events"""
        
        try:
            data = {
                "session_id": session_id,
                "user_id": user_id,
                "risk_level": risk_level,
                "details": details,
                "timestamp": datetime.now().isoformat()
            }
            
            async with self.session.post(
                f"{self.backend_url}/api/v1/security/risk_event",
                json=data
            ) as resp:
                return resp.status == 200
                
        except Exception as e:
            logger.error(f"Failed to notify risk event: {e}")
            return False

def start_api_server(host: str = "0.0.0.0", port: int = 8001):
    """Start the ML Engine API server"""
    uvicorn.run(
        "api_integration:app",
        host=host,
        port=port,
        reload=False,
        access_log=True
    )

if __name__ == "__main__":
    # Start the API server
    start_api_server()

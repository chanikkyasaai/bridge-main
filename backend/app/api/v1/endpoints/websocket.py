import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Query
from typing import Dict, Any, Optional
import json
import asyncio
from datetime import datetime
from app.core.session_manager import session_manager
from app.core.security import extract_session_info

# ML-Engine Integration
try:
    from ml_hooks import behavioral_event_hook
    ML_INTEGRATION_AVAILABLE = True

    print("Imported ML-Engine integration in websocket")
except ImportError:
    print("ML-Engine integration not available in websocket")
    async def behavioral_event_hook(*args, **kwargs):
        return None
    ML_INTEGRATION_AVAILABLE = False

router = APIRouter()

class WebSocketManager:
    """Manages WebSocket connections for behavioral data collection"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        
    
    async def connect(self, websocket: WebSocket, session_id: str):
        """Accept WebSocket connection and link to session"""
        await websocket.accept()
        self.active_connections[session_id] = websocket
        
        # Link WebSocket to session
        session = session_manager.get_session(session_id)
        if session:
            session.websocket_connection = websocket
        
        print(f"WebSocket connected for session: {session_id}")
    
    def disconnect(self, session_id: str):
        """Remove WebSocket connection"""
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        
        # Unlink from session
        session = session_manager.get_session(session_id)
        if session:
            session.websocket_connection = None
        
        print(f"WebSocket disconnected for session: {session_id}")
    
    async def send_message(self, session_id: str, message: dict):
        """Send message to specific session"""
        if session_id in self.active_connections:
            websocket = self.active_connections[session_id]
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                print(f"Failed to send message to {session_id}: {e}")
                self.disconnect(session_id)
    
    async def broadcast_to_user(self, user_id: str, message: dict):
        """Send message to all sessions of a user"""
        user_sessions = session_manager.get_user_sessions(user_id)
        for session in user_sessions:
            await self.send_message(session.session_id, message)

# Global WebSocket manager
websocket_manager = WebSocketManager()

@router.websocket("/behavior/{session_id}")
async def behavioral_websocket(websocket: WebSocket, session_id: str, token: str = Query(...)):
    """
    WebSocket endpoint for real-time behavioral data collection
    """
    try:
        # Verify session token
        session_info = extract_session_info(token)
        if not session_info:
            await websocket.close(code=1008, reason="Invalid session token")
            return
        
        # Get session from session manager
        session = session_manager.get_session(session_id)
        if not session:
            await websocket.close(code=1008, reason="Session not found")
            return
            
        if session.is_blocked:
            await websocket.close(code=1008, reason="Session is blocked")
            return
        
        # Verify that the token belongs to this session's user
        token_user_id = session_info.get("user_id")
        token_phone = session_info.get("user_phone")
        
        if token_user_id != session.user_id or token_phone != session.phone:
            await websocket.close(code=1008, reason="Token does not match session user")
            return
        
        await websocket_manager.connect(websocket, session_id)
        
        # Send initial connection confirmation
        await websocket.send_text(json.dumps({
            "type": "connection_established",
            "session_id": session_id,
            "message": "Behavioral data collection started",
            "timestamp": datetime.utcnow().isoformat()
        }))

        while True:
            # Receive behavioral data from client
            data = await websocket.receive_text()

            try:
                behavioral_event = json.loads(data)
                # print(f"Received behavioral event: {behavioral_event}")
                await process_behavioral_data(session_id, behavioral_event)

                # Send acknowledgment
                await websocket.send_text(json.dumps({
                    "type": "data_received",
                    "status": "processed",
                    "timestamp": datetime.utcnow().isoformat()
                }))

            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON format",
                    "timestamp": datetime.utcnow().isoformat()
                }))
            except Exception as e:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"Processing error: {str(e)}",
                    "timestamp": datetime.utcnow().isoformat()
                }))

    except WebSocketDisconnect:
        websocket_manager.disconnect(session_id)
        print(f"WebSocket disconnected for session: {session_id}")
        
        # Handle graceful session cleanup on WebSocket disconnect
        await handle_websocket_disconnect(session_id)
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await websocket.close(code=1011, reason=f"Server error: {str(e)}")
        except:
            pass

async def process_behavioral_data(session_id: str, behavioral_event: Dict[str, Any]):
    """
    Process incoming behavioral data and update session
    """
    session = session_manager.get_session(session_id)
    if not session:
        return
    
    # Validate required fields
    if "event_type" not in behavioral_event:
        raise ValueError("Missing event_type in behavioral data")
    
    event_type = behavioral_event["event_type"]
    event_data = behavioral_event["features"]
    
    # Log event    
    print(f"Processing behavioral event: {event_type} @ websocket")
    print(f"Event data: {event_data}")
    
    # Store behavioral data
    session.add_behavioral_data(event_type, event_data)
    
    # Process through ML-Engine for real-time authentication
    ml_response = None
    
    combined_event = {
        "event_type": event_type,
        "event_data": event_data
    }

    
    print(f"ML-Engine integration available: {ML_INTEGRATION_AVAILABLE}")
    if ML_INTEGRATION_AVAILABLE:
        try:
            # Collect recent events for ML analysis (last 10 events including current)
            recent_events = session.behavioral_buffer[-9:] if len(
                session.behavioral_buffer) > 9 else session.behavioral_buffer[:]
            
            # Add current event to the list for analysis
            current_event = {
                "event_type": event_type,
                "timestamp": datetime.utcnow().isoformat(),
                "data": event_data
            }
            recent_events.append(current_event)
            
            # Call ML Engine for analysis
            ml_response = await behavioral_event_hook(
                session.user_id, 
                session_id, 
                recent_events
            )
        except Exception as e:
            print(f"ML-Engine processing error: {e}")
    
    # Analyze behavior and update risk score
    await analyze_behavioral_pattern(session, event_type, event_data, ml_response)

async def analyze_behavioral_pattern(session, event_type: str, event_data: Dict[str, Any], ml_response: Dict[str, Any] = None):
    """
    Analyze behavioral patterns and update risk score
    Integrates ML-Engine response with rule-based analysis
    """
    current_risk = session.risk_score
    risk_adjustment = 0.0
    
    # Use ML-Engine response if available
    if ml_response and ML_INTEGRATION_AVAILABLE and ml_response.get("status") == "success":
        ml_decision = ml_response.get('decision', 'allow')
        ml_confidence = ml_response.get('confidence', 0.0)
        ml_similarity_score = ml_response.get('similarity_score', 0.0)
        
        # Handle ML decisions based on confidence
        if ml_confidence > 0.8:  # High confidence ML decision
            if ml_decision == 'block':
                session.block_session(f"ML-Engine: High risk behavior detected (confidence: {ml_confidence:.2f})")
                return
            elif ml_decision == 'challenge':
                session.request_mpin_verification()
                return
            elif ml_decision == 'allow':
                # Lower risk for high-confidence allow decisions
                risk_adjustment = -0.1
        elif ml_confidence > 0.6:  # Medium confidence
            if ml_decision == 'block':
                risk_adjustment = 0.3  # Increase risk but don't block immediately
            elif ml_decision == 'challenge':
                risk_adjustment = 0.2
            elif ml_decision == 'allow':
                risk_adjustment = -0.05
        else:
            # Low confidence - use similarity score as risk factor
            if ml_similarity_score < 0.5:
                risk_adjustment = 0.15  # Unusual behavior pattern
            elif ml_similarity_score > 0.8:
                risk_adjustment = -0.05  # Very familiar behavior
    
    # Fallback to rule-based risk scoring when ML is not available or has low confidence
    risk_factors = {
        # Suspicious patterns
        "rapid_clicks": 0.1,
        "unusual_navigation": 0.15,
        "copy_paste_behavior": 0.05,
        "idle_timeout": -0.05,
        "normal_typing": -0.02,
        
        # Transaction-related
        "large_transaction": 0.2,
        "new_beneficiary": 0.15,
        "off_hours_activity": 0.1,
        
        # Authentication
        "mpin_failed": 0.25,
        "mpin_verified": -0.1,
        "multiple_login_attempts": 0.3,
        
        "shake_detected": 0.1,
        "flip_detected": 0.15,
        "fast_rotation": 0.08,
    }
    
    # Apply risk adjustment based on event type
    if event_type in risk_factors:
        risk_adjustment = risk_factors[event_type]
    
    # Additional context-based adjustments
    if event_type == "transaction_attempt":
        amount = event_data.get("amount", 0)
        if amount > 50000:  # Large amount
            risk_adjustment += 0.15
    
    elif event_type == "navigation_pattern":
        page_switches = event_data.get("page_switches_per_minute", 0)
        if page_switches > 10:  # Rapid navigation
            risk_adjustment += 0.1
    
    elif event_type == "typing_pattern":
        typing_speed = event_data.get("words_per_minute", 0)
        if typing_speed > 100 or typing_speed < 10:  # Unusual typing speed
            risk_adjustment += 0.05
    
    # Update risk score
    new_risk_score = max(0.0, min(1.0, current_risk + risk_adjustment))
    session.update_risk_score(new_risk_score)
    
    # Log risk score change
    if risk_adjustment != 0:
        session.add_behavioral_data("risk_score_update", {
            "session_id": session.session_id,
            "previous_score": current_risk,
            "new_score": new_risk_score,
            "adjustment": risk_adjustment,
            "trigger_event": event_type
        })
    
    print(f"Session {session.session_id}: Risk score updated from {current_risk:.3f} to {new_risk_score:.3f} (Δ{risk_adjustment:+.3f}) due to {event_type}")

@router.get("/sessions/{session_id}/behavior-summary")
async def get_behavior_summary(session_id: str):
    """
    Get behavioral analysis summary for a session
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Analyze behavioral buffer
    event_types = {}
    for behavior_data in session.behavioral_buffer:
        event_type = behavior_data.event_type
        event_types[event_type] = event_types.get(event_type, 0) + 1
    
    return {
        "session_id": session_id,
        "risk_score": session.risk_score,
        "total_events": len(session.behavioral_buffer),
        "event_breakdown": event_types,
        "session_duration_minutes": (datetime.utcnow() - session.created_at).total_seconds() / 60,
        "last_activity": session.last_activity.isoformat(),
        "is_blocked": session.is_blocked
    }

@router.post("/sessions/{session_id}/simulate-ml-analysis")
async def simulate_ml_analysis(session_id: str):
    """
    Simulate ML model analysis and risk score update
    This endpoint simulates what your ML model would do
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Simulate ML model prediction
    import random
    simulated_risk_score = random.uniform(0.0, 1.0)
    
    session.update_risk_score(simulated_risk_score)
    
    # Add ML analysis data
    session.add_behavioral_data("ml_analysis_result", {
        "session_id": session_id,
        "predicted_risk": simulated_risk_score,
        "model_version": "v1.0_simulation",
        "confidence": random.uniform(0.7, 0.95),
        "features_analyzed": len(session.behavioral_buffer)
    })
    
    return {
        "message": "ML analysis simulation completed",
        "session_id": session_id,
        "predicted_risk_score": simulated_risk_score,
        "action_taken": "block" if simulated_risk_score >= 0.9 else "monitor" if simulated_risk_score >= 0.7 else "normal"
    }

@router.get("/debug/token/{token}")
async def debug_token(token: str):
    """Debug endpoint to inspect token contents"""
    try:
        from app.core.security import get_token_payload
        
        # Get token payload without verification
        payload = get_token_payload(token)
        
        if not payload:
            return {"error": "Invalid token format"}
        
        # Also try to verify it
        session_info = extract_session_info(token)
        
        return {
            "token_payload": payload,
            "session_info": session_info,
            "token_type": payload.get("type"),
            "expires": payload.get("exp"),
            "issued_at": payload.get("iat")
        }
        
    except Exception as e:
        return {"error": f"Debug failed: {str(e)}"}

async def handle_websocket_disconnect(session_id: str):
    """
    Handle WebSocket disconnection and decide whether to terminate the session
    """
    # Use the new lifecycle event handler
    await session_manager.handle_app_lifecycle_event(
        session_id, 
        "websocket_disconnect",
        {"reason": "client_disconnect"}
    )

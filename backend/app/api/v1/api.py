from fastapi import APIRouter
from app.api.v1.endpoints import auth, websocket, logging

# ML-Engine endpoints (optional import)
try:
    from app.api.v1.endpoints import ml_engine
    ML_ENDPOINTS_AVAILABLE = True
except ImportError:
    ML_ENDPOINTS_AVAILABLE = False

api_router = APIRouter()

# Core authentication and session management
api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])

# Behavioral logging endpoints
api_router.include_router(logging.router, prefix="/log", tags=["behavioral-logging"])

# Real-time behavioral analysis via WebSocket
api_router.include_router(websocket.router, prefix="/ws", tags=["behavioral-analysis"])

# ML-Engine endpoints (if available)
if ML_ENDPOINTS_AVAILABLE:
    api_router.include_router(ml_engine.router, prefix="/ml", tags=["ml-engine"])

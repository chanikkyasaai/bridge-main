from fastapi import APIRouter
from app.api.v1.endpoints import auth, websocket, logging

api_router = APIRouter()

# Core authentication and session management
api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])

# Behavioral logging endpoints
api_router.include_router(logging.router, prefix="/log", tags=["behavioral-logging"])

# Real-time behavioral analysis via WebSocket
api_router.include_router(websocket.router, prefix="/ws", tags=["behavioral-analysis"])

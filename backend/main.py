from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
import os
import asyncio
from dotenv import load_dotenv

from app.api.v1.api import api_router
from app.core.config import settings
from app.core.session_manager import cleanup_sessions_task

# ML-Engine Integration
try:
    from ml_hooks import initialize_ml_integration, shutdown_ml_integration
    ML_INTEGRATION_AVAILABLE = True
except ImportError:
    async def initialize_ml_integration():
        return False
    async def shutdown_ml_integration():
        pass
    ML_INTEGRATION_AVAILABLE = False

# Load environment variables
load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting up Canara AI Security Backend...")
    print("Initializing behavioral analysis system...")
    
    # Create session buffers directory
    os.makedirs("session_buffers", exist_ok=True)
    
    # Initialize ML-Engine integration
    if ML_INTEGRATION_AVAILABLE:
        print("Initializing ML-Engine integration...")
        ml_success = await initialize_ml_integration()
        if ml_success:
            print("ML-Engine integration initialized successfully")
        else:
            print("Warning: ML-Engine integration failed to initialize")
    else:
        print("Warning: ML-Engine integration not available")
    
    # Start background task for session cleanup
    cleanup_task = asyncio.create_task(cleanup_sessions_task())
    
    yield
    
    # Shutdown
    print("Shutting down Canara AI Security Backend...")
    
    # Shutdown ML-Engine
    if ML_INTEGRATION_AVAILABLE:
        print("Shutting down ML-Engine integration...")
        await shutdown_ml_integration()
    
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=settings.DESCRIPTION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
)

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/")
async def root():
    return {
        "message": "Welcome to Canara AI Security Backend with Supabase Integration",
        "version": settings.VERSION,
        "description": "ML-powered behavioral analysis for banking security with Supabase storage",
        "features": [
            "Real-time behavioral analysis via WebSocket",
            "Session-based security monitoring",
            "MPIN verification system",
            "Supabase database integration",
            "Behavioral data storage in Supabase Storage",
            "Structured JSON logging with user_id/session_id"
        ],
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "auth": {
                "register": "POST /api/v1/auth/register",
                "login": "POST /api/v1/auth/login",
                "verify_mpin": "POST /api/v1/auth/verify-mpin",
                "logout": "POST /api/v1/auth/logout"
            },
            "logging": {
                "start_session": "POST /api/v1/log/start-session",
                "log_behavior": "POST /api/v1/log/behavior-data",
                "end_session": "POST /api/v1/log/end-session",
                "session_status": "GET /api/v1/log/session/{id}/status",
                "get_logs": "GET /api/v1/log/session/{id}/logs"
            },
            "websocket": "ws://localhost:8000/api/v1/ws/behavior/{session_id}?token={session_token}"
        },
        "supabase_integration": {
            "database_tables": ["users", "sessions", "security_events"],
            "storage_bucket": "behavior-logs",
            "log_format": "logs/{user_id}/{session_id}.json"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "canara-ai-backend"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="debug"
    )

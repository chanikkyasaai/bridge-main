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

# Load environment variables
load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting up Canara AI Security Backend...")
    print("Initializing behavioral analysis system...")
    
    # Create session buffers directory
    os.makedirs("session_buffers", exist_ok=True)
    
    # Start background task for session cleanup
    cleanup_task = asyncio.create_task(cleanup_sessions_task())
    
    yield
    
    # Shutdown
    print("Shutting down Canara AI Security Backend...")
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
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/")
async def root():
    return {
        "message": "Welcome to Canara AI Security Backend",
        "version": settings.VERSION,
        "description": "ML-powered behavioral analysis for banking security",
        "features": [
            "Real-time behavioral analysis",
            "Session-based security monitoring",
            "MPIN verification system",
            "WebSocket behavioral data collection"
        ],
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "websocket": "/api/v1/ws/behavior/{session_id}?token={session_token}"
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
        log_level="info"
    )

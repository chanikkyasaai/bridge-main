import logging
import logging.handlers
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
import os
import asyncio
import time
import json
from datetime import datetime
from dotenv import load_dotenv

from app.api.v1.api import api_router
from app.core.config import settings
from app.core.session_manager import cleanup_sessions_task

# Configure comprehensive logging for backend
detailed_formatter = logging.Formatter(
    '%(asctime)s | %(levelname)-8s | %(name)-15s | %(funcName)-15s | %(message)s'
)

# Setup backend log file with rotation
backend_log_handler = logging.handlers.RotatingFileHandler(
    'backend_detailed.log',
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
backend_log_handler.setFormatter(detailed_formatter)
backend_log_handler.setLevel(logging.DEBUG)

# Setup console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(detailed_formatter)
console_handler.setLevel(logging.INFO)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    handlers=[backend_log_handler, console_handler]
)

# Create specialized loggers
backend_logger = logging.getLogger("BACKEND")
api_logger = logging.getLogger("API_REQUESTS")
auth_logger = logging.getLogger("AUTHENTICATION")
security_logger = logging.getLogger("SECURITY")

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    backend_logger.info("🚀 BACKEND_STARTUP | Starting Canara AI Security Backend...")
    backend_logger.info("🔧 INITIALIZING | Behavioral analysis system...")
    
    # Create session buffers directory
    os.makedirs("session_buffers", exist_ok=True)
    backend_logger.info("📁 DIRECTORY_CREATED | session_buffers directory ready")
    
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Generate request ID
    request_id = f"{int(time.time() * 1000)}_{id(request)}"
    start_time = time.time()
    
    # Log incoming request
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")
    
    api_logger.info(f"📥 REQUEST_IN | ID: {request_id} | Method: {request.method} | Path: {request.url.path} | IP: {client_ip}")
    api_logger.debug(f"📋 REQUEST_DETAILS | ID: {request_id} | Headers: {dict(request.headers)} | UserAgent: {user_agent}")
    
    # Process request
    try:
        response = await call_next(request)
        duration = (time.time() - start_time) * 1000
        
        # Log response
        api_logger.info(f"📤 REQUEST_OUT | ID: {request_id} | Status: {response.status_code} | Duration: {duration:.2f}ms")
        
        return response
    except Exception as e:
        duration = (time.time() - start_time) * 1000
        security_logger.error(f"🚨 REQUEST_ERROR | ID: {request_id} | Error: {str(e)} | Duration: {duration:.2f}ms")
        raise

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

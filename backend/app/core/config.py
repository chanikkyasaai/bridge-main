import os
from typing import List
from pydantic import BaseSettings

class Settings(BaseSettings):
    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Canara AI Security Backend"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "ML-powered behavioral analysis backend for banking security"
    
    # Security
    SECRET_KEY: str = "change-this-secret-key-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    SESSION_EXPIRE_MINUTES: int = 60
    
    # MPIN Configuration
    MPIN_LENGTH: int = 6
    MAX_MPIN_ATTEMPTS: int = 3
    MPIN_LOCKOUT_MINUTES: int = 15
    
    # Behavioral Analysis
    BEHAVIOR_BUFFER_SIZE: int = 1000  # Number of events to store per session
    SUSPICIOUS_THRESHOLD: float = 0.7  # ML model threshold for suspicious behavior
    HIGH_RISK_THRESHOLD: float = 0.9   # Threshold to immediately block session
    
    # Session Management
    SESSION_CLEANUP_INTERVAL: int = 300  # 5 minutes
    
    # CORS
    BACKEND_CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8080"]
    
    # Development
    DEBUG: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()

"""
Application settings and configuration management.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    model_config = {
        'protected_namespaces': ('settings_',),
        'env_file': '.env',
        'case_sensitive': False,
        'extra': 'ignore'  # Allow extra fields to be ignored
    }
    
    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    # API Configuration
    api_host: str = Field(default="127.0.0.1", env="ML_ENGINE_HOST")
    api_port: int = Field(default=8001, env="ML_ENGINE_PORT")
    api_prefix: str = Field(default="/api/v1", env="API_PREFIX")
    
    # Database Configuration - Supabase
    supabase_url: str = Field(default="", env="SUPABASE_URL")
    supabase_service_key: str = Field(default="", env="SUPABASE_SERVICE_KEY")
    supabase_storage_bucket: str = Field(default="behavior-logs", env="SUPABASE_STORAGE_BUCKET")
    
    # Security Configuration
    secret_key: str = Field(default="canara-ai-ml-engine-secret-key-2025", env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # Behavioral Analysis Configuration
    similarity_threshold: float = Field(default=0.7, env="SIMILARITY_THRESHOLD")
    min_vectors_for_search: int = Field(default=5, env="MIN_VECTORS_FOR_SEARCH")
    vector_dimension: int = Field(default=90, env="VECTOR_DIMENSION")
    
    # Phase 1 Learning Configuration
    learning_session_threshold: int = Field(default=5, env="LEARNING_SESSION_THRESHOLD")
    gradual_risk_threshold: int = Field(default=15, env="GRADUAL_RISK_THRESHOLD")
    confidence_threshold: float = Field(default=0.7, env="CONFIDENCE_THRESHOLD")
    min_vectors_per_session: int = Field(default=3, env="MIN_VECTORS_PER_SESSION")
    
    # Phase 2 Continuous Analysis Configuration
    similarity_threshold_high: float = Field(default=0.85, env="SIMILARITY_THRESHOLD_HIGH")
    similarity_threshold_medium: float = Field(default=0.70, env="SIMILARITY_THRESHOLD_MEDIUM")
    similarity_threshold_low: float = Field(default=0.50, env="SIMILARITY_THRESHOLD_LOW")
    
    # Behavioral Drift Detection
    drift_detection_window: int = Field(default=20, env="DRIFT_DETECTION_WINDOW")
    drift_threshold: float = Field(default=0.15, env="DRIFT_THRESHOLD")
    baseline_adaptation_threshold: float = Field(default=0.3, env="BASELINE_ADAPTATION_THRESHOLD")
    
    # Ensemble Decision Weights
    faiss_layer_weight: float = Field(default=0.4, env="FAISS_LAYER_WEIGHT")
    gnn_transformer_weight: float = Field(default=0.6, env="GNN_TRANSFORMER_WEIGHT")
    
    # Storage Configuration
    vector_store_path: str = Field(default="./data/vectors", env="VECTOR_STORE_PATH")
    model_store_path: str = Field(default="./data/models", env="MODEL_STORE_PATH")
    
    # Development Configuration
    debug: bool = Field(default=True, env="DEBUG")
    reload: bool = Field(default=True, env="RELOAD")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # ML Configuration  
    faiss_index_type: str = Field(default="IndexFlatIP", env="FAISS_INDEX_TYPE")
    batch_size: int = Field(default=32, env="BATCH_SIZE")
    ml_update_interval: int = Field(default=3600, env="MODEL_UPDATE_INTERVAL")  # Renamed to avoid namespace conflict
    
    # FAISS Layer Settings
    similarity_threshold: float = Field(default=0.7, env="SIMILARITY_THRESHOLD")
    min_vectors_for_search: int = Field(default=3, env="MIN_VECTORS_FOR_SEARCH")
    
    # Adaptive Layer Settings
    adaptive_learning_rate: float = Field(default=0.01, env="ADAPTIVE_LEARNING_RATE")
    adaptation_threshold: float = Field(default=0.1, env="ADAPTATION_THRESHOLD")
    pattern_retention_days: int = Field(default=30, env="PATTERN_RETENTION_DAYS")
    min_feedback_samples: int = Field(default=3, env="MIN_FEEDBACK_SAMPLES")
    
    # Storage Configuration
    vector_storage_path: str = Field(default="./data/vectors/", env="VECTOR_STORAGE_PATH")
    hdf5_chunk_size: int = Field(default=1000, env="HDF5_CHUNK_SIZE")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    
    # Performance Configuration
    max_concurrent_users: int = Field(default=1000, env="MAX_CONCURRENT_USERS")
    rate_limit_per_minute: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")
    request_timeout: int = Field(default=30, env="REQUEST_TIMEOUT")


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the current settings instance."""
    return settings


def is_development() -> bool:
    """Check if running in development environment."""
    return settings.environment.lower() == "development"


def is_production() -> bool:
    """Check if running in production environment."""
    return settings.environment.lower() == "production"


def get_database_url() -> Optional[str]:
    """Get the database URL if configured."""
    if settings.supabase_url:
        return settings.supabase_url
    return None


def get_vector_storage_path() -> str:
    """Get the absolute path for vector storage."""
    path = os.path.abspath(settings.vector_storage_path)
    os.makedirs(path, exist_ok=True)
    return path

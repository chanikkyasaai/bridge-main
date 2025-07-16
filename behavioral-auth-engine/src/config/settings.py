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
        'case_sensitive': False
    }
    
    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8001, env="API_PORT")
    api_prefix: str = Field(default="/api/v1", env="API_PREFIX")
    
    # Database Configuration
    supabase_url: Optional[str] = Field(default=None, env="SUPABASE_URL")
    supabase_key: Optional[str] = Field(default=None, env="SUPABASE_KEY")
    supabase_service_key: Optional[str] = Field(default=None, env="SUPABASE_SERVICE_KEY")
    
    # Redis Configuration
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    redis_db: int = Field(default=0, env="REDIS_DB")
    
    # Security Configuration
    secret_key: str = Field(default="your_secret_key_here_change_in_production", env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # ML Configuration
    faiss_index_type: str = Field(default="IndexFlatIP", env="FAISS_INDEX_TYPE")
    vector_dimension: int = Field(default=90, env="VECTOR_DIMENSION")
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

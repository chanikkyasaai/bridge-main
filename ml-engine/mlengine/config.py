from pathlib import Path
from types import SimpleNamespace

# Base directory for all model and data files
BASE_DIR = Path(__file__).parent

_config_dict = {
    # Model and index paths
    "MODEL_BASE_PATH": str(BASE_DIR / "models"),
    "FAISS_INDEX_PATH": str(BASE_DIR / "models" / "faiss_index"),
    "TRANSFORMER_MODEL_PATH": str(BASE_DIR / "models" / "transformer"),
    "GNN_MODEL_PATH": str(BASE_DIR / "models" / "gnn"),
    "DRIFT_MODEL_PATH": str(BASE_DIR / "models" / "drift"),
    # Security and authentication
    "SESSION_TIMEOUT_MINUTES": 15,
    "MAX_CONCURRENT_SESSIONS": 1000,
    "MAX_WORKERS": 8,
    "COLD_START_THRESHOLD": 5,  # Number of events before full profile
    "DRIFT_DETECTION_WINDOW": 50,
    "DRIFT_ALERT_THRESHOLD": 0.7,
    "L1_FAISS_THRESHOLD": 0.85,
    "L2_TRANSFORMER_THRESHOLD": 0.80,
    "L2_GNN_THRESHOLD": 0.80,
    "RISK_SCORE_THRESHOLD": 0.75,
    # FAISS configuration
    "FAISS_INDEX_TYPE": "IndexFlatIP",
    "FAISS_NLIST": 100,
    "L1_HIGH_CONFIDENCE_THRESHOLD": 0.85,
    "L1_MEDIUM_CONFIDENCE_THRESHOLD": 0.70,
    "L1_LOW_CONFIDENCE_THRESHOLD": 0.55,
    "USER_PROFILES_PATH": str(BASE_DIR / "profiles"),
    # Layer 2 configuration
    "L2_ANALYSIS_TIMEOUT_MS": 80,
    # Logging and monitoring
    "LOG_LEVEL": "INFO",
    "AUDIT_LOG_PATH": str(BASE_DIR / "logs" / "audit.log"),
    "PERFORMANCE_LOG_PATH": str(BASE_DIR / "logs" / "performance.log"),
    # API and integration
    "API_HOST": "0.0.0.0",
    "API_PORT": 8080,
    "BACKEND_HOST": "localhost",
    "BACKEND_PORT": 8000,
    # Explainability
    "EXPLAINABILITY_ENABLED": True,
    # Miscellaneous
    "RANDOM_SEED": 42,
    "ENVIRONMENT": "production",
    # ML/Model-specific
    "BEHAVIORAL_VECTOR_DIM": 64,
    "TRANSFORMER_MAX_LENGTH": 128,
    "GNN_HIDDEN_DIM": 64,
    "GNN_DROPOUT": 0.1,
    "GRAPH_EMBEDDING_DIM": 32,
    "MAX_SESSION_NODES": 100,
}

class Config(SimpleNamespace):
    def __getitem__(self, key):
        return getattr(self, key)
    def get(self, key, default=None):
        return getattr(self, key, default)

CONFIG = Config(**_config_dict)

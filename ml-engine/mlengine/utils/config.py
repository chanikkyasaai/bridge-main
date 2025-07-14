"""
BRIDGE ML-Engine Configuration
Behavioral Risk Intelligence for Dynamic Guarded Entry

Configuration settings for the ML components
"""

import os
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class BRIDGEConfig:
    """Main configuration for BRIDGE ML-Engine"""
    
    # Model Paths
    MODEL_BASE_PATH: str = "models/"
    FAISS_INDEX_PATH: str = "faiss/indexes/"
    TRANSFORMER_MODEL_PATH: str = "transformers/"
    GNN_MODEL_PATH: str = "gnn/"
    
    # FAISS Configuration
    FAISS_INDEX_TYPE: str = "IndexFlatIP"  # Inner Product for cosine similarity
    FAISS_DIMENSION: int = 128
    FAISS_NLIST: int = 100  # For IVF indexes
    FAISS_NPROBE: int = 10
    
    # Vector Configuration
    BEHAVIORAL_VECTOR_DIM: int = 128
    SLIDING_WINDOW_SIZE: int = 2  # seconds
    VECTOR_BATCH_SIZE: int = 32
    
    # Layer 1 (FAISS) Thresholds
    L1_HIGH_CONFIDENCE_THRESHOLD: float = 0.85
    L1_MEDIUM_CONFIDENCE_THRESHOLD: float = 0.65
    L1_LOW_CONFIDENCE_THRESHOLD: float = 0.45
    
    # Layer 2 (Transformer + GNN) Configuration
    TRANSFORMER_MODEL_NAME: str = "distilbert-base-uncased"
    TRANSFORMER_MAX_LENGTH: int = 512
    GNN_HIDDEN_DIM: int = 64
    GNN_NUM_LAYERS: int = 3
    GNN_DROPOUT: float = 0.1
    
    # Risk Scoring
    RISK_WEIGHTS: Dict[str, float] = None
    HIGH_RISK_THRESHOLD: float = 0.8
    MEDIUM_RISK_THRESHOLD: float = 0.5
    
    # Drift Detection
    DRIFT_DETECTION_WINDOW: int = 100  # samples
    DRIFT_THRESHOLD: float = 0.1
    DRIFT_WARNING_THRESHOLD: float = 0.05
    
    # Session Graph
    MAX_SESSION_NODES: int = 1000
    GRAPH_EMBEDDING_DIM: int = 64
    
    # Performance
    INFERENCE_BATCH_SIZE: int = 16
    MAX_CONCURRENT_SESSIONS: int = 1000
    CACHE_SIZE: int = 10000
    
    # Security
    ENCRYPTION_KEY_SIZE: int = 256
    VECTOR_ENCRYPTION: bool = True
    
    def __post_init__(self):
        if self.RISK_WEIGHTS is None:
            self.RISK_WEIGHTS = {
                "faiss_similarity": 0.3,
                "transformer_confidence": 0.25,
                "gnn_anomaly": 0.25,
                "drift_score": 0.1,
                "context_score": 0.1
            }

# Global configuration instance
CONFIG = BRIDGEConfig()

# Environment-specific overrides
if os.getenv("BRIDGE_ENV") == "production":
    CONFIG.FAISS_NLIST = 500
    CONFIG.MAX_CONCURRENT_SESSIONS = 5000
    CONFIG.CACHE_SIZE = 50000
elif os.getenv("BRIDGE_ENV") == "development":
    CONFIG.FAISS_NLIST = 50
    CONFIG.MAX_CONCURRENT_SESSIONS = 100
    CONFIG.CACHE_SIZE = 1000

"""
ML model configurations and hyperparameters.
"""

from typing import Dict, Any
from dataclasses import dataclass
from src.utils.constants import *


@dataclass
class FAISSConfig:
    """Configuration for FAISS vector search."""
    index_type: str = DEFAULT_FAISS_INDEX_TYPE
    dimension: int = TOTAL_VECTOR_DIM
    nprobe: int = 10  # number of clusters to search
    nlist: int = 100  # number of clusters for IndexIVFFlat
    similarity_threshold: float = 0.7


@dataclass
class TransformerConfig:
    """Configuration for transformer-based behavioral analysis."""
    model_name: str = "distilbert-base-uncased"
    max_length: int = 128
    hidden_size: int = 256
    num_attention_heads: int = 8
    num_hidden_layers: int = 6
    dropout_rate: float = 0.1
    learning_rate: float = 2e-5
    batch_size: int = DEFAULT_BATCH_SIZE


@dataclass
class DriftDetectorConfig:
    """Configuration for behavioral drift detection."""
    window_size: int = 50  # number of recent sessions to consider
    drift_threshold: float = DRIFT_DETECTION_THRESHOLD
    min_samples: int = 10  # minimum samples needed for drift detection
    statistical_test: str = "ks_test"  # kolmogorov-smirnov test
    confidence_level: float = 0.95


@dataclass
class PolicyEngineConfig:
    """Configuration for risk-based policy decisions."""
    new_user_threshold: float = NEW_USER_RISK_THRESHOLD
    moderate_user_threshold: float = MODERATE_USER_RISK_THRESHOLD
    experienced_user_threshold: float = EXPERIENCED_USER_RISK_THRESHOLD
    false_positive_tolerance: float = 0.05
    adaptive_threshold_enabled: bool = True
    threshold_adjustment_rate: float = 0.01


@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing."""
    typing_window_size: int = TYPING_WINDOW_SIZE
    touch_window_size: int = TOUCH_WINDOW_SIZE
    navigation_window_size: int = NAVIGATION_WINDOW_SIZE
    normalization_method: str = "min_max"  # "min_max", "z_score", "robust"
    outlier_detection_method: str = "iqr"  # "iqr", "z_score", "isolation_forest"
    outlier_threshold: float = 3.0


@dataclass
class VectorStorageConfig:
    """Configuration for vector storage."""
    storage_backend: str = "hdf5"  # "hdf5", "redis", "postgresql"
    chunk_size: int = DEFAULT_HDF5_CHUNK_SIZE
    compression: str = "gzip"
    compression_level: int = 6
    max_memory_usage: int = 1024  # MB
    cache_size: int = 100  # number of user profiles to cache


class MLConfig:
    """Central ML configuration class."""
    
    def __init__(self):
        self.faiss = FAISSConfig()
        self.transformer = TransformerConfig()
        self.drift_detector = DriftDetectorConfig()
        self.policy_engine = PolicyEngineConfig()
        self.preprocessing = PreprocessingConfig()
        self.vector_storage = VectorStorageConfig()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "faiss": self.faiss.__dict__,
            "transformer": self.transformer.__dict__,
            "drift_detector": self.drift_detector.__dict__,
            "policy_engine": self.policy_engine.__dict__,
            "preprocessing": self.preprocessing.__dict__,
            "vector_storage": self.vector_storage.__dict__,
        }
    
    def update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration from dictionary."""
        for component_name, component_config in config_dict.items():
            if hasattr(self, component_name):
                component = getattr(self, component_name)
                for key, value in component_config.items():
                    if hasattr(component, key):
                        setattr(component, key, value)


# Global ML configuration instance
ml_config = MLConfig()


def get_ml_config() -> MLConfig:
    """Get the current ML configuration instance."""
    return ml_config

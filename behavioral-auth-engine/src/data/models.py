"""
Data models and schemas for behavioral authentication.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator
from enum import Enum
import uuid


class SessionPhase(str, Enum):
    """Session learning phases."""
    LEARNING = "learning"
    GRADUAL_RISK = "gradual_risk"
    FULL_AUTH = "full_auth"


class LearningPhase(str, Enum):
    """Learning phases for user behavioral profile development"""
    COLD_START = "cold_start"
    LEARNING = "learning"
    GRADUAL_RISK = "gradual_risk"
    FULL_AUTH = "full_auth"


class RiskLevel(str, Enum):
    """Risk assessment levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AuthenticationDecision(str, Enum):
    """Authentication decision outcomes."""
    ALLOW = "allow"
    CHALLENGE = "challenge"
    BLOCK = "block"
    LEARN = "learn"


class BehavioralFeatures(BaseModel):
    """Raw behavioral features from user interaction."""
    
    # Typing features (25 dimensions)
    typing_speed: float = Field(..., description="Average typing speed in WPM")
    keystroke_intervals: List[float] = Field(..., description="Intervals between keystrokes")
    typing_rhythm_variance: float = Field(..., description="Variance in typing rhythm")
    backspace_frequency: float = Field(..., description="Frequency of backspace usage")
    typing_pressure: List[float] = Field(default=[], description="Typing pressure patterns")
    
    # Touch features (30 dimensions)
    touch_pressure: List[float] = Field(..., description="Touch pressure patterns")
    touch_duration: List[float] = Field(..., description="Touch duration patterns")
    touch_area: List[float] = Field(..., description="Touch contact area patterns")
    swipe_velocity: List[float] = Field(..., description="Swipe velocity patterns")
    touch_coordinates: List[Dict[str, float]] = Field(..., description="Touch coordinate patterns")
    
    # Navigation features (20 dimensions)
    navigation_patterns: List[str] = Field(..., description="App navigation sequences")
    screen_time_distribution: Dict[str, float] = Field(..., description="Time spent on different screens")
    interaction_frequency: float = Field(..., description="Frequency of user interactions")
    session_duration: float = Field(..., description="Session duration in seconds")
    
    # Contextual features (15 dimensions)
    device_orientation: str = Field(..., description="Device orientation")
    time_of_day: int = Field(..., description="Hour of the day (0-23)")
    day_of_week: int = Field(..., description="Day of week (0-6)")
    location_context: Optional[str] = Field(default=None, description="Location context if available")
    app_version: str = Field(..., description="Application version")
    
    @field_validator('keystroke_intervals', 'typing_pressure', 'touch_pressure', 'touch_duration', 
              'touch_area', 'swipe_velocity')
    @classmethod
    def validate_feature_arrays(cls, v):
        """Validate that feature arrays are not empty and contain valid values."""
        if not v:
            return []
        if any(val < 0 for val in v if isinstance(val, (int, float))):
            raise ValueError("Feature values cannot be negative")
        return v


class BehavioralVector(BaseModel):
    """Processed behavioral vector for ML operations."""
    
    user_id: str = Field(..., description="User identifier")
    session_id: str = Field(..., description="Session identifier")
    vector: List[float] = Field(..., description="90-dimensional behavioral vector")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0)
    feature_source: BehavioralFeatures = Field(..., description="Source features")
    
    @field_validator('vector')
    @classmethod
    def validate_vector_dimension(cls, v):
        """Validate vector has correct dimensions."""
        from src.utils.constants import TOTAL_VECTOR_DIM
        if len(v) != TOTAL_VECTOR_DIM:
            raise ValueError(f"Vector must have {TOTAL_VECTOR_DIM} dimensions, got {len(v)}")
        return v


class UserProfile(BaseModel):
    """User behavioral profile and history."""
    
    user_id: str = Field(..., description="User identifier")
    session_count: int = Field(default=0, description="Total number of sessions")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Learning phase information
    current_phase: SessionPhase = Field(default=SessionPhase.LEARNING)
    risk_threshold: float = Field(default=0.8, description="Current risk threshold")
    
    # Behavioral statistics
    baseline_vector: Optional[List[float]] = Field(default=None, description="Baseline behavioral vector")
    vector_variance: Optional[List[float]] = Field(default=None, description="Vector variance")
    false_positive_rate: float = Field(default=0.0, description="Historical false positive rate")
    
    # Recent session history
    recent_vectors: List[BehavioralVector] = Field(default=[], description="Recent behavioral vectors")
    drift_score: float = Field(default=0.0, description="Behavioral drift score")
    
    def add_session_vector(self, vector: BehavioralVector) -> None:
        """Add a new session vector to the profile."""
        self.recent_vectors.append(vector)
        self.session_count += 1
        self.updated_at = datetime.utcnow()
        
        # Keep only last 50 vectors for memory efficiency
        if len(self.recent_vectors) > 50:
            self.recent_vectors = self.recent_vectors[-50:]
    
    def update_phase(self) -> None:
        """Update the learning phase based on session count."""
        from src.utils.constants import LEARNING_PHASE_SESSIONS, GRADUAL_RISK_SESSIONS
        
        if self.session_count <= LEARNING_PHASE_SESSIONS:
            self.current_phase = SessionPhase.LEARNING
        elif self.session_count <= GRADUAL_RISK_SESSIONS:
            self.current_phase = SessionPhase.GRADUAL_RISK
        else:
            self.current_phase = SessionPhase.FULL_AUTH


class AuthenticationRequest(BaseModel):
    """Authentication request with behavioral data."""
    
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = Field(..., description="User identifier")
    session_id: str = Field(..., description="Session identifier")
    behavioral_data: BehavioralFeatures = Field(..., description="Raw behavioral features")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Request metadata
    device_id: Optional[str] = Field(default=None, description="Device identifier")
    ip_address: Optional[str] = Field(default=None, description="Client IP address")
    user_agent: Optional[str] = Field(default=None, description="Client user agent")


class AuthenticationResponse(BaseModel):
    """Authentication response with decision and explanation."""
    
    request_id: str = Field(..., description="Request identifier")
    user_id: str = Field(..., description="User identifier")
    decision: AuthenticationDecision = Field(..., description="Authentication decision")
    risk_level: RiskLevel = Field(..., description="Risk assessment level")
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Risk score (0-1)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Decision confidence")
    
    # Timing information
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Explanation
    decision_factors: List[str] = Field(default=[], description="Factors influencing the decision")
    similarity_scores: Dict[str, float] = Field(default={}, description="Similarity scores by layer")
    
    # Session information
    session_phase: SessionPhase = Field(..., description="Current learning phase")
    session_count: int = Field(..., description="User's total session count")


class DriftDetectionResult(BaseModel):
    """Result of behavioral drift detection."""
    
    user_id: str = Field(..., description="User identifier")
    drift_detected: bool = Field(..., description="Whether drift was detected")
    drift_score: float = Field(..., ge=0.0, le=1.0, description="Drift severity score")
    drift_components: Dict[str, float] = Field(default={}, description="Drift by feature component")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Statistical information
    test_statistic: float = Field(..., description="Statistical test result")
    p_value: float = Field(..., description="Statistical significance")
    confidence_level: float = Field(default=0.95, description="Confidence level used")


class ModelPerformanceMetrics(BaseModel):
    """Performance metrics for ML models."""
    
    model_config = {"protected_namespaces": ()}
    
    model_name: str = Field(..., description="Model identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Accuracy metrics
    true_positives: int = Field(default=0)
    false_positives: int = Field(default=0)
    true_negatives: int = Field(default=0)
    false_negatives: int = Field(default=0)
    
    # Computed metrics
    @property
    def accuracy(self) -> float:
        total = self.true_positives + self.false_positives + self.true_negatives + self.false_negatives
        if total == 0:
            return 0.0
        return (self.true_positives + self.true_negatives) / total
    
    @property
    def precision(self) -> float:
        if (self.true_positives + self.false_positives) == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)
    
    @property
    def recall(self) -> float:
        if (self.true_positives + self.false_negatives) == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)
    
    @property
    def f1_score(self) -> float:
        if (self.precision + self.recall) == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)
    
    # Performance metrics
    avg_response_time_ms: float = Field(default=0.0, description="Average response time")
    throughput_rps: float = Field(default=0.0, description="Requests per second")
    memory_usage_mb: float = Field(default=0.0, description="Memory usage in MB")

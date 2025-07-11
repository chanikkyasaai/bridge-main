"""
Behavioral Vector Processing Module
Handles real-time behavioral data vectorization and encoding
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from collections import deque

from ml_engine.config import CONFIG

logger = logging.getLogger(__name__)

@dataclass
class BehavioralEvent:
    """Single behavioral event structure"""
    timestamp: datetime
    event_type: str  # 'touch', 'swipe', 'keystroke', 'navigation'
    features: Dict[str, float]
    session_id: str
    user_id: str

@dataclass
class BehavioralVector:
    """Processed behavioral vector"""
    vector: np.ndarray
    timestamp: datetime
    confidence: float
    session_id: str
    user_id: str
    source_events: List[BehavioralEvent]

class BehavioralEncoder(nn.Module):
    """Neural encoder for behavioral features"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Tanh()  # Normalize to [-1, 1]
        )
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=output_dim,
            hidden_size=output_dim // 2,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        self.final_projection = nn.Linear(output_dim // 2, output_dim)
        
    def forward(self, x: torch.Tensor, temporal_context: Optional[torch.Tensor] = None):
        """
        Args:
            x: Current behavioral features [batch_size, input_dim]
            temporal_context: Previous vectors [batch_size, seq_len, output_dim]
        """
        # Encode current features
        encoded = self.encoder(x)
        
        if temporal_context is not None:
            # Add temporal modeling
            combined = torch.cat([temporal_context, encoded.unsqueeze(1)], dim=1)
            lstm_out, _ = self.lstm(combined)
            # Take the last output
            final_encoding = self.final_projection(lstm_out[:, -1, :])
        else:
            final_encoding = encoded
            
        # L2 normalize for cosine similarity
        return torch.nn.functional.normalize(final_encoding, p=2, dim=1)

class FeatureExtractor:
    """Extracts and normalizes features from behavioral events"""
    
    TOUCH_FEATURES = [
        'x_coord', 'y_coord', 'pressure', 'touch_major', 'touch_minor',
        'orientation', 'tool_type', 'dwell_time'
    ]
    
    SWIPE_FEATURES = [
        'start_x', 'start_y', 'end_x', 'end_y', 'velocity', 'acceleration',
        'path_deviation', 'gesture_time', 'pressure_variation'
    ]
    
    KEYSTROKE_FEATURES = [
        'key_code', 'dwell_time', 'flight_time', 'pressure', 
        'typing_speed', 'correction_count', 'pause_ratio'
    ]
    
    NAVIGATION_FEATURES = [
        'screen_id', 'transition_time', 'scroll_velocity', 'interaction_density',
        'revisit_count', 'time_on_screen'
    ]
    
    def __init__(self):
        self.feature_stats = {}  # For normalization
        self.feature_dim = len(self.TOUCH_FEATURES) + len(self.SWIPE_FEATURES) + \
                          len(self.KEYSTROKE_FEATURES) + len(self.NAVIGATION_FEATURES)
    
    def extract_features(self, event: BehavioralEvent) -> np.ndarray:
        """Extract normalized feature vector from behavioral event"""
        features = np.zeros(self.feature_dim)
        
        if event.event_type == 'touch':
            features[:len(self.TOUCH_FEATURES)] = self._extract_touch_features(event.features)
        elif event.event_type == 'swipe':
            start_idx = len(self.TOUCH_FEATURES)
            features[start_idx:start_idx+len(self.SWIPE_FEATURES)] = \
                self._extract_swipe_features(event.features)
        elif event.event_type == 'keystroke':
            start_idx = len(self.TOUCH_FEATURES) + len(self.SWIPE_FEATURES)
            features[start_idx:start_idx+len(self.KEYSTROKE_FEATURES)] = \
                self._extract_keystroke_features(event.features)
        elif event.event_type == 'navigation':
            start_idx = len(self.TOUCH_FEATURES) + len(self.SWIPE_FEATURES) + len(self.KEYSTROKE_FEATURES)
            features[start_idx:start_idx+len(self.NAVIGATION_FEATURES)] = \
                self._extract_navigation_features(event.features)
        
        return self._normalize_features(features, event.event_type)
    
    def _extract_touch_features(self, raw_features: Dict[str, float]) -> np.ndarray:
        """Extract touch-specific features"""
        features = np.zeros(len(self.TOUCH_FEATURES))
        for i, feature_name in enumerate(self.TOUCH_FEATURES):
            features[i] = raw_features.get(feature_name, 0.0)
        return features
    
    def _extract_swipe_features(self, raw_features: Dict[str, float]) -> np.ndarray:
        """Extract swipe-specific features"""
        features = np.zeros(len(self.SWIPE_FEATURES))
        for i, feature_name in enumerate(self.SWIPE_FEATURES):
            features[i] = raw_features.get(feature_name, 0.0)
        return features
    
    def _extract_keystroke_features(self, raw_features: Dict[str, float]) -> np.ndarray:
        """Extract keystroke-specific features"""
        features = np.zeros(len(self.KEYSTROKE_FEATURES))
        for i, feature_name in enumerate(self.KEYSTROKE_FEATURES):
            features[i] = raw_features.get(feature_name, 0.0)
        return features
    
    def _extract_navigation_features(self, raw_features: Dict[str, float]) -> np.ndarray:
        """Extract navigation-specific features"""
        features = np.zeros(len(self.NAVIGATION_FEATURES))
        for i, feature_name in enumerate(self.NAVIGATION_FEATURES):
            features[i] = raw_features.get(feature_name, 0.0)
        return features
    
    def _normalize_features(self, features: np.ndarray, event_type: str) -> np.ndarray:
        """Normalize features to [0, 1] range"""
        # Simple min-max normalization (in production, use learned statistics)
        normalized = np.clip(features / (features.max() + 1e-8), 0, 1)
        return normalized

class BehavioralVectorProcessor:
    """Main processor for converting behavioral events to vectors"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.feature_extractor = FeatureExtractor()
        self.encoder = BehavioralEncoder(
            input_dim=self.feature_extractor.feature_dim,
            output_dim=CONFIG.BEHAVIORAL_VECTOR_DIM
        )
        
        if model_path:
            self.load_model(model_path)
        
        # Sliding window buffer for temporal context
        self.temporal_buffers = {}  # session_id -> deque of vectors
        self.max_temporal_context = 10
        
    async def initialize(self):
        """Initialize the behavioral vector processor"""
        logger.info("Initializing Behavioral Vector Processor...")
        
        # Initialize encoder in evaluation mode
        self.encoder.eval()
        
        # Initialize temporal buffers
        self.temporal_buffers = {}
        
        logger.info("✓ Behavioral Vector Processor initialized")
    
    async def process_events(self, events: List[BehavioralEvent]) -> List[BehavioralVector]:
        """Process a batch of behavioral events into vectors (async version)"""
        if not events:
            return []
        
        # Validate events
        self._validate_events(events)
        
        # Group events by session
        session_events = {}
        for event in events:
            if event.session_id not in session_events:
                session_events[event.session_id] = []
            session_events[event.session_id].append(event)
        
        vectors = []
        for session_id, session_event_list in session_events.items():
            session_vectors = await self._process_session_events(session_id, session_event_list)
            vectors.extend(session_vectors)
        
        return vectors
    
    def _validate_events(self, events: List[BehavioralEvent]):
        """Validate event data"""
        if not events:
            raise ValueError("Empty event list")
        
        for event in events:
            # Check for invalid features
            for key, value in event.features.items():
                if np.isnan(value) or np.isinf(value):
                    raise ValueError(f"Invalid feature value: {key}={value}")
        
        # Check timestamp ordering
        timestamps = [event.timestamp for event in events]
        if timestamps != sorted(timestamps):
            raise ValueError("Events must be in chronological order")
    
    async def _process_session_events(self, session_id: str, events: List[BehavioralEvent]) -> List[BehavioralVector]:
        """Process events for a single session"""
        vectors = []
        
        for i, event in enumerate(events):
            # Extract features
            features = self.feature_extractor.extract_features(event)
            
            # Convert to tensor
            feature_tensor = torch.FloatTensor(features).unsqueeze(0)
            
            # Encode to behavioral vector
            with torch.no_grad():
                vector_tensor = self.encoder(feature_tensor)
                vector = vector_tensor.squeeze().numpy()
            
            # Calculate confidence based on feature consistency
            confidence = self._calculate_confidence(features, session_id)
            
            # Create behavioral vector
            behavioral_vector = BehavioralVector(
                vector=vector,
                timestamp=event.timestamp,
                confidence=confidence,
                session_id=session_id,
                user_id=event.user_id,
                source_events=[event]
            )
            
            vectors.append(behavioral_vector)
            
            # Update temporal buffer
            self._update_temporal_buffer(session_id, behavioral_vector)
        
        return vectors

    def _calculate_confidence(self, features: np.ndarray, session_id: str) -> float:
        """Calculate confidence score for the behavioral vector"""
        # Base confidence
        confidence = 0.8
        
        # Reduce confidence for extreme values
        if np.any(features > 3.0) or np.any(features < -3.0):
            confidence -= 0.2
        
        # Check consistency with recent vectors in session
        if session_id in self.temporal_buffers and len(self.temporal_buffers[session_id]) > 0:
            recent_vectors = [v.vector for v in list(self.temporal_buffers[session_id])[-3:]]
            current_vector = features  # Use features as proxy for vector
            
            # Calculate similarity to recent vectors
            similarities = []
            for recent_vector in recent_vectors:
                sim = np.dot(current_vector, recent_vector[:len(current_vector)]) / (
                    np.linalg.norm(current_vector) * np.linalg.norm(recent_vector[:len(current_vector)]) + 1e-8
                )
                similarities.append(sim)
            
            if similarities:
                avg_similarity = np.mean(similarities)
                if avg_similarity < 0.3:  # Low similarity to recent behavior
                    confidence -= 0.3
                elif avg_similarity > 0.8:  # High similarity
                    confidence += 0.1
        
        return max(0.1, min(1.0, confidence))
    
    def _update_temporal_buffer(self, session_id: str, vector: BehavioralVector):
        """Update temporal buffer for session"""
        if session_id not in self.temporal_buffers:
            self.temporal_buffers[session_id] = deque(maxlen=self.max_temporal_context)
        
        self.temporal_buffers[session_id].append(vector)

    def clear_session_buffer(self, session_id: str):
        """Clear temporal buffer for a session"""
        if session_id in self.temporal_buffers:
            del self.temporal_buffers[session_id]
    
    def save_model(self, path: str):
        """Save the encoder model"""
        torch.save(self.encoder.state_dict(), path)
        logger.info(f"Behavioral encoder saved to {path}")
    
    def load_model(self, path: str):
        """Load the encoder model"""
        self.encoder.load_state_dict(torch.load(path))
        self.encoder.eval()
        logger.info(f"Behavioral encoder loaded from {path}")

# Utility functions for behavioral data validation
def validate_behavioral_event(event: BehavioralEvent) -> bool:
    """Validate behavioral event for basic sanity checks"""
    if not event.timestamp or not event.event_type or not event.session_id:
        return False
    
    if not event.features:
        return False
    
    # Check for reasonable timestamp
    now = datetime.now()
    if abs((now - event.timestamp).total_seconds()) > 3600:  # 1 hour threshold
        return False
    
    return True

def is_synthetic_behavior(features: Dict[str, float]) -> bool:
    """Detect obviously synthetic/bot behavior"""
    # Check for perfectly regular timing (bot behavior)
    if 'dwell_time' in features and features['dwell_time'] > 0:
        if features['dwell_time'] % 10 == 0:  # Suspiciously round numbers
            return True
    
    # Check for impossible pressure values
    if 'pressure' in features:
        if features['pressure'] < 0 or features['pressure'] > 1:
            return True
    
    # Check for impossible coordinates
    if 'x_coord' in features and 'y_coord' in features:
        if features['x_coord'] < 0 or features['y_coord'] < 0:
            return True
    
    return False

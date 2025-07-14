"""
Behavioral Preprocessing Pipeline
Handles preprocessing of behavioral data for banking security ML models
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import asyncio

logger = logging.getLogger(__name__)

class BehavioralPreprocessingPipeline:
    """Production-ready behavioral data preprocessing pipeline"""
    
    def __init__(self):
        self.is_initialized = False
        self.feature_scalers = {}
        self.feature_statistics = {}
        
        # Preprocessing configuration
        self.config = {
            "touch_pressure_range": (0.0, 1.0),
            "velocity_range": (0.0, 10.0),
            "duration_range": (0.01, 5.0),
            "coordinate_normalization": True,
            "outlier_threshold": 3.0  # Standard deviations
        }
        
        logger.info("🔧 Behavioral Preprocessing Pipeline initialized")
    
    async def initialize(self):
        """Initialize the preprocessing pipeline"""
        try:
            # Load any pre-trained scalers or statistics
            await self._load_preprocessing_models()
            
            self.is_initialized = True
            logger.info("✅ Behavioral Preprocessing Pipeline ready")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize preprocessing pipeline: {e}")
            raise
    
    async def preprocess_behavioral_events(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Preprocess a list of behavioral events"""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            processed_events = []
            
            for event in events:
                processed_event = await self._preprocess_single_event(event)
                if processed_event:  # Only add valid events
                    processed_events.append(processed_event)
            
            logger.debug(f"Preprocessed {len(processed_events)} events from {len(events)} input events")
            return processed_events
            
        except Exception as e:
            logger.error(f"Error preprocessing behavioral events: {e}")
            return []
    
    async def _preprocess_single_event(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Preprocess a single behavioral event"""
        try:
            processed_event = event.copy()
            features = event.get('features', {})
            
            # Normalize touch pressure
            if 'pressure' in features:
                features['pressure'] = self._normalize_value(
                    features['pressure'], 
                    self.config["touch_pressure_range"]
                )
            
            # Normalize velocity
            if 'velocity' in features:
                features['velocity'] = self._normalize_value(
                    features['velocity'], 
                    self.config["velocity_range"]
                )
            
            # Normalize duration
            if 'duration' in features:
                features['duration'] = self._normalize_value(
                    features['duration'], 
                    self.config["duration_range"]
                )
            
            # Normalize coordinates if enabled
            if self.config["coordinate_normalization"]:
                if 'x_coordinate' in features and 'y_coordinate' in features:
                    # Assume screen dimensions for normalization
                    screen_width = features.get('screen_width', 1080)
                    screen_height = features.get('screen_height', 1920)
                    
                    features['x_coordinate'] = features['x_coordinate'] / screen_width
                    features['y_coordinate'] = features['y_coordinate'] / screen_height
            
            # Remove outliers
            if not self._is_outlier(features):
                processed_event['features'] = features
                return processed_event
            else:
                logger.debug(f"Filtered out outlier event: {event.get('event_type', 'unknown')}")
                return None
                
        except Exception as e:
            logger.warning(f"Error preprocessing event: {e}")
            return None
    
    def _normalize_value(self, value: float, value_range: Tuple[float, float]) -> float:
        """Normalize a value to 0-1 range"""
        min_val, max_val = value_range
        
        # Clamp to range
        value = max(min_val, min(max_val, value))
        
        # Normalize to 0-1
        if max_val > min_val:
            return (value - min_val) / (max_val - min_val)
        else:
            return 0.5  # Default if range is invalid
    
    def _is_outlier(self, features: Dict[str, Any]) -> bool:
        """Check if features contain outlier values"""
        try:
            # Check for impossible values
            pressure = features.get('pressure', 0.5)
            velocity = features.get('velocity', 1.0)
            duration = features.get('duration', 0.1)
            
            # Basic sanity checks
            if pressure < 0 or pressure > 1:
                return True
            if velocity < 0 or velocity > 20:  # Very high velocity
                return True
            if duration < 0.001 or duration > 10:  # Very short or long duration
                return True
                
            return False
            
        except Exception:
            return True  # Treat errors as outliers
    
    async def extract_behavioral_features(self, events: List[Dict[str, Any]]) -> np.ndarray:
        """Extract feature vector from behavioral events"""
        try:
            if not events:
                return np.zeros(64)  # Default feature vector size
            
            # Aggregate features across events
            features = []
            
            # Basic statistical features
            pressures = [e.get('features', {}).get('pressure', 0.5) for e in events]
            velocities = [e.get('features', {}).get('velocity', 1.0) for e in events]
            durations = [e.get('features', {}).get('duration', 0.1) for e in events]
            
            # Statistical measures
            features.extend([
                np.mean(pressures), np.std(pressures), np.min(pressures), np.max(pressures),
                np.mean(velocities), np.std(velocities), np.min(velocities), np.max(velocities),
                np.mean(durations), np.std(durations), np.min(durations), np.max(durations)
            ])
            
            # Event type distribution
            event_types = [e.get('event_type', 'unknown') for e in events]
            type_counts = {
                'touch': event_types.count('touch'),
                'swipe': event_types.count('swipe'),
                'type': event_types.count('type'),
                'scroll': event_types.count('scroll'),
                'tap': event_types.count('tap')
            }
            
            total_events = len(events)
            features.extend([
                type_counts['touch'] / total_events,
                type_counts['swipe'] / total_events,
                type_counts['type'] / total_events,
                type_counts['scroll'] / total_events,
                type_counts['tap'] / total_events
            ])
            
            # Temporal features
            if len(events) > 1:
                timestamps = [datetime.fromisoformat(e.get('timestamp', datetime.now().isoformat())) for e in events]
                time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() for i in range(len(timestamps)-1)]
                
                features.extend([
                    np.mean(time_diffs),
                    np.std(time_diffs),
                    np.min(time_diffs),
                    np.max(time_diffs)
                ])
            else:
                features.extend([0.1, 0.0, 0.1, 0.1])  # Default values
            
            # Pad or truncate to 64 features
            while len(features) < 64:
                features.append(0.0)
            
            return np.array(features[:64], dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error extracting behavioral features: {e}")
            return np.zeros(64, dtype=np.float32)
    
    async def _load_preprocessing_models(self):
        """Load any pre-trained preprocessing models"""
        try:
            # Placeholder for loading pre-trained scalers or statistics
            # In production, this would load from saved files
            logger.debug("Preprocessing models loaded (placeholder)")
            
        except Exception as e:
            logger.warning(f"Could not load preprocessing models: {e}")
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get preprocessing pipeline statistics"""
        return {
            "pipeline_type": "Behavioral Preprocessing Pipeline",
            "initialized": self.is_initialized,
            "configuration": self.config,
            "feature_vector_size": 64
        }

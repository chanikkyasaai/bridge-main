"""
Feature Extractor for Behavioral Authentication
Converts behavioral logs to 48-dimensional feature vectors
"""

import numpy as np
from typing import List, Dict, Any, Tuple
import logging
from collections import defaultdict
from datetime import datetime, timedelta
import math

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Extracts features from behavioral events and creates normalized vectors"""
    
    def __init__(self, vector_dimensions: int = 48):
        self.vector_dimensions = vector_dimensions
        self.feature_config = {
            # Touch patterns (10 dimensions)
            'touch_coords_x_mean': 0,
            'touch_coords_y_mean': 1,
            'touch_coords_x_std': 2,
            'touch_coords_y_std': 3,
            'touch_pressure_mean': 4,
            'touch_duration_mean': 5,
            'touch_duration_std': 6,
            'inter_touch_gap_mean': 7,
            'inter_touch_gap_std': 8,
            'touch_frequency': 9,
            
            # Accelerometer (8 dimensions)
            'accel_x_mean': 10,
            'accel_y_mean': 11,
            'accel_z_mean': 12,
            'accel_magnitude_mean': 13,
            'accel_magnitude_std': 14,
            'accel_jerk_mean': 15,
            'accel_stability': 16,
            'accel_variance': 17,
            
            # Gyroscope (8 dimensions)
            'gyro_x_mean': 18,
            'gyro_y_mean': 19,
            'gyro_z_mean': 20,
            'gyro_magnitude_mean': 21,
            'gyro_magnitude_std': 22,
            'gyro_angular_velocity': 23,
            'gyro_rotation_stability': 24,
            'gyro_variance': 25,
            
            # Scroll patterns (6 dimensions)
            'scroll_velocity_mean': 26,
            'scroll_velocity_std': 27,
            'scroll_direction_consistency': 28,
            'scroll_distance_mean': 29,
            'scroll_frequency': 30,
            'scroll_smoothness': 31,
            
            # Typing patterns (8 dimensions)
            'typing_speed_mean': 32,
            'typing_speed_std': 33,
            'keystroke_rhythm_consistency': 34,
            'delete_ratio': 35,
            'words_per_minute': 36,
            'typing_pressure_mean': 37,
            'first_key_delay_mean': 38,
            'typing_pattern_regularity': 39,
            
            # Context patterns (6 dimensions)
            'brightness_level': 40,
            'orientation_changes': 41,
            'navigation_pattern_consistency': 42,
            'session_context_stability': 43,
            'device_interaction_rhythm': 44,
            'overall_activity_level': 45,
            
            # Statistical features (2 dimensions)
            'temporal_pattern_consistency': 46,
            'behavioral_entropy': 47
        }
    
    async def extract_features(self, events: List[Dict[str, Any]], 
                             window_duration: int = 15) -> np.ndarray:
        """
        Extract 48-dimensional feature vector from behavioral events
        
        Args:
            events: List of behavioral events
            window_duration: Time window in seconds
            
        Returns:
            48-dimensional numpy array
        """
        try:
            # Initialize feature vector
            features = np.zeros(self.vector_dimensions)
            
            if not events:
                return features
            
            # Group events by type
            event_groups = self._group_events_by_type(events)
            
            # Extract touch features (10 dimensions)
            touch_features = self._extract_touch_features(event_groups.get('touch', []))
            features[0:10] = touch_features
            
            # Extract accelerometer features (8 dimensions)
            accel_features = self._extract_accel_features(event_groups.get('accel', []))
            features[10:18] = accel_features
            
            # Extract gyroscope features (8 dimensions)
            gyro_features = self._extract_gyro_features(event_groups.get('gyro', []))
            features[18:26] = gyro_features
            
            # Extract scroll features (6 dimensions)
            scroll_features = self._extract_scroll_features(event_groups.get('scroll', []))
            features[26:32] = scroll_features
            
            # Extract typing features (8 dimensions)
            typing_features = self._extract_typing_features(event_groups.get('typing', []))
            features[32:40] = typing_features
            
            # Extract context features (6 dimensions)
            context_features = self._extract_context_features(event_groups, events)
            features[40:46] = context_features
            
            # Extract statistical features (2 dimensions)
            statistical_features = self._extract_statistical_features(events)
            features[46:48] = statistical_features
            
            # Normalize features
            features = self._normalize_features(features)
            
            logger.info(f"Extracted {len(features)}-dimensional feature vector from {len(events)} events")
            return features
            
        except Exception as e:
            logger.error(f"Failed to extract features: {e}")
            return np.zeros(self.vector_dimensions)
    
    def _group_events_by_type(self, events: List[Dict]) -> Dict[str, List[Dict]]:
        """Group events by behavioral type"""
        groups = defaultdict(list)
        
        for event in events:
            event_type = event.get('event_type', '')
            
            if 'touch' in event_type:
                groups['touch'].append(event)
            elif 'accel' in event_type:
                groups['accel'].append(event)
            elif 'gyro' in event_type:
                groups['gyro'].append(event)
            elif 'scroll' in event_type:
                groups['scroll'].append(event)
            elif 'typing' in event_type:
                groups['typing'].append(event)
            else:
                groups['context'].append(event)
                
        return dict(groups)
    
    def _extract_touch_features(self, touch_events: List[Dict]) -> np.ndarray:
        """Extract touch-related features (10 dimensions)"""
        features = np.zeros(10)
        
        if not touch_events:
            return features
        
        # Extract coordinate and pressure data
        coords_x, coords_y, pressures, durations, gaps = [], [], [], [], []
        
        prev_timestamp = None
        for event in touch_events:
            data = event.get('data', {})
            
            if 'coordinates' in data:
                coords_x.append(data['coordinates'][0])
                coords_y.append(data['coordinates'][1])
            
            if 'pressure' in data:
                pressures.append(data['pressure'])
            
            if 'touch_duration_ms' in data:
                durations.append(data['touch_duration_ms'])
            
            if 'inter_touch_gap_ms' in data:
                gaps.append(data['inter_touch_gap_ms'])
        
        # Calculate features
        if coords_x:
            features[0] = np.mean(coords_x) / 1000.0  # Normalize coordinates
            features[1] = np.mean(coords_y) / 1000.0
            features[2] = np.std(coords_x) / 1000.0 if len(coords_x) > 1 else 0
            features[3] = np.std(coords_y) / 1000.0 if len(coords_y) > 1 else 0
        
        if pressures:
            features[4] = np.mean(pressures)
        
        if durations:
            features[5] = np.mean(durations) / 1000.0  # Convert to seconds
            features[6] = np.std(durations) / 1000.0 if len(durations) > 1 else 0
        
        if gaps:
            features[7] = np.mean(gaps) / 1000.0  # Convert to seconds
            features[8] = np.std(gaps) / 1000.0 if len(gaps) > 1 else 0
        
        # Touch frequency (touches per second)
        if touch_events:
            time_span = self._calculate_time_span(touch_events)
            features[9] = len(touch_events) / max(time_span, 1)
        
        return features
    
    def _extract_accel_features(self, accel_events: List[Dict]) -> np.ndarray:
        """Extract accelerometer features (8 dimensions)"""
        features = np.zeros(8)
        
        if not accel_events:
            return features
        
        # Extract accelerometer data
        x_vals, y_vals, z_vals, magnitudes = [], [], [], []
        
        for event in accel_events:
            data = event.get('data', {})
            x, y, z = data.get('x', 0), data.get('y', 0), data.get('z', 0)
            
            x_vals.append(x)
            y_vals.append(y)
            z_vals.append(z)
            magnitudes.append(math.sqrt(x*x + y*y + z*z))
        
        if x_vals:
            features[0] = np.mean(x_vals)
            features[1] = np.mean(y_vals)
            features[2] = np.mean(z_vals)
            features[3] = np.mean(magnitudes)
            features[4] = np.std(magnitudes) if len(magnitudes) > 1 else 0
            
            # Calculate jerk (rate of change of acceleration)
            if len(magnitudes) > 1:
                jerk = np.diff(magnitudes)
                features[5] = np.mean(np.abs(jerk))
            
            # Stability (inverse of variance)
            features[6] = 1.0 / (1.0 + np.var(magnitudes))
            features[7] = np.var(magnitudes)
        
        return features
    
    def _extract_gyro_features(self, gyro_events: List[Dict]) -> np.ndarray:
        """Extract gyroscope features (8 dimensions)"""
        features = np.zeros(8)
        
        if not gyro_events:
            return features
        
        # Extract gyroscope data
        x_vals, y_vals, z_vals, magnitudes = [], [], [], []
        
        for event in gyro_events:
            data = event.get('data', {})
            x, y, z = data.get('x', 0), data.get('y', 0), data.get('z', 0)
            
            x_vals.append(x)
            y_vals.append(y)
            z_vals.append(z)
            magnitudes.append(math.sqrt(x*x + y*y + z*z))
        
        if x_vals:
            features[0] = np.mean(x_vals)
            features[1] = np.mean(y_vals)
            features[2] = np.mean(z_vals)
            features[3] = np.mean(magnitudes)
            features[4] = np.std(magnitudes) if len(magnitudes) > 1 else 0
            
            # Angular velocity consistency
            features[5] = np.mean(magnitudes)
            
            # Rotation stability
            features[6] = 1.0 / (1.0 + np.var(magnitudes))
            features[7] = np.var(magnitudes)
        
        return features
    
    def _extract_scroll_features(self, scroll_events: List[Dict]) -> np.ndarray:
        """Extract scroll features (6 dimensions)"""
        features = np.zeros(6)
        
        if not scroll_events:
            return features
        
        velocities, distances = [], []
        directions = []  # 1 for down, -1 for up, 0 for no change
        
        prev_pixels = None
        for event in scroll_events:
            data = event.get('data', {})
            
            if 'velocity' in data:
                velocities.append(abs(data['velocity']))
            
            pixels = data.get('pixels', 0)
            if prev_pixels is not None:
                diff = pixels - prev_pixels
                distances.append(abs(diff))
                directions.append(1 if diff > 0 else (-1 if diff < 0 else 0))
            prev_pixels = pixels
        
        if velocities:
            features[0] = np.mean(velocities) / 1000.0  # Normalize
            features[1] = np.std(velocities) / 1000.0 if len(velocities) > 1 else 0
        
        if directions:
            # Direction consistency (how often user scrolls in same direction)
            direction_changes = sum(1 for i in range(1, len(directions)) 
                                  if directions[i] != directions[i-1])
            features[2] = 1.0 - (direction_changes / max(len(directions), 1))
        
        if distances:
            features[3] = np.mean(distances) / 100.0  # Normalize
        
        # Scroll frequency
        if scroll_events:
            time_span = self._calculate_time_span(scroll_events)
            features[4] = len(scroll_events) / max(time_span, 1)
        
        # Scroll smoothness (inverse of velocity variance)
        if velocities and len(velocities) > 1:
            features[5] = 1.0 / (1.0 + np.var(velocities))
        
        return features
    
    def _extract_typing_features(self, typing_events: List[Dict]) -> np.ndarray:
        """Extract typing features (8 dimensions)"""
        features = np.zeros(8)
        
        if not typing_events:
            return features
        
        speeds, rhythms, pressures, delays, delete_counts, wpm_values = [], [], [], [], [], []
        
        for event in typing_events:
            data = event.get('data', {})
            
            if 'typing_speed' in data:
                speeds.append(data['typing_speed'])
            
            if 'keystroke_dynamics' in data and data['keystroke_dynamics']:
                rhythms.extend(data['keystroke_dynamics'])
            
            if 'touch_pressure' in data:
                pressures.append(data['touch_pressure'])
            
            if 'first_key_delay_ms' in data:
                delays.append(data['first_key_delay_ms'])
            
            if 'delete_count' in data:
                delete_counts.append(data['delete_count'])
            
            if 'words_per_minute' in data:
                wpm_values.append(data['words_per_minute'])
        
        if speeds:
            features[0] = np.mean(speeds)
            features[1] = np.std(speeds) if len(speeds) > 1 else 0
        
        if rhythms:
            # Rhythm consistency (inverse of variance in keystroke timing)
            features[2] = 1.0 / (1.0 + np.var(rhythms))
        
        if delete_counts and len(typing_events) > 0:
            # Delete ratio
            total_keystrokes = sum(event.get('data', {}).get('keystroke_count', 0) 
                                 for event in typing_events)
            features[3] = sum(delete_counts) / max(total_keystrokes, 1)
        
        if wpm_values:
            features[4] = np.mean(wpm_values)
        
        if pressures:
            features[5] = np.mean(pressures)
        
        if delays:
            features[6] = np.mean(delays) / 1000.0  # Convert to seconds
        
        # Typing pattern regularity
        if speeds and len(speeds) > 1:
            features[7] = 1.0 / (1.0 + np.std(speeds))
        
        return features
    
    def _extract_context_features(self, event_groups: Dict, all_events: List[Dict]) -> np.ndarray:
        """Extract contextual features (6 dimensions)"""
        features = np.zeros(6)
        
        # Brightness level
        brightness_events = [e for e in all_events if e.get('event_type') == 'brightness_change']
        if brightness_events:
            brightness_levels = [e.get('data', {}).get('brightness_level', 0.5) 
                               for e in brightness_events]
            features[0] = np.mean(brightness_levels)
        
        # Orientation changes
        orientation_events = [e for e in all_events if e.get('event_type') == 'orientation_change']
        features[1] = len(orientation_events) / max(len(all_events), 1)
        
        # Navigation pattern consistency
        nav_events = [e for e in all_events if e.get('event_type') == 'navigation_pattern']
        if nav_events:
            # Simple consistency measure based on navigation frequency
            time_span = self._calculate_time_span(nav_events)
            features[2] = len(nav_events) / max(time_span, 1)
        
        # Session context stability (fewer context changes = more stable)
        context_changes = len(brightness_events) + len(orientation_events)
        features[3] = 1.0 / (1.0 + context_changes)
        
        # Device interaction rhythm
        if all_events:
            timestamps = [self._parse_timestamp(e.get('timestamp')) for e in all_events]
            timestamps = [t for t in timestamps if t is not None]
            
            if len(timestamps) > 1:
                intervals = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                           for i in range(len(timestamps)-1)]
                features[4] = 1.0 / (1.0 + np.std(intervals)) if intervals else 0
        
        # Overall activity level
        if all_events:
            time_span = self._calculate_time_span(all_events)
            features[5] = len(all_events) / max(time_span, 1)
        
        return features
    
    def _extract_statistical_features(self, events: List[Dict]) -> np.ndarray:
        """Extract statistical features (2 dimensions)"""
        features = np.zeros(2)
        
        if not events:
            return features
        
        # Temporal pattern consistency
        timestamps = [self._parse_timestamp(e.get('timestamp')) for e in events]
        timestamps = [t for t in timestamps if t is not None]
        
        if len(timestamps) > 2:
            intervals = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                        for i in range(len(timestamps)-1)]
            features[0] = 1.0 / (1.0 + np.std(intervals)) if intervals else 0
        
        # Behavioral entropy (diversity of event types)
        event_types = [e.get('event_type', '') for e in events]
        unique_types = set(event_types)
        
        if unique_types:
            type_counts = {t: event_types.count(t) for t in unique_types}
            total = len(event_types)
            
            # Calculate Shannon entropy
            entropy = -sum((count/total) * math.log2(count/total) 
                          for count in type_counts.values())
            features[1] = entropy / math.log2(len(unique_types))  # Normalized
        
        return features
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features to [0, 1] range"""
        # Clip extreme values
        features = np.clip(features, -10, 10)
        
        # Apply tanh normalization for smooth bounded output
        normalized = (np.tanh(features) + 1) / 2
        
        return normalized
    
    def _calculate_time_span(self, events: List[Dict]) -> float:
        """Calculate time span of events in seconds"""
        if len(events) < 2:
            return 1.0
        
        timestamps = [self._parse_timestamp(e.get('timestamp')) for e in events]
        timestamps = [t for t in timestamps if t is not None]
        
        if len(timestamps) < 2:
            return 1.0
        
        return (max(timestamps) - min(timestamps)).total_seconds() or 1.0
    
    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse timestamp string to datetime object"""
        try:
            # Handle different timestamp formats
            if timestamp_str:
                return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except:
            pass
        return None

    async def extract_session_vector(self, all_session_events: List[Dict]) -> np.ndarray:
        """
        Extract final session vector for clustering/storage
        Uses entire session data to create representative vector
        """
        return await self.extract_features(all_session_events, window_duration=None)

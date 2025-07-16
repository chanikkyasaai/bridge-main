"""
Behavioral data processing and feature extraction.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import math
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from src.data.models import BehavioralFeatures, BehavioralVector
from src.config.ml_config import get_ml_config
from src.utils.constants import *


class BehavioralProcessor:
    """Processes raw behavioral data into feature vectors."""
    
    def __init__(self):
        self.ml_config = get_ml_config()
        self.scaler = self._get_scaler()
        self._feature_cache = {}
    
    def _get_scaler(self):
        """Get the appropriate scaler based on configuration."""
        normalization_method = self.ml_config.preprocessing.normalization_method
        
        if normalization_method == "min_max":
            return MinMaxScaler()
        elif normalization_method == "z_score":
            return StandardScaler()
        elif normalization_method == "robust":
            return RobustScaler()
        else:
            return MinMaxScaler()  # Default
    
    async def process_behavioral_data(
        self, 
        behavioral_data: BehavioralFeatures, 
        user_id: str,
        session_id: str
    ) -> BehavioralVector:
        """Process raw behavioral data into a feature vector."""
        
        # Extract features from each domain
        typing_features = self._extract_typing_features(behavioral_data)
        touch_features = self._extract_touch_features(behavioral_data)
        navigation_features = self._extract_navigation_features(behavioral_data)
        contextual_features = self._extract_contextual_features(behavioral_data)
        
        # Combine all features
        combined_features = (
            typing_features + 
            touch_features + 
            navigation_features + 
            contextual_features
        )
        
        # Ensure we have exactly the right number of features
        if len(combined_features) != TOTAL_VECTOR_DIM:
            # Pad or truncate to match expected dimensions
            if len(combined_features) < TOTAL_VECTOR_DIM:
                combined_features.extend([0.0] * (TOTAL_VECTOR_DIM - len(combined_features)))
            else:
                combined_features = combined_features[:TOTAL_VECTOR_DIM]
        
        # Apply normalization
        normalized_features = self._normalize_features(combined_features)
        
        # Detect and handle outliers
        cleaned_features = self._handle_outliers(normalized_features)
        
        return BehavioralVector(
            user_id=user_id,
            session_id=session_id,
            vector=cleaned_features,
            feature_source=behavioral_data,
            confidence_score=self._calculate_confidence_score(behavioral_data)
        )
    
    def _extract_typing_features(self, data: BehavioralFeatures) -> List[float]:
        """Extract typing-related features (25 dimensions)."""
        features = []
        
        # Basic typing metrics
        features.append(data.typing_speed)
        features.append(data.typing_rhythm_variance)
        features.append(data.backspace_frequency)
        
        # Keystroke interval statistics
        if data.keystroke_intervals:
            intervals = np.array(data.keystroke_intervals)
            features.extend([
                np.mean(intervals),
                np.std(intervals),
                np.median(intervals),
                np.min(intervals),
                np.max(intervals),
                stats.skew(intervals) if len(intervals) > 2 else 0.0,
                stats.kurtosis(intervals) if len(intervals) > 3 else 0.0
            ])
        else:
            features.extend([0.0] * 7)
        
        # Typing pressure statistics
        if data.typing_pressure:
            pressure = np.array(data.typing_pressure)
            features.extend([
                np.mean(pressure),
                np.std(pressure),
                np.median(pressure),
                np.min(pressure),
                np.max(pressure)
            ])
        else:
            features.extend([0.0] * 5)
        
        # Advanced typing patterns
        features.extend([
            self._calculate_typing_rhythm_consistency(data.keystroke_intervals),
            self._calculate_typing_speed_variance(data.keystroke_intervals),
            self._calculate_burst_typing_pattern(data.keystroke_intervals),
            self._calculate_pause_pattern(data.keystroke_intervals),
            self._calculate_pressure_consistency(data.typing_pressure),
            self._calculate_typing_acceleration(data.keystroke_intervals),
            self._calculate_inter_key_variability(data.keystroke_intervals),
            self._calculate_typing_flow_score(data.keystroke_intervals),
            self._calculate_error_correction_pattern(data.backspace_frequency)
        ])
        
        # Ensure exactly 25 features
        return self._pad_or_truncate(features, TYPING_FEATURES_DIM)
    
    def _extract_touch_features(self, data: BehavioralFeatures) -> List[float]:
        """Extract touch-related features (30 dimensions)."""
        features = []
        
        # Touch pressure statistics
        if data.touch_pressure:
            pressure = np.array(data.touch_pressure)
            features.extend([
                np.mean(pressure),
                np.std(pressure),
                np.median(pressure),
                np.min(pressure),
                np.max(pressure),
                stats.skew(pressure) if len(pressure) > 2 else 0.0,
                stats.kurtosis(pressure) if len(pressure) > 3 else 0.0
            ])
        else:
            features.extend([0.0] * 7)
        
        # Touch duration statistics
        if data.touch_duration:
            duration = np.array(data.touch_duration)
            features.extend([
                np.mean(duration),
                np.std(duration),
                np.median(duration),
                np.min(duration),
                np.max(duration)
            ])
        else:
            features.extend([0.0] * 5)
        
        # Touch area statistics
        if data.touch_area:
            area = np.array(data.touch_area)
            features.extend([
                np.mean(area),
                np.std(area),
                np.median(area),
                np.min(area),
                np.max(area)
            ])
        else:
            features.extend([0.0] * 5)
        
        # Swipe velocity statistics
        if data.swipe_velocity:
            velocity = np.array(data.swipe_velocity)
            features.extend([
                np.mean(velocity),
                np.std(velocity),
                np.median(velocity),
                np.min(velocity),
                np.max(velocity)
            ])
        else:
            features.extend([0.0] * 5)
        
        # Advanced touch patterns
        features.extend([
            self._calculate_touch_consistency(data.touch_pressure, data.touch_duration),
            self._calculate_touch_rhythm(data.touch_duration),
            self._calculate_pressure_duration_correlation(data.touch_pressure, data.touch_duration),
            self._calculate_touch_area_variance(data.touch_area),
            self._calculate_swipe_pattern_consistency(data.swipe_velocity),
            self._calculate_touch_coordinate_patterns(data.touch_coordinates),
            self._calculate_touch_force_distribution(data.touch_pressure),
            self._calculate_multi_touch_patterns(data.touch_coordinates)
        ])
        
        # Ensure exactly 30 features
        return self._pad_or_truncate(features, TOUCH_FEATURES_DIM)
    
    def _extract_navigation_features(self, data: BehavioralFeatures) -> List[float]:
        """Extract navigation-related features (20 dimensions)."""
        features = []
        
        # Basic navigation metrics
        features.extend([
            len(data.navigation_patterns),
            data.interaction_frequency,
            data.session_duration
        ])
        
        # Screen time distribution analysis
        if data.screen_time_distribution:
            screen_times = list(data.screen_time_distribution.values())
            features.extend([
                np.mean(screen_times),
                np.std(screen_times) if len(screen_times) > 1 else 0.0,
                max(screen_times) if screen_times else 0.0,
                min(screen_times) if screen_times else 0.0,
                len(data.screen_time_distribution)  # Number of unique screens visited
            ])
        else:
            features.extend([0.0] * 5)
        
        # Navigation pattern analysis
        features.extend([
            self._calculate_navigation_entropy(data.navigation_patterns),
            self._calculate_navigation_sequence_consistency(data.navigation_patterns),
            self._calculate_screen_transition_speed(data.screen_time_distribution),
            self._calculate_navigation_depth(data.navigation_patterns),
            self._calculate_backtrack_frequency(data.navigation_patterns),
            self._calculate_focus_distribution(data.screen_time_distribution),
            self._calculate_interaction_burst_pattern(data.interaction_frequency),
            self._calculate_session_engagement_score(data.session_duration, data.interaction_frequency),
            self._calculate_navigation_efficiency(data.navigation_patterns),
            self._calculate_user_journey_complexity(data.navigation_patterns),
            self._calculate_attention_span_metric(data.screen_time_distribution),
            self._calculate_multitasking_indicator(data.screen_time_distribution)
        ])
        
        # Ensure exactly 20 features
        return self._pad_or_truncate(features, NAVIGATION_FEATURES_DIM)
    
    def _extract_contextual_features(self, data: BehavioralFeatures) -> List[float]:
        """Extract contextual features (15 dimensions)."""
        features = []
        
        # Time-based features
        features.extend([
            data.time_of_day / 24.0,  # Normalize to 0-1
            data.day_of_week / 7.0,   # Normalize to 0-1
            math.sin(2 * math.pi * data.time_of_day / 24),  # Circular encoding for hour
            math.cos(2 * math.pi * data.time_of_day / 24),
            math.sin(2 * math.pi * data.day_of_week / 7),   # Circular encoding for day
            math.cos(2 * math.pi * data.day_of_week / 7)
        ])
        
        # Device orientation (encoded as binary features)
        orientation_features = [0.0, 0.0, 0.0]  # portrait, landscape, unknown
        if data.device_orientation == "portrait":
            orientation_features[0] = 1.0
        elif data.device_orientation == "landscape":
            orientation_features[1] = 1.0
        else:
            orientation_features[2] = 1.0
        features.extend(orientation_features)
        
        # App version (hash to numeric value)
        app_version_hash = hash(data.app_version) % 1000 / 1000.0  # Normalize to 0-1
        features.append(app_version_hash)
        
        # Location context (if available)
        location_feature = 0.0
        if data.location_context:
            location_feature = hash(data.location_context) % 100 / 100.0
        features.append(location_feature)
        
        # Advanced contextual patterns
        features.extend([
            self._calculate_time_consistency_score(data.time_of_day),
            self._calculate_usage_pattern_score(data.time_of_day, data.day_of_week),
            self._calculate_device_familiarity_score(data.device_orientation),
            self._calculate_app_version_stability_score(data.app_version)
        ])
        
        # Ensure exactly 15 features
        return self._pad_or_truncate(features, CONTEXTUAL_FEATURES_DIM)
    
    def _normalize_features(self, features: List[float]) -> List[float]:
        """Normalize features using the configured method."""
        features_array = np.array(features).reshape(1, -1)
        
        # Handle any infinite or NaN values
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=1.0, neginf=0.0)
        
        try:
            normalized = self.scaler.fit_transform(features_array)
            return normalized[0].tolist()
        except Exception:
            # Fallback to simple min-max normalization
            min_val, max_val = min(features), max(features)
            if max_val == min_val:
                return [0.5] * len(features)  # All values are the same
            return [(f - min_val) / (max_val - min_val) for f in features]
    
    def _handle_outliers(self, features: List[float]) -> List[float]:
        """Detect and handle outliers in feature vectors."""
        method = self.ml_config.preprocessing.outlier_detection_method
        threshold = self.ml_config.preprocessing.outlier_threshold
        
        features_array = np.array(features)
        
        if method == "iqr":
            Q1 = np.percentile(features_array, 25)
            Q3 = np.percentile(features_array, 75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            # Clip outliers
            features_array = np.clip(features_array, lower_bound, upper_bound)
            
        elif method == "z_score":
            z_scores = np.abs(stats.zscore(features_array))
            # Replace outliers with median
            median_val = np.median(features_array)
            features_array[z_scores > threshold] = median_val
        
        return features_array.tolist()
    
    def _calculate_confidence_score(self, data: BehavioralFeatures) -> float:
        """Calculate confidence score based on data quality."""
        score = 1.0
        
        # Reduce confidence for missing data
        if not data.keystroke_intervals:
            score -= 0.2
        if not data.typing_pressure:
            score -= 0.2
        if not data.touch_pressure:
            score -= 0.1
        if not data.navigation_patterns:
            score -= 0.1
        if not data.screen_time_distribution:
            score -= 0.1
        
        # Reduce confidence for very short sessions
        if data.session_duration < 30:  # Less than 30 seconds
            score -= 0.2
        
        # Reduce confidence for very low interaction
        if data.interaction_frequency < 0.1:
            score -= 0.1
        
        return max(0.1, score)  # Minimum confidence of 0.1
    
    def _pad_or_truncate(self, features: List[float], target_length: int) -> List[float]:
        """Pad with zeros or truncate to match target length."""
        if len(features) < target_length:
            features.extend([0.0] * (target_length - len(features)))
        elif len(features) > target_length:
            features = features[:target_length]
        return features
    
    # Helper methods for advanced feature extraction
    def _calculate_typing_rhythm_consistency(self, intervals: List[float]) -> float:
        """Calculate consistency of typing rhythm."""
        if not intervals or len(intervals) < 3:
            return 0.0
        return 1.0 / (1.0 + np.std(intervals))
    
    def _calculate_typing_speed_variance(self, intervals: List[float]) -> float:
        """Calculate variance in typing speed."""
        if not intervals or len(intervals) < 2:
            return 0.0
        speeds = [1.0 / (interval + 0.001) for interval in intervals]  # Avoid division by zero
        return np.var(speeds)
    
    def _calculate_burst_typing_pattern(self, intervals: List[float]) -> float:
        """Detect burst typing patterns."""
        if not intervals or len(intervals) < 5:
            return 0.0
        
        # Count rapid consecutive keystrokes (< 0.1s intervals)
        rapid_count = sum(1 for interval in intervals if interval < 0.1)
        return rapid_count / len(intervals)
    
    def _calculate_pause_pattern(self, intervals: List[float]) -> float:
        """Detect pause patterns in typing."""
        if not intervals:
            return 0.0
        
        # Count long pauses (> 0.5s intervals)
        pause_count = sum(1 for interval in intervals if interval > 0.5)
        return pause_count / len(intervals)
    
    def _calculate_pressure_consistency(self, pressure: List[float]) -> float:
        """Calculate consistency of typing pressure."""
        if not pressure or len(pressure) < 2:
            return 0.0
        return 1.0 / (1.0 + np.std(pressure))
    
    def _calculate_typing_acceleration(self, intervals: List[float]) -> float:
        """Calculate typing acceleration patterns."""
        if not intervals or len(intervals) < 3:
            return 0.0
        
        accelerations = []
        for i in range(1, len(intervals) - 1):
            accel = intervals[i+1] - intervals[i]
            accelerations.append(accel)
        
        return np.mean(np.abs(accelerations)) if accelerations else 0.0
    
    def _calculate_inter_key_variability(self, intervals: List[float]) -> float:
        """Calculate variability between different key combinations."""
        if not intervals or len(intervals) < 2:
            return 0.0
        return np.std(intervals) / (np.mean(intervals) + 0.001)
    
    def _calculate_typing_flow_score(self, intervals: List[float]) -> float:
        """Calculate overall typing flow score."""
        if not intervals:
            return 0.0
        
        # Combine rhythm consistency and speed consistency
        rhythm_score = self._calculate_typing_rhythm_consistency(intervals)
        speed_variance = self._calculate_typing_speed_variance(intervals)
        
        return rhythm_score / (1.0 + speed_variance)
    
    def _calculate_error_correction_pattern(self, backspace_frequency: float) -> float:
        """Analyze error correction patterns."""
        # Higher frequency indicates more corrections/errors
        return min(1.0, backspace_frequency * 10)  # Scale to 0-1
    
    def _calculate_touch_consistency(self, pressure: List[float], duration: List[float]) -> float:
        """Calculate touch behavior consistency."""
        if not pressure or not duration:
            return 0.0
        
        pressure_consistency = 1.0 / (1.0 + np.std(pressure))
        duration_consistency = 1.0 / (1.0 + np.std(duration))
        
        return (pressure_consistency + duration_consistency) / 2.0
    
    def _calculate_touch_rhythm(self, duration: List[float]) -> float:
        """Calculate touch rhythm patterns."""
        if not duration or len(duration) < 3:
            return 0.0
        
        return 1.0 / (1.0 + np.std(duration))
    
    def _calculate_pressure_duration_correlation(self, pressure: List[float], duration: List[float]) -> float:
        """Calculate correlation between pressure and duration."""
        if not pressure or not duration or len(pressure) != len(duration) or len(pressure) < 2:
            return 0.0
        
        try:
            correlation, _ = stats.pearsonr(pressure, duration)
            return abs(correlation) if not np.isnan(correlation) else 0.0
        except:
            return 0.0
    
    def _calculate_touch_area_variance(self, area: List[float]) -> float:
        """Calculate variance in touch contact area."""
        if not area or len(area) < 2:
            return 0.0
        return np.var(area)
    
    def _calculate_swipe_pattern_consistency(self, velocity: List[float]) -> float:
        """Calculate swipe pattern consistency."""
        if not velocity or len(velocity) < 2:
            return 0.0
        return 1.0 / (1.0 + np.std(velocity))
    
    def _calculate_touch_coordinate_patterns(self, coordinates: List[Dict[str, float]]) -> float:
        """Analyze touch coordinate patterns."""
        if not coordinates or len(coordinates) < 2:
            return 0.0
        
        # Calculate average distance between consecutive touches
        distances = []
        for i in range(1, len(coordinates)):
            prev = coordinates[i-1]
            curr = coordinates[i]
            if 'x' in prev and 'y' in prev and 'x' in curr and 'y' in curr:
                dist = math.sqrt((curr['x'] - prev['x'])**2 + (curr['y'] - prev['y'])**2)
                distances.append(dist)
        
        return np.mean(distances) if distances else 0.0
    
    def _calculate_touch_force_distribution(self, pressure: List[float]) -> float:
        """Analyze touch force distribution patterns."""
        if not pressure:
            return 0.0
        
        # Calculate coefficient of variation
        mean_pressure = np.mean(pressure)
        if mean_pressure == 0:
            return 0.0
        
        return np.std(pressure) / mean_pressure
    
    def _calculate_multi_touch_patterns(self, coordinates: List[Dict[str, float]]) -> float:
        """Detect multi-touch usage patterns."""
        if not coordinates:
            return 0.0
        
        # Simple heuristic: frequency of coordinate changes
        return min(1.0, len(coordinates) / 100.0)  # Normalize to 0-1
    
    def _calculate_navigation_entropy(self, patterns: List[str]) -> float:
        """Calculate entropy of navigation patterns."""
        if not patterns:
            return 0.0
        
        # Count frequency of each pattern
        freq_dist = {}
        for pattern in patterns:
            freq_dist[pattern] = freq_dist.get(pattern, 0) + 1
        
        # Calculate entropy
        total = len(patterns)
        entropy = 0.0
        for count in freq_dist.values():
            probability = count / total
            if probability > 0:
                entropy -= probability * math.log2(probability)
        
        return entropy
    
    def _calculate_navigation_sequence_consistency(self, patterns: List[str]) -> float:
        """Calculate consistency of navigation sequences."""
        if not patterns or len(patterns) < 2:
            return 0.0
        
        # Calculate transition consistency
        transitions = {}
        for i in range(len(patterns) - 1):
            transition = (patterns[i], patterns[i+1])
            transitions[transition] = transitions.get(transition, 0) + 1
        
        if not transitions:
            return 0.0
        
        # Higher consistency = more repeated transitions
        max_transitions = max(transitions.values())
        total_transitions = sum(transitions.values())
        
        return max_transitions / total_transitions
    
    def _calculate_screen_transition_speed(self, screen_times: Dict[str, float]) -> float:
        """Calculate average screen transition speed."""
        if not screen_times or len(screen_times) < 2:
            return 0.0
        
        times = list(screen_times.values())
        avg_time = np.mean(times)
        
        # Faster transitions = higher score
        return 1.0 / (avg_time + 1.0)  # Avoid division by zero
    
    def _calculate_navigation_depth(self, patterns: List[str]) -> float:
        """Calculate navigation depth complexity."""
        if not patterns:
            return 0.0
        
        unique_screens = len(set(patterns))
        total_navigations = len(patterns)
        
        return unique_screens / max(1, total_navigations)
    
    def _calculate_backtrack_frequency(self, patterns: List[str]) -> float:
        """Calculate frequency of backtracking in navigation."""
        if not patterns or len(patterns) < 2:
            return 0.0
        
        backtrack_count = 0
        for i in range(2, len(patterns)):
            if patterns[i] == patterns[i-2]:  # Returned to previous screen
                backtrack_count += 1
        
        return backtrack_count / max(1, len(patterns) - 2)
    
    def _calculate_focus_distribution(self, screen_times: Dict[str, float]) -> float:
        """Calculate focus distribution across screens."""
        if not screen_times:
            return 0.0
        
        times = list(screen_times.values())
        if len(times) < 2:
            return 1.0
        
        # Calculate coefficient of variation
        mean_time = np.mean(times)
        if mean_time == 0:
            return 0.0
        
        return np.std(times) / mean_time
    
    def _calculate_interaction_burst_pattern(self, frequency: float) -> float:
        """Detect interaction burst patterns."""
        # Simple frequency-based heuristic
        return min(1.0, frequency * 2.0)  # Scale to 0-1
    
    def _calculate_session_engagement_score(self, duration: float, frequency: float) -> float:
        """Calculate overall session engagement score."""
        if duration == 0:
            return 0.0
        
        # Combine duration and interaction frequency
        normalized_duration = min(1.0, duration / 300.0)  # Normalize to 5 minutes max
        normalized_frequency = min(1.0, frequency)
        
        return (normalized_duration + normalized_frequency) / 2.0
    
    def _calculate_navigation_efficiency(self, patterns: List[str]) -> float:
        """Calculate navigation efficiency score."""
        if not patterns:
            return 0.0
        
        unique_screens = len(set(patterns))
        total_steps = len(patterns)
        
        # Higher efficiency = fewer steps to visit unique screens
        return unique_screens / max(1, total_steps)
    
    def _calculate_user_journey_complexity(self, patterns: List[str]) -> float:
        """Calculate complexity of user journey."""
        if not patterns:
            return 0.0
        
        # Based on unique transitions
        transitions = set()
        for i in range(len(patterns) - 1):
            transitions.add((patterns[i], patterns[i+1]))
        
        return len(transitions) / max(1, len(patterns) - 1)
    
    def _calculate_attention_span_metric(self, screen_times: Dict[str, float]) -> float:
        """Calculate attention span metric."""
        if not screen_times:
            return 0.0
        
        times = list(screen_times.values())
        if not times:
            return 0.0
        
        # Longer average time per screen indicates better attention span
        avg_time = np.mean(times)
        return min(1.0, avg_time / 60.0)  # Normalize to 1 minute max
    
    def _calculate_multitasking_indicator(self, screen_times: Dict[str, float]) -> float:
        """Calculate multitasking behavior indicator."""
        if not screen_times or len(screen_times) < 2:
            return 0.0
        
        # More screens with shorter times indicates multitasking
        num_screens = len(screen_times)
        avg_time = np.mean(list(screen_times.values()))
        
        if avg_time == 0:
            return 1.0
        
        # Higher multitasking = more screens, shorter time per screen
        return min(1.0, num_screens / (avg_time + 1.0))
    
    def _calculate_time_consistency_score(self, time_of_day: int) -> float:
        """Calculate time usage consistency score."""
        # This would be enhanced with historical data in production
        # For now, return a placeholder based on time
        if 9 <= time_of_day <= 17:  # Business hours
            return 0.8
        elif 18 <= time_of_day <= 22:  # Evening
            return 0.6
        else:  # Night/early morning
            return 0.3
    
    def _calculate_usage_pattern_score(self, time_of_day: int, day_of_week: int) -> float:
        """Calculate usage pattern score based on time and day."""
        # Weekday vs weekend, business hours vs personal time
        if day_of_week < 5:  # Weekday
            if 9 <= time_of_day <= 17:  # Business hours
                return 0.9
            else:
                return 0.6
        else:  # Weekend
            if 9 <= time_of_day <= 22:  # Daytime/evening
                return 0.7
            else:
                return 0.4
    
    def _calculate_device_familiarity_score(self, orientation: str) -> float:
        """Calculate device familiarity score."""
        # Portrait mode typically indicates more comfortable/familiar usage
        if orientation == "portrait":
            return 0.8
        elif orientation == "landscape":
            return 0.6
        else:
            return 0.4
    
    def _calculate_app_version_stability_score(self, app_version: str) -> float:
        """Calculate app version stability score."""
        # This would be enhanced with version history in production
        # For now, return a placeholder
        return 0.7

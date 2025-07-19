"""
Enhanced Behavioral Data Processor
Converts raw mobile behavioral data into meaningful vector embeddings for FAISS storage
"""

import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timezone
from dataclasses import dataclass
from collections import defaultdict
import math

logger = logging.getLogger(__name__)

@dataclass
class ProcessedBehavioralFeatures:
    """Processed behavioral features ready for vector embedding"""
    # Touch patterns (15 features)
    touch_pressure_stats: List[float]  # mean, std, min, max, percentiles
    touch_duration_stats: List[float]  # mean, std, min, max, percentiles  
    inter_touch_gap_stats: List[float] # mean, std, min, max, percentiles
    
    # Motion patterns (18 features)
    accelerometer_stats: List[float]  # x,y,z mean, std, magnitude stats
    gyroscope_stats: List[float]     # x,y,z mean, std, magnitude stats
    
    # Scroll behavior (12 features)
    scroll_velocity_stats: List[float] # mean, std, min, max stats
    scroll_pixel_stats: List[float]    # mean, std, distance stats
    scroll_pattern_stats: List[float]  # frequency, smoothness stats
    
    # Device interaction (10 features)
    orientation_changes: float
    brightness_adjustments: float
    session_duration: float
    event_frequency: float
    interaction_rhythm: List[float]
    
    # Contextual features (15 features)
    temporal_patterns: List[float]  # time of day, session timing
    device_stability: List[float]   # device position consistency
    user_confidence: List[float]    # authentication success patterns
    
    # Behavioral consistency (20 features) 
    consistency_metrics: List[float] # pattern stability across session
    anomaly_indicators: List[float]  # deviation from expected patterns
    
    def to_vector(self) -> np.ndarray:
        """Convert processed features to 90-dimensional vector"""
        vector = []
        
        # Touch patterns (15)
        vector.extend(self.touch_pressure_stats)
        vector.extend(self.touch_duration_stats) 
        vector.extend(self.inter_touch_gap_stats)
        
        # Motion patterns (18)
        vector.extend(self.accelerometer_stats)
        vector.extend(self.gyroscope_stats)
        
        # Scroll behavior (12)
        vector.extend(self.scroll_velocity_stats)
        vector.extend(self.scroll_pixel_stats)
        vector.extend(self.scroll_pattern_stats)
        
        # Device interaction (10)
        vector.append(self.orientation_changes)
        vector.append(self.brightness_adjustments)
        vector.append(self.session_duration)
        vector.append(self.event_frequency)
        vector.extend(self.interaction_rhythm)
        
        # Contextual features (15)
        vector.extend(self.temporal_patterns)
        vector.extend(self.device_stability)
        vector.extend(self.user_confidence)
        
        # Behavioral consistency (20)
        vector.extend(self.consistency_metrics)
        vector.extend(self.anomaly_indicators)
        
        # Ensure exactly 90 dimensions
        while len(vector) < 90:
            vector.append(0.0)
        
        return np.array(vector[:90], dtype=np.float32)

class EnhancedBehavioralProcessor:
    """Enhanced processor for converting mobile behavioral data to embeddings"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def _parse_timestamp(self, timestamp):
        """Parse timestamp in multiple formats"""
        try:
            if isinstance(timestamp, (int, float)):
                # Numeric timestamp (milliseconds)
                return datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)
            elif isinstance(timestamp, str):
                # ISO string timestamp
                return datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            else:
                # Fallback to current time
                return datetime.now(tz=timezone.utc)
        except Exception:
            # Fallback to current time on any error
            return datetime.now(tz=timezone.utc)
        
    def process_behavioral_logs(self, logs: List[Dict[str, Any]]) -> ProcessedBehavioralFeatures:
        """
        Process raw behavioral logs from mobile into structured features
        
        Args:
            logs: List of behavioral event logs from mobile app
            
        Returns:
            ProcessedBehavioralFeatures with 90-dimensional vector capability
        """
        try:
            # Organize events by type
            events_by_type = defaultdict(list)
            for log in logs:
                event_type = log.get('event_type', '')
                events_by_type[event_type].append(log)
            
            # Extract and process each feature category
            touch_features = self._extract_touch_features(events_by_type)
            motion_features = self._extract_motion_features(events_by_type)
            scroll_features = self._extract_scroll_features(events_by_type)
            device_features = self._extract_device_features(events_by_type, logs)
            contextual_features = self._extract_contextual_features(events_by_type, logs)
            consistency_features = self._extract_consistency_features(events_by_type, logs)
            
            return ProcessedBehavioralFeatures(
                touch_pressure_stats=touch_features['pressure_stats'],
                touch_duration_stats=touch_features['duration_stats'],
                inter_touch_gap_stats=touch_features['gap_stats'],
                accelerometer_stats=motion_features['accel_stats'],
                gyroscope_stats=motion_features['gyro_stats'],
                scroll_velocity_stats=scroll_features['velocity_stats'],
                scroll_pixel_stats=scroll_features['pixel_stats'],
                scroll_pattern_stats=scroll_features['pattern_stats'],
                orientation_changes=device_features['orientation_changes'],
                brightness_adjustments=device_features['brightness_adjustments'],
                session_duration=device_features['session_duration'],
                event_frequency=device_features['event_frequency'],
                interaction_rhythm=device_features['interaction_rhythm'],
                temporal_patterns=contextual_features['temporal_patterns'],
                device_stability=contextual_features['device_stability'],
                user_confidence=contextual_features['user_confidence'],
                consistency_metrics=consistency_features['consistency_metrics'],
                anomaly_indicators=consistency_features['anomaly_indicators']
            )
            
        except Exception as e:
            self.logger.error(f"Error processing behavioral logs: {e}")
            return self._get_default_features()
    
    def _extract_touch_features(self, events_by_type: Dict[str, List]) -> Dict[str, List[float]]:
        """Extract touch-related behavioral features"""
        # Handle both old format and new touch_sequence format
        touch_events = events_by_type.get('touch', []) + events_by_type.get('touch_down', []) + events_by_type.get('touch_up', [])
        
        # NEW: Handle touch_sequence format
        touch_sequence_events = events_by_type.get('touch_sequence', [])
        
        # Extract touch data from touch_sequence events
        pressures = []
        durations = []
        touch_positions = []
        
        # Process old format
        for event in touch_events:
            if 'data' in event and 'pressure' in event['data']:
                pressures.append(event['data']['pressure'])
                
        # Process NEW format (touch_sequence)
        for event in touch_sequence_events:
            if 'data' in event and 'touch_events' in event['data']:
                for touch_event in event['data']['touch_events']:
                    # Extract pressure
                    if 'pressure' in touch_event:
                        pressures.append(float(touch_event['pressure']))
                    
                    # Extract duration
                    if 'duration' in touch_event:
                        durations.append(float(touch_event['duration']))
                    
                    # Extract position
                    if 'x' in touch_event and 'y' in touch_event:
                        touch_positions.append((float(touch_event['x']), float(touch_event['y'])))

        pressure_stats = self._calculate_stats(pressures, 5)
        duration_stats = self._calculate_stats(durations, 5)
        
        # Calculate inter-touch gaps from positions (movement distances)
        gaps = []
        if len(touch_positions) > 1:
            for i in range(1, len(touch_positions)):
                dx = touch_positions[i][0] - touch_positions[i-1][0]
                dy = touch_positions[i][1] - touch_positions[i-1][1]
                gap = math.sqrt(dx*dx + dy*dy)
                gaps.append(gap)

        gap_stats = self._calculate_stats(gaps, 5)
        
        return {
            'pressure_stats': pressure_stats,
            'duration_stats': duration_stats,
            'gap_stats': gap_stats
        }
    
    def _extract_motion_features(self, events_by_type: Dict[str, List]) -> Dict[str, List[float]]:
        """Extract motion sensor features (accelerometer & gyroscope)"""
        # Handle both old format and new touch_sequence format
        accel_events = events_by_type.get('accelerometer', []) + events_by_type.get('accel_data', [])
        gyro_events = events_by_type.get('gyroscope', []) + events_by_type.get('gyro_data', [])
        
        # NEW: Handle touch_sequence format which contains accelerometer/gyroscope data
        touch_sequence_events = events_by_type.get('touch_sequence', [])
        
        # Accelerometer statistics
        accel_x = []
        accel_y = []
        accel_z = []
        
        # Process old format
        for event in accel_events:
            if 'data' in event:
                accel_x.append(float(event['data'].get('x', 0)))
                accel_y.append(float(event['data'].get('y', 0)))
                accel_z.append(float(event['data'].get('z', 0)))
        
        # Process NEW format (motion data from touch_sequence)
        for event in touch_sequence_events:
            if 'data' in event:
                # Extract accelerometer data
                if 'accelerometer' in event['data']:
                    accel_data = event['data']['accelerometer']
                    if isinstance(accel_data, list):
                        # Handle array of accelerometer readings
                        for accel_item in accel_data:
                            if isinstance(accel_item, dict):
                                accel_x.append(float(accel_item.get('x', 0)))
                                accel_y.append(float(accel_item.get('y', 0)))
                                accel_z.append(float(accel_item.get('z', 0)))
                    elif isinstance(accel_data, dict):
                        # Handle single accelerometer reading
                        accel_x.append(float(accel_data.get('x', 0)))
                        accel_y.append(float(accel_data.get('y', 0)))
                        accel_z.append(float(accel_data.get('z', 0)))
        
        # Calculate magnitude for each reading
        accel_magnitude = []
        if accel_x and accel_y and accel_z:
            accel_magnitude = [math.sqrt(x**2 + y**2 + z**2) for x, y, z in zip(accel_x, accel_y, accel_z)]
        
        accel_stats = []
        accel_stats.extend(self._calculate_stats(accel_x, 3))
        accel_stats.extend(self._calculate_stats(accel_y, 3)) 
        accel_stats.extend(self._calculate_stats(accel_z, 3))
        accel_stats.extend(self._calculate_stats(accel_magnitude, 3))
        
        # Gyroscope statistics
        gyro_x = []
        gyro_y = []
        gyro_z = []
        
        # Process old format
        for event in gyro_events:
            if 'data' in event:
                gyro_x.append(float(event['data'].get('x', 0)))
                gyro_y.append(float(event['data'].get('y', 0)))
                gyro_z.append(float(event['data'].get('z', 0)))
        
        # Process NEW format (motion data from touch_sequence)
        for event in touch_sequence_events:
            if 'data' in event:
                # Extract gyroscope data
                if 'gyroscope' in event['data']:
                    gyro_data = event['data']['gyroscope']
                    if isinstance(gyro_data, list):
                        # Handle array of gyroscope readings
                        for gyro_item in gyro_data:
                            if isinstance(gyro_item, dict):
                                gyro_x.append(float(gyro_item.get('x', 0)))
                                gyro_y.append(float(gyro_item.get('y', 0)))
                                gyro_z.append(float(gyro_item.get('z', 0)))
                    elif isinstance(gyro_data, dict):
                        # Handle single gyroscope reading
                        gyro_x.append(float(gyro_data.get('x', 0)))
                        gyro_y.append(float(gyro_data.get('y', 0)))
                        gyro_z.append(float(gyro_data.get('z', 0)))
        
        # Calculate gyroscope magnitude
        gyro_magnitude = []
        if gyro_x and gyro_y and gyro_z:
            gyro_magnitude = [math.sqrt(x**2 + y**2 + z**2) for x, y, z in zip(gyro_x, gyro_y, gyro_z)]
        
        gyro_stats = []
        gyro_stats.extend(self._calculate_stats(gyro_x, 1))
        gyro_stats.extend(self._calculate_stats(gyro_y, 1))
        gyro_stats.extend(self._calculate_stats(gyro_z, 1))
        gyro_stats.extend(self._calculate_stats(gyro_magnitude, 3))
        
        return {
            'accel_stats': accel_stats,
            'gyro_stats': gyro_stats
        }
    
    def _extract_scroll_features(self, events_by_type: Dict[str, List]) -> Dict[str, List[float]]:
        """Extract scrolling behavior features"""
        # Look for scroll data in touch_sequence events
        touch_sequence_events = events_by_type.get('touch_sequence', [])
        scroll_events = events_by_type.get('scroll', [])  # Keep for backward compatibility
        
        # Velocity statistics from scroll events
        velocities = []
        pixels = []
        
        # Extract from touch_sequence events first
        for event in touch_sequence_events:
            if 'data' in event and 'scroll' in event['data']:
                scroll_data = event['data']['scroll']
                # Extract velocity and delta information
                if isinstance(scroll_data, dict):
                    velocities.append(scroll_data.get('velocity', 0))
                    # Calculate pixels from delta_y if available
                    delta_y = scroll_data.get('delta_y', 0)
                    pixels.append(abs(delta_y))
                elif isinstance(scroll_data, list):
                    # Handle array of scroll events
                    for scroll_item in scroll_data:
                        if isinstance(scroll_item, dict):
                            velocities.append(scroll_item.get('velocity', 0))
                            delta_y = scroll_item.get('delta_y', 0)
                            pixels.append(abs(delta_y))
        
        # Also process standalone scroll events for backward compatibility
        for event in scroll_events:
            if 'data' in event:
                velocities.append(event['data'].get('velocity', 0))
                # Calculate pixels from delta_y if available
                delta_y = event['data'].get('delta_y', 0)
                pixels.append(abs(delta_y))
        
        velocity_stats = self._calculate_stats(velocities, 4)
        
        # Pixel movement statistics
        pixel_stats = self._calculate_stats(pixels, 4)
        
        # Pattern analysis
        pattern_stats = []
        total_scroll_events = len(touch_sequence_events) + len(scroll_events)
        if total_scroll_events > 0:
            # Scroll frequency
            pattern_stats.append(total_scroll_events)
            
            # Scroll smoothness (velocity variance)
            vel_variance = np.var(velocities) if velocities else 0
            pattern_stats.append(vel_variance)
            
            # Direction consistency
            positive_scrolls = sum(1 for v in velocities if v > 0)
            direction_consistency = positive_scrolls / len(velocities) if velocities else 0.5
            pattern_stats.append(direction_consistency)
            
            # Scroll acceleration patterns
            if len(velocities) > 1:
                accelerations = [velocities[i+1] - velocities[i] for i in range(len(velocities)-1)]
                pattern_stats.append(np.mean(accelerations) if accelerations else 0)
            else:
                pattern_stats.append(0)
        else:
            pattern_stats = [0, 0, 0.5, 0]
        
        return {
            'velocity_stats': velocity_stats,
            'pixel_stats': pixel_stats,
            'pattern_stats': pattern_stats
        }
    
    def _extract_device_features(self, events_by_type: Dict[str, List], all_logs: List) -> Dict[str, Any]:
        """Extract device interaction features"""
        orientation_events = events_by_type.get('orientation_change', [])
        brightness_events = events_by_type.get('brightness_change', [])
        
        # Count changes
        orientation_changes = float(len(orientation_events))
        brightness_adjustments = float(len(brightness_events))
        
        # Session duration
        if all_logs:
            timestamps = [self._parse_timestamp(log['timestamp']) for log in all_logs]
            session_duration = (max(timestamps) - min(timestamps)).total_seconds()
        else:
            session_duration = 0.0
        
        # Event frequency
        event_frequency = len(all_logs) / max(session_duration, 1.0)
        
        # Interaction rhythm (time between consecutive events)
        if len(all_logs) > 1:
            timestamps = [self._parse_timestamp(log['timestamp']) for log in all_logs]
            intervals = [(timestamps[i+1] - timestamps[i]).total_seconds() for i in range(len(timestamps)-1)]
            rhythm_stats = self._calculate_stats(intervals, 6)
        else:
            rhythm_stats = [0.0] * 6
        
        return {
            'orientation_changes': orientation_changes,
            'brightness_adjustments': brightness_adjustments,
            'session_duration': session_duration,
            'event_frequency': event_frequency,
            'interaction_rhythm': rhythm_stats
        }
    
    def _extract_contextual_features(self, events_by_type: Dict[str, List], all_logs: List) -> Dict[str, List[float]]:
        """Extract contextual behavioral features"""
        
        # Temporal patterns
        temporal_patterns = []
        if all_logs:
            # Time of day analysis
            timestamps = [self._parse_timestamp(log['timestamp']) for log in all_logs]
            hours = [t.hour for t in timestamps]
            temporal_patterns.extend(self._calculate_stats(hours, 3))
            
            # Session timing consistency
            start_time = min(timestamps)
            hour_of_day = start_time.hour
            temporal_patterns.append(hour_of_day / 24.0)  # Normalized hour
            temporal_patterns.append(start_time.weekday() / 7.0)  # Normalized day of week
        else:
            temporal_patterns = [0.0] * 5
        
        # Device stability (accelerometer consistency)
        # Look in touch_sequence events first
        touch_sequence_events = events_by_type.get('touch_sequence', [])
        accel_events = events_by_type.get('accel_data', [])  # Backward compatibility
        
        accel_magnitudes = []
        
        # Extract from touch_sequence events
        for event in touch_sequence_events:
            if 'data' in event and 'accelerometer' in event['data']:
                accel_data = event['data']['accelerometer']
                if isinstance(accel_data, list):
                    for accel_item in accel_data:
                        if isinstance(accel_item, dict):
                            x, y, z = accel_item.get('x', 0), accel_item.get('y', 0), accel_item.get('z', 0)
                            magnitude = math.sqrt(x**2 + y**2 + z**2)
                            accel_magnitudes.append(magnitude)
                elif isinstance(accel_data, dict):
                    x, y, z = accel_data.get('x', 0), accel_data.get('y', 0), accel_data.get('z', 0)
                    magnitude = math.sqrt(x**2 + y**2 + z**2)
                    accel_magnitudes.append(magnitude)
        
        # Also process standalone accelerometer events
        for event in accel_events:
            x, y, z = event['data'].get('x', 0), event['data'].get('y', 0), event['data'].get('z', 0)
            magnitude = math.sqrt(x**2 + y**2 + z**2)
            accel_magnitudes.append(magnitude)
        
        if accel_magnitudes:
            stability_stats = self._calculate_stats(accel_magnitudes, 5)
        else:
            stability_stats = [0.0] * 5
        
        # User confidence (authentication patterns)
        mpin_events = events_by_type.get('mpin_verified', [])
        confidence_stats = []
        if mpin_events:
            success_rate = sum(1 for event in mpin_events if event['data'].get('success', False)) / len(mpin_events)
            confidence_stats.append(success_rate)
            confidence_stats.append(float(len(mpin_events)))
            confidence_stats.extend([0.0] * 3)  # Reserved for future confidence metrics
        else:
            confidence_stats = [1.0, 0.0, 0.0, 0.0, 0.0]  # Default high confidence
        
        return {
            'temporal_patterns': temporal_patterns,
            'device_stability': stability_stats,
            'user_confidence': confidence_stats
        }
    
    def _extract_consistency_features(self, events_by_type: Dict[str, List], all_logs: List) -> Dict[str, List[float]]:
        """Extract behavioral consistency and anomaly indicators"""
        
        # Pattern consistency across session
        consistency_metrics = []
        
        # Touch consistency
        # Look in touch_sequence events
        touch_sequence_events = events_by_type.get('touch_sequence', [])
        touch_events = events_by_type.get('touch_down', []) + events_by_type.get('touch_up', [])  # Backward compatibility
        
        coordinates = []
        
        # Extract coordinates from touch_sequence events
        for event in touch_sequence_events:
            if 'data' in event and 'touch_events' in event['data']:
                touch_data = event['data']['touch_events']
                if isinstance(touch_data, list):
                    for touch_item in touch_data:
                        if isinstance(touch_item, dict) and 'coordinates' in touch_item:
                            coords = touch_item['coordinates']
                            if len(coords) >= 2:
                                coordinates.append([coords[0], coords[1]])
        
        # Also process standalone touch events
        for event in touch_events:
            if 'coordinates' in event['data']:
                coords = event['data']['coordinates']
                if len(coords) >= 2:
                    coordinates.append([coords[0], coords[1]])
        
        if len(coordinates) > 5:
            coord_array = np.array(coordinates)
            x_variance = np.var(coord_array[:, 0])
            y_variance = np.var(coord_array[:, 1])
            consistency_metrics.extend([x_variance, y_variance])
        else:
            consistency_metrics.extend([0.0, 0.0])
        
        # Motion consistency
        # Use the same accelerometer data we extracted above
        x_values = []
        y_values = []
        z_values = []
        
        # Extract from touch_sequence events
        for event in touch_sequence_events:
            if 'data' in event and 'accelerometer' in event['data']:
                accel_data = event['data']['accelerometer']
                if isinstance(accel_data, list):
                    for accel_item in accel_data:
                        if isinstance(accel_item, dict):
                            x_values.append(accel_item.get('x', 0))
                            y_values.append(accel_item.get('y', 0))
                            z_values.append(accel_item.get('z', 0))
                elif isinstance(accel_data, dict):
                    x_values.append(accel_data.get('x', 0))
                    y_values.append(accel_data.get('y', 0))
                    z_values.append(accel_data.get('z', 0))
        
        # Also get from standalone accelerometer events
        standalone_accel_events = events_by_type.get('accel_data', [])
        for event in standalone_accel_events:
            x_values.append(event['data'].get('x', 0))
            y_values.append(event['data'].get('y', 0))
            z_values.append(event['data'].get('z', 0))
        
        if x_values:
            consistency_metrics.append(np.var(x_values))
            consistency_metrics.append(np.var(y_values))
            consistency_metrics.append(np.var(z_values))
        else:
            consistency_metrics.extend([0.0, 0.0, 0.0])
        
        # Timing consistency
        if len(all_logs) > 1:
            timestamps = [self._parse_timestamp(log['timestamp']) for log in all_logs]
            intervals = [(timestamps[i+1] - timestamps[i]).total_seconds() for i in range(len(timestamps)-1)]
            interval_variance = np.var(intervals)
            consistency_metrics.append(interval_variance)
        else:
            consistency_metrics.append(0.0)
        
        # Pad to 10 consistency metrics
        while len(consistency_metrics) < 10:
            consistency_metrics.append(0.0)
        
        # Anomaly indicators
        anomaly_indicators = []
        
        # Extreme values detection
        pressures = []
        
        # Extract pressures from touch_sequence events
        for event in touch_sequence_events:
            if 'data' in event and 'touch_events' in event['data']:
                touch_data = event['data']['touch_events']
                if isinstance(touch_data, list):
                    for touch_item in touch_data:
                        if isinstance(touch_item, dict) and 'pressure' in touch_item:
                            pressures.append(touch_item['pressure'])
        
        # Also get from standalone touch events
        standalone_touch_events = events_by_type.get('touch_down', [])
        for event in standalone_touch_events:
            if 'pressure' in event['data']:
                pressures.append(event['data']['pressure'])
        
        if pressures:
            pressure_mean = np.mean(pressures)
            pressure_std = np.std(pressures)
            extreme_pressure_count = sum(1 for p in pressures if abs(p - pressure_mean) > 2 * pressure_std)
            anomaly_indicators.append(extreme_pressure_count / len(pressures))
        else:
            anomaly_indicators.append(0.0)
        
        # Rapid event frequency
        total_events = len(all_logs)
        session_duration = self._extract_device_features(events_by_type, all_logs)['session_duration']
        if session_duration > 0:
            event_rate = total_events / session_duration
            # Flag high event rates as potentially anomalous
            anomaly_indicators.append(min(event_rate / 10.0, 1.0))  # Normalize to 0-1
        else:
            anomaly_indicators.append(0.0)
        
        # Pattern breaks - use existing scroll velocity data
        scroll_velocities = []
        
        # Extract scroll velocities from touch_sequence events
        for event in touch_sequence_events:
            if 'data' in event and 'scroll' in event['data']:
                scroll_data = event['data']['scroll']
                if isinstance(scroll_data, dict):
                    scroll_velocities.append(scroll_data.get('velocity', 0))
                elif isinstance(scroll_data, list):
                    for scroll_item in scroll_data:
                        if isinstance(scroll_item, dict):
                            scroll_velocities.append(scroll_item.get('velocity', 0))
        
        # Also get from standalone scroll events
        standalone_scroll_events = events_by_type.get('scroll', [])
        for event in standalone_scroll_events:
            if 'velocity' in event['data']:
                scroll_velocities.append(event['data']['velocity'])
        
        if len(scroll_velocities) > 2:
            # Detect sudden velocity changes
            velocity_changes = [abs(scroll_velocities[i+1] - scroll_velocities[i]) for i in range(len(scroll_velocities)-1)]
            mean_change = np.mean(velocity_changes)
            large_changes = sum(1 for change in velocity_changes if change > 3 * mean_change)
            anomaly_indicators.append(large_changes / len(velocity_changes))
        else:
            anomaly_indicators.append(0.0)
        
        # Pad to 10 anomaly indicators
        while len(anomaly_indicators) < 10:
            anomaly_indicators.append(0.0)
        
        return {
            'consistency_metrics': consistency_metrics,
            'anomaly_indicators': anomaly_indicators
        }
    
    def _calculate_stats(self, values: List[float], num_stats: int) -> List[float]:
        """Calculate statistical features for a list of values"""
        if not values:
            return [0.0] * num_stats
        
        stats = []
        arr = np.array(values)
        
        # Basic statistics
        if num_stats >= 1:
            stats.append(float(np.mean(arr)))
        if num_stats >= 2:
            stats.append(float(np.std(arr)))
        if num_stats >= 3:
            stats.append(float(np.min(arr)))
        if num_stats >= 4:
            stats.append(float(np.max(arr)))
        if num_stats >= 5:
            stats.append(float(np.median(arr)))
        if num_stats >= 6:
            stats.append(float(np.percentile(arr, 25)))
        
        # Pad with zeros if needed
        while len(stats) < num_stats:
            stats.append(0.0)
        
        return stats[:num_stats]
    
    def _get_default_features(self) -> ProcessedBehavioralFeatures:
        """Return default features when processing fails"""
        return ProcessedBehavioralFeatures(
            touch_pressure_stats=[1.0, 0.0, 1.0, 1.0, 1.0],
            touch_duration_stats=[100.0, 50.0, 50.0, 200.0, 100.0],
            inter_touch_gap_stats=[500.0, 200.0, 300.0, 1000.0, 500.0],
            accelerometer_stats=[0.0] * 12,
            gyroscope_stats=[0.0] * 6,
            scroll_velocity_stats=[0.0] * 4,
            scroll_pixel_stats=[0.0] * 4,
            scroll_pattern_stats=[0.0] * 4,
            orientation_changes=0.0,
            brightness_adjustments=0.0,
            session_duration=0.0,
            event_frequency=0.0,
            interaction_rhythm=[0.0] * 6,
            temporal_patterns=[12.0, 0.0, 0.0, 0.5, 0.5],
            device_stability=[9.8, 0.0, 9.8, 9.8, 9.8],
            user_confidence=[1.0, 0.0, 0.0, 0.0, 0.0],
            consistency_metrics=[0.0] * 10,
            anomaly_indicators=[0.0] * 10
        )

    def process_mobile_behavioral_data(self, behavioral_data: Dict[str, Any]) -> np.ndarray:
        """
        Main method to process mobile behavioral data in the exact format you provided
        
        Args:
            behavioral_data: Dict containing user_id, session_id, and logs array
            
        Returns:
            np.ndarray: 90-dimensional vector embedding
        """
        try:
            self.logger.info(f"Processing mobile behavioral data for user {behavioral_data.get('user_id')} session {behavioral_data.get('session_id')}")
            
            # Extract logs from the mobile data format
            logs = behavioral_data.get('logs', [])
            
            if not logs:
                self.logger.warning("No logs found in behavioral data, returning default features")
                return self._get_default_features().to_vector()
            
            # Process the logs using existing method
            features = self.process_behavioral_logs(logs)
            
            # Convert to vector
            vector = features.to_vector()
            
            self.logger.info(f"Successfully processed {len(logs)} behavioral events into 90-dimensional vector")
            
            # Verify vector quality (not all zeros)
            vector_sum = np.sum(np.abs(vector))
            if vector_sum == 0:
                self.logger.warning("Generated vector is all zeros, using default features")
                return self._get_default_features().to_vector()
            
            return vector
            
        except Exception as e:
            self.logger.error(f"Error processing mobile behavioral data: {e}")
            return self._get_default_features().to_vector()

    def create_vector_embedding(self, behavioral_data: Dict[str, Any]) -> List[float]:
        """
        Backward compatibility method - converts mobile behavioral data to vector list
        
        Args:
            behavioral_data: Mobile behavioral data
            
        Returns:
            List[float]: 90-dimensional vector as list
        """
        vector = self.process_mobile_behavioral_data(behavioral_data)
        return vector.tolist()

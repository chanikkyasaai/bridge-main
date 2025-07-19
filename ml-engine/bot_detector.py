"""
Bot Detector for Banking Application
Detects automated/robotic behavior patterns that indicate malicious bots
"""

import numpy as np
from typing import List, Dict, Any, Tuple
import logging
from datetime import datetime
import math

logger = logging.getLogger(__name__)

class BotDetector:
    """Detects bot/automated behavior in banking applications"""
    
    def __init__(self):
        self.bot_threshold = 0.8  # High threshold for banking security
        self.suspicious_patterns = {
            'perfect_timing': 0.9,
            'impossible_speed': 0.95,
            'perfect_coordinates': 0.85,
            'unnatural_pressure': 0.8,
            'robotic_acceleration': 0.9,
            'missing_human_variance': 0.85
        }
    
    async def analyze_events(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze events for bot behavior
        
        Returns:
            Dict with is_bot, confidence, reason, and details
        """
        try:
            if not events:
                return {"is_bot": False, "confidence": 0.0, "reason": "no_events"}
            
            # Run all bot detection tests
            bot_scores = {}
            
            # 1. Perfect timing patterns
            bot_scores['timing'] = await self._detect_perfect_timing(events)
            
            # 2. Impossible human speeds
            bot_scores['speed'] = await self._detect_impossible_speeds(events)
            
            # 3. Perfect coordinate patterns
            bot_scores['coordinates'] = await self._detect_perfect_coordinates(events)
            
            # 4. Unnatural pressure patterns
            bot_scores['pressure'] = await self._detect_unnatural_pressure(events)
            
            # 5. Robotic accelerometer patterns
            bot_scores['acceleration'] = await self._detect_robotic_acceleration(events)
            
            # 6. Missing human variance
            bot_scores['variance'] = await self._detect_missing_variance(events)
            
            # 7. Sequence patterns
            bot_scores['sequence'] = await self._detect_sequence_patterns(events)
            
            # Calculate overall bot confidence
            max_score = max(bot_scores.values())
            avg_score = np.mean(list(bot_scores.values()))
            
            # Use weighted combination (max score has more weight for security)
            confidence = 0.7 * max_score + 0.3 * avg_score
            
            is_bot = confidence >= self.bot_threshold
            
            # Find primary reason
            primary_reason = max(bot_scores.items(), key=lambda x: x[1])
            
            result = {
                "is_bot": is_bot,
                "confidence": confidence,
                "reason": primary_reason[0],
                "details": {
                    "scores": bot_scores,
                    "threshold": self.bot_threshold,
                    "max_score": max_score,
                    "avg_score": avg_score
                }
            }
            
            if is_bot:
                logger.warning(f"Bot detected: {primary_reason[0]} (confidence: {confidence:.2f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Bot detection failed: {e}")
            return {"is_bot": False, "confidence": 0.0, "reason": "detection_error"}
    
    async def _detect_perfect_timing(self, events: List[Dict]) -> float:
        """Detect suspiciously perfect timing patterns"""
        try:
            timestamps = []
            for event in events:
                if event.get('timestamp'):
                    try:
                        ts = datetime.fromisoformat(event['timestamp'].replace('Z', '+00:00'))
                        timestamps.append(ts)
                    except:
                        continue
            
            if len(timestamps) < 3:
                return 0.0
            
            # Calculate intervals between events
            intervals = []
            for i in range(1, len(timestamps)):
                interval = (timestamps[i] - timestamps[i-1]).total_seconds() * 1000  # milliseconds
                intervals.append(interval)
            
            if not intervals:
                return 0.0
            
            # Check for perfect timing (very low variance)
            variance = np.var(intervals)
            mean_interval = np.mean(intervals)
            
            if mean_interval == 0:
                return 0.9  # All events at same time = suspicious
            
            coefficient_of_variation = math.sqrt(variance) / mean_interval
            
            # Human timing usually has CV > 0.1, bots often have CV < 0.05
            if coefficient_of_variation < 0.02:
                return 0.95
            elif coefficient_of_variation < 0.05:
                return 0.8
            elif coefficient_of_variation < 0.1:
                return 0.4
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Perfect timing detection failed: {e}")
            return 0.0
    
    async def _detect_impossible_speeds(self, events: List[Dict]) -> float:
        """Detect impossible human interaction speeds"""
        try:
            touch_events = [e for e in events if 'touch' in e.get('event_type', '')]
            
            if len(touch_events) < 2:
                return 0.0
            
            max_suspicious_score = 0.0
            
            for i in range(1, len(touch_events)):
                prev_event = touch_events[i-1]
                curr_event = touch_events[i]
                
                # Get coordinates
                prev_data = prev_event.get('data', {})
                curr_data = curr_event.get('data', {})
                
                prev_coords = prev_data.get('coordinates', [0, 0])
                curr_coords = curr_data.get('coordinates', [0, 0])
                
                # Calculate distance
                dx = curr_coords[0] - prev_coords[0]
                dy = curr_coords[1] - prev_coords[1]
                distance = math.sqrt(dx*dx + dy*dy)
                
                # Calculate time difference
                try:
                    prev_time = datetime.fromisoformat(prev_event['timestamp'].replace('Z', '+00:00'))
                    curr_time = datetime.fromisoformat(curr_event['timestamp'].replace('Z', '+00:00'))
                    time_diff = (curr_time - prev_time).total_seconds()
                except:
                    continue
                
                if time_diff <= 0:
                    max_suspicious_score = max(max_suspicious_score, 0.9)
                    continue
                
                # Calculate speed (pixels per second)
                speed = distance / time_diff
                
                # Impossible speeds for human touch
                if speed > 5000:  # > 5000 pixels/sec is very suspicious
                    max_suspicious_score = max(max_suspicious_score, 0.95)
                elif speed > 3000:
                    max_suspicious_score = max(max_suspicious_score, 0.8)
                elif speed > 2000:
                    max_suspicious_score = max(max_suspicious_score, 0.4)
            
            return max_suspicious_score
            
        except Exception as e:
            logger.error(f"Speed detection failed: {e}")
            return 0.0
    
    async def _detect_perfect_coordinates(self, events: List[Dict]) -> float:
        """Detect suspiciously perfect coordinate patterns"""
        try:
            touch_events = [e for e in events if 'touch' in e.get('event_type', '')]
            
            if len(touch_events) < 3:
                return 0.0
            
            coordinates = []
            for event in touch_events:
                coords = event.get('data', {}).get('coordinates', [])
                if coords:
                    coordinates.append((coords[0], coords[1]))
            
            if len(coordinates) < 3:
                return 0.0
            
            # Check for perfect patterns
            x_coords = [c[0] for c in coordinates]
            y_coords = [c[1] for c in coordinates]
            
            # Check for repeated exact coordinates
            unique_coords = set(coordinates)
            if len(unique_coords) == 1:  # All touches at exact same spot
                return 0.95
            
            # Check for perfect grid patterns
            x_diffs = [abs(x_coords[i] - x_coords[i-1]) for i in range(1, len(x_coords))]
            y_diffs = [abs(y_coords[i] - y_coords[i-1]) for i in range(1, len(y_coords))]
            
            # Check if differences are suspiciously regular
            if x_diffs and all(abs(d - x_diffs[0]) < 1 for d in x_diffs):
                return 0.8
            if y_diffs and all(abs(d - y_diffs[0]) < 1 for d in y_diffs):
                return 0.8
            
            # Check for perfect integer coordinates (bots often use exact pixels)
            perfect_integers = sum(1 for x, y in coordinates if x == int(x) and y == int(y))
            if perfect_integers / len(coordinates) > 0.9:
                return 0.7
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Perfect coordinates detection failed: {e}")
            return 0.0
    
    async def _detect_unnatural_pressure(self, events: List[Dict]) -> float:
        """Detect unnatural pressure patterns"""
        try:
            touch_events = [e for e in events if 'touch' in e.get('event_type', '')]
            
            pressures = []
            for event in touch_events:
                pressure = event.get('data', {}).get('pressure')
                if pressure is not None:
                    pressures.append(pressure)
            
            if len(pressures) < 3:
                return 0.0
            
            # Check for constant pressure (very suspicious)
            unique_pressures = set(pressures)
            if len(unique_pressures) == 1:
                return 0.9
            
            # Check for unnatural pressure values
            perfect_ones = sum(1 for p in pressures if p == 1.0)
            if perfect_ones / len(pressures) > 0.95:  # Always perfect pressure
                return 0.8
            
            # Check for impossible pressure variance (too low)
            pressure_variance = np.var(pressures)
            if pressure_variance < 0.001:  # Very low variance
                return 0.7
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Pressure detection failed: {e}")
            return 0.0
    
    async def _detect_robotic_acceleration(self, events: List[Dict]) -> float:
        """Detect robotic accelerometer patterns"""
        try:
            accel_events = [e for e in events if e.get('event_type') == 'accel_data']
            
            if len(accel_events) < 5:
                return 0.0
            
            # Extract acceleration values
            accelerations = []
            for event in accel_events:
                data = event.get('data', {})
                x, y, z = data.get('x', 0), data.get('y', 0), data.get('z', 0)
                magnitude = math.sqrt(x*x + y*y + z*z)
                accelerations.append(magnitude)
            
            if not accelerations:
                return 0.0
            
            # Check for constant acceleration (impossible for real device)
            unique_values = set(round(a, 6) for a in accelerations)  # Round to avoid floating point issues
            if len(unique_values) == 1:
                return 0.95
            
            # Check for suspiciously low variance
            acceleration_variance = np.var(accelerations)
            if acceleration_variance < 0.001:
                return 0.9
            
            # Check for perfect mathematical patterns
            if len(accelerations) > 3:
                diffs = [accelerations[i] - accelerations[i-1] for i in range(1, len(accelerations))]
                if diffs and all(abs(d - diffs[0]) < 0.001 for d in diffs):
                    return 0.85  # Perfect linear pattern
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Acceleration detection failed: {e}")
            return 0.0
    
    async def _detect_missing_variance(self, events: List[Dict]) -> float:
        """Detect missing natural human variance across all interactions"""
        try:
            if len(events) < 10:
                return 0.0
            
            variance_scores = []
            
            # Check touch duration variance
            touch_durations = []
            for event in events:
                if 'touch' in event.get('event_type', ''):
                    duration = event.get('data', {}).get('touch_duration_ms')
                    if duration:
                        touch_durations.append(duration)
            
            if len(touch_durations) > 3:
                duration_cv = np.std(touch_durations) / max(np.mean(touch_durations), 1)
                if duration_cv < 0.05:
                    variance_scores.append(0.8)
                elif duration_cv < 0.15:
                    variance_scores.append(0.4)
            
            # Check inter-event timing variance
            timestamps = []
            for event in events:
                try:
                    ts = datetime.fromisoformat(event['timestamp'].replace('Z', '+00:00'))
                    timestamps.append(ts)
                except:
                    continue
            
            if len(timestamps) > 3:
                intervals = [(timestamps[i] - timestamps[i-1]).total_seconds() 
                           for i in range(1, len(timestamps))]
                if intervals:
                    interval_cv = np.std(intervals) / max(np.mean(intervals), 0.001)
                    if interval_cv < 0.02:
                        variance_scores.append(0.9)
                    elif interval_cv < 0.1:
                        variance_scores.append(0.5)
            
            return max(variance_scores) if variance_scores else 0.0
            
        except Exception as e:
            logger.error(f"Variance detection failed: {e}")
            return 0.0
    
    async def _detect_sequence_patterns(self, events: List[Dict]) -> float:
        """Detect robotic sequence patterns"""
        try:
            if len(events) < 5:
                return 0.0
            
            event_types = [e.get('event_type', '') for e in events]
            
            # Check for repetitive sequences
            sequence_length = min(3, len(event_types) // 3)
            if sequence_length < 2:
                return 0.0
            
            # Look for repeating patterns
            for length in range(2, sequence_length + 1):
                pattern = event_types[:length]
                repetitions = 0
                
                for i in range(0, len(event_types) - length + 1, length):
                    if event_types[i:i+length] == pattern:
                        repetitions += 1
                
                # If pattern repeats too many times, it's suspicious
                if repetitions >= len(event_types) // length * 0.8:  # 80% of possible repetitions
                    return 0.8
            
            # Check for unnatural event type distribution
            type_counts = {}
            for event_type in event_types:
                type_counts[event_type] = type_counts.get(event_type, 0) + 1
            
            # If one event type dominates too much, it might be robotic
            max_count = max(type_counts.values())
            if max_count / len(event_types) > 0.9:
                return 0.6
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Sequence detection failed: {e}")
            return 0.0
    
    def get_bot_detection_summary(self) -> Dict[str, Any]:
        """Get bot detection configuration summary"""
        return {
            "bot_threshold": self.bot_threshold,
            "detection_methods": [
                "perfect_timing",
                "impossible_speeds", 
                "perfect_coordinates",
                "unnatural_pressure",
                "robotic_acceleration",
                "missing_variance",
                "sequence_patterns"
            ],
            "thresholds": self.suspicious_patterns
        }

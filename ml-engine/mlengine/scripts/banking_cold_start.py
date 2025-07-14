"""
Banking-Grade Cold Start Handler and Early Threat Detection

This module handles the cold start problem for new users and provides
early threat detection capabilities suitable for banking applications.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import asyncio
import statistics

logger = logging.getLogger(__name__)

class UserProfileStage(Enum):
    """User profile development stages"""
    COLD_START = "cold_start"              # 0 sessions
    OBSERVATION = "observation"            # 1-3 sessions
    LEARNING = "learning"                  # 4-5 sessions
    ESTABLISHED = "established"            # 6+ sessions
    SUSPICIOUS = "suspicious"              # Flagged behavior

class ThreatLevel(Enum):
    """Early threat detection levels"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ColdStartMetrics:
    """Metrics for cold start analysis"""
    session_count: int = 0
    total_events: int = 0
    avg_session_duration: float = 0.0
    behavioral_consistency: float = 0.0
    threat_indicators: List[str] = None
    profile_stage: UserProfileStage = UserProfileStage.COLD_START
    
    def __post_init__(self):
        if self.threat_indicators is None:
            self.threat_indicators = []

@dataclass
class EarlyThreatIndicators:
    """Early threat detection indicators"""
    bot_score: float = 0.0
    automation_score: float = 0.0
    speed_anomaly_score: float = 0.0
    pattern_anomaly_score: float = 0.0
    device_compromise_score: float = 0.0
    overall_threat_level: ThreatLevel = ThreatLevel.NONE
    specific_threats: List[str] = None
    
    def __post_init__(self):
        if self.specific_threats is None:
            self.specific_threats = []

class BankingColdStartHandler:
    """
    Banking-grade cold start handler with observation mode and early threat detection
    """
    
    def __init__(self):
        self.user_profiles = {}  # In production, this should be persistent storage
        self.session_histories = {}
        
        # Banking-specific thresholds
        self.observation_sessions = 3
        self.learning_sessions = 5
        self.established_threshold = 6
        
        # Early threat detection thresholds
        self.bot_detection_threshold = 0.7
        self.speed_anomaly_threshold = 0.8
        self.automation_threshold = 0.75
        
        logger.info("Banking Cold Start Handler initialized")
    
    async def get_user_profile_stage(self, user_id: str) -> UserProfileStage:
        """Get current profile development stage for user"""
        if user_id not in self.user_profiles:
            return UserProfileStage.COLD_START
        
        metrics = self.user_profiles[user_id]
        
        if metrics.session_count == 0:
            return UserProfileStage.COLD_START
        elif metrics.session_count <= self.observation_sessions:
            return UserProfileStage.OBSERVATION
        elif metrics.session_count <= self.learning_sessions:
            return UserProfileStage.LEARNING
        else:
            return UserProfileStage.ESTABLISHED
    
    async def should_use_observation_mode(self, user_id: str) -> bool:
        """Determine if user should be in observation-only mode"""
        stage = await self.get_user_profile_stage(user_id)
        return stage in [UserProfileStage.COLD_START, UserProfileStage.OBSERVATION]
    
    async def detect_early_threats(self, session_id: str, user_id: str, 
                                 behavioral_events: List[Dict[str, Any]]) -> EarlyThreatIndicators:
        """
        Early threat detection for banking security
        Works even without established behavioral profiles
        """
        try:
            if not behavioral_events:
                return EarlyThreatIndicators()
            
            # Validate and filter events to ensure they have proper structure
            valid_events = []
            for event in behavioral_events:
                try:
                    # Ensure event is a dictionary with required fields
                    if not isinstance(event, dict):
                        continue
                    
                    # Check for basic required structure
                    if 'features' not in event:
                        continue
                    
                    if not isinstance(event.get('features'), dict):
                        continue
                        
                    valid_events.append(event)
                except Exception as e:
                    logger.warning(f"Skipping malformed event: {e}")
                    continue
            
            if not valid_events:
                logger.warning(f"No valid events found for threat detection - user {user_id}, session {session_id}")
                return EarlyThreatIndicators()
            
            # PATCH: Aggressively flag extreme values as attack/bot
            for event in valid_events:
                features = event.get('features', {})
                if features.get('pressure', 0) >= 1.0 or features.get('velocity', 0) >= 5.0:
                    logger.warning(f"[PATCH] Aggressive threat flag: Extreme pressure/velocity detected for user {user_id}, session {session_id}. Features: {features}")
                    return EarlyThreatIndicators(
                        bot_score=1.0,
                        automation_score=1.0,
                        speed_anomaly_score=1.0,
                        pattern_anomaly_score=1.0,
                        device_compromise_score=0.0,
                        overall_threat_level=ThreatLevel.CRITICAL,
                        specific_threats=["EXTREME_PRESSURE_OR_VELOCITY"]
                    )
            
            # Analyze behavioral patterns for threats
            bot_score = await self._detect_bot_behavior(valid_events)
            automation_score = await self._detect_automation(valid_events)
            speed_score = await self._detect_speed_anomalies(valid_events)
            pattern_score = await self._detect_pattern_anomalies(valid_events)
            device_score = await self._detect_device_compromise(valid_events)
            
            # Calculate overall threat level
            max_score = max(bot_score, automation_score, speed_score, pattern_score, device_score)
            threat_level = self._calculate_threat_level(max_score)
            
            # Identify specific threats
            specific_threats = []
            if bot_score > self.bot_detection_threshold:
                specific_threats.append(f"BOT_BEHAVIOR (score: {bot_score:.2f})")
            if automation_score > self.automation_threshold:
                specific_threats.append(f"AUTOMATION_DETECTED (score: {automation_score:.2f})")
            if speed_score > self.speed_anomaly_threshold:
                specific_threats.append(f"SPEED_ANOMALY (score: {speed_score:.2f})")
            if pattern_score > 0.7:
                specific_threats.append(f"PATTERN_ANOMALY (score: {pattern_score:.2f})")
            if device_score > 0.6:
                specific_threats.append(f"DEVICE_COMPROMISE (score: {device_score:.2f})")
            
            return EarlyThreatIndicators(
                bot_score=bot_score,
                automation_score=automation_score,
                speed_anomaly_score=speed_score,
                pattern_anomaly_score=pattern_score,
                device_compromise_score=device_score,
                overall_threat_level=threat_level,
                specific_threats=specific_threats
            )
            
        except Exception as e:
            logger.error(f"Error in threat detection for user {user_id}, session {session_id}: {e}")
            # Return safe default in case of errors
            return EarlyThreatIndicators(
                bot_score=0.0,
                automation_score=0.0,
                speed_anomaly_score=0.0,
                pattern_anomaly_score=0.0,
                device_compromise_score=0.0,
                overall_threat_level=ThreatLevel.NONE,
                specific_threats=["ERROR_IN_THREAT_DETECTION"]
            )
    
    async def _detect_bot_behavior(self, events: List[Dict[str, Any]]) -> float:
        """Detect bot-like behavior patterns"""
        try:
            if len(events) < 5:
                return 0.0
            
            bot_indicators = 0
            total_checks = 0
            
            # Check for perfect timing patterns (bots often have regular intervals)
            touch_events = [e for e in events if e.get('event_type') == 'touch']
            if len(touch_events) >= 3:
                intervals = []
                for i in range(1, len(touch_events)):
                    if 'timestamp' in touch_events[i] and 'timestamp' in touch_events[i-1]:
                        try:
                            t1 = datetime.fromisoformat(touch_events[i]['timestamp'].replace('Z', ''))
                            t2 = datetime.fromisoformat(touch_events[i-1]['timestamp'].replace('Z', ''))
                            intervals.append((t1 - t2).total_seconds())
                        except Exception:
                            continue
                
                if intervals:
                    total_checks += 1
                    # Perfect regularity is suspicious
                    if len(set([round(i, 1) for i in intervals])) <= 2:
                        bot_indicators += 1
            
            # Check for inhuman precision in touch coordinates
            touch_coords = [(e.get('features', {}).get('x_position', 0), 
                            e.get('features', {}).get('y_position', 0)) 
                           for e in touch_events if e.get('features')]
            
            if len(touch_coords) >= 5:
                total_checks += 1
                # Check for pixel-perfect repetition
                unique_coords = set(touch_coords)
                if len(unique_coords) / len(touch_coords) < 0.3:  # Too many repeated exact coordinates
                    bot_indicators += 1
            
            # Check for inhuman speed/pressure consistency
            pressures = [e.get('features', {}).get('pressure', 0) for e in touch_events 
                        if e.get('features', {}).get('pressure')]
            if len(pressures) >= 5:
                total_checks += 1
                pressure_variance = np.var(pressures) if pressures else 0
                if pressure_variance < 0.01:  # Too consistent pressure
                    bot_indicators += 1
            
            return bot_indicators / max(total_checks, 1)
        
        except Exception as e:
            logger.error(f"Error in bot behavior detection: {e}")
            return 0.0
    
    async def _detect_automation(self, events: List[Dict[str, Any]]) -> float:
        """Detect automated/scripted behavior"""
        if len(events) < 10:
            return 0.0
        
        automation_score = 0.0
        
        # Check for rapid-fire events (faster than human possible)
        rapid_events = 0
        for i in range(1, len(events)):
            try:
                t1 = datetime.fromisoformat(events[i]['timestamp'].replace('Z', ''))
                t2 = datetime.fromisoformat(events[i-1]['timestamp'].replace('Z', ''))
                interval = (t1 - t2).total_seconds()
                if interval < 0.05:  # Faster than 50ms is suspicious
                    rapid_events += 1
            except:
                continue
        
        if rapid_events > len(events) * 0.3:  # More than 30% rapid events
            automation_score += 0.5
        
        # Check for perfect geometric patterns (automation often follows exact paths)
        touch_events = [e for e in events if e.get('event_type') == 'touch']
        if len(touch_events) >= 10:
            coords = [(e.get('features', {}).get('x_position', 0), 
                      e.get('features', {}).get('y_position', 0)) 
                     for e in touch_events]
            
            # Check for straight lines (perfect automation)
            if self._is_perfect_line(coords):
                automation_score += 0.3
            
            # Check for perfect circles/curves
            if self._is_perfect_curve(coords):
                automation_score += 0.3
        
        return min(automation_score, 1.0)
    
    async def _detect_speed_anomalies(self, events: List[Dict[str, Any]]) -> float:
        """Detect inhuman speed patterns"""
        try:
            touch_events = [e for e in events if e.get('event_type') == 'touch']
            if len(touch_events) < 5:
                return 0.0
            
            speeds = []
            for i in range(1, len(touch_events)):
                try:
                    # Calculate movement speed
                    e1, e2 = touch_events[i-1], touch_events[i]
                    x1, y1 = e1.get('features', {}).get('x_position', 0), e1.get('features', {}).get('y_position', 0)
                    x2, y2 = e2.get('features', {}).get('x_position', 0), e2.get('features', {}).get('y_position', 0)
                    
                    t1 = datetime.fromisoformat(e1['timestamp'].replace('Z', ''))
                    t2 = datetime.fromisoformat(e2['timestamp'].replace('Z', ''))
                    
                    distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    time_diff = (t2-t1).total_seconds()
                    
                    if time_diff > 0:
                        speed = distance / time_diff  # pixels per second
                        speeds.append(speed)
                except Exception:
                    continue
            
            if not speeds:
                return 0.0
            
            # Detect abnormally high speeds (possible bot/hacking)
            max_human_speed = 2000  # pixels per second (reasonable for mobile)
            super_fast_moves = sum(1 for s in speeds if s > max_human_speed)
            
            # Detect impossible acceleration
            impossible_moves = sum(1 for s in speeds if s > 5000)
            
            speed_anomaly_ratio = (super_fast_moves + impossible_moves * 2) / len(speeds)
            return min(speed_anomaly_ratio, 1.0)
        
        except Exception as e:
            logger.error(f"Error in speed anomaly detection: {e}")
            return 0.0
    
    async def _detect_pattern_anomalies(self, events: List[Dict[str, Any]]) -> float:
        """Detect anomalous behavioral patterns"""
        if len(events) < 15:
            return 0.0
        
        anomaly_score = 0.0
        
        # Check for missing typical human variations
        touch_events = [e for e in events if e.get('event_type') == 'touch']
        if len(touch_events) >= 10:
            # Human touch typically varies in pressure, duration, etc.
            pressures = [e.get('features', {}).get('pressure', 0.5) for e in touch_events]
            durations = [e.get('features', {}).get('duration', 0.1) for e in touch_events]
            
            # Too little variation suggests automation
            if pressures and np.std(pressures) < 0.05:
                anomaly_score += 0.3
            
            if durations and np.std(durations) < 0.02:
                anomaly_score += 0.3
        
        # Check for missing typical human errors/corrections
        # Humans typically have some back-and-forth movements, corrections
        movement_directions = []
        touch_coords = [(e.get('features', {}).get('x_position', 0), 
                        e.get('features', {}).get('y_position', 0)) 
                       for e in touch_events]
        
        for i in range(1, len(touch_coords)):
            dx = touch_coords[i][0] - touch_coords[i-1][0]
            dy = touch_coords[i][1] - touch_coords[i-1][1]
            if abs(dx) > 10 or abs(dy) > 10:  # Significant movement
                movement_directions.append((dx, dy))
        
        # Perfect unidirectional movement is suspicious
        if len(movement_directions) >= 5:
            direction_changes = 0
            for i in range(1, len(movement_directions)):
                prev_dir = movement_directions[i-1]
                curr_dir = movement_directions[i]
                # Check for direction change
                if (prev_dir[0] * curr_dir[0] < 0) or (prev_dir[1] * curr_dir[1] < 0):
                    direction_changes += 1
            
            if direction_changes / len(movement_directions) < 0.1:  # Too few direction changes
                anomaly_score += 0.4
        
        return min(anomaly_score, 1.0)
    
    async def _detect_device_compromise(self, events: List[Dict[str, Any]]) -> float:
        """Detect potential device compromise indicators"""
        compromise_score = 0.0
        
        # Check for multiple simultaneous touch points (possible injection)
        simultaneous_touches = 0
        touch_events = [e for e in events if e.get('event_type') == 'touch']
        
        # Group events by timestamp (within 10ms window)
        timestamp_groups = {}
        for event in touch_events:
            try:
                ts = datetime.fromisoformat(event['timestamp'].replace('Z', ''))
                ts_key = int(ts.timestamp() * 100)  # 10ms precision
                if ts_key not in timestamp_groups:
                    timestamp_groups[ts_key] = []
                timestamp_groups[ts_key].append(event)
            except:
                continue
        
        # Count simultaneous touches
        for group in timestamp_groups.values():
            if len(group) > 2:  # More than 2 simultaneous touches is suspicious
                simultaneous_touches += 1
        
        if simultaneous_touches > 0:
            compromise_score += min(simultaneous_touches * 0.2, 0.5)
        
        # Check for impossible multi-touch patterns
        # Humans can't typically do complex multi-touch at banking speeds
        complex_multitouch = 0
        for group in timestamp_groups.values():
            if len(group) >= 3:
                # Check if touches are too far apart for human hands
                coords = [(e.get('features', {}).get('x_position', 0), 
                          e.get('features', {}).get('y_position', 0)) for e in group]
                max_distance = 0
                for i in range(len(coords)):
                    for j in range(i+1, len(coords)):
                        dist = np.sqrt((coords[i][0] - coords[j][0])**2 + (coords[i][1] - coords[j][1])**2)
                        max_distance = max(max_distance, dist)
                
                if max_distance > 800:  # Too far for human hand span
                    complex_multitouch += 1
        
        if complex_multitouch > 0:
            compromise_score += min(complex_multitouch * 0.3, 0.4)
        
        return min(compromise_score, 1.0)
    
    def _is_perfect_line(self, coords: List[Tuple[float, float]]) -> bool:
        """Check if coordinates form a suspiciously perfect line"""
        if len(coords) < 5:
            return False
        
        # Calculate if points are nearly collinear
        deviations = []
        for i in range(2, len(coords)):
            # Calculate distance from point to line formed by first and last points
            x1, y1 = coords[0]
            x2, y2 = coords[-1]
            x0, y0 = coords[i]
            
            # Distance from point to line formula
            if (x2 - x1) == 0 and (y2 - y1) == 0:
                continue
                
            distance = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1) / np.sqrt((y2-y1)**2 + (x2-x1)**2)
            deviations.append(distance)
        
        if deviations:
            avg_deviation = sum(deviations) / len(deviations)
            return avg_deviation < 5.0  # Very small deviation suggests automation
        
        return False
    
    def _is_perfect_curve(self, coords: List[Tuple[float, float]]) -> bool:
        """Check if coordinates form a suspiciously perfect curve"""
        if len(coords) < 8:
            return False
        
        # Simple check for perfect circular motion
        # Calculate center and check if all points are equidistant
        center_x = sum(c[0] for c in coords) / len(coords)
        center_y = sum(c[1] for c in coords) / len(coords)
        
        distances = [np.sqrt((c[0] - center_x)**2 + (c[1] - center_y)**2) for c in coords]
        if distances:
            variance = np.var(distances)
            return variance < 100  # Very low variance suggests perfect circle
        
        return False
    
    def _calculate_threat_level(self, max_score: float) -> ThreatLevel:
        """Calculate overall threat level from scores"""
        if max_score >= 0.9:
            return ThreatLevel.CRITICAL
        elif max_score >= 0.75:
            return ThreatLevel.HIGH
        elif max_score >= 0.5:
            return ThreatLevel.MEDIUM
        elif max_score >= 0.25:
            return ThreatLevel.LOW
        else:
            return ThreatLevel.NONE
    
    async def process_session_learning(self, user_id: str, session_id: str, 
                                     session_events: List[Dict[str, Any]], 
                                     session_duration: float) -> ColdStartMetrics:
        """Process session for progressive learning"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = ColdStartMetrics()
        
        metrics = self.user_profiles[user_id]
        metrics.session_count += 1
        metrics.total_events += len(session_events)
        
        # Update average session duration
        if metrics.session_count == 1:
            metrics.avg_session_duration = session_duration
        else:
            metrics.avg_session_duration = (
                (metrics.avg_session_duration * (metrics.session_count - 1) + session_duration) 
                / metrics.session_count
            )
        
        # Store session history for pattern learning
        if user_id not in self.session_histories:
            self.session_histories[user_id] = []
        
        self.session_histories[user_id].append({
            'session_id': session_id,
            'events': session_events,
            'duration': session_duration,
            'timestamp': datetime.utcnow()
        })
        
        # Keep only last 10 sessions
        if len(self.session_histories[user_id]) > 10:
            self.session_histories[user_id] = self.session_histories[user_id][-10:]
        
        # Calculate behavioral consistency (for users with multiple sessions)
        if metrics.session_count >= 2:
            metrics.behavioral_consistency = await self._calculate_behavioral_consistency(user_id)
        
        # Update profile stage
        metrics.profile_stage = await self.get_user_profile_stage(user_id)
        
        return metrics
    
    async def _calculate_behavioral_consistency(self, user_id: str) -> float:
        """Calculate behavioral consistency across sessions"""
        if user_id not in self.session_histories or len(self.session_histories[user_id]) < 2:
            return 0.0
        
        sessions = self.session_histories[user_id]
        consistency_scores = []
        
        # Compare session characteristics
        for i in range(1, len(sessions)):
            current = sessions[i]
            previous = sessions[i-1]
            
            # Duration consistency
            duration_diff = abs(current['duration'] - previous['duration'])
            duration_consistency = max(0, 1 - duration_diff / max(current['duration'], previous['duration']))
            
            # Event count consistency
            current_events = len(current['events'])
            previous_events = len(previous['events'])
            if max(current_events, previous_events) > 0:
                event_consistency = min(current_events, previous_events) / max(current_events, previous_events)
            else:
                event_consistency = 1.0
            
            # Combined consistency
            session_consistency = (duration_consistency + event_consistency) / 2
            consistency_scores.append(session_consistency)
        
        return sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.0
    
    async def get_banking_security_decision(self, user_id: str, session_id: str, 
                                          behavioral_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get banking-specific security decision with cold start handling
        """
        # Get user profile stage
        profile_stage = await self.get_user_profile_stage(user_id)
        observation_mode = await self.should_use_observation_mode(user_id)
        
        # Run early threat detection regardless of profile stage
        threat_indicators = await self.detect_early_threats(session_id, user_id, behavioral_events)
        
        # PATCH: Aggressively block attack/bot scenarios
        if (
            threat_indicators.bot_score > 0.6 or
            threat_indicators.automation_score > 0.6 or
            threat_indicators.speed_anomaly_score > 0.6 or
            threat_indicators.pattern_anomaly_score > 0.6
        ):
            logger.warning(f"[PATCH] Aggressive block: Detected attack/bot indicators for user {user_id}, session {session_id}. Scores: bot={threat_indicators.bot_score}, automation={threat_indicators.automation_score}, speed={threat_indicators.speed_anomaly_score}, pattern={threat_indicators.pattern_anomaly_score}")
            action = "block"
            reason = f"Aggressive block: Attack/bot indicators detected (bot={threat_indicators.bot_score:.2f}, automation={threat_indicators.automation_score:.2f}, speed={threat_indicators.speed_anomaly_score:.2f}, pattern={threat_indicators.pattern_anomaly_score:.2f})"
            return {
                "action": action,
                "reason": reason,
                "profile_stage": profile_stage.value,
                "observation_mode": observation_mode,
                "threat_level": "critical",
                "threat_indicators": {
                    "bot_score": threat_indicators.bot_score,
                    "automation_score": threat_indicators.automation_score,
                    "speed_anomaly_score": threat_indicators.speed_anomaly_score,
                    "pattern_anomaly_score": threat_indicators.pattern_anomaly_score,
                    "device_compromise_score": threat_indicators.device_compromise_score,
                    "specific_threats": threat_indicators.specific_threats
                },
                "session_count": self.user_profiles.get(user_id, ColdStartMetrics()).session_count,
                "requires_profile_building": observation_mode
            }
        
        # Determine action based on profile stage and threats
        if threat_indicators.overall_threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            # Block immediately regardless of profile stage
            action = "block"
            reason = f"Critical threat detected: {', '.join(threat_indicators.specific_threats)}"
        elif threat_indicators.overall_threat_level == ThreatLevel.MEDIUM:
            if observation_mode:
                action = "flag_and_continue"
                reason = "Medium threat during observation - flagged for review"
            else:
                action = "step_up_auth"
                reason = "Medium threat detected - additional verification required"
        elif observation_mode:
            # Observation mode - collect data but don't block
            action = "observe"
            reason = f"Observation mode - Session {self.user_profiles.get(user_id, ColdStartMetrics()).session_count + 1}/{self.observation_sessions + 1}"
        else:
            # Normal operation for established users
            action = "continue"
            reason = "Normal behavior - continue session"
        
        return {
            "action": action,
            "reason": reason,
            "profile_stage": profile_stage.value,
            "observation_mode": observation_mode,
            "threat_level": threat_indicators.overall_threat_level.value,
            "threat_indicators": {
                "bot_score": threat_indicators.bot_score,
                "automation_score": threat_indicators.automation_score,
                "speed_anomaly_score": threat_indicators.speed_anomaly_score,
                "pattern_anomaly_score": threat_indicators.pattern_anomaly_score,
                "device_compromise_score": threat_indicators.device_compromise_score,
                "specific_threats": threat_indicators.specific_threats
            },
            "session_count": self.user_profiles.get(user_id, ColdStartMetrics()).session_count,
            "requires_profile_building": observation_mode
        }

# Global instance
banking_cold_start_handler = BankingColdStartHandler()

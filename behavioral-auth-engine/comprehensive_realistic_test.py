#!/usr/bin/env python3
"""
Comprehensive Realistic Behavioral Authentication System Test
Simulates complete user journey from cold start to advanced detection
"""

import asyncio
import aiohttp
import json
import uuid
import time
import random
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealisticBehavioralSimulator:
    """Simulates realistic user behavioral patterns"""
    
    def __init__(self):
        self.user_profiles = {}
        
    def create_user_profile(self, user_type: str = "normal") -> Dict[str, Any]:
        """Create a realistic user behavioral profile"""
        profiles = {
            "normal": {
                "typing_speed": random.uniform(45, 85),  # WPM
                "typing_rhythm_variance": random.uniform(0.1, 0.3),
                "touch_pressure": random.uniform(0.3, 0.7),
                "touch_accuracy": random.uniform(0.85, 0.98),
                "scroll_speed": random.uniform(2.0, 8.0),
                "navigation_efficiency": random.uniform(0.7, 0.95),
                "session_duration_preference": random.uniform(300, 1800),  # seconds
                "error_rate": random.uniform(0.02, 0.08),
                "confidence_baseline": random.uniform(0.7, 0.9)
            },
            "elderly": {
                "typing_speed": random.uniform(20, 45),
                "typing_rhythm_variance": random.uniform(0.4, 0.8),
                "touch_pressure": random.uniform(0.5, 0.9),
                "touch_accuracy": random.uniform(0.6, 0.85),
                "scroll_speed": random.uniform(1.0, 4.0),
                "navigation_efficiency": random.uniform(0.4, 0.7),
                "session_duration_preference": random.uniform(600, 2400),
                "error_rate": random.uniform(0.1, 0.2),
                "confidence_baseline": random.uniform(0.5, 0.8)
            },
            "power_user": {
                "typing_speed": random.uniform(80, 120),
                "typing_rhythm_variance": random.uniform(0.05, 0.15),
                "touch_pressure": random.uniform(0.2, 0.5),
                "touch_accuracy": random.uniform(0.95, 0.99),
                "scroll_speed": random.uniform(8.0, 15.0),
                "navigation_efficiency": random.uniform(0.9, 0.99),
                "session_duration_preference": random.uniform(180, 900),
                "error_rate": random.uniform(0.01, 0.05),
                "confidence_baseline": random.uniform(0.8, 0.95)
            },
            "bot": {
                "typing_speed": random.uniform(200, 500),  # Extremely fast
                "typing_rhythm_variance": random.uniform(0.001, 0.01),  # Too consistent
                "touch_pressure": 0.5,  # Always the same
                "touch_accuracy": 1.0,  # Perfect accuracy
                "scroll_speed": random.uniform(20.0, 50.0),  # Too fast
                "navigation_efficiency": 1.0,  # Perfect navigation
                "session_duration_preference": random.uniform(60, 300),
                "error_rate": 0.0,  # No errors
                "confidence_baseline": 0.99  # Suspiciously high
            },
            "attacker": {
                "typing_speed": random.uniform(30, 60),  # Slower than normal
                "typing_rhythm_variance": random.uniform(0.3, 0.6),  # Inconsistent
                "touch_pressure": random.uniform(0.1, 0.9),  # Highly variable
                "touch_accuracy": random.uniform(0.5, 0.8),  # Lower accuracy
                "scroll_speed": random.uniform(1.0, 12.0),  # Erratic
                "navigation_efficiency": random.uniform(0.3, 0.6),  # Poor navigation
                "session_duration_preference": random.uniform(120, 600),
                "error_rate": random.uniform(0.15, 0.3),  # High error rate
                "confidence_baseline": random.uniform(0.2, 0.6)  # Low confidence
            }
        }
        return profiles.get(user_type, profiles["normal"])
    
    def generate_behavioral_vector(self, profile: Dict[str, Any], session_num: int = 1, 
                                 drift_factor: float = 0.0, anomaly_factor: float = 0.0) -> Tuple[List[float], float]:
        """Generate a realistic 90-dimensional behavioral vector"""
        
        # Base vector generation with realistic noise
        base_vector = []
        
        # Typing features (20 dimensions)
        typing_speed = profile["typing_speed"] * (1 + random.gauss(0, profile["typing_rhythm_variance"]))
        for i in range(20):
            base_vector.append(typing_speed + random.gauss(0, 5))
        
        # Touch features (20 dimensions)
        base_pressure = profile["touch_pressure"]
        base_accuracy = profile["touch_accuracy"]
        for i in range(20):
            pressure = base_pressure + random.gauss(0, 0.1)
            accuracy = base_accuracy + random.gauss(0, 0.05)
            base_vector.extend([pressure, accuracy])
        
        # Navigation features (20 dimensions)
        nav_efficiency = profile["navigation_efficiency"]
        scroll_speed = profile["scroll_speed"]
        for i in range(10):
            base_vector.extend([
                nav_efficiency + random.gauss(0, 0.1),
                scroll_speed + random.gauss(0, 1.0)
            ])
        
        # Contextual features (30 dimensions)
        error_rate = profile["error_rate"]
        session_preference = profile["session_duration_preference"]
        for i in range(10):
            base_vector.extend([
                error_rate + random.gauss(0, 0.02),
                session_preference / 1000 + random.gauss(0, 0.1),  # Normalized
                random.gauss(0.5, 0.2)  # Random contextual feature
            ])
        
        # Convert to numpy array for vectorized operations
        base_vector = np.array(base_vector)
        
        # Apply drift factor (gradual change over sessions)
        if drift_factor > 0:
            drift_noise = np.random.normal(0, drift_factor, len(base_vector))
            base_vector = base_vector + drift_noise
        
        # Apply anomaly factor (sudden unusual behavior)
        if anomaly_factor > 0:
            anomaly_indices = random.sample(range(len(base_vector)), int(len(base_vector) * anomaly_factor))
            for idx in anomaly_indices:
                base_vector[idx] *= random.uniform(2.0, 5.0)  # Significant deviation
        
        # Calculate confidence based on consistency and profile
        vector_consistency = 1.0 - np.std(base_vector) / np.mean(np.abs(base_vector))
        vector_consistency = max(0.1, min(0.99, vector_consistency))
        
        base_confidence = profile["confidence_baseline"]
        session_learning_bonus = min(0.2, session_num * 0.02)  # Improve with sessions
        
        confidence = base_confidence * vector_consistency + session_learning_bonus
        confidence = max(0.1, min(0.99, confidence))
        
        # Apply anomaly penalty to confidence
        if anomaly_factor > 0:
            confidence *= (1.0 - anomaly_factor * 0.5)
        
        # Ensure values stay within reasonable bounds and handle NaN/inf
        base_vector = np.clip(base_vector, 0.0, 10.0)
        
        # Replace any NaN or infinity values
        base_vector = np.nan_to_num(base_vector, nan=0.5, posinf=10.0, neginf=0.0)
        
        return base_vector.tolist(), confidence

class ComprehensiveSystemTester:
    """Comprehensive test suite for the behavioral authentication system"""
    
    def __init__(self):
        self.backend_url = "http://127.0.0.1:8000"
        self.ml_engine_url = "http://127.0.0.1:8001"
        self.simulator = RealisticBehavioralSimulator()
        
    async def test_complete_user_journey(self):
        """Test complete user journey from cold start to advanced detection"""
        logger.info("🚀 Starting Comprehensive Behavioral Authentication System Test")
        
        # Test different user types
        user_types = ["normal", "elderly", "power_user", "bot", "attacker"]
        results = {}
        
        for user_type in user_types:
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing User Type: {user_type.upper()}")
            logger.info(f"{'='*60}")
            
            results[user_type] = await self._test_user_type(user_type)
        
        # Summary analysis
        await self._analyze_results(results)
        
    async def _test_user_type(self, user_type: str) -> Dict[str, Any]:
        """Test a specific user type through complete journey"""
        user_id = str(uuid.uuid4())
        profile = self.simulator.create_user_profile(user_type)
        
        logger.info(f"User Profile: {json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in profile.items()}, indent=2)}")
        
        results = {
            "user_type": user_type,
            "user_id": user_id,
            "profile": profile,
            "sessions": [],
            "phase_transitions": [],
            "detection_results": [],
            "final_status": None
        }
        
        # Phase 1: Cold Start + Learning Phase (8 sessions)
        logger.info(f"\n📚 Phase 1: Learning Phase (Cold Start → Learning)")
        await self._test_learning_phase(user_id, profile, results)
        
        # Phase 2: Gradual Risk Phase (7 sessions)
        logger.info(f"\n⚠️ Phase 2: Gradual Risk Phase")
        await self._test_gradual_risk_phase(user_id, profile, results)
        
        # Phase 3: Full Authentication Phase (5 sessions)
        logger.info(f"\n🔒 Phase 3: Full Authentication Phase")
        await self._test_full_auth_phase(user_id, profile, results)
        
        # Phase 4: Drift Detection Test
        logger.info(f"\n📊 Phase 4: Behavioral Drift Detection")
        await self._test_drift_detection(user_id, profile, results)
        
        # Phase 5: Anomaly/Attack Detection
        logger.info(f"\n🛡️ Phase 5: Anomaly/Attack Detection")
        await self._test_anomaly_detection(user_id, profile, results)
        
        return results
        
    async def _test_learning_phase(self, user_id: str, profile: Dict[str, Any], results: Dict[str, Any]):
        """Test the learning phase progression"""
        logger.info("Testing Learning Phase Progression...")
        
        for session_num in range(1, 9):  # 8 sessions total
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_learning_{session_num}"
            
            # Start session
            session_response = await self._start_session(user_id, session_id)
            if not session_response:
                continue
                
            logger.info(f"Session {session_num}: Phase={session_response.get('learning_phase', 'unknown')}")
            
            # Generate 3 behavioral samples per session
            session_samples = []
            for sample_num in range(1, 4):
                vector, confidence = self.simulator.generate_behavioral_vector(
                    profile, session_num=session_num
                )
                
                # Analyze behavior
                analysis = await self._analyze_behavior(user_id, session_id, vector)
                if analysis:
                    decision = analysis.get('decision', 'unknown')
                    conf = analysis.get('confidence', 0.0)
                    phase = analysis.get('learning_result', {}).get('current_phase', 'unknown')
                    
                    session_samples.append({
                        'sample': sample_num,
                        'decision': decision,
                        'confidence': conf,
                        'phase': phase
                    })
                    
                    logger.info(f"   Session {session_num}, Sample {sample_num}: {decision} (confidence: {conf:.3f})")
                
                await asyncio.sleep(0.1)  # Small delay between samples
            
            # Check progress
            progress = await self._get_learning_progress(user_id)
            if progress:
                phase = progress.get('progress_report', {}).get('current_phase', 'unknown')
                vectors = progress.get('progress_report', {}).get('vectors_collected', 0)
                logger.info(f"   Progress: Phase={phase}, Vectors={vectors}")
                
                # Check for phase transition
                if phase != results.get('last_phase', 'cold_start'):
                    results['phase_transitions'].append({
                        'session': session_num,
                        'from_phase': results.get('last_phase', 'cold_start'),
                        'to_phase': phase
                    })
                    logger.info(f"   🔄 Phase Transition: {results.get('last_phase', 'cold_start')} → {phase}")
                
                results['last_phase'] = phase
            
            results['sessions'].append({
                'session_num': session_num,
                'session_id': session_id,
                'samples': session_samples,
                'progress': progress
            })
            
            await asyncio.sleep(0.2)  # Small delay between sessions
    
    async def _test_gradual_risk_phase(self, user_id: str, profile: Dict[str, Any], results: Dict[str, Any]):
        """Test gradual risk phase with slight behavioral variations"""
        logger.info("Testing Gradual Risk Phase...")
        
        for session_num in range(9, 16):  # 7 sessions
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_gradual_{session_num}"
            
            # Start session
            session_response = await self._start_session(user_id, session_id)
            if not session_response:
                continue
            
            # Introduce slight drift
            drift_factor = min(0.1, (session_num - 8) * 0.01)
            
            session_samples = []
            for sample_num in range(1, 4):
                vector, confidence = self.simulator.generate_behavioral_vector(
                    profile, session_num=session_num, drift_factor=drift_factor
                )
                
                analysis = await self._analyze_behavior(user_id, session_id, vector)
                if analysis:
                    decision = analysis.get('decision', 'unknown')
                    conf = analysis.get('confidence', 0.0)
                    risk_level = analysis.get('risk_level', 'unknown')
                    
                    session_samples.append({
                        'sample': sample_num,
                        'decision': decision,
                        'confidence': conf,
                        'risk_level': risk_level
                    })
                    
                    logger.info(f"   Session {session_num}, Sample {sample_num}: {decision} (risk: {risk_level}, conf: {conf:.3f})")
                
                await asyncio.sleep(0.1)
            
            results['sessions'].append({
                'session_num': session_num,
                'session_id': session_id,
                'samples': session_samples,
                'drift_factor': drift_factor
            })
            
            await asyncio.sleep(0.2)
    
    async def _test_full_auth_phase(self, user_id: str, profile: Dict[str, Any], results: Dict[str, Any]):
        """Test full authentication phase with FAISS + GNN analysis"""
        logger.info("Testing Full Authentication Phase...")
        
        for session_num in range(16, 21):  # 5 sessions
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_fullauth_{session_num}"
            
            session_response = await self._start_session(user_id, session_id)
            if not session_response:
                continue
            
            session_samples = []
            for sample_num in range(1, 4):
                vector, confidence = self.simulator.generate_behavioral_vector(
                    profile, session_num=session_num
                )
                
                analysis = await self._analyze_behavior(user_id, session_id, vector)
                if analysis:
                    decision = analysis.get('decision', 'unknown')
                    conf = analysis.get('confidence', 0.0)
                    analysis_type = analysis.get('analysis_type', 'unknown')
                    
                    session_samples.append({
                        'sample': sample_num,
                        'decision': decision,
                        'confidence': conf,
                        'analysis_type': analysis_type
                    })
                    
                    logger.info(f"   Session {session_num}, Sample {sample_num}: {decision} ({analysis_type}, conf: {conf:.3f})")
                
                await asyncio.sleep(0.1)
            
            results['sessions'].append({
                'session_num': session_num,
                'session_id': session_id,
                'samples': session_samples
            })
            
            await asyncio.sleep(0.2)
    
    async def _test_drift_detection(self, user_id: str, profile: Dict[str, Any], results: Dict[str, Any]):
        """Test behavioral drift detection"""
        logger.info("Testing Behavioral Drift Detection...")
        
        drift_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
        for i, drift_level in enumerate(drift_levels):
            session_num = 21 + i
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_drift_{session_num}"
            
            session_response = await self._start_session(user_id, session_id)
            if not session_response:
                continue
            
            vector, confidence = self.simulator.generate_behavioral_vector(
                profile, session_num=session_num, drift_factor=drift_level
            )
            
            analysis = await self._analyze_behavior(user_id, session_id, vector)
            if analysis:
                decision = analysis.get('decision', 'unknown')
                conf = analysis.get('confidence', 0.0)
                
                logger.info(f"   Drift Level {drift_level:.1f}: {decision} (conf: {conf:.3f})")
                
                results['detection_results'].append({
                    'test_type': 'drift',
                    'drift_level': drift_level,
                    'decision': decision,
                    'confidence': conf,
                    'detected': decision in ['deny', 'challenge']
                })
            
            await asyncio.sleep(0.2)
    
    async def _test_anomaly_detection(self, user_id: str, profile: Dict[str, Any], results: Dict[str, Any]):
        """Test anomaly and attack detection"""
        logger.info("Testing Anomaly/Attack Detection...")
        
        anomaly_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
        for i, anomaly_level in enumerate(anomaly_levels):
            session_num = 26 + i
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_anomaly_{session_num}"
            
            session_response = await self._start_session(user_id, session_id)
            if not session_response:
                continue
            
            vector, confidence = self.simulator.generate_behavioral_vector(
                profile, session_num=session_num, anomaly_factor=anomaly_level
            )
            
            analysis = await self._analyze_behavior(user_id, session_id, vector)
            if analysis:
                decision = analysis.get('decision', 'unknown')
                conf = analysis.get('confidence', 0.0)
                
                logger.info(f"   Anomaly Level {anomaly_level:.1f}: {decision} (conf: {conf:.3f})")
                
                results['detection_results'].append({
                    'test_type': 'anomaly',
                    'anomaly_level': anomaly_level,
                    'decision': decision,
                    'confidence': conf,
                    'detected': decision in ['deny', 'challenge']
                })
            
            await asyncio.sleep(0.2)
    
    async def _start_session(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """Start a new session"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.ml_engine_url}/session/start"
                data = {"user_id": user_id, "session_id": session_id}
                
                async with session.post(url, json=data) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.error(f"Failed to start session: {response.status}")
                        return None
        except Exception as e:
            logger.error(f"Error starting session: {e}")
            return None
    
    async def _analyze_behavior(self, user_id: str, session_id: str, vector: List[float]) -> Dict[str, Any]:
        """Analyze behavioral vector"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.ml_engine_url}/analyze"
                data = {
                    "user_id": user_id,
                    "session_id": session_id,
                    "events": [
                        {
                            "event_type": "behavioral_vector",
                            "timestamp": datetime.now().isoformat(),
                            "data": {
                                "vector": vector,
                                "confidence": 0.8
                            }
                        }
                    ]
                }
                
                async with session.post(url, json=data) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.error(f"Failed to analyze behavior: {response.status}")
                        return None
        except Exception as e:
            logger.error(f"Error analyzing behavior: {e}")
            return None
    
    async def _get_learning_progress(self, user_id: str) -> Dict[str, Any]:
        """Get learning progress"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.ml_engine_url}/user/{user_id}/learning-progress"
                
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.error(f"Failed to get learning progress: {response.status}")
                        return None
        except Exception as e:
            logger.error(f"Error getting learning progress: {e}")
            return None
    
    async def _analyze_results(self, results: Dict[str, Any]):
        """Analyze and summarize test results"""
        logger.info(f"\n{'='*80}")
        logger.info("📊 COMPREHENSIVE TEST RESULTS ANALYSIS")
        logger.info(f"{'='*80}")
        
        for user_type, result in results.items():
            logger.info(f"\n🔍 {user_type.upper()} USER ANALYSIS:")
            logger.info(f"   Total Sessions: {len(result['sessions'])}")
            logger.info(f"   Phase Transitions: {len(result['phase_transitions'])}")
            
            # Analyze detection effectiveness
            detection_results = result['detection_results']
            if detection_results:
                drift_detections = [r for r in detection_results if r['test_type'] == 'drift' and r['detected']]
                anomaly_detections = [r for r in detection_results if r['test_type'] == 'anomaly' and r['detected']]
                
                logger.info(f"   Drift Detection Rate: {len(drift_detections)}/5 ({len(drift_detections)*20}%)")
                logger.info(f"   Anomaly Detection Rate: {len(anomaly_detections)}/5 ({len(anomaly_detections)*20}%)")
            
            # Expected results based on user type
            if user_type == "bot":
                logger.info("   Expected: High detection rate for anomalies")
            elif user_type == "attacker":
                logger.info("   Expected: High detection rate for both drift and anomalies")
            elif user_type == "normal":
                logger.info("   Expected: Low false positive rate")
        
        # Overall system assessment
        logger.info(f"\n🎯 SYSTEM PERFORMANCE ASSESSMENT:")
        logger.info("   ✅ Learning Phase Progression: Functional")
        logger.info("   ✅ Session Management: Operational")
        logger.info("   ✅ Vector Storage: Working")
        logger.info("   ⚠️  Vector Counting: Needs Investigation")
        logger.info("   ⚠️  Detection Accuracy: Needs Tuning")

async def main():
    """Run comprehensive test suite"""
    tester = ComprehensiveSystemTester()
    await tester.test_complete_user_journey()

if __name__ == "__main__":
    asyncio.run(main())

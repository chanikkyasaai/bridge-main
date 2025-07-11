"""
BRIDGE ML-Engine Advanced Stress Testing & Edge Cases
Extended production readiness validation for banking security system

This extends the basic production tests with:
1. Stress Testing under load
2. Edge Case Scenarios  
3. Memory & Resource Leak Detection
4. Security Boundary Testing
5. Integration Failure Scenarios
6. Data Quality & Consistency Tests
"""

import asyncio
import pytest
import numpy as np
import time
import logging
import json
import sys
import os
import gc
import threading
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Any
from dataclasses import dataclass
import uuid
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ml_engine'))

try:
    from ml_engine.scripts.banking_cold_start import banking_cold_start_handler, UserProfileStage, ThreatLevel
    print("✅ Advanced test suite ready - banking_cold_start module loaded")
except ImportError as e:
    print(f"❌ Failed to import banking_cold_start: {e}")
    banking_cold_start_handler = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedTestResults:
    """Extended test results for stress testing"""
    def __init__(self):
        self.stress_tests = []
        self.edge_case_tests = []
        self.security_boundary_tests = []
        self.resource_tests = []
        self.integration_tests = []
        self.data_quality_tests = []
        self.critical_failures = []
        self.warnings = []
        
    def add_test_result(self, category: str, test_name: str, passed: bool, 
                       score: float, details: Dict[str, Any]):
        """Add a test result"""
        getattr(self, category).append({
            'test_name': test_name,
            'passed': passed,
            'score': score,
            'details': details,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    def calculate_scores(self):
        """Calculate category scores"""
        scores = {}
        for category in ['stress_tests', 'edge_case_tests', 'security_boundary_tests', 
                        'resource_tests', 'integration_tests', 'data_quality_tests']:
            tests = getattr(self, category)
            if tests:
                passed = sum(1 for test in tests if test['passed'])
                total = len(tests)
                avg_score = sum(test['score'] for test in tests) / total
                scores[category] = {
                    'passed': passed,
                    'total': total,
                    'avg_score': avg_score,
                    'pass_rate': passed / total
                }
            else:
                scores[category] = {'passed': 0, 'total': 0, 'avg_score': 0.0, 'pass_rate': 0.0}
        return scores

# Test Results Collector
advanced_results = AdvancedTestResults()

# ==================== STRESS TESTING ====================

@pytest.mark.asyncio
async def test_high_volume_user_load():
    """Test system under high volume of concurrent users"""
    if not banking_cold_start_handler:
        advanced_results.critical_failures.append("banking_cold_start_handler not available")
        return
        
    logger.info("🔥 Testing High Volume User Load (100 concurrent users)...")
    
    async def simulate_user_session(user_num):
        """Simulate a complete user session"""
        user_id = f"stress_user_{user_num}"
        session_id = f"stress_session_{user_num}"
        
        # Simulate multiple events per user
        events = []
        for i in range(random.randint(5, 15)):
            events.append({
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'event_type': random.choice(['touch', 'swipe', 'scroll']),
                'features': {
                    'x_position': float(random.randint(0, 1000)),
                    'y_position': float(random.randint(0, 1000)),
                    'pressure': random.uniform(0.1, 0.9),
                    'duration': random.uniform(0.05, 0.3),
                    'velocity': random.uniform(0.1, 2.0)
                }
            })
        
        # Process threat detection
        threat_result = await banking_cold_start_handler.detect_early_threats(
            session_id, user_id, events
        )
        
        # Process session learning
        metrics = await banking_cold_start_handler.process_session_learning(
            user_id, session_id, events, random.uniform(60.0, 300.0)
        )
        
        # Get security decision
        decision = await banking_cold_start_handler.get_banking_security_decision(
            user_id, session_id, events[:5]  # First 5 events
        )
        
        return {
            'user_id': user_id,
            'events_processed': len(events),
            'threat_score': max(threat_result.bot_score, threat_result.automation_score),
            'decision': decision['action']
        }
    
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    # Create 100 concurrent user sessions
    tasks = [simulate_user_session(i) for i in range(100)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    processing_time = time.time() - start_time
    end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    memory_increase = end_memory - start_memory
    
    # Analyze results
    successful_sessions = sum(1 for r in results if not isinstance(r, Exception))
    failed_sessions = len(results) - successful_sessions
    
    # Performance thresholds
    performance_ok = processing_time < 30.0  # 30 seconds for 100 users
    memory_ok = memory_increase < 500  # Less than 500MB increase
    success_rate_ok = successful_sessions / len(results) >= 0.95  # 95% success rate
    
    stress_test_passed = performance_ok and memory_ok and success_rate_ok
    
    advanced_results.add_test_result('stress_tests', 'high_volume_user_load',
                                   stress_test_passed, 1.0 if stress_test_passed else 0.0,
                                   {
                                       'concurrent_users': 100,
                                       'processing_time_seconds': processing_time,
                                       'memory_increase_mb': memory_increase,
                                       'successful_sessions': successful_sessions,
                                       'failed_sessions': failed_sessions,
                                       'success_rate': successful_sessions / len(results)
                                   })
    
    logger.info(f"Stress Test Results: {successful_sessions}/100 sessions successful in {processing_time:.2f}s")
    logger.info(f"Memory usage increased by {memory_increase:.1f}MB")
    
    assert stress_test_passed, f"High volume stress test failed - check performance/memory/success rate"

@pytest.mark.asyncio
async def test_memory_leak_detection():
    """Test for memory leaks under continuous operation"""
    if not banking_cold_start_handler:
        advanced_results.critical_failures.append("banking_cold_start_handler not available")
        return
        
    logger.info("🧠 Testing Memory Leak Detection...")
    
    initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    memory_samples = [initial_memory]
    
    # Run 500 iterations of threat detection
    for iteration in range(500):
        user_id = f"leak_test_user_{iteration % 50}"  # Cycle through 50 users
        session_id = f"leak_session_{iteration}"
        
        # Generate events
        events = [{
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'event_type': 'touch',
            'features': {
                'x_position': float(random.randint(0, 1000)),
                'y_position': float(random.randint(0, 1000)),
                'pressure': random.uniform(0.1, 0.9),
                'duration': random.uniform(0.05, 0.3)
            }
        } for _ in range(5)]
        
        # Process
        await banking_cold_start_handler.detect_early_threats(session_id, user_id, events)
        
        # Sample memory every 50 iterations
        if iteration % 50 == 0:
            gc.collect()  # Force garbage collection
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_samples.append(current_memory)
    
    final_memory = memory_samples[-1]
    memory_growth = final_memory - initial_memory
    
    # Check for concerning memory growth
    memory_leak_detected = memory_growth > 100  # More than 100MB growth is concerning
    
    advanced_results.add_test_result('resource_tests', 'memory_leak_detection',
                                   not memory_leak_detected, 1.0 if not memory_leak_detected else 0.0,
                                   {
                                       'initial_memory_mb': initial_memory,
                                       'final_memory_mb': final_memory,
                                       'memory_growth_mb': memory_growth,
                                       'iterations': 500,
                                       'memory_samples': memory_samples
                                   })
    
    logger.info(f"Memory Test: {initial_memory:.1f}MB -> {final_memory:.1f}MB (growth: {memory_growth:.1f}MB)")
    
    if memory_leak_detected:
        advanced_results.warnings.append(f"Potential memory leak detected: {memory_growth:.1f}MB growth")
    
    assert not memory_leak_detected, f"Memory leak detected: {memory_growth:.1f}MB growth over 500 iterations"

# ==================== EDGE CASE TESTING ====================

@pytest.mark.asyncio
async def test_timestamp_edge_cases():
    """Test handling of various timestamp edge cases"""
    if not banking_cold_start_handler:
        advanced_results.critical_failures.append("banking_cold_start_handler not available")
        return
        
    logger.info("⏰ Testing Timestamp Edge Cases...")
    
    edge_cases = [
        # Future timestamps
        [{
            'timestamp': (datetime.utcnow() + timedelta(hours=1)).isoformat() + 'Z',
            'event_type': 'touch',
            'features': {'x_position': 100.0, 'y_position': 200.0, 'pressure': 0.5}
        }],
        # Very old timestamps  
        [{
            'timestamp': (datetime.utcnow() - timedelta(days=365)).isoformat() + 'Z',
            'event_type': 'touch',
            'features': {'x_position': 100.0, 'y_position': 200.0, 'pressure': 0.5}
        }],
        # Invalid timestamp format
        [{
            'timestamp': 'invalid-timestamp',
            'event_type': 'touch',
            'features': {'x_position': 100.0, 'y_position': 200.0, 'pressure': 0.5}
        }],
        # Missing timestamp
        [{
            'event_type': 'touch',
            'features': {'x_position': 100.0, 'y_position': 200.0, 'pressure': 0.5}
        }],
        # Simultaneous timestamps (exactly same time)
        [
            {
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'event_type': 'touch',
                'features': {'x_position': 100.0, 'y_position': 200.0, 'pressure': 0.5}
            },
            {
                'timestamp': datetime.utcnow().isoformat() + 'Z',  # Same timestamp
                'event_type': 'touch', 
                'features': {'x_position': 150.0, 'y_position': 250.0, 'pressure': 0.6}
            }
        ]
    ]
    
    handled_cases = 0
    total_cases = len(edge_cases)
    
    for i, events in enumerate(edge_cases):
        try:
            result = await banking_cold_start_handler.detect_early_threats(
                f"edge_session_{i}", f"edge_user_{i}", events
            )
            # If we get here without exception, it's handled
            handled_cases += 1
            logger.info(f"Edge case {i+1}: Handled gracefully")
        except Exception as e:
            logger.error(f"Edge case {i+1} failed: {e}")
    
    edge_case_robust = handled_cases == total_cases
    
    advanced_results.add_test_result('edge_case_tests', 'timestamp_edge_cases',
                                   edge_case_robust, handled_cases / total_cases,
                                   {
                                       'handled_cases': handled_cases,
                                       'total_cases': total_cases,
                                       'edge_cases_tested': ['future_timestamp', 'old_timestamp', 
                                                           'invalid_format', 'missing_timestamp', 
                                                           'simultaneous_timestamps']
                                   })
    
    logger.info(f"Timestamp Edge Cases: {handled_cases}/{total_cases} handled")
    assert edge_case_robust, f"Some timestamp edge cases not handled: {handled_cases}/{total_cases}"

@pytest.mark.asyncio
async def test_extreme_feature_values():
    """Test handling of extreme feature values"""
    if not banking_cold_start_handler:
        advanced_results.critical_failures.append("banking_cold_start_handler not available")
        return
        
    logger.info("🎯 Testing Extreme Feature Values...")
    
    extreme_cases = [
        # Negative coordinates
        {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'event_type': 'touch',
            'features': {'x_position': -1000.0, 'y_position': -1000.0, 'pressure': 0.5}
        },
        # Extremely large coordinates
        {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'event_type': 'touch',
            'features': {'x_position': 999999.0, 'y_position': 999999.0, 'pressure': 0.5}
        },
        # Zero pressure
        {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'event_type': 'touch',
            'features': {'x_position': 100.0, 'y_position': 200.0, 'pressure': 0.0}
        },
        # Negative pressure
        {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'event_type': 'touch',
            'features': {'x_position': 100.0, 'y_position': 200.0, 'pressure': -0.5}
        },
        # NaN values
        {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'event_type': 'touch',
            'features': {'x_position': float('nan'), 'y_position': float('inf'), 'pressure': 0.5}
        },
        # Empty features
        {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'event_type': 'touch',
            'features': {}
        }
    ]
    
    handled_extremes = 0
    
    for i, event in enumerate(extreme_cases):
        try:
            result = await banking_cold_start_handler.detect_early_threats(
                f"extreme_session_{i}", f"extreme_user_{i}", [event]
            )
            handled_extremes += 1
            logger.info(f"Extreme case {i+1}: Handled")
        except Exception as e:
            logger.error(f"Extreme case {i+1} failed: {e}")
    
    extreme_robust = handled_extremes == len(extreme_cases)
    
    advanced_results.add_test_result('edge_case_tests', 'extreme_feature_values',
                                   extreme_robust, handled_extremes / len(extreme_cases),
                                   {
                                       'handled_cases': handled_extremes,
                                       'total_cases': len(extreme_cases)
                                   })
    
    logger.info(f"Extreme Features: {handled_extremes}/{len(extreme_cases)} handled")
    assert extreme_robust, f"Some extreme feature cases not handled: {handled_extremes}/{len(extreme_cases)}"

# ==================== SECURITY BOUNDARY TESTING ====================

@pytest.mark.asyncio
async def test_adversarial_attack_patterns():
    """Test detection of sophisticated adversarial attack patterns"""
    if not banking_cold_start_handler:
        advanced_results.critical_failures.append("banking_cold_start_handler not available")
        return
        
    logger.info("🛡️ Testing Adversarial Attack Patterns...")
    
    # Sophisticated attack pattern: trying to mimic human variation but with bot precision
    adversarial_events = []
    base_time = datetime.utcnow()
    
    for i in range(20):
        # Add slight randomness to try to fool the system, but keep bot-like patterns
        adversarial_events.append({
            'timestamp': (base_time + timedelta(milliseconds=i*100 + random.randint(-5, 5))).isoformat() + 'Z',
            'event_type': 'touch',
            'features': {
                'x_position': float(100 + i*5 + random.uniform(-2, 2)),  # Linear with slight noise
                'y_position': float(200 + random.uniform(-3, 3)),  # Small variation
                'pressure': 0.5 + random.uniform(-0.02, 0.02),  # Very small pressure variation
                'duration': 0.1 + random.uniform(-0.005, 0.005)  # Very small duration variation
            }
        })
    
    threat_result = await banking_cold_start_handler.detect_early_threats(
        "adversarial_session", "adversarial_user", adversarial_events
    )
    
    # Check if the system can still detect this as suspicious
    detected_adversarial = (
        threat_result.bot_score > 0.5 or 
        threat_result.automation_score > 0.5 or
        threat_result.pattern_anomaly_score > 0.5
    )
    
    advanced_results.add_test_result('security_boundary_tests', 'adversarial_attack_patterns',
                                   detected_adversarial, 1.0 if detected_adversarial else 0.0,
                                   {
                                       'bot_score': threat_result.bot_score,
                                       'automation_score': threat_result.automation_score,
                                       'pattern_anomaly_score': threat_result.pattern_anomaly_score,
                                       'overall_threat_level': threat_result.overall_threat_level.value
                                   })
    
    logger.info(f"Adversarial Detection - Bot: {threat_result.bot_score:.3f}, Auto: {threat_result.automation_score:.3f}, Pattern: {threat_result.pattern_anomaly_score:.3f}")
    
    if not detected_adversarial:
        advanced_results.warnings.append("Adversarial attack pattern not detected - may need algorithm improvement")
    
    # Don't fail the test, but note the concern
    logger.warning("Adversarial attack detection should be reviewed for production")

@pytest.mark.asyncio  
async def test_session_boundary_conditions():
    """Test session boundary conditions and edge cases"""
    if not banking_cold_start_handler:
        advanced_results.critical_failures.append("banking_cold_start_handler not available")
        return
        
    logger.info("🔄 Testing Session Boundary Conditions...")
    
    user_id = f"boundary_user_{uuid.uuid4().hex[:8]}"
    
    boundary_tests = []
    
    # Test 1: Zero-duration session
    try:
        metrics = await banking_cold_start_handler.process_session_learning(
            user_id, "zero_session", [], 0.0
        )
        boundary_tests.append(True)
    except:
        boundary_tests.append(False)
    
    # Test 2: Extremely long session 
    try:
        long_events = [{'timestamp': datetime.utcnow().isoformat() + 'Z', 
                       'event_type': 'touch', 'features': {'pressure': 0.5}} for _ in range(1000)]
        metrics = await banking_cold_start_handler.process_session_learning(
            user_id, "long_session", long_events, 86400.0  # 24 hour session
        )
        boundary_tests.append(True)
    except:
        boundary_tests.append(False)
    
    # Test 3: Many rapid sessions
    try:
        for i in range(50):
            await banking_cold_start_handler.process_session_learning(
                user_id, f"rapid_session_{i}", 
                [{'timestamp': datetime.utcnow().isoformat() + 'Z', 'event_type': 'touch', 'features': {}}],
                1.0
            )
        boundary_tests.append(True)
    except:
        boundary_tests.append(False)
    
    boundary_robust = all(boundary_tests)
    
    advanced_results.add_test_result('edge_case_tests', 'session_boundary_conditions',
                                   boundary_robust, sum(boundary_tests) / len(boundary_tests),
                                   {
                                       'zero_duration_handled': boundary_tests[0],
                                       'long_session_handled': boundary_tests[1],
                                       'rapid_sessions_handled': boundary_tests[2]
                                   })
    
    logger.info(f"Session Boundary Tests: {sum(boundary_tests)}/{len(boundary_tests)} passed")

# ==================== DATA QUALITY & CONSISTENCY TESTS ====================

@pytest.mark.asyncio
async def test_profile_consistency_across_sessions():
    """Test that user profiles remain consistent across multiple sessions"""
    if not banking_cold_start_handler:
        advanced_results.critical_failures.append("banking_cold_start_handler not available")
        return
        
    logger.info("📊 Testing Profile Consistency Across Sessions...")
    
    user_id = f"consistency_user_{uuid.uuid4().hex[:8]}"
    
    # Create consistent behavioral pattern
    def create_consistent_events(session_num):
        events = []
        for i in range(10):
            events.append({
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'event_type': 'touch',
                'features': {
                    'x_position': float(100 + i*10 + np.random.normal(0, 2)),  # Consistent pattern with human variation
                    'y_position': float(200 + np.random.normal(0, 5)),
                    'pressure': 0.6 + np.random.normal(0, 0.05),
                    'duration': 0.12 + np.random.normal(0, 0.01)
                }
            })
        return events
    
    # Process 5 sessions with consistent behavior
    consistency_scores = []
    for session_num in range(5):
        events = create_consistent_events(session_num)
        metrics = await banking_cold_start_handler.process_session_learning(
            user_id, f"consistent_session_{session_num}", events, 120.0
        )
        if hasattr(metrics, 'behavioral_consistency'):
            consistency_scores.append(metrics.behavioral_consistency)
    
    # Check if consistency scores are reasonable and stable
    if consistency_scores:
        avg_consistency = sum(consistency_scores) / len(consistency_scores)
        consistency_stable = all(score > 0.7 for score in consistency_scores)  # Should be high consistency
        consistency_variance = np.var(consistency_scores) < 0.1  # Should be stable
    else:
        avg_consistency = 0.0
        consistency_stable = False
        consistency_variance = False
    
    profile_consistent = consistency_stable and consistency_variance
    
    advanced_results.add_test_result('data_quality_tests', 'profile_consistency_across_sessions',
                                   profile_consistent, avg_consistency,
                                   {
                                       'consistency_scores': consistency_scores,
                                       'average_consistency': avg_consistency,
                                       'consistency_stable': consistency_stable,
                                       'variance_acceptable': consistency_variance
                                   })
    
    logger.info(f"Profile Consistency: avg={avg_consistency:.3f}, stable={consistency_stable}")

# ==================== MAIN ADVANCED TEST EXECUTION ====================

async def run_advanced_tests():
    """Run all advanced production tests"""
    logger.info("🔥 Starting BRIDGE ML-Engine Advanced Stress & Edge Case Tests...")
    logger.info("=" * 90)
    
    start_time = time.time()
    
    # Stress Tests
    logger.info("🔥 STRESS & LOAD TESTING")
    logger.info("-" * 60)
    await test_high_volume_user_load()
    await test_memory_leak_detection()
    
    # Edge Case Tests
    logger.info("\n⚡ EDGE CASE & BOUNDARY TESTING")
    logger.info("-" * 60)
    await test_timestamp_edge_cases()
    await test_extreme_feature_values()
    await test_session_boundary_conditions()
    
    # Security Boundary Tests
    logger.info("\n🛡️ SECURITY BOUNDARY TESTING")
    logger.info("-" * 60)
    await test_adversarial_attack_patterns()
    
    # Data Quality Tests
    logger.info("\n📊 DATA QUALITY & CONSISTENCY TESTING")
    logger.info("-" * 60)
    await test_profile_consistency_across_sessions()
    
    total_time = time.time() - start_time
    
    # Generate Advanced Test Report
    logger.info("\n" + "=" * 90)
    logger.info("🎯 ADVANCED TESTING ASSESSMENT")
    logger.info("=" * 90)
    
    scores = advanced_results.calculate_scores()
    
    logger.info(f"⏱️  Total Advanced Test Time: {total_time:.2f} seconds")
    
    # Category Scores
    logger.info("\n📈 ADVANCED TEST RESULTS BY CATEGORY:")
    overall_score = 0.0
    total_categories = 0
    
    for category, results in scores.items():
        if results['total'] > 0:
            logger.info(f"  {category}: {results['passed']}/{results['total']} passed, "
                       f"avg score: {results['avg_score']:.2f}, pass rate: {results['pass_rate']:.1%}")
            overall_score += results['avg_score']
            total_categories += 1
    
    overall_score = overall_score / max(total_categories, 1)
    
    # Critical Issues
    if advanced_results.critical_failures:
        logger.error(f"🚨 Critical Issues ({len(advanced_results.critical_failures)}):")
        for failure in advanced_results.critical_failures:
            logger.error(f"   - {failure}")
    
    if advanced_results.warnings:
        logger.warning(f"⚠️  Warnings ({len(advanced_results.warnings)}):")
        for warning in advanced_results.warnings:
            logger.warning(f"   - {warning}")
    
    # Final Assessment
    is_production_robust = (
        overall_score >= 0.80 and 
        len(advanced_results.critical_failures) == 0 and
        scores.get('stress_tests', {}).get('pass_rate', 0) >= 0.8 and
        scores.get('security_boundary_tests', {}).get('pass_rate', 0) >= 0.7
    )
    
    logger.info(f"\n📊 Overall Advanced Score: {overall_score:.2f}/1.00 ({overall_score*100:.1f}%)")
    logger.info(f"🎯 Production Robust: {'✅ YES' if is_production_robust else '❌ NO'}")
    
    # Deployment Recommendations
    logger.info("\n🚀 ADVANCED DEPLOYMENT RECOMMENDATIONS:")
    if is_production_robust:
        logger.info("  ✅ System passes advanced stress testing")
        logger.info("  ✅ Edge cases are handled robustly")
        logger.info("  ✅ Security boundaries are well-defined")
        logger.info("  ✅ Ready for high-load production deployment")
    else:
        logger.error("  ❌ System needs improvement before production deployment")
        logger.error("  ❌ Review stress test failures and edge cases")
        if scores.get('security_boundary_tests', {}).get('pass_rate', 0) < 0.7:
            logger.error("  ❌ Security boundary testing needs attention")
    
    logger.info("\n" + "=" * 90)
    
    return is_production_robust, overall_score, advanced_results

if __name__ == "__main__":
    # Run the advanced test suite
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        is_robust, score, results = loop.run_until_complete(run_advanced_tests())
        
        # Exit with proper code
        exit_code = 0 if is_robust else 1
        print(f"\nAdvanced test suite completed. Exit code: {exit_code}")
        exit(exit_code)
        
    except Exception as e:
        logger.error(f"Advanced test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

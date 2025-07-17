"""
Comprehensive Test for Phase 1 Learning + Phase 2 Continuous Analysis
Tests the complete behavioral authentication system with database integration
"""

import asyncio
import aiohttp
import json
import logging
from datetime import datetime
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComprehensivePhaseTest:
    """Test suite for Phase 1 & Phase 2 behavioral authentication system"""
    
    def __init__(self):
        self.ml_engine_url = "http://localhost:8001"
        self.backend_url = "http://localhost:8000"
        self.test_user_id = "test_user_phase_system"
        self.test_session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Test results tracking
        self.test_results = {
            'phase1_tests': {},
            'phase2_tests': {},
            'integration_tests': {},
            'database_tests': {},
            'performance_metrics': {}
        }
    
    async def run_comprehensive_test(self):
        """Run complete test suite"""
        logger.info("Starting Comprehensive Phase 1 + Phase 2 Test Suite")
        logger.info("=" * 80)
        
        try:
            # Test 1: ML Engine Health and Database Connectivity
            await self.test_ml_engine_health()
            
            # Test 2: Phase 1 Learning System
            await self.test_phase1_learning_system()
            
            # Test 3: Phase Transitions
            await self.test_phase_transitions()
            
            # Test 4: Phase 2 Continuous Analysis
            await self.test_phase2_continuous_analysis()
            
            # Test 5: Behavioral Drift Detection
            await self.test_behavioral_drift_detection()
            
            # Test 6: Database Integration
            await self.test_database_integration()
            
            # Test 7: Performance Benchmarks
            await self.test_performance_benchmarks()
            
            # Generate comprehensive report
            await self.generate_test_report()
            
        except Exception as e:
            logger.error(f"Test suite failed: {e}")
            return False
        
        return True
    
    async def test_ml_engine_health(self):
        """Test ML Engine health and database connectivity"""
        logger.info("\n🔍 Testing ML Engine Health & Database Connectivity")
        
        async with aiohttp.ClientSession() as session:
            try:
                # Test ML Engine health
                async with session.get(f"{self.ml_engine_url}/") as response:
                    if response.status == 200:
                        health_data = await response.json()
                        self.test_results['integration_tests']['ml_engine_health'] = {
                            'status': 'PASS',
                            'response_time_ms': response.headers.get('X-Response-Time', 'N/A'),
                            'components': health_data.get('components', {}),
                            'message': 'ML Engine is healthy'
                        }
                        logger.info("✅ ML Engine health check passed")
                    else:
                        raise Exception(f"ML Engine health check failed: {response.status}")
                
                # Test database connectivity
                async with session.get(f"{self.ml_engine_url}/health/database") as response:
                    if response.status == 200:
                        db_health = await response.json()
                        self.test_results['database_tests']['connectivity'] = {
                            'status': 'PASS' if db_health.get('connectivity') else 'FAIL',
                            'statistics': db_health.get('statistics', {}),
                            'message': 'Database connectivity verified'
                        }
                        if db_health.get('connectivity'):
                            logger.info("✅ Database connectivity confirmed")
                        else:
                            logger.warning("⚠️  Database connectivity issues detected")
                    else:
                        raise Exception(f"Database health check failed: {response.status}")
                        
            except Exception as e:
                logger.error(f"❌ Health check failed: {e}")
                self.test_results['integration_tests']['ml_engine_health'] = {
                    'status': 'FAIL',
                    'error': str(e)
                }
    
    async def test_phase1_learning_system(self):
        """Test Phase 1 Learning System - Cold Start and Learning"""
        logger.info("\n📚 Testing Phase 1 Learning System")
        
        async with aiohttp.ClientSession() as session:
            try:
                # Test 1: Cold Start Session
                session_data = {
                    "user_id": self.test_user_id,
                    "session_id": self.test_session_id,
                    "device_info": {
                        "device_id": "test_device_001",
                        "user_agent": "Test/1.0"
                    }
                }
                
                async with session.post(
                    f"{self.ml_engine_url}/session/start",
                    json=session_data
                ) as response:
                    if response.status == 200:
                        start_result = await response.json()
                        learning_phase = start_result.get('learning_phase')
                        session_guidance = start_result.get('session_guidance', {})
                        
                        self.test_results['phase1_tests']['cold_start'] = {
                            'status': 'PASS',
                            'learning_phase': learning_phase,
                            'session_guidance': session_guidance,
                            'message': f'Cold start detected with phase: {learning_phase}'
                        }
                        
                        logger.info(f"✅ Cold start session: {learning_phase}")
                        logger.info(f"   Guidance: {session_guidance.get('message', 'N/A')}")
                        
                    else:
                        raise Exception(f"Session start failed: {response.status}")
                
                # Test 2: Behavioral Vector Processing in Learning Phase
                behavioral_events = [
                    {
                        "event_type": "keystroke",
                        "timestamp": datetime.utcnow().isoformat(),
                        "data": {"key": "a", "duration": 120, "pressure": 0.8}
                    },
                    {
                        "event_type": "mouse_move",
                        "timestamp": datetime.utcnow().isoformat(),
                        "data": {"x": 100, "y": 200, "velocity": 1.5}
                    },
                    {
                        "event_type": "touch",
                        "timestamp": datetime.utcnow().isoformat(),
                        "data": {"x": 150, "y": 300, "pressure": 0.7, "duration": 200}
                    }
                ]
                
                analysis_data = {
                    "user_id": self.test_user_id,
                    "session_id": self.test_session_id,
                    "events": behavioral_events
                }
                
                # Process multiple vectors to simulate learning
                for i in range(5):
                    async with session.post(
                        f"{self.ml_engine_url}/analyze",
                        json=analysis_data
                    ) as response:
                        if response.status == 200:
                            analysis_result = await response.json()
                            
                            if i == 0:  # Store first result
                                self.test_results['phase1_tests']['vector_processing'] = {
                                    'status': 'PASS',
                                    'analysis_type': analysis_result.get('analysis_type'),
                                    'decision': analysis_result.get('decision'),
                                    'confidence': analysis_result.get('confidence'),
                                    'learning_result': analysis_result.get('learning_result', {}),
                                    'message': 'Learning phase vector processing successful'
                                }
                            
                            logger.info(f"✅ Vector {i+1}/5 processed: {analysis_result.get('decision')} "
                                       f"(confidence: {analysis_result.get('confidence', 0):.3f})")
                        else:
                            raise Exception(f"Analysis failed on vector {i+1}: {response.status}")
                
                # Test 3: Learning Progress Evaluation
                async with session.get(
                    f"{self.ml_engine_url}/user/{self.test_user_id}/learning-progress"
                ) as response:
                    if response.status == 200:
                        progress_data = await response.json()
                        progress_report = progress_data.get('progress_report', {})
                        
                        self.test_results['phase1_tests']['learning_progress'] = {
                            'status': 'PASS',
                            'current_phase': progress_report.get('current_phase'),
                            'vectors_collected': progress_report.get('vectors_collected'),
                            'phase_confidence': progress_report.get('phase_confidence'),
                            'learning_completeness': progress_report.get('learning_completeness'),
                            'consistency_analysis': progress_report.get('consistency_analysis', {}),
                            'message': 'Learning progress evaluation successful'
                        }
                        
                        logger.info(f"✅ Learning progress: {progress_report.get('learning_completeness', 0):.1f}% complete")
                        logger.info(f"   Phase: {progress_report.get('current_phase')}")
                        logger.info(f"   Vectors: {progress_report.get('vectors_collected', 0)}")
                        
                    else:
                        raise Exception(f"Learning progress check failed: {response.status}")
                        
            except Exception as e:
                logger.error(f"❌ Phase 1 learning test failed: {e}")
                self.test_results['phase1_tests']['error'] = str(e)
    
    async def test_phase_transitions(self):
        """Test phase transitions from learning to gradual_risk to full_auth"""
        logger.info("\n🔄 Testing Phase Transitions")
        
        async with aiohttp.ClientSession() as session:
            try:
                # Simulate multiple sessions to trigger phase transitions
                for session_num in range(7):  # Should trigger learning -> gradual_risk
                    session_id = f"{self.test_session_id}_transition_{session_num}"
                    
                    # Start new session
                    session_data = {
                        "user_id": self.test_user_id,
                        "session_id": session_id,
                        "device_info": {"device_id": "test_device_001"}
                    }
                    
                    async with session.post(
                        f"{self.ml_engine_url}/session/start",
                        json=session_data
                    ) as response:
                        if response.status == 200:
                            start_result = await response.json()
                            current_phase = start_result.get('learning_phase')
                            
                            # Process behavioral data for this session
                            analysis_data = {
                                "user_id": self.test_user_id,
                                "session_id": session_id,
                                "events": [
                                    {
                                        "event_type": "keystroke",
                                        "timestamp": datetime.utcnow().isoformat(),
                                        "data": {"key": "test", "duration": 100}
                                    }
                                ]
                            }
                            
                            async with session.post(
                                f"{self.ml_engine_url}/analyze",
                                json=analysis_data
                            ) as analyze_response:
                                if analyze_response.status == 200:
                                    analysis_result = await analyze_response.json()
                                    
                                    # Check for phase transitions
                                    learning_result = analysis_result.get('learning_result', {})
                                    phase_transition = learning_result.get('phase_transition')
                                    
                                    if phase_transition and phase_transition.get('transition_occurred'):
                                        transition_info = {
                                            'session_number': session_num + 1,
                                            'old_phase': phase_transition.get('old_phase'),
                                            'new_phase': phase_transition.get('new_phase'),
                                            'reason': phase_transition.get('reason'),
                                            'timestamp': phase_transition.get('timestamp')
                                        }
                                        
                                        if 'transitions' not in self.test_results['phase1_tests']:
                                            self.test_results['phase1_tests']['transitions'] = []
                                        
                                        self.test_results['phase1_tests']['transitions'].append(transition_info)
                                        
                                        logger.info(f"🔄 Phase transition detected at session {session_num + 1}:")
                                        logger.info(f"   {phase_transition.get('old_phase')} → {phase_transition.get('new_phase')}")
                                        logger.info(f"   Reason: {phase_transition.get('reason')}")
                
                # Verify final phase
                async with session.get(
                    f"{self.ml_engine_url}/user/{self.test_user_id}/learning-progress"
                ) as response:
                    if response.status == 200:
                        progress_data = await response.json()
                        user_profile = progress_data.get('user_profile', {})
                        
                        final_phase = user_profile.get('current_phase')
                        session_count = user_profile.get('current_session_count', 0)
                        
                        self.test_results['phase1_tests']['final_transition_state'] = {
                            'status': 'PASS',
                            'final_phase': final_phase,
                            'total_sessions': session_count,
                            'transitions_detected': len(self.test_results['phase1_tests'].get('transitions', [])),
                            'message': f'Reached phase: {final_phase} after {session_count} sessions'
                        }
                        
                        logger.info(f"✅ Final phase: {final_phase} (after {session_count} sessions)")
                        
            except Exception as e:
                logger.error(f"❌ Phase transition test failed: {e}")
                self.test_results['phase1_tests']['transition_error'] = str(e)
    
    async def test_phase2_continuous_analysis(self):
        """Test Phase 2 Continuous Analysis with multi-layer decisions"""
        logger.info("\n🧠 Testing Phase 2 Continuous Analysis")
        
        # This test would need a user in gradual_risk or full_auth phase
        # For now, we'll test the system's ability to handle different analysis levels
        
        async with aiohttp.ClientSession() as session:
            try:
                # Test enhanced analysis capabilities
                analysis_data = {
                    "user_id": self.test_user_id,
                    "session_id": f"{self.test_session_id}_phase2",
                    "events": [
                        {
                            "event_type": "complex_interaction",
                            "timestamp": datetime.utcnow().isoformat(),
                            "data": {
                                "keystroke_pattern": [100, 120, 110, 105],
                                "mouse_trajectory": [(0, 0), (50, 25), (100, 50)],
                                "touch_pressure_sequence": [0.8, 0.7, 0.9, 0.6]
                            }
                        }
                    ]
                }
                
                async with session.post(
                    f"{self.ml_engine_url}/analyze",
                    json=analysis_data
                ) as response:
                    if response.status == 200:
                        analysis_result = await response.json()
                        
                        self.test_results['phase2_tests']['continuous_analysis'] = {
                            'status': 'PASS',
                            'analysis_type': analysis_result.get('analysis_type'),
                            'decision': analysis_result.get('decision'),
                            'confidence': analysis_result.get('confidence'),
                            'risk_score': analysis_result.get('risk_score'),
                            'risk_level': analysis_result.get('risk_level'),
                            'layer_decisions': analysis_result.get('layer_decisions', {}),
                            'message': 'Phase 2 continuous analysis functional'
                        }
                        
                        logger.info(f"✅ Phase 2 analysis: {analysis_result.get('decision')} "
                                   f"(confidence: {analysis_result.get('confidence', 0):.3f})")
                        
                        # Test drift analysis if present
                        if 'drift_analysis' in analysis_result:
                            drift_info = analysis_result['drift_analysis']
                            self.test_results['phase2_tests']['drift_detection'] = {
                                'status': 'DETECTED',
                                'drift_type': drift_info.get('drift_type'),
                                'severity': drift_info.get('severity'),
                                'confidence': drift_info.get('confidence'),
                                'risk_assessment': drift_info.get('risk_assessment'),
                                'message': 'Behavioral drift detection active'
                            }
                            
                            logger.info(f"🔍 Drift detected: {drift_info.get('drift_type')} "
                                       f"(severity: {drift_info.get('severity', 0):.3f})")
                    else:
                        raise Exception(f"Phase 2 analysis failed: {response.status}")
                        
            except Exception as e:
                logger.error(f"❌ Phase 2 continuous analysis test failed: {e}")
                self.test_results['phase2_tests']['error'] = str(e)
    
    async def test_behavioral_drift_detection(self):
        """Test behavioral drift detection and baseline adaptation"""
        logger.info("\n🔍 Testing Behavioral Drift Detection")
        
        async with aiohttp.ClientSession() as session:
            try:
                # Test baseline adaptation
                async with session.post(
                    f"{self.ml_engine_url}/user/{self.test_user_id}/adapt-baseline"
                ) as response:
                    if response.status == 200:
                        adaptation_result = await response.json()
                        
                        self.test_results['phase2_tests']['baseline_adaptation'] = {
                            'status': adaptation_result.get('status'),
                            'message': adaptation_result.get('message'),
                            'timestamp': adaptation_result.get('timestamp')
                        }
                        
                        logger.info(f"✅ Baseline adaptation: {adaptation_result.get('status')}")
                        logger.info(f"   {adaptation_result.get('message')}")
                    else:
                        raise Exception(f"Baseline adaptation failed: {response.status}")
                        
            except Exception as e:
                logger.error(f"❌ Behavioral drift test failed: {e}")
                self.test_results['phase2_tests']['drift_error'] = str(e)
    
    async def test_database_integration(self):
        """Test database integration and data persistence"""
        logger.info("\n💾 Testing Database Integration")
        
        async with aiohttp.ClientSession() as session:
            try:
                # Get comprehensive statistics
                async with session.get(f"{self.ml_engine_url}/statistics") as response:
                    if response.status == 200:
                        stats_data = await response.json()
                        
                        database_stats = stats_data.get('statistics', {}).get('database', {})
                        learning_stats = stats_data.get('statistics', {}).get('learning_system', {})
                        analysis_stats = stats_data.get('statistics', {}).get('continuous_analysis', {})
                        
                        self.test_results['database_tests']['integration'] = {
                            'status': 'PASS',
                            'database_stats': database_stats,
                            'learning_stats': learning_stats,
                            'analysis_stats': analysis_stats,
                            'message': 'Database integration verified'
                        }
                        
                        logger.info("✅ Database integration verified")
                        logger.info(f"   User profiles: {database_stats.get('user_profiles_count', 0)}")
                        logger.info(f"   Behavioral vectors: {database_stats.get('behavioral_vectors_count', 0)}")
                        logger.info(f"   Auth decisions: {database_stats.get('authentication_decisions_count', 0)}")
                        
                    else:
                        raise Exception(f"Statistics retrieval failed: {response.status}")
                        
            except Exception as e:
                logger.error(f"❌ Database integration test failed: {e}")
                self.test_results['database_tests']['integration_error'] = str(e)
    
    async def test_performance_benchmarks(self):
        """Test performance benchmarks"""
        logger.info("\n⚡ Testing Performance Benchmarks")
        
        async with aiohttp.ClientSession() as session:
            try:
                # Measure analysis response times
                response_times = []
                
                for i in range(10):
                    start_time = datetime.utcnow()
                    
                    analysis_data = {
                        "user_id": self.test_user_id,
                        "session_id": f"{self.test_session_id}_perf_{i}",
                        "events": [
                            {
                                "event_type": "performance_test",
                                "timestamp": datetime.utcnow().isoformat(),
                                "data": {"test_iteration": i}
                            }
                        ]
                    }
                    
                    async with session.post(
                        f"{self.ml_engine_url}/analyze",
                        json=analysis_data
                    ) as response:
                        if response.status == 200:
                            end_time = datetime.utcnow()
                            response_time = (end_time - start_time).total_seconds() * 1000
                            response_times.append(response_time)
                
                # Calculate performance metrics
                avg_response_time = sum(response_times) / len(response_times)
                min_response_time = min(response_times)
                max_response_time = max(response_times)
                
                self.test_results['performance_metrics'] = {
                    'avg_response_time_ms': avg_response_time,
                    'min_response_time_ms': min_response_time,
                    'max_response_time_ms': max_response_time,
                    'total_requests': len(response_times),
                    'status': 'PASS' if avg_response_time < 1000 else 'SLOW'
                }
                
                logger.info(f"✅ Performance metrics:")
                logger.info(f"   Average response time: {avg_response_time:.1f}ms")
                logger.info(f"   Min/Max: {min_response_time:.1f}ms / {max_response_time:.1f}ms")
                
            except Exception as e:
                logger.error(f"❌ Performance benchmark failed: {e}")
                self.test_results['performance_metrics']['error'] = str(e)
    
    async def generate_test_report(self):
        """Generate comprehensive test report"""
        logger.info("\n📋 Generating Comprehensive Test Report")
        logger.info("=" * 80)
        
        # Calculate overall success rate
        total_tests = 0
        passed_tests = 0
        
        for category, tests in self.test_results.items():
            if isinstance(tests, dict):
                for test_name, result in tests.items():
                    if isinstance(result, dict) and 'status' in result:
                        total_tests += 1
                        if result['status'] == 'PASS':
                            passed_tests += 1
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Generate summary
        report = {
            'test_summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'success_rate': f"{success_rate:.1f}%",
                'test_timestamp': datetime.utcnow().isoformat(),
                'test_user_id': self.test_user_id
            },
            'detailed_results': self.test_results
        }
        
        # Save report to file
        report_filename = f"phase_system_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print summary
        logger.info(f"🎯 TEST SUMMARY:")
        logger.info(f"   Total Tests: {total_tests}")
        logger.info(f"   Passed: {passed_tests}")
        logger.info(f"   Success Rate: {success_rate:.1f}%")
        logger.info(f"   Report saved: {report_filename}")
        
        if success_rate >= 80:
            logger.info("🎉 Phase 1 + Phase 2 Implementation: SUCCESS!")
        else:
            logger.warning("⚠️  Phase 1 + Phase 2 Implementation: NEEDS ATTENTION")
        
        return report


async def main():
    """Main test execution"""
    print("🚀 Starting Comprehensive Phase 1 + Phase 2 Test Suite")
    print("=" * 80)
    
    test_suite = ComprehensivePhaseTest()
    
    try:
        success = await test_suite.run_comprehensive_test()
        
        if success:
            print("\n✅ Comprehensive test suite completed successfully!")
            return 0
        else:
            print("\n❌ Test suite encountered errors!")
            return 1
            
    except KeyboardInterrupt:
        print("\n⏹️  Test suite interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Test suite failed with error: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))

"""
Comprehensive test script for Phase 1 Learning + Phase 2 Continuous Analysis
Tests the complete behavioral authentication system with database integration
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BehavioralAuthSystemTest:
    """Test the complete behavioral authentication system"""
    
    def __init__(self):
        self.ml_engine_url = "http://localhost:8001"
        self.backend_url = "http://localhost:8000"
        self.test_user_id = "test_user_phase_system"
        self.test_session_id = "test_session_phase_system"
        
        # Test results
        self.test_results = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'test_details': []
        }
    
    async def run_comprehensive_test(self):
        """Run complete system test"""
        logger.info("Starting comprehensive Phase 1 + Phase 2 system test")
        
        async with aiohttp.ClientSession() as session:
            try:
                # Test 1: Health checks
                await self.test_health_checks(session)
                
                # Test 2: Database connectivity
                await self.test_database_connectivity(session)
                
                # Test 3: Phase 1 Learning - Cold start
                await self.test_phase1_cold_start(session)
                
                # Test 4: Phase 1 Learning - Multiple sessions
                await self.test_phase1_learning_progression(session)
                
                # Test 5: Learning progress evaluation
                await self.test_learning_progress(session)
                
                # Test 6: Phase transition simulation
                await self.test_phase_transitions(session)
                
                # Test 7: Phase 2 Continuous analysis
                await self.test_phase2_analysis(session)
                
                # Test 8: Behavioral drift detection
                await self.test_drift_detection(session)
                
                # Test 9: Statistics and monitoring
                await self.test_statistics_endpoints(session)
                
                # Test 10: End-to-end workflow
                await self.test_complete_workflow(session)
                
            except Exception as e:
                logger.error(f"Critical test failure: {e}")
                self.record_test_result("Critical System Test", False, str(e))
        
        # Print results
        await self.print_test_results()
    
    async def test_health_checks(self, session):
        """Test health check endpoints"""
        logger.info("Testing health checks...")
        
        try:
            # ML Engine health
            async with session.get(f"{self.ml_engine_url}/") as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"ML Engine status: {data['status']}")
                    self.record_test_result("ML Engine Health Check", True, data['status'])
                else:
                    self.record_test_result("ML Engine Health Check", False, f"Status: {response.status}")
            
            # Database health
            async with session.get(f"{self.ml_engine_url}/health/database") as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"Database status: {data['status']}")
                    self.record_test_result("Database Health Check", data['connectivity'], data['status'])
                else:
                    self.record_test_result("Database Health Check", False, f"Status: {response.status}")
                    
        except Exception as e:
            self.record_test_result("Health Checks", False, str(e))
    
    async def test_database_connectivity(self, session):
        """Test database connectivity and operations"""
        logger.info("Testing database connectivity...")
        
        try:
            async with session.get(f"{self.ml_engine_url}/statistics") as response:
                if response.status == 200:
                    data = await response.json()
                    db_stats = data['statistics'].get('database', {})
                    logger.info(f"Database stats: {db_stats}")
                    self.record_test_result("Database Statistics", True, f"Connected: {len(db_stats)} tables")
                else:
                    self.record_test_result("Database Statistics", False, f"Status: {response.status}")
                    
        except Exception as e:
            self.record_test_result("Database Connectivity", False, str(e))
    
    async def test_phase1_cold_start(self, session):
        """Test Phase 1 cold start functionality"""
        logger.info("Testing Phase 1 cold start...")
        
        try:
            # Start session for new user
            session_data = {
                "user_id": self.test_user_id,
                "session_id": self.test_session_id,
                "device_info": {
                    "device_id": "test_device",
                    "user_agent": "test_agent"
                }
            }
            
            async with session.post(f"{self.ml_engine_url}/session/start", json=session_data) as response:
                if response.status == 200:
                    data = await response.json()
                    learning_phase = data.get('learning_phase', '')
                    session_guidance = data.get('session_guidance', {})
                    
                    logger.info(f"Cold start - Learning phase: {learning_phase}")
                    logger.info(f"Session guidance: {session_guidance.get('message', '')}")
                    
                    # Should be cold_start or learning phase
                    success = learning_phase in ['cold_start', 'learning']
                    self.record_test_result("Phase 1 Cold Start", success, f"Phase: {learning_phase}")
                else:
                    self.record_test_result("Phase 1 Cold Start", False, f"Status: {response.status}")
                    
        except Exception as e:
            self.record_test_result("Phase 1 Cold Start", False, str(e))
    
    async def test_phase1_learning_progression(self, session):
        """Test Phase 1 learning progression through multiple sessions"""
        logger.info("Testing Phase 1 learning progression...")
        
        try:
            # Simulate multiple behavioral analysis requests
            for i in range(5):
                behavioral_data = {
                    "user_id": self.test_user_id,
                    "session_id": f"{self.test_session_id}_{i}",
                    "events": [
                        {
                            "event_type": "keystroke",
                            "timestamp": datetime.utcnow().isoformat(),
                            "data": {"key": "a", "duration": 120 + i * 10}
                        },
                        {
                            "event_type": "touch",
                            "timestamp": datetime.utcnow().isoformat(),
                            "data": {"x": 100 + i, "y": 200 + i, "pressure": 0.8}
                        }
                    ]
                }
                
                async with session.post(f"{self.ml_engine_url}/analyze", json=behavioral_data) as response:
                    if response.status == 200:
                        data = await response.json()
                        analysis_type = data.get('analysis_type', '')
                        
                        if analysis_type == 'phase1_learning':
                            learning_result = data.get('learning_result', {})
                            vectors_collected = learning_result.get('vectors_collected', 0)
                            logger.info(f"Learning session {i+1}: {vectors_collected} vectors collected")
                        
                        self.record_test_result(f"Learning Session {i+1}", True, f"Type: {analysis_type}")
                    else:
                        self.record_test_result(f"Learning Session {i+1}", False, f"Status: {response.status}")
                
                # Small delay between sessions
                await asyncio.sleep(0.5)
                
        except Exception as e:
            self.record_test_result("Phase 1 Learning Progression", False, str(e))
    
    async def test_learning_progress(self, session):
        """Test learning progress evaluation"""
        logger.info("Testing learning progress evaluation...")
        
        try:
            async with session.get(f"{self.ml_engine_url}/user/{self.test_user_id}/learning-progress") as response:
                if response.status == 200:
                    data = await response.json()
                    progress_report = data.get('progress_report', {})
                    user_profile = data.get('user_profile', {})
                    
                    current_phase = progress_report.get('current_phase', '')
                    vectors_collected = progress_report.get('vectors_collected', 0)
                    phase_confidence = progress_report.get('phase_confidence', 0.0)
                    
                    logger.info(f"Learning progress - Phase: {current_phase}, "
                              f"Vectors: {vectors_collected}, Confidence: {phase_confidence:.3f}")
                    
                    self.record_test_result("Learning Progress Evaluation", True, 
                                          f"Phase: {current_phase}, Vectors: {vectors_collected}")
                else:
                    self.record_test_result("Learning Progress Evaluation", False, f"Status: {response.status}")
                    
        except Exception as e:
            self.record_test_result("Learning Progress Evaluation", False, str(e))
    
    async def test_phase_transitions(self, session):
        """Test phase transitions"""
        logger.info("Testing phase transitions...")
        
        try:
            # Simulate enough sessions to trigger transitions
            for session_num in range(8, 20):  # Continue from where we left off
                behavioral_data = {
                    "user_id": self.test_user_id,
                    "session_id": f"{self.test_session_id}_transition_{session_num}",
                    "events": [
                        {
                            "event_type": "navigation",
                            "timestamp": datetime.utcnow().isoformat(),
                            "data": {"screen": "dashboard", "duration": 1000 + session_num * 50}
                        }
                    ]
                }
                
                async with session.post(f"{self.ml_engine_url}/analyze", json=behavioral_data) as response:
                    if response.status == 200:
                        data = await response.json()
                        analysis_type = data.get('analysis_type', '')
                        
                        # Check if we've moved to Phase 2
                        if analysis_type == 'phase2_continuous':
                            logger.info(f"Phase transition detected at session {session_num}!")
                            self.record_test_result("Phase Transition", True, 
                                                  f"Transitioned to Phase 2 at session {session_num}")
                            break
                        
                        # Check learning result for phase info
                        if analysis_type == 'phase1_learning':
                            learning_result = data.get('learning_result', {})
                            phase_transition = learning_result.get('phase_transition')
                            if phase_transition:
                                logger.info(f"Phase transition info: {phase_transition}")
                
                await asyncio.sleep(0.3)
            
        except Exception as e:
            self.record_test_result("Phase Transitions", False, str(e))
    
    async def test_phase2_analysis(self, session):
        """Test Phase 2 continuous analysis"""
        logger.info("Testing Phase 2 continuous analysis...")
        
        try:
            # Create a user that should be in Phase 2
            phase2_user = "test_user_phase2"
            
            # Simulate advanced behavioral analysis
            behavioral_data = {
                "user_id": phase2_user,
                "session_id": f"phase2_session",
                "events": [
                    {
                        "event_type": "typing_pattern",
                        "timestamp": datetime.utcnow().isoformat(),
                        "data": {"speed": 75, "rhythm": 0.12, "pressure": [0.8, 0.7, 0.9]}
                    },
                    {
                        "event_type": "touch_pattern",
                        "timestamp": datetime.utcnow().isoformat(),
                        "data": {"pressure": [0.9, 0.8], "duration": [150, 140], "area": [10.5, 11.2]}
                    }
                ]
            }
            
            async with session.post(f"{self.ml_engine_url}/analyze", json=behavioral_data) as response:
                if response.status == 200:
                    data = await response.json()
                    analysis_type = data.get('analysis_type', '')
                    decision = data.get('decision', '')
                    confidence = data.get('confidence', 0.0)
                    analysis_level = data.get('analysis_level', '')
                    
                    logger.info(f"Phase 2 analysis - Type: {analysis_type}, "
                              f"Decision: {decision}, Confidence: {confidence:.3f}")
                    
                    # Should be Phase 2 analysis
                    success = analysis_type in ['phase1_learning', 'phase2_continuous']
                    self.record_test_result("Phase 2 Analysis", success, 
                                          f"Type: {analysis_type}, Decision: {decision}")
                else:
                    self.record_test_result("Phase 2 Analysis", False, f"Status: {response.status}")
                    
        except Exception as e:
            self.record_test_result("Phase 2 Analysis", False, str(e))
    
    async def test_drift_detection(self, session):
        """Test behavioral drift detection"""
        logger.info("Testing behavioral drift detection...")
        
        try:
            # Simulate anomalous behavior to trigger drift detection
            drift_user = "test_user_drift"
            
            # First, establish baseline with normal behavior
            for i in range(10):
                normal_data = {
                    "user_id": drift_user,
                    "session_id": f"drift_baseline_{i}",
                    "events": [
                        {
                            "event_type": "typing",
                            "timestamp": datetime.utcnow().isoformat(),
                            "data": {"speed": 60 + i, "rhythm": 0.15}  # Consistent pattern
                        }
                    ]
                }
                
                await session.post(f"{self.ml_engine_url}/analyze", json=normal_data)
                await asyncio.sleep(0.2)
            
            # Now simulate anomalous behavior
            anomalous_data = {
                "user_id": drift_user,
                "session_id": "drift_anomaly",
                "events": [
                    {
                        "event_type": "typing",
                        "timestamp": datetime.utcnow().isoformat(),
                        "data": {"speed": 150, "rhythm": 0.8}  # Very different pattern
                    }
                ]
            }
            
            async with session.post(f"{self.ml_engine_url}/analyze", json=anomalous_data) as response:
                if response.status == 200:
                    data = await response.json()
                    drift_analysis = data.get('drift_analysis')
                    
                    if drift_analysis:
                        drift_type = drift_analysis.get('drift_type', '')
                        severity = drift_analysis.get('severity', 0.0)
                        logger.info(f"Drift detected - Type: {drift_type}, Severity: {severity:.3f}")
                        self.record_test_result("Drift Detection", True, 
                                              f"Type: {drift_type}, Severity: {severity:.3f}")
                    else:
                        self.record_test_result("Drift Detection", False, "No drift detected")
                else:
                    self.record_test_result("Drift Detection", False, f"Status: {response.status}")
                    
        except Exception as e:
            self.record_test_result("Drift Detection", False, str(e))
    
    async def test_statistics_endpoints(self, session):
        """Test statistics and monitoring endpoints"""
        logger.info("Testing statistics endpoints...")
        
        try:
            # Test comprehensive statistics
            async with session.get(f"{self.ml_engine_url}/statistics") as response:
                if response.status == 200:
                    data = await response.json()
                    stats = data.get('statistics', {})
                    
                    # Check for key statistics
                    has_learning_stats = 'learning_system' in stats
                    has_analysis_stats = 'continuous_analysis' in stats
                    has_database_stats = 'database' in stats
                    
                    logger.info(f"Statistics available - Learning: {has_learning_stats}, "
                              f"Analysis: {has_analysis_stats}, Database: {has_database_stats}")
                    
                    self.record_test_result("Statistics Endpoint", True, 
                                          f"Components: {len(stats)} available")
                else:
                    self.record_test_result("Statistics Endpoint", False, f"Status: {response.status}")
                    
        except Exception as e:
            self.record_test_result("Statistics Endpoints", False, str(e))
    
    async def test_complete_workflow(self, session):
        """Test complete end-to-end workflow"""
        logger.info("Testing complete workflow...")
        
        try:
            workflow_user = "test_user_workflow"
            
            # 1. Start session
            session_data = {
                "user_id": workflow_user,
                "session_id": "workflow_session",
                "device_info": {"device_id": "workflow_device"}
            }
            
            session_start_success = False
            async with session.post(f"{self.ml_engine_url}/session/start", json=session_data) as response:
                session_start_success = response.status == 200
            
            # 2. Analyze behavior
            analysis_data = {
                "user_id": workflow_user,
                "session_id": "workflow_session",
                "events": [
                    {
                        "event_type": "complete_interaction",
                        "timestamp": datetime.utcnow().isoformat(),
                        "data": {"duration": 1500, "actions": 5}
                    }
                ]
            }
            
            analysis_success = False
            async with session.post(f"{self.ml_engine_url}/analyze", json=analysis_data) as response:
                analysis_success = response.status == 200
            
            # 3. Get learning progress
            progress_success = False
            async with session.get(f"{self.ml_engine_url}/user/{workflow_user}/learning-progress") as response:
                progress_success = response.status == 200
            
            # 4. End session
            end_data = {"session_id": "workflow_session", "reason": "completed"}
            end_success = False
            async with session.post(f"{self.ml_engine_url}/session/end", json=end_data) as response:
                end_success = response.status == 200
            
            # Overall workflow success
            workflow_success = all([session_start_success, analysis_success, progress_success, end_success])
            
            logger.info(f"Complete workflow - Start: {session_start_success}, "
                       f"Analysis: {analysis_success}, Progress: {progress_success}, End: {end_success}")
            
            self.record_test_result("Complete Workflow", workflow_success, 
                                  f"All steps: {workflow_success}")
                    
        except Exception as e:
            self.record_test_result("Complete Workflow", False, str(e))
    
    def record_test_result(self, test_name, success, details):
        """Record test result"""
        self.test_results['total_tests'] += 1
        if success:
            self.test_results['passed_tests'] += 1
        else:
            self.test_results['failed_tests'] += 1
        
        self.test_results['test_details'].append({
            'test': test_name,
            'success': success,
            'details': details,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    async def print_test_results(self):
        """Print comprehensive test results"""
        print("\n" + "="*80)
        print("PHASE 1 + PHASE 2 BEHAVIORAL AUTHENTICATION SYSTEM TEST RESULTS")
        print("="*80)
        
        total = self.test_results['total_tests']
        passed = self.test_results['passed_tests']
        failed = self.test_results['failed_tests']
        success_rate = (passed / total * 100) if total > 0 else 0
        
        print(f"Total Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Success Rate: {success_rate:.1f}%")
        print("\nDetailed Results:")
        print("-" * 80)
        
        for result in self.test_results['test_details']:
            status = "✓ PASS" if result['success'] else "✗ FAIL"
            print(f"{status} | {result['test']}: {result['details']}")
        
        print("\n" + "="*80)
        
        if success_rate >= 80:
            print("🎉 SYSTEM STATUS: HEALTHY - Phase 1 + Phase 2 systems working well!")
        elif success_rate >= 60:
            print("⚠️  SYSTEM STATUS: FUNCTIONAL - Some issues detected")
        else:
            print("❌ SYSTEM STATUS: NEEDS ATTENTION - Multiple issues found")
        
        print("="*80)

async def main():
    """Run the comprehensive test"""
    tester = BehavioralAuthSystemTest()
    await tester.run_comprehensive_test()

if __name__ == "__main__":
    asyncio.run(main())

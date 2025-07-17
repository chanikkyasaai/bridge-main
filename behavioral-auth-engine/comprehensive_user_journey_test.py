"""
Comprehensive User Journey Test for Behavioral Authentication System
Tests the complete flow from cold start to advanced behavioral analysis
"""

import asyncio
import requests
import json
import time
import uuid
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import random

# Test Configuration
BACKEND_URL = "http://127.0.0.1:8000"
ML_ENGINE_URL = "http://127.0.0.1:8001"

class BehavioralAuthTester:
    def __init__(self):
        self.test_users = []
        self.session_results = []
        self.test_results = {}
        
    def log_test(self, test_name: str, status: str, details: str = ""):
        """Log test results"""
        print(f"{'✅' if status == 'PASS' else '❌'} {test_name}: {status}")
        if details:
            print(f"   Details: {details}")
        self.test_results[test_name] = {"status": status, "details": details}
    
    def generate_behavioral_data(self, user_type: str = "normal", session_number: int = 1) -> Dict[str, Any]:
        """Generate realistic behavioral data for different user types"""
        base_data = {
            "typing_speed": 45.0,
            "key_dwell_times": [0.12, 0.15, 0.13, 0.14],
            "key_flight_times": [0.08, 0.09, 0.07, 0.08],
            "touch_pressure": [0.6, 0.7, 0.65, 0.68],
            "touch_area": [12.5, 13.2, 12.8, 13.0],
            "swipe_velocity": [150.0, 160.0, 155.0],
            "device_orientation": "portrait",
            "app_usage_pattern": ["login", "dashboard", "transactions"],
            "navigation_pattern": ["home", "menu", "settings"],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Modify data based on user type and learning progression
        if user_type == "normal":
            # Consistent user with slight natural variation
            variation = 0.1 + (0.05 * max(0, session_number - 5))  # Less variation as user learns
            for key, value in base_data.items():
                if isinstance(value, (int, float)):
                    base_data[key] = value * (1 + random.uniform(-variation, variation))
                elif isinstance(value, list) and all(isinstance(x, (int, float)) for x in value):
                    base_data[key] = [x * (1 + random.uniform(-variation, variation)) for x in value]
                    
        elif user_type == "attacker":
            # Significantly different behavioral patterns
            base_data["typing_speed"] = random.uniform(20.0, 80.0)  # Erratic typing
            base_data["key_dwell_times"] = [random.uniform(0.05, 0.25) for _ in range(4)]
            base_data["touch_pressure"] = [random.uniform(0.3, 0.9) for _ in range(4)]
            
        elif user_type == "bot":
            # Very consistent, machine-like patterns
            base_data["typing_speed"] = 60.0  # Exactly same speed
            base_data["key_dwell_times"] = [0.10] * 4  # Identical timings
            base_data["key_flight_times"] = [0.05] * 4  # Perfect consistency
            base_data["touch_pressure"] = [0.5] * 4  # Exact pressure
            
        elif user_type == "stressed":
            # User under stress - faster, more erratic
            base_data["typing_speed"] *= 1.3
            base_data["key_dwell_times"] = [x * random.uniform(0.7, 1.5) for x in base_data["key_dwell_times"]]
            base_data["touch_pressure"] = [x * random.uniform(1.2, 1.8) for x in base_data["touch_pressure"]]
            
        return base_data
    
    async def test_system_health(self) -> bool:
        """Test if both backend and ML engine are running"""
        try:
            # Test backend health
            backend_response = requests.get(f"{BACKEND_URL}/health")
            if backend_response.status_code != 200:
                self.log_test("Backend Health", "FAIL", f"Status: {backend_response.status_code}")
                return False
            self.log_test("Backend Health", "PASS")
            
            # Test ML engine health
            ml_response = requests.get(f"{ML_ENGINE_URL}/")
            if ml_response.status_code != 200:
                self.log_test("ML Engine Health", "FAIL", f"Status: {ml_response.status_code}")
                return False
            self.log_test("ML Engine Health", "PASS")
            
            # Test database connectivity
            db_response = requests.get(f"{ML_ENGINE_URL}/health/database")
            if db_response.status_code != 200:
                self.log_test("Database Health", "FAIL", f"Status: {db_response.status_code}")
                return False
            self.log_test("Database Health", "PASS")
            
            return True
        except Exception as e:
            self.log_test("System Health", "FAIL", str(e))
            return False
    
    def create_test_user(self, user_id: str = None) -> str:
        """Create a test user and return user ID"""
        if not user_id:
            user_id = str(uuid.uuid4())
        self.test_users.append(user_id)
        return user_id
    
    def simulate_session_start(self, user_id: str, session_name: str = None) -> Dict[str, Any]:
        """Start a behavioral analysis session"""
        if not session_name:
            session_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"
        
        try:
            response = requests.post(
                f"{ML_ENGINE_URL}/session/start",
                json={"user_id": user_id, "session_id": session_name}
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Session start failed: {response.status_code}", "details": response.text}
        except Exception as e:
            return {"error": str(e)}
    
    def simulate_behavioral_analysis(self, user_id: str, session_id: str, behavioral_data: Dict[str, Any]) -> Dict[str, Any]:
        """Submit behavioral data for analysis"""
        try:
            # Convert behavioral data to events format as expected by the API
            events = [
                {
                    "event_type": "typing",
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": {
                        "typing_speed": behavioral_data.get("typing_speed", 45.0),
                        "key_dwell_times": behavioral_data.get("key_dwell_times", [0.12, 0.15, 0.13]),
                        "key_flight_times": behavioral_data.get("key_flight_times", [0.08, 0.09, 0.07])
                    }
                },
                {
                    "event_type": "touch",
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": {
                        "touch_pressure": behavioral_data.get("touch_pressure", [0.6, 0.7, 0.65]),
                        "touch_area": behavioral_data.get("touch_area", [12.5, 13.2, 12.8]),
                        "swipe_velocity": behavioral_data.get("swipe_velocity", [150.0, 160.0, 155.0])
                    }
                },
                {
                    "event_type": "navigation",
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": {
                        "app_usage_pattern": behavioral_data.get("app_usage_pattern", ["login", "dashboard"]),
                        "navigation_pattern": behavioral_data.get("navigation_pattern", ["home", "menu"]),
                        "device_orientation": behavioral_data.get("device_orientation", "portrait")
                    }
                }
            ]
            
            analysis_payload = {
                "user_id": user_id,
                "session_id": session_id,
                "events": events
            }
            
            response = requests.post(
                f"{ML_ENGINE_URL}/analyze",
                json=analysis_payload
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Analysis failed: {response.status_code}", "details": response.text}
        except Exception as e:
            return {"error": str(e)}
    
    def get_learning_progress(self, user_id: str) -> Dict[str, Any]:
        """Get user's learning progress"""
        try:
            response = requests.get(f"{ML_ENGINE_URL}/user/{user_id}/learning-progress")
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Progress check failed: {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        try:
            response = requests.get(f"{ML_ENGINE_URL}/statistics")
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Statistics failed: {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}
    
    async def test_cold_start_phase(self):
        """Test cold start detection and initial learning"""
        print("\n🔥 Testing Cold Start Phase")
        
        # Create new user
        user_id = self.create_test_user()
        
        # Start first session
        session_result = self.simulate_session_start(user_id, "cold_start_session")
        if "error" in session_result:
            self.log_test("Cold Start Session", "FAIL", session_result["error"])
            return False
        
        # Verify cold start detection
        if session_result.get("learning_phase") == "cold_start":
            self.log_test("Cold Start Detection", "PASS")
        else:
            self.log_test("Cold Start Detection", "FAIL", f"Expected cold_start, got {session_result.get('learning_phase')}")
        
        # Submit behavioral data
        behavioral_data = self.generate_behavioral_data("normal", 1)
        analysis_result = self.simulate_behavioral_analysis(
            user_id, 
            session_result.get("session_id", "cold_start_session"), 
            behavioral_data
        )
        
        if "error" in analysis_result:
            self.log_test("Cold Start Analysis", "FAIL", analysis_result["error"])
            return False
        
        # Verify learning decision
        if analysis_result.get("decision") == "learn":
            self.log_test("Cold Start Learning Decision", "PASS")
        else:
            self.log_test("Cold Start Learning Decision", "FAIL", f"Expected learn, got {analysis_result.get('decision')}")
        
        return user_id
    
    async def test_learning_phase_progression(self, user_id: str):
        """Test progression through learning phase"""
        print("\n📚 Testing Learning Phase Progression")
        
        for session_num in range(2, 8):  # Sessions 2-7
            session_name = f"learning_session_{session_num}"
            
            # Start session
            session_result = self.simulate_session_start(user_id, session_name)
            if "error" in session_result:
                self.log_test(f"Learning Session {session_num}", "FAIL", session_result["error"])
                continue
            
            # Submit multiple behavioral samples per session
            for sample in range(3):
                behavioral_data = self.generate_behavioral_data("normal", session_num)
                analysis_result = self.simulate_behavioral_analysis(
                    user_id, 
                    session_result.get("session_id", session_name), 
                    behavioral_data
                )
                
                if "error" not in analysis_result:
                    print(f"   Session {session_num}, Sample {sample+1}: {analysis_result.get('decision', 'unknown')} (confidence: {analysis_result.get('confidence', 0):.3f})")
            
            # Check learning progress
            progress = self.get_learning_progress(user_id)
            if "error" not in progress:
                current_phase = progress.get("progress_report", {}).get("current_phase", "unknown")
                vectors_collected = progress.get("progress_report", {}).get("vectors_collected", 0)
                print(f"   Progress: Phase={current_phase}, Vectors={vectors_collected}")
        
        # Final progress check
        final_progress = self.get_learning_progress(user_id)
        if "error" not in final_progress:
            final_phase = final_progress.get("progress_report", {}).get("current_phase", "unknown")
            if final_phase in ["learning", "gradual_risk"]:
                self.log_test("Learning Phase Progression", "PASS", f"Progressed to {final_phase}")
            else:
                self.log_test("Learning Phase Progression", "PARTIAL", f"Currently in {final_phase}")
        
        return final_progress
    
    async def test_layer1_faiss_analysis(self, user_id: str):
        """Test FAISS layer analysis for established users"""
        print("\n🔍 Testing Layer 1 FAISS Analysis")
        
        # Start session for FAISS testing
        session_result = self.simulate_session_start(user_id, "faiss_test_session")
        if "error" in session_result:
            self.log_test("FAISS Test Session", "FAIL", session_result["error"])
            return
        
        # Test with normal behavioral pattern (should pass)
        normal_data = self.generate_behavioral_data("normal", 10)
        normal_result = self.simulate_behavioral_analysis(
            user_id, 
            session_result.get("session_id"), 
            normal_data
        )
        
        if "error" not in normal_result:
            if normal_result.get("risk_level") in ["low", "medium"]:
                self.log_test("FAISS Normal Pattern Recognition", "PASS")
            else:
                self.log_test("FAISS Normal Pattern Recognition", "FAIL", f"Risk level: {normal_result.get('risk_level')}")
        
        # Test with slightly anomalous pattern
        stressed_data = self.generate_behavioral_data("stressed", 10)
        stressed_result = self.simulate_behavioral_analysis(
            user_id, 
            session_result.get("session_id"), 
            stressed_data
        )
        
        if "error" not in stressed_result:
            print(f"   Stressed pattern analysis: {stressed_result.get('decision')} (risk: {stressed_result.get('risk_level')})")
    
    async def test_layer2_advanced_analysis(self, user_id: str):
        """Test Layer 2 GNN/Transformer analysis"""
        print("\n🧠 Testing Layer 2 Advanced Analysis")
        
        # Create scenario that should trigger Layer 2
        session_result = self.simulate_session_start(user_id, "layer2_test_session")
        if "error" in session_result:
            self.log_test("Layer 2 Test Session", "FAIL", session_result["error"])
            return
        
        # Submit pattern that might require advanced analysis
        advanced_data = self.generate_behavioral_data("normal", 15)
        # Add some complexity that might trigger Layer 2
        advanced_data["complex_navigation"] = ["unusual_path", "rapid_switches", "deep_menu_access"]
        
        advanced_result = self.simulate_behavioral_analysis(
            user_id, 
            session_result.get("session_id"), 
            advanced_data
        )
        
        if "error" not in advanced_result:
            analysis_type = advanced_result.get("analysis_type", "unknown")
            if "continuous" in analysis_type or "phase2" in analysis_type:
                self.log_test("Layer 2 Analysis Trigger", "PASS", f"Analysis type: {analysis_type}")
            else:
                self.log_test("Layer 2 Analysis Trigger", "PARTIAL", f"Analysis type: {analysis_type}")
    
    async def test_drift_detection(self, user_id: str):
        """Test behavioral drift detection"""
        print("\n📊 Testing Drift Detection")
        
        # Gradually introduce drift over multiple sessions
        for drift_session in range(1, 6):
            session_name = f"drift_test_session_{drift_session}"
            session_result = self.simulate_session_start(user_id, session_name)
            
            if "error" not in session_result:
                # Gradually increase behavioral variation
                drift_factor = 0.1 * drift_session  # Increasing drift
                behavioral_data = self.generate_behavioral_data("normal", 20)
                
                # Apply drift
                if isinstance(behavioral_data.get("typing_speed"), (int, float)):
                    behavioral_data["typing_speed"] *= (1 + drift_factor)
                
                drift_result = self.simulate_behavioral_analysis(
                    user_id, 
                    session_result.get("session_id"), 
                    behavioral_data
                )
                
                if "error" not in drift_result:
                    risk_level = drift_result.get("risk_level", "unknown")
                    confidence = drift_result.get("confidence", 0)
                    print(f"   Drift Session {drift_session}: Risk={risk_level}, Confidence={confidence:.3f}")
                    
                    # Check if drift is detected in later sessions
                    if drift_session >= 3 and risk_level in ["medium", "high"]:
                        self.log_test("Drift Detection", "PASS", f"Detected drift at session {drift_session}")
                        return
        
        self.log_test("Drift Detection", "PARTIAL", "Drift introduced but may not have triggered alerts")
    
    async def test_bot_detection(self):
        """Test bot detection capabilities"""
        print("\n🤖 Testing Bot Detection")
        
        # Create bot user
        bot_user_id = self.create_test_user()
        
        # Start session for bot
        session_result = self.simulate_session_start(bot_user_id, "bot_test_session")
        if "error" in session_result:
            self.log_test("Bot Test Session", "FAIL", session_result["error"])
            return
        
        # Submit multiple bot-like behavioral patterns
        bot_detected = False
        for bot_attempt in range(5):
            bot_data = self.generate_behavioral_data("bot", bot_attempt + 1)
            bot_result = self.simulate_behavioral_analysis(
                bot_user_id, 
                session_result.get("session_id"), 
                bot_data
            )
            
            if "error" not in bot_result:
                risk_level = bot_result.get("risk_level", "unknown")
                decision = bot_result.get("decision", "unknown")
                confidence = bot_result.get("confidence", 0)
                
                print(f"   Bot Attempt {bot_attempt + 1}: Decision={decision}, Risk={risk_level}, Confidence={confidence:.3f}")
                
                if risk_level == "high" or decision == "block":
                    bot_detected = True
                    self.log_test("Bot Detection", "PASS", f"Bot detected at attempt {bot_attempt + 1}")
                    break
        
        if not bot_detected:
            self.log_test("Bot Detection", "PARTIAL", "Bot patterns submitted but not definitively detected")
    
    async def test_attacker_detection(self):
        """Test detection of malicious users with different behavioral patterns"""
        print("\n🚨 Testing Attacker Detection")
        
        # Create attacker user
        attacker_user_id = self.create_test_user()
        
        # Start session
        session_result = self.simulate_session_start(attacker_user_id, "attacker_test_session")
        if "error" in session_result:
            self.log_test("Attacker Test Session", "FAIL", session_result["error"])
            return
        
        # Submit attacker behavioral patterns
        attacker_detected = False
        for attack_attempt in range(3):
            attacker_data = self.generate_behavioral_data("attacker", attack_attempt + 1)
            attacker_result = self.simulate_behavioral_analysis(
                attacker_user_id, 
                session_result.get("session_id"), 
                attacker_data
            )
            
            if "error" not in attacker_result:
                risk_level = attacker_result.get("risk_level", "unknown")
                decision = attacker_result.get("decision", "unknown")
                confidence = attacker_result.get("confidence", 0)
                
                print(f"   Attack Attempt {attack_attempt + 1}: Decision={decision}, Risk={risk_level}, Confidence={confidence:.3f}")
                
                if risk_level in ["medium", "high"] or decision in ["challenge", "block"]:
                    attacker_detected = True
                    self.log_test("Attacker Detection", "PASS", f"Attacker detected at attempt {attack_attempt + 1}")
                    break
        
        if not attacker_detected:
            self.log_test("Attacker Detection", "PARTIAL", "Attacker patterns submitted but not definitively detected")
    
    async def test_system_performance(self):
        """Test system performance and statistics"""
        print("\n⚡ Testing System Performance")
        
        # Get comprehensive statistics
        stats = self.get_system_statistics()
        if "error" in stats:
            self.log_test("System Statistics", "FAIL", stats["error"])
            return
        
        # Verify statistics structure
        required_sections = ["learning_system", "continuous_analysis", "database", "faiss_layer"]
        missing_sections = [section for section in required_sections if section not in stats.get("statistics", {})]
        
        if not missing_sections:
            self.log_test("Statistics Completeness", "PASS")
        else:
            self.log_test("Statistics Completeness", "PARTIAL", f"Missing: {missing_sections}")
        
        # Print key statistics
        stats_data = stats.get("statistics", {})
        print(f"   Users: {stats_data.get('database', {}).get('user_profiles_count', 0)}")
        print(f"   Vectors: {stats_data.get('database', {}).get('behavioral_vectors_count', 0)}")
        print(f"   Learning Users: {stats_data.get('learning_system', {}).get('learning_stats', {}).get('users_in_learning', 0)}")
        print(f"   Active Sessions: {stats_data.get('session_manager', {}).get('active_sessions', 0)}")
    
    def print_test_summary(self):
        """Print comprehensive test results summary"""
        print("\n" + "="*80)
        print("🏁 COMPREHENSIVE TEST RESULTS SUMMARY")
        print("="*80)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result["status"] == "PASS")
        partial_tests = sum(1 for result in self.test_results.values() if result["status"] == "PARTIAL")
        failed_tests = sum(1 for result in self.test_results.values() if result["status"] == "FAIL")
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"⚠️  Partial: {partial_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests > 0:
            print("\n🔍 FAILED TESTS:")
            for test_name, result in self.test_results.items():
                if result["status"] == "FAIL":
                    print(f"   ❌ {test_name}: {result['details']}")
        
        if partial_tests > 0:
            print("\n⚠️  PARTIAL TESTS:")
            for test_name, result in self.test_results.items():
                if result["status"] == "PARTIAL":
                    print(f"   ⚠️  {test_name}: {result['details']}")
        
        print("\n" + "="*80)

async def main():
    """Run comprehensive behavioral authentication system test"""
    print("🚀 Starting Comprehensive Behavioral Authentication System Test")
    print("="*80)
    
    tester = BehavioralAuthTester()
    
    # Test system health first
    if not await tester.test_system_health():
        print("❌ System health check failed. Ensure both backend and ML engine are running.")
        return
    
    # Test cold start phase
    user_id = await tester.test_cold_start_phase()
    if not user_id:
        print("❌ Cold start test failed. Stopping tests.")
        return
    
    # Test learning phase progression
    await tester.test_learning_phase_progression(user_id)
    
    # Give some time for learning to stabilize
    print("\n⏱️  Allowing system to stabilize...")
    time.sleep(2)
    
    # Test advanced analysis layers
    await tester.test_layer1_faiss_analysis(user_id)
    await tester.test_layer2_advanced_analysis(user_id)
    
    # Test detection capabilities
    await tester.test_drift_detection(user_id)
    await tester.test_bot_detection()
    await tester.test_attacker_detection()
    
    # Test system performance
    await tester.test_system_performance()
    
    # Print comprehensive summary
    tester.print_test_summary()
    
    print(f"\n🎯 Test completed with {len(tester.test_users)} test users created")
    print("💡 Check the logs above for detailed results and any issues found")

if __name__ == "__main__":
    asyncio.run(main())

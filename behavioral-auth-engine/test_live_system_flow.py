"""
Live System End-to-End Test
Tests the complete system: ML Engine + Backend
Shows real FAISS scores, escalation, and blocking decisions
"""
import requests
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List
import asyncio

# System endpoints
ML_ENGINE_BASE = "http://localhost:8001"
BACKEND_BASE = "http://localhost:8000"

class LiveSystemTester:
    """Test the complete live system with real behavioral data"""
    
    def __init__(self):
        self.test_users = []
        self.test_sessions = []
        
    def create_normal_user_data(self, user_id: str) -> Dict[str, Any]:
        """Create normal behavioral data for a user"""
        return {
            "user_id": user_id,
            "session_id": f"session_{int(datetime.now().timestamp())}_{user_id}",
            "device_info": {
                "device_type": "mobile",
                "os": "iOS",
                "screen_resolution": "1170x2532",
                "device_model": "iPhone13",
                "user_agent": "CanaraAI/2.1.5 (iOS 15.0)"
            },
            "session_context": {
                "location": "home",
                "time_of_day": "morning",
                "app_version": "2.1.5",
                "network_type": "wifi"
            },
            "behavioral_logs": [
                {
                    "timestamp": datetime.now().isoformat(),
                    "event_type": "touch_sequence",
                    "data": {
                        "touch_events": [
                            {"x": 150, "y": 400, "pressure": 0.6, "duration": 120},
                            {"x": 155, "y": 405, "pressure": 0.65, "duration": 115},
                            {"x": 160, "y": 410, "pressure": 0.7, "duration": 125}
                        ],
                        "accelerometer": {"x": 0.02, "y": 0.15, "z": 9.78},
                        "gyroscope": {"x": 0.001, "y": 0.002, "z": 0.0015}
                    }
                },
                {
                    "timestamp": (datetime.now() + timedelta(seconds=1)).isoformat(),
                    "event_type": "keystroke_sequence",
                    "data": {
                        "keystrokes": [
                            {"key": "1", "dwell_time": 95, "pressure": 0.55},
                            {"key": "2", "dwell_time": 105, "pressure": 0.6},
                            {"key": "3", "dwell_time": 88, "pressure": 0.58},
                            {"key": "4", "dwell_time": 110, "pressure": 0.62}
                        ],
                        "typing_rhythm": [85, 92, 78, 88],
                        "inter_key_intervals": [0.12, 0.15, 0.11, 0.13]
                    }
                }
            ]
        }
    
    def create_suspicious_user_data(self, user_id: str) -> Dict[str, Any]:
        """Create suspicious behavioral data"""
        return {
            "user_id": user_id,
            "session_id": f"session_{int(datetime.now().timestamp())}_{user_id}",
            "device_info": {
                "device_type": "desktop",
                "os": "Windows",
                "screen_resolution": "1920x1080",
                "device_model": "Generic PC",
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
            },
            "session_context": {
                "location": "unknown",
                "time_of_day": "night",
                "app_version": "2.0.1",
                "network_type": "vpn"
            },
            "behavioral_logs": [
                {
                    "timestamp": datetime.now().isoformat(),
                    "event_type": "touch_sequence", 
                    "data": {
                        "touch_events": [
                            {"x": 50, "y": 100, "pressure": 0.3, "duration": 50},
                            {"x": 55, "y": 105, "pressure": 0.35, "duration": 45},
                            {"x": 60, "y": 110, "pressure": 0.4, "duration": 55}
                        ],
                        "accelerometer": {"x": 0.8, "y": 1.2, "z": 8.5},
                        "gyroscope": {"x": 0.05, "y": 0.08, "z": 0.12}
                    }
                },
                {
                    "timestamp": (datetime.now() + timedelta(seconds=1)).isoformat(),
                    "event_type": "keystroke_sequence",
                    "data": {
                        "keystrokes": [
                            {"key": "9", "dwell_time": 200, "pressure": 0.9},
                            {"key": "8", "dwell_time": 190, "pressure": 0.85},
                            {"key": "7", "dwell_time": 210, "pressure": 0.95},
                            {"key": "6", "dwell_time": 180, "pressure": 0.8}
                        ],
                        "typing_rhythm": [180, 200, 190, 185],
                        "inter_key_intervals": [0.5, 0.6, 0.55, 0.58]
                    }
                }
            ]
        }
    
    def create_bot_user_data(self, user_id: str) -> Dict[str, Any]:
        """Create bot-like behavioral data (very suspicious)"""
        return {
            "user_id": user_id,
            "session_id": f"session_{int(datetime.now().timestamp())}_{user_id}",
            "device_info": {
                "device_type": "desktop",
                "os": "Linux",
                "screen_resolution": "1024x768",
                "device_model": "Bot Device",
                "user_agent": "AutomatedBot/1.0"
            },
            "session_context": {
                "location": "data_center",
                "time_of_day": "night",
                "app_version": "1.0.0",
                "network_type": "proxy"
            },
            "behavioral_logs": [
                {
                    "timestamp": datetime.now().isoformat(),
                    "event_type": "touch_sequence",
                    "data": {
                        "touch_events": [
                            {"x": 100, "y": 200, "pressure": 1.0, "duration": 10},
                            {"x": 100, "y": 200, "pressure": 1.0, "duration": 10},
                            {"x": 100, "y": 200, "pressure": 1.0, "duration": 10}
                        ],
                        "accelerometer": {"x": 0.0, "y": 0.0, "z": 0.0},
                        "gyroscope": {"x": 0.0, "y": 0.0, "z": 0.0}
                    }
                },
                {
                    "timestamp": (datetime.now() + timedelta(milliseconds=10)).isoformat(),
                    "event_type": "keystroke_sequence",
                    "data": {
                        "keystrokes": [
                            {"key": "1", "dwell_time": 5, "pressure": 1.0},
                            {"key": "2", "dwell_time": 5, "pressure": 1.0},
                            {"key": "3", "dwell_time": 5, "pressure": 1.0},
                            {"key": "4", "dwell_time": 5, "pressure": 1.0}
                        ],
                        "typing_rhythm": [5, 5, 5, 5],
                        "inter_key_intervals": [0.001, 0.001, 0.001, 0.001]
                    }
                }
            ]
        }
    
    def register_user(self, user_id: str, user_type: str) -> bool:
        """Register a new user with the backend"""
        print(f"\n{'='*60}")
        print(f"   REGISTERING USER: {user_id} ({user_type})")
        print(f"{'='*60}")
        
        try:
            registration_data = {
                "phone": f"{''.join([str(i%10) for i in range(10)])}{user_id[-3:]}",
                "password": "testpassword123", 
                "mpin": "1234"
            }
            
            response = requests.post(
                f"{BACKEND_BASE}/api/v1/auth/register",
                json=registration_data,
                timeout=30
            )
            
            print(f"Registration request sent for {user_id}")
            print(f"Response status: {response.status_code}")
            print(f"Response: {response.text}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ User {user_id} registered successfully")
                print(f"Profile created: {result.get('success', False)}")
                return True
            else:
                print(f"❌ Registration failed: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Registration error: {e}")
            return False
    
    def create_learning_sessions(self, user_id: str, user_type: str, num_sessions: int = 3):
        """Create multiple learning sessions for a user"""
        print(f"\n{'='*60}")
        print(f"   CREATING {num_sessions} LEARNING SESSIONS FOR {user_id}")
        print(f"{'='*60}")
        
        sessions_created = 0
        for i in range(num_sessions):
            try:
                # Get appropriate behavioral data
                if user_type == "normal":
                    behavioral_data = self.create_normal_user_data(user_id)
                elif user_type == "suspicious":
                    behavioral_data = self.create_suspicious_user_data(user_id)
                else:  # bot
                    behavioral_data = self.create_bot_user_data(user_id)
                
                print(f"\n--- Learning Session {i+1} for {user_id} ---")
                print(f"Session ID: {behavioral_data['session_id']}")
                
                # Send to ML Engine for processing
                response = requests.post(
                    f"{ML_ENGINE_BASE}/api/v1/behavioral/analyze",
                    json=behavioral_data,
                    timeout=30
                )
                
                print(f"ML Engine response status: {response.status_code}")
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"✅ Session {i+1} processed")
                    print(f"Decision: {result.get('decision', 'Unknown')}")
                    print(f"Confidence: {result.get('confidence', 'Unknown')}")
                    if 'faiss_similarity' in result:
                        print(f"FAISS Similarity: {result['faiss_similarity']:.3f}")
                    sessions_created += 1
                else:
                    print(f"❌ Session {i+1} failed: {response.text}")
                
                # Wait between sessions
                time.sleep(2)
                
            except Exception as e:
                print(f"❌ Error creating session {i+1}: {e}")
        
        print(f"\n✅ Created {sessions_created}/{num_sessions} learning sessions for {user_id}")
        return sessions_created
    
    def test_authentication_attempt(self, user_id: str, user_type: str, attempt_type: str):
        """Test authentication attempt and show detailed scoring"""
        print(f"\n{'='*60}")
        print(f"   AUTHENTICATION TEST: {user_id} ({attempt_type})")
        print(f"{'='*60}")
        
        try:
            # Create behavioral data based on attempt type
            if attempt_type == "same_user_good":
                behavioral_data = self.create_normal_user_data(user_id)
            elif attempt_type == "same_user_different":
                # Same user but slightly different behavior
                behavioral_data = self.create_normal_user_data(user_id)
                # Modify some values to create slight differences
                behavioral_data['session_context']['time_of_day'] = "evening"
                behavioral_data['behavioral_logs'][0]['data']['touch_events'][0]['pressure'] = 0.4
            elif attempt_type == "different_user":
                behavioral_data = self.create_suspicious_user_data(user_id)
            else:  # bot_attack
                behavioral_data = self.create_bot_user_data(user_id)
            
            print(f"\n--- STEP 1: BEHAVIORAL DATA INPUT ---")
            print(f"User ID: {behavioral_data['user_id']}")
            print(f"Device Type: {behavioral_data['device_info']['device_type']}")
            print(f"Location: {behavioral_data['session_context']['location']}")
            print(f"Touch Events: {len(behavioral_data['behavioral_logs'][0]['data']['touch_events'])}")
            print(f"Keystrokes: {len(behavioral_data['behavioral_logs'][1]['data']['keystrokes'])}")
            
            # Send to ML Engine for detailed analysis
            print(f"\n--- STEP 2: ML ENGINE PROCESSING ---")
            response = requests.post(
                f"{ML_ENGINE_BASE}/api/v1/behavioral/analyze",
                json=behavioral_data,
                timeout=30
            )
            
            print(f"ML Engine response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                
                print(f"\n--- STEP 3: FAISS SIMILARITY ANALYSIS ---")
                faiss_sim = result.get('faiss_similarity', 0.0)
                print(f"FAISS Similarity Score: {faiss_sim:.3f}")
                print(f"Similarity Threshold: 0.700")
                print(f"FAISS Decision: {'PASS' if faiss_sim > 0.7 else 'ESCALATE'}")
                
                print(f"\n--- STEP 4: LAYER PROCESSING ---")
                print(f"Layer 1 (FAISS): {result.get('layer_1_decision', 'N/A')}")
                if 'layer_2_decision' in result:
                    print(f"Layer 2 (Adaptive): {result['layer_2_decision']}")
                if 'layer_3_decision' in result:
                    print(f"Layer 3 (GNN): {result['layer_3_decision']}")
                
                print(f"\n--- STEP 5: RISK ANALYSIS ---")
                risk_score = result.get('risk_score', 0.0)
                print(f"Risk Score: {risk_score:.3f}")
                print(f"Risk Level: {result.get('risk_level', 'Unknown')}")
                
                print(f"\n--- STEP 6: FINAL DECISION ---")
                decision = result.get('decision', 'Unknown')
                confidence = result.get('confidence', 0.0)
                print(f"Final Decision: {decision}")
                print(f"Confidence: {confidence:.1f}%")
                
                if decision == "ALLOW":
                    print("✅ AUTHENTICATION SUCCESSFUL")
                elif decision == "CHALLENGE":
                    print("⚠️ ADDITIONAL VERIFICATION REQUIRED")
                elif decision == "BLOCK":
                    print("❌ AUTHENTICATION BLOCKED")
                else:
                    print("🔄 AUTHENTICATION UNDER REVIEW")
                
                return result
                
            else:
                print(f"❌ ML Engine error: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Authentication test error: {e}")
            return None
    
    def run_complete_system_test(self):
        """Run complete end-to-end system test"""
        print("🚀 LIVE BEHAVIORAL AUTHENTICATION SYSTEM TEST")
        print("=" * 80)
        print("Testing real ML Engine + Backend integration")
        print("=" * 80)
        
        # Create test users
        test_users = [
            ("normal_user_001", "normal"),
            ("suspicious_user_002", "suspicious"), 
            ("bot_user_003", "bot")
        ]
        
        # Phase 1: User Registration
        print(f"\n🔹 PHASE 1: USER REGISTRATION")
        print("-" * 40)
        registered_users = []
        
        for user_id, user_type in test_users:
            if self.register_user(user_id, user_type):
                registered_users.append((user_id, user_type))
        
        print(f"\n✅ Registered {len(registered_users)} users successfully")
        
        # Phase 2: Learning Phase (Create baseline behavioral profiles)
        print(f"\n🔹 PHASE 2: LEARNING PHASE (Creating Behavioral Baselines)")
        print("-" * 60)
        
        for user_id, user_type in registered_users:
            sessions_created = self.create_learning_sessions(user_id, user_type, 3)
            print(f"User {user_id}: {sessions_created} learning sessions created")
        
        print(f"\n✅ Learning phase complete - User profiles established")
        
        # Wait for data to settle
        print(f"\n⏳ Waiting 5 seconds for data processing...")
        time.sleep(5)
        
        # Phase 3: Authentication Testing
        print(f"\n🔹 PHASE 3: AUTHENTICATION TESTING")
        print("-" * 40)
        
        test_scenarios = [
            ("normal_user_001", "normal", "same_user_good", "Normal user, good behavior"),
            ("normal_user_001", "normal", "same_user_different", "Normal user, slight changes"),
            ("suspicious_user_002", "suspicious", "different_user", "Different user attempting access"),
            ("bot_user_003", "bot", "bot_attack", "Bot attack attempt"),
        ]
        
        results = []
        
        for user_id, user_type, attempt_type, description in test_scenarios:
            print(f"\n🧪 TEST SCENARIO: {description}")
            print("-" * 50)
            result = self.test_authentication_attempt(user_id, user_type, attempt_type)
            if result:
                results.append({
                    "user_id": user_id,
                    "attempt_type": attempt_type,
                    "description": description,
                    "decision": result.get('decision'),
                    "confidence": result.get('confidence'),
                    "faiss_similarity": result.get('faiss_similarity'),
                    "risk_score": result.get('risk_score')
                })
            
            # Wait between tests
            time.sleep(3)
        
        # Phase 4: Results Summary
        print(f"\n🔹 PHASE 4: RESULTS SUMMARY")
        print("=" * 80)
        
        for i, result in enumerate(results, 1):
            print(f"\nTest {i}: {result['description']}")
            print(f"  User: {result['user_id']}")
            print(f"  FAISS Similarity: {result.get('faiss_similarity', 0):.3f}")
            print(f"  Risk Score: {result.get('risk_score', 0):.3f}")
            print(f"  Decision: {result.get('decision', 'Unknown')}")
            print(f"  Confidence: {result.get('confidence', 0):.1f}%")
            
            if result.get('decision') == "ALLOW":
                print(f"  Result: ✅ AUTHENTICATED")
            elif result.get('decision') == "CHALLENGE":
                print(f"  Result: ⚠️ NEEDS VERIFICATION")
            elif result.get('decision') == "BLOCK":
                print(f"  Result: ❌ BLOCKED")
            else:
                print(f"  Result: 🔄 UNDER REVIEW")
        
        print(f"\n✅ COMPLETE SYSTEM TEST FINISHED")
        print(f"📊 Processed {len(results)} authentication scenarios")
        print(f"🎯 System demonstrating real FAISS scoring, escalation, and blocking")
        
        return results

def main():
    """Run the live system test"""
    tester = LiveSystemTester()
    
    # Check if services are running
    try:
        ml_health = requests.get(f"{ML_ENGINE_BASE}/health", timeout=5)
        backend_health = requests.get(f"{BACKEND_BASE}/health", timeout=5)
        
        if ml_health.status_code != 200:
            print("❌ ML Engine not running on port 8001")
            return False
            
        if backend_health.status_code != 200:
            print("❌ Backend not running on port 8000")
            return False
            
        print("✅ Both ML Engine and Backend are running")
        
    except Exception as e:
        print(f"❌ Service check failed: {e}")
        print("Make sure both ML Engine (8001) and Backend (8000) are running")
        return False
    
    # Run the complete test
    results = tester.run_complete_system_test()
    return len(results) > 0

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 LIVE SYSTEM TEST COMPLETED SUCCESSFULLY!")
    else:
        print("\n❌ LIVE SYSTEM TEST FAILED!")

"""
Live Behavioral Analysis Test
Focus on ML Engine behavioral analysis with real FAISS scoring
"""
import requests
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List

# System endpoints
ML_ENGINE_BASE = "http://localhost:8001"

class BehavioralAnalysisTester:
    """Test behavioral analysis with real FAISS scoring"""
    
    def create_normal_user_data(self, user_id: str) -> Dict[str, Any]:
        """Create normal behavioral data"""
        return {
            "user_id": user_id,
            "session_id": f"session_{int(datetime.now().timestamp())}_{user_id}",
            "logs": [
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
            "logs": [
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
        """Create bot-like behavioral data"""
        return {
            "user_id": user_id,
            "session_id": f"session_{int(datetime.now().timestamp())}_{user_id}",
            "logs": [
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
    
    def test_behavioral_analysis(self, user_id: str, data_type: str, behavioral_data: Dict[str, Any]):
        """Test behavioral analysis and show detailed results"""
        print(f"\n{'='*80}")
        print(f"   BEHAVIORAL ANALYSIS TEST: {user_id} ({data_type})")
        print(f"{'='*80}")
        
        print(f"\n--- STEP 1: INPUT BEHAVIORAL DATA ---")
        print(f"User ID: {behavioral_data['user_id']}")
        print(f"Session ID: {behavioral_data['session_id']}")
        print(f"Behavioral Logs: {len(behavioral_data['logs'])}")
        print(f"Touch Events: {len(behavioral_data['logs'][0]['data']['touch_events'])}")
        print(f"Keystrokes: {len(behavioral_data['logs'][1]['data']['keystrokes'])}")
        
        try:
            print(f"\n--- STEP 2: SENDING TO ML ENGINE ---")
            print(f"Endpoint: {ML_ENGINE_BASE}/analyze-mobile")
            
            response = requests.post(
                f"{ML_ENGINE_BASE}/analyze-mobile",
                json=behavioral_data,
                timeout=30
            )
            
            print(f"Response Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                
                print(f"\n--- STEP 3: PREPROCESSING RESULTS ---")
                if 'preprocessing_info' in result:
                    preprocessing = result['preprocessing_info']
                    print(f"Vector Dimension: {preprocessing.get('vector_dimension', 'N/A')}")
                    print(f"Feature Count: {preprocessing.get('feature_count', 'N/A')}")
                    print(f"Processing Time: {preprocessing.get('processing_time_ms', 'N/A')} ms")
                
                print(f"\n--- STEP 4: FAISS SIMILARITY ANALYSIS ---")
                faiss_sim = result.get('faiss_similarity', 0.0)
                faiss_distance = result.get('faiss_distance', 1.0)
                threshold = result.get('similarity_threshold', 0.7)
                
                print(f"FAISS Similarity Score: {faiss_sim:.4f}")
                print(f"FAISS Distance: {faiss_distance:.4f}")
                print(f"Similarity Threshold: {threshold:.3f}")
                print(f"FAISS Decision: {'PASS (High Similarity)' if faiss_sim > threshold else 'ESCALATE (Low Similarity)'}")
                
                print(f"\n--- STEP 5: LAYER PROCESSING ---")
                layer_info = result.get('layer_processing', {})
                print(f"Layer 1 (FAISS): {layer_info.get('layer_1_decision', result.get('layer_1_decision', 'N/A'))}")
                
                if 'layer_2_decision' in result or 'layer_2_decision' in layer_info:
                    print(f"Layer 2 (Adaptive): {layer_info.get('layer_2_decision', result.get('layer_2_decision', 'N/A'))}")
                
                if 'layer_3_decision' in result or 'layer_3_decision' in layer_info:
                    print(f"Layer 3 (GNN): {layer_info.get('layer_3_decision', result.get('layer_3_decision', 'N/A'))}")
                
                print(f"\n--- STEP 6: RISK ASSESSMENT ---")
                risk_score = result.get('risk_score', 0.0)
                risk_level = result.get('risk_level', 'Unknown')
                risk_factors = result.get('risk_factors', [])
                
                print(f"Risk Score: {risk_score:.4f}")
                print(f"Risk Level: {risk_level}")
                if risk_factors:
                    print(f"Risk Factors: {', '.join(risk_factors)}")
                
                print(f"\n--- STEP 7: FINAL DECISION ---")
                decision = result.get('decision', 'Unknown')
                confidence = result.get('confidence', 0.0)
                
                print(f"Final Decision: {decision}")
                print(f"Confidence: {confidence:.2f}%")
                
                if decision == "ALLOW":
                    print("✅ AUTHENTICATION APPROVED - User behavior matches profile")
                elif decision == "CHALLENGE":
                    print("⚠️ ADDITIONAL VERIFICATION REQUIRED - Moderate risk detected")
                elif decision == "BLOCK" or decision == "REJECT":
                    print("❌ AUTHENTICATION BLOCKED - High risk/Bot behavior detected")
                else:
                    print("🔄 AUTHENTICATION UNDER REVIEW - Analyzing...")
                
                return result
                
            else:
                print(f"❌ ML Engine Error:")
                print(f"Status: {response.status_code}")
                print(f"Response: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Test Error: {e}")
            return None
    
    def create_learning_profile(self, user_id: str, data_type: str, num_sessions: int = 3):
        """Create learning sessions to build user profile"""
        print(f"\n{'='*80}")
        print(f"   CREATING LEARNING PROFILE: {user_id} ({num_sessions} sessions)")
        print(f"{'='*80}")
        
        sessions_created = 0
        
        for i in range(num_sessions):
            print(f"\n--- Learning Session {i+1}/{num_sessions} ---")
            
            # Create behavioral data based on type
            if data_type == "normal":
                behavioral_data = self.create_normal_user_data(user_id)
            elif data_type == "suspicious":
                behavioral_data = self.create_suspicious_user_data(user_id)
            else:
                behavioral_data = self.create_bot_user_data(user_id)
            
            try:
                response = requests.post(
                    f"{ML_ENGINE_BASE}/analyze-mobile",
                    json=behavioral_data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"✅ Session {i+1} processed")
                    print(f"   Decision: {result.get('decision', 'Unknown')}")
                    print(f"   FAISS Similarity: {result.get('faiss_similarity', 0.0):.3f}")
                    sessions_created += 1
                else:
                    print(f"❌ Session {i+1} failed: {response.status_code}")
                
            except Exception as e:
                print(f"❌ Session {i+1} error: {e}")
            
            time.sleep(1)  # Brief pause between sessions
        
        print(f"\n✅ Created {sessions_created}/{num_sessions} learning sessions")
        return sessions_created
    
    def run_comprehensive_test(self):
        """Run comprehensive behavioral analysis test"""
        print("🚀 LIVE BEHAVIORAL AUTHENTICATION ANALYSIS")
        print("=" * 80)
        print("Testing ML Engine with Real FAISS Scoring & Layer Processing")
        print("=" * 80)
        
        # Test users with different behavioral patterns
        test_users = [
            ("user_normal_001", "normal"),
            ("user_suspicious_002", "suspicious"),
            ("user_bot_003", "bot")
        ]
        
        # Phase 1: Create Learning Profiles
        print(f"\n🔹 PHASE 1: LEARNING PROFILE CREATION")
        print("-" * 50)
        
        for user_id, data_type in test_users:
            sessions = self.create_learning_profile(user_id, data_type, 3)
            print(f"User {user_id}: {sessions} sessions created")
        
        print(f"\n⏳ Waiting 3 seconds for profile establishment...")
        time.sleep(3)
        
        # Phase 2: Authentication Tests
        print(f"\n🔹 PHASE 2: AUTHENTICATION SCENARIO TESTING")
        print("-" * 50)
        
        test_scenarios = [
            ("user_normal_001", "normal", "Normal user with good behavior"),
            ("user_normal_001", "suspicious", "Normal user with suspicious behavior"),
            ("user_suspicious_002", "suspicious", "Suspicious user maintaining pattern"), 
            ("user_suspicious_002", "normal", "Suspicious user trying normal behavior"),
            ("user_bot_003", "bot", "Bot user with automated behavior"),
            ("user_bot_003", "normal", "Bot trying to mimic normal behavior")
        ]
        
        results = []
        
        for user_id, behavior_type, description in test_scenarios:
            print(f"\n🧪 TEST SCENARIO: {description}")
            print("-" * 60)
            
            # Create appropriate behavioral data
            if behavior_type == "normal":
                behavioral_data = self.create_normal_user_data(user_id)
            elif behavior_type == "suspicious":
                behavioral_data = self.create_suspicious_user_data(user_id)
            else:
                behavioral_data = self.create_bot_user_data(user_id)
            
            result = self.test_behavioral_analysis(user_id, behavior_type, behavioral_data)
            
            if result:
                results.append({
                    "user_id": user_id,
                    "behavior_type": behavior_type,
                    "description": description,
                    "faiss_similarity": result.get('faiss_similarity', 0.0),
                    "risk_score": result.get('risk_score', 0.0),
                    "decision": result.get('decision', 'Unknown'),
                    "confidence": result.get('confidence', 0.0)
                })
            
            time.sleep(2)  # Pause between tests
        
        # Phase 3: Results Analysis
        print(f"\n🔹 PHASE 3: RESULTS ANALYSIS")
        print("=" * 80)
        
        for i, result in enumerate(results, 1):
            print(f"\nTest {i}: {result['description']}")
            print(f"  User ID: {result['user_id']}")
            print(f"  Behavior Type: {result['behavior_type']}")
            print(f"  FAISS Similarity: {result['faiss_similarity']:.4f}")
            print(f"  Risk Score: {result['risk_score']:.4f}")
            print(f"  Decision: {result['decision']}")
            print(f"  Confidence: {result['confidence']:.2f}%")
            
            if result['decision'] in ['ALLOW', 'APPROVE']:
                status = "✅ APPROVED"
            elif result['decision'] == 'CHALLENGE':
                status = "⚠️ CHALLENGE"
            elif result['decision'] in ['BLOCK', 'REJECT']:
                status = "❌ BLOCKED"
            else:
                status = "🔄 REVIEW"
            
            print(f"  Result: {status}")
        
        # Summary Statistics
        print(f"\n📊 SUMMARY STATISTICS")
        print("-" * 40)
        print(f"Total Tests: {len(results)}")
        
        if results:
            avg_faiss = sum(r['faiss_similarity'] for r in results) / len(results)
            avg_risk = sum(r['risk_score'] for r in results) / len(results)
            
            decisions = [r['decision'] for r in results]
            approved = sum(1 for d in decisions if d in ['ALLOW', 'APPROVE'])
            challenged = sum(1 for d in decisions if d == 'CHALLENGE')
            blocked = sum(1 for d in decisions if d in ['BLOCK', 'REJECT'])
            
            print(f"Average FAISS Similarity: {avg_faiss:.4f}")
            print(f"Average Risk Score: {avg_risk:.4f}")
            print(f"Approved: {approved}")
            print(f"Challenged: {challenged}")  
            print(f"Blocked: {blocked}")
        
        print(f"\n✅ COMPREHENSIVE BEHAVIORAL ANALYSIS COMPLETE")
        print(f"🎯 Real FAISS scoring, layer processing, and decision making demonstrated")
        
        return results

def main():
    """Run the behavioral analysis test"""
    # Check ML Engine health
    try:
        health_response = requests.get(f"{ML_ENGINE_BASE}/health", timeout=5)
        if health_response.status_code != 200:
            print("❌ ML Engine not running on port 8001")
            return False
        print("✅ ML Engine is running and healthy")
    except Exception as e:
        print(f"❌ ML Engine health check failed: {e}")
        return False
    
    # Run the test
    tester = BehavioralAnalysisTester()
    results = tester.run_comprehensive_test()
    
    return len(results) > 0

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 BEHAVIORAL ANALYSIS TEST COMPLETED!")
    else:
        print("\n❌ BEHAVIORAL ANALYSIS TEST FAILED!")

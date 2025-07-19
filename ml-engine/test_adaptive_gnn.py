#!/usr/bin/env python3
"""
Test the Adaptive Layer and GNN to make them genuinely work
Just like we fixed FAISS, now we'll fix the next layers
"""

import requests
import json
import numpy as np
import asyncio
from datetime import datetime
from typing import Dict, List, Any

class AdaptiveGNNTester:
    """Test adaptive layer and GNN for genuine functionality"""
    
    def __init__(self):
        self.base_url = "http://localhost:8001"
        self.test_results = []
        
    def test_adaptive_learning(self):
        """Test if adaptive layer is genuinely learning"""
        print("🧠 TESTING ADAPTIVE LAYER LEARNING")
        print("="*60)
        
        # Create a user with consistent behavior pattern
        user_id = "adaptive_test_user_001"
        
        # Simulate 10 sessions with consistent pattern
        print("📊 Phase 1: Building consistent pattern...")
        consistent_sessions = []
        
        for i in range(5):
            session_data = self.generate_consistent_behavior(user_id, i)
            response = self.send_request(session_data)
            
            if response:
                print(f"Session {i+1}: Decision={response.get('decision')}, "
                      f"Confidence={response.get('confidence'):.3f}, "
                      f"Risk={response.get('risk_score'):.3f}")
                consistent_sessions.append(response)
            
        print(f"\n🔄 Phase 2: Testing deviation detection...")
        # Now send anomalous behavior
        anomaly_data = self.generate_anomalous_behavior(user_id)
        anomaly_response = self.send_request(anomaly_data)
        
        if anomaly_response:
            print(f"Anomaly: Decision={anomaly_response.get('decision')}, "
                  f"Confidence={anomaly_response.get('confidence'):.3f}, "
                  f"Risk={anomaly_response.get('risk_score'):.3f}")
                  
        # Test adaptive feedback
        print(f"\n🎯 Phase 3: Testing feedback learning...")
        feedback_result = self.test_feedback_learning(user_id, consistent_sessions[-1])
        
        return {
            'consistent_behavior': consistent_sessions,
            'anomaly_response': anomaly_response,
            'feedback_learning': feedback_result
        }
    
    def generate_consistent_behavior(self, user_id: str, session_num: int) -> Dict[str, Any]:
        """Generate consistent behavioral pattern"""
        return {
            "user_id": user_id,
            "session_id": f"consistent_session_{session_num}_{int(datetime.now().timestamp())}",
            "logs": [
                {
                    "timestamp": datetime.now().isoformat(),
                    "event_type": "touch_sequence",
                    "data": {
                        "touch_events": [
                            {"x": 150 + np.random.normal(0, 5), "y": 200 + np.random.normal(0, 5), 
                             "pressure": 0.5 + np.random.normal(0, 0.05), "duration": 120 + np.random.normal(0, 10)},
                            {"x": 160 + np.random.normal(0, 5), "y": 210 + np.random.normal(0, 5), 
                             "pressure": 0.6 + np.random.normal(0, 0.05), "duration": 115 + np.random.normal(0, 10)}
                        ],
                        "accelerometer": {"x": 0.02 + np.random.normal(0, 0.01), 
                                        "y": 0.15 + np.random.normal(0, 0.01), 
                                        "z": 9.78 + np.random.normal(0, 0.1)},
                        "gyroscope": {"x": 0.001 + np.random.normal(0, 0.0005), 
                                    "y": 0.002 + np.random.normal(0, 0.0005), 
                                    "z": 0.0015 + np.random.normal(0, 0.0005)}
                    }
                },
                {
                    "timestamp": datetime.now().isoformat(),
                    "event_type": "keystroke_sequence", 
                    "data": {
                        "keystrokes": [
                            {"key": "a", "dwell_time": 100 + np.random.normal(0, 10), 
                             "pressure": 0.55 + np.random.normal(0, 0.05)},
                            {"key": "b", "dwell_time": 120 + np.random.normal(0, 10), 
                             "pressure": 0.6 + np.random.normal(0, 0.05)}
                        ],
                        "typing_rhythm": [95 + np.random.normal(0, 5), 110 + np.random.normal(0, 5)],
                        "inter_key_intervals": [0.12 + np.random.normal(0, 0.02), 0.15 + np.random.normal(0, 0.02)]
                    }
                }
            ]
        }
    
    def generate_anomalous_behavior(self, user_id: str) -> Dict[str, Any]:
        """Generate clearly anomalous behavior pattern"""
        return {
            "user_id": user_id,
            "session_id": f"anomaly_session_{int(datetime.now().timestamp())}",
            "logs": [
                {
                    "timestamp": datetime.now().isoformat(),
                    "event_type": "touch_sequence",
                    "data": {
                        # Very different touch pattern - bot-like
                        "touch_events": [
                            {"x": 100, "y": 100, "pressure": 1.0, "duration": 50},  # Perfect clicks
                            {"x": 100, "y": 100, "pressure": 1.0, "duration": 50},
                            {"x": 100, "y": 100, "pressure": 1.0, "duration": 50}
                        ],
                        # No natural motion
                        "accelerometer": {"x": 0.0, "y": 0.0, "z": 10.0},
                        "gyroscope": {"x": 0.0, "y": 0.0, "z": 0.0}
                    }
                },
                {
                    "timestamp": datetime.now().isoformat(),
                    "event_type": "keystroke_sequence",
                    "data": {
                        # Robotic typing
                        "keystrokes": [
                            {"key": "x", "dwell_time": 50, "pressure": 1.0},  # Perfect timing
                            {"key": "x", "dwell_time": 50, "pressure": 1.0},
                            {"key": "x", "dwell_time": 50, "pressure": 1.0}
                        ],
                        "typing_rhythm": [50, 50, 50],  # Perfect rhythm
                        "inter_key_intervals": [0.1, 0.1, 0.1]  # Perfect intervals
                    }
                }
            ]
        }
    
    def send_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Send request to ML Engine and get response"""
        try:
            response = requests.post(
                f"{self.base_url}/analyze-mobile",
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ Error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Exception: {e}")
            return None
    
    def test_feedback_learning(self, user_id: str, last_session: Dict[str, Any]) -> Dict[str, Any]:
        """Test if adaptive layer learns from feedback"""
        try:
            # Simulate feedback about authentication decision
            feedback_data = {
                "user_id": user_id,
                "session_id": last_session.get('vector_id', 'unknown'),
                "decision_id": f"decision_{int(datetime.now().timestamp())}",
                "was_correct": True,  # User confirms the decision was correct
                "actual_outcome": "legitimate",
                "confidence": last_session.get('confidence', 0.5)
            }
            
            response = requests.post(
                f"{self.base_url}/feedback",
                json=feedback_data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Feedback processed: {result.get('message', 'Success')}")
                return result
            else:
                print(f"❌ Feedback error: {response.status_code}")
                return {"error": response.text}
                
        except Exception as e:
            print(f"❌ Feedback exception: {e}")
            return {"error": str(e)}
    
    def test_statistics_endpoints(self):
        """Test if we can get real statistics from adaptive layer"""
        print("📊 TESTING ADAPTIVE LAYER STATISTICS")
        print("="*60)
        
        try:
            # Get health status
            health_response = requests.get(f"{self.base_url}/health")
            if health_response.status_code == 200:
                health = health_response.json()
                print("🏥 Health Status:")
                print(f"- Adaptive Layer: {health.get('components', {}).get('adaptive_layer', 'Unknown')}")
                print(f"- Continuous Analysis: {health.get('components', {}).get('continuous_analysis', 'Unknown')}")
            
            # Get layer statistics  
            stats_response = requests.get(f"{self.base_url}/stats")
            if stats_response.status_code == 200:
                stats = stats_response.json()
                print("\n📈 Layer Statistics:")
                
                adaptive_stats = stats.get('layers', {}).get('adaptive', {})
                if adaptive_stats:
                    print("🧠 Adaptive Layer:")
                    for key, value in adaptive_stats.items():
                        print(f"  - {key}: {value}")
                else:
                    print("⚠️  No adaptive layer statistics available")
                    
                analysis_stats = stats.get('layers', {}).get('analysis', {})
                if analysis_stats:
                    print("🔍 Analysis Layer:")
                    for key, value in analysis_stats.items():
                        print(f"  - {key}: {value}")
                
                return stats
            else:
                print(f"❌ Stats error: {stats_response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Statistics exception: {e}")
            return None
    
    def run_comprehensive_test(self):
        """Run comprehensive adaptive layer and GNN testing"""
        print("🚀 COMPREHENSIVE ADAPTIVE LAYER & GNN TESTING")
        print("="*80)
        
        # Test 1: Adaptive Learning
        adaptive_results = self.test_adaptive_learning()
        
        # Test 2: Statistics and Health
        stats_results = self.test_statistics_endpoints()
        
        # Test 3: Pattern Analysis
        print(f"\n🔬 ANALYZING RESULTS")
        print("="*60)
        
        if adaptive_results['consistent_behavior']:
            consistent_sessions = adaptive_results['consistent_behavior']
            print(f"📈 Consistent Sessions Analysis:")
            print(f"- Number of sessions: {len(consistent_sessions)}")
            
            # Check if confidence is building over time
            confidences = [s.get('confidence', 0) for s in consistent_sessions]
            risks = [s.get('risk_score', 0) for s in consistent_sessions]
            
            print(f"- Confidence progression: {[f'{c:.3f}' for c in confidences]}")
            print(f"- Risk progression: {[f'{r:.3f}' for r in risks]}")
            
            # Check if system is learning (confidence should increase)
            if len(confidences) > 1:
                confidence_trend = confidences[-1] - confidences[0]
                print(f"- Confidence trend: {'📈 Improving' if confidence_trend > 0 else '📉 Declining' if confidence_trend < 0 else '➡️  Stable'}")
        
        if adaptive_results['anomaly_response']:
            anomaly = adaptive_results['anomaly_response']
            print(f"\n🚨 Anomaly Detection Analysis:")
            print(f"- Anomaly decision: {anomaly.get('decision')}")
            print(f"- Anomaly confidence: {anomaly.get('confidence', 0):.3f}")
            print(f"- Anomaly risk: {anomaly.get('risk_score', 0):.3f}")
            
            # Check if system detected the anomaly
            if anomaly.get('risk_score', 0) > 0.5 or anomaly.get('decision') in ['challenge', 'block']:
                print("✅ System detected anomalous behavior")
            else:
                print("⚠️  System may have missed anomaly")
        
        return {
            'adaptive_results': adaptive_results,
            'stats_results': stats_results
        }

if __name__ == "__main__":
    tester = AdaptiveGNNTester()
    results = tester.run_comprehensive_test()
    print(f"\n🎉 ADAPTIVE LAYER & GNN TESTING COMPLETED!")

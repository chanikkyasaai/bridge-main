#!/usr/bin/env python3
"""
Fix and test Adaptive Layer + GNN integration for genuine anomaly detection
"""

import requests
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Any

def test_fixed_endpoints():
    """Test the endpoints with correct format"""
    print("🔧 TESTING FIXED ENDPOINTS")
    print("="*60)
    
    base_url = "http://localhost:8001"
    
    # Test correct stats endpoint
    try:
        print("📊 Testing stats endpoint...")
        stats_response = requests.get(f"{base_url}/api/v1/system/stats")
        print(f"Stats Response Status: {stats_response.status_code}")
        
        if stats_response.status_code == 200:
            stats = stats_response.json()
            print("✅ Stats endpoint working")
            print(json.dumps(stats, indent=2))
        else:
            print(f"❌ Stats error: {stats_response.text}")
    except Exception as e:
        print(f"❌ Stats exception: {e}")
    
    # Test health endpoint
    try:
        print(f"\n🏥 Testing health endpoint...")
        health_response = requests.get(f"{base_url}/health")
        print(f"Health Response Status: {health_response.status_code}")
        
        if health_response.status_code == 200:
            health = health_response.json()
            print("✅ Health endpoint working")
            print(json.dumps(health, indent=2))
        else:
            print(f"❌ Health error: {health_response.text}")
    except Exception as e:
        print(f"❌ Health exception: {e}")
    
    # Test feedback with correct format
    try:
        print(f"\n💬 Testing feedback endpoint...")
        feedback_data = {
            "user_id": "test_user_feedback",
            "session_id": "test_session_feedback",
            "decision_id": f"decision_{int(datetime.now().timestamp())}",
            "was_correct": True,
            "feedback_source": "manual_test"
        }
        
        feedback_response = requests.post(
            f"{base_url}/feedback",
            json=feedback_data
        )
        print(f"Feedback Response Status: {feedback_response.status_code}")
        
        if feedback_response.status_code == 200:
            result = feedback_response.json()
            print("✅ Feedback endpoint working")
            print(json.dumps(result, indent=2))
        else:
            print(f"❌ Feedback error: {feedback_response.text}")
    except Exception as e:
        print(f"❌ Feedback exception: {e}")

def test_anomaly_detection_detailed():
    """Test anomaly detection with detailed analysis"""
    print("🚨 TESTING DETAILED ANOMALY DETECTION")
    print("="*80)
    
    base_url = "http://localhost:8001"
    user_id = "anomaly_detection_test"
    
    # Step 1: Build normal profile
    print("📊 Step 1: Building normal user profile...")
    normal_sessions = []
    
    for i in range(3):
        # Consistent human-like behavior
        normal_data = {
            "user_id": user_id,
            "session_id": f"normal_{i}_{int(datetime.now().timestamp())}",
            "logs": [
                {
                    "timestamp": datetime.now().isoformat(),
                    "event_type": "touch_sequence",
                    "data": {
                        "touch_events": [
                            {
                                "x": 150 + np.random.normal(0, 15),  # Natural variation
                                "y": 200 + np.random.normal(0, 15), 
                                "pressure": 0.5 + np.random.normal(0, 0.1), 
                                "duration": 120 + np.random.normal(0, 20)
                            },
                            {
                                "x": 160 + np.random.normal(0, 15), 
                                "y": 210 + np.random.normal(0, 15), 
                                "pressure": 0.6 + np.random.normal(0, 0.1), 
                                "duration": 115 + np.random.normal(0, 20)
                            }
                        ],
                        "accelerometer": {
                            "x": 0.02 + np.random.normal(0, 0.02), 
                            "y": 0.15 + np.random.normal(0, 0.05), 
                            "z": 9.78 + np.random.normal(0, 0.3)
                        },
                        "gyroscope": {
                            "x": 0.001 + np.random.normal(0, 0.002), 
                            "y": 0.002 + np.random.normal(0, 0.002), 
                            "z": 0.0015 + np.random.normal(0, 0.002)
                        }
                    }
                }
            ]
        }
        
        response = requests.post(f"{base_url}/analyze-mobile", json=normal_data)
        if response.status_code == 200:
            result = response.json()
            normal_sessions.append(result)
            print(f"Normal {i+1}: Risk={result.get('risk_score', 0):.4f}, "
                  f"Confidence={result.get('confidence', 0):.3f}, "
                  f"Decision={result.get('decision')}")
    
    # Step 2: Test different types of anomalies
    anomaly_tests = [
        {
            "name": "Perfect Bot Behavior",
            "data": {
                "user_id": user_id,
                "session_id": f"bot_test_{int(datetime.now().timestamp())}",
                "logs": [{
                    "timestamp": datetime.now().isoformat(),
                    "event_type": "touch_sequence", 
                    "data": {
                        "touch_events": [
                            {"x": 100, "y": 100, "pressure": 1.0, "duration": 50},
                            {"x": 100, "y": 100, "pressure": 1.0, "duration": 50},
                            {"x": 100, "y": 100, "pressure": 1.0, "duration": 50}
                        ],
                        "accelerometer": {"x": 0.0, "y": 0.0, "z": 10.0},
                        "gyroscope": {"x": 0.0, "y": 0.0, "z": 0.0}
                    }
                }]
            }
        },
        {
            "name": "Extreme Speed Typing",
            "data": {
                "user_id": user_id,
                "session_id": f"speed_test_{int(datetime.now().timestamp())}",
                "logs": [{
                    "timestamp": datetime.now().isoformat(),
                    "event_type": "keystroke_sequence",
                    "data": {
                        "keystrokes": [
                            {"key": "a", "dwell_time": 10, "pressure": 1.0},  # Impossibly fast
                            {"key": "b", "dwell_time": 10, "pressure": 1.0},
                            {"key": "c", "dwell_time": 10, "pressure": 1.0}
                        ],
                        "typing_rhythm": [10, 10, 10],
                        "inter_key_intervals": [0.01, 0.01, 0.01]  # Superhuman speed
                    }
                }]
            }
        },
        {
            "name": "Impossible Device Motion",
            "data": {
                "user_id": user_id,
                "session_id": f"motion_test_{int(datetime.now().timestamp())}",
                "logs": [{
                    "timestamp": datetime.now().isoformat(),
                    "event_type": "touch_sequence",
                    "data": {
                        "touch_events": [
                            {"x": 150, "y": 200, "pressure": 0.5, "duration": 120}
                        ],
                        "accelerometer": {"x": 50.0, "y": 50.0, "z": 50.0},  # Impossible values
                        "gyroscope": {"x": 100.0, "y": 100.0, "z": 100.0}   # Device spinning wildly
                    }
                }]
            }
        }
    ]
    
    print(f"\n🔍 Step 2: Testing anomaly detection...")
    anomaly_results = []
    
    for test_case in anomaly_tests:
        print(f"\n--- Testing {test_case['name']} ---")
        response = requests.post(f"{base_url}/analyze-mobile", json=test_case['data'])
        
        if response.status_code == 200:
            result = response.json()
            anomaly_results.append(result)
            
            risk_score = result.get('risk_score', 0)
            confidence = result.get('confidence', 0)
            decision = result.get('decision', 'unknown')
            similarity = result.get('similarity_score', 0)
            
            print(f"Risk Score: {risk_score:.4f}")
            print(f"Confidence: {confidence:.3f}")
            print(f"Decision: {decision}")
            print(f"Similarity: {similarity:.4f}")
            print(f"Risk Factors: {result.get('risk_factors', [])}")
            
            # Check if anomaly was detected
            if risk_score > 0.3 or decision in ['challenge', 'block']:
                print("✅ ANOMALY DETECTED")
            else:
                print("❌ ANOMALY MISSED")
        else:
            print(f"❌ Request failed: {response.status_code}")
    
    # Step 3: Analysis
    print(f"\n🔬 ANALYSIS SUMMARY")
    print("="*60)
    
    if normal_sessions:
        normal_risks = [s.get('risk_score', 0) for s in normal_sessions]
        print(f"Normal behavior risks: {[f'{r:.4f}' for r in normal_risks]}")
        avg_normal_risk = np.mean(normal_risks)
        print(f"Average normal risk: {avg_normal_risk:.4f}")
    
    if anomaly_results:
        anomaly_risks = [s.get('risk_score', 0) for s in anomaly_results]
        print(f"Anomaly behavior risks: {[f'{r:.4f}' for r in anomaly_risks]}")
        avg_anomaly_risk = np.mean(anomaly_risks)
        print(f"Average anomaly risk: {avg_anomaly_risk:.4f}")
        
        # Check discrimination ability
        if avg_anomaly_risk > avg_normal_risk * 2:
            print("✅ System shows good discrimination between normal and anomalous behavior")
        else:
            print("⚠️  System may not be discriminating well between normal and anomalous behavior")
    
    return {
        'normal_sessions': normal_sessions,
        'anomaly_results': anomaly_results,
        'discrimination_ratio': avg_anomaly_risk / avg_normal_risk if normal_sessions and anomaly_results else 0
    }

if __name__ == "__main__":
    # Test fixed endpoints
    test_fixed_endpoints()
    
    print("\n" + "="*100 + "\n")
    
    # Test anomaly detection in detail
    results = test_anomaly_detection_detailed()
    
    print(f"\n🎯 FINAL ASSESSMENT:")
    print("="*60)
    
    discrimination = results.get('discrimination_ratio', 0)
    if discrimination > 2:
        print(f"✅ Excellent anomaly detection (ratio: {discrimination:.2f})")
    elif discrimination > 1.5:
        print(f"⚡ Good anomaly detection (ratio: {discrimination:.2f})")
    elif discrimination > 1.0:
        print(f"⚠️  Fair anomaly detection (ratio: {discrimination:.2f})")
    else:
        print(f"❌ Poor anomaly detection (ratio: {discrimination:.2f})")
    
    print(f"\n🎉 ADAPTIVE LAYER & ANOMALY TESTING COMPLETED!")

#!/usr/bin/env python3
"""
Force users into different learning phases and test real anomaly detection
"""

import requests
import json
import numpy as np
from datetime import datetime
from typing import Dict, Any

def force_user_phase_and_test():
    """Force users into different phases and test anomaly detection"""
    print("⚡ FORCING USERS INTO ADVANCED PHASES FOR REAL TESTING")
    print("="*80)
    
    base_url = "http://localhost:8001"
    
    # Test different user phases
    phase_tests = [
        {
            "phase_name": "Learning Phase (Current)",
            "user_id": "learning_user",
            "expected_learning": True,
            "sessions_to_send": 3
        },
        {
            "phase_name": "Gradual Risk Phase (Forced)",
            "user_id": "gradual_user", 
            "expected_learning": False,
            "sessions_to_send": 8  # Should transition to gradual risk
        },
        {
            "phase_name": "Full Auth Phase (Forced)",
            "user_id": "full_auth_user",
            "expected_learning": False,
            "sessions_to_send": 15  # Should transition to full auth
        }
    ]
    
    results = {}
    
    for phase_test in phase_tests:
        print(f"\n🧪 Testing {phase_test['phase_name']}")
        print("-" * 60)
        
        user_id = phase_test["user_id"]
        
        # Build up user profile with many sessions
        print(f"📊 Building profile with {phase_test['sessions_to_send']} sessions...")
        
        normal_sessions = []
        for i in range(phase_test['sessions_to_send']):
            normal_data = {
                "user_id": user_id,
                "session_id": f"profile_{i}_{int(datetime.now().timestamp())}",
                "logs": [{
                    "timestamp": datetime.now().isoformat(),
                    "event_type": "touch_sequence",
                    "data": {
                        "touch_events": [
                            {
                                "x": 150 + np.random.normal(0, 10),
                                "y": 200 + np.random.normal(0, 10), 
                                "pressure": 0.5 + np.random.normal(0, 0.08), 
                                "duration": 120 + np.random.normal(0, 15)
                            }
                        ],
                        "accelerometer": {
                            "x": 0.02 + np.random.normal(0, 0.01), 
                            "y": 0.15 + np.random.normal(0, 0.03), 
                            "z": 9.78 + np.random.normal(0, 0.2)
                        },
                        "gyroscope": {
                            "x": 0.001 + np.random.normal(0, 0.001), 
                            "y": 0.002 + np.random.normal(0, 0.001), 
                            "z": 0.0015 + np.random.normal(0, 0.001)
                        }
                    }
                }]
            }
            
            response = requests.post(f"{base_url}/analyze-mobile", json=normal_data)
            if response.status_code == 200:
                result = response.json()
                normal_sessions.append(result)
                
                # Print every few sessions to show progress
                if (i + 1) % 3 == 0:
                    print(f"  Session {i+1}: Decision={result.get('decision')}, "
                          f"Confidence={result.get('confidence', 0):.3f}, "
                          f"Risk={result.get('risk_score', 0):.4f}")
        
        # Now test with clear anomaly
        print(f"\n🚨 Testing anomaly detection for {phase_test['phase_name']}...")
        
        bot_anomaly = {
            "user_id": user_id,
            "session_id": f"bot_anomaly_{int(datetime.now().timestamp())}",
            "logs": [{
                "timestamp": datetime.now().isoformat(),
                "event_type": "touch_sequence",
                "data": {
                    "touch_events": [
                        {"x": 100, "y": 100, "pressure": 1.0, "duration": 50},
                        {"x": 100, "y": 100, "pressure": 1.0, "duration": 50},
                        {"x": 100, "y": 100, "pressure": 1.0, "duration": 50},
                        {"x": 100, "y": 100, "pressure": 1.0, "duration": 50}
                    ],
                    "accelerometer": {"x": 0.0, "y": 0.0, "z": 10.0},
                    "gyroscope": {"x": 0.0, "y": 0.0, "z": 0.0}
                }
            }]
        }
        
        anomaly_response = requests.post(f"{base_url}/analyze-mobile", json=bot_anomaly)
        if anomaly_response.status_code == 200:
            anomaly_result = anomaly_response.json()
            
            print(f"\n📋 ANOMALY RESULT:")
            print(f"  Decision: {anomaly_result.get('decision')}")
            print(f"  Confidence: {anomaly_result.get('confidence', 0):.3f}")
            print(f"  Risk Score: {anomaly_result.get('risk_score', 0):.4f}")
            print(f"  Similarity: {anomaly_result.get('similarity_score', 0):.4f}")
            print(f"  Risk Factors: {anomaly_result.get('risk_factors', [])}")
            
            # Check if anomaly was detected
            risk_score = anomaly_result.get('risk_score', 0)
            decision = anomaly_result.get('decision', 'unknown')
            
            if risk_score > 0.3 or decision in ['challenge', 'block']:
                print(f"  ✅ ANOMALY DETECTED - Phase working!")
            elif phase_test['expected_learning'] and decision == 'learn':
                print(f"  ⚠️  Still learning (expected for learning phase)")
            else:
                print(f"  ❌ ANOMALY MISSED - Phase not working properly")
            
            results[phase_test['phase_name']] = {
                'normal_sessions': normal_sessions,
                'anomaly_result': anomaly_result,
                'anomaly_detected': risk_score > 0.3 or decision in ['challenge', 'block']
            }
        
        # Also test extreme behavioral variation
        extreme_anomaly = {
            "user_id": user_id,
            "session_id": f"extreme_anomaly_{int(datetime.now().timestamp())}",
            "logs": [{
                "timestamp": datetime.now().isoformat(),
                "event_type": "touch_sequence",
                "data": {
                    "touch_events": [
                        {"x": 1000, "y": 2000, "pressure": 0.0, "duration": 1000},  # Extreme outlier
                        {"x": -500, "y": -1000, "pressure": 2.0, "duration": 1}    # Impossible values
                    ],
                    "accelerometer": {"x": 100.0, "y": 200.0, "z": 500.0},  # Impossible acceleration
                    "gyroscope": {"x": 1000.0, "y": 2000.0, "z": 3000.0}    # Device impossible to hold
                }
            }]
        }
        
        extreme_response = requests.post(f"{base_url}/analyze-mobile", json=extreme_anomaly)
        if extreme_response.status_code == 200:
            extreme_result = extreme_response.json()
            
            print(f"\n🔥 EXTREME ANOMALY RESULT:")
            print(f"  Decision: {extreme_result.get('decision')}")
            print(f"  Risk Score: {extreme_result.get('risk_score', 0):.4f}")
            print(f"  Similarity: {extreme_result.get('similarity_score', 0):.4f}")
            
            extreme_risk = extreme_result.get('risk_score', 0)
            if extreme_risk > 0.5 or extreme_result.get('decision') in ['challenge', 'block']:
                print(f"  ✅ EXTREME ANOMALY DETECTED")
            else:
                print(f"  ❌ EXTREME ANOMALY MISSED - System not sensitive enough")
    
    # Final analysis
    print(f"\n🎯 PHASE TESTING SUMMARY")
    print("="*60)
    
    detection_count = 0
    total_phases = len(results)
    
    for phase_name, result in results.items():
        detected = result.get('anomaly_detected', False)
        detection_count += 1 if detected else 0
        status = "✅ WORKING" if detected else "❌ NOT WORKING"
        print(f"{phase_name}: {status}")
    
    detection_rate = detection_count / total_phases if total_phases > 0 else 0
    print(f"\nOverall Detection Rate: {detection_rate:.1%} ({detection_count}/{total_phases})")
    
    if detection_rate >= 0.7:
        print("✅ Good anomaly detection across phases")
    elif detection_rate >= 0.3:
        print("⚠️  Partial anomaly detection - needs improvement")
    else:
        print("❌ Poor anomaly detection - major issues")
    
    return results

def debug_vector_similarity_issue():
    """Debug why all similarities are 1.0"""
    print("\n🔍 DEBUGGING VECTOR SIMILARITY ISSUE")
    print("="*60)
    
    base_url = "http://localhost:8001"
    user_id = "similarity_debug"
    
    # Send very different behavioral patterns
    patterns = [
        {
            "name": "Slow Careful User",
            "data": {
                "user_id": user_id,
                "session_id": f"slow_{int(datetime.now().timestamp())}",
                "logs": [{
                    "timestamp": datetime.now().isoformat(),
                    "event_type": "touch_sequence",
                    "data": {
                        "touch_events": [
                            {"x": 200, "y": 300, "pressure": 0.3, "duration": 300}  # Slow, light touch
                        ],
                        "accelerometer": {"x": 0.01, "y": 0.05, "z": 9.8},
                        "gyroscope": {"x": 0.0001, "y": 0.0002, "z": 0.0001}
                    }
                }]
            }
        },
        {
            "name": "Fast Aggressive User",
            "data": {
                "user_id": user_id,
                "session_id": f"fast_{int(datetime.now().timestamp())}",
                "logs": [{
                    "timestamp": datetime.now().isoformat(),
                    "event_type": "touch_sequence",
                    "data": {
                        "touch_events": [
                            {"x": 50, "y": 80, "pressure": 0.9, "duration": 30}  # Fast, hard touch
                        ],
                        "accelerometer": {"x": 0.1, "y": 0.3, "z": 9.5},
                        "gyroscope": {"x": 0.01, "y": 0.02, "z": 0.015}
                    }
                }]
            }
        },
        {
            "name": "Robot User",
            "data": {
                "user_id": user_id,
                "session_id": f"robot_{int(datetime.now().timestamp())}",
                "logs": [{
                    "timestamp": datetime.now().isoformat(),
                    "event_type": "touch_sequence",
                    "data": {
                        "touch_events": [
                            {"x": 100, "y": 100, "pressure": 1.0, "duration": 50}  # Perfect robot
                        ],
                        "accelerometer": {"x": 0.0, "y": 0.0, "z": 10.0},
                        "gyroscope": {"x": 0.0, "y": 0.0, "z": 0.0}
                    }
                }]
            }
        }
    ]
    
    vectors = []
    for pattern in patterns:
        print(f"\nTesting {pattern['name']}...")
        response = requests.post(f"{base_url}/analyze-mobile", json=pattern['data'])
        
        if response.status_code == 200:
            result = response.json()
            session_vector = result.get('session_vector', [])
            
            if session_vector:
                vectors.append({
                    'name': pattern['name'],
                    'vector': session_vector,
                    'similarity': result.get('similarity_score', 0),
                    'risk': result.get('risk_score', 0)
                })
                
                # Show vector stats
                vector_array = np.array(session_vector)
                non_zero = np.count_nonzero(vector_array)
                print(f"  Vector length: {len(session_vector)}")
                print(f"  Non-zero elements: {non_zero}")
                print(f"  Vector mean: {np.mean(vector_array):.6f}")
                print(f"  Vector std: {np.std(vector_array):.6f}")
                print(f"  Similarity score: {result.get('similarity_score', 0):.6f}")
            else:
                print(f"  ❌ No vector returned")
    
    # Compare vectors manually
    if len(vectors) >= 2:
        print(f"\n🔬 MANUAL VECTOR COMPARISON:")
        
        for i in range(len(vectors)):
            for j in range(i+1, len(vectors)):
                v1 = np.array(vectors[i]['vector'])
                v2 = np.array(vectors[j]['vector'])
                
                # Manual cosine similarity
                dot_product = np.dot(v1, v2)
                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)
                
                if norm1 == 0 or norm2 == 0:
                    cosine_sim = 0.0
                else:
                    cosine_sim = dot_product / (norm1 * norm2)
                
                # Euclidean distance
                euclidean_dist = np.linalg.norm(v1 - v2)
                
                print(f"\n{vectors[i]['name']} vs {vectors[j]['name']}:")
                print(f"  Cosine similarity: {cosine_sim:.6f}")
                print(f"  Euclidean distance: {euclidean_dist:.6f}")
                print(f"  Vectors identical: {np.array_equal(v1, v2)}")
                
                if np.array_equal(v1, v2):
                    print("  ⚠️  VECTORS ARE IDENTICAL - This explains 1.0 similarity!")
                elif cosine_sim > 0.99:
                    print("  ⚠️  VECTORS ARE NEARLY IDENTICAL")
                elif cosine_sim < 0.3:
                    print("  ✅ VECTORS ARE CLEARLY DIFFERENT")

if __name__ == "__main__":
    # Force users into different phases and test
    phase_results = force_user_phase_and_test()
    
    # Debug vector similarity issue
    debug_vector_similarity_issue()
    
    print(f"\n🎉 COMPREHENSIVE PHASE AND VECTOR TESTING COMPLETED!")

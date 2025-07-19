#!/usr/bin/env python3
"""Test GNN integration and anomaly detection with established baseline"""

import requests
import json
import time

def test_gnn_anomaly_detection():
    """Test GNN anomaly detection after establishing baseline"""
    print("🔍 TESTING GNN ANOMALY DETECTION WITH ESTABLISHED BASELINE")
    print("="*80)
    
    user_id = "gnn_test_user"
    
    # Step 1: Establish baseline with normal behavior
    print("\n🏗️  STEP 1: Establishing Baseline with Normal Behavior")
    print("-" * 60)
    
    for i in range(3):
        normal_data = {
            "user_id": user_id,
            "session_id": f"normal_baseline_{i+1}",
            "logs": [
                {
                    "timestamp": f"2024-01-01T10:0{i}:00",
                    "event_type": "touch_sequence",
                    "data": {
                        "touch_events": [
                            {"coordinates": [100+i*5, 200+i*5], "pressure": 0.7+i*0.1, "duration": 120+i*5},
                            {"coordinates": [105+i*5, 205+i*5], "pressure": 0.8+i*0.1, "duration": 115+i*5}
                        ],
                        "accelerometer": [
                            {"x": 0.1+i*0.02, "y": 0.2+i*0.02, "z": 9.8+i*0.01}
                        ],
                        "gyroscope": [
                            {"x": 0.01+i*0.005, "y": 0.02+i*0.005, "z": 0.01+i*0.005}
                        ]
                    }
                }
            ]
        }
        
        response = requests.post("http://localhost:8001/analyze-mobile", json=normal_data, timeout=30)
        if response.status_code == 200:
            result = response.json()
            print(f"   Session {i+1}: Decision={result.get('decision')}, Confidence={result.get('confidence')}, Risk={result.get('risk_score', 0):.3f}")
        time.sleep(0.5)
    
    # Step 2: Test with highly anomalous behavior
    print(f"\n🚨 STEP 2: Testing Anomalous Robot Behavior")
    print("-" * 60)
    
    anomaly_data = {
        "user_id": user_id,
        "session_id": "suspicious_robot_attack",
        "logs": [
            {
                "timestamp": "2024-01-01T10:10:00",
                "event_type": "touch_sequence",
                "data": {
                    "touch_events": [
                        {"coordinates": [0, 0], "pressure": 1.0, "duration": 100},
                        {"coordinates": [0, 0], "pressure": 1.0, "duration": 100}, 
                        {"coordinates": [0, 0], "pressure": 1.0, "duration": 100},
                        {"coordinates": [0, 0], "pressure": 1.0, "duration": 100},
                        {"coordinates": [0, 0], "pressure": 1.0, "duration": 100}
                    ],
                    "accelerometer": [
                        {"x": 0.0, "y": 0.0, "z": 9.8},
                        {"x": 0.0, "y": 0.0, "z": 9.8},
                        {"x": 0.0, "y": 0.0, "z": 9.8}
                    ],
                    "gyroscope": [
                        {"x": 0.0, "y": 0.0, "z": 0.0},
                        {"x": 0.0, "y": 0.0, "z": 0.0}, 
                        {"x": 0.0, "y": 0.0, "z": 0.0}
                    ]
                }
            }
        ]
    }
    
    response = requests.post("http://localhost:8001/analyze-mobile", json=anomaly_data, timeout=30)
    if response.status_code == 200:
        result = response.json()
        
        print(f"🔍 ANOMALY TEST RESULTS:")
        print(f"   Decision: {result.get('decision')}")
        print(f"   Confidence: {result.get('confidence')}")
        print(f"   Risk Score: {result.get('risk_score', 0):.6f}")
        print(f"   Similarity: {result.get('similarity_score', 0):.6f}")
        print(f"   Learning Phase: {result.get('learning_phase')}")
        
        risk_factors = result.get('risk_factors', [])
        if risk_factors:
            print(f"   Risk Factors:")
            for factor in risk_factors:
                print(f"     - {factor}")
        
        # Check if GNN or adaptive layers detected the anomaly
        if result.get('decision') == 'deny' or result.get('risk_score', 0) > 0.5:
            print(f"\n✅ SUCCESS: Anomaly Detection Working!")
            print(f"   - High risk behavior was detected")
            print(f"   - System correctly identified suspicious patterns")
        elif result.get('risk_score', 0) > 0.1:
            print(f"\n⚠️  PARTIAL SUCCESS: Anomaly Detected but Not Blocked")
            print(f"   - System detected suspicious patterns (risk > 0.1)")
            print(f"   - May need threshold adjustment or more baseline data")
        else:
            print(f"\n❌ ISSUE: Anomaly Not Detected")
            print(f"   - Identical robot behavior not flagged as suspicious")
            print(f"   - GNN integration may need work")
    
    # Step 3: Test extreme behavioral anomaly
    print(f"\n💥 STEP 3: Testing Extreme Anomalous Behavior")
    print("-" * 60)
    
    extreme_anomaly = {
        "user_id": user_id,
        "session_id": "extreme_attack_pattern",
        "logs": [
            {
                "timestamp": "2024-01-01T10:15:00", 
                "event_type": "touch_sequence",
                "data": {
                    "touch_events": [
                        {"coordinates": [0, 0], "pressure": 5.0, "duration": 1},
                        {"coordinates": [1000, 1000], "pressure": 5.0, "duration": 1},
                        {"coordinates": [0, 1000], "pressure": 5.0, "duration": 1},
                        {"coordinates": [1000, 0], "pressure": 5.0, "duration": 1}
                    ],
                    "accelerometer": [
                        {"x": 10.0, "y": 10.0, "z": 20.0}
                    ],
                    "gyroscope": [
                        {"x": 5.0, "y": 5.0, "z": 5.0}
                    ]
                }
            }
        ]
    }
    
    response = requests.post("http://localhost:8001/analyze-mobile", json=extreme_anomaly, timeout=30)
    if response.status_code == 200:
        result = response.json()
        
        print(f"🔍 EXTREME ANOMALY TEST RESULTS:")
        print(f"   Decision: {result.get('decision')}")
        print(f"   Risk Score: {result.get('risk_score', 0):.6f}")
        print(f"   Similarity: {result.get('similarity_score', 0):.6f}")
        
        if result.get('decision') == 'deny' or result.get('risk_score', 0) > 0.3:
            print(f"\n🎯 EXCELLENT: Extreme Anomaly Detected!")
        else:
            print(f"\n❓ Extreme anomaly risk: {result.get('risk_score', 0):.6f}")

if __name__ == "__main__":
    test_gnn_anomaly_detection()

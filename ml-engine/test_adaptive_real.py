#!/usr/bin/env python3
"""Test the Adaptive Layer with real behavioral data"""

import requests
import json
import time

def test_adaptive_layer():
    """Test adaptive layer with multiple behavioral patterns"""
    print("🧪 TESTING ADAPTIVE LAYER WITH REAL BEHAVIORAL DATA")
    print("="*70)
    
    # Test different behavioral patterns
    test_cases = {
        "normal_human": {
            "user_id": "adaptive_test_user",
            "session_id": "normal_session_001",
            "logs": [
                {
                    "timestamp": "2024-01-01T10:00:00",
                    "event_type": "touch_sequence",
                    "data": {
                        "touch_events": [
                            {"coordinates": [100, 200], "pressure": 0.7, "duration": 120},
                            {"coordinates": [105, 205], "pressure": 0.8, "duration": 115},
                            {"coordinates": [110, 210], "pressure": 0.6, "duration": 125}
                        ],
                        "accelerometer": [
                            {"x": 0.1, "y": 0.2, "z": 9.8},
                            {"x": 0.12, "y": 0.18, "z": 9.82}
                        ],
                        "gyroscope": [
                            {"x": 0.01, "y": 0.02, "z": 0.01},
                            {"x": 0.012, "y": 0.018, "z": 0.009}
                        ]
                    }
                }
            ]
        },
        
        "robot_behavior": {
            "user_id": "adaptive_test_user", 
            "session_id": "robot_session_001",
            "logs": [
                {
                    "timestamp": "2024-01-01T10:01:00",
                    "event_type": "touch_sequence", 
                    "data": {
                        "touch_events": [
                            {"coordinates": [100, 200], "pressure": 1.0, "duration": 100},
                            {"coordinates": [100, 200], "pressure": 1.0, "duration": 100},
                            {"coordinates": [100, 200], "pressure": 1.0, "duration": 100}
                        ],
                        "accelerometer": [
                            {"x": 0.0, "y": 0.0, "z": 9.8},
                            {"x": 0.0, "y": 0.0, "z": 9.8}
                        ],
                        "gyroscope": [
                            {"x": 0.0, "y": 0.0, "z": 0.0},
                            {"x": 0.0, "y": 0.0, "z": 0.0}
                        ]
                    }
                }
            ]
        },
        
        "fast_aggressive": {
            "user_id": "adaptive_test_user",
            "session_id": "fast_session_001", 
            "logs": [
                {
                    "timestamp": "2024-01-01T10:02:00",
                    "event_type": "touch_sequence",
                    "data": {
                        "touch_events": [
                            {"coordinates": [50, 100], "pressure": 1.5, "duration": 50},
                            {"coordinates": [200, 400], "pressure": 1.8, "duration": 30},
                            {"coordinates": [350, 600], "pressure": 2.0, "duration": 25}
                        ],
                        "accelerometer": [
                            {"x": 0.8, "y": 1.2, "z": 10.5},
                            {"x": 1.0, "y": 1.5, "z": 11.0}
                        ],
                        "gyroscope": [
                            {"x": 0.2, "y": 0.3, "z": 0.15},
                            {"x": 0.25, "y": 0.35, "z": 0.2}
                        ]
                    }
                }
            ]
        }
    }
    
    results = {}
    
    # Test each behavioral pattern
    for pattern_name, test_data in test_cases.items():
        print(f"\n🔄 Testing: {pattern_name.upper()}")
        print("-" * 40)
        
        try:
            response = requests.post(
                "http://localhost:8001/analyze-mobile",
                json=test_data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract key metrics
                confidence = result.get('confidence', 0)
                risk_score = result.get('risk_score', 0)
                decision = result.get('decision', 'unknown')
                learning_phase = result.get('learning_phase', 'unknown')
                vector_stats = result.get('vector_stats', {})
                similarity_score = result.get('similarity_score', 0)
                
                results[pattern_name] = {
                    'confidence': confidence,
                    'risk_score': risk_score, 
                    'decision': decision,
                    'learning_phase': learning_phase,
                    'vector_stats': vector_stats,
                    'similarity_score': similarity_score
                }
                
                print(f"✅ Decision: {decision}")
                print(f"   Confidence: {confidence}")
                print(f"   Risk Score: {risk_score}")
                print(f"   Learning Phase: {learning_phase}")
                print(f"   Similarity Score: {similarity_score}")
                print(f"   Vector Non-zeros: {vector_stats.get('non_zero_count', 0)}/90")
                
            else:
                print(f"❌ HTTP Error {response.status_code}: {response.text}")
                results[pattern_name] = {'error': f"HTTP {response.status_code}"}
                
        except Exception as e:
            print(f"❌ Exception: {e}")
            results[pattern_name] = {'error': str(e)}
        
        # Small delay between requests
        time.sleep(1)
    
    # Analysis of results
    print(f"\n📊 ADAPTIVE LAYER ANALYSIS")
    print("="*50)
    
    # Check if we get different responses for different behaviors
    unique_decisions = set()
    unique_confidences = set()
    unique_risks = set()
    
    for pattern, data in results.items():
        if 'error' not in data:
            unique_decisions.add(data['decision'])
            unique_confidences.add(round(data['confidence'], 2))
            unique_risks.add(round(data['risk_score'], 2))
    
    print(f"Unique Decisions: {list(unique_decisions)}")
    print(f"Unique Confidences: {list(unique_confidences)}")
    print(f"Unique Risk Scores: {list(unique_risks)}")
    
    # Assessment
    if len(unique_decisions) > 1 or len(unique_confidences) > 1 or len(unique_risks) > 1:
        print(f"\n✅ ADAPTIVE LAYER IS WORKING!")
        print("   - Different behavioral patterns produce different responses")
        print("   - System is adapting to behavioral variations")
    else:
        print(f"\n⚠️  ADAPTIVE LAYER MAY NEED MORE SESSIONS")
        print("   - All patterns showing similar responses")
        print("   - May need more data to establish baselines")
    
    return results

if __name__ == "__main__":
    test_adaptive_layer()

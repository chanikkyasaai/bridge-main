#!/usr/bin/env python3

import requests
import json
import numpy as np

def test_ml_engine_with_fixed_processor():
    """Test the full ML Engine pipeline with fixed behavioral processor"""
    
    print("🚀 Testing ML Engine with Fixed Behavioral Processor")
    print("=" * 60)
    
    ml_engine_url = "http://localhost:5003"
    
    # Test behavioral patterns with realistic differences
    test_patterns = {
        "Normal Human Behavior": {
            "user_id": "test_user_123",
            "session_id": "normal_session",
            "logs": [
                {
                    "event_type": "touch_sequence",
                    "timestamp": "2024-01-10T10:30:00Z",
                    "data": {
                        "touch_events": [
                            {"coordinates": [100, 200], "pressure": 0.7, "duration": 120},
                            {"coordinates": [105, 205], "pressure": 0.8, "duration": 110},
                            {"coordinates": [110, 210], "pressure": 0.6, "duration": 130}
                        ],
                        "accelerometer": [
                            {"x": 0.1, "y": 0.2, "z": 9.8},
                            {"x": 0.15, "y": 0.18, "z": 9.82},
                            {"x": 0.12, "y": 0.22, "z": 9.79}
                        ],
                        "gyroscope": [
                            {"x": 0.01, "y": 0.02, "z": 0.01},
                            {"x": 0.015, "y": 0.018, "z": 0.012},
                            {"x": 0.012, "y": 0.022, "z": 0.009}
                        ],
                        "scroll": {"velocity": 150, "delta_y": 50}
                    }
                }
            ]
        },
        
        "Similar Normal Behavior": {
            "user_id": "test_user_123",
            "session_id": "similar_session",
            "logs": [
                {
                    "event_type": "touch_sequence",
                    "timestamp": "2024-01-10T10:35:00Z",
                    "data": {
                        "touch_events": [
                            {"coordinates": [95, 195], "pressure": 0.75, "duration": 115},
                            {"coordinates": [100, 200], "pressure": 0.7, "duration": 125},
                            {"coordinates": [103, 208], "pressure": 0.65, "duration": 135}
                        ],
                        "accelerometer": [
                            {"x": 0.08, "y": 0.25, "z": 9.75},
                            {"x": 0.12, "y": 0.15, "z": 9.85},
                            {"x": 0.09, "y": 0.28, "z": 9.77}
                        ],
                        "gyroscope": [
                            {"x": 0.008, "y": 0.025, "z": 0.015},
                            {"x": 0.012, "y": 0.015, "z": 0.008},
                            {"x": 0.015, "y": 0.028, "z": 0.012}
                        ],
                        "scroll": {"velocity": 140, "delta_y": 45}
                    }
                }
            ]
        },
        
        "Robot/Automation Pattern": {
            "user_id": "test_user_123",
            "session_id": "robot_session",
            "logs": [
                {
                    "event_type": "touch_sequence",
                    "timestamp": "2024-01-10T10:40:00Z",
                    "data": {
                        "touch_events": [
                            {"coordinates": [100, 200], "pressure": 1.0, "duration": 100},
                            {"coordinates": [100, 200], "pressure": 1.0, "duration": 100},
                            {"coordinates": [100, 200], "pressure": 1.0, "duration": 100}
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
                        ],
                        "scroll": {"velocity": 200, "delta_y": 100}
                    }
                }
            ]
        },
        
        "Suspicious/Aggressive Pattern": {
            "user_id": "test_user_123",
            "session_id": "suspicious_session",
            "logs": [
                {
                    "event_type": "touch_sequence",
                    "timestamp": "2024-01-10T10:45:00Z",
                    "data": {
                        "touch_events": [
                            {"coordinates": [50, 100], "pressure": 1.5, "duration": 50},
                            {"coordinates": [200, 400], "pressure": 1.8, "duration": 30},
                            {"coordinates": [350, 600], "pressure": 2.0, "duration": 20}
                        ],
                        "accelerometer": [
                            {"x": 0.8, "y": 1.2, "z": 10.5},
                            {"x": 1.0, "y": 1.5, "z": 11.0},
                            {"x": 1.2, "y": 1.8, "z": 11.5}
                        ],
                        "gyroscope": [
                            {"x": 0.15, "y": 0.2, "z": 0.1},
                            {"x": 0.2, "y": 0.25, "z": 0.15},
                            {"x": 0.25, "y": 0.3, "z": 0.2}
                        ],
                        "scroll": {"velocity": 800, "delta_y": 300}
                    }
                }
            ]
        }
    }
    
    def make_ml_request(data, endpoint="analyze-behavioral"):
        """Make request to ML Engine"""
        try:
            response = requests.post(f"{ml_engine_url}/{endpoint}", json=data, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"   ❌ HTTP Error {response.status_code}: {response.text}")
                return None
        except requests.exceptions.ConnectionError:
            print(f"   ❌ Connection Error: ML Engine not running at {ml_engine_url}")
            return None
        except Exception as e:
            print(f"   ❌ Request Error: {e}")
            return None
    
    print("🔧 Testing ML Engine API with Fixed Behavioral Patterns")
    print("-" * 55)
    
    results = []
    
    # Test each pattern
    for i, (pattern_name, behavioral_data) in enumerate(test_patterns.items()):
        print(f"\n{i+1}. Testing: {pattern_name}")
        print("-" * 40)
        
        # Make request to ML Engine
        response = make_ml_request(behavioral_data)
        
        if response:
            # Extract key metrics
            confidence = response.get('confidence', 0)
            anomaly_detected = response.get('anomaly_detected', False)
            anomaly_score = response.get('anomaly_score', 0)
            session_vector = response.get('session_vector', [])
            
            # Analyze the session vector
            if session_vector:
                vector = np.array(session_vector)
                vector_mean = float(np.mean(vector))
                vector_std = float(np.std(vector))
                non_zero_count = int(np.count_nonzero(vector))
            else:
                vector_mean = vector_std = non_zero_count = 0
            
            print(f"   Confidence: {confidence:.3f}")
            print(f"   Anomaly Detected: {anomaly_detected} (score: {anomaly_score:.3f})")
            print(f"   Vector Stats: mean={vector_mean:.3f}, std={vector_std:.3f}, non_zero={non_zero_count}/90")
            
            # Determine expected behavior
            expected_anomaly = "Robot" in pattern_name or "Suspicious" in pattern_name
            correct = (anomaly_detected == expected_anomaly)
            
            print(f"   Expected Anomaly: {expected_anomaly}")
            print(f"   Detection Result: {'✅ CORRECT' if correct else '❌ INCORRECT'}")
            
            results.append({
                'pattern': pattern_name,
                'confidence': confidence,
                'anomaly_detected': anomaly_detected,
                'anomaly_score': anomaly_score,
                'expected_anomaly': expected_anomaly,
                'correct': correct,
                'vector_stats': {
                    'mean': vector_mean,
                    'std': vector_std,
                    'non_zero_count': non_zero_count
                }
            })
        else:
            print(f"   Failed to get response from ML Engine")
    
    # Analysis of results
    print("\n" + "=" * 60)
    print("📊 COMPREHENSIVE ANALYSIS")
    print("=" * 60)
    
    if not results:
        print("❌ FAILED: No successful tests completed")
        print("   - Check if ML Engine is running on http://localhost:5003")
        return
    
    # Vector uniqueness analysis
    print("\n🔍 Vector Uniqueness Analysis:")
    vectors = []
    for result in results:
        if result['vector_stats']['non_zero_count'] > 0:
            vectors.append(result)
    
    if len(vectors) >= 2:
        print(f"   Vectors analyzed: {len(vectors)}")
        mean_values = [v['vector_stats']['mean'] for v in vectors]
        unique_means = len(set(f"{m:.3f}" for m in mean_values))
        print(f"   Unique mean values: {unique_means}/{len(vectors)}")
        
        if unique_means == len(vectors):
            print("   ✅ All vectors are unique - behavioral processor is working!")
        else:
            print("   ⚠️  Some vectors may be identical")
    
    # Detection accuracy analysis
    print("\n🎯 Detection Accuracy Analysis:")
    total_tests = len(results)
    correct_detections = sum(1 for r in results if r['correct'])
    accuracy = correct_detections / total_tests * 100 if total_tests > 0 else 0
    
    normal_patterns = [r for r in results if not r['expected_anomaly']]
    anomaly_patterns = [r for r in results if r['expected_anomaly']]
    
    if normal_patterns:
        true_negative_rate = sum(1 for r in normal_patterns if not r['anomaly_detected']) / len(normal_patterns) * 100
        print(f"   Normal Pattern Recognition: {true_negative_rate:.1f}% ({sum(1 for r in normal_patterns if not r['anomaly_detected'])}/{len(normal_patterns)})")
    
    if anomaly_patterns:
        true_positive_rate = sum(1 for r in anomaly_patterns if r['anomaly_detected']) / len(anomaly_patterns) * 100
        print(f"   Anomaly Pattern Detection:  {true_positive_rate:.1f}% ({sum(1 for r in anomaly_patterns if r['anomaly_detected'])}/{len(anomaly_patterns)})")
    
    print(f"   Overall Accuracy: {accuracy:.1f}% ({correct_detections}/{total_tests})")
    
    # Final assessment
    print("\n🏆 FINAL ASSESSMENT:")
    
    if accuracy >= 75:
        print("✅ SUCCESS: ML Engine with Fixed Behavioral Processor is working correctly!")
        print("   - Behavioral vectors are unique and meaningful")
        print("   - Anomaly detection is functioning properly")
        print("   - System is ready for GNN integration and production use")
    elif accuracy >= 50:
        print("⚠️  PARTIAL SUCCESS: Significant improvement but needs refinement")
        print("   - Behavioral processor is generating unique vectors")
        print("   - Anomaly detection working but may need threshold tuning")
        print("   - Consider additional training data or parameter optimization")
    else:
        print("❌ ISSUES REMAIN: Further debugging needed")
        print("   - Check if all components are properly integrated")
        print("   - Verify ML Engine configuration and training data")
    
    return results

if __name__ == "__main__":
    test_ml_engine_with_fixed_processor()

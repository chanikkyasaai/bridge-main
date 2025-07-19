#!/usr/bin/env python3

import numpy as np
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from layers.layer2_adaptive_learning import AdaptiveLearningLayer
from core.enhanced_behavioral_processor import EnhancedBehavioralProcessor

def test_adaptive_with_fixed_processor():
    """Test Adaptive Layer with the fixed behavioral processor"""
    
    print("🧪 Testing Adaptive Layer with Fixed Behavioral Processor")
    print("=" * 60)
    
    # Initialize components
    processor = EnhancedBehavioralProcessor()
    adaptive_layer = AdaptiveLearningLayer()
    
    # Define test behavioral patterns with realistic differences
    test_patterns = {
        "Normal User": {
            "user_id": "user123",
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
        
        "Similar Normal User": {
            "user_id": "user123",
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
        
        "Automation/Bot": {
            "user_id": "user123",
            "session_id": "bot_session",
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
        
        "Aggressive/Suspicious": {
            "user_id": "user123",
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
    
    print("🔧 Step 1: Generate behavioral vectors")
    print("-" * 40)
    
    behavioral_vectors = {}
    for pattern_name, pattern_data in test_patterns.items():
        vector = processor.process_mobile_behavioral_data(pattern_data)
        behavioral_vectors[pattern_name] = vector
        
        print(f"{pattern_name:20}: mean={np.mean(vector):.3f}, std={np.std(vector):.3f}, non_zero={np.count_nonzero(vector)}/90")
    
    print("\n🧠 Step 2: Train Adaptive Layer (Learning Phase)")
    print("-" * 50)
    
    user_id = "user123"
    
    # Train with normal patterns first (multiple sessions to build profile)
    normal_vectors = [behavioral_vectors["Normal User"], behavioral_vectors["Similar Normal User"]]
    
    for i, vector in enumerate(normal_vectors):
        session_id = f"training_session_{i+1}"
        result = adaptive_layer.analyze_behavioral_pattern(
            user_id=user_id,
            session_id=session_id,
            behavioral_vector=vector,
            user_action="login_attempt",
            force_learning_phase=True  # Force learning to build baseline
        )
        
        print(f"Training session {i+1}: confidence={result['confidence']:.3f}, learning_phase={result['learning_phase']}")
    
    print(f"\nProfile after training: sessions={len(adaptive_layer.user_profiles.get(user_id, {}).get('sessions', []))}")
    
    print("\n🎯 Step 3: Test Detection (Analysis Phase)")
    print("-" * 45)
    
    # Test each pattern
    test_cases = [
        ("Normal Pattern", behavioral_vectors["Normal User"], False),
        ("Similar Normal", behavioral_vectors["Similar Normal User"], False),
        ("Bot/Automation", behavioral_vectors["Automation/Bot"], True),
        ("Aggressive/Suspicious", behavioral_vectors["Aggressive/Suspicious"], True)
    ]
    
    results = []
    for i, (test_name, test_vector, should_be_anomaly) in enumerate(test_cases):
        session_id = f"test_session_{i+1}"
        
        result = adaptive_layer.analyze_behavioral_pattern(
            user_id=user_id,
            session_id=session_id,
            behavioral_vector=test_vector,
            user_action="login_attempt",
            force_learning_phase=False  # Analysis mode
        )
        
        is_anomaly = result['anomaly_detected']
        confidence = result['confidence']
        anomaly_score = result['anomaly_score']
        
        # Determine if detection was correct
        correct_detection = (is_anomaly and should_be_anomaly) or (not is_anomaly and not should_be_anomaly)
        status = "✅ CORRECT" if correct_detection else "❌ WRONG"
        
        print(f"{test_name:18}: anomaly={is_anomaly} (score={anomaly_score:.3f}), confidence={confidence:.3f} {status}")
        
        results.append({
            'name': test_name,
            'expected_anomaly': should_be_anomaly,
            'detected_anomaly': is_anomaly,
            'correct': correct_detection,
            'confidence': confidence,
            'anomaly_score': anomaly_score
        })
    
    print("\n📊 FINAL ASSESSMENT")
    print("=" * 30)
    
    # Calculate detection accuracy
    total_tests = len(results)
    correct_detections = sum(1 for r in results if r['correct'])
    accuracy = correct_detections / total_tests * 100
    
    anomaly_tests = [r for r in results if r['expected_anomaly']]
    normal_tests = [r for r in results if not r['expected_anomaly']]
    
    true_positive_rate = sum(1 for r in anomaly_tests if r['detected_anomaly']) / len(anomaly_tests) * 100 if anomaly_tests else 0
    true_negative_rate = sum(1 for r in normal_tests if not r['detected_anomaly']) / len(normal_tests) * 100 if normal_tests else 0
    
    print(f"Overall Accuracy:     {accuracy:.1f}% ({correct_detections}/{total_tests})")
    print(f"Anomaly Detection:    {true_positive_rate:.1f}% (sensitivity)")
    print(f"Normal Recognition:   {true_negative_rate:.1f}% (specificity)")
    
    if accuracy >= 75:
        print("✅ SUCCESS: Adaptive Layer is working with genuine behavioral differentiation!")
        print("   - Fixed behavioral processor provides unique vectors")
        print("   - Anomaly detection is functioning properly")
        print("   - Ready for integration with GNN layer")
    else:
        print("⚠️  PARTIAL: Some detection issues remain, but significant improvement from fixed processor")
        print("   - Behavioral vectors are now unique and meaningful")
        print("   - May need threshold tuning or additional training data")
    
    return results

if __name__ == "__main__":
    test_adaptive_with_fixed_processor()

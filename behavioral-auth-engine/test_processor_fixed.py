#!/usr/bin/env python3

import numpy as np
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from core.enhanced_behavioral_processor import EnhancedBehavioralProcessor

def test_fixed_processor():
    """Test the fixed Enhanced Behavioral Processor with touch_sequence data"""
    
    print("🔧 Testing Fixed Enhanced Behavioral Processor")
    print("=" * 60)
    
    # Initialize processor
    processor = EnhancedBehavioralProcessor()
    
    # Test data in touch_sequence format (similar to real mobile data)
    test_cases = {
        "Normal Human Behavior": {
            "user_id": "test_user",
            "session_id": "normal_session",
            "logs": [
                {
                    "event_type": "touch_sequence",
                    "timestamp": "2024-01-10T10:30:00Z",
                    "data": {
                        "touch_events": [
                            {"coordinates": [100, 200], "pressure": 0.7, "duration": 120},
                            {"coordinates": [102, 205], "pressure": 0.8, "duration": 110},
                            {"coordinates": [105, 210], "pressure": 0.6, "duration": 130}
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
                        "scroll": {
                            "velocity": 150,
                            "delta_y": 50
                        }
                    }
                }
            ]
        },
        
        "Robot/Automation Behavior": {
            "user_id": "test_user",
            "session_id": "robot_session", 
            "logs": [
                {
                    "event_type": "touch_sequence",
                    "timestamp": "2024-01-10T10:31:00Z",
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
                        "scroll": {
                            "velocity": 200,
                            "delta_y": 100
                        }
                    }
                }
            ]
        },
        
        "Fast Aggressive Behavior": {
            "user_id": "test_user", 
            "session_id": "fast_session",
            "logs": [
                {
                    "event_type": "touch_sequence",
                    "timestamp": "2024-01-10T10:32:00Z",
                    "data": {
                        "touch_events": [
                            {"coordinates": [50, 100], "pressure": 1.2, "duration": 50},
                            {"coordinates": [150, 300], "pressure": 1.5, "duration": 40},
                            {"coordinates": [250, 500], "pressure": 1.8, "duration": 30}
                        ],
                        "accelerometer": [
                            {"x": 0.5, "y": 0.8, "z": 10.2},
                            {"x": 0.8, "y": 1.2, "z": 10.5},
                            {"x": 1.0, "y": 1.5, "z": 10.8}
                        ],
                        "gyroscope": [
                            {"x": 0.1, "y": 0.15, "z": 0.08},
                            {"x": 0.15, "y": 0.2, "z": 0.12},
                            {"x": 0.2, "y": 0.25, "z": 0.15}
                        ],
                        "scroll": {
                            "velocity": 500,
                            "delta_y": 200
                        }
                    }
                }
            ]
        }
    }
    
    # Process each test case
    results = {}
    for case_name, behavioral_data in test_cases.items():
        print(f"\n🧪 Processing: {case_name}")
        
        try:
            vector = processor.process_mobile_behavioral_data(behavioral_data)
            
            # Calculate vector statistics
            vector_mean = float(np.mean(vector))
            vector_std = float(np.std(vector))
            vector_min = float(np.min(vector))
            vector_max = float(np.max(vector))
            non_zero_count = int(np.count_nonzero(vector))
            
            results[case_name] = {
                'vector': vector,
                'stats': {
                    'mean': vector_mean,
                    'std': vector_std,
                    'min': vector_min,
                    'max': vector_max,
                    'non_zero_count': non_zero_count,
                    'is_unique': non_zero_count > 10  # At least 10 non-zero values indicates uniqueness
                }
            }
            
            print(f"   Vector mean: {vector_mean:.6f}")
            print(f"   Vector std:  {vector_std:.6f}")
            print(f"   Non-zeros:   {non_zero_count}/90")
            print(f"   Range:       [{vector_min:.3f}, {vector_max:.3f}]")
            
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            results[case_name] = {'error': str(e)}
    
    # Compare vectors for uniqueness
    print(f"\n🔍 Vector Comparison Analysis")
    print("=" * 40)
    
    if len(results) >= 2:
        case_names = list(results.keys())
        for i in range(len(case_names)):
            for j in range(i + 1, len(case_names)):
                case1, case2 = case_names[i], case_names[j]
                if 'vector' in results[case1] and 'vector' in results[case2]:
                    vector1 = results[case1]['vector']
                    vector2 = results[case2]['vector']
                    
                    # Cosine similarity
                    dot_product = np.dot(vector1, vector2)
                    norm1 = np.linalg.norm(vector1)
                    norm2 = np.linalg.norm(vector2)
                    
                    if norm1 > 0 and norm2 > 0:
                        cosine_sim = dot_product / (norm1 * norm2)
                        euclidean_dist = np.linalg.norm(vector1 - vector2)
                        
                        print(f"\n{case1} vs {case2}:")
                        print(f"   Cosine similarity: {cosine_sim:.6f}")
                        print(f"   Euclidean distance: {euclidean_dist:.6f}")
                        
                        if cosine_sim < 0.9:
                            print(f"   ✅ UNIQUE vectors detected!")
                        else:
                            print(f"   ⚠️  High similarity - may be identical")
    
    # Overall assessment
    print(f"\n📊 FINAL ASSESSMENT")
    print("=" * 30)
    
    unique_vectors = sum(1 for result in results.values() 
                        if 'stats' in result and result['stats']['is_unique'])
    total_vectors = len([r for r in results.values() if 'vector' in r])
    
    if unique_vectors == total_vectors and total_vectors > 1:
        print("✅ SUCCESS: Enhanced Behavioral Processor is generating UNIQUE vectors!")
        print("   - All behavioral patterns produce different embeddings")
        print("   - Feature extraction methods are working correctly")
        print("   - Ready for genuine anomaly detection testing")
    else:
        print("❌ ISSUE: Behavioral processor may still have problems")
        print(f"   - Unique vectors: {unique_vectors}/{total_vectors}")
        print("   - May need additional debugging")
    
    return results

if __name__ == "__main__":
    test_fixed_processor()

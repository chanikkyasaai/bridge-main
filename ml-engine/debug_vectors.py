#!/usr/bin/env python3
"""Debug vector generation and similarity calculation"""

import requests
import numpy as np
import json
from typing import Dict, List

def test_vector_quality():
    """Test if vectors are being generated with enough diversity"""
    print("🔍 DEBUGGING VECTOR GENERATION")
    print("="*60)
    
    # Test data for different behaviors
    test_scenarios = [
        {
            "name": "Normal User",
            "data": {
                "user_id": "debug_user_001",
                "session_id": "debug_session_normal",
                "logs": [
                    {
                        "timestamp": "2024-01-01T10:00:00",
                        "event_type": "touch_sequence",
                        "data": {
                            "touch_events": [
                                {"x": 150, "y": 200, "pressure": 0.5, "duration": 120},
                                {"x": 160, "y": 210, "pressure": 0.6, "duration": 115},
                                {"x": 170, "y": 220, "pressure": 0.4, "duration": 125}
                            ],
                            "accelerometer": {"x": 0.02, "y": 0.15, "z": 9.78},
                            "gyroscope": {"x": 0.001, "y": 0.002, "z": 0.0015}
                        }
                    },
                    {
                        "timestamp": "2024-01-01T10:00:01",
                        "event_type": "keystroke_sequence",
                        "data": {
                            "keystrokes": [
                                {"key": "a", "dwell_time": 100, "pressure": 0.55},
                                {"key": "b", "dwell_time": 120, "pressure": 0.6},
                                {"key": "c", "dwell_time": 90, "pressure": 0.45}
                            ],
                            "typing_rhythm": [95, 110, 85],
                            "inter_key_intervals": [0.12, 0.15, 0.18]
                        }
                    }
                ]
            }
        },
        {
            "name": "Bot User",
            "data": {
                "user_id": "debug_user_002",
                "session_id": "debug_session_bot",
                "logs": [
                    {
                        "timestamp": "2024-01-01T10:00:00",
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
                    },
                    {
                        "timestamp": "2024-01-01T10:00:01",
                        "event_type": "keystroke_sequence",
                        "data": {
                            "keystrokes": [
                                {"key": "x", "dwell_time": 50, "pressure": 1.0},
                                {"key": "x", "dwell_time": 50, "pressure": 1.0},
                                {"key": "x", "dwell_time": 50, "pressure": 1.0}
                            ],
                            "typing_rhythm": [50, 50, 50],
                            "inter_key_intervals": [0.1, 0.1, 0.1]
                        }
                    }
                ]
            }
        }
    ]
    
    results = []
    
    for i, scenario in enumerate(test_scenarios):
        print(f"\n📊 Testing {scenario['name']}")
        print("-" * 40)
        
        try:
            # Send to ML Engine
            response = requests.post(
                "http://localhost:8001/analyze-mobile",
                json=scenario["data"],
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Response received")
                print(f"Session Vector Length: {len(result.get('session_vector', []))}")
                
                # Analyze vector
                vector = np.array(result.get('session_vector', []))
                non_zero = np.count_nonzero(vector)
                total_elements = len(vector)
                diversity_pct = (non_zero / total_elements * 100) if total_elements > 0 else 0
                
                print(f"Non-zero elements: {non_zero}/{total_elements} ({diversity_pct:.1f}%)")
                print(f"Vector mean: {vector.mean():.6f}")
                print(f"Vector std: {vector.std():.6f}")
                print(f"Vector min: {vector.min():.6f}")
                print(f"Vector max: {vector.max():.6f}")
                
                # Show some sample values
                print(f"First 10 values: {vector[:10]}")
                
                results.append({
                    'name': scenario['name'],
                    'vector': vector,
                    'diversity': diversity_pct,
                    'mean': vector.mean(),
                    'std': vector.std()
                })
                
            else:
                print(f"❌ Error: {response.status_code}")
                print(f"Response: {response.text}")
                
        except Exception as e:
            print(f"❌ Exception: {e}")
    
    # Compare vectors if we have at least 2
    if len(results) >= 2:
        print(f"\n🔬 VECTOR COMPARISON")
        print("="*60)
        
        v1 = results[0]['vector']
        v2 = results[1]['vector']
        
        # Manual cosine similarity
        def cosine_similarity(vec1, vec2):
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
        
        similarity = cosine_similarity(v1, v2)
        print(f"Manual Cosine Similarity: {similarity:.6f}")
        
        # Check if vectors are identical
        if np.array_equal(v1, v2):
            print("⚠️  Vectors are IDENTICAL!")
        else:
            print("✅ Vectors are different")
            
        # Check for zero vectors
        if np.all(v1 == 0):
            print("⚠️  Vector 1 is all zeros!")
        if np.all(v2 == 0):
            print("⚠️  Vector 2 is all zeros!")

if __name__ == "__main__":
    test_vector_quality()

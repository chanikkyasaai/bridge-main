#!/usr/bin/env python3
"""
Debug the Enhanced Behavioral Processor to find why all vectors are identical
"""

import sys
import os
sys.path.append('c:\\Users\\Hp\\OneDrive\\Desktop\\bridge\\bridge\\behavioral-auth-engine')

import numpy as np
from src.core.enhanced_behavioral_processor import EnhancedBehavioralProcessor
import json

def debug_behavioral_processor():
    """Debug the behavioral processor step by step"""
    print("🔍 DEBUGGING ENHANCED BEHAVIORAL PROCESSOR")
    print("="*60)
    
    processor = EnhancedBehavioralProcessor()
    
    # Test different behavioral patterns
    test_cases = [
        {
            "name": "Normal Human Behavior",
            "data": {
                "user_id": "test_user_1",
                "session_id": "session_1",
                "logs": [
                    {
                        "timestamp": "2024-01-01T10:00:00",
                        "event_type": "touch_sequence",
                        "data": {
                            "touch_events": [
                                {"x": 150, "y": 200, "pressure": 0.5, "duration": 120},
                                {"x": 160, "y": 210, "pressure": 0.6, "duration": 115}
                            ],
                            "accelerometer": {"x": 0.02, "y": 0.15, "z": 9.78},
                            "gyroscope": {"x": 0.001, "y": 0.002, "z": 0.0015}
                        }
                    }
                ]
            }
        },
        {
            "name": "Robot Behavior",
            "data": {
                "user_id": "test_user_2",
                "session_id": "session_2",
                "logs": [
                    {
                        "timestamp": "2024-01-01T10:00:00",
                        "event_type": "touch_sequence",
                        "data": {
                            "touch_events": [
                                {"x": 100, "y": 100, "pressure": 1.0, "duration": 50},
                                {"x": 100, "y": 100, "pressure": 1.0, "duration": 50}
                            ],
                            "accelerometer": {"x": 0.0, "y": 0.0, "z": 10.0},
                            "gyroscope": {"x": 0.0, "y": 0.0, "z": 0.0}
                        }
                    }
                ]
            }
        },
        {
            "name": "Empty Data",
            "data": {
                "user_id": "test_user_3",
                "session_id": "session_3",
                "logs": []
            }
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        print(f"\n--- Testing {test_case['name']} ---")
        
        try:
            # Process the data
            vector = processor.process_mobile_behavioral_data(test_case['data'])
            
            # Analyze the vector
            vector_stats = {
                'length': len(vector),
                'sum': float(np.sum(vector)),
                'mean': float(np.mean(vector)),
                'std': float(np.std(vector)),
                'non_zero_count': int(np.count_nonzero(vector)),
                'min': float(np.min(vector)),
                'max': float(np.max(vector)),
                'first_10': vector[:10].tolist(),
                'last_10': vector[-10:].tolist()
            }
            
            print(f"Vector length: {vector_stats['length']}")
            print(f"Vector sum: {vector_stats['sum']:.6f}")
            print(f"Non-zero elements: {vector_stats['non_zero_count']}")
            print(f"Mean: {vector_stats['mean']:.6f}")
            print(f"Std: {vector_stats['std']:.6f}")
            print(f"Range: [{vector_stats['min']:.6f}, {vector_stats['max']:.6f}]")
            
            results.append({
                'name': test_case['name'],
                'vector': vector,
                'stats': vector_stats,
                'data': test_case['data']
            })
            
            # Check if this looks like default features
            if vector_stats['sum'] == 0 or vector_stats['non_zero_count'] < 10:
                print("⚠️  Vector appears to be mostly empty (likely default features)")
            else:
                print("✅ Vector has meaningful content")
                
        except Exception as e:
            print(f"❌ Error processing {test_case['name']}: {e}")
            import traceback
            traceback.print_exc()
    
    # Compare vectors
    print(f"\n🔬 VECTOR COMPARISON")
    print("="*60)
    
    if len(results) >= 2:
        for i in range(len(results)):
            for j in range(i+1, len(results)):
                v1 = results[i]['vector']
                v2 = results[j]['vector']
                
                # Check if identical
                are_identical = np.array_equal(v1, v2)
                
                # Calculate similarity
                cosine_sim = 0.0
                if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                    cosine_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                
                print(f"\n{results[i]['name']} vs {results[j]['name']}:")
                print(f"  Identical: {are_identical}")
                print(f"  Cosine similarity: {cosine_sim:.6f}")
                
                if are_identical:
                    print("  🚨 PROBLEM: Vectors are identical despite different inputs!")
    
    # Test the default features directly
    print(f"\n🔧 TESTING DEFAULT FEATURES")
    print("="*60)
    
    try:
        default_features = processor._get_default_features()
        default_vector = default_features.to_vector()
        
        print(f"Default vector stats:")
        print(f"  Length: {len(default_vector)}")
        print(f"  Non-zero: {np.count_nonzero(default_vector)}")
        print(f"  Sum: {np.sum(default_vector):.6f}")
        print(f"  Mean: {np.mean(default_vector):.6f}")
        
        # Check if any of our results match the default
        for result in results:
            if np.array_equal(result['vector'], default_vector):
                print(f"  🚨 {result['name']} is using DEFAULT FEATURES!")
            else:
                print(f"  ✅ {result['name']} has unique features")
                
    except Exception as e:
        print(f"Error testing default features: {e}")
    
    return results

def debug_individual_extraction_methods():
    """Debug individual feature extraction methods"""
    print(f"\n🔧 DEBUGGING INDIVIDUAL FEATURE EXTRACTION")
    print("="*60)
    
    processor = EnhancedBehavioralProcessor()
    
    # Test data
    test_logs = [
        {
            "timestamp": "2024-01-01T10:00:00",
            "event_type": "touch_sequence",
            "data": {
                "touch_events": [
                    {"x": 150, "y": 200, "pressure": 0.5, "duration": 120},
                    {"x": 160, "y": 210, "pressure": 0.6, "duration": 115}
                ],
                "accelerometer": {"x": 0.02, "y": 0.15, "z": 9.78},
                "gyroscope": {"x": 0.001, "y": 0.002, "z": 0.0015}
            }
        }
    ]
    
    try:
        # Process logs to get organized events
        from collections import defaultdict
        events_by_type = defaultdict(list)
        for log in test_logs:
            event_type = log.get('event_type', '')
            events_by_type[event_type].append(log)
        
        print(f"Events by type: {dict(events_by_type)}")
        
        # Test each extraction method
        try:
            touch_features = processor._extract_touch_features(events_by_type)
            print(f"✅ Touch features: {touch_features}")
        except Exception as e:
            print(f"❌ Touch feature extraction failed: {e}")
            
        try:
            motion_features = processor._extract_motion_features(events_by_type)
            print(f"✅ Motion features: {motion_features}")
        except Exception as e:
            print(f"❌ Motion feature extraction failed: {e}")
            
        # Test other methods...
        
    except Exception as e:
        print(f"Error in individual method testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Debug main processor
    results = debug_behavioral_processor()
    
    # Debug individual methods
    debug_individual_extraction_methods()
    
    print(f"\n🎯 SUMMARY")
    print("="*60)
    
    if results:
        identical_count = 0
        total_comparisons = 0
        
        for i in range(len(results)):
            for j in range(i+1, len(results)):
                total_comparisons += 1
                if np.array_equal(results[i]['vector'], results[j]['vector']):
                    identical_count += 1
        
        if total_comparisons > 0:
            identical_rate = identical_count / total_comparisons
            print(f"Identical vector rate: {identical_rate:.1%} ({identical_count}/{total_comparisons})")
            
            if identical_rate > 0.5:
                print("🚨 CRITICAL: Most vectors are identical - processor is broken!")
            elif identical_rate > 0:
                print("⚠️  Some vectors are identical - partial issues detected")
            else:
                print("✅ All vectors are unique - processor working correctly")
    
    print(f"\n🔧 NEXT STEPS:")
    print("1. Fix the behavioral processor feature extraction methods")
    print("2. Ensure different inputs generate different vectors") 
    print("3. Re-test anomaly detection with fixed vectors")
    print("4. Integrate GNN layer for enhanced anomaly detection")

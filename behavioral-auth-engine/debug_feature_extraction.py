#!/usr/bin/env python3

import numpy as np
import sys
import os
import traceback

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from core.enhanced_behavioral_processor import EnhancedBehavioralProcessor

def test_feature_extraction_debug():
    """Debug the feature extraction process step by step"""
    
    print("🔧 Debugging Enhanced Behavioral Processor Feature Extraction")
    print("=" * 60)
    
    # Initialize processor
    processor = EnhancedBehavioralProcessor()
    
    # Test data in touch_sequence format
    test_data = {
        "user_id": "test_user",
        "session_id": "debug_session",
        "logs": [
            {
                "event_type": "touch_sequence",
                "timestamp": "2024-01-10T10:30:00Z",
                "data": {
                    "touch_events": [
                        {"coordinates": [100, 200], "pressure": 0.7, "duration": 120}
                    ],
                    "accelerometer": [
                        {"x": 0.1, "y": 0.2, "z": 9.8}
                    ],
                    "gyroscope": [
                        {"x": 0.01, "y": 0.02, "z": 0.01}
                    ],
                    "scroll": {
                        "velocity": 150,
                        "delta_y": 50
                    }
                }
            }
        ]
    }
    
    print("📋 Test data structure:")
    print(f"   Logs count: {len(test_data['logs'])}")
    print(f"   First event type: {test_data['logs'][0]['event_type']}")
    print(f"   Touch events: {len(test_data['logs'][0]['data']['touch_events'])}")
    
    # Test the process_behavioral_logs method directly
    try:
        logs = test_data['logs']
        print(f"\n🧪 Testing process_behavioral_logs with {len(logs)} logs")
        
        # Organize events by type (like the processor does)
        from collections import defaultdict
        events_by_type = defaultdict(list)
        for log in logs:
            event_type = log.get('event_type', '')
            events_by_type[event_type].append(log)
        
        print(f"   Events organized: {dict(events_by_type)}")
        
        # Test each feature extraction method individually
        print(f"\n🔍 Testing individual feature extraction methods:")
        
        try:
            print("   1. Testing _extract_touch_features...")
            touch_features = processor._extract_touch_features(events_by_type)
            print(f"      ✅ Success: {list(touch_features.keys())}")
            print(f"      Pressure stats: {touch_features.get('pressure_stats', [])[:3]}...")
        except Exception as e:
            print(f"      ❌ Error in touch features: {e}")
            traceback.print_exc()
        
        try:
            print("   2. Testing _extract_motion_features...")
            motion_features = processor._extract_motion_features(events_by_type)
            print(f"      ✅ Success: {list(motion_features.keys())}")
            print(f"      Accel stats: {motion_features.get('accel_stats', [])[:3]}...")
        except Exception as e:
            print(f"      ❌ Error in motion features: {e}")
            traceback.print_exc()
        
        try:
            print("   3. Testing _extract_scroll_features...")
            scroll_features = processor._extract_scroll_features(events_by_type)
            print(f"      ✅ Success: {list(scroll_features.keys())}")
            print(f"      Velocity stats: {scroll_features.get('velocity_stats', [])[:3]}...")
        except Exception as e:
            print(f"      ❌ Error in scroll features: {e}")
            traceback.print_exc()
        
        try:
            print("   4. Testing _extract_device_features...")
            device_features = processor._extract_device_features(events_by_type, logs)
            print(f"      ✅ Success: {list(device_features.keys())}")
            print(f"      Session duration: {device_features.get('session_duration', 0)}")
        except Exception as e:
            print(f"      ❌ Error in device features: {e}")
            traceback.print_exc()
        
        try:
            print("   5. Testing _extract_contextual_features...")
            contextual_features = processor._extract_contextual_features(events_by_type, logs)
            print(f"      ✅ Success: {list(contextual_features.keys())}")
        except Exception as e:
            print(f"      ❌ Error in contextual features: {e}")
            traceback.print_exc()
        
        try:
            print("   6. Testing _extract_consistency_features...")
            consistency_features = processor._extract_consistency_features(events_by_type, logs)
            print(f"      ✅ Success: {list(consistency_features.keys())}")
        except Exception as e:
            print(f"      ❌ Error in consistency features: {e}")
            traceback.print_exc()
        
        # Now test the full process
        print(f"\n🎯 Testing full process_behavioral_logs...")
        features = processor.process_behavioral_logs(logs)
        vector = features.to_vector()
        
        print(f"   Vector shape: {vector.shape}")
        print(f"   Vector mean: {np.mean(vector):.6f}")
        print(f"   Non-zero count: {np.count_nonzero(vector)}")
        print(f"   First 10 values: {vector[:10]}")
        
    except Exception as e:
        print(f"❌ FULL PROCESS ERROR: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_feature_extraction_debug()

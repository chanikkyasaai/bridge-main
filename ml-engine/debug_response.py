#!/usr/bin/env python3
"""Debug FAISS → Adaptive → GNN Pipeline with detailed data flow tracing"""

import requests
import json
import time

def debug_pipeline_flow():
    """Debug the complete pipeline flow: FAISS → Adaptive → GNN"""
    print("🔍 DEBUGGING COMPLETE PIPELINE: FAISS → ADAPTIVE → GNN")
    print("="*80)
    print("📊 Tracing data flow through each layer with detailed logging")
    print("="*80)
    
    
    # Test Data 1: Simple behavioral pattern
    simple_test_data = {
        "user_id": "pipeline_debug_user_001",
        "session_id": "simple_session_001",
        "logs": [
            {
                "timestamp": "2024-01-01T10:00:00",
                "event_type": "touch_sequence",
                "data": {
                    "touch_events": [
                        {"x": 150, "y": 200, "pressure": 0.5, "duration": 120},
                        {"x": 160, "y": 210, "pressure": 0.6, "duration": 115},
                        {"x": 155, "y": 205, "pressure": 0.55, "duration": 118}
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
                        {"key": "c", "dwell_time": 110, "pressure": 0.58}
                    ],
                    "typing_rhythm": [95, 110, 105],
                    "inter_key_intervals": [0.12, 0.15, 0.13]
                }
            }
        ]
    }
    
    # Test Data 2: Complex behavioral pattern (more data points)
    complex_test_data = {
        "user_id": "pipeline_debug_user_002", 
        "session_id": "complex_session_001",
        "logs": []
    }
    
    # Generate 15 touch sequences with varied patterns
    for i in range(15):
        complex_test_data["logs"].append({
            "timestamp": f"2024-01-01T10:{i:02d}:00",
            "event_type": "touch_sequence",
            "data": {
                "touch_events": [
                    {"x": 150 + i*5, "y": 200 + i*3, "pressure": 0.5 + i*0.02, "duration": 120 + i*2},
                    {"x": 160 + i*4, "y": 210 + i*2, "pressure": 0.6 + i*0.01, "duration": 115 + i*3},
                    {"x": 155 + i*6, "y": 205 + i*4, "pressure": 0.55 + i*0.015, "duration": 118 + i}
                ],
                "accelerometer": {"x": 0.02 + i*0.001, "y": 0.15 + i*0.005, "z": 9.78 + i*0.002},
                "gyroscope": {"x": 0.001 + i*0.0001, "y": 0.002 + i*0.0002, "z": 0.0015 + i*0.0001}
            }
        })
    
    # Generate 10 keystroke sequences
    for i in range(10):
        complex_test_data["logs"].append({
            "timestamp": f"2024-01-01T10:{i+15:02d}:00", 
            "event_type": "keystroke_sequence",
            "data": {
                "keystrokes": [
                    {"key": chr(97 + i % 26), "dwell_time": 100 + i*5, "pressure": 0.55 + i*0.01},
                    {"key": chr(98 + i % 25), "dwell_time": 120 + i*3, "pressure": 0.6 + i*0.008},
                    {"key": chr(99 + i % 24), "dwell_time": 110 + i*4, "pressure": 0.58 + i*0.012}
                ],
                "typing_rhythm": [95 + i*2, 110 + i*3, 105 + i],
                "inter_key_intervals": [0.12 + i*0.01, 0.15 + i*0.005, 0.13 + i*0.008]
            }
        })
    
    # Test Data 3: Bot-like pattern (very consistent)
    bot_test_data = {
        "user_id": "pipeline_debug_bot_001",
        "session_id": "bot_session_001", 
        "logs": []
    }
    
    # Generate identical touch patterns (bot-like)
    for i in range(10):
        bot_test_data["logs"].append({
            "timestamp": f"2024-01-01T11:{i:02d}:00",
            "event_type": "touch_sequence",
            "data": {
                "touch_events": [
                    {"x": 150, "y": 200, "pressure": 0.5, "duration": 120},  # Identical
                    {"x": 150, "y": 200, "pressure": 0.5, "duration": 120},  # Identical
                    {"x": 150, "y": 200, "pressure": 0.5, "duration": 120}   # Identical
                ],
                "accelerometer": {"x": 0.02, "y": 0.15, "z": 9.78},  # Identical
                "gyroscope": {"x": 0.001, "y": 0.002, "z": 0.0015}   # Identical
            }
        })
    
    test_scenarios = [
        ("🧪 TEST 1: SIMPLE PATTERN", simple_test_data),
        ("🧪 TEST 2: COMPLEX HUMAN PATTERN", complex_test_data),
        ("🧪 TEST 3: BOT-LIKE PATTERN", bot_test_data)
    ]
    
    for test_name, test_data in test_scenarios:
        print(f"\n{test_name}")
        print("="*60)
        print(f"👤 User: {test_data['user_id']}")
        print(f"🔗 Session: {test_data['session_id']}")
        print(f"📊 Behavioral Logs: {len(test_data['logs'])}")
        
        try:
            print(f"\n📤 Sending request to ML Engine pipeline...")
            start_time = time.time()
            
            response = requests.post(
                "http://localhost:8001/analyze-mobile",
                json=test_data,
                timeout=60
            )
            
            response_time = time.time() - start_time
            print(f"📥 Response Status: {response.status_code}")
            print(f"⏱️  Total Processing Time: {response_time:.3f}s")
            
            if response.status_code == 200:
                result = response.json()
                
                print(f"\n🎯 PIPELINE OUTPUT ANALYSIS:")
                print("-" * 40)
                
                # Main results
                print(f"✅ Final Decision: {result.get('decision', 'N/A')}")
                print(f"📊 Risk Score: {result.get('risk_score', 'N/A')}")
                print(f"🔍 Analysis Type: {result.get('analysis_type', 'N/A')}")
                print(f"🧠 Confidence: {result.get('confidence', 'N/A')}")
                print(f"⚡ Processing Time: {result.get('processing_time_ms', 'N/A')}ms")
                
                # Vector analysis
                vector_stats = result.get('vector_stats', {})
                print(f"\n� BEHAVIORAL VECTOR ANALYSIS:")
                print(f"   Length: {vector_stats.get('length', 'N/A')}")
                print(f"   Non-Zero Features: {vector_stats.get('non_zero_count', 'N/A')}/{vector_stats.get('length', 'N/A')}")
                print(f"   Data Richness: {vector_stats.get('non_zero_percentage', 'N/A')}%")
                print(f"   Vector Mean: {vector_stats.get('mean', 'N/A')}")
                print(f"   Vector Std: {vector_stats.get('std', 'N/A')}")
                print(f"   Is Meaningful: {vector_stats.get('is_meaningful', 'N/A')}")
                
                # FAISS analysis
                print(f"\n🔍 FAISS LAYER OUTPUT:")
                print(f"   Similarity Score: {result.get('similarity_score', 'N/A')}")
                print(f"   Similar Vectors Found: {len(result.get('similar_vectors', []))}")
                print(f"   Learning Phase: {result.get('learning_phase', 'N/A')}")
                
                # GNN analysis 
                gnn_analysis = result.get('gnn_analysis', {})
                print(f"\n🧠 GNN LAYER OUTPUT:")
                print(f"   Anomaly Score: {gnn_analysis.get('anomaly_score', 'N/A')}")
                print(f"   GNN Confidence: {gnn_analysis.get('gnn_confidence', 'N/A')}")
                print(f"   Risk Adjustment: {gnn_analysis.get('risk_adjustment', 'N/A')}")
                print(f"   Anomaly Types: {gnn_analysis.get('anomaly_types', [])}")
                
                # Risk factors
                risk_factors = result.get('risk_factors', [])
                if risk_factors:
                    print(f"\n⚠️  RISK FACTORS DETECTED:")
                    for i, factor in enumerate(risk_factors, 1):
                        print(f"   {i}. {factor}")
                
                # Vector sample (first 15 values for analysis)
                session_vector = result.get('session_vector', [])
                if session_vector:
                    print(f"\n🔢 BEHAVIORAL VECTOR SAMPLE (First 15 values):")
                    sample_values = session_vector[:15]
                    print(f"   {[round(v, 6) for v in sample_values]}")
                    
                    # Check for patterns
                    non_zero_sample = [v for v in sample_values if v != 0.0]
                    zero_sample = [v for v in sample_values if v == 0.0]
                    print(f"   Non-zeros in sample: {len(non_zero_sample)}/15")
                    print(f"   Zeros in sample: {len(zero_sample)}/15")
                    
                    if len(non_zero_sample) > 0:
                        print(f"   Non-zero range: {min(non_zero_sample):.6f} to {max(non_zero_sample):.6f}")
                
                # Check if this looks like bot behavior
                bot_indicators = []
                if vector_stats.get('non_zero_percentage', 0) < 40:
                    bot_indicators.append("Low feature diversity")
                if gnn_analysis.get('anomaly_score', 0) == 0.0:
                    bot_indicators.append("No GNN anomaly detected") 
                if result.get('similarity_score', 0) == 0.0:
                    bot_indicators.append("No FAISS similarity")
                
                if bot_indicators:
                    print(f"\n🤖 POTENTIAL BOT INDICATORS:")
                    for indicator in bot_indicators:
                        print(f"   - {indicator}")
                
                print(f"\n" + "="*60)
                
            else:
                print(f"❌ Error Response: {response.text}")
                
        except Exception as e:
            print(f"❌ Exception during pipeline test: {e}")
        
        print("\n" + "⏸️ " * 20)  # Separator between tests
        time.sleep(1)  # Brief pause between tests

def run_focused_layer_analysis():
    """Run additional focused analysis on each layer"""
    print(f"\n🎯 FOCUSED LAYER-BY-LAYER ANALYSIS")
    print("="*60)
    
    # This will show the detailed logging from each layer
    test_data = {
        "user_id": "layer_analysis_user",
        "session_id": "layer_analysis_session", 
        "logs": [
            {
                "timestamp": "2024-01-01T12:00:00",
                "event_type": "touch_sequence",
                "data": {
                    "touch_events": [
                        {"x": 100, "y": 150, "pressure": 0.4, "duration": 100},
                        {"x": 120, "y": 170, "pressure": 0.6, "duration": 130},
                        {"x": 110, "y": 160, "pressure": 0.5, "duration": 115},
                        {"x": 125, "y": 175, "pressure": 0.7, "duration": 140},
                        {"x": 105, "y": 155, "pressure": 0.45, "duration": 105}
                    ],
                    "accelerometer": {"x": 0.05, "y": 0.12, "z": 9.81},
                    "gyroscope": {"x": 0.003, "y": 0.004, "z": 0.002}
                }
            },
            {
                "timestamp": "2024-01-01T12:00:02", 
                "event_type": "keystroke_sequence",
                "data": {
                    "keystrokes": [
                        {"key": "h", "dwell_time": 85, "pressure": 0.5},
                        {"key": "e", "dwell_time": 95, "pressure": 0.55},
                        {"key": "l", "dwell_time": 90, "pressure": 0.52},
                        {"key": "l", "dwell_time": 88, "pressure": 0.53},
                        {"key": "o", "dwell_time": 92, "pressure": 0.54}
                    ],
                    "typing_rhythm": [88, 92, 89, 91, 90],
                    "inter_key_intervals": [0.11, 0.13, 0.12, 0.14, 0.13]
                }
            }
        ]
    }
    
    print("📤 Sending focused analysis request...")
    print("🔍 Check backend and ML Engine logs for detailed layer-by-layer processing...")
    
    try:
        response = requests.post(
            "http://localhost:8001/analyze-mobile",
            json=test_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Focused analysis completed successfully")
            print(f"📊 Final Risk Score: {result.get('risk_score')}")
            print(f"🧠 Final Confidence: {result.get('confidence')}")
            print(f"🎯 Final Decision: {result.get('decision')}")
        else:
            print(f"❌ Error in focused analysis: {response.text}")
    except Exception as e:
        print(f"❌ Exception in focused analysis: {e}")

if __name__ == "__main__":
    debug_pipeline_flow()
    run_focused_layer_analysis()
    
    print(f"\n🏆 PIPELINE DEBUGGING COMPLETE!")
    print("="*60)
    print("📋 Check the above output for detailed pipeline flow analysis")
    print("📋 Check backend logs for detailed FAISS → Adaptive → GNN processing steps")

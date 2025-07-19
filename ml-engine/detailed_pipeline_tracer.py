#!/usr/bin/env python3
"""
🔬 DETAILED PIPELINE FLOW TRACER
===============================
Traces data through FAISS → Adaptive → GNN with detailed inspection at each step
"""

import requests
import json
import time
import numpy as np
from typing import Dict, List, Any

def create_detailed_test_data() -> Dict[str, Any]:
    """Create rich test data for pipeline tracing"""
    return {
        "user_id": "pipeline_tracer_user",
        "session_id": "detailed_trace_session",
        "logs": [
            # Rich touch sequence data
            {
                "timestamp": "2024-01-01T10:00:00",
                "event_type": "touch_sequence",
                "data": {
                    "touch_events": [
                        {"x": 120, "y": 180, "pressure": 0.45, "duration": 95},
                        {"x": 135, "y": 195, "pressure": 0.52, "duration": 110},
                        {"x": 142, "y": 188, "pressure": 0.48, "duration": 105},
                        {"x": 128, "y": 175, "pressure": 0.55, "duration": 120},
                        {"x": 155, "y": 202, "pressure": 0.42, "duration": 98}
                    ],
                    "accelerometer": {"x": 0.035, "y": 0.125, "z": 9.79},
                    "gyroscope": {"x": 0.0025, "y": 0.0035, "z": 0.0018}
                }
            },
            # Rich keystroke sequence data
            {
                "timestamp": "2024-01-01T10:00:02",
                "event_type": "keystroke_sequence", 
                "data": {
                    "keystrokes": [
                        {"key": "h", "dwell_time": 92, "pressure": 0.48},
                        {"key": "e", "dwell_time": 88, "pressure": 0.52},
                        {"key": "l", "dwell_time": 95, "pressure": 0.45},
                        {"key": "l", "dwell_time": 90, "pressure": 0.50},
                        {"key": "o", "dwell_time": 98, "pressure": 0.47}
                    ],
                    "typing_rhythm": [90, 92, 88, 94, 96],
                    "inter_key_intervals": [0.11, 0.14, 0.13, 0.12, 0.15]
                }
            },
            # Additional complex touch patterns
            {
                "timestamp": "2024-01-01T10:00:04",
                "event_type": "touch_sequence",
                "data": {
                    "touch_events": [
                        {"x": 200, "y": 250, "pressure": 0.65, "duration": 140},
                        {"x": 195, "y": 245, "pressure": 0.68, "duration": 135},
                        {"x": 205, "y": 255, "pressure": 0.62, "duration": 145}
                    ],
                    "accelerometer": {"x": 0.08, "y": 0.18, "z": 9.83},
                    "gyroscope": {"x": 0.008, "y": 0.012, "z": 0.006}
                }
            },
            # Complex keystroke patterns
            {
                "timestamp": "2024-01-01T10:00:06",
                "event_type": "keystroke_sequence",
                "data": {
                    "keystrokes": [
                        {"key": "w", "dwell_time": 105, "pressure": 0.58},
                        {"key": "o", "dwell_time": 98, "pressure": 0.55},
                        {"key": "r", "dwell_time": 102, "pressure": 0.62},
                        {"key": "l", "dwell_time": 95, "pressure": 0.52},
                        {"key": "d", "dwell_time": 108, "pressure": 0.60}
                    ],
                    "typing_rhythm": [98, 100, 105, 92, 110],
                    "inter_key_intervals": [0.13, 0.16, 0.14, 0.18, 0.12]
                }
            }
        ]
    }

def analyze_vector_characteristics(vector: List[float]) -> Dict[str, Any]:
    """Analyze detailed characteristics of a behavioral vector"""
    if not vector:
        return {"error": "Empty vector"}
    
    vector_array = np.array(vector)
    non_zero_mask = vector_array != 0.0
    non_zeros = vector_array[non_zero_mask]
    zeros = vector_array[~non_zero_mask]
    
    analysis = {
        "total_features": len(vector),
        "non_zero_count": len(non_zeros),
        "zero_count": len(zeros),
        "data_density": len(non_zeros) / len(vector) * 100,
        "vector_stats": {
            "mean": float(np.mean(vector_array)),
            "std": float(np.std(vector_array)),
            "min": float(np.min(vector_array)),
            "max": float(np.max(vector_array)),
            "median": float(np.median(vector_array))
        },
        "non_zero_stats": {
            "mean": float(np.mean(non_zeros)) if len(non_zeros) > 0 else 0.0,
            "std": float(np.std(non_zeros)) if len(non_zeros) > 0 else 0.0,
            "min": float(np.min(non_zeros)) if len(non_zeros) > 0 else 0.0,
            "max": float(np.max(non_zeros)) if len(non_zeros) > 0 else 0.0,
            "range": float(np.max(non_zeros) - np.min(non_zeros)) if len(non_zeros) > 0 else 0.0
        },
        "distribution_analysis": {
            "very_small_values": len(vector_array[(vector_array > 0) & (vector_array < 0.001)]),
            "small_values": len(vector_array[(vector_array >= 0.001) & (vector_array < 0.01)]),
            "medium_values": len(vector_array[(vector_array >= 0.01) & (vector_array < 0.1)]),
            "large_values": len(vector_array[vector_array >= 0.1]),
        },
        "feature_patterns": {
            "identical_values": len(vector) - len(set(vector)),
            "unique_values": len(set(vector)),
            "repeated_patterns": "detected" if len(vector) - len(set(vector)) > 5 else "minimal"
        }
    }
    
    return analysis

def trace_pipeline_with_multiple_sessions():
    """Trace the pipeline with multiple sessions to see learning progression"""
    print("🔬 DETAILED PIPELINE FLOW TRACER")
    print("="*80)
    print("📊 Tracing FAISS → Adaptive → GNN with detailed inspection")
    print("="*80)
    
    base_user_id = "pipeline_tracer"
    sessions_to_test = 5
    
    all_results = []
    
    for session_num in range(1, sessions_to_test + 1):
        print(f"\n🧪 SESSION {session_num}/{sessions_to_test}: DETAILED ANALYSIS")
        print("="*60)
        
        # Create test data with slight variations per session
        test_data = create_detailed_test_data()
        test_data["user_id"] = f"{base_user_id}_{session_num:03d}"
        test_data["session_id"] = f"trace_session_{session_num:03d}"
        
        # Add session-specific variations to show learning
        for log_entry in test_data["logs"]:
            if log_entry["event_type"] == "touch_sequence":
                # Add slight variations in pressure and timing
                for touch in log_entry["data"]["touch_events"]:
                    touch["pressure"] += (session_num - 1) * 0.01
                    touch["duration"] += (session_num - 1) * 2
            elif log_entry["event_type"] == "keystroke_sequence":
                # Add variations in typing rhythm
                for i, rhythm in enumerate(log_entry["data"]["typing_rhythm"]):
                    log_entry["data"]["typing_rhythm"][i] += (session_num - 1) * 1
        
        print(f"👤 User ID: {test_data['user_id']}")
        print(f"🔗 Session ID: {test_data['session_id']}")
        print(f"📊 Behavioral Logs: {len(test_data['logs'])}")
        
        # Send request
        try:
            print(f"\n📤 SENDING REQUEST TO ML PIPELINE...")
            start_time = time.time()
            
            response = requests.post(
                "http://localhost:8001/analyze-mobile",
                json=test_data,
                timeout=60
            )
            
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                
                print(f"✅ Response Status: {response.status_code}")
                print(f"⏱️  Total Processing Time: {processing_time:.3f}s")
                
                # DETAILED PIPELINE ANALYSIS
                print(f"\n🎯 PHASE 1: BEHAVIORAL VECTOR ANALYSIS")
                print("-" * 50)
                
                session_vector = result.get("session_vector", [])
                if session_vector:
                    vector_analysis = analyze_vector_characteristics(session_vector)
                    
                    print(f"📈 Vector Characteristics:")
                    print(f"   Total Features: {vector_analysis['total_features']}")
                    print(f"   Non-Zero Features: {vector_analysis['non_zero_count']} ({vector_analysis['data_density']:.1f}%)")
                    print(f"   Zero Features: {vector_analysis['zero_count']}")
                    
                    print(f"\n📊 Statistical Analysis:")
                    stats = vector_analysis['vector_stats']
                    print(f"   Mean: {stats['mean']:.6f}")
                    print(f"   Std: {stats['std']:.6f}")
                    print(f"   Range: {stats['min']:.6f} to {stats['max']:.6f}")
                    print(f"   Median: {stats['median']:.6f}")
                    
                    print(f"\n🎨 Distribution Analysis:")
                    dist = vector_analysis['distribution_analysis']
                    print(f"   Very Small (0-0.001): {dist['very_small_values']} features")
                    print(f"   Small (0.001-0.01): {dist['small_values']} features")
                    print(f"   Medium (0.01-0.1): {dist['medium_values']} features")
                    print(f"   Large (≥0.1): {dist['large_values']} features")
                    
                    print(f"\n🔍 Pattern Analysis:")
                    patterns = vector_analysis['feature_patterns']
                    print(f"   Unique Values: {patterns['unique_values']}")
                    print(f"   Repeated Values: {patterns['identical_values']}")
                    print(f"   Pattern Assessment: {patterns['repeated_patterns']}")
                
                print(f"\n🎯 PHASE 2: FAISS LAYER ANALYSIS")
                print("-" * 50)
                print(f"   Similarity Score: {result.get('similarity_score', 'N/A')}")
                print(f"   Learning Phase: {result.get('learning_phase', 'N/A')}")
                print(f"   Similar Vectors: {len(result.get('similar_vectors', []))}")
                
                print(f"\n🎯 PHASE 3: GNN LAYER ANALYSIS")
                print("-" * 50)
                gnn_analysis = result.get("gnn_analysis", {})
                print(f"   Anomaly Score: {gnn_analysis.get('anomaly_score', 'N/A')}")
                print(f"   GNN Confidence: {gnn_analysis.get('gnn_confidence', 'N/A')}")
                print(f"   Risk Adjustment: {gnn_analysis.get('risk_adjustment', 'N/A')}")
                print(f"   Anomaly Types: {gnn_analysis.get('anomaly_types', [])}")
                
                print(f"\n🎯 PHASE 4: FINAL DECISION ANALYSIS")
                print("-" * 50)
                print(f"   Final Decision: {result.get('decision', 'N/A')}")
                print(f"   Risk Score: {result.get('risk_score', 'N/A')}")
                print(f"   Confidence: {result.get('confidence', 'N/A')}")
                print(f"   Analysis Type: {result.get('analysis_type', 'N/A')}")
                
                risk_factors = result.get("risk_factors", [])
                if risk_factors:
                    print(f"\n⚠️  Risk Factors:")
                    for i, factor in enumerate(risk_factors, 1):
                        print(f"      {i}. {factor}")
                
                # Store results for progression analysis
                all_results.append({
                    "session_num": session_num,
                    "user_id": test_data["user_id"],
                    "session_id": test_data["session_id"],
                    "decision": result.get("decision"),
                    "risk_score": result.get("risk_score"),
                    "confidence": result.get("confidence"),
                    "similarity_score": result.get("similarity_score"),
                    "vector_analysis": vector_analysis if 'vector_analysis' in locals() else None,
                    "processing_time": processing_time
                })
                
            else:
                print(f"❌ Error Response: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ Exception: {e}")
        
        print("\n" + "⏸️ " * 25)
        time.sleep(1)  # Brief pause between sessions
    
    # PROGRESSION ANALYSIS
    print(f"\n📈 LEARNING PROGRESSION ANALYSIS")
    print("="*60)
    
    if all_results:
        for i, result in enumerate(all_results):
            print(f"Session {result['session_num']}: Decision={result['decision']}, Risk={result['risk_score']:.4f}, Confidence={result['confidence']:.2f}, Processing={result['processing_time']:.2f}s")
        
        # Check if confidence is improving
        confidences = [r['confidence'] for r in all_results if r['confidence'] is not None]
        if len(confidences) > 1:
            print(f"\n📊 Confidence Trend: {confidences[0]:.2f} → {confidences[-1]:.2f}")
            if confidences[-1] > confidences[0]:
                print("✅ System is learning and confidence is building!")
            else:
                print("⚠️  Confidence not building as expected")
    
    return all_results

if __name__ == "__main__":
    print("🚀 Starting Detailed Pipeline Flow Tracing...")
    results = trace_pipeline_with_multiple_sessions()
    
    print(f"\n🏆 PIPELINE TRACING COMPLETE!")
    print("="*60)
    print(f"📊 Sessions Analyzed: {len(results)}")
    print("📋 Check above output for detailed FAISS → Adaptive → GNN flow analysis")
    print("📋 Each phase shows detailed data transformations and decision logic")

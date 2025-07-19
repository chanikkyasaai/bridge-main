#!/usr/bin/env python3
"""
🚨 GNN BOT DETECTION TEST
========================
Specifically test GNN with obvious bot patterns to see if it detects them
"""

import requests
import json

def test_obvious_bot_patterns():
    """Test GNN with VERY obvious bot patterns"""
    print("🚨 GNN BOT DETECTION TEST")
    print("="*60)
    
    # Create EXTREMELY obvious bot pattern - all identical touches
    obvious_bot_data = {
        "user_id": "obvious_bot_test",
        "session_id": "obvious_bot_session",
        "logs": []
    }
    
    # Generate 20 IDENTICAL touch sequences (perfect bot behavior)
    for i in range(20):
        obvious_bot_data["logs"].append({
            "timestamp": f"2024-01-01T10:{i:02d}:00",
            "event_type": "touch_sequence",
            "data": {
                "touch_events": [
                    {"x": 100, "y": 100, "pressure": 0.5, "duration": 100},  # IDENTICAL
                    {"x": 100, "y": 100, "pressure": 0.5, "duration": 100},  # IDENTICAL
                    {"x": 100, "y": 100, "pressure": 0.5, "duration": 100},  # IDENTICAL
                    {"x": 100, "y": 100, "pressure": 0.5, "duration": 100},  # IDENTICAL
                    {"x": 100, "y": 100, "pressure": 0.5, "duration": 100}   # IDENTICAL
                ],
                "accelerometer": {"x": 0.01, "y": 0.01, "z": 9.80},  # IDENTICAL
                "gyroscope": {"x": 0.001, "y": 0.001, "z": 0.001}    # IDENTICAL
            }
        })
    
    print(f"🤖 Testing OBVIOUS bot with {len(obvious_bot_data['logs'])} identical touch sequences")
    print(f"   Each sequence has 5 identical touches at (100,100)")
    print(f"   All pressures: 0.5, all durations: 100ms")
    print(f"   All sensor readings identical")
    
    try:
        response = requests.post(
            "http://localhost:8001/analyze-mobile",
            json=obvious_bot_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n📊 GNN ANALYSIS RESULTS:")
            print(f"   Decision: {result.get('decision')}")
            print(f"   Risk Score: {result.get('risk_score')}")
            print(f"   Confidence: {result.get('confidence')}")
            
            gnn_analysis = result.get('gnn_analysis', {})
            print(f"\n🧠 GNN Layer Results:")
            print(f"   Anomaly Score: {gnn_analysis.get('anomaly_score')}")
            print(f"   GNN Confidence: {gnn_analysis.get('gnn_confidence')}")
            print(f"   Anomaly Types: {gnn_analysis.get('anomaly_types', [])}")
            print(f"   Risk Adjustment: {gnn_analysis.get('risk_adjustment')}")
            
            # Analyze the behavioral vector
            vector_stats = result.get('vector_stats', {})
            print(f"\n📈 Vector Analysis:")
            print(f"   Non-Zero Features: {vector_stats.get('non_zero_count')}/90")
            print(f"   Data Richness: {vector_stats.get('non_zero_percentage')}%")
            print(f"   Vector Mean: {vector_stats.get('mean')}")
            print(f"   Vector Std: {vector_stats.get('std')}")
            
            # Check if GNN detected the obvious bot pattern
            anomaly_score = gnn_analysis.get('anomaly_score', 0)
            if anomaly_score > 0.5:
                print(f"\n✅ GNN CORRECTLY DETECTED bot pattern! Score: {anomaly_score}")
            elif anomaly_score > 0.0:
                print(f"\n⚠️  GNN detected some anomaly but score is low: {anomaly_score}")
            else:
                print(f"\n❌ GNN FAILED to detect obvious bot pattern! Score: {anomaly_score}")
                print("🔍 This indicates the GNN detection logic needs debugging")
                
        else:
            print(f"❌ Request failed: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")

def test_human_vs_bot_comparison():
    """Test human pattern vs bot pattern side by side"""
    print(f"\n🧪 HUMAN vs BOT COMPARISON TEST")
    print("="*60)
    
    # Human pattern (varied)
    human_data = {
        "user_id": "human_test",
        "session_id": "human_session",
        "logs": []
    }
    
    # Generate 10 VARIED human touch sequences
    for i in range(10):
        human_data["logs"].append({
            "timestamp": f"2024-01-01T11:{i:02d}:00",
            "event_type": "touch_sequence",
            "data": {
                "touch_events": [
                    {"x": 100 + i*5, "y": 100 + i*3, "pressure": 0.4 + i*0.02, "duration": 95 + i*2},
                    {"x": 110 + i*4, "y": 105 + i*2, "pressure": 0.5 + i*0.015, "duration": 100 + i*3},
                    {"x": 95 + i*6, "y": 108 + i*4, "pressure": 0.45 + i*0.01, "duration": 98 + i}
                ],
                "accelerometer": {"x": 0.01 + i*0.002, "y": 0.02 + i*0.001, "z": 9.78 + i*0.01},
                "gyroscope": {"x": 0.001 + i*0.0001, "y": 0.002 + i*0.0002, "z": 0.0015 + i*0.00005}
            }
        })
    
    # Bot pattern (identical)  
    bot_data = {
        "user_id": "bot_test",
        "session_id": "bot_session",
        "logs": []
    }
    
    # Generate 10 IDENTICAL bot touch sequences
    for i in range(10):
        bot_data["logs"].append({
            "timestamp": f"2024-01-01T12:{i:02d}:00",
            "event_type": "touch_sequence",
            "data": {
                "touch_events": [
                    {"x": 200, "y": 200, "pressure": 0.6, "duration": 120},  # IDENTICAL
                    {"x": 200, "y": 200, "pressure": 0.6, "duration": 120},  # IDENTICAL
                    {"x": 200, "y": 200, "pressure": 0.6, "duration": 120}   # IDENTICAL
                ],
                "accelerometer": {"x": 0.02, "y": 0.03, "z": 9.81},  # IDENTICAL
                "gyroscope": {"x": 0.002, "y": 0.003, "z": 0.002}    # IDENTICAL
            }
        })
    
    print("👤 Testing HUMAN pattern (varied touches, pressures, durations)")
    test_pattern("HUMAN", human_data)
    
    print("\n🤖 Testing BOT pattern (identical touches, pressures, durations)")
    test_pattern("BOT", bot_data)

def test_pattern(pattern_type, test_data):
    """Test a specific pattern"""
    try:
        response = requests.post(
            "http://localhost:8001/analyze-mobile",
            json=test_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            gnn_analysis = result.get('gnn_analysis', {})
            
            anomaly_score = gnn_analysis.get('anomaly_score', 0)
            print(f"   {pattern_type} GNN Anomaly Score: {anomaly_score}")
            print(f"   {pattern_type} Decision: {result.get('decision')}")
            print(f"   {pattern_type} Risk Score: {result.get('risk_score')}")
            
        else:
            print(f"   {pattern_type} request failed: {response.status_code}")
            
    except Exception as e:
        print(f"   {pattern_type} exception: {e}")

if __name__ == "__main__":
    test_obvious_bot_patterns()
    test_human_vs_bot_comparison()
    
    print(f"\n🏁 GNN BOT DETECTION TEST COMPLETE")
    print("="*60)
    print("If GNN anomaly scores are all 0.0, the detection logic needs fixing")

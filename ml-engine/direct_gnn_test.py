#!/usr/bin/env python3
"""
🔬 DIRECT GNN METHOD TEST
========================
Test GNN methods directly without going through the full pipeline
"""

import sys
import os

# Add the path to import the GNN module  
gnn_path = r"c:\Users\Hp\OneDrive\Desktop\bridge\bridge\behavioral-auth-engine\src\layers\gnn_anomaly_detector.py"

import importlib.util
spec = importlib.util.spec_from_file_location("gnn_anomaly_detector", gnn_path)
gnn_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gnn_module)
GNNAnomalyDetector = gnn_module.GNNAnomalyDetector
import numpy as np
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

def test_automation_detection_directly():
    """Test the automation detection method directly"""
    print("🔬 DIRECT GNN AUTOMATION DETECTION TEST")
    print("="*60)
    
    # Create GNN detector
    gnn_detector = GNNAnomalyDetector()
    
    # Create obvious bot behavioral logs
    obvious_bot_logs = [
        {
            "timestamp": "2024-01-01T10:00:00",
            "event_type": "touch_sequence",
            "data": {
                "touch_events": [
                    {"x": 100, "y": 100, "pressure": 0.5, "duration": 100},
                    {"x": 100, "y": 100, "pressure": 0.5, "duration": 100},
                    {"x": 100, "y": 100, "pressure": 0.5, "duration": 100},
                    {"x": 100, "y": 100, "pressure": 0.5, "duration": 100},
                    {"x": 100, "y": 100, "pressure": 0.5, "duration": 100}
                ]
            }
        },
        {
            "timestamp": "2024-01-01T10:00:01",
            "event_type": "touch_sequence", 
            "data": {
                "touch_events": [
                    {"x": 100, "y": 100, "pressure": 0.5, "duration": 100},
                    {"x": 100, "y": 100, "pressure": 0.5, "duration": 100},
                    {"x": 100, "y": 100, "pressure": 0.5, "duration": 100}
                ]
            }
        }
    ]
    
    # Test automation detection directly
    print("🤖 Testing direct automation detection...")
    automation_score = gnn_detector._detect_automation_patterns(obvious_bot_logs)
    print(f"   Automation Score: {automation_score}")
    
    if automation_score > 0.0:
        print(f"✅ SUCCESS: Automation detected! Score: {automation_score}")
    else:
        print(f"❌ FAILED: No automation detected. Score: {automation_score}")
    
    # Test spatial detection
    print("\n📍 Testing direct spatial detection...")
    spatial_score = gnn_detector._detect_spatial_anomalies(obvious_bot_logs)
    print(f"   Spatial Score: {spatial_score}")
    
    # Test vector detection
    print("\n📊 Testing direct vector detection...")
    # Create a low-variation vector (like a bot would have)
    bot_vector = np.array([0.1] * 20 + [0.0] * 70)  # Low variation
    vector_score = gnn_detector._detect_vector_anomalies(bot_vector)
    print(f"   Vector Score: {vector_score}")
    
    # Test human pattern
    print(f"\n👤 COMPARISON: Testing human pattern...")
    human_logs = [
        {
            "timestamp": "2024-01-01T11:00:00",
            "event_type": "touch_sequence",
            "data": {
                "touch_events": [
                    {"x": 100, "y": 100, "pressure": 0.4, "duration": 95},
                    {"x": 105, "y": 103, "pressure": 0.55, "duration": 108},
                    {"x": 98, "y": 107, "pressure": 0.48, "duration": 102},
                    {"x": 110, "y": 95, "pressure": 0.62, "duration": 115},
                    {"x": 102, "y": 101, "pressure": 0.51, "duration": 99}
                ]
            }
        }
    ]
    
    human_automation_score = gnn_detector._detect_automation_patterns(human_logs)
    print(f"   Human Automation Score: {human_automation_score}")
    
    print(f"\n📊 SUMMARY:")
    print(f"   Bot Automation Score: {automation_score}")
    print(f"   Human Automation Score: {human_automation_score}")
    print(f"   Difference: {automation_score - human_automation_score}")
    
    if automation_score > human_automation_score:
        print(f"✅ SUCCESS: Bot detection working correctly!")
    else:
        print(f"❌ ISSUE: Bot score should be higher than human score")

if __name__ == "__main__":
    test_automation_detection_directly()

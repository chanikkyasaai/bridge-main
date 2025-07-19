#!/usr/bin/env python3
"""
🔬 ISOLATED GNN AUTOMATION LOGIC TEST
=====================================
Test just the automation detection logic without complex imports
"""

import numpy as np
from typing import Dict, List, Any

def detect_automation_patterns(behavioral_logs: List[Dict[str, Any]]) -> float:
    """
    Simplified version of the automation detection logic from GNN
    """
    print("🔍 Starting automation detection...")
    
    if not behavioral_logs:
        print("   No logs provided")
        return 0.0
    
    automation_indicators = []
    
    for i, log in enumerate(behavioral_logs):
        print(f"   Processing log {i+1}/{len(behavioral_logs)}")
        
        if 'data' not in log:
            print(f"   Log {i+1}: No 'data' field found")
            continue
            
        if 'touch_events' not in log['data']:
            print(f"   Log {i+1}: No 'touch_events' found")
            continue
        
        touch_events = log['data']['touch_events']
        print(f"   Log {i+1}: Found {len(touch_events)} touch events")
        
        if len(touch_events) < 3:
            print(f"   Log {i+1}: Not enough touch events for analysis")
            continue
        
        # Extract coordinates, pressures, durations
        coordinates = []
        pressures = []
        durations = []
        
        for j, touch in enumerate(touch_events):
            print(f"   Touch {j+1}: {touch}")
            
            # Extract coordinates
            x = touch.get('x', 0)
            y = touch.get('y', 0)
            coordinates.append([x, y])
            
            # Extract pressure and duration
            pressures.append(touch.get('pressure', 0.5))
            durations.append(touch.get('duration', 100))
        
        print(f"   Coordinates: {coordinates}")
        print(f"   Pressures: {pressures}")
        print(f"   Durations: {durations}")
        
        # Check for identical coordinates
        unique_coords = len(set(tuple(coord) for coord in coordinates))
        total_coords = len(coordinates)
        coord_variation = unique_coords / total_coords if total_coords > 0 else 1.0
        
        print(f"   Coordinate variation: {coord_variation} ({unique_coords}/{total_coords} unique)")
        
        # Check for identical pressures  
        unique_pressures = len(set(pressures))
        pressure_variation = unique_pressures / len(pressures) if pressures else 1.0
        
        print(f"   Pressure variation: {pressure_variation} ({unique_pressures}/{len(pressures)} unique)")
        
        # Check for identical durations
        unique_durations = len(set(durations))
        duration_variation = unique_durations / len(durations) if durations else 1.0
        
        print(f"   Duration variation: {duration_variation} ({unique_durations}/{len(durations)} unique)")
        
        # Calculate automation score for this sequence
        # Low variation = high automation
        automation_score = 1.0 - (coord_variation * pressure_variation * duration_variation)
        print(f"   Sequence automation score: {automation_score}")
        
        # Check if this sequence meets automation threshold
        automation_threshold = 0.8
        if automation_score >= automation_threshold:
            print(f"   ✅ AUTOMATION DETECTED in sequence {i+1}!")
            automation_indicators.append(automation_score)
        else:
            print(f"   ❌ No automation in sequence {i+1}")
    
    print(f"🎯 Automation indicators found: {len(automation_indicators)}")
    print(f"   Scores: {automation_indicators}")
    
    if not automation_indicators:
        print("   No automation patterns detected")
        return 0.0
    
    # Calculate overall automation score
    max_automation = max(automation_indicators)
    avg_automation = sum(automation_indicators) / len(automation_indicators)
    
    # Use weighted combination
    overall_score = 0.7 * max_automation + 0.3 * avg_automation
    
    print(f"📊 FINAL AUTOMATION ANALYSIS:")
    print(f"   Max automation: {max_automation}")
    print(f"   Average automation: {avg_automation}")  
    print(f"   Overall score: {overall_score}")
    
    return overall_score

def test_automation_detection():
    """Test the automation detection with obvious bot patterns"""
    print("🤖 TESTING AUTOMATION DETECTION")
    print("="*60)
    
    # Create extremely obvious bot pattern
    bot_logs = [
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
        }
    ]
    
    print("🔥 Testing with OBVIOUS BOT PATTERN:")
    print("   5 identical touches: [100,100], pressure=0.5, duration=100")
    
    bot_score = detect_automation_patterns(bot_logs)
    
    print(f"\n🎯 RESULT: Bot automation score = {bot_score}")
    
    if bot_score > 0.8:
        print("✅ SUCCESS: Bot detected correctly!")
    elif bot_score > 0.0:
        print(f"⚠️  PARTIAL: Some automation detected but score low ({bot_score})")
    else:
        print("❌ FAILED: No automation detected!")
    
    # Test human pattern
    print(f"\n👤 Testing HUMAN PATTERN for comparison:")
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
    
    human_score = detect_automation_patterns(human_logs)
    
    print(f"\n🎯 COMPARISON RESULTS:")
    print(f"   Bot Score: {bot_score}")
    print(f"   Human Score: {human_score}")
    print(f"   Difference: {bot_score - human_score}")
    
    if bot_score > human_score:
        print("✅ SUCCESS: Bot score > Human score ✓")
    else:
        print("❌ ISSUE: Bot score should be higher than human score")

if __name__ == "__main__":
    test_automation_detection()

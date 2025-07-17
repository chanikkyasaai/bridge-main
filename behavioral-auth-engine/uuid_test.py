#!/usr/bin/env python3
"""
Quick test for Phase 1/2 system with proper UUID handling
"""

import asyncio
import json
import uuid
import requests
from datetime import datetime

ML_ENGINE_URL = "http://127.0.0.1:8001"

# Generate a valid UUID for testing
TEST_USER_ID = str(uuid.uuid4())
TEST_SESSION_BASE = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def test_behavioral_features():
    """Generate realistic behavioral features for testing"""
    return {
        # Typing features (25 dimensions)
        "typing_speed": 65.5,
        "keystroke_intervals": [0.12, 0.15, 0.13, 0.18, 0.11],
        "typing_rhythm_variance": 0.045,
        "backspace_frequency": 0.08,
        "shift_key_usage": 0.15,
        "space_bar_timing": [0.25, 0.23, 0.27],
        "common_bigram_speed": {"th": 0.11, "er": 0.13, "on": 0.12},
        "error_correction_patterns": ["backspace", "select_all"],
        "typing_acceleration": 0.02,
        "pause_patterns": [1.2, 0.8, 2.1],
        "finger_transition_speed": {"left_to_right": 0.14, "same_hand": 0.09},
        "caps_lock_behavior": 0.01,
        "number_row_usage": 0.12,
        "special_char_timing": 0.18,
        "word_completion_speed": 0.95,
        
        # Mouse features (25 dimensions)
        "mouse_movement_velocity": [150.2, 180.5, 165.8],
        "click_patterns": {"single": 0.85, "double": 0.12, "right": 0.03},
        "scroll_behavior": {"vertical": 0.75, "horizontal": 0.25},
        "mouse_acceleration": 0.15,
        "cursor_trajectory_smoothness": 0.82,
        "hover_duration": 1.2,
        "drag_and_drop_patterns": ["smooth", "jerky"],
        "mouse_pressure": 0.6,
        "click_hold_duration": 0.15,
        "movement_efficiency": 0.78,
        "tremor_patterns": 0.03,
        "directional_preference": {"horizontal": 0.55, "vertical": 0.45},
        "precision_tasks_accuracy": 0.89,
        "gesture_fluidity": 0.76,
        "mouse_rest_positions": [(500, 300), (450, 280)],
        
        # Navigation features (25 dimensions)
        "page_dwell_time": [30.5, 45.2, 28.8],
        "scroll_speed": 2.5,
        "navigation_patterns": ["linear", "jump", "back_forward"],
        "menu_interaction_style": "hover",
        "tab_switching_frequency": 0.15,
        "bookmark_usage": 0.08,
        "search_behavior": {"query_length": 4.2, "refinement_rate": 0.3},
        "link_click_precision": 0.92,
        "form_filling_speed": 1.8,
        "error_recovery_time": 2.1,
        "navigation_efficiency": 0.84,
        "breadcrumb_usage": 0.25,
        "back_button_reliance": 0.18,
        "keyboard_shortcuts": 0.35,
        "multitasking_patterns": ["sequential", "parallel"],
        
        # Context features (15 dimensions)
        "session_duration": 25.5,
        "time_of_day": "14:30",
        "device_type": "desktop",
        "screen_resolution": "1920x1080",
        "browser_type": "chrome",
        "network_latency": 45.2,
        "geolocation_consistency": 0.98,
        "ip_address_stability": 1.0,
        "user_agent_consistency": 1.0,
        "timezone_behavior": "consistent",
        "login_frequency": 0.85,
        "concurrent_sessions": 1,
        "device_fingerprint_match": 0.97,
        "behavioral_consistency": 0.88,
        "anomaly_indicators": []
    }

async def main():
    print("🔄 Testing Phase 1/2 Behavioral Authentication System")
    print(f"📋 Test User ID: {TEST_USER_ID}")
    print(f"📋 Test Session Base: {TEST_SESSION_BASE}")
    
    # Test 1: Health check
    print("\n=== Test 1: ML Engine Health Check ===")
    try:
        response = requests.get(f"{ML_ENGINE_URL}/")
        if response.status_code == 200:
            print("✅ ML Engine health check passed")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Cannot connect to ML Engine: {e}")
        return
        
    # Test 2: Database health
    print("\n=== Test 2: Database Health Check ===")
    try:
        response = requests.get(f"{ML_ENGINE_URL}/health/database")
        if response.status_code == 200:
            print("✅ Database health check passed")
        else:
            print(f"❌ Database health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Database health check error: {e}")
        
    # Test 3: Start learning phase session
    print("\n=== Test 3: Learning Phase Session Start ===")
    try:
        session_id = f"{TEST_SESSION_BASE}_learning"
        response = requests.post(f"{ML_ENGINE_URL}/session/start", json={
            "user_id": TEST_USER_ID,
            "session_id": session_id
        })
        print(f"Session start response: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Session started: {result}")
        else:
            print(f"❌ Session start failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Session start error: {e}")
        
    # Test 4: Behavioral analysis (learning phase)
    print("\n=== Test 4: Learning Phase Behavioral Analysis ===")
    try:
        # Create behavioral events in the correct format
        behavioral_events = [
            {
                "event_type": "keypress",
                "timestamp": datetime.now().isoformat(),
                "data": {
                    "key": "a",
                    "press_duration": 120,
                    "interval_since_last": 150
                }
            },
            {
                "event_type": "mouse_click", 
                "timestamp": datetime.now().isoformat(),
                "data": {
                    "button": "left",
                    "position": {"x": 100, "y": 200},
                    "click_duration": 80
                }
            },
            {
                "event_type": "navigation",
                "timestamp": datetime.now().isoformat(), 
                "data": {
                    "url": "/dashboard",
                    "dwell_time": 2500,
                    "scroll_behavior": {"vertical": 250, "horizontal": 0}
                }
            }
        ]
        
        response = requests.post(f"{ML_ENGINE_URL}/analyze", json={
            "user_id": TEST_USER_ID,
            "session_id": session_id,
            "events": behavioral_events
        })
        print(f"Analysis response: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Learning phase analysis: {json.dumps(result, indent=2)}")
        else:
            print(f"❌ Analysis failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        
    # Test 5: Check learning progress
    print("\n=== Test 5: Learning Progress Check ===")
    try:
        response = requests.get(f"{ML_ENGINE_URL}/user/{TEST_USER_ID}/learning-progress")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Learning progress: {json.dumps(result, indent=2)}")
        else:
            print(f"❌ Learning progress failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Learning progress error: {e}")
        
    # Test 6: Statistics
    print("\n=== Test 6: System Statistics ===")
    try:
        response = requests.get(f"{ML_ENGINE_URL}/statistics")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ System statistics: {json.dumps(result, indent=2)}")
        else:
            print(f"❌ Statistics failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Statistics error: {e}")

if __name__ == "__main__":
    asyncio.run(main())

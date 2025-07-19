#!/usr/bin/env python3
"""
🧪 DIRECT API CALL TEST
======================
Test the ML Engine API directly to see what's in the response
"""

import requests
import json

def test_direct_api_response():
    """Test the ML Engine API and inspect the full response"""
    print("🔍 DIRECT API RESPONSE INSPECTION")
    print("="*50)
    
    # Create bot data
    bot_data = {
        "user_id": "api_test_bot",
        "session_id": "api_test_session", 
        "logs": [
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
                    ],
                    "accelerometer": {"x": 0.01, "y": 0.01, "z": 9.80},
                    "gyroscope": {"x": 0.001, "y": 0.001, "z": 0.001}
                }
            }
        ]
    }
    
    try:
        print(f"📤 Sending request to ML Engine API...")
        response = requests.post(
            "http://localhost:8001/analyze-mobile",
            json=bot_data,
            timeout=30
        )
        
        print(f"📥 Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n📋 FULL RESPONSE KEYS:")
            for key in result.keys():
                print(f"   - {key}: {type(result[key])}")
            
            print(f"\n🔍 DETAILED ANALYSIS:")
            print(f"   Decision: {result.get('decision')}")
            print(f"   Risk Score: {result.get('risk_score')}")
            print(f"   Confidence: {result.get('confidence')}")
            
            # Check if session_vector exists in response
            session_vector = result.get('session_vector')
            print(f"\n📊 SESSION VECTOR:")
            print(f"   Exists: {'Yes' if session_vector else 'No'}")
            if session_vector:
                print(f"   Length: {len(session_vector)}")
                print(f"   Non-zero count: {sum(1 for v in session_vector if v != 0)}")
                print(f"   Sample values: {session_vector[:10]}")
            
            # Check GNN analysis
            gnn_analysis = result.get('gnn_analysis')
            print(f"\n🧠 GNN ANALYSIS:")
            print(f"   Exists: {'Yes' if gnn_analysis else 'No'}")
            if gnn_analysis:
                print(f"   Keys: {list(gnn_analysis.keys())}")
                for key, value in gnn_analysis.items():
                    print(f"   {key}: {value}")
            else:
                print(f"   ❌ GNN analysis is missing from response!")
                print(f"   This means the GNN conditional check failed")
        
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")

if __name__ == "__main__":
    test_direct_api_response()

"""
Quick test to verify the FAISS similarity calculation fixes
"""
import requests
import json
from datetime import datetime

def test_single_behavioral_analysis():
    """Test a single behavioral analysis call"""
    
    # Test data
    behavioral_data = {
        "user_id": "test_user_001",
        "session_id": f"test_session_{int(datetime.now().timestamp())}",
        "logs": [
            {
                "timestamp": datetime.now().isoformat(),
                "event_type": "touch_sequence",
                "data": {
                    "touch_events": [
                        {"x": 150, "y": 400, "pressure": 0.6, "duration": 120},
                        {"x": 155, "y": 405, "pressure": 0.65, "duration": 115}
                    ],
                    "accelerometer": {"x": 0.02, "y": 0.15, "z": 9.78},
                    "gyroscope": {"x": 0.001, "y": 0.002, "z": 0.0015}
                }
            },
            {
                "timestamp": datetime.now().isoformat(),
                "event_type": "keystroke_sequence",
                "data": {
                    "keystrokes": [
                        {"key": "1", "dwell_time": 95, "pressure": 0.55},
                        {"key": "2", "dwell_time": 105, "pressure": 0.6}
                    ],
                    "typing_rhythm": [85, 92],
                    "inter_key_intervals": [0.12, 0.15]
                }
            }
        ]
    }
    
    try:
        # Send to ML Engine
        response = requests.post(
            "http://localhost:8001/analyze-mobile",
            json=behavioral_data,
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n=== ANALYSIS RESULT ===")
            print(f"Decision: {result.get('decision')}")
            print(f"Confidence: {result.get('confidence')}")
            print(f"Similarity Score: {result.get('similarity_score')}")
            print(f"Risk Score: {result.get('risk_score')}")
            print(f"Risk Level: {result.get('risk_level')}")
            print(f"Risk Factors: {result.get('risk_factors')}")
            print(f"Learning Phase: {result.get('learning_phase')}")
            print(f"Vector ID: {result.get('vector_id')}")
            
            if 'vector_stats' in result:
                print(f"\n=== VECTOR STATS ===")
                stats = result['vector_stats']
                print(f"Non-zero count: {stats.get('non_zero_count')}/{stats.get('length')}")
                print(f"Non-zero %: {stats.get('non_zero_percentage'):.1f}%")
                print(f"Is meaningful: {stats.get('is_meaningful')}")
            
            return True
        else:
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"Request failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing FAISS fixes...")
    
    # Test 3 sessions for the same user
    for i in range(3):
        print(f"\n--- Session {i+1} ---")
        success = test_single_behavioral_analysis()
        if not success:
            print("❌ Test failed")
            break
        print("✅ Session processed")
    
    print("\n🎉 Quick fix test completed!")

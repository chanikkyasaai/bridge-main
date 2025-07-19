import requests
import json
from datetime import datetime

# Test with different behavioral patterns to create more diverse vectors
behavioral_data = {
    'user_id': 'diverse_test_001',
    'session_id': f'diverse_session_{int(datetime.now().timestamp())}',
    'logs': [
        {
            'timestamp': datetime.now().isoformat(),
            'event_type': 'touch_sequence',
            'data': {
                'touch_events': [
                    {'x': 200, 'y': 500, 'pressure': 0.8, 'duration': 200},  # Different values
                    {'x': 210, 'y': 510, 'pressure': 0.9, 'duration': 180}
                ],
                'accelerometer': {'x': 0.5, 'y': 0.8, 'z': 9.2},  # Different sensor data
                'gyroscope': {'x': 0.1, 'y': 0.15, 'z': 0.08}
            }
        },
        {
            'timestamp': datetime.now().isoformat(),
            'event_type': 'keystroke_sequence', 
            'data': {
                'keystrokes': [
                    {'key': '5', 'dwell_time': 150, 'pressure': 0.75},  # Different typing
                    {'key': '6', 'dwell_time': 160, 'pressure': 0.8},
                    {'key': '7', 'dwell_time': 140, 'pressure': 0.7}
                ],
                'typing_rhythm': [140, 150, 130],
                'inter_key_intervals': [0.2, 0.25, 0.18]
            }
        }
    ]
}

response = requests.post('http://localhost:8001/analyze-mobile', json=behavioral_data, timeout=30)
print('Status:', response.status_code)
if response.status_code == 200:
    result = response.json()
    print('Similarity:', result.get('similarity_score'))
    print('Vector ID:', result.get('vector_id'))
    print('Risk Factors:', result.get('risk_factors'))
    if 'vector_stats' in result:
        stats = result['vector_stats']
        print(f"Non-zero: {stats.get('non_zero_count')}/{stats.get('length')} ({stats.get('non_zero_percentage'):.1f}%)")
else:
    print('Error:', response.text)

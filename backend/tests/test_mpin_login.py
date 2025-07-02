#!/usr/bin/env python3
"""
Test the new MPIN-only login endpoint
"""

import requests
import json

def test_mpin_login():
    """Test the MPIN-only login endpoint"""
    
    print("🧪 Testing MPIN-only login endpoint...")
    
    try:
        response = requests.post('http://localhost:8000/api/v1/auth/mpin-login', json={
            'phone': '9876543210',
            'mpin': '123456',
            'device_id': 'test_device'
        })
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ MPIN-only login endpoint works!")
            print(f"Access token: {data.get('access_token', 'N/A')[:50]}...")
            print(f"Refresh token: {data.get('refresh_token', 'N/A')[:50]}...")
            print(f"Session token: {data.get('session_token', 'N/A')[:50]}...")
            print(f"Session ID: {data.get('session_id', 'N/A')}")
            print(f"Behavioral logging: {data.get('behavioral_logging', 'N/A')}")
            print(f"Message: {data.get('message', 'N/A')}")
            return True
        else:
            print(f"❌ Error: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed - make sure the server is running on localhost:8000")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_mpin_login()

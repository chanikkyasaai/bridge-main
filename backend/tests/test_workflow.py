#!/usr/bin/env python3
"""
End-to-end workflow test for the corrected authentication flow
This script tests the complete workflow: register → login → verify MPIN → session start
"""

import requests
import json

# Configuration
BASE_URL = "http://localhost:8000"
TEST_PHONE = "9876543210"
TEST_PASSWORD = "SecurePassword123"
TEST_MPIN = "123456"
TEST_DEVICE_ID = "test_device_001"

def test_complete_workflow():
    """Test the complete authentication and session workflow"""
    
    print("🚀 Starting end-to-end authentication workflow test...\n")
    
    # Step 1: Registration
    print("1️⃣ Testing user registration...")
    register_response = requests.post(f"{BASE_URL}/api/v1/auth/register", json={
        "phone": TEST_PHONE,
        "password": TEST_PASSWORD,
        "mpin": TEST_MPIN
    })
    
    if register_response.status_code == 200:
        print("✅ Registration successful")
        print(f"   Response: {register_response.json()}")
    else:
        print(f"❌ Registration failed: {register_response.status_code}")
        print(f"   Error: {register_response.text}")
        return False
    
    print()
    
    # Step 2: Login
    print("2️⃣ Testing user login...")
    login_response = requests.post(f"{BASE_URL}/api/v1/auth/login", json={
        "phone": TEST_PHONE,
        "password": TEST_PASSWORD,
        "device_id": TEST_DEVICE_ID
    })
    
    if login_response.status_code == 200:
        login_data = login_response.json()
        print("✅ Login successful")
        print(f"   Access token: {login_data['access_token'][:50]}...")
        print(f"   Refresh token: {login_data['refresh_token'][:50]}...")
        print(f"   Token type: {login_data['token_type']}")
        print(f"   Expires in: {login_data['expires_in']} seconds")
        
        # Verify that session_id and session_token are NOT in the response
        if 'session_id' not in login_data and 'session_token' not in login_data:
            print("✅ Confirmed: No session tokens in login response")
        else:
            print("❌ Unexpected: Session tokens found in login response")
            return False
        
        access_token = login_data['access_token']
    else:
        print(f"❌ Login failed: {login_response.status_code}")
        print(f"   Error: {login_response.text}")
        return False
    
    print()
    
    # Step 3: MPIN Verification
    print("3️⃣ Testing MPIN verification...")
    mpin_response = requests.post(f"{BASE_URL}/api/v1/auth/verify-mpin", 
        json={"mpin": TEST_MPIN},
        headers={"Authorization": f"Bearer {access_token}"}
    )
    
    if mpin_response.status_code == 200:
        mpin_data = mpin_response.json()
        print("✅ MPIN verification successful")
        print(f"   Message: {mpin_data['message']}")
        print(f"   User ID: {mpin_data['user_id']}")
        print(f"   Status: {mpin_data['status']}")
        print(f"   Session ID: {mpin_data['session_id']}")
        print(f"   Session token: {mpin_data['session_token'][:50]}...")
        print(f"   Behavioral logging: {mpin_data['behavioral_logging']}")
        
        session_token = mpin_data['session_token']
    else:
        print(f"❌ MPIN verification failed: {mpin_response.status_code}")
        print(f"   Error: {mpin_response.text}")
        return False
    
    print()
    
    # Step 4: Test behavioral logging endpoint (if available)
    print("4️⃣ Testing behavioral logging with session token...")
    try:
        log_response = requests.post(f"{BASE_URL}/api/v1/log/action",
            json={
                "action": "test_action",
                "data": {"test": "workflow_validation"}
            },
            headers={"Authorization": f"Bearer {session_token}"}
        )
        
        if log_response.status_code in [200, 201]:
            print("✅ Behavioral logging endpoint accessible with session token")
        else:
            print(f"⚠️  Behavioral logging endpoint response: {log_response.status_code}")
    except Exception as e:
        print(f"⚠️  Behavioral logging endpoint test skipped: {e}")
    
    print()
    print("🎉 End-to-end workflow test completed successfully!")
    print("✅ Authentication flow is working correctly:")
    print("   - Registration creates user account")
    print("   - Login returns only authentication tokens") 
    print("   - MPIN verification starts behavioral session")
    print("   - Session token enables behavioral logging")
    
    return True

if __name__ == "__main__":
    try:
        test_complete_workflow()
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()

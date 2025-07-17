#!/usr/bin/env python3
"""
Register demo user for behavioral authentication testing
"""

import requests
import json

def register_demo_user():
    """Register a demo user for testing"""
    backend_url = "http://localhost:8000"
    
    # Register user
    user_data = {
        "phone": "9876543210",
        "password": "DemoPass123",
        "mpin": "123456"
    }
    
    try:
        print("🔐 Registering demo user...")
        response = requests.post(f"{backend_url}/api/v1/auth/register", json=user_data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ User registered successfully!")
            print(f"   User ID: {result['user_id']}")
            print(f"   Phone: {result['phone']}")
            return True
        elif response.status_code == 400:
            error_detail = response.json().get("detail", "Unknown error")
            if "already registered" in error_detail:
                print("✅ User already exists - ready for demo")
                return True
            else:
                print(f"❌ Registration failed: {error_detail}")
                return False
        else:
            print(f"❌ Registration failed: {response.status_code} - {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to backend")
        print("💡 Make sure backend is running: backend/start_backend.bat")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = register_demo_user()
    if success:
        print("\n🚀 Ready to run behavioral demo!")
        print("   Run: python websocket_behavioral_demo.py")
    else:
        print("\n❌ Demo setup failed")

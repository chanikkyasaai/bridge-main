#!/usr/bin/env python3
"""
Test Session Token Database Storage Fix
Verifies that session tokens are stored correctly in the database
"""

import asyncio
import json
import httpx
from datetime import datetime

# Configuration
BACKEND_URL = "http://127.0.0.1:8000"

class SessionTokenTest:
    def __init__(self):
        self.session_data = {}
        self.access_token = None
        self.session_token = None
        
    async def test_session_token_storage(self):
        """Test that session tokens are stored correctly in database"""
        print("🔧 Testing Session Token Database Storage Fix")
        print("=" * 50)
        
        try:
            # Step 1: Register user
            await self.register_user()
            
            # Step 2: Login and get tokens
            await self.login_user()
            
            # Step 3: Verify MPIN (this should create session with correct token)
            await self.verify_mpin()
            
            # Step 4: Check session status to verify token
            await self.check_session_status()
            
            # Step 5: Test behavioral data with session token
            await self.test_behavioral_data()
            
            # Step 6: Logout
            await self.logout_user()
            
            print("✅ Session token storage test completed successfully!")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            raise
    
    async def register_user(self):
        """Register a test user"""
        print("\n👤 Registering test user...")
        
        user_data = {
            "phone": "9876543211",
            "password": "testpassword123",
            "mpin": "12345"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BACKEND_URL}/api/v1/auth/register",
                json=user_data
            )
            
            if response.status_code == 200:
                result = response.json()
                self.session_data["user_id"] = result["user_id"]
                self.session_data["phone"] = user_data["phone"]
                print(f"✅ User registered: {result['user_id']}")
            else:
                print(f"❌ Registration failed: {response.text}")
                raise Exception("User registration failed")
    
    async def login_user(self):
        """Login user and get tokens"""
        print("\n🔐 Logging in user...")
        
        login_data = {
            "phone": self.session_data["phone"],
            "password": "testpassword123",
            "device_id": "test-device-002"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BACKEND_URL}/api/v1/auth/login",
                json=login_data
            )
            
            if response.status_code == 200:
                result = response.json()
                self.access_token = result["access_token"]
                self.session_data["device_id"] = login_data["device_id"]
                print("✅ Login successful")
            else:
                print(f"❌ Login failed: {response.text}")
                raise Exception("Login failed")
    
    async def verify_mpin(self):
        """Verify MPIN and create session with token"""
        print("\n🔑 Verifying MPIN...")
        
        mpin_data = {
            "mpin": "12345"
        }
        
        headers = {"Authorization": f"Bearer {self.access_token}"}
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BACKEND_URL}/api/v1/auth/verify-mpin",
                json=mpin_data,
                headers=headers
            )
            
            if response.status_code == 200:
                result = response.json()
                self.session_token = result["session_token"]
                self.session_data["session_id"] = result["session_id"]
                print(f"✅ MPIN verified, session started: {result['session_id']}")
                print(f"✅ Session token received: {self.session_token[:20]}...")
            else:
                print(f"❌ MPIN verification failed: {response.text}")
                raise Exception("MPIN verification failed")
    
    async def check_session_status(self):
        """Check session status to verify token is working"""
        print("\n📊 Checking session status...")
        
        headers = {"Authorization": f"Bearer {self.access_token}"}
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{BACKEND_URL}/api/v1/auth/session-status",
                headers=headers
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Session status: {result}")
                
                # Verify session ID matches
                if result.get("session_id") == self.session_data["session_id"]:
                    print("✅ Session ID matches")
                else:
                    print("❌ Session ID mismatch")
                    raise Exception("Session ID mismatch")
            else:
                print(f"❌ Session status failed: {response.text}")
                raise Exception("Session status check failed")
    
    async def test_behavioral_data(self):
        """Test behavioral data with session token"""
        print("\n📈 Testing behavioral data with session token...")
        
        behavioral_data = {
            "session_id": self.session_data["session_id"],
            "event_type": "test_event",
            "data": {
                "test": "session_token_verification",
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
        headers = {"Authorization": f"Bearer {self.session_token}"}
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BACKEND_URL}/api/v1/log/behavior-data",
                json=behavioral_data,
                headers=headers
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Behavioral data logged successfully: {result}")
            else:
                print(f"❌ Behavioral data logging failed: {response.text}")
                raise Exception("Behavioral data logging failed")
    
    async def logout_user(self):
        """Logout user"""
        print("\n🚪 Logging out user...")
        
        headers = {"Authorization": f"Bearer {self.access_token}"}
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BACKEND_URL}/api/v1/auth/logout",
                headers=headers
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Logout successful: {result}")
            else:
                print(f"❌ Logout failed: {response.text}")
    
    async def debug_session_token(self):
        """Debug session token contents"""
        print("\n🔍 Debugging session token...")
        
        if not self.session_token:
            print("❌ No session token available")
            return
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{BACKEND_URL}/api/v1/ws/debug/token/{self.session_token}"
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Token debug info: {result}")
            else:
                print(f"❌ Token debug failed: {response.text}")

async def main():
    """Main test runner"""
    test = SessionTokenTest()
    await test.test_session_token_storage()

if __name__ == "__main__":
    asyncio.run(main()) 
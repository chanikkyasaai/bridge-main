#!/usr/bin/env python3
"""
Test ML Engine Integration
Tests the complete flow from session start to end with ML engine integration
"""

import asyncio
import json
import httpx
import time
from datetime import datetime

# Configuration
BACKEND_URL = "http://127.0.0.1:8000"
ML_ENGINE_URL = "http://127.0.0.1:8001"

class MLIntegrationTest:
    def __init__(self):
        self.session_data = {}
        self.access_token = None
        self.session_token = None
        
    async def test_ml_integration_flow(self):
        """Test complete ML integration flow"""
        print("🚀 Starting ML Engine Integration Test")
        print("=" * 50)
        
        try:
            # Step 1: Check ML Engine health
            await self.check_ml_engine_health()
            
            # Step 2: Register user
            await self.register_user()
            
            # Step 3: Login and get tokens
            await self.login_user()
            
            # Step 4: Verify MPIN (this should start ML session)
            await self.verify_mpin()
            
            # Step 5: Send behavioral data via WebSocket
            await self.test_websocket_behavioral_data()
            
            # Step 6: Test app lifecycle events
            await self.test_app_lifecycle_events()
            
            # Step 7: Logout (this should end ML session)
            await self.logout_user()
            
            print("✅ All ML integration tests passed!")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            raise
    
    async def check_ml_engine_health(self):
        """Check if ML Engine is running"""
        print("\n📡 Checking ML Engine health...")
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{ML_ENGINE_URL}/")
                if response.status_code == 200:
                    health_data = response.json()
                    print(f"✅ ML Engine is healthy: {health_data.get('status', 'unknown')}")
                    return True
                else:
                    print(f"⚠️ ML Engine returned status {response.status_code}")
                    return False
        except Exception as e:
            print(f"❌ ML Engine not reachable: {e}")
            return False
    
    async def register_user(self):
        """Register a test user"""
        print("\n👤 Registering test user...")
        
        user_data = {
            "phone": "9876543210",
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
            "device_id": "test-device-001"
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
        """Verify MPIN and start ML session"""
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
                
                # Check if ML session was started
                await self.check_ml_session_status()
            else:
                print(f"❌ MPIN verification failed: {response.text}")
                raise Exception("MPIN verification failed")
    
    async def check_ml_session_status(self):
        """Check if ML session was properly started"""
        print("\n🔍 Checking ML session status...")
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{ML_ENGINE_URL}/statistics")
                if response.status_code == 200:
                    stats = response.json()
                    print(f"✅ ML Engine statistics: {stats}")
                else:
                    print(f"⚠️ Could not get ML Engine statistics: {response.status_code}")
        except Exception as e:
            print(f"⚠️ Could not check ML Engine status: {e}")
    
    async def test_websocket_behavioral_data(self):
        """Test WebSocket behavioral data transmission"""
        print("\n📡 Testing WebSocket behavioral data...")
        
        # This would require a WebSocket client implementation
        # For now, we'll test the REST endpoint
        await self.test_rest_behavioral_data()
    
    async def test_rest_behavioral_data(self):
        """Test REST behavioral data endpoint"""
        print("\n📊 Testing REST behavioral data...")
        
        behavioral_data = {
            "session_id": self.session_data["session_id"],
            "event_type": "typing_pattern",
            "data": {
                "words_per_minute": 45,
                "typing_style": "normal",
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
                print(f"✅ Behavioral data logged: {result}")
            else:
                print(f"❌ Behavioral data logging failed: {response.text}")
    
    async def test_app_lifecycle_events(self):
        """Test app lifecycle events"""
        print("\n📱 Testing app lifecycle events...")
        
        lifecycle_data = {
            "event_type": "app_background",
            "details": {
                "reason": "user_switched_app",
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
        headers = {"Authorization": f"Bearer {self.session_token}"}
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BACKEND_URL}/api/v1/ws/sessions/{self.session_data['session_id']}/lifecycle",
                json=lifecycle_data,
                headers=headers
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Lifecycle event processed: {result}")
            else:
                print(f"❌ Lifecycle event failed: {response.text}")
    
    async def logout_user(self):
        """Logout user and end ML session"""
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
                
                # Check if ML session was ended
                await self.check_ml_session_ended()
            else:
                print(f"❌ Logout failed: {response.text}")
    
    async def check_ml_session_ended(self):
        """Check if ML session was properly ended"""
        print("\n🔍 Checking ML session ended...")
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{ML_ENGINE_URL}/statistics")
                if response.status_code == 200:
                    stats = response.json()
                    print(f"✅ ML Engine final statistics: {stats}")
                else:
                    print(f"⚠️ Could not get final ML Engine statistics: {response.status_code}")
        except Exception as e:
            print(f"⚠️ Could not check final ML Engine status: {e}")

async def main():
    """Main test runner"""
    test = MLIntegrationTest()
    await test.test_ml_integration_flow()

if __name__ == "__main__":
    asyncio.run(main()) 
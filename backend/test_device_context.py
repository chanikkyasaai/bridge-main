#!/usr/bin/env python3
"""
Test Device Context Integration
Verifies that device context information is properly handled in MPIN verification
"""

import asyncio
import json
import httpx
from datetime import datetime

# Configuration
BACKEND_URL = "http://127.0.0.1:8000"

class DeviceContextTest:
    def __init__(self):
        self.session_data = {}
        self.access_token = None
        self.session_token = None
        
    async def test_device_context_integration(self):
        """Test device context integration in MPIN verification"""
        print("📱 Testing Device Context Integration")
        print("=" * 50)
        
        try:
            # Step 1: Register user
            await self.register_user()
            
            # Step 2: Login and get tokens
            await self.login_user()
            
            # Step 3: Verify MPIN with device context
            await self.verify_mpin_with_context()
            
            # Step 4: Test MPIN login with device context
            await self.test_mpin_login_with_context()
            
            print("✅ Device context integration test completed successfully!")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            raise
    
    async def register_user(self):
        """Register a test user"""
        print("\n👤 Registering test user...")
        
        user_data = {
            "phone": "9876543212",
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
            "device_id": "test-device-003"
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
    
    async def verify_mpin_with_context(self):
        """Verify MPIN with device context"""
        print("\n🔑 Verifying MPIN with device context...")
        
        # Create comprehensive device context
        device_context = {
            "device_id": "test-device-003",
            "device_type": "mobile",
            "device_model": "iPhone 15 Pro",
            "os_version": "iOS 17.2",
            "app_version": "1.2.3",
            "network_type": "wifi",
            "location_data": {
                "latitude": 12.9716,
                "longitude": 77.5946,
                "city": "Bangalore",
                "country": "India"
            },
            "user_agent": "CanaraBankApp/1.2.3 (iPhone; iOS 17.2; Scale/3.00)",
            "ip_address": "192.168.1.100"
        }
        
        mpin_data = {
            "mpin": "12345",
            "context": device_context
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
                print(f"✅ MPIN verified with context, session started: {result['session_id']}")
                print(f"✅ Session token received: {self.session_token[:20]}...")
            else:
                print(f"❌ MPIN verification failed: {response.text}")
                raise Exception("MPIN verification failed")
    
    async def test_mpin_login_with_context(self):
        """Test MPIN login with device context"""
        print("\n🔐 Testing MPIN login with device context...")
        
        # Create device context for MPIN login
        device_context = {
            "device_id": "test-device-004",
            "device_type": "tablet",
            "device_model": "iPad Pro 12.9",
            "os_version": "iPadOS 17.2",
            "app_version": "1.2.3",
            "network_type": "cellular",
            "location_data": {
                "latitude": 19.0760,
                "longitude": 72.8777,
                "city": "Mumbai",
                "country": "India"
            },
            "user_agent": "CanaraBankApp/1.2.3 (iPad; iPadOS 17.2; Scale/2.00)",
            "ip_address": "10.0.0.50"
        }
        
        mpin_login_data = {
            "phone": self.session_data["phone"],
            "mpin": "12345",
            "device_id": "test-device-004",
            "context": device_context
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BACKEND_URL}/api/v1/auth/mpin-login",
                json=mpin_login_data
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ MPIN login with context successful: {result['session_id']}")
                print(f"✅ Access token: {result['access_token'][:20]}...")
                print(f"✅ Session token: {result['session_token'][:20]}...")
            else:
                print(f"❌ MPIN login failed: {response.text}")
                raise Exception("MPIN login failed")
    
    async def test_behavioral_data_with_context(self):
        """Test behavioral data with session token from context"""
        print("\n📈 Testing behavioral data with context session...")
        
        behavioral_data = {
            "session_id": self.session_data["session_id"],
            "event_type": "device_context_test",
            "data": {
                "test": "device_context_verification",
                "timestamp": datetime.utcnow().isoformat(),
                "context_verified": True
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
                print(f"✅ Behavioral data logged with context: {result}")
            else:
                print(f"❌ Behavioral data logging failed: {response.text}")
    
    async def test_minimal_context(self):
        """Test with minimal device context"""
        print("\n🔧 Testing minimal device context...")
        
        # Test with only required device_id
        minimal_context = {
            "device_id": "minimal-device-001"
        }
        
        mpin_data = {
            "mpin": "12345",
            "context": minimal_context
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
                print(f"✅ Minimal context MPIN verification successful: {result['session_id']}")
            else:
                print(f"❌ Minimal context MPIN verification failed: {response.text}")
    
    async def test_no_context(self):
        """Test without device context (fallback)"""
        print("\n🔄 Testing without device context (fallback)...")
        
        mpin_data = {
            "mpin": "12345"
            # No context field
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
                print(f"✅ No context MPIN verification successful: {result['session_id']}")
            else:
                print(f"❌ No context MPIN verification failed: {response.text}")

async def main():
    """Main test runner"""
    test = DeviceContextTest()
    await test.test_device_context_integration()

if __name__ == "__main__":
    asyncio.run(main()) 
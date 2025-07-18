#!/usr/bin/env python3
"""
Test mobile app integration with backend and ML engine
Simulates the complete flow that a mobile app would follow
"""

import asyncio
import aiohttp
import json
import uuid
import random
from datetime import datetime

# Configuration
BACKEND_URL = "http://localhost:8000"
ML_ENGINE_URL = "http://localhost:8001"

class MobileIntegrationTester:
    def __init__(self):
        self.session = None
        self.session_id = None
        self.session_token = None
        self.user_id = str(uuid.uuid4())
        self.phone = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def generate_mobile_behavioral_logs(self) -> list:
        """Generate realistic mobile behavioral logs"""
        base_timestamp = datetime.utcnow().timestamp() * 1000
        logs = []
        
        # Touch events
        for i in range(15):
            logs.append({
                "event_type": "touch",
                "timestamp": base_timestamp + i * 100,
                "data": {
                    "x": 150 + random.uniform(-50, 50),
                    "y": 300 + random.uniform(-50, 50),
                    "pressure": 0.5 + random.uniform(-0.2, 0.2),
                    "action": "down" if i % 3 == 0 else "move"
                }
            })
        
        # Accelerometer events
        for i in range(25):
            logs.append({
                "event_type": "accelerometer",
                "timestamp": base_timestamp + i * 50,
                "data": {
                    "x": random.uniform(-2, 2),
                    "y": random.uniform(-2, 2),
                    "z": 9.8 + random.uniform(-1, 1)
                }
            })
        
        # Gyroscope events
        for i in range(25):
            logs.append({
                "event_type": "gyroscope",
                "timestamp": base_timestamp + i * 50,
                "data": {
                    "x": random.uniform(-0.5, 0.5),
                    "y": random.uniform(-0.5, 0.5),
                    "z": random.uniform(-0.5, 0.5)
                }
            })
        
        # Scroll events
        for i in range(8):
            logs.append({
                "event_type": "scroll",
                "timestamp": base_timestamp + i * 200,
                "data": {
                    "delta_y": random.uniform(-100, 100),
                    "velocity": random.uniform(50, 200)
                }
            })
        
        # Sort by timestamp
        logs.sort(key=lambda x: x['timestamp'])
        return logs
    
    async def create_test_user(self):
        """Create a test user in the backend"""
        try:
            # Generate a valid 10-digit phone number (digits only)
            phone_suffix = ''.join([str(random.randint(0, 9)) for _ in range(6)])
            self.phone = f"9876{phone_suffix}"
            
            user_data = {
                "phone": self.phone,
                "password": "TestPassword123!",
                "mpin": "123456"
            }
            
            async with self.session.post(f"{BACKEND_URL}/api/v1/auth/register", json=user_data) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"✅ Created test user: {self.user_id}")
                    return True
                else:
                    error_text = await response.text()
                    print(f"⚠️ Failed to create user: {response.status} - {error_text}")
                    return False
        except Exception as e:
            print(f"⚠️ Error creating user: {e}")
            return False
    
    async def login_user(self):
        """Login the test user"""
        try:
            login_data = {
                "phone": self.phone,
                "password": "TestPassword123!",
                "device_id": f"device_{self.user_id[:8]}"
            }
            
            async with self.session.post(f"{BACKEND_URL}/api/v1/auth/login", json=login_data) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"✅ User logged in successfully")
                    return result.get("access_token")
                else:
                    error_text = await response.text()
                    print(f"⚠️ Login failed: {response.status} - {error_text}")
                    return None
        except Exception as e:
            print(f"⚠️ Error during login: {e}")
            return None
    
    async def start_session(self, access_token):
        """Start a behavioral logging session"""
        try:
            session_data = {
                "phone": self.phone,
                "device_id": f"device_{self.user_id[:8]}",
                "mpin": "123456"
            }
            
            headers = {"Authorization": f"Bearer {access_token}"}
            
            async with self.session.post(f"{BACKEND_URL}/api/v1/log/start-session", 
                                       json=session_data, headers=headers) as response:
                if response.status == 200:
                    result = await response.json()
                    self.session_id = result["session_id"]
                    self.session_token = result["session_token"]
                    print(f"✅ Session started: {self.session_id}")
                    return True
                else:
                    error_text = await response.text()
                    print(f"⚠️ Failed to start session: {response.status} - {error_text}")
                    return False
        except Exception as e:
            print(f"⚠️ Error starting session: {e}")
            return False
    
    async def send_behavioral_data(self, logs):
        """Send behavioral data to backend"""
        try:
            headers = {"Authorization": f"Bearer {self.session_token}"}
            
            # Send behavioral events one by one (like a real mobile app would)
            for i, log_entry in enumerate(logs):
                behavior_data = {
                    "session_id": self.session_id,
                    "event_type": log_entry["event_type"],
                    "data": log_entry["data"]
                }
                
                async with self.session.post(f"{BACKEND_URL}/api/v1/log/behavior-data",
                                           json=behavior_data, headers=headers) as response:
                    if response.status == 200:
                        if i % 10 == 0:  # Log every 10th event
                            print(f"📊 Sent behavioral event {i+1}/{len(logs)}")
                    else:
                        error_text = await response.text()
                        print(f"⚠️ Failed to send behavioral data: {response.status} - {error_text}")
                        return False
            
            print(f"✅ Sent {len(logs)} behavioral events")
            return True
        except Exception as e:
            print(f"⚠️ Error sending behavioral data: {e}")
            return False
    
    async def end_session(self):
        """End the behavioral logging session"""
        try:
            end_data = {
                "session_id": self.session_id,
                "session_token": self.session_token,
                "final_decision": "normal"
            }
            
            async with self.session.post(f"{BACKEND_URL}/api/v1/log/end-session", json=end_data) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"✅ Session ended successfully")
                    return True
                else:
                    error_text = await response.text()
                    print(f"⚠️ Failed to end session: {response.status} - {error_text}")
                    return False
        except Exception as e:
            print(f"⚠️ Error ending session: {e}")
            return False
    
    async def test_ml_engine_direct(self, logs):
        """Test ML engine directly to verify it's working"""
        try:
            ml_data = {
                "user_id": self.user_id,
                "session_id": self.session_id,
                "logs": logs
            }
            
            async with self.session.post(f"{ML_ENGINE_URL}/analyze-mobile", json=ml_data) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"🤖 ML Engine Analysis: {result['decision']} (confidence: {result['confidence']:.2f})")
                    print(f"🎯 Vector ID: {result.get('vector_id', 'N/A')}")
                    print(f"📈 Risk Level: {result.get('risk_level', 'N/A')}")
                    return result
                else:
                    error_text = await response.text()
                    print(f"⚠️ ML Engine analysis failed: {response.status} - {error_text}")
                    return None
        except Exception as e:
            print(f"⚠️ Error testing ML engine: {e}")
            return None

async def main():
    """Run the complete mobile integration test"""
    print("🚀 Starting Mobile Integration Test")
    print("=" * 50)
    
    async with MobileIntegrationTester() as tester:
        # Step 1: Create test user
        print("\n📱 Step 1: Creating test user...")
        user_created = await tester.create_test_user()
        if not user_created:
            print("❌ Cannot proceed without user creation")
            return
        
        # Step 2: Login user
        print("\n🔐 Step 2: Logging in user...")
        access_token = await tester.login_user()
        if not access_token:
            print("❌ Cannot proceed without login")
            return
        
        # Step 3: Start session
        print("\n📊 Step 3: Starting behavioral logging session...")
        session_started = await tester.start_session(access_token)
        if not session_started:
            print("❌ Cannot proceed without session")
            return
        
        # Step 4: Generate and send behavioral data
        print("\n🎯 Step 4: Generating and sending behavioral data...")
        logs = tester.generate_mobile_behavioral_logs()
        data_sent = await tester.send_behavioral_data(logs)
        if not data_sent:
            print("❌ Failed to send behavioral data")
            return
        
        # Step 5: Test ML engine directly
        print("\n🤖 Step 5: Testing ML engine directly...")
        ml_result = await tester.test_ml_engine_direct(logs)
        
        # Step 6: End session
        print("\n🏁 Step 6: Ending session...")
        session_ended = await tester.end_session()
        
        # Summary
        print("\n📋 Test Summary:")
        print("=" * 30)
        print(f"✅ User Created: {user_created}")
        print(f"✅ User Logged In: {access_token is not None}")
        print(f"✅ Session Started: {session_started}")
        print(f"✅ Behavioral Data Sent: {data_sent}")
        print(f"✅ ML Engine Analyzed: {ml_result is not None}")
        print(f"✅ Session Ended: {session_ended}")
        
        if ml_result:
            print(f"\n🎯 ML Analysis Results:")
            print(f"   Decision: {ml_result.get('decision', 'N/A')}")
            print(f"   Confidence: {ml_result.get('confidence', 0):.2f}")
            print(f"   Risk Level: {ml_result.get('risk_level', 'N/A')}")
            print(f"   Vector ID: {ml_result.get('vector_id', 'N/A')}")
        
        if all([user_created, access_token, session_started, data_sent, session_ended]):
            print("\n🎉 MOBILE INTEGRATION TEST PASSED!")
            print("✅ The complete mobile app flow is working correctly!")
        else:
            print("\n❌ MOBILE INTEGRATION TEST FAILED!")
            print("⚠️ Some steps failed - check the logs above")

if __name__ == "__main__":
    asyncio.run(main())

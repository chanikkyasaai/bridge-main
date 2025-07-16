"""
Complete Integration Test Script
Tests the full flow between Backend and ML Engine
"""

import asyncio
import json
import websockets
import httpx
from datetime import datetime
import uuid

# Configuration
BACKEND_URL = "http://localhost:8000"
ML_ENGINE_URL = "http://localhost:8001"
TEST_USER_PHONE = "8825624303"
TEST_USER_PASSWORD = "testpass123"
TEST_USER_MPIN = "1234"

class IntegrationTester:
    def __init__(self):
        self.access_token = None
        self.session_id = None
        self.session_token = None
        self.user_id = None
        
    async def test_complete_flow(self):
        """Test the complete integration flow"""
        print("🚀 Starting Complete Integration Test")
        print("=" * 50)
        
        try:
            # Step 1: Check ML Engine Health
            await self.test_ml_engine_health()
            
            # Step 2: Register/Login User
            await self.test_user_authentication()
            
            # Step 3: Verify MPIN and Start Session
            await self.test_mpin_verification()
            
            # Step 4: Test WebSocket Behavioral Data
            await self.test_websocket_behavioral_data()
            
            # Step 5: Test ML Analysis
            await self.test_ml_analysis()
            
            # Step 6: Test Feedback
            await self.test_feedback_submission()
            
            # Step 7: Test Session Cleanup
            await self.test_session_cleanup()
            
            print("\n✅ All tests completed successfully!")
            
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            raise
    
    async def test_ml_engine_health(self):
        """Test ML Engine availability"""
        print("\n1️⃣ Testing ML Engine Health...")
        
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{ML_ENGINE_URL}/")
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ ML Engine Status: {data['status']}")
                print(f"   📊 Components: {data['components']}")
            else:
                raise Exception(f"ML Engine health check failed: {response.status_code}")
    
    async def test_user_authentication(self):
        """Test user login"""
        print("\n2️⃣ Testing User Authentication...")
        
        async with httpx.AsyncClient() as client:
            # Try to login
            login_data = {
                "phone": TEST_USER_PHONE,
                "password": TEST_USER_PASSWORD,
                "device_id": "test_device_123"
            }
            
            response = await client.post(f"{BACKEND_URL}/api/v1/auth/login", json=login_data)
            
            if response.status_code == 200:
                data = response.json()
                self.access_token = data["access_token"]
                print(f"   ✅ Login successful")
                print(f"   🔑 Access token obtained")
            else:
                # Try to register if login fails
                print("   ℹ️ Login failed, attempting registration...")
                register_data = {
                    "phone": TEST_USER_PHONE,
                    "password": TEST_USER_PASSWORD,
                    "mpin": TEST_USER_MPIN
                }
                
                reg_response = await client.post(f"{BACKEND_URL}/api/v1/auth/register", json=register_data)
                if reg_response.status_code == 201:
                    print("   ✅ Registration successful")
                    # Now login
                    login_response = await client.post(f"{BACKEND_URL}/api/v1/auth/login", json=login_data)
                    if login_response.status_code == 200:
                        data = login_response.json()
                        self.access_token = data["access_token"]
                        print(f"   ✅ Login after registration successful")
                    else:
                        raise Exception(f"Login after registration failed: {login_response.text}")
                else:
                    raise Exception(f"Registration failed: {reg_response.text}")
    
    async def test_mpin_verification(self):
        """Test MPIN verification and session start"""
        print("\n3️⃣ Testing MPIN Verification & Session Start...")
        
        async with httpx.AsyncClient() as client:
            headers = {"Authorization": f"Bearer {self.access_token}"}
            mpin_data = {"mpin": TEST_USER_MPIN}
            
            response = await client.post(
                f"{BACKEND_URL}/api/v1/auth/verify-mpin", 
                json=mpin_data,
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                self.session_id = data["session_id"]
                self.session_token = data["session_token"]
                self.user_id = data["user_id"]
                print(f"   ✅ MPIN verified successfully")
                print(f"   🎯 Session ID: {self.session_id}")
                print(f"   👤 User ID: {self.user_id}")
            else:
                raise Exception(f"MPIN verification failed: {response.text}")
    
    async def test_websocket_behavioral_data(self):
        """Test WebSocket behavioral data collection"""
        print("\n4️⃣ Testing WebSocket Behavioral Data...")
        
        ws_url = f"ws://localhost:8000/api/v1/ws/behavior/{self.session_id}?token={self.session_token}"
        
        try:
            async with websockets.connect(ws_url) as websocket:
                # Wait for connection confirmation
                confirmation = await websocket.recv()
                confirm_data = json.loads(confirmation)
                print(f"   ✅ WebSocket connected: {confirm_data['type']}")
                
                # Send sample behavioral events
                sample_events = [
                    {
                        "event_type": "touch_event",
                        "data": {
                            "x": 150.5,
                            "y": 300.2,
                            "pressure": 0.8,
                            "touch_area": 12.5
                        }
                    },
                    {
                        "event_type": "keystroke",
                        "data": {
                            "key": "a",
                            "dwell_time": 120,
                            "flight_time": 80
                        }
                    },
                    {
                        "event_type": "accel_data",
                        "data": {
                            "x": 0.1,
                            "y": 5.4,
                            "z": 8.3
                        }
                    }
                ]
                
                for event in sample_events:
                    await websocket.send(json.dumps(event))
                    response = await websocket.recv()
                    resp_data = json.loads(response)
                    print(f"   📤 Sent {event['event_type']}: {resp_data['status']}")
                
                print(f"   ✅ Behavioral data sent successfully")
                
        except Exception as e:
            print(f"   ❌ WebSocket test failed: {e}")
            raise
    
    async def test_ml_analysis(self):
        """Test direct ML analysis"""
        print("\n5️⃣ Testing ML Analysis...")
        
        async with httpx.AsyncClient() as client:
            # Prepare sample events for ML analysis
            events = [
                {
                    "event_type": "touch_event",
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": {"x": 150.5, "y": 300.2, "pressure": 0.8}
                },
                {
                    "event_type": "keystroke",
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": {"key": "a", "dwell_time": 120}
                }
            ]
            
            analysis_data = {
                "user_id": self.user_id,
                "session_id": self.session_id,
                "events": events
            }
            
            response = await client.post(f"{ML_ENGINE_URL}/analyze", json=analysis_data)
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ ML Analysis completed")
                print(f"   🎯 Decision: {data.get('decision')}")
                print(f"   📊 Confidence: {data.get('confidence'):.2f}")
                print(f"   🔍 Similarity Score: {data.get('similarity_score', 'N/A')}")
            else:
                print(f"   ⚠️ ML Analysis failed: {response.text}")
    
    async def test_feedback_submission(self):
        """Test feedback submission"""
        print("\n6️⃣ Testing Feedback Submission...")
        
        async with httpx.AsyncClient() as client:
            headers = {"Authorization": f"Bearer {self.access_token}"}
            
            feedback_data = {
                "user_id": self.user_id,
                "session_id": self.session_id,
                "decision_id": f"test_decision_{uuid.uuid4()}",
                "was_correct": True,
                "feedback_source": "integration_test"
            }
            
            response = await client.post(
                f"{BACKEND_URL}/api/v1/ml/feedback",
                json=feedback_data,
                headers=headers
            )
            
            if response.status_code == 200:
                print(f"   ✅ Feedback submitted successfully")
            else:
                print(f"   ⚠️ Feedback submission failed: {response.text}")
    
    async def test_session_cleanup(self):
        """Test session cleanup"""
        print("\n7️⃣ Testing Session Cleanup...")
        
        async with httpx.AsyncClient() as client:
            headers = {"Authorization": f"Bearer {self.access_token}"}
            
            response = await client.post(f"{BACKEND_URL}/api/v1/auth/logout", headers=headers)
            
            if response.status_code == 200:
                print(f"   ✅ Session cleanup successful")
            else:
                print(f"   ⚠️ Session cleanup failed: {response.text}")

async def main():
    """Run the complete integration test"""
    tester = IntegrationTester()
    await tester.test_complete_flow()

if __name__ == "__main__":
    print("🧪 Behavioral Authentication Integration Test")
    print("Make sure both Backend (8000) and ML Engine (8001) are running!")
    print()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()

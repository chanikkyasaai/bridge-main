#!/usr/bin/env python3
"""
Test Event Batching System
Verifies that behavioral events are properly batched and sent to ML engine
"""

import asyncio
import json
import httpx
import time
from datetime import datetime

# Configuration
BACKEND_URL = "http://127.0.0.1:8000"

class EventBatchingTest:
    def __init__(self):
        self.session_data = {}
        self.access_token = None
        self.session_token = None
        
    async def test_event_batching(self):
        """Test event batching functionality"""
        print("📦 Testing Event Batching System")
        print("=" * 50)
        
        try:
            # Step 1: Register and login user
            await self.setup_user()
            
            # Step 2: Verify MPIN to start session
            await self.verify_mpin()
            
            # Step 3: Test batching with multiple events
            await self.test_batch_processing()
            
            # Step 4: Test batch statistics
            await self.test_batch_statistics()
            
            # Step 5: Test manual flush
            await self.test_manual_flush()
            
            # Step 6: Test session termination with batch flush
            await self.test_session_termination()
            
            print("✅ Event batching test completed successfully!")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            raise
    
    async def setup_user(self):
        """Setup user for testing"""
        print("\n👤 Setting up test user...")
        
        # Register user
        user_data = {
            "phone": "9876543213",
            "password": "testpassword123",
            "mpin": "12345"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BACKEND_URL}/api/v1/auth/register",
                json=user_data
            )
            
            if response.status_code != 200:
                print(f"⚠️ User might already exist: {response.text}")
            
            # Login user
            login_data = {
                "phone": user_data["phone"],
                "password": user_data["password"],
                "device_id": "test-device-batch"
            }
            
            response = await client.post(
                f"{BACKEND_URL}/api/v1/auth/login",
                json=login_data
            )
            
            if response.status_code == 200:
                result = response.json()
                self.access_token = result["access_token"]
                self.session_data["phone"] = user_data["phone"]
                self.session_data["device_id"] = login_data["device_id"]
                print("✅ User setup completed")
            else:
                print(f"❌ Login failed: {response.text}")
                raise Exception("Login failed")
    
    async def verify_mpin(self):
        """Verify MPIN to start session"""
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
            else:
                print(f"❌ MPIN verification failed: {response.text}")
                raise Exception("MPIN verification failed")
    
    async def test_batch_processing(self):
        """Test batching with multiple events"""
        print("\n📦 Testing batch processing...")
        
        # Send 25 events (should trigger 2 batches: 20 + 5)
        event_types = [
            "typing_pattern", "mouse_movement", "click_event", 
            "scroll_event", "focus_change", "blur_event",
            "key_press", "key_release", "touch_start", "touch_end",
            "gesture_swipe", "gesture_pinch", "device_orientation",
            "network_change", "battery_level", "app_foreground",
            "app_background", "session_start", "session_end",
            "transaction_start", "transaction_end", "beneficiary_add",
            "amount_entered", "mpin_attempt", "security_check"
        ]
        
        headers = {"Authorization": f"Bearer {self.session_token}"}
        
        async with httpx.AsyncClient() as client:
            for i, event_type in enumerate(event_types):
                behavioral_data = {
                    "session_id": self.session_data["session_id"],
                    "event_type": event_type,
                    "data": {
                        "event_id": i + 1,
                        "timestamp": datetime.utcnow().isoformat(),
                        "test_batch": True,
                        "batch_number": (i // 20) + 1
                    }
                }
                
                response = await client.post(
                    f"{BACKEND_URL}/api/v1/log/behavior-data",
                    json=behavioral_data,
                    headers=headers
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"✅ Event {i+1} ({event_type}) logged: {result.get('total_events', 'N/A')}")
                else:
                    print(f"❌ Event {i+1} failed: {response.text}")
                
                # Small delay between events
                await asyncio.sleep(0.1)
        
        # Wait a bit for batch processing
        print("⏳ Waiting for batch processing...")
        await asyncio.sleep(2)
    
    async def test_batch_statistics(self):
        """Test batch statistics endpoint"""
        print("\n📊 Testing batch statistics...")
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{BACKEND_URL}/api/v1/ws/debug/event-batcher"
            )
            
            if response.status_code == 200:
                result = response.json()
                stats = result.get("event_batcher_stats", {})
                
                print(f"✅ Event batcher stats:")
                print(f"  - Total sessions: {stats.get('total_sessions', 0)}")
                print(f"  - Active batches: {stats.get('active_batches', 0)}")
                print(f"  - Pending events: {stats.get('total_pending_events', 0)}")
                print(f"  - Batch size: {stats.get('config', {}).get('batch_size', 0)}")
                print(f"  - Max wait time: {stats.get('config', {}).get('max_wait_time', 0)}s")
                print(f"  - Batching enabled: {stats.get('config', {}).get('enable_batching', False)}")
                print(f"  - Is running: {stats.get('is_running', False)}")
            else:
                print(f"❌ Failed to get stats: {response.text}")
    
    async def test_manual_flush(self):
        """Test manual batch flush"""
        print("\n🔄 Testing manual batch flush...")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BACKEND_URL}/api/v1/ws/debug/flush-all-batches"
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Manual flush successful: {result.get('message', 'N/A')}")
            else:
                print(f"❌ Manual flush failed: {response.text}")
    
    async def test_session_termination(self):
        """Test session termination with batch flush"""
        print("\n🚪 Testing session termination...")
        
        # Send a few more events
        headers = {"Authorization": f"Bearer {self.session_token}"}
        
        async with httpx.AsyncClient() as client:
            for i in range(5):
                behavioral_data = {
                    "session_id": self.session_data["session_id"],
                    "event_type": f"termination_test_{i}",
                    "data": {
                        "event_id": i + 1,
                        "timestamp": datetime.utcnow().isoformat(),
                        "test_termination": True
                    }
                }
                
                response = await client.post(
                    f"{BACKEND_URL}/api/v1/log/behavior-data",
                    json=behavioral_data,
                    headers=headers
                )
                
                if response.status_code == 200:
                    print(f"✅ Termination test event {i+1} logged")
                else:
                    print(f"❌ Termination test event {i+1} failed: {response.text}")
                
                await asyncio.sleep(0.1)
        
        # Logout to trigger session termination and batch flush
        logout_headers = {"Authorization": f"Bearer {self.access_token}"}
        
        response = await client.post(
            f"{BACKEND_URL}/api/v1/auth/logout",
            headers=logout_headers
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Logout successful: {result.get('message', 'N/A')}")
        else:
            print(f"❌ Logout failed: {response.text}")
    
    async def test_websocket_batching(self):
        """Test batching via WebSocket (simulated)"""
        print("\n🔌 Testing WebSocket batching simulation...")
        
        # This would require a WebSocket client implementation
        # For now, we'll simulate by sending events rapidly
        
        headers = {"Authorization": f"Bearer {self.session_token}"}
        
        async with httpx.AsyncClient() as client:
            # Send events rapidly to simulate WebSocket behavior
            tasks = []
            for i in range(30):
                behavioral_data = {
                    "session_id": self.session_data["session_id"],
                    "event_type": f"websocket_sim_{i}",
                    "data": {
                        "event_id": i + 1,
                        "timestamp": datetime.utcnow().isoformat(),
                        "websocket_sim": True,
                        "rapid_fire": True
                    }
                }
                
                task = client.post(
                    f"{BACKEND_URL}/api/v1/log/behavior-data",
                    json=behavioral_data,
                    headers=headers
                )
                tasks.append(task)
            
            # Execute all requests concurrently
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            success_count = 0
            for i, response in enumerate(responses):
                if isinstance(response, Exception):
                    print(f"❌ WebSocket sim event {i+1} failed: {response}")
                elif response.status_code == 200:
                    success_count += 1
                else:
                    print(f"❌ WebSocket sim event {i+1} failed: {response.text}")
            
            print(f"✅ WebSocket simulation: {success_count}/30 events successful")
    
    async def test_error_handling(self):
        """Test error handling in batching"""
        print("\n⚠️ Testing error handling...")
        
        # Test with invalid session
        headers = {"Authorization": f"Bearer {self.session_token}"}
        
        async with httpx.AsyncClient() as client:
            behavioral_data = {
                "session_id": "invalid-session-id",
                "event_type": "error_test",
                "data": {
                    "test_error": True,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            
            response = await client.post(
                f"{BACKEND_URL}/api/v1/log/behavior-data",
                json=behavioral_data,
                headers=headers
            )
            
            if response.status_code == 404:
                print("✅ Error handling working: Invalid session rejected")
            else:
                print(f"⚠️ Unexpected response for invalid session: {response.status_code}")

async def main():
    """Main test runner"""
    test = EventBatchingTest()
    await test.test_event_batching()

if __name__ == "__main__":
    asyncio.run(main()) 
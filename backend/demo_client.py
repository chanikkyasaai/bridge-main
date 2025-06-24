"""
Sample client demonstrating how to use the Canara AI Security Backend
This script shows how to:
1. Register a user
2. Login and get session token
3. Connect to WebSocket for behavioral data
4. Send behavioral events
5. Handle MPIN challenges
"""

import asyncio
import json
import websockets
import requests
from datetime import datetime
import random
import time

BASE_URL = "http://localhost:8000/api/v1"
WS_URL = "ws://localhost:8000/api/v1/ws"

class CanaraAIClient:
    def __init__(self):
        self.session_token = None
        self.session_id = None
        self.user_email = None
        self.websocket = None
    
    def register(self, email: str, password: str, mpin: str):
        """Register a new user"""
        data = {
            "email": email,
            "password": password,
            "mpin": mpin
        }
        
        response = requests.post(f"{BASE_URL}/auth/register", json=data)
        if response.status_code == 200:
            print(f"✅ User registered successfully: {email}")
            return response.json()
        else:
            print(f"❌ Registration failed: {response.json()}")
            return None
    
    def login(self, email: str, password: str, device_id: str = "demo_device"):
        """Login and get session token"""
        data = {
            "email": email,
            "password": password,
            "device_id": device_id
        }
        
        response = requests.post(f"{BASE_URL}/auth/login", json=data)
        if response.status_code == 200:
            result = response.json()
            self.session_token = result["session_token"]
            self.session_id = result["session_id"]
            self.user_email = email
            print(f"✅ Login successful. Session ID: {self.session_id}")
            return result
        else:
            print(f"❌ Login failed: {response.json()}")
            return None
    
    def verify_mpin(self, mpin: str):
        """Verify MPIN"""
        headers = {"Authorization": f"Bearer {self.session_token}"}
        data = {"mpin": mpin}
        
        response = requests.post(f"{BASE_URL}/auth/verify-mpin", json=data, headers=headers)
        if response.status_code == 200:
            print("✅ MPIN verified successfully")
            return response.json()
        else:
            print(f"❌ MPIN verification failed: {response.json()}")
            return None
    
    async def connect_websocket(self):
        """Connect to WebSocket for behavioral data"""
        if not self.session_token or not self.session_id:
            print("❌ Need to login first")
            return
        
        uri = f"{WS_URL}/behavior/{self.session_id}?token={self.session_token}"
        
        try:
            self.websocket = await websockets.connect(uri)
            print("✅ WebSocket connected for behavioral data collection")
            
            # Listen for server messages
            asyncio.create_task(self._listen_to_server())
            
        except Exception as e:
            print(f"❌ WebSocket connection failed: {e}")
    
    async def _listen_to_server(self):
        """Listen for messages from server"""
        try:
            async for message in self.websocket:
                data = json.loads(message)
                await self._handle_server_message(data)
        except websockets.exceptions.ConnectionClosed:
            print("🔌 WebSocket connection closed")
        except Exception as e:
            print(f"❌ Error listening to server: {e}")
    
    async def _handle_server_message(self, data):
        """Handle messages from server"""
        message_type = data.get("type")
        
        if message_type == "connection_established":
            print("📡 Behavioral data collection started")
        
        elif message_type == "mpin_required":
            print("🔐 MPIN verification required!")
            print(f"Reason: {data.get('message', 'Suspicious behavior detected')}")
            
            # Simulate MPIN entry (in real app, this would be user input)
            await asyncio.sleep(1)
            mpin = "1234"  # Demo MPIN
            self.verify_mpin(mpin)
        
        elif message_type == "session_blocked":
            print(f"🚫 Session blocked: {data.get('reason')}")
            if self.websocket:
                await self.websocket.close()
        
        elif message_type == "data_received":
            print("📊 Behavioral data processed by server")
        
        elif message_type == "error":
            print(f"⚠️ Server error: {data.get('message')}")
        
        else:
            print(f"📨 Server message: {data}")
    
    async def send_behavioral_event(self, event_type: str, event_data: dict):
        """Send behavioral event to server"""
        if not self.websocket:
            print("❌ WebSocket not connected")
            return
        
        behavioral_event = {
            "event_type": event_type,
            "data": {
                **event_data,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
        try:
            await self.websocket.send(json.dumps(behavioral_event))
            print(f"📤 Sent behavioral event: {event_type}")
        except Exception as e:
            print(f"❌ Failed to send behavioral event: {e}")
    
    async def simulate_user_behavior(self, duration_seconds: int = 60):
        """Simulate realistic user behavior patterns"""
        print(f"🎭 Simulating user behavior for {duration_seconds} seconds...")
        
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            # Simulate different types of behavioral events
            
            # Normal typing behavior
            if random.random() < 0.3:
                await self.send_behavioral_event("typing_pattern", {
                    "words_per_minute": random.randint(40, 80),
                    "key_press_intervals": [random.uniform(0.1, 0.3) for _ in range(5)]
                })
            
            # Navigation behavior
            if random.random() < 0.2:
                await self.send_behavioral_event("navigation_pattern", {
                    "page_switches_per_minute": random.randint(2, 8),
                    "current_page": random.choice(["dashboard", "transfer", "history", "profile"])
                })
            
            # Mouse movement (simplified)
            if random.random() < 0.4:
                await self.send_behavioral_event("mouse_movement", {
                    "movement_speed": random.uniform(50, 200),
                    "click_pattern": "normal",
                    "idle_time": random.uniform(0.5, 3.0)
                })
            
            # Occasional suspicious behavior for testing
            if random.random() < 0.05:  # 5% chance
                suspicious_events = [
                    ("rapid_clicks", {"clicks_per_second": random.randint(8, 15)}),
                    ("copy_paste_behavior", {"paste_events": random.randint(3, 7)}),
                    ("unusual_navigation", {"rapid_page_switches": True})
                ]
                
                event_type, data = random.choice(suspicious_events)
                await self.send_behavioral_event(event_type, data)
                print("⚠️ Sent suspicious behavioral pattern")
            
            # Random delay between events
            await asyncio.sleep(random.uniform(1, 5))
        
        print("✅ Behavior simulation completed")

async def demo_flow():
    """Demonstrate the complete flow"""
    client = CanaraAIClient()
    
    # Demo user credentials
    email = "demo@canara.com"
    password = "demo123"
    mpin = "1234"
    
    print("🚀 Starting Canara AI Security Backend Demo")
    print("=" * 50)
    
    # Step 1: Register user
    print("\n1️⃣ Registering user...")
    client.register(email, password, mpin)
    
    # Step 2: Login
    print("\n2️⃣ Logging in...")
    client.login(email, password)
    
    if not client.session_token:
        print("❌ Login failed, stopping demo")
        return
    
    # Step 3: Connect WebSocket
    print("\n3️⃣ Connecting to behavioral analysis WebSocket...")
    await client.connect_websocket()
    
    # Step 4: Verify MPIN initially
    print("\n4️⃣ Verifying MPIN...")
    client.verify_mpin(mpin)
    
    # Step 5: Simulate behavioral data
    print("\n5️⃣ Simulating user behavior...")
    await client.simulate_user_behavior(30)  # 30 seconds of behavior
    
    # Step 6: Check session status
    print("\n6️⃣ Checking session status...")
    headers = {"Authorization": f"Bearer {client.session_token}"}
    response = requests.get(f"{BASE_URL}/auth/session-status", headers=headers)
    if response.status_code == 200:
        stats = response.json()
        print(f"📊 Session Risk Score: {stats['risk_score']:.3f}")
        print(f"📊 Behavioral Events: {stats['behavioral_events_count']}")
    
    # Keep connection alive for a bit more
    print("\n7️⃣ Monitoring for security events...")
    await asyncio.sleep(10)
    
    # Cleanup
    if client.websocket:
        await client.websocket.close()
    
    print("\n✅ Demo completed!")

if __name__ == "__main__":
    print("Make sure the FastAPI server is running on http://localhost:8000")
    print("Start server with: python main.py")
    input("Press Enter to start demo...")
    
    asyncio.run(demo_flow())

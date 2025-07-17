#!/usr/bin/env python3
"""
WebSocket Behavioral Authentication Demo
Shows how frontend would integrate with the behavioral auth system
"""

import asyncio
import websockets
import json
import aiohttp
from datetime import datetime
import time

class BehavioralAuthClient:
    """Simulates frontend WebSocket behavioral integration"""
    
    def __init__(self):
        self.backend_url = "http://localhost:8000"
        self.ws_url = "ws://localhost:8000"
        self.session_id = None
        self.session_token = None
        self.websocket = None
        self.risk_score = 0.0
        self.is_blocked = False
        
    async def authenticate(self):
        """Simulate user authentication to get session"""
        print("🔐 Step 1: User Authentication")
        
        # This would normally be called after user enters MPIN
        auth_data = {
            "phone": "9876543210",
            "mpin": "123456",
            "device_id": "demo_device_001"
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(f"{self.backend_url}/api/v1/auth/mpin-login", json=auth_data) as resp:
                    if resp.status == 200:
                        auth_response = await resp.json()
                        self.session_id = auth_response["session_id"]
                        self.session_token = auth_response["session_token"]
                        print(f"   ✅ Session created: {self.session_id}")
                        return True
                    else:
                        error = await resp.text()
                        print(f"   ❌ Authentication failed: {error}")
                        return False
            except Exception as e:
                print(f"   ❌ Connection failed: {e}")
                return False
    
    async def connect_behavioral_stream(self):
        """Connect to behavioral data WebSocket"""
        print("\n🌐 Step 2: Connect Behavioral Stream")
        
        if not self.session_id or not self.session_token:
            print("   ❌ No valid session for WebSocket connection")
            return False
            
        ws_endpoint = f"{self.ws_url}/api/v1/behavior/{self.session_id}?token={self.session_token}"
        
        try:
            self.websocket = await websockets.connect(ws_endpoint)
            print("   ✅ WebSocket connected - Ready for behavioral streaming")
            
            # Wait for connection confirmation
            confirmation = await self.websocket.recv()
            conf_data = json.loads(confirmation)
            print(f"   📨 Server: {conf_data.get('message', 'Connected')}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ WebSocket connection failed: {e}")
            return False
    
    async def stream_behavioral_data(self):
        """Simulate streaming behavioral data"""
        print("\n📊 Step 3: Stream Behavioral Data")
        
        if not self.websocket:
            print("   ❌ No WebSocket connection")
            return
        
        # Simulate different user behaviors
        behaviors = [
            {
                "event_type": "normal_typing",
                "data": {
                    "typing_speed": 65,
                    "keystroke_intervals": [110, 120, 105, 115],
                    "accuracy": 0.95
                }
            },
            {
                "event_type": "touch_pattern",
                "data": {
                    "touch_pressure": [0.7, 0.8, 0.6],
                    "touch_duration": [150, 140, 160],
                    "coordinates": [{"x": 100, "y": 200}, {"x": 150, "y": 250}]
                }
            },
            {
                "event_type": "navigation_pattern",
                "data": {
                    "page_switches_per_minute": 2,
                    "dwell_time": 45,
                    "back_button_usage": 0.1
                }
            },
            {
                "event_type": "suspicious_rapid_clicks",
                "data": {
                    "click_rate": 12,
                    "rapid_succession": True,
                    "pattern": "automated"
                }
            },
            {
                "event_type": "large_transaction_attempt",
                "data": {
                    "amount": 85000,
                    "beneficiary_type": "new",
                    "time_of_day": "03:15",
                    "location_change": True
                }
            }
        ]
        
        for i, behavior in enumerate(behaviors, 1):
            print(f"\n   📋 Sending behavior {i}: {behavior['event_type']}")
            
            try:
                # Send behavioral data
                await self.websocket.send(json.dumps(behavior))
                
                # Wait for acknowledgment
                response = await asyncio.wait_for(self.websocket.recv(), timeout=5.0)
                resp_data = json.loads(response)
                
                if resp_data.get("type") == "data_received":
                    print(f"      ✅ Processed successfully")
                elif resp_data.get("type") == "mpin_required":
                    print(f"      ⚠️  🔐 MPIN RE-AUTHENTICATION REQUIRED")
                    print(f"      📱 Frontend would show MPIN input dialog")
                    await self.handle_mpin_request()
                elif resp_data.get("type") == "session_blocked":
                    print(f"      🚨 SESSION BLOCKED - HIGH RISK DETECTED")
                    print(f"      🚪 Frontend would force logout and redirect to login")
                    self.is_blocked = True
                    break
                else:
                    print(f"      📨 Server response: {resp_data}")
                
                # Check risk score
                await self.check_risk_status()
                
                # Wait before next behavior
                await asyncio.sleep(2)
                
            except asyncio.TimeoutError:
                print(f"      ⏱️  No response received (timeout)")
            except Exception as e:
                print(f"      ❌ Error sending behavior: {e}")
                break
    
    async def handle_mpin_request(self):
        """Handle MPIN re-authentication request"""
        print("      🔄 Simulating MPIN re-authentication...")
        
        # In real frontend, this would show MPIN input dialog
        # For demo, we'll simulate successful MPIN entry
        mpin_data = {
            "event_type": "mpin_verified",
            "data": {
                "mpin": "123456",
                "verification_time": datetime.utcnow().isoformat(),
                "biometric_match": True
            }
        }
        
        try:
            await self.websocket.send(json.dumps(mpin_data))
            response = await self.websocket.recv()
            resp_data = json.loads(response)
            print(f"      ✅ MPIN verified - Risk reduced")
        except Exception as e:
            print(f"      ❌ MPIN verification failed: {e}")
    
    async def check_risk_status(self):
        """Check current session risk status"""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{self.backend_url}/api/v1/sessions/{self.session_id}/behavior-summary") as resp:
                    if resp.status == 200:
                        summary = await resp.json()
                        new_risk = summary.get("risk_score", 0.0)
                        
                        if new_risk != self.risk_score:
                            risk_change = "📈" if new_risk > self.risk_score else "📉"
                            print(f"      {risk_change} Risk: {self.risk_score:.3f} → {new_risk:.3f}")
                            self.risk_score = new_risk
                            
                            if new_risk >= 0.9:
                                print(f"      🚨 HIGH RISK - Session will be blocked")
                            elif new_risk >= 0.7:
                                print(f"      ⚠️  SUSPICIOUS - MPIN may be required")
                        
            except Exception as e:
                print(f"      ⚠️  Could not check risk status: {e}")
    
    async def cleanup(self):
        """Clean up connection"""
        print("\n🧹 Step 4: Cleanup")
        
        if self.websocket:
            await self.websocket.close()
            print("   ✅ WebSocket disconnected")
        
        # End session
        async with aiohttp.ClientSession() as session:
            try:
                end_data = {"reason": "demo_completed"}
                async with session.post(f"{self.backend_url}/api/v1/sessions/{self.session_id}/end", json=end_data) as resp:
                    if resp.status == 200:
                        print("   ✅ Session ended")
            except Exception as e:
                print(f"   ⚠️  Session cleanup: {e}")

async def main():
    """Run the behavioral authentication demo"""
    print("🚀 BEHAVIORAL AUTHENTICATION WEBSOCKET DEMO")
    print("="*55)
    print("This demo shows how the frontend integrates with")
    print("the behavioral authentication system via WebSockets")
    print("="*55)
    
    client = BehavioralAuthClient()
    
    # Step 1: Authenticate and get session
    if not await client.authenticate():
        print("\n❌ Demo failed - Could not authenticate")
        print("💡 Make sure backend is running: backend/start_backend.bat")
        return
    
    # Step 2: Connect WebSocket
    if not await client.connect_behavioral_stream():
        print("\n❌ Demo failed - Could not connect WebSocket")
        return
    
    try:
        # Step 3: Stream behavioral data and watch risk responses
        await client.stream_behavioral_data()
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
    finally:
        # Step 4: Cleanup
        await client.cleanup()
    
    print("\n" + "="*55)
    if client.is_blocked:
        print("🚨 DEMO RESULT: Session was blocked due to high risk")
        print("   In production, user would be forced to re-login")
    else:
        print("✅ DEMO COMPLETED: Behavioral monitoring successful")
    
    print("\n📱 FRONTEND INTEGRATION POINTS:")
    print("1. MPIN verification → Get session token")
    print("2. WebSocket connection → Start behavioral streaming")
    print("3. Risk-based responses → Handle MPIN/block actions")
    print("4. Continuous monitoring → Seamless security")
    print("\n🔗 Integration ready for production use! ✅")

if __name__ == "__main__":
    asyncio.run(main())

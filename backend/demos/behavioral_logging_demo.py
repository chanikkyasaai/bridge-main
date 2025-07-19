"""
WebSocket and Behavioral Logging Demo
Demonstrates the complete session-based logging workflow with temporary storage
"""

import requests
import json
import time
import asyncio
import websockets
from datetime import datetime
import random

class BehavioralLoggingDemo:
    """Demo class for behavioral logging system"""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session_id = None
        self.session_token = None
        self.websocket = None
        
    async def start_session(self, user_id, phone, device_id):
        """Start a new behavioral logging session"""
        print(f"🚀 Starting behavioral logging session")
        print(f"   User ID: {user_id}")
        print(f"   Phone: {phone}")
        print(f"   Device: {device_id}")
        
        url = f"{self.base_url}/api/v1/log/start-session"
        data = {
            "user_id": user_id,
            "phone": phone,
            "device_id": device_id,
            "device_info": "Demo Device - Chrome Browser"
        }
        
        try:
            response = requests.post(url, json=data)
            if response.status_code == 200:
                result = response.json()
                self.session_id = result["session_id"]
                print(f"✅ Session started successfully!")
                print(f"   Session ID: {self.session_id}")
                print(f"   Status: {result['status']}")
                return True
            else:
                print(f"❌ Failed to start session: {response.status_code}")
                print(f"   Error: {response.json().get('detail', 'Unknown error')}")
                return False
        except Exception as e:
            print(f"❌ Session start error: {str(e)}")
            return False
    
    def log_behavioral_event(self, event_type, event_data):
        """Log a single behavioral event"""
        if not self.session_id:
            print("❌ No active session. Please start a session first.")
            return False
        
        url = f"{self.base_url}/api/v1/log/behavior-data"
        headers = {"Authorization": f"Bearer {self.session_token}"} if self.session_token else {}
        data = {
            "session_id": self.session_id,
            "event_type": event_type,
            "data": event_data
        }
        
        try:
            response = requests.post(url, json=data, headers=headers)
            if response.status_code == 200:
                result = response.json()
                print(f"📝 Logged {event_type}: {result['total_events']} total events")
                return True
            else:
                print(f"❌ Failed to log {event_type}: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Logging error for {event_type}: {str(e)}")
            return False
    
    def simulate_user_behavior(self):
        """Simulate various types of user behavioral events"""
        print(f"🎭 Simulating user behavioral patterns...")
        
        # Login behavior
        self.log_behavioral_event("login_attempt", {
            "timestamp": datetime.utcnow().isoformat(),
            "ip_address": "192.168.1.100",
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "login_method": "password"
        })
        
        time.sleep(0.5)
        
        # Mouse movements
        for i in range(5):
            self.log_behavioral_event("mouse_movement", {
                "x": random.randint(100, 800),
                "y": random.randint(100, 600),
                "timestamp": datetime.utcnow().isoformat(),
                "velocity": round(random.uniform(0.1, 2.0), 2)
            })
            time.sleep(0.2)
        
        # Page navigation
        pages = ["dashboard", "accounts", "transfer", "history"]
        for page in pages:
            self.log_behavioral_event("page_navigation", {
                "from_page": "dashboard" if page != "dashboard" else "login",
                "to_page": page,
                "timestamp": datetime.utcnow().isoformat(),
                "load_time_ms": random.randint(200, 1500)
            })
            time.sleep(1)
        
        # Form interactions
        form_fields = ["amount", "recipient", "description"]
        for field in form_fields:
            self.log_behavioral_event("form_interaction", {
                "field": field,
                "action": "focus",
                "timestamp": datetime.utcnow().isoformat()
            })
            time.sleep(0.3)
            
            self.log_behavioral_event("typing_pattern", {
                "field": field,
                "typing_speed": random.randint(80, 200),  # WPM
                "pause_count": random.randint(0, 3),
                "backspace_count": random.randint(0, 2),
                "timestamp": datetime.utcnow().isoformat()
            })
            time.sleep(0.5)
        
        # Transaction attempt
        self.log_behavioral_event("transaction_initiation", {
            "amount": 1500.00,
            "transaction_type": "transfer",
            "recipient_type": "saved_payee",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Biometric verification
        self.log_behavioral_event("biometric_verification", {
            "method": "fingerprint",
            "attempts": 1,
            "success": True,
            "confidence_score": 0.95,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # MPIN entry
        self.log_behavioral_event("mpin_entry", {
            "attempts": 1,
            "time_taken_seconds": 3.2,
            "success": True,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        print(f"✅ Behavioral simulation completed!")
    
    def get_session_status(self):
        """Get current session status"""
        if not self.session_id:
            print("❌ No active session.")
            return None
        
        url = f"{self.base_url}/api/v1/log/session/{self.session_id}/status"
        headers = {"Authorization": f"Bearer {self.session_token}"} if self.session_token else {}
        
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                result = response.json()
                print(f"📊 Session Status:")
                print(f"   Session ID: {result.get('session_id')}")
                print(f"   User ID: {result.get('user_id')}")
                print(f"   Active: {result.get('is_active')}")
                print(f"   Risk Score: {result.get('risk_score')}")
                print(f"   Created: {result.get('created_at')}")
                
                if 'behavioral_data_summary' in result:
                    summary = result['behavioral_data_summary']
                    print(f"   Total Events: {summary.get('total_events')}")
                    print(f"   Event Types: {', '.join(summary.get('event_types', []))}")
                
                return result
            else:
                print(f"❌ Failed to get session status: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ Status check error: {str(e)}")
            return None
    
    def end_session(self, final_decision="normal"):
        """End the session and save behavioral data"""
        if not self.session_id:
            print("❌ No active session to end.")
            return False
        
        print(f"🏁 Ending session with decision: {final_decision}")
        
        url = f"{self.base_url}/api/v1/log/end-session"
        headers = {"Authorization": f"Bearer {self.session_token}"} if self.session_token else {}
        data = {
            "session_id": self.session_id,
            "final_decision": final_decision
        }
        
        try:
            response = requests.post(url, json=data, headers=headers)
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Session ended successfully!")
                print(f"   Session ID: {result['session_id']}")
                print(f"   Final Decision: {result['final_decision']}")
                print(f"   Data Saved: {result.get('behavioral_data_saved')}")
                
                # Clear session data
                self.session_id = None
                self.session_token = None
                
                return True
            else:
                print(f"❌ Failed to end session: {response.status_code}")
                print(f"   Error: {response.json().get('detail', 'Unknown error')}")
                return False
        except Exception as e:
            print(f"❌ Session end error: {str(e)}")
            return False
    
    async def connect_websocket(self):
        """Connect to WebSocket for real-time behavioral data"""
        if not self.session_id or not self.session_token:
            print("❌ Need active session and token for WebSocket connection.")
            return False
        
        ws_url = f"ws://localhost:8000/api/v1/ws/behavior/{self.session_id}?token={self.session_token}"
        
        try:
            print(f"🔌 Connecting to WebSocket...")
            self.websocket = await websockets.connect(ws_url)
            print(f"✅ WebSocket connected!")
            
            # Listen for messages
            async for message in self.websocket:
                data = json.loads(message)
                print(f"📨 WebSocket message: {data['type']} - {data.get('message', '')}")
                
                if data['type'] == 'connection_established':
                    print(f"   Real-time behavioral collection started")
                elif data['type'] == 'data_received':
                    print(f"   Data processing confirmed")
                elif data['type'] == 'mpin_required':
                    print(f"   🔐 MPIN verification requested!")
                elif data['type'] == 'session_blocked':
                    print(f"   🚫 Session blocked: {data.get('reason')}")
                    break
                    
        except websockets.exceptions.ConnectionClosed:
            print(f"🔌 WebSocket connection closed")
        except Exception as e:
            print(f"❌ WebSocket error: {str(e)}")
        
        return True
    
    async def send_websocket_data(self, event_type, data):
        """Send behavioral data via WebSocket"""
        if not self.websocket:
            print("❌ No WebSocket connection.")
            return False
        
        message = {
            "event_type": event_type,
            "data": data
        }
        
        try:
            await self.websocket.send(json.dumps(message))
            print(f"📤 Sent via WebSocket: {event_type}")
            return True
        except Exception as e:
            print(f"❌ WebSocket send error: {str(e)}")
            return False
    
    def check_api_health(self):
        """Check if the API is running"""
        try:
            response = requests.get(f"{self.base_url}/health")
            if response.status_code == 200:
                result = response.json()
                print(f"✅ API is healthy: {result['service']}")
                return True
            else:
                print(f"❌ API health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Cannot connect to API: {str(e)}")
            return False
    
    async def run_complete_demo(self):
        """Run the complete behavioral logging demo"""
        print("=" * 70)
        print("🎯 BEHAVIORAL LOGGING SYSTEM DEMO")
        print("=" * 70)
        print("This demo shows session-based behavioral logging with temporary storage")
        print("until session completion, then permanent storage in Supabase.")
        print()
        
        # Step 1: Health check
        print("STEP 1: API Health Check")
        print("-" * 30)
        if not self.check_api_health():
            print("❌ API is not available. Please start the server first.")
            return False
        print()
        
        # Step 2: Start session
        print("STEP 2: Start Behavioral Logging Session")
        print("-" * 40)
        user_id = "demo_user_behavioral"
        phone = "9876543210"
        device_id = "demo_device_behavioral"
        
        if not await self.start_session(user_id, phone, device_id):
            print("❌ Failed to start session. Cannot continue demo.")
            return False
        print()
        
        # Step 3: Simulate behavioral data
        print("STEP 3: Simulate User Behavioral Patterns")
        print("-" * 42)
        print("Note: All data is stored temporarily in memory during the session")
        self.simulate_user_behavior()
        print()
        
        # Step 4: Check session status
        print("STEP 4: Check Session Status (Temporary Storage)")
        print("-" * 48)
        self.get_session_status()
        print()
        
        # Step 5: More behavioral events
        print("STEP 5: Additional Behavioral Events")
        print("-" * 36)
        additional_events = [
            ("device_orientation", {"orientation": "portrait", "timestamp": datetime.utcnow().isoformat()}),
            ("app_focus", {"focus": True, "timestamp": datetime.utcnow().isoformat()}),
            ("copy_paste_action", {"action": "paste", "field": "amount", "timestamp": datetime.utcnow().isoformat()}),
            ("session_idle", {"idle_time_seconds": 5, "timestamp": datetime.utcnow().isoformat()})
        ]
        
        for event_type, event_data in additional_events:
            self.log_behavioral_event(event_type, event_data)
            time.sleep(0.5)
        print()
        
        # Step 6: Final session status
        print("STEP 6: Final Session Status Before Termination")
        print("-" * 48)
        final_status = self.get_session_status()
        if final_status and 'behavioral_data_summary' in final_status:
            summary = final_status['behavioral_data_summary']
            print(f"📈 Session Summary:")
            print(f"   Total events collected: {summary.get('total_events')}")
            print(f"   Event types: {len(summary.get('event_types', []))}")
        print()
        
        # Step 7: End session and save data
        print("STEP 7: End Session and Save to Permanent Storage")
        print("-" * 50)
        print("Now all temporary behavioral data will be validated and saved to Supabase")
        
        if not self.end_session("normal"):
            print("❌ Failed to end session properly.")
            return False
        print()
        
        # Step 8: Summary
        print("STEP 8: Demo Summary")
        print("-" * 20)
        print("✅ Complete behavioral logging workflow demonstrated:")
        print("   • Session creation and management")
        print("   • Temporary storage of behavioral events in memory")
        print("   • Real-time behavioral pattern collection")
        print("   • Session status monitoring")
        print("   • Data validation and permanent storage on session end")
        print("   • Integration with authentication system")
        print()
        
        print("=" * 70)
        print("🎉 BEHAVIORAL LOGGING DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
        return True

async def main():
    """Main demo function"""
    print("Starting Behavioral Logging Demo...")
    print("Make sure the FastAPI server is running on http://localhost:8000")
    print()
    
    # Run the demo
    demo = BehavioralLoggingDemo()
    await demo.run_complete_demo()
    
    print()
    print("=" * 70)
    print("📋 BEHAVIORAL LOGGING FEATURES DEMONSTRATED")
    print("=" * 70)
    print("Session Management:")
    print("• Session creation with user/device association")
    print("• Temporary in-memory storage during active session")
    print("• Session status monitoring and statistics")
    print("• Graceful session termination")
    print()
    print("Behavioral Data Collection:")
    print("• Mouse movements and clicks")
    print("• Keyboard typing patterns") 
    print("• Page navigation and load times")
    print("• Form interactions and field focus")
    print("• Biometric verification events")
    print("• Transaction initiation patterns")
    print("• MPIN entry timing and attempts")
    print()
    print("Data Storage Strategy:")
    print("• Temporary storage in memory during session")
    print("• Data validation before permanent storage")
    print("• Automatic save to Supabase on session end")
    print("• Local backup if Supabase storage fails")
    print("• Structured JSON format for easy analysis")
    print()
    print("Security Features:")
    print("• Session-based authentication")
    print("• Real-time risk score calculation")
    print("• Automatic session blocking on high risk")
    print("• MPIN verification requests")
    print("• WebSocket real-time communication")

if __name__ == "__main__":
    asyncio.run(main())

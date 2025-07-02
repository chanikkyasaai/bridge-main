"""
Supabase Behavioral Logging Demo Client
This script demonstrates the new Supabase-integrated behavioral logging system
"""

import asyncio
import json
import websockets
import httpx
from datetime import datetime
import time

# API Base URL
BASE_URL = "http://localhost:8000/api/v1"

async def demo_supabase_behavioral_logging():
    """
    Demo script showing the complete behavioral logging workflow with Supabase
    """
    print("🚀 Starting Canara AI Behavioral Logging Demo with Supabase")
    print("=" * 60)
    
    async with httpx.AsyncClient() as client:
        # Step 1: Register a user
        print("\n📝 Step 1: Registering a new user...")
        register_data = {
            "phone": "9876543210",
            "password": "SecurePassword123",
            "mpin": "123456"
        }
        
        try:
            response = await client.post(f"{BASE_URL}/auth/register", json=register_data)
            if response.status_code == 200:
                reg_result = response.json()
                user_id = reg_result.get("user_id", "user_123")
                print(f"✅ User registered: {reg_result}")
            else:
                print(f"⚠️ Registration response: {response.status_code} - {response.text}")
                user_id = "user_123"  # Fallback for demo
        except Exception as e:
            print(f"❌ Registration failed: {e}")
            user_id = "user_123"  # Fallback for demo
        
        # Step 2: Login
        print("\n🔐 Step 2: Logging in...")
        login_data = {
            "phone": "9876543210",
            "password": "SecurePassword123",
            "device_id": "demo_device_001"
        }
        
        try:
            response = await client.post(f"{BASE_URL}/auth/login", json=login_data)
            if response.status_code == 200:
                login_result = response.json()
                session_token = login_result["session_token"]
                session_id = login_result["session_id"]
                print(f"✅ Login successful! Session ID: {session_id}")
            else:
                print(f"❌ Login failed: {response.status_code} - {response.text}")
                return
        except Exception as e:
            print(f"❌ Login failed: {e}")
            return
        
        # Step 3: Start behavioral logging session
        print("\n📊 Step 3: Starting behavioral logging session...")
        start_session_data = {
            "user_id": user_id,
            "phone": "9876543210",
            "device_id": "demo_device_001",
            "device_info": "Android 12, Chrome 98"
        }
        
        headers = {"Authorization": f"Bearer {session_token}"}
        
        try:
            response = await client.post(f"{BASE_URL}/log/start-session", 
                                       json=start_session_data, headers=headers)
            if response.status_code == 200:
                log_session = response.json()
                log_session_id = log_session["session_id"]
                print(f"✅ Logging session started: {log_session_id}")
            else:
                print(f"❌ Failed to start logging session: {response.status_code} - {response.text}")
                log_session_id = session_id  # Use main session as fallback
        except Exception as e:
            print(f"❌ Failed to start logging session: {e}")
            log_session_id = session_id  # Use main session as fallback
        
        # Step 4: Connect WebSocket for real-time behavioral data
        print("\n🔌 Step 4: Connecting WebSocket for real-time data...")
        websocket_url = f"ws://localhost:8000/api/v1/ws/behavior/{session_id}?token={session_token}"
        
        try:
            async with websockets.connect(websocket_url) as websocket:
                # Receive connection confirmation
                confirmation = await websocket.recv()
                print(f"✅ WebSocket connected: {json.loads(confirmation)}")
                
                # Step 5: Send behavioral data via WebSocket
                print("\n📈 Step 5: Sending behavioral data via WebSocket...")
                
                behavioral_events = [
                    {
                        "event_type": "login_success",
                        "data": {
                            "timestamp": datetime.now().isoformat(),
                            "device_fingerprint": "abc123def456",
                            "ip_address": "192.168.1.100"
                        }
                    },
                    {
                        "event_type": "page_view",
                        "data": {
                            "page": "dashboard",
                            "timestamp": datetime.now().isoformat(),
                            "load_time": 1.2,
                            "viewport": "1920x1080"
                        }
                    },
                    {
                        "event_type": "button_click",
                        "data": {
                            "button_id": "transfer_btn",
                            "click_duration": 0.15,
                            "coordinates": [150, 200],
                            "pressure": 0.8
                        }
                    },
                    {
                        "event_type": "typing_pattern",
                        "data": {
                            "field": "amount",
                            "typing_speed": 45,
                            "keystroke_dynamics": [0.1, 0.12, 0.08, 0.11],
                            "dwell_times": [0.08, 0.09, 0.07, 0.10]
                        }
                    },
                    {
                        "event_type": "mouse_movement",
                        "data": {
                            "pattern": "smooth",
                            "velocity": 150,
                            "path_length": 300,
                            "acceleration": 12.5
                        }
                    },
                    {
                        "event_type": "form_interaction",
                        "data": {
                            "form_name": "transfer_form",
                            "field_focus_order": ["amount", "beneficiary", "remarks"],
                            "time_spent": 45.2
                        }
                    },
                    {
                        "event_type": "transaction_attempt",
                        "data": {
                            "amount": 5000,
                            "beneficiary": "John Doe",
                            "account_type": "savings",
                            "transaction_type": "neft"
                        }
                    }
                ]
                
                for i, event in enumerate(behavioral_events):
                    await websocket.send(json.dumps(event))
                    response = await websocket.recv()
                    response_data = json.loads(response)
                    print(f"  📊 Event {i+1}: {event['event_type']} - {response_data['status']}")
                    await asyncio.sleep(0.5)  # Simulate real-time gaps
                
                print("✅ All WebSocket behavioral events sent successfully!")
                
        except Exception as e:
            print(f"❌ WebSocket connection failed: {e}")
        
        # Step 6: Send additional behavioral data via REST API
        print("\n📋 Step 6: Sending additional data via REST API...")
        
        rest_events = [
            {
                "session_id": session_id,
                "event_type": "security_check",
                "data": {
                    "check_type": "device_fingerprint",
                    "result": "verified",
                    "confidence": 0.95,
                    "factors_checked": ["screen_resolution", "timezone", "plugins"]
                }
            },
            {
                "session_id": session_id,
                "event_type": "mpin_entry_pattern",
                "data": {
                    "entry_speed": 2.5,
                    "accuracy": 1.0,
                    "attempts": 1,
                    "finger_pressure_pattern": [0.7, 0.8, 0.6, 0.9, 0.7, 0.8]
                }
            },
            {
                "session_id": session_id,
                "event_type": "navigation_pattern",
                "data": {
                    "pages_visited": ["dashboard", "transfer", "beneficiary"],
                    "time_per_page": [30, 120, 45],
                    "back_button_usage": 2
                }
            },
            {
                "session_id": session_id,
                "event_type": "idle_behavior",
                "data": {
                    "idle_duration": 180,
                    "idle_reason": "phone_call",
                    "resumed_activity": "transaction_continue"
                }
            }
        ]
        
        for event in rest_events:
            try:
                response = await client.post(f"{BASE_URL}/log/behavior-data", 
                                           json=event, headers=headers)
                if response.status_code == 200:
                    result = response.json()
                    print(f"  📊 REST Event: {event['event_type']} - Total: {result['total_events']} events")
                else:
                    print(f"  ❌ REST Event failed: {response.status_code}")
            except Exception as e:
                print(f"  ❌ REST Event error: {e}")
            
            await asyncio.sleep(0.2)
        
        # Step 7: Check session status
        print("\n📊 Step 7: Checking session status...")
        try:
            response = await client.get(f"{BASE_URL}/log/session/{session_id}/status", 
                                      headers=headers)
            if response.status_code == 200:
                status = response.json()
                print(f"✅ Session Status:")
                print(f"  - Session ID: {status['session_id']}")
                print(f"  - Active: {status['is_active']}")
                print(f"  - Risk Score: {status['risk_score']}")
                print(f"  - Total Events: {status['behavioral_data_summary']['total_events']}")
                print(f"  - Event Types: {', '.join(status['behavioral_data_summary']['event_types'])}")
                if status['behavioral_data_summary']['last_event']:
                    print(f"  - Last Event: {status['behavioral_data_summary']['last_event']}")
            else:
                print(f"❌ Failed to get session status: {response.status_code}")
        except Exception as e:
            print(f"❌ Status check error: {e}")
        
        # Step 8: Test MPIN verification (creates security event)
        print("\n🔒 Step 8: Testing MPIN verification...")
        mpin_data = {"mpin": "123456"}
        
        try:
            response = await client.post(f"{BASE_URL}/auth/verify-mpin", 
                                       json=mpin_data, headers=headers)
            if response.status_code == 200:
                result = response.json()
                print(f"✅ MPIN verified: {result['message']}")
            else:
                print(f"⚠️ MPIN verification: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ MPIN verification error: {e}")
        
        # Step 9: End session (this uploads all data to Supabase Storage)
        print("\n🔚 Step 9: Ending session and uploading to Supabase Storage...")
        end_session_data = {
            "session_id": session_id,
            "final_decision": "normal"
        }
        
        try:
            response = await client.post(f"{BASE_URL}/log/end-session", 
                                       json=end_session_data, headers=headers)
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Session ended successfully!")
                print(f"  - Session ID: {result['session_id']}")
                print(f"  - Behavioral data saved to Supabase: {result['behavioral_data_saved']}")
                print(f"  - Final decision: {result['final_decision']}")
                print(f"  - Upload timestamp: {result['timestamp']}")
            else:
                print(f"❌ Failed to end session: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ End session error: {e}")
        
        # Step 10: Try to retrieve logs from Supabase Storage
        print("\n📥 Step 10: Retrieving logs from Supabase Storage...")
        try:
            # Note: This might fail if session is already terminated
            response = await client.get(f"{BASE_URL}/log/session/{session_id}/logs", 
                                      headers=headers)
            if response.status_code == 200:
                logs_data = response.json()
                print(f"✅ Logs retrieved successfully!")
                print(f"  - File path: {logs_data['file_path']}")
                print(f"  - Total events in file: {len(logs_data['logs']['logs'])}")
                print(f"  - User ID: {logs_data['logs']['user_id']}")
                print(f"  - Session ID: {logs_data['logs']['session_id']}")
            else:
                print(f"⚠️ Could not retrieve logs: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Log retrieval error: {e}")
    
    print("\n🎉 Demo completed successfully!")
    print("\n" + "="*60)
    print("📝 SUMMARY OF SUPABASE INTEGRATION:")
    print("="*60)
    print("1. ✅ User registration in Supabase 'users' table")
    print("2. ✅ Session creation in Supabase 'sessions' table")
    print("3. ✅ Real-time behavioral data collection (stored in memory)")
    print("4. ✅ Security events logged in 'security_events' table")
    print("5. ✅ Session termination with data upload to Supabase Storage")
    print("6. ✅ JSON logs saved in structured format:")
    print("     📁 Bucket: behavior-logs")
    print("     📄 Path: logs/{user_id}/{session_id}.json")
    print("7. ✅ Log retrieval from Supabase Storage")
    print("\n🗂️ Data Storage Architecture:")
    print("  📊 Database Tables:")
    print("    - users: User credentials and profile")
    print("    - sessions: Session metadata and status")
    print("    - security_events: Security decisions and scores")
    print("  💾 Storage Bucket:")
    print("    - behavior-logs: Complete behavioral data as JSON")
    print("\n💡 Key Features Demonstrated:")
    print("  - Continuous WebSocket data collection")
    print("  - In-memory buffering during session")
    print("  - Batch upload to storage on session end")
    print("  - Structured JSON with user_id, session_id, and logs")
    print("  - Integration with ML risk scoring")
    print("  - Security event logging")

def print_setup_instructions():
    """Print setup instructions for the demo"""
    print("\n🔧 SETUP REQUIREMENTS:")
    print("="*40)
    print("1. 📋 Environment Setup:")
    print("   - Ensure .env file has SUPABASE_URL and SUPABASE_SERVICE_KEY")
    print("   - Install requirements: pip install -r requirements.txt")
    print("\n2. 🗄️ Supabase Setup:")
    print("   - Create the following tables using the provided schema:")
    print("     • users (id, phone, password_hash, mpin_hash, created_at)")
    print("     • sessions (id, user_id, started_at, ended_at, device_info, etc.)")
    print("     • security_events (id, session_id, level, decision, reason, etc.)")
    print("   - Create storage bucket: 'behavior-logs'")
    print("   - Ensure bucket allows public access or configure policies")
    print("\n3. 🚀 Start the Backend:")
    print("   - Run: python main.py")
    print("   - Server should start on http://localhost:8000")
    print("\n4. 📡 Network Requirements:")
    print("   - Ensure WebSocket connections are allowed")
    print("   - Check firewall settings for port 8000")

if __name__ == "__main__":
    print_setup_instructions()
    print("\n" + "="*60)
    print("Press Enter to start the Supabase Behavioral Logging Demo...")
    input()
    
    asyncio.run(demo_supabase_behavioral_logging())

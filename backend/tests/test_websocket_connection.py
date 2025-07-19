#!/usr/bin/env python3
"""
Test WebSocket connection for behavioral logging
"""

import asyncio
import websockets
import json
import requests

BASE_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000"

async def test_websocket_connection():
    """Test WebSocket behavioral logging connection"""
    
    print("🧪 Testing WebSocket Behavioral Logging Connection...")
    
    try:
        # Step 1: First, let's do MPIN login to get session tokens
        print("\n1️⃣ Performing MPIN login to get session tokens...")
        
        login_response = requests.post(f"{BASE_URL}/api/v1/auth/mpin-login", json={
            'phone': '9876543210',
            'mpin': '123456',
            'device_id': 'test_device_ws'
        })
        
        if login_response.status_code != 200:
            print(f"❌ MPIN login failed: {login_response.status_code} - {login_response.text}")
            return False
        
        login_data = login_response.json()
        session_id = login_data.get('session_id')
        session_token = login_data.get('session_token')
        
        print(f"✅ Login successful!")
        print(f"   Session ID: {session_id}")
        print(f"   Session Token: {session_token[:50]}...")
        
        if not session_id or not session_token:
            print("❌ Missing session_id or session_token in response")
            return False
        
        # Step 2: Connect to WebSocket
        print(f"\n2️⃣ Connecting to WebSocket...")
        ws_url = f"{WS_URL}/api/v1/ws/behavior/{session_id}?token={session_token}"
        print(f"   WebSocket URL: ws://localhost:8000/api/v1/ws/behavior/{session_id}?token={session_token[:30]}...")
        
        async with websockets.connect(ws_url) as websocket:
            print("✅ WebSocket connected successfully!")
            
            # Step 3: Wait for connection confirmation
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                connection_msg = json.loads(response)
                print(f"✅ Connection confirmed: {connection_msg}")
                
                # Step 4: Send test behavioral data
                print(f"\n3️⃣ Sending test behavioral data...")
                test_events = [
                    {
                        "event_type": "mouse_movement",
                        "data": {
                            "x": 100,
                            "y": 200,
                            "timestamp": "2025-07-03T10:30:00Z"
                        }
                    },
                    {
                        "event_type": "key_press",
                        "data": {
                            "key": "Enter",
                            "timestamp": "2025-07-03T10:30:01Z"
                        }
                    }
                ]
                
                for event in test_events:
                    await websocket.send(json.dumps(event))
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    ack = json.loads(response)
                    print(f"✅ Event sent and acknowledged: {event['event_type']} -> {ack.get('status')}")
                
                print(f"\n🎉 WebSocket test completed successfully!")
                return True
                
            except asyncio.TimeoutError:
                print("❌ Timeout waiting for WebSocket response")
                return False
            except json.JSONDecodeError as e:
                print(f"❌ JSON decode error: {e}")
                return False
                
    except websockets.exceptions.ConnectionClosed as e:
        print(f"❌ WebSocket connection closed: {e}")
        return False
    except websockets.exceptions.InvalidURI as e:
        print(f"❌ Invalid WebSocket URI: {e}")
        return False
    except websockets.exceptions.WebSocketException as e:
        print(f"❌ WebSocket error: {e}")
        return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to backend server. Make sure it's running on localhost:8000")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(test_websocket_connection())

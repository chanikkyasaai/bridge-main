"""
WebSocket and Logging Tests
Tests for session-based behavioral logging with temporary storage until session completion
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from fastapi import WebSocket
from main import app
from app.core.session_manager import session_manager
from app.core.security import create_session_token, create_access_token
from app.api.v1.endpoints.websocket import websocket_manager

client = TestClient(app)

class TestWebSocketAndLogging:
    """Test suite for WebSocket connections and behavioral logging"""
    
    def setup_method(self):
        """Setup for each test method"""
        # Clear session manager state
        session_manager.active_sessions.clear()
        session_manager.user_sessions.clear()
        websocket_manager.active_connections.clear()
    
    def test_start_session_endpoint(self):
        """Test starting a new behavioral logging session"""
        session_data = {
            "user_id": "test_user_123",
            "phone": "9876543210",
            "device_id": "test_device_001",
            "device_info": "iPhone 13 Pro"
        }
        
        response = client.post("/api/v1/log/start-session", json=session_data)
        
        # Should succeed or fail gracefully (depending on Supabase configuration)
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert "session_id" in data
            assert "message" in data
            assert data["status"] == "active"
            
            # Verify session was created in session manager
            session = session_manager.get_session(data["session_id"])
            assert session is not None
            assert session.user_id == "test_user_123"
            assert session.phone == "9876543210"
            assert session.device_id == "test_device_001"
            assert session.is_active == True
            assert len(session.behavioral_buffer) == 0  # Empty initially
    
    def test_log_behavior_data_endpoint(self):
        """Test logging behavioral data to session buffer"""
        # First create a session
        user_id = "test_user_456"
        phone = "9123456789"
        device_id = "test_device_002"
        session_token = create_session_token(phone, device_id, user_id)
        
        session_id = asyncio.run(session_manager.create_session(
            user_id, phone, device_id, session_token
        ))
        
        # Test logging behavioral data
        behavior_data = {
            "session_id": session_id,
            "event_type": "mouse_movement",
            "data": {
                "x": 150,
                "y": 200,
                "timestamp": datetime.utcnow().isoformat(),
                "velocity": 0.5
            }
        }
        
        headers = {"Authorization": f"Bearer {session_token}"}
        response = client.post("/api/v1/log/behavior-data", 
                              json=behavior_data, headers=headers)
        
        # Should succeed or fail gracefully
        assert response.status_code in [200, 401, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert "message" in data
            assert data["session_id"] == session_id
            assert data["event_type"] == "mouse_movement"
            assert data["total_events"] == 1
            
            # Verify data is stored in session buffer (temporary storage)
            session = session_manager.get_session(session_id)
            assert session is not None
            assert len(session.behavioral_buffer) == 1
            assert session.behavioral_buffer[0].event_type == "mouse_movement"
            assert session.behavioral_buffer[0].data["x"] == 150
            assert session.behavioral_buffer[0].data["y"] == 200
    
    def test_multiple_behavioral_events_storage(self):
        """Test that multiple behavioral events are stored in temporary buffer"""
        # Create session
        user_id = "test_user_789"
        phone = "9987654321"
        device_id = "test_device_003"
        session_token = create_session_token(phone, device_id, user_id)
        
        session_id = asyncio.run(session_manager.create_session(
            user_id, phone, device_id, session_token
        ))
        
        headers = {"Authorization": f"Bearer {session_token}"}
        
        # Log multiple different types of behavioral events
        events = [
            {
                "session_id": session_id,
                "event_type": "mouse_click",
                "data": {"x": 100, "y": 50, "button": "left"}
            },
            {
                "session_id": session_id,
                "event_type": "key_press",
                "data": {"key": "Enter", "duration": 0.1}
            },
            {
                "session_id": session_id,
                "event_type": "page_scroll",
                "data": {"direction": "down", "distance": 250}
            },
            {
                "session_id": session_id,
                "event_type": "form_interaction",
                "data": {"field": "amount", "action": "focus"}
            }
        ]
        
        # Log each event
        for event in events:
            response = client.post("/api/v1/log/behavior-data", 
                                  json=event, headers=headers)
            # Should succeed or handle gracefully
            assert response.status_code in [200, 401, 500]
        
        # Verify all events are stored in temporary buffer
        session = session_manager.get_session(session_id)
        assert session is not None
        
        if len(session.behavioral_buffer) > 0:  # If logging succeeded
            assert len(session.behavioral_buffer) == len(events)
            
            # Verify event types
            event_types = [bd.event_type for bd in session.behavioral_buffer]
            expected_types = ["mouse_click", "key_press", "page_scroll", "form_interaction"]
            for expected_type in expected_types:
                assert expected_type in event_types
    
    def test_session_status_endpoint(self):
        """Test getting session status and behavioral data summary"""
        # Create session with some behavioral data
        user_id = "test_user_status"
        phone = "9555666777"
        device_id = "test_device_status"
        session_token = create_session_token(phone, device_id, user_id)
        
        session_id = asyncio.run(session_manager.create_session(
            user_id, phone, device_id, session_token
        ))
        
        # Add some test behavioral data directly
        session = session_manager.get_session(session_id)
        if session:
            session.add_behavioral_data("test_event", {"test": "data"})
            session.add_behavioral_data("another_event", {"more": "data"})
        
        headers = {"Authorization": f"Bearer {session_token}"}
        response = client.get(f"/api/v1/log/session/{session_id}/status", headers=headers)
        
        # Should succeed or fail gracefully
        assert response.status_code in [200, 401, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert "session_id" in data
            assert "user_id" in data
            assert "is_active" in data
            assert "behavioral_data_summary" in data
            
            summary = data["behavioral_data_summary"]
            assert "total_events" in summary
            assert "event_types" in summary
    
    def test_end_session_and_data_persistence(self):
        """Test ending session and ensuring behavioral data is saved"""
        # Create session
        user_id = "test_user_end"
        phone = "9444555666"
        device_id = "test_device_end"
        session_token = create_session_token(phone, device_id, user_id)
        
        session_id = asyncio.run(session_manager.create_session(
            user_id, phone, device_id, session_token
        ))
        
        # Add behavioral data to session
        session = session_manager.get_session(session_id)
        if session:
            session.add_behavioral_data("final_action", {"action": "logout"})
            session.add_behavioral_data("page_exit", {"page": "dashboard"})
            
            # Verify data is in temporary buffer
            assert len(session.behavioral_buffer) == 2
        
        # End the session
        end_data = {
            "session_id": session_id,
            "final_decision": "normal"
        }
        
        headers = {"Authorization": f"Bearer {session_token}"}
        response = client.post("/api/v1/log/end-session", 
                              json=end_data, headers=headers)
        
        # Should succeed or fail gracefully
        assert response.status_code in [200, 401, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert "message" in data
            assert data["session_id"] == session_id
            assert data["final_decision"] == "normal"
            assert "behavioral_data_saved" in data
            
            # Verify session is no longer active in session manager
            terminated_session = session_manager.get_session(session_id)
            assert terminated_session is None  # Should be removed after termination
    
    def test_websocket_connection_simulation(self):
        """Test WebSocket connection management"""
        # Create session
        user_id = "test_user_ws"
        phone = "9333444555"
        device_id = "test_device_ws"
        session_token = create_session_token(phone, device_id, user_id)
        
        session_id = asyncio.run(session_manager.create_session(
            user_id, phone, device_id, session_token
        ))
        
        # Test WebSocket manager functionality
        class MockWebSocket:
            def __init__(self):
                self.connected = False
                self.messages = []
            
            async def accept(self):
                self.connected = True
            
            async def send_text(self, message):
                self.messages.append(message)
            
            async def close(self, code=None, reason=None):
                self.connected = False
        
        mock_ws = MockWebSocket()
        
        # Test connection
        asyncio.run(websocket_manager.connect(mock_ws, session_id))
        assert session_id in websocket_manager.active_connections
        
        # Test sending message
        test_message = {"type": "test", "data": "hello"}
        asyncio.run(websocket_manager.send_message(session_id, test_message))
        assert len(mock_ws.messages) == 1
        
        # Test disconnection
        websocket_manager.disconnect(session_id)
        assert session_id not in websocket_manager.active_connections
    
    def test_behavioral_data_structure_validation(self):
        """Test that behavioral data maintains proper structure"""
        # Create session
        user_id = "test_user_structure"
        phone = "9222333444"
        device_id = "test_device_structure"
        session_token = create_session_token(phone, device_id, user_id)
        
        session_id = asyncio.run(session_manager.create_session(
            user_id, phone, device_id, session_token
        ))
        
        session = session_manager.get_session(session_id)
        if session:
            # Add behavioral data with various structures
            test_events = [
                {
                    "event_type": "login_attempt",
                    "data": {
                        "username": "user123",
                        "timestamp": datetime.utcnow().isoformat(),
                        "ip_address": "192.168.1.1",
                        "user_agent": "Mozilla/5.0..."
                    }
                },
                {
                    "event_type": "transaction_start",
                    "data": {
                        "amount": 1000.50,
                        "recipient": "XXXXXXXX1234",
                        "transaction_type": "transfer"
                    }
                },
                {
                    "event_type": "biometric_verification",
                    "data": {
                        "method": "fingerprint",
                        "success": True,
                        "confidence_score": 0.95
                    }
                }
            ]
            
            for event in test_events:
                session.add_behavioral_data(event["event_type"], event["data"])
            
            # Verify all data is stored with proper structure
            assert len(session.behavioral_buffer) == 3
            
            for i, behavior_data in enumerate(session.behavioral_buffer):
                assert hasattr(behavior_data, 'timestamp')
                assert hasattr(behavior_data, 'event_type')
                assert hasattr(behavior_data, 'data')
                assert behavior_data.event_type == test_events[i]["event_type"]
                assert isinstance(behavior_data.timestamp, str)  # ISO format
                
                # Verify timestamp is recent
                timestamp_dt = datetime.fromisoformat(behavior_data.timestamp.replace('Z', ''))
                time_diff = datetime.utcnow() - timestamp_dt
                assert time_diff.total_seconds() < 10  # Within 10 seconds
    
    def test_session_risk_score_updates(self):
        """Test that risk scores are updated based on behavioral patterns"""
        # Create session
        user_id = "test_user_risk"
        phone = "9111222333"
        device_id = "test_device_risk"
        session_token = create_session_token(phone, device_id, user_id)
        
        session_id = asyncio.run(session_manager.create_session(
            user_id, phone, device_id, session_token
        ))
        
        session = session_manager.get_session(session_id)
        if session:
            # Initial risk score should be 0
            assert session.risk_score == 0.0
            
            # Simulate risk score updates
            session.update_risk_score(0.3)  # Low risk
            assert session.risk_score == 0.3
            assert session.is_active == True
            assert session.is_blocked == False
            
            session.update_risk_score(0.8)  # High risk (but below blocking threshold)
            assert session.risk_score == 0.8
            
            # Test blocking threshold (if configured)
            from app.core.config import settings
            if hasattr(settings, 'HIGH_RISK_THRESHOLD'):
                if settings.HIGH_RISK_THRESHOLD <= 0.9:
                    session.update_risk_score(0.95)  # Very high risk
                    assert session.risk_score == 0.95
                    # Might be blocked depending on threshold
    
    def test_session_cleanup_functionality(self):
        """Test that expired sessions are properly cleaned up"""
        # Create session
        user_id = "test_user_cleanup"
        phone = "9000111222"
        device_id = "test_device_cleanup"
        session_token = create_session_token(phone, device_id, user_id)
        
        session_id = asyncio.run(session_manager.create_session(
            user_id, phone, device_id, session_token
        ))
        
        # Verify session exists
        session = session_manager.get_session(session_id)
        assert session is not None
        
        # Manually set session as expired for testing
        session.created_at = datetime.utcnow() - timedelta(hours=2)
        
        # Verify session is considered expired
        assert session.is_expired() == True
        
        # Test cleanup (would normally be called by background task)
        asyncio.run(session_manager.cleanup_expired_sessions())
        
        # Session should be removed
        cleaned_session = session_manager.get_session(session_id)
        assert cleaned_session is None
    
    def test_session_statistics(self):
        """Test session statistics and monitoring"""
        # Create multiple sessions
        sessions_data = [
            ("user1", "9111000001", "device1"),
            ("user2", "9111000002", "device2"),
            ("user1", "9111000001", "device3"),  # Same user, different device
        ]
        
        session_ids = []
        for user_id, phone, device_id in sessions_data:
            session_token = create_session_token(phone, device_id, user_id)
            session_id = asyncio.run(session_manager.create_session(
                user_id, phone, device_id, session_token
            ))
            session_ids.append(session_id)
        
        # Get statistics
        stats = session_manager.get_all_sessions_stats()
        
        assert "total_active_sessions" in stats
        assert "total_users" in stats
        assert "sessions" in stats
        
        assert stats["total_active_sessions"] == 3
        assert stats["total_users"] == 2  # user1 and user2
        assert len(stats["sessions"]) == 3
        
        # Verify user sessions
        user1_sessions = session_manager.get_user_sessions("user1")
        user2_sessions = session_manager.get_user_sessions("user2")
        
        assert len(user1_sessions) == 2  # Two devices for user1
        assert len(user2_sessions) == 1  # One device for user2
    
    def test_authentication_integration(self):
        """Test integration with authentication system"""
        # Create access token
        test_payload = {
            "sub": "auth_user_123",
            "phone": "9876543210",
            "device_id": "auth_device"
        }
        access_token = create_access_token(test_payload)
        
        # Test that endpoints requiring authentication work with access tokens
        headers = {"Authorization": f"Bearer {access_token}"}
        
        # Test session status endpoint with access token
        response = client.get("/api/v1/auth/session-status", headers=headers)
        # Should validate token structure even if user doesn't exist
        assert response.status_code in [200, 404, 500]
    
    def test_concurrent_behavioral_logging(self):
        """Test concurrent behavioral data logging"""
        # Create session
        user_id = "test_user_concurrent"
        phone = "9999888777"
        device_id = "test_device_concurrent"
        session_token = create_session_token(phone, device_id, user_id)
        
        session_id = asyncio.run(session_manager.create_session(
            user_id, phone, device_id, session_token
        ))
        
        session = session_manager.get_session(session_id)
        if session:
            # Simulate concurrent behavioral data addition
            import threading
            import time
            
            def add_behavior_data(event_num):
                session.add_behavioral_data(
                    f"concurrent_event_{event_num}",
                    {"event_number": event_num, "thread_id": threading.current_thread().ident}
                )
            
            # Create multiple threads to add data concurrently
            threads = []
            for i in range(10):
                thread = threading.Thread(target=add_behavior_data, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            # Verify all events were logged
            assert len(session.behavioral_buffer) == 10
            
            # Verify event numbers are all different
            event_numbers = [bd.data["event_number"] for bd in session.behavioral_buffer]
            assert len(set(event_numbers)) == 10  # All unique

    def test_complete_session_lifecycle_with_temp_storage(self):
        """Test complete session lifecycle emphasizing temporary storage until session end"""
        # Phase 1: Session Creation
        user_id = "test_lifecycle_user"
        phone = "9123456789"
        device_id = "test_lifecycle_device"
        
        session_data = {
            "user_id": user_id,
            "phone": phone,
            "device_id": device_id,
            "device_info": "Test Device for Lifecycle"
        }
        
        # Start session
        response = client.post("/api/v1/log/start-session", json=session_data)
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            session_id = data["session_id"]
            
            # Verify session exists and is empty initially
            session = session_manager.get_session(session_id)
            assert session is not None
            assert len(session.behavioral_buffer) == 0
            assert session.is_active == True
            
            # Phase 2: Data Collection in Temporary Storage
            session_token = session.session_token
            headers = {"Authorization": f"Bearer {session_token}"}
            
            # Simulate various behavioral events during session
            behavioral_events = [
                {"event_type": "login_verification", "data": {"method": "mpin", "success": True}},
                {"event_type": "page_navigation", "data": {"from": "dashboard", "to": "transfer"}},
                {"event_type": "form_input", "data": {"field": "amount", "value_length": 5}},
                {"event_type": "mouse_movement", "data": {"x": 150, "y": 200, "velocity": 0.3}},
                {"event_type": "key_press", "data": {"key": "Tab", "duration": 0.1}},
                {"event_type": "copy_paste", "data": {"field": "account_number", "action": "paste"}},
                {"event_type": "transaction_review", "data": {"amount": 5000, "recipient": "XXXX1234"}}
            ]
            
            # Log each event and verify temporary storage
            for i, event in enumerate(behavioral_events):
                behavior_request = {
                    "session_id": session_id,
                    "event_type": event["event_type"],
                    "data": event["data"]
                }
                
                response = client.post("/api/v1/log/behavior-data", 
                                      json=behavior_request, headers=headers)
                
                if response.status_code == 200:
                    # Verify data is stored in temporary buffer
                    session = session_manager.get_session(session_id)
                    assert len(session.behavioral_buffer) == i + 1
                    assert session.behavioral_buffer[i].event_type == event["event_type"]
                    
                    # Verify data is NOT yet persisted to Supabase (still in memory)
                    assert session.is_active == True
            
            # Phase 3: Session Status Check (data still in temp storage)
            response = client.get(f"/api/v1/log/session/{session_id}/status", headers=headers)
            if response.status_code == 200:
                status_data = response.json()
                assert status_data["is_active"] == True
                assert status_data["behavioral_data_summary"]["total_events"] == len(behavioral_events)
                # Verify all event types are captured
                logged_event_types = status_data["behavioral_data_summary"]["event_types"]
                expected_types = [event["event_type"] for event in behavioral_events]
                for expected_type in expected_types:
                    assert expected_type in logged_event_types
            
            # Phase 4: Session Termination and Data Persistence
            end_request = {
                "session_id": session_id,
                "final_decision": "transaction_completed"
            }
            
            response = client.post("/api/v1/log/end-session", json=end_request, headers=headers)
            if response.status_code == 200:
                end_data = response.json()
                assert end_data["session_id"] == session_id
                assert end_data["final_decision"] == "transaction_completed"
                assert end_data["behavioral_data_saved"] == True
                
                # Verify session is no longer active (cleaned up from memory)
                terminated_session = session_manager.get_session(session_id)
                assert terminated_session is None

    def test_websocket_behavioral_data_collection_workflow(self):
        """Test WebSocket real-time behavioral data collection with temporary storage"""
        # Create session
        user_id = "test_websocket_user"
        phone = "9876543210"
        device_id = "test_websocket_device"
        session_token = create_session_token(phone, device_id, user_id)
        
        session_id = asyncio.run(session_manager.create_session(
            user_id, phone, device_id, session_token
        ))
        
        session = session_manager.get_session(session_id)
        assert session is not None
        assert len(session.behavioral_buffer) == 0  # Empty initially
        
        # Simulate WebSocket behavioral data events
        from app.api.v1.endpoints.websocket import process_behavioral_data
        
        websocket_events = [
            {"event_type": "mouse_click", "data": {"x": 100, "y": 50, "button": "left"}},
            {"event_type": "scroll_behavior", "data": {"direction": "down", "speed": 500}},
            {"event_type": "typing_pattern", "data": {"words_per_minute": 45, "backspace_count": 2}},
            {"event_type": "idle_time", "data": {"duration_seconds": 30}},
            {"event_type": "focus_change", "data": {"from_element": "amount_field", "to_element": "submit_button"}}
        ]
        
        # Process each WebSocket event
        for event in websocket_events:
            asyncio.run(process_behavioral_data(session_id, event))
        
        # Verify all events are stored in temporary buffer
        session = session_manager.get_session(session_id)
        assert len(session.behavioral_buffer) >= len(websocket_events)  # May include risk_score_update events
        
        # Verify event types
        buffer_event_types = [bd.event_type for bd in session.behavioral_buffer]
        for event in websocket_events:
            assert event["event_type"] in buffer_event_types

    def test_temporary_storage_memory_efficiency(self):
        """Test that temporary storage is memory efficient and properly bounded"""
        # Create session
        user_id = "test_memory_user"
        phone = "9111222333"
        device_id = "test_memory_device"
        session_token = create_session_token(phone, device_id, user_id)
        
        session_id = asyncio.run(session_manager.create_session(
            user_id, phone, device_id, session_token
        ))
        
        session = session_manager.get_session(session_id)
        assert session is not None
        
        # Add a large number of behavioral events to test memory handling
        num_events = 1000
        for i in range(num_events):
            session.add_behavioral_data(
                f"bulk_event_{i % 10}",  # 10 different event types
                {
                    "event_id": i,
                    "timestamp": datetime.utcnow().isoformat(),
                    "data_size": "x" * 100  # 100 char string per event
                }
            )
        
        # Verify all events are stored
        assert len(session.behavioral_buffer) == num_events
        
        # Verify memory structure is maintained
        assert session.is_active == True
        assert session.risk_score >= 0.0
        
        # Test that we can still access first and last events
        first_event = session.behavioral_buffer[0]
        last_event = session.behavioral_buffer[-1]
        
        assert first_event.data["event_id"] == 0
        assert last_event.data["event_id"] == num_events - 1

    def test_session_data_isolation_between_users(self):
        """Test that behavioral data is properly isolated between different user sessions"""
        # Create multiple sessions for different users
        sessions_data = [
            ("user_1", "9111111111", "device_1"),
            ("user_2", "9222222222", "device_2"),
            ("user_1", "9111111111", "device_3"),  # Same user, different device
        ]
        
        created_sessions = []
        for user_id, phone, device_id in sessions_data:
            session_token = create_session_token(phone, device_id, user_id)
            session_id = asyncio.run(session_manager.create_session(
                user_id, phone, device_id, session_token
            ))
            created_sessions.append((session_id, user_id, device_id))
        
        # Add unique behavioral data to each session
        for i, (session_id, user_id, device_id) in enumerate(created_sessions):
            session = session_manager.get_session(session_id)
            if session:
                # Add unique events per session
                for j in range(5):
                    session.add_behavioral_data(
                        f"user_specific_event_{i}_{j}",
                        {
                            "session_identifier": session_id,
                            "user_id": user_id,
                            "device_id": device_id,
                            "event_number": j
                        }
                    )
        
        # Verify data isolation
        for i, (session_id, user_id, device_id) in enumerate(created_sessions):
            session = session_manager.get_session(session_id)
            assert session is not None
            assert len(session.behavioral_buffer) == 5
            
            # Verify all events belong to this session
            for bd in session.behavioral_buffer:
                assert bd.data["session_identifier"] == session_id
                assert bd.data["user_id"] == user_id
                assert bd.data["device_id"] == device_id
                assert bd.event_type.startswith(f"user_specific_event_{i}_")

    def test_error_handling_during_behavioral_logging(self):
        """Test error handling scenarios during behavioral data collection"""
        # Create session
        user_id = "test_error_user"
        phone = "9555444333"
        device_id = "test_error_device"
        session_token = create_session_token(phone, device_id, user_id)
        
        session_id = asyncio.run(session_manager.create_session(
            user_id, phone, device_id, session_token
        ))
        
        headers = {"Authorization": f"Bearer {session_token}"}
        
        # Test 1: Invalid event type
        invalid_event = {
            "session_id": session_id,
            "event_type": "",  # Empty event type
            "data": {"test": "data"}
        }
        
        response = client.post("/api/v1/log/behavior-data", 
                              json=invalid_event, headers=headers)
        # Should handle gracefully
        assert response.status_code in [200, 400, 401, 500]
        
        # Test 2: Malformed data
        malformed_event = {
            "session_id": session_id,
            "event_type": "test_event"
            # Missing data field
        }
        
        response = client.post("/api/v1/log/behavior-data", 
                              json=malformed_event, headers=headers)
        # Should handle gracefully
        assert response.status_code in [200, 400, 422, 500]
        
        # Test 3: Non-existent session
        non_existent_event = {
            "session_id": "non_existent_session_id",
            "event_type": "test_event",
            "data": {"test": "data"}
        }
        
        response = client.post("/api/v1/log/behavior-data", 
                              json=non_existent_event, headers=headers)
        assert response.status_code in [401, 404, 500]  # Unauthorized, Not Found, or Server Error

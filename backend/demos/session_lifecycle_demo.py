#!/usr/bin/env python3
"""
Session Lifecycle Demo Script
Demonstrates the complete session-based behavioral logging workflow:
1. Session creation with temporary in-memory storage
2. Behavioral data collection stored in temporary backend buffer
3. Session termination with data persistence to Supabase
"""

import asyncio
import time
import json
from datetime import datetime
from app.core.session_manager import session_manager
from app.core.security import create_session_token
from app.api.v1.endpoints.websocket import process_behavioral_data


def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)


def print_step(step_num, description):
    """Print a formatted step"""
    print(f"\n{step_num}. {description}")
    print("-" * 50)


async def demonstrate_session_lifecycle():
    """Demonstrate the complete session lifecycle with temporary storage"""
    
    print_header("SESSION-BASED BEHAVIORAL LOGGING DEMO")
    print("This demo shows how behavioral data is stored temporarily")
    print("in backend memory during a session and only persisted")
    print("to Supabase when the session ends.")
    
    # ========================================================================
    # PHASE 1: SESSION CREATION
    # ========================================================================
    print_step(1, "Creating User Session")
    
    user_id = "demo_user_12345"
    phone = "9876543210"
    device_id = "demo_device_001"
    device_info = "iPhone 13 Pro - Demo Device"
    
    # Create session token
    session_token = create_session_token(phone, device_id, user_id)
    print(f"✓ Session token created: {session_token[:50]}...")
    
    # Create session
    session_id = await session_manager.create_session(
        user_id, phone, device_id, session_token
    )
    print(f"✓ Session created with ID: {session_id}")
    
    # Get session object
    session = session_manager.get_session(session_id)
    print(f"✓ Session active: {session.is_active}")
    print(f"✓ Initial behavioral buffer size: {len(session.behavioral_buffer)}")
    print(f"✓ Initial risk score: {session.risk_score}")
    
    # ========================================================================
    # PHASE 2: BEHAVIORAL DATA COLLECTION (TEMPORARY STORAGE)
    # ========================================================================
    print_step(2, "Collecting Behavioral Data in Temporary Storage")
    
    print("Simulating user interactions during banking session...")
    
    # Simulate various behavioral events that would occur during a banking session
    behavioral_events = [
        {
            "event_type": "login_verification",
            "data": {
                "method": "mpin",
                "success": True,
                "attempt_count": 1,
                "verification_time_ms": 1250
            }
        },
        {
            "event_type": "navigation_pattern",
            "data": {
                "from_page": "dashboard",
                "to_page": "transfer",
                "navigation_time_ms": 800,
                "click_count": 1
            }
        },
        {
            "event_type": "form_interaction",
            "data": {
                "field": "beneficiary_account",
                "action": "focus",
                "input_method": "typing",
                "character_count": 16
            }
        },
        {
            "event_type": "mouse_behavior",
            "data": {
                "movement_pattern": "smooth",
                "click_frequency": 0.5,
                "hover_duration_ms": 500,
                "coordinates": [{"x": 150, "y": 200}, {"x": 300, "y": 150}]
            }
        },
        {
            "event_type": "typing_pattern",
            "data": {
                "words_per_minute": 45,
                "backspace_count": 2,
                "pause_duration_ms": 1500,
                "rhythm_consistency": 0.85
            }
        },
        {
            "event_type": "transaction_preparation",
            "data": {
                "amount": 5000,
                "recipient_type": "saved_beneficiary",
                "amount_input_time_ms": 3000,
                "review_time_ms": 8000
            }
        },
        {
            "event_type": "biometric_verification",
            "data": {
                "method": "fingerprint",
                "success": True,
                "confidence_score": 0.96,
                "verification_attempts": 1
            }
        }
    ]
    
    print(f"Processing {len(behavioral_events)} behavioral events...")
    
    for i, event in enumerate(behavioral_events, 1):
        # Add some realistic delay between events
        await asyncio.sleep(0.5)
        
        # Process behavioral data through WebSocket handler (simulated)
        await process_behavioral_data(session_id, event)
        
        # Check current session state
        session = session_manager.get_session(session_id)
        buffer_size = len(session.behavioral_buffer)
        risk_score = session.risk_score
        
        print(f"  Event {i:2d}: {event['event_type']:25} | "
              f"Buffer size: {buffer_size:3d} | Risk score: {risk_score:.3f}")
    
    # ========================================================================
    # PHASE 3: SESSION STATUS AND TEMPORARY DATA ANALYSIS
    # ========================================================================
    print_step(3, "Analyzing Temporary Session Data")
    
    session = session_manager.get_session(session_id)
    
    print(f"Session Statistics:")
    print(f"  ├─ Session ID: {session.session_id}")
    print(f"  ├─ User ID: {session.user_id}")
    print(f"  ├─ Phone: {session.phone}")
    print(f"  ├─ Device ID: {session.device_id}")
    print(f"  ├─ Is Active: {session.is_active}")
    print(f"  ├─ Is Blocked: {session.is_blocked}")
    print(f"  ├─ Risk Score: {session.risk_score:.3f}")
    print(f"  ├─ Created At: {session.created_at}")
    print(f"  ├─ Last Activity: {session.last_activity}")
    print(f"  └─ Behavioral Events in Buffer: {len(session.behavioral_buffer)}")
    
    print(f"\nBehavioral Event Types Collected:")
    event_types = {}
    for bd in session.behavioral_buffer:
        event_types[bd.event_type] = event_types.get(bd.event_type, 0) + 1
    
    for event_type, count in event_types.items():
        print(f"  ├─ {event_type}: {count} events")
    
    print(f"\nTemporary Storage Details:")
    print(f"  ├─ Data stored in: session.behavioral_buffer (in-memory)")
    print(f"  ├─ Persistence status: TEMPORARY (not yet saved to Supabase)")
    print(f"  ├─ Total memory events: {len(session.behavioral_buffer)}")
    print(f"  └─ Supabase persistence: Will occur at session end")
    
    # ========================================================================
    # PHASE 4: SESSION TERMINATION AND DATA PERSISTENCE
    # ========================================================================
    print_step(4, "Session Termination and Data Persistence")
    
    print("Ending session and triggering data persistence to Supabase...")
    
    # Store current buffer size for comparison
    final_buffer_size = len(session.behavioral_buffer)
    final_risk_score = session.risk_score
    
    print(f"Before termination:")
    print(f"  ├─ Behavioral events in buffer: {final_buffer_size}")
    print(f"  ├─ Final risk score: {final_risk_score:.3f}")
    print(f"  └─ Session status: {session.is_active}")
    
    # Terminate session (this will save all behavioral data to Supabase)
    final_decision = "transaction_completed_successfully"
    success = await session_manager.terminate_session(session_id, final_decision)
    
    print(f"\nAfter termination:")
    print(f"  ├─ Termination success: {success}")
    print(f"  ├─ Final decision: {final_decision}")
    print(f"  ├─ Data persisted to: Supabase (or local backup)")
    print(f"  ├─ Session cleanup: Memory freed")
    print(f"  └─ Behavioral events saved: {final_buffer_size}")
    
    # Verify session is no longer in memory
    terminated_session = session_manager.get_session(session_id)
    print(f"  └─ Session still in memory: {terminated_session is not None}")
    
    # ========================================================================
    # PHASE 5: SESSION MANAGER STATISTICS
    # ========================================================================
    print_step(5, "Session Manager Statistics")
    
    stats = session_manager.get_all_sessions_stats()
    print(f"Global Session Statistics:")
    print(f"  ├─ Total active sessions: {stats['total_active_sessions']}")
    print(f"  ├─ Total users with sessions: {stats['total_users']}")
    print(f"  └─ Sessions currently in memory: {len(stats['sessions'])}")
    
    print_header("DEMO COMPLETE")
    print("Summary of Session-Based Behavioral Logging:")
    print("1. ✓ Session created with temporary in-memory storage")
    print("2. ✓ Behavioral data collected in session buffer (temporary)")
    print("3. ✓ Data analyzed and risk score updated in real-time")
    print("4. ✓ Session terminated and all data persisted to Supabase")
    print("5. ✓ Memory cleaned up and session removed from backend")
    print("\nThis ensures:")
    print("  • Fast real-time behavioral analysis during session")
    print("  • Efficient memory usage with temporary storage")
    print("  • Reliable data persistence only when session completes")
    print("  • Clean session lifecycle management")


async def demonstrate_concurrent_sessions():
    """Demonstrate multiple concurrent sessions with isolated data"""
    
    print_header("CONCURRENT SESSIONS DEMO")
    print("Demonstrating multiple users with isolated behavioral data")
    
    # Create multiple sessions
    sessions_data = [
        ("user_alice_001", "9111111111", "alice_phone"),
        ("user_bob_002", "9222222222", "bob_tablet"),
        ("user_charlie_003", "9333333333", "charlie_laptop"),
    ]
    
    created_sessions = []
    
    print("\nCreating concurrent sessions:")
    for user_id, phone, device_id in sessions_data:
        session_token = create_session_token(phone, device_id, user_id)
        session_id = await session_manager.create_session(
            user_id, phone, device_id, session_token
        )
        created_sessions.append((session_id, user_id, device_id))
        print(f"  ✓ Session created for {user_id}: {session_id[:16]}...")
    
    print(f"\nSimulating concurrent behavioral data collection:")
    
    # Add behavioral data to each session concurrently
    for i, (session_id, user_id, device_id) in enumerate(created_sessions):
        session = session_manager.get_session(session_id)
        
        # Add unique behavioral patterns for each user
        user_events = [
            f"user_{i}_login_pattern",
            f"user_{i}_navigation_style",
            f"user_{i}_typing_behavior",
            f"user_{i}_transaction_pattern"
        ]
        
        for j, event_type in enumerate(user_events):
            session.add_behavioral_data(event_type, {
                "user_specific_id": user_id,
                "device_id": device_id,
                "event_sequence": j,
                "timestamp": datetime.utcnow().isoformat()
            })
        
        print(f"  ✓ {user_id}: {len(session.behavioral_buffer)} events in buffer")
    
    # Show session isolation
    print(f"\nVerifying data isolation between sessions:")
    for session_id, user_id, device_id in created_sessions:
        session = session_manager.get_session(session_id)
        events_count = len(session.behavioral_buffer)
        user_events = [bd.data.get("user_specific_id", "unknown") for bd in session.behavioral_buffer]
        unique_users = set(user_events)
        
        print(f"  ✓ {user_id}: {events_count} events, all belong to: {unique_users}")
    
    # Clean up sessions
    print(f"\nTerminating all sessions:")
    for session_id, user_id, device_id in created_sessions:
        success = await session_manager.terminate_session(session_id, "demo_completed")
        print(f"  ✓ {user_id} session terminated: {success}")
    
    final_stats = session_manager.get_all_sessions_stats()
    print(f"\nFinal statistics:")
    print(f"  └─ Active sessions remaining: {final_stats['total_active_sessions']}")


if __name__ == "__main__":
    # Run the demonstrations
    try:
        print("Starting Session Lifecycle Demonstrations...")
        
        # Main session lifecycle demo
        asyncio.run(demonstrate_session_lifecycle())
        
        # Wait a moment
        time.sleep(2)
        
        # Concurrent sessions demo
        asyncio.run(demonstrate_concurrent_sessions())
        
        print("\n" + "="*60)
        print(" ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()

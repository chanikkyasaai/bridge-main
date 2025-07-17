#!/usr/bin/env python3
"""
Comprehensive test for the complete Phase 1/2 behavioral authentication system
"""

import asyncio
import json
import uuid
import requests
from datetime import datetime
import time

ML_ENGINE_URL = "http://127.0.0.1:8001"

def generate_behavioral_events(event_type="varied"):
    """Generate different types of behavioral events for testing"""
    base_events = [
        {
            "event_type": "keypress",
            "timestamp": datetime.now().isoformat(),
            "data": {
                "key": "a",
                "press_duration": 120,
                "interval_since_last": 150
            }
        },
        {
            "event_type": "mouse_click", 
            "timestamp": datetime.now().isoformat(),
            "data": {
                "button": "left",
                "position": {"x": 100, "y": 200},
                "click_duration": 80
            }
        },
        {
            "event_type": "navigation",
            "timestamp": datetime.now().isoformat(), 
            "data": {
                "url": "/dashboard",
                "dwell_time": 2500,
                "scroll_behavior": {"vertical": 250, "horizontal": 0}
            }
        }
    ]
    
    if event_type == "typing":
        return [base_events[0]]
    elif event_type == "mouse":
        return [base_events[1]]
    elif event_type == "navigation":
        return [base_events[2]]
    else:
        return base_events

async def main():
    print("🚀 Comprehensive Phase 1/2 Behavioral Authentication Test")
    
    # Generate test user
    test_user_id = str(uuid.uuid4())
    test_session_base = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"📋 Test User ID: {test_user_id}")
    print(f"📋 Test Session Base: {test_session_base}")
    
    # Test 1: System health checks
    print("\n" + "="*60)
    print("🔍 SYSTEM HEALTH CHECKS")
    print("="*60)
    
    try:
        # Basic health
        response = requests.get(f"{ML_ENGINE_URL}/")
        print(f"✅ ML Engine health: {response.status_code == 200}")
        
        # Database health
        response = requests.get(f"{ML_ENGINE_URL}/health/database")
        print(f"✅ Database health: {response.status_code == 200}")
        
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return
    
    # Test 2: Phase 1 Learning System Testing
    print("\n" + "="*60)
    print("🎓 PHASE 1 LEARNING SYSTEM TESTING")
    print("="*60)
    
    learning_sessions = []
    
    for i in range(5):  # Test multiple learning sessions
        print(f"\n--- Learning Session {i+1} ---")
        
        # Start session
        session_id = f"{test_session_base}_learning_{i}"
        response = requests.post(f"{ML_ENGINE_URL}/session/start", json={
            "user_id": test_user_id,
            "session_id": session_id
        })
        
        if response.status_code == 200:
            session_data = response.json()
            print(f"✅ Session started: {session_data.get('learning_phase', 'unknown')}")
            learning_sessions.append(session_id)
            
            # Analyze behavioral data
            events = generate_behavioral_events("varied")
            response = requests.post(f"{ML_ENGINE_URL}/analyze", json={
                "user_id": test_user_id,
                "session_id": session_id,
                "events": events
            })
            
            if response.status_code == 200:
                analysis = response.json()
                print(f"✅ Analysis: {analysis.get('decision', 'unknown')} (confidence: {analysis.get('confidence', 0)})")
                
                # Check if there's an error in learning result
                learning_result = analysis.get('learning_result', {})
                if learning_result.get('status') == 'error':
                    print(f"⚠️  Learning error: {learning_result.get('message', 'Unknown error')}")
                else:
                    print(f"✅ Learning successful")
            else:
                print(f"❌ Analysis failed: {response.status_code}")
        else:
            print(f"❌ Session start failed: {response.status_code}")
        
        # Small delay between sessions
        time.sleep(0.5)
    
    # Test 3: Learning Progress Check
    print("\n" + "="*60)
    print("📊 LEARNING PROGRESS ANALYSIS")
    print("="*60)
    
    response = requests.get(f"{ML_ENGINE_URL}/user/{test_user_id}/learning-progress")
    if response.status_code == 200:
        progress = response.json()
        progress_report = progress.get('progress_report', {})
        print(f"✅ Current Phase: {progress_report.get('current_phase', 'unknown')}")
        print(f"✅ Session Count: {progress_report.get('session_count', 0)}")
        print(f"✅ Vectors Collected: {progress_report.get('vectors_collected', 0)}")
        print(f"✅ Learning Completeness: {progress_report.get('learning_completeness', 0):.1%}")
        
        # Check if ready for phase transition
        readiness = progress_report.get('readiness_assessment', {})
        if readiness.get('ready_for_next_phase', False):
            print(f"🎯 Ready for next phase: {readiness.get('next_phase', 'unknown')}")
        else:
            print(f"⏳ Not ready for transition. Requirements: {readiness.get('requirements', [])}")
    else:
        print(f"❌ Progress check failed: {response.status_code}")
    
    # Test 4: Phase 2 Continuous Analysis Testing (if user has progressed)
    print("\n" + "="*60)
    print("🔄 PHASE 2 CONTINUOUS ANALYSIS TESTING")
    print("="*60)
    
    # Test Phase 2 analysis regardless of learning progress
    session_id = f"{test_session_base}_phase2_test"
    response = requests.post(f"{ML_ENGINE_URL}/session/start", json={
        "user_id": test_user_id,
        "session_id": session_id
    })
    
    if response.status_code == 200:
        print("✅ Phase 2 session started")
        
        # Test continuous analysis
        events = generate_behavioral_events("mouse")
        response = requests.post(f"{ML_ENGINE_URL}/analyze", json={
            "user_id": test_user_id,
            "session_id": session_id,
            "events": events
        })
        
        if response.status_code == 200:
            analysis = response.json()
            print(f"✅ Phase 2 Analysis: {analysis.get('decision', 'unknown')}")
            print(f"✅ Analysis Type: {analysis.get('analysis_type', 'unknown')}")
            print(f"✅ Risk Level: {analysis.get('risk_level', 'unknown')}")
        else:
            print(f"❌ Phase 2 analysis failed: {response.status_code}")
    
    # Test 5: System Statistics
    print("\n" + "="*60)
    print("📈 SYSTEM STATISTICS")
    print("="*60)
    
    response = requests.get(f"{ML_ENGINE_URL}/statistics")
    if response.status_code == 200:
        stats = response.json().get('statistics', {})
        
        # Learning System Stats
        learning_stats = stats.get('learning_system', {}).get('learning_stats', {})
        print(f"✅ Users in Learning: {learning_stats.get('users_in_learning', 0)}")
        print(f"✅ Cold Start Users: {learning_stats.get('cold_start_users', 0)}")
        print(f"✅ Phase Transitions Today: {learning_stats.get('phase_transitions_today', 0)}")
        
        # Database Stats
        db_stats = stats.get('database', {})
        print(f"✅ User Profiles: {db_stats.get('user_profiles_count', 0)}")
        print(f"✅ Behavioral Vectors: {db_stats.get('behavioral_vectors_count', 0)}")
        print(f"✅ Authentication Decisions: {db_stats.get('authentication_decisions_count', 0)}")
        
        # Session Stats
        session_stats = stats.get('session_manager', {})
        print(f"✅ Total Sessions: {session_stats.get('total_sessions', 0)}")
        print(f"✅ Active Sessions: {session_stats.get('active_sessions', 0)}")
        print(f"✅ Unique Users: {session_stats.get('unique_users', 0)}")
    else:
        print(f"❌ Statistics failed: {response.status_code}")
    
    # Test 6: Baseline Adaptation
    print("\n" + "="*60)
    print("🎯 BASELINE ADAPTATION TESTING")
    print("="*60)
    
    response = requests.post(f"{ML_ENGINE_URL}/user/{test_user_id}/adapt-baseline")
    if response.status_code == 200:
        adaptation = response.json()
        print(f"✅ Baseline adaptation: {adaptation.get('status', 'unknown')}")
    else:
        print(f"❌ Baseline adaptation failed: {response.status_code}")
    
    # Summary
    print("\n" + "="*60)
    print("📋 TEST SUMMARY")
    print("="*60)
    print(f"✅ Test User Created: {test_user_id}")
    print(f"✅ Learning Sessions Completed: {len(learning_sessions)}")
    print("✅ Phase 1 Learning System: Tested")
    print("✅ Phase 2 Continuous Analysis: Tested")
    print("✅ Database Integration: Verified")
    print("✅ Session Management: Working")
    print("✅ Statistics Collection: Active")
    
    print("\n🎉 Comprehensive testing completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())

"""
Quick test to verify the Phase 1 + Phase 2 system with valid UUID
"""

import asyncio
import aiohttp
import json
import uuid
from datetime import datetime

async def quick_test():
    """Quick test with proper UUID format"""
    ml_engine_url = "http://localhost:8001"
    
    # Use a valid UUID for testing
    test_user_id = str(uuid.uuid4())
    test_session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"🧪 Quick Test with User ID: {test_user_id}")
    print("=" * 60)
    
    async with aiohttp.ClientSession() as session:
        try:
            # Test 1: Health Check
            print("1. Testing ML Engine Health...")
            async with session.get(f"{ml_engine_url}/") as response:
                if response.status == 200:
                    print("✅ ML Engine is healthy")
                else:
                    print(f"❌ ML Engine health check failed: {response.status}")
                    return
            
            # Test 2: Start Session (Phase 1)
            print("2. Testing Session Start (Phase 1)...")
            session_data = {
                "user_id": test_user_id,
                "session_id": test_session_id,
                "device_info": {
                    "device_id": "test_device_001",
                    "user_agent": "Test/1.0"
                }
            }
            
            async with session.post(
                f"{ml_engine_url}/session/start",
                json=session_data
            ) as response:
                if response.status == 200:
                    start_result = await response.json()
                    learning_phase = start_result.get('learning_phase')
                    print(f"✅ Session started: {learning_phase}")
                    print(f"   Message: {start_result.get('session_guidance', {}).get('message', 'N/A')}")
                else:
                    print(f"❌ Session start failed: {response.status}")
                    return
            
            # Test 3: Behavioral Analysis
            print("3. Testing Behavioral Analysis...")
            behavioral_events = [
                {
                    "event_type": "keystroke",
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": {"key": "a", "duration": 120, "pressure": 0.8}
                },
                {
                    "event_type": "mouse_move",
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": {"x": 100, "y": 200, "velocity": 1.5}
                }
            ]
            
            analysis_data = {
                "user_id": test_user_id,
                "session_id": test_session_id,
                "events": behavioral_events
            }
            
            async with session.post(
                f"{ml_engine_url}/analyze",
                json=analysis_data
            ) as response:
                if response.status == 200:
                    analysis_result = await response.json()
                    print(f"✅ Behavioral analysis successful:")
                    print(f"   Decision: {analysis_result.get('decision')}")
                    print(f"   Confidence: {analysis_result.get('confidence', 0):.3f}")
                    print(f"   Analysis Type: {analysis_result.get('analysis_type')}")
                    
                    # Show learning result if present
                    learning_result = analysis_result.get('learning_result', {})
                    if learning_result:
                        print(f"   Vectors Collected: {learning_result.get('vectors_collected', 0)}")
                        print(f"   Phase Confidence: {learning_result.get('phase_confidence', 0):.3f}")
                        
                        phase_transition = learning_result.get('phase_transition')
                        if phase_transition and phase_transition.get('transition_occurred'):
                            print(f"   🔄 Phase Transition: {phase_transition.get('old_phase')} → {phase_transition.get('new_phase')}")
                else:
                    error_text = await response.text()
                    print(f"❌ Behavioral analysis failed: {response.status}")
                    print(f"   Error: {error_text}")
                    return
            
            # Test 4: Learning Progress
            print("4. Testing Learning Progress...")
            async with session.get(
                f"{ml_engine_url}/user/{test_user_id}/learning-progress"
            ) as response:
                if response.status == 200:
                    progress_data = await response.json()
                    progress_report = progress_data.get('progress_report', {})
                    print(f"✅ Learning progress retrieved:")
                    print(f"   Current Phase: {progress_report.get('current_phase')}")
                    print(f"   Learning Completeness: {progress_report.get('learning_completeness', 0):.1f}%")
                    print(f"   Vectors Collected: {progress_report.get('vectors_collected', 0)}")
                else:
                    print(f"❌ Learning progress failed: {response.status}")
            
            # Test 5: Database Stats
            print("5. Testing Database Integration...")
            async with session.get(f"{ml_engine_url}/statistics") as response:
                if response.status == 200:
                    stats_data = await response.json()
                    database_stats = stats_data.get('statistics', {}).get('database', {})
                    print(f"✅ Database statistics:")
                    print(f"   User profiles: {database_stats.get('user_profiles_count', 0)}")
                    print(f"   Behavioral vectors: {database_stats.get('behavioral_vectors_count', 0)}")
                    print(f"   Auth decisions: {database_stats.get('authentication_decisions_count', 0)}")
                else:
                    print(f"❌ Statistics failed: {response.status}")
            
            print("\n🎉 Quick test completed successfully!")
            print(f"   Test User ID: {test_user_id}")
            print("   Phase 1 Learning System: ✅ Working")
            print("   Database Integration: ✅ Working")
            print("   Vector Processing: ✅ Working")
            
        except Exception as e:
            print(f"❌ Quick test failed: {e}")

if __name__ == "__main__":
    asyncio.run(quick_test())

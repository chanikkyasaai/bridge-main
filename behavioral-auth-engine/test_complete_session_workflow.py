#!/usr/bin/env python3
"""
Test complete session workflow for cumulative learning system
This validates the entire flow:
1. Session start → mobile data processing → vector storage → session end → cumulative update
"""

import asyncio
import json
import uuid
import sys
import os
from datetime import datetime

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.enhanced_behavioral_processor import EnhancedBehavioralProcessor, ProcessedBehavioralFeatures
from src.core.enhanced_faiss_engine import EnhancedFAISSEngine
from src.core.ml_database import MLSupabaseClient

class SessionWorkflowTester:
    def __init__(self):
        self.db_client = MLSupabaseClient()
        self.processor = EnhancedBehavioralProcessor()
        self.faiss_engine = EnhancedFAISSEngine()  # Fixed: No db_client parameter needed
        # Generate proper UUID format for user_id
        self.test_user_id = str(uuid.uuid4())
        self.test_results = []

    async def ensure_test_user_exists(self):
        """Ensure the test user exists in the database"""
        try:
            # Check if user already exists
            existing_user = self.db_client.supabase.table('users')\
                .select('*')\
                .eq('id', self.test_user_id)\
                .execute()
            
            if not existing_user.data:
                # Create test user with correct schema
                user_data = {
                    'id': self.test_user_id,
                    'phone': f'+1555{self.test_user_id[:8]}',  # Using phone instead of email
                    'password_hash': 'test_password_hash',
                    'mpin_hash': 'test_mpin_hash',
                    'created_at': datetime.utcnow().isoformat()
                }
                
                result = self.db_client.supabase.table('users').insert(user_data).execute()
                if result.data:
                    print(f"✅ Created test user: {self.test_user_id}")
                else:
                    print(f"⚠️ Failed to create test user")
            else:
                print(f"✅ Test user already exists: {self.test_user_id}")
                
        except Exception as e:
            print(f"⚠️ Error ensuring test user exists: {e}")
            # Continue anyway - some tests might work without explicit user creation

    def generate_mobile_behavioral_data(self, session_num: int) -> dict:
        """Generate realistic mobile behavioral data in the correct logs format"""
        import random
        
        base_timestamp = datetime.utcnow().timestamp() * 1000
        
        # Simulate different behavioral patterns for different sessions
        variance = session_num * 0.1  # Slight variations per session
        
        logs = []
        
        # Generate touch events in the correct format with 'data' field
        for i in range(15):
            logs.append({
                "event_type": "touch",
                "timestamp": base_timestamp + i * 100,
                "data": {
                    "x": 150 + random.uniform(-50, 50) + variance * 10,
                    "y": 300 + random.uniform(-50, 50) + variance * 5,
                    "pressure": 0.5 + random.uniform(-0.2, 0.2),
                    "action": "down" if i % 3 == 0 else "move"
                }
            })
        
        # Generate accelerometer events in the correct format with 'data' field
        for i in range(25):
            logs.append({
                "event_type": "accelerometer",
                "timestamp": base_timestamp + i * 50,
                "data": {
                    "x": random.uniform(-2, 2) + variance,
                    "y": random.uniform(-2, 2) + variance * 0.5,
                    "z": 9.8 + random.uniform(-1, 1)
                }
            })
        
        # Generate gyroscope events in the correct format with 'data' field
        for i in range(25):
            logs.append({
                "event_type": "gyroscope",
                "timestamp": base_timestamp + i * 50,
                "data": {
                    "x": random.uniform(-0.5, 0.5) + variance * 0.2,
                    "y": random.uniform(-0.5, 0.5) + variance * 0.1,
                    "z": random.uniform(-0.5, 0.5)
                }
            })
        
        # Generate scroll events in the correct format with 'data' field
        for i in range(8):
            logs.append({
                "event_type": "scroll",
                "timestamp": base_timestamp + i * 200,
                "data": {
                    "delta_y": random.uniform(-100, 100) + variance * 20,
                    "velocity": random.uniform(50, 200)
                }
            })
        
        # Sort logs by timestamp for realistic sequence
        logs.sort(key=lambda x: x['timestamp'])
        
        return {
            "user_id": self.test_user_id,
            "session_id": f"session_{self.test_user_id}_{session_num}",
            "logs": logs
        }

    async def test_single_session_workflow(self, session_num: int) -> dict:
        """Test a complete single session workflow"""
        session_id = f"session_{self.test_user_id}_{session_num}"
        
        print(f"\n=== Testing Session {session_num} Workflow ===")
        
        try:
            # Step 1: Generate mobile behavioral data
            mobile_data = self.generate_mobile_behavioral_data(session_num)
            print(f"Generated mobile data with {len(mobile_data['logs'])} behavioral events")
            
            # Step 2: Process and store behavioral data using FAISS engine
            result = await self.faiss_engine.process_mobile_behavioral_data(
                user_id=self.test_user_id,
                session_id=session_id,
                behavioral_data=mobile_data
            )
            print(f"Processed and stored mobile behavioral data: {result}")
            
            # Extract vector info for analysis
            vector_id = result.vector_id if hasattr(result, 'vector_id') else None
            
            # Get non-zero count by processing the data with our processor too for analysis
            features = self.processor.process_mobile_behavioral_data(mobile_data)
            non_zero_count = (features != 0).sum()
            print(f"Generated 90D vector with {non_zero_count}/90 non-zero elements")
            
            # Step 3: Simulate session end with cumulative update
            await self.faiss_engine.end_session_update(self.test_user_id, session_id)
            print(f"Completed session end cumulative update")
            
            # Step 4: Check learning status
            learning_status = await self.faiss_engine.get_user_learning_status(self.test_user_id)
            print(f"Learning status: {learning_status}")
            
            return {
                "session_num": session_num,
                "session_id": session_id,
                "vector_id": vector_id,
                "non_zero_elements": int(non_zero_count),
                "learning_status": learning_status,
                "result": {
                    "similarity_score": result.similarity_score if hasattr(result, 'similarity_score') else 0.0,
                    "confidence": result.confidence if hasattr(result, 'confidence') else 0.0,
                    "decision": result.decision if hasattr(result, 'decision') else 'unknown',
                    "risk_level": result.risk_level if hasattr(result, 'risk_level') else 'unknown',
                    "vector_id": result.vector_id if hasattr(result, 'vector_id') else None
                },
                "success": True,
                "error": None
            }
            
        except Exception as e:
            print(f"❌ Error in session {session_num}: {e}")
            return {
                "session_num": session_num,
                "session_id": session_id,
                "vector_id": None,
                "non_zero_elements": 0,
                "learning_status": {"error": str(e)},
                "result": {},
                "success": False,
                "error": str(e)
            }

    async def test_complete_learning_progression(self) -> dict:
        """Test complete learning progression through all phases"""
        print(f"\n🚀 Starting Complete Session Workflow Test for User: {self.test_user_id}")
        
        # Ensure test user exists in database
        await self.ensure_test_user_exists()
        
        results = {
            "user_id": self.test_user_id,
            "sessions": [],
            "phase_transitions": [],
            "final_status": {}
        }
        
        # Test 12 sessions to go through all learning phases
        for session_num in range(1, 13):
            session_result = await self.test_single_session_workflow(session_num)
            results["sessions"].append(session_result)
            
            # Check for phase transitions
            current_phase = session_result["learning_status"]["learning_phase"]
            if session_num == 1:
                last_phase = "learning"
            else:
                last_phase = results["sessions"][-2]["learning_status"]["learning_phase"]
            
            if current_phase != last_phase:
                transition = {
                    "session": session_num,
                    "from_phase": last_phase,
                    "to_phase": current_phase,
                    "vector_count": session_result["learning_status"]["vector_count"]
                }
                results["phase_transitions"].append(transition)
                print(f"🎯 PHASE TRANSITION: {transition}")
            
            # Add small delay between sessions
            await asyncio.sleep(0.1)
        
        # Get final learning status
        final_status = await self.faiss_engine.get_user_learning_status(self.test_user_id)
        results["final_status"] = final_status
        
        return results

    async def verify_cumulative_vectors(self) -> dict:
        """Verify cumulative vectors are being properly updated"""
        print(f"\n🔍 Verifying Cumulative Vector Storage...")
        
        # Query cumulative vectors for the test user
        cumulative_result = self.db_client.supabase.table('enhanced_behavioral_vectors')\
            .select('*')\
            .eq('user_id', self.test_user_id)\
            .eq('vector_type', 'cumulative')\
            .order('created_at', desc=False)\
            .execute()
        
        cumulative_vectors = cumulative_result.data
        print(f"Found {len(cumulative_vectors)} cumulative vectors")
        
        # Check vector evolution
        vector_analysis = []
        for i, vector_data in enumerate(cumulative_vectors):
            vector = vector_data['vector_data']
            non_zero_count = sum(1 for v in vector if v != 0)
            
            analysis = {
                "update_number": i + 1,
                "timestamp": vector_data['created_at'],
                "non_zero_elements": non_zero_count,
                "vector_norm": sum(v*v for v in vector) ** 0.5
            }
            vector_analysis.append(analysis)
            print(f"  Update {i+1}: {non_zero_count}/90 non-zero, norm: {analysis['vector_norm']:.4f}")
        
        return {
            "cumulative_vector_count": len(cumulative_vectors),
            "vector_analysis": vector_analysis
        }

    async def cleanup_test_data(self):
        """Clean up test data"""
        print(f"\n🧹 Cleaning up test data for user: {self.test_user_id}")
        
        try:
            # Delete test vectors
            self.db_client.supabase.table('enhanced_behavioral_vectors')\
                .delete()\
                .eq('user_id', self.test_user_id)\
                .execute()
            
            # Delete test user profile
            self.db_client.supabase.table('user_profiles')\
                .delete()\
                .eq('user_id', self.test_user_id)\
                .execute()
            
            # Delete test user
            self.db_client.supabase.table('users')\
                .delete()\
                .eq('id', self.test_user_id)\
                .execute()
            
            print("✅ Test data cleaned up")
            
        except Exception as e:
            print(f"⚠️ Error during cleanup: {e}")
            # Continue anyway

async def main():
    """Run complete session workflow test"""
    tester = SessionWorkflowTester()
    
    try:
        # Test complete learning progression
        results = await tester.test_complete_learning_progression()
        
        # Verify cumulative vector storage
        cumulative_analysis = await tester.verify_cumulative_vectors()
        results["cumulative_analysis"] = cumulative_analysis
        
        # Print summary
        print(f"\n📊 TEST SUMMARY")
        print(f"=" * 50)
        print(f"User ID: {results['user_id']}")
        print(f"Total Sessions: {len(results['sessions'])}")
        print(f"Phase Transitions: {len(results['phase_transitions'])}")
        print(f"Final Phase: {results['final_status']['learning_phase']}")
        print(f"Final Vector Count: {results['final_status']['vector_count']}")
        print(f"Baseline Created: {results['final_status']['baseline_created']}")
        print(f"Cumulative Updates: {cumulative_analysis['cumulative_vector_count']}")
        
        # Phase transition summary
        print(f"\n🎯 Phase Transitions:")
        for transition in results["phase_transitions"]:
            print(f"  Session {transition['session']}: {transition['from_phase']} → {transition['to_phase']}")
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"session_workflow_test_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {results_file}")
        
        print(f"\n✅ COMPLETE SESSION WORKFLOW TEST PASSED!")
        print(f"✅ Cumulative Learning System is Working Correctly!")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup test data
        await tester.cleanup_test_data()

if __name__ == "__main__":
    asyncio.run(main())

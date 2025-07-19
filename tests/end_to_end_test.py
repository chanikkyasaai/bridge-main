"""
End-to-End Test for FAISS Behavioral Authentication System
Tests complete flow from user registration to session analysis
"""

import asyncio
import json
import httpx
import time
import logging
from pathlib import Path
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EndToEndTester:
    def __init__(self, backend_url="http://localhost:8000", ml_engine_url="http://localhost:8001"):
        self.backend_url = backend_url
        self.ml_engine_url = ml_engine_url
        self.data_dir = Path("data")
        
        # Test user details
        self.test_user = {
            "phone_number": "9876543210",
            "password": "TestPassword123",
            "mpin": "1234"
        }
        
        self.user_id = None
        self.session_results = []
        
    async def check_services_health(self):
        """Check if backend and ML engine are running"""
        logger.info("🔍 Checking services health...")
        
        async with httpx.AsyncClient() as client:
            # Check backend
            try:
                response = await client.get(f"{self.backend_url}/")
                logger.info(f"✅ Backend: {response.status_code}")
            except Exception as e:
                logger.error(f"❌ Backend not available: {e}")
                return False
            
            # Check ML Engine
            try:
                response = await client.get(f"{self.ml_engine_url}/")
                result = response.json()
                logger.info(f"✅ ML Engine: {result.get('status', 'unknown')}")
                logger.info(f"   Active sessions: {result.get('statistics', {}).get('active_sessions', 0)}")
            except Exception as e:
                logger.error(f"❌ ML Engine not available: {e}")
                return False
        
        return True
    
    async def register_test_user(self):
        """Register test user in backend"""
        logger.info("👤 Registering test user...")
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(f"{self.backend_url}/api/v1/auth/register", json={
                    "phone_number": self.test_user["phone_number"],
                    "password": self.test_user["password"],
                    "mpin": self.test_user["mpin"]
                })
                
                if response.status_code == 200:
                    result = response.json()
                    self.user_id = result.get("user_id")
                    logger.info(f"✅ User registered: {self.user_id}")
                    return True
                else:
                    logger.error(f"❌ Registration failed: {response.status_code} - {response.text}")
                    return False
                    
            except Exception as e:
                logger.error(f"❌ Registration error: {e}")
                return False
    
    async def login_user(self):
        """Login test user"""
        logger.info("🔐 Logging in test user...")
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(f"{self.backend_url}/api/v1/auth/login", json={
                    "phone_number": self.test_user["phone_number"],
                    "password": self.test_user["password"]
                })
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"✅ Login successful: {result.get('message', 'Success')}")
                    return result
                else:
                    logger.error(f"❌ Login failed: {response.status_code} - {response.text}")
                    return None
                    
            except Exception as e:
                logger.error(f"❌ Login error: {e}")
                return None
    
    async def run_session_test(self, session_num: int):
        """Run test with one session file"""
        session_file = self.data_dir / f"test_user_session_{session_num:02d}.json"
        
        if not session_file.exists():
            logger.error(f"❌ Session file not found: {session_file}")
            return None
            
        logger.info(f"📱 Running Session {session_num} test...")
        
        # Load session data
        with open(session_file, 'r') as f:
            session_data = json.load(f)
        
        user_id = session_data["user_id"] 
        session_id = session_data["session_id"]
        events = session_data["logs"]
        
        logger.info(f"   📊 Session ID: {session_id}")
        logger.info(f"   📈 Events count: {len(events)}")
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                # Start ML session
                start_result = await client.post(f"{self.ml_engine_url}/session/start", json={
                    "user_id": user_id,
                    "session_id": session_id,
                    "device_info": {"device_id": "test_device", "platform": "test"}
                })
                
                if start_result.status_code == 200:
                    start_data = start_result.json()
                    logger.info(f"   🚀 Session started in {start_data.get('phase', 'unknown')} mode")
                else:
                    logger.error(f"   ❌ Session start failed: {start_result.status_code}")
                    return None
                
                # Send behavioral data in chunks (simulating real-time)
                chunk_size = 20  # Events per analysis
                analysis_results = []
                
                for i in range(0, len(events), chunk_size):
                    chunk = events[i:i+chunk_size]
                    
                    # Analyze chunk
                    analyze_result = await client.post(f"{self.ml_engine_url}/analyze-mobile", json={
                        "user_id": user_id,
                        "session_id": session_id,
                        "logs": chunk
                    })
                    
                    if analyze_result.status_code == 200:
                        result = analyze_result.json()
                        analysis_results.append(result)
                        
                        decision = result.get("decision", "unknown")
                        confidence = result.get("confidence", 0)
                        phase = result.get("phase", "unknown")
                        
                        logger.info(f"   📊 Chunk {i//chunk_size + 1}: {decision} (confidence: {confidence:.3f}, phase: {phase})")
                        
                        # If blocked, stop session
                        if decision == "block":
                            logger.warning(f"   🚫 Session BLOCKED at chunk {i//chunk_size + 1}")
                            break
                    else:
                        logger.error(f"   ❌ Analysis failed for chunk {i//chunk_size + 1}")
                
                # End session
                end_result = await client.post(f"{self.ml_engine_url}/session/end", json={
                    "session_id": session_id,
                    "reason": "completed"
                })
                
                if end_result.status_code == 200:
                    end_data = end_result.json()
                    logger.info(f"   ✅ Session ended: {end_data.get('message', 'Success')}")
                else:
                    logger.error(f"   ❌ Session end failed: {end_result.status_code}")
                
                # Compile session results
                session_result = {
                    "session_num": session_num,
                    "session_id": session_id,
                    "total_events": len(events),
                    "chunks_processed": len(analysis_results),
                    "final_decision": analysis_results[-1].get("decision") if analysis_results else "unknown",
                    "final_confidence": analysis_results[-1].get("confidence", 0) if analysis_results else 0,
                    "phase": analysis_results[-1].get("phase") if analysis_results else "unknown",
                    "session_count": analysis_results[-1].get("session_count") if analysis_results else 0,
                    "blocked": any(r.get("decision") == "block" for r in analysis_results),
                    "analysis_results": analysis_results
                }
                
                return session_result
                
            except Exception as e:
                logger.error(f"   ❌ Session test error: {e}")
                return None
    
    async def run_complete_test(self):
        """Run complete end-to-end test with all 10 sessions"""
        logger.info("🚀 Starting Complete End-to-End Test")
        logger.info("="*60)
        
        # Check services
        if not await self.check_services_health():
            logger.error("❌ Services not available. Please start backend and ML engine.")
            return
        
        # Register user
        if not await self.register_test_user():
            logger.error("❌ User registration failed.")
            return
        
        # Login user
        login_result = await self.login_user()
        if not login_result:
            logger.error("❌ User login failed.")
            return
        
        logger.info("="*60)
        logger.info("📱 Starting Session Tests (1-10)")
        logger.info("="*60)
        
        # Run all 10 sessions
        for session_num in range(1, 11):
            result = await self.run_session_test(session_num)
            if result:
                self.session_results.append(result)
                
                # Summary for this session
                logger.info(f"📋 Session {session_num} Summary:")
                logger.info(f"   Decision: {result['final_decision']}")
                logger.info(f"   Confidence: {result['final_confidence']:.3f}")
                logger.info(f"   Phase: {result['phase']}")
                logger.info(f"   Session Count: {result['session_count']}")
                logger.info(f"   Events: {result['total_events']}")
                logger.info(f"   Blocked: {'Yes' if result['blocked'] else 'No'}")
                
            logger.info("-" * 40)
            await asyncio.sleep(1)  # Brief pause between sessions
        
        # Final analysis
        await self.print_final_analysis()
    
    async def print_final_analysis(self):
        """Print comprehensive analysis of all sessions"""
        logger.info("="*60)
        logger.info("📊 FINAL ANALYSIS")
        logger.info("="*60)
        
        if not self.session_results:
            logger.error("No session results to analyze")
            return
        
        learning_sessions = [r for r in self.session_results if r["phase"] == "learning"]
        auth_sessions = [r for r in self.session_results if r["phase"] == "authentication"]
        
        logger.info(f"👨‍🎓 Learning Phase Sessions: {len(learning_sessions)}")
        logger.info(f"🔒 Authentication Phase Sessions: {len(auth_sessions)}")
        
        if learning_sessions:
            logger.info("\n📚 Learning Phase Analysis:")
            for session in learning_sessions:
                logger.info(f"   Session {session['session_num']}: {session['final_decision']} (count: {session['session_count']})")
        
        if auth_sessions:
            logger.info("\n🔐 Authentication Phase Analysis:")
            for session in auth_sessions:
                status = "🚫 BLOCKED" if session['blocked'] else "✅ ALLOWED"
                logger.info(f"   Session {session['session_num']}: {status} (confidence: {session['final_confidence']:.3f})")
            
            # Authentication statistics
            allowed_count = len([s for s in auth_sessions if not s['blocked']])
            blocked_count = len([s for s in auth_sessions if s['blocked']])
            avg_confidence = sum(s['final_confidence'] for s in auth_sessions) / len(auth_sessions)
            
            logger.info(f"\n📈 Authentication Statistics:")
            logger.info(f"   Allowed: {allowed_count}/{len(auth_sessions)}")
            logger.info(f"   Blocked: {blocked_count}/{len(auth_sessions)}")
            logger.info(f"   Average Confidence: {avg_confidence:.3f}")
        
        logger.info("\n🎯 System Behavior Verification:")
        logger.info(f"   ✓ Learning phase should handle first 6 sessions")
        logger.info(f"   ✓ Authentication phase should start from session 7+")
        logger.info(f"   ✓ Similar user behavior should be allowed")
        logger.info(f"   ✓ System should learn and adapt")
        
        # Check expected behavior
        expected_learning = 6
        actual_learning = len(learning_sessions)
        
        if actual_learning >= expected_learning:
            logger.info("   ✅ Learning phase completed as expected")
        else:
            logger.warning(f"   ⚠️ Learning phase incomplete: {actual_learning}/{expected_learning}")
        
        if auth_sessions and allowed_count > 0:
            logger.info("   ✅ Authentication phase working correctly")
        elif auth_sessions:
            logger.warning("   ⚠️ All authentication sessions blocked - may need threshold adjustment")
        
        logger.info("="*60)
        logger.info("🏆 End-to-End Test Complete!")
        logger.info("="*60)

async def main():
    """Main test execution"""
    tester = EndToEndTester()
    await tester.run_complete_test()

if __name__ == "__main__":
    asyncio.run(main())

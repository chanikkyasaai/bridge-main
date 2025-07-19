#!/usr/bin/env python3
"""
Comprehensive ML Engine Direct Test
===================================

This script directly tests the ML Engine with all 10 user sessions and logs:
- Vector representations (48-dimensional)
- Matching scores and similarity calculations
- Learning phase (sessions 1-6) vs Authentication phase (sessions 7-10)
- All ML engine methods and their responses
- FAISS vector storage and retrieval
- Bot detection algorithms
- Feature extraction details
"""

import asyncio
import aiohttp
import json
import numpy as np
from datetime import datetime
from pathlib import Path
import logging

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Configuration
ML_ENGINE_URL = "http://localhost:8001"
DATA_DIR = Path("data")

class MLEngineDirectTester:
    def __init__(self):
        self.session = None
        self.test_user_id = "test-user-ml-engine"
        
    async def setup(self):
        """Setup HTTP session"""
        self.session = aiohttp.ClientSession()
        
    async def cleanup(self):
        """Cleanup HTTP session"""
        if self.session:
            await self.session.close()
    
    async def call_ml_engine(self, method: str, endpoint: str, data: dict = None):
        """Call ML engine with detailed logging"""
        url = f"{ML_ENGINE_URL}{endpoint}"
        print(f"\n🔗 ML ENGINE CALL: {method} {endpoint}")
        
        try:
            if method.upper() == "POST":
                async with self.session.post(url, json=data) as response:
                    result = await response.json()
                    print(f"✅ Response Status: {response.status}")
                    print(f"📄 Response Data: {json.dumps(result, indent=2)}")
                    return result
            else:
                async with self.session.get(url) as response:
                    result = await response.json()
                    print(f"✅ Response Status: {response.status}")
                    print(f"📄 Response Data: {json.dumps(result, indent=2)}")
                    return result
                    
        except Exception as e:
            print(f"❌ ML Engine Error: {e}")
            return None
    
    async def test_health_check(self):
        """Test ML Engine health and capabilities"""
        print("\n" + "="*60)
        print("🏥 TESTING ML ENGINE HEALTH & CAPABILITIES")
        print("="*60)
        
        result = await self.call_ml_engine("GET", "/")
        if result:
            print(f"🟢 ML Engine Status: {result.get('status')}")
            components = result.get('components', {})
            for component, status in components.items():
                print(f"   📦 {component}: {status}")
            
            statistics = result.get('statistics', {})
            for stat, value in statistics.items():
                print(f"   📊 {stat}: {value}")
        
        return result is not None
    
    async def test_session_lifecycle(self, session_number: int, session_id: str):
        """Test session start and end"""
        print(f"\n🚀 TESTING SESSION LIFECYCLE - Session {session_number}")
        print("-" * 50)
        
        # Start session
        start_data = {
            "user_id": self.test_user_id,
            "session_id": session_id,
            "device_info": {
                "platform": "test",
                "version": "1.0",
                "session_number": session_number
            }
        }
        
        start_result = await self.call_ml_engine("POST", "/session/start", start_data)
        
        if start_result:
            print(f"✅ Session {session_number} started successfully")
            return True
        else:
            print(f"❌ Session {session_number} start failed")
            return False
    
    async def test_behavioral_analysis(self, session_number: int, session_id: str, session_file: Path):
        """Test behavioral data analysis with detailed vector logging"""
        print(f"\n🧠 TESTING BEHAVIORAL ANALYSIS - Session {session_number}")
        print("-" * 50)
        
        # Load behavioral data
        try:
            with open(session_file, 'r') as f:
                session_data = json.load(f)
            
            logs = session_data["logs"]
            print(f"📊 Loaded {len(logs)} behavioral events from {session_file.name}")
            
            # Analyze behavior
            analysis_data = {
                "user_id": self.test_user_id,
                "session_id": session_id,
                "logs": logs
            }
            
            print(f"🔍 Sending behavioral data for analysis...")
            analysis_result = await self.call_ml_engine("POST", "/analyze-mobile", analysis_data)
            
            if analysis_result:
                print(f"\n📈 DETAILED ANALYSIS RESULTS - Session {session_number}:")
                print(f"   🎯 Decision: {analysis_result.get('decision', 'N/A')}")
                print(f"   🎲 Confidence: {analysis_result.get('confidence', 'N/A')}")
                print(f"   🤖 Bot Detected: {analysis_result.get('is_bot_detected', 'N/A')}")
                print(f"   📊 Risk Score: {analysis_result.get('risk_score', 'N/A')}")
                print(f"   🔢 Session Count: {analysis_result.get('session_count', 'N/A')}")
                print(f"   📋 Status: {analysis_result.get('status', 'N/A')}")
                
                # Extract vector information if available
                if 'vector_info' in analysis_result:
                    vector_info = analysis_result['vector_info']
                    print(f"   🔢 Vector Dimension: {vector_info.get('dimensions', 'N/A')}")
                    print(f"   📏 Vector Norm: {vector_info.get('norm', 'N/A')}")
                    print(f"   🎯 Vector Preview: {vector_info.get('preview', 'N/A')}")
                
                # Show similarity scores if in authentication phase
                if 'similarity_scores' in analysis_result:
                    scores = analysis_result['similarity_scores']
                    print(f"   🔍 Similarity Analysis:")
                    for score_type, value in scores.items():
                        print(f"       {score_type}: {value}")
                
                # Phase-specific information
                phase = "LEARNING" if session_number <= 6 else "AUTHENTICATION"
                print(f"   🎭 Phase: {phase}")
                
                if phase == "LEARNING":
                    print(f"   🧠 Learning Progress: Building behavioral profile...")
                    print(f"   📚 Session {session_number}/6 in learning phase")
                else:
                    print(f"   🔐 Authentication Mode: Validating against learned profile...")
                    print(f"   ⚖️  Authentication Session {session_number - 6}/4")
                
                return analysis_result
            else:
                print(f"❌ Behavioral analysis failed for session {session_number}")
                return None
                
        except Exception as e:
            print(f"❌ Error in behavioral analysis: {e}")
            return None
    
    async def test_session_end(self, session_number: int, session_id: str):
        """Test session ending"""
        print(f"\n🔚 ENDING SESSION {session_number}")
        print("-" * 30)
        
        end_data = {
            "session_id": session_id,
            "reason": "completed"
        }
        
        end_result = await self.call_ml_engine("POST", "/session/end", end_data)
        
        if end_result:
            print(f"✅ Session {session_number} ended successfully")
            return True
        else:
            print(f"❌ Session {session_number} end failed")
            return False
    
    async def test_statistics(self, session_number: int):
        """Test getting ML engine statistics"""
        print(f"\n📊 CHECKING ML ENGINE STATISTICS after Session {session_number}")
        print("-" * 50)
        
        stats_result = await self.call_ml_engine("GET", "/statistics")
        
        if stats_result:
            print(f"📈 Current Statistics:")
            for key, value in stats_result.items():
                print(f"   {key}: {value}")
            return stats_result
        else:
            print(f"❌ Failed to get statistics")
            return None
    
    async def test_user_profile(self, session_number: int):
        """Test getting user profile information"""
        print(f"\n👤 CHECKING USER PROFILE after Session {session_number}")
        print("-" * 40)
        
        profile_result = await self.call_ml_engine("GET", f"/user/{self.test_user_id}/profile")
        
        if profile_result:
            print(f"📋 User Profile Information:")
            for key, value in profile_result.items():
                if isinstance(value, dict):
                    print(f"   {key}:")
                    for sub_key, sub_value in value.items():
                        print(f"     {sub_key}: {sub_value}")
                elif isinstance(value, list) and len(value) > 0:
                    print(f"   {key}: [{len(value)} items]")
                    if len(value) <= 3:  # Show first few items
                        for item in value:
                            print(f"     - {item}")
                else:
                    print(f"   {key}: {value}")
            return profile_result
        else:
            print(f"❌ Failed to get user profile")
            return None
    
    async def run_comprehensive_test(self):
        """Run complete ML engine test with all 10 sessions"""
        print("\n" + "="*80)
        print("🚀 COMPREHENSIVE ML ENGINE DIRECT TEST")
        print("   Testing Learning Phase (Sessions 1-6) & Authentication Phase (Sessions 7-10)")
        print("="*80)
        
        # Test health first
        if not await self.test_health_check():
            print("❌ ML Engine health check failed - aborting test")
            return
        
        # Find all session files
        session_files = []
        for i in range(1, 11):
            session_file = DATA_DIR / f"test_user_session_{i:02d}.json"
            if session_file.exists():
                session_files.append((session_file, i))
            else:
                print(f"⚠️  Missing session file: {session_file}")
        
        if not session_files:
            print("❌ No session files found!")
            return
        
        print(f"\n📁 Found {len(session_files)} session files to test")
        
        successful_sessions = 0
        analysis_results = []
        
        # Test each session
        for session_file, session_number in session_files:
            session_id = f"ml-test-session-{session_number:02d}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            
            print(f"\n" + "🔥"*60)
            print(f"🎯 TESTING SESSION {session_number}: {session_file.name}")
            print(f"   Session ID: {session_id}")
            print("🔥"*60)
            
            try:
                # Step 1: Start session
                if await self.test_session_lifecycle(session_number, session_id):
                    # Step 2: Analyze behavioral data
                    analysis_result = await self.test_behavioral_analysis(session_number, session_id, session_file)
                    
                    if analysis_result:
                        analysis_results.append({
                            'session_number': session_number,
                            'session_id': session_id,
                            'result': analysis_result
                        })
                        successful_sessions += 1
                    
                    # Step 3: End session
                    await self.test_session_end(session_number, session_id)
                    
                    # Step 4: Check statistics
                    await self.test_statistics(session_number)
                    
                    # Step 5: Check user profile (every few sessions)
                    if session_number in [3, 6, 10]:
                        await self.test_user_profile(session_number)
                
                # Wait before next session
                if session_number < len(session_files):
                    print(f"\n⏳ Waiting before next session...")
                    await asyncio.sleep(2)
                    
            except Exception as e:
                print(f"❌ Error in session {session_number}: {e}")
        
        # Final summary
        print(f"\n" + "="*80)
        print("🏁 COMPREHENSIVE ML ENGINE TEST COMPLETE")
        print("="*80)
        print(f"✅ Successful sessions: {successful_sessions}/{len(session_files)}")
        
        # Analyze learning vs authentication phase results
        learning_sessions = [r for r in analysis_results if r['session_number'] <= 6]
        auth_sessions = [r for r in analysis_results if r['session_number'] > 6]
        
        print(f"\n📊 PHASE ANALYSIS:")
        print(f"🧠 Learning Phase (Sessions 1-6): {len(learning_sessions)} sessions")
        for session in learning_sessions:
            result = session['result']
            print(f"   Session {session['session_number']}: {result.get('decision')} (confidence: {result.get('confidence', 'N/A')})")
        
        print(f"\n🔐 Authentication Phase (Sessions 7-10): {len(auth_sessions)} sessions")
        for session in auth_sessions:
            result = session['result']
            decision = result.get('decision')
            confidence = result.get('confidence', 'N/A')
            print(f"   Session {session['session_number']}: {decision} (confidence: {confidence})")
            
            if decision == 'allow':
                print(f"       ✅ Authentication SUCCESS")
            elif decision == 'reject':
                print(f"       ❌ Authentication FAILED")
            elif decision == 'challenge':
                print(f"       ⚠️  Authentication CHALLENGE")
        
        print(f"\n🎯 ML ENGINE BEHAVIOR SUMMARY:")
        if len(learning_sessions) == 6:
            print(f"   ✅ Learning phase completed successfully (6/6 sessions)")
        else:
            print(f"   ⚠️  Learning phase incomplete ({len(learning_sessions)}/6 sessions)")
        
        if len(auth_sessions) > 0:
            auth_successes = sum(1 for s in auth_sessions if s['result'].get('decision') == 'allow')
            print(f"   🔐 Authentication phase: {auth_successes}/{len(auth_sessions)} sessions authenticated")
        else:
            print(f"   ⚠️  No authentication sessions tested")
        
        return successful_sessions, analysis_results

async def main():
    """Main test function"""
    tester = MLEngineDirectTester()
    
    try:
        await tester.setup()
        successful_sessions, results = await tester.run_comprehensive_test()
        
        print(f"\n🎉 TEST COMPLETE!")
        print(f"📊 Total sessions tested: {successful_sessions}")
        print(f"📋 Analysis results collected: {len(results)}")
        
    finally:
        await tester.cleanup()

if __name__ == "__main__":
    print("🎯 ML Engine Comprehensive Direct Test")
    print("=" * 80)
    print("This test will:")
    print("• Test ML Engine health and capabilities")
    print("• Process all 10 behavioral sessions directly")
    print("• Log detailed vector representations and matching scores")
    print("• Show learning phase (1-6) vs authentication phase (7-10)")
    print("• Test all ML engine methods and endpoints")
    print("• Provide comprehensive behavioral analysis")
    print("=" * 80)
    
    asyncio.run(main())

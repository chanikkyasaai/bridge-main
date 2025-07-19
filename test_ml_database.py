"""
Comprehensive ML Engine Test with Database Integration
Tests the complete behavioral authentication flow with actual database persistence
"""

import asyncio
import logging
import json
import os
import numpy as np
from datetime import datetime
import aiohttp

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test configuration with proper UUIDs
ML_ENGINE_URL = "http://localhost:8001"
USER_ID = "123e4567-e89b-12d3-a456-426614174000"
SESSION_IDS = [
    "223e4567-e89b-12d3-a456-426614174001",
    "223e4567-e89b-12d3-a456-426614174002",
    "223e4567-e89b-12d3-a456-426614174003",
    "223e4567-e89b-12d3-a456-426614174004",
    "223e4567-e89b-12d3-a456-426614174005",
    "223e4567-e89b-12d3-a456-426614174006",
    "223e4567-e89b-12d3-a456-426614174007",
    "223e4567-e89b-12d3-a456-426614174008",
    "223e4567-e89b-12d3-a456-426614174009",
    "223e4567-e89b-12d3-a456-426614174010",
]

def load_test_data(file_number: int) -> dict:
    """Load test data from JSON files"""
    try:
        # Use absolute path
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        file_path = os.path.join(data_dir, f"test_user_session_{file_number:02d}.json")
        with open(file_path, 'r') as f:
            data = json.load(f)
            logger.info(f"✅ Loaded test data from {file_path}")
            return data
    except FileNotFoundError:
        logger.warning(f"Test file {file_number} not found at {file_path}, generating synthetic data")
        return generate_synthetic_session_data()

def generate_synthetic_session_data() -> dict:
    """Generate synthetic behavioral data for testing"""
    return {
        "behavioral_data": [
            {
                "timestamp": datetime.utcnow().isoformat(),
                "mouse": {
                    "x": int(np.random.randint(100, 800)),
                    "y": int(np.random.randint(100, 600)),
                    "click": bool(np.random.choice([True, False], p=[0.1, 0.9]))
                },
                "keyboard": {
                    "key": "a" if np.random.random() > 0.5 else "space",
                    "dwell_time": float(np.random.uniform(50, 200)),
                    "flight_time": float(np.random.uniform(20, 100))
                },
                "touch": {
                    "x": int(np.random.randint(0, 400)),
                    "y": int(np.random.randint(0, 800)),
                    "pressure": float(np.random.uniform(0.1, 1.0)),
                    "size": float(np.random.uniform(5, 15))
                }
            } for _ in range(int(np.random.randint(50, 150)))
        ]
    }

async def test_session_analysis(session_num: int, session_id: str, session_data: dict):
    """Test a single session analysis"""
    logger.info(f"\n{'='*60}")
    logger.info(f"TESTING SESSION {session_num}: {session_id}")
    logger.info(f"{'='*60}")
    
    async with aiohttp.ClientSession() as client:
        try:
            # 1. Start session
            logger.info(f"🔄 Starting session analysis...")
            
            # Handle both formats: "logs" (actual test files) and "behavioral_data" (synthetic)
            logs_data = session_data.get("logs", session_data.get("behavioral_data", []))
            
            start_payload = {
                "user_id": USER_ID,
                "session_id": session_id,
                "logs": logs_data
            }
            
            async with client.post(f"{ML_ENGINE_URL}/analyze-mobile", json=start_payload) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"✅ Session analysis completed:")
                    logger.info(f"   🎯 Decision: {result.get('decision', 'unknown')}")
                    logger.info(f"   � Phase: {result.get('phase', 'unknown')}")
                    logger.info(f"   📈 Session Count: {result.get('session_count', 0)}")
                    logger.info(f"   � Confidence: {result.get('confidence', 0.0):.4f}")
                    
                    if result.get('phase') == 'authentication':
                        logger.info(f"   🔐 Auth Decision: {result.get('decision', 'unknown')}")
                    
                    if result.get('similarity_score') is not None:
                        logger.info(f"   📍 Similarity: {result.get('similarity_score'):.4f}")
                    
                    if result.get('bot_probability') is not None:
                        logger.info(f"   🤖 Bot Risk: {result.get('bot_probability'):.4f}")
                    
                    # Now end the session to increment session count
                    logger.info(f"🔚 Ending session...")
                    end_payload = {
                        "session_id": session_id,
                        "reason": "completed"
                    }
                    
                    async with client.post(f"{ML_ENGINE_URL}/session/end", json=end_payload) as end_response:
                        if end_response.status == 200:
                            end_result = await end_response.json()
                            logger.info(f"✅ Session ended: {end_result.get('message', 'success')}")
                        else:
                            logger.error(f"❌ Session end failed: HTTP {end_response.status}")
                    
                    # Store key metrics for summary
                    return {
                        'session_id': session_id,
                        'decision': result.get('decision'),
                        'phase': result.get('phase'),
                        'session_count': result.get('session_count', 0),
                        'confidence': result.get('confidence', 0.0),
                        'similarity_score': result.get('similarity_score'),
                        'bot_probability': result.get('bot_probability'),
                        'message': result.get('message', '')
                    }
                else:
                    error_text = await response.text()
                    logger.error(f"❌ Session analysis failed: HTTP {response.status} - {error_text}")
                    return None
                    
        except Exception as e:
            logger.error(f"❌ Exception during session analysis: {e}")
            return None

async def test_system_status():
    """Test system status endpoint"""
    logger.info(f"\n{'='*60}")
    logger.info(f"TESTING SYSTEM STATUS")
    logger.info(f"{'='*60}")
    
    async with aiohttp.ClientSession() as client:
        try:
            async with client.get(f"{ML_ENGINE_URL}/") as response:
                if response.status == 200:
                    status = await response.json()
                    logger.info(f"✅ System Status:")
                    logger.info(f"   🏥 Health: {status.get('status', 'unknown')}")
                    logger.info(f"   � Users: {status.get('statistics', {}).get('total_users', 0)}")
                    logger.info(f"   � Sessions: {status.get('statistics', {}).get('total_sessions', 0)}")
                    logger.info(f"   � Active Sessions: {status.get('statistics', {}).get('active_sessions', 0)}")
                    logger.info(f"   🧠 Components: {status.get('components', {})}")
                    return status
                else:
                    logger.error(f"❌ System status check failed: HTTP {response.status}")
                    return None
        except Exception as e:
            logger.error(f"❌ Exception during system status check: {e}")
            return None

async def main():
    """Main test execution"""
    logger.info(f"🚀 Starting Comprehensive ML Engine Database Test")
    logger.info(f"🎯 Target: {ML_ENGINE_URL}")
    logger.info(f"👤 User ID: {USER_ID}")
    logger.info(f"📱 Sessions: {len(SESSION_IDS)} total")
    
    # Test system status first
    await test_system_status()
    
    # Test all sessions
    session_results = []
    for i, session_id in enumerate(SESSION_IDS, 1):
        session_data = load_test_data(i)
        result = await test_session_analysis(i, session_id, session_data)
        if result:
            session_results.append(result)
        
        # Small delay between sessions
        await asyncio.sleep(0.5)
    
    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info(f"COMPREHENSIVE TEST SUMMARY")
    logger.info(f"{'='*60}")
    
    if session_results:
        learning_sessions = [r for r in session_results if r.get('phase') == 'learning']
        auth_sessions = [r for r in session_results if r.get('phase') == 'authentication']
        
        logger.info(f"📚 Learning Phase Sessions: {len(learning_sessions)}")
        logger.info(f"🔐 Authentication Phase Sessions: {len(auth_sessions)}")
        
        if auth_sessions:
            allow_decisions = [r for r in auth_sessions if r.get('decision') == 'allow']
            logger.info(f"✅ Authentication Allowed: {len(allow_decisions)}/{len(auth_sessions)}")
            
            avg_confidence = np.mean([r['confidence'] for r in session_results if r['confidence'] is not None])
            logger.info(f"📊 Average Confidence: {avg_confidence:.4f}")
        
        # Show session progression
        logger.info(f"\n📈 Session Count Progress:")
        for i, result in enumerate(session_results, 1):
            logger.info(f"   Session {i:2d}: Count={result['session_count']} Phase={result['phase']} Decision={result['decision']}")
        
        logger.info(f"\n🎯 DATABASE PERSISTENCE TEST: {'✅ SUCCESS' if any(r['session_count'] > 0 for r in session_results) else '❌ FAILED'}")
    else:
        logger.error(f"❌ No successful session tests completed")
    
    # Final system status
    await test_system_status()
    
    logger.info(f"\n🏁 Test completed!")

if __name__ == "__main__":
    asyncio.run(main())

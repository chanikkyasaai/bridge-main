#!/usr/bin/env python3
"""
Bot Detection Test for ML Engine
Tests if the authentication system can detect and block bot/attack behavior
"""

import asyncio
import aiohttp
import json
import logging
import time
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
ML_ENGINE_BASE_URL = "http://localhost:8001"
LEGITIMATE_USER_ID = "123e4567-e89b-12d3-a456-426614174000"  # User who completed learning
BOT_SESSION_FILE = "data/test_bot_session.json"

class BotDetectionTester:
    def __init__(self):
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_system_status(self):
        """Get current system status"""
        try:
            async with self.session.get(f"{ML_ENGINE_BASE_URL}/") as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    logger.error(f"Failed to get system status: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return None
    
    async def analyze_session(self, session_data):
        """Send session data for analysis"""
        try:
            # Prepare the request data
            request_data = {
                "user_id": LEGITIMATE_USER_ID,  # Use the legitimate user's ID
                "session_id": session_data["session_id"],
                "logs": session_data["logs"]
            }
            
            async with self.session.post(
                f"{ML_ENGINE_BASE_URL}/analyze-mobile",
                json=request_data,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result
                else:
                    error_text = await response.text()
                    logger.error(f"Analysis failed with status {response.status}: {error_text}")
                    return None
        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            return None
    
    async def end_session(self, session_id):
        """End the session"""
        try:
            request_data = {
                "session_id": session_id,
                "reason": "completed"
            }
            
            async with self.session.post(
                f"{ML_ENGINE_BASE_URL}/session/end",
                json=request_data,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result
                else:
                    error_text = await response.text()
                    logger.error(f"Session end failed with status {response.status}: {error_text}")
                    return None
        except Exception as e:
            logger.error(f"Error ending session: {e}")
            return None

async def main():
    logger.info("🚀 Starting Bot Detection Test")
    logger.info(f"🎯 Target: {ML_ENGINE_BASE_URL}")
    logger.info(f"👤 Legitimate User ID: {LEGITIMATE_USER_ID}")
    logger.info(f"🤖 Bot Session File: {BOT_SESSION_FILE}")
    logger.info("")
    
    # Load bot session data
    bot_file_path = Path(BOT_SESSION_FILE)
    if not bot_file_path.exists():
        logger.error(f"❌ Bot session file not found: {BOT_SESSION_FILE}")
        return
    
    with open(bot_file_path, 'r') as f:
        bot_session_data = json.load(f)
    
    logger.info(f"✅ Loaded bot session data with {len(bot_session_data['logs'])} events")
    
    async with BotDetectionTester() as tester:
        # Get system status first
        logger.info("============================================================")
        logger.info("TESTING SYSTEM STATUS")
        logger.info("============================================================")
        
        status = await tester.get_system_status()
        if status:
            logger.info("✅ System Status:")
            logger.info(f"   🏥 Health: {status['status']}")
            logger.info(f"   👥 Users: {status['statistics']['total_users']}")
            logger.info(f"   📱 Sessions: {status['statistics']['total_sessions']}")
            logger.info(f"   🔄 Active Sessions: {status['statistics']['active_sessions']}")
            logger.info(f"   🧠 Components: {status['components']}")
        else:
            logger.error("❌ Failed to get system status")
            return
        
        logger.info("")
        logger.info("============================================================")
        logger.info("🤖 TESTING BOT SESSION DETECTION")
        logger.info("============================================================")
        logger.info(f"📊 Bot Session: {bot_session_data['session_id']}")
        logger.info(f"👤 Using legitimate user profile: {LEGITIMATE_USER_ID}")
        logger.info(f"📋 Events to analyze: {len(bot_session_data['logs'])}")
        logger.info("")
        
        # Analyze the bot session using the legitimate user's profile
        logger.info("🔄 Analyzing bot behavior against legitimate user profile...")
        
        start_time = time.time()
        result = await tester.analyze_session(bot_session_data)
        analysis_time = time.time() - start_time
        
        if result:
            logger.info("✅ Analysis completed:")
            logger.info(f"   🎯 Decision: {result.get('decision', 'unknown')}")
            logger.info(f"   📊 Phase: {result.get('phase', 'unknown')}")
            logger.info(f"   📈 Session Count: {result.get('session_count', 'unknown')}")
            logger.info(f"   🎗️ Confidence: {result.get('confidence', 0):.4f}")
            logger.info(f"   ⏱️ Analysis Time: {analysis_time:.2f}s")
            
            # Check for authentication details
            if 'similarity' in result:
                logger.info(f"   🔍 Similarity: {result['similarity']:.4f}")
            if 'reason' in result:
                logger.info(f"   💭 Reason: {result['reason']}")
            if 'message' in result:
                logger.info(f"   💬 Message: {result['message']}")
            
            # Determine test result
            decision = result.get('decision', '').lower()
            if decision == 'block':
                logger.info("   🚨 BOT DETECTION: ✅ SUCCESS - Bot behavior detected and blocked!")
            elif decision == 'allow':
                logger.warning("   ⚠️ BOT DETECTION: ❌ FAILED - Bot behavior was not detected!")
            else:
                logger.warning(f"   ❓ BOT DETECTION: UNKNOWN - Unexpected decision: {decision}")
            
            # End the session
            logger.info("")
            logger.info("🔚 Ending bot session...")
            
            end_result = await tester.end_session(bot_session_data['session_id'])
            if end_result:
                logger.info("✅ Bot session ended successfully")
            else:
                logger.warning("⚠️ Failed to end bot session")
                
        else:
            logger.error("❌ Bot analysis failed")
        
        # Final system status
        logger.info("")
        logger.info("============================================================")
        logger.info("FINAL SYSTEM STATUS")
        logger.info("============================================================")
        
        final_status = await tester.get_system_status()
        if final_status:
            logger.info("✅ Final System Status:")
            logger.info(f"   🏥 Health: {final_status['status']}")
            logger.info(f"   👥 Users: {final_status['statistics']['total_users']}")
            logger.info(f"   📱 Sessions: {final_status['statistics']['total_sessions']}")
            logger.info(f"   🔄 Active Sessions: {final_status['statistics']['active_sessions']}")
        
        logger.info("")
        logger.info("🏁 Bot Detection Test Completed!")

if __name__ == "__main__":
    asyncio.run(main())

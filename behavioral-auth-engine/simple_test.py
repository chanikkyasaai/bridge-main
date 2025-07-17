#!/usr/bin/env python3
"""
Simple Test to verify basic behavioral authentication system functionality
"""

import asyncio
import aiohttp
import json
import uuid
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleSystemTest:
    def __init__(self):
        self.backend_url = "http://127.0.0.1:8000"
        self.ml_engine_url = "http://127.0.0.1:8001"
    
    async def test_basic_workflow(self):
        """Test basic workflow: create user, start session, analyze behavior"""
        logger.info("🚀 Starting Simple Behavioral Authentication Test")
        
        # Generate test user
        user_id = str(uuid.uuid4())
        session_id = f"test_session_{int(datetime.now().timestamp())}"
        
        logger.info(f"Testing with User ID: {user_id}")
        logger.info(f"Session ID: {session_id}")
        
        try:
            # 1. Test ML Engine health
            await self._test_ml_engine_health()
            
            # 2. Start a session
            await self._start_session(user_id, session_id)
            
            # 3. Send behavioral data for analysis
            for i in range(3):
                logger.info(f"Sending behavioral sample {i+1}")
                await self._analyze_behavior(user_id, session_id, i+1)
                await asyncio.sleep(1)
            
            # 4. Check learning progress
            await self._check_learning_progress(user_id)
            
            logger.info("✅ Basic workflow test completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Test failed: {e}")
    
    async def _test_ml_engine_health(self):
        """Test ML Engine health endpoint"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.ml_engine_url}/") as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"✅ ML Engine is healthy: {data.get('status')}")
                else:
                    raise Exception(f"ML Engine health check failed: {response.status}")
    
    async def _start_session(self, user_id: str, session_id: str):
        """Start a new session"""
        async with aiohttp.ClientSession() as session:
            url = f"{self.ml_engine_url}/session/start"
            data = {"user_id": user_id, "session_id": session_id}
            
            async with session.post(url, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"✅ Session started: {result.get('message', 'Success')}")
                else:
                    text = await response.text()
                    raise Exception(f"Failed to start session: {response.status} - {text}")
    
    async def _analyze_behavior(self, user_id: str, session_id: str, sample_num: int):
        """Send behavioral data for analysis"""
        # Generate simple test vector (90 dimensions)
        test_vector = []
        for i in range(90):
            # Generate predictable but varied values
            value = 1.0 + (i * 0.01) + (sample_num * 0.1)
            test_vector.append(value)
        
        async with aiohttp.ClientSession() as session:
            url = f"{self.ml_engine_url}/analyze"
            data = {
                "user_id": user_id,
                "session_id": session_id,
                "events": [
                    {
                        "event_type": "behavioral_vector",
                        "timestamp": datetime.now().isoformat(),
                        "data": {
                            "vector": test_vector,
                            "confidence": 0.8
                        }
                    }
                ]
            }
            
            async with session.post(url, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    decision = result.get('decision', 'unknown')
                    confidence = result.get('confidence', 0.0)
                    logger.info(f"✅ Sample {sample_num}: {decision} (confidence: {confidence:.3f})")
                else:
                    text = await response.text()
                    logger.error(f"❌ Analysis failed for sample {sample_num}: {response.status} - {text}")
    
    async def _check_learning_progress(self, user_id: str):
        """Check learning progress"""
        async with aiohttp.ClientSession() as session:
            url = f"{self.ml_engine_url}/user/{user_id}/learning-progress"
            
            async with session.get(url) as response:
                if response.status == 200:
                    result = await response.json()
                    phase = result.get('current_phase', 'unknown')
                    vectors = result.get('vectors_collected', 0)
                    logger.info(f"✅ Learning Progress: Phase={phase}, Vectors={vectors}")
                else:
                    text = await response.text()
                    logger.warning(f"⚠️ Could not get learning progress: {response.status} - {text}")

async def main():
    tester = SimpleSystemTest()
    await tester.test_basic_workflow()

if __name__ == "__main__":
    asyncio.run(main())

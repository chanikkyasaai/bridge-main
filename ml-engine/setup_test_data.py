"""
Setup Test Data for ML Engine Testing
Creates test user and sessions in the database
"""

import os
import asyncio
import logging
from datetime import datetime
from dotenv import load_dotenv
from database import DatabaseManager

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def setup_test_data():
    """Create test user and sessions for ML engine testing"""
    db_manager = DatabaseManager()
    await db_manager.initialize()
    
    # Test user data
    test_user_id = "123e4567-e89b-12d3-a456-426614174000"  # UUID format
    test_sessions = [
        "223e4567-e89b-12d3-a456-426614174001",
        "223e4567-e89b-12d3-a456-426614174002", 
        "223e4567-e89b-12d3-a456-426614174003",
        "223e4567-e89b-12d3-a456-426614174004",
        "223e4567-e89b-12d3-a456-426614174005",
        "223e4567-e89b-12d3-a456-426614174006",
        "223e4567-e89b-12d3-a456-426614174007",
        "223e4567-e89b-12d3-a456-426614174008",
        "223e4567-e89b-12d3-a456-426614174009",
        "223e4567-e89b-12d3-a456-426614174010"
    ]
    
    try:
        logger.info("Setting up test data...")
        
        # 1. Create test user (use Supabase directly since our DB manager doesn't have user creation)
        user_data = {
            'id': test_user_id,
            'phone_number': '+1234567890',
            'password_hash': 'test_password_hash',
            'mpin_hash': 'test_mpin_hash',
            'sessions_count': 0
        }
        
        # Insert or update user
        result = db_manager.supabase.table('users').upsert(user_data).execute()
        if result.data:
            logger.info(f"✅ Created test user: {test_user_id}")
        else:
            logger.error("❌ Failed to create test user")
            return False
        
        # 2. Create test sessions
        for i, session_id in enumerate(test_sessions, 1):
            session_data = {
                'id': session_id,
                'user_id': test_user_id,
                'session_token': f'test_token_{i}',
                'started_at': datetime.utcnow().isoformat()
            }
            
            # Insert or update session
            result = db_manager.supabase.table('sessions').upsert(session_data).execute()
            if result.data:
                logger.info(f"✅ Created test session {i}: {session_id}")
            else:
                logger.error(f"❌ Failed to create test session {i}")
        
        # 3. Display mapping for test file
        logger.info("\n" + "="*60)
        logger.info("TEST DATA SETUP COMPLETE")
        logger.info("="*60)
        logger.info(f"Test User ID: {test_user_id}")
        logger.info("Test Session IDs:")
        for i, session_id in enumerate(test_sessions, 1):
            logger.info(f"  Session {i:2d}: {session_id}")
        
        logger.info("\n📝 Update your test files to use these UUIDs:")
        logger.info(f'USER_ID = "{test_user_id}"')
        print("SESSION_IDS = [")
        for session_id in test_sessions:
            print(f'    "{session_id}",')
        print("]")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to setup test data: {e}")
        return False
    
    finally:
        await db_manager.close()

if __name__ == "__main__":
    asyncio.run(setup_test_data())

"""
Reset Test Sessions Script
Removes end times from test sessions so they can be properly tested again
"""

import os
from supabase import create_client, Client
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Supabase configuration
SUPABASE_URL = 'https://zuyoowgeytuqfysomovy.supabase.co'
SUPABASE_SERVICE_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inp1eW9vd2dleXR1cWZ5c29tb3Z5Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1MTQyNzMwOSwiZXhwIjoyMDY3MDAzMzA5fQ.bpom1qKQCQ3Bz_XhNy9jsFQF1KlJcZoxIzRAXFqbfpE'

def reset_sessions():
    """Reset test sessions by removing ended_at timestamps"""
    
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    
    # User ID for test user
    user_id = "123e4567-e89b-12d3-a456-426614174000"
    
    logger.info("🔄 Resetting test sessions...")
    
    try:
        # Get test sessions
        result = supabase.table('sessions').select('id, ended_at').eq('user_id', user_id).execute()
        
        if result.data:
            logger.info(f"Found {len(result.data)} sessions for user {user_id}")
            
            # Reset ended_at to NULL for all sessions
            update_result = supabase.table('sessions').update({
                'ended_at': None
            }).eq('user_id', user_id).execute()
            
            if update_result.data:
                logger.info(f"✅ Reset {len(update_result.data)} sessions - removed end times")
            else:
                logger.warning("No sessions were updated")
                
        else:
            logger.warning(f"No sessions found for user {user_id}")
            
        # Also reset the user's session count to 0
        user_update = supabase.table('users').update({
            'sessions_count': 0
        }).eq('id', user_id).execute()
        
        if user_update.data:
            logger.info("✅ Reset user session count to 0")
        else:
            logger.warning("Failed to reset user session count")
            
        logger.info("🎯 Session reset complete!")
        
    except Exception as e:
        logger.error(f"Failed to reset sessions: {e}")

if __name__ == "__main__":
    reset_sessions()

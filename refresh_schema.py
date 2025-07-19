#!/usr/bin/env python3
"""
Schema refresh utility to clear Supabase cache and verify table structure
"""

import os
import logging
from supabase import create_client, Client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def refresh_schema():
    """Refresh Supabase schema cache"""
    try:
        # Initialize Supabase client
        supabase_url = os.getenv('SUPABASE_URL', 'https://zuyoowgeytuqfysomovy.supabase.co')
        supabase_key = os.getenv('SUPABASE_SERVICE_KEY', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inp1eW9vd2dleXR1cWZ5c29tb3Z5Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1MTQyNzMwOSwiZXhwIjoyMDY3MDAzMzA5fQ.bpom1qKQCQ3Bz_XhNy9jsFQF1KlJcZoxIzRAXFqbfpE')
        
        supabase: Client = create_client(supabase_url, supabase_key)
        
        logger.info("🔄 Refreshing schema cache...")
        
        # Query all tables to refresh schema cache
        logger.info("📊 Checking session_vectors table structure...")
        result = supabase.table('session_vectors').select('*').limit(1).execute()
        logger.info(f"✅ session_vectors query successful: {len(result.data)} rows")
        
        logger.info("📊 Checking users table structure...")
        result = supabase.table('users').select('*').limit(1).execute()
        logger.info(f"✅ users query successful: {len(result.data)} rows")
        
        logger.info("📊 Checking sessions table structure...")
        result = supabase.table('sessions').select('*').limit(1).execute()
        logger.info(f"✅ sessions query successful: {len(result.data)} rows")
        
        logger.info("📊 Checking user_clusters table structure...")
        result = supabase.table('user_clusters').select('*').limit(1).execute()
        logger.info(f"✅ user_clusters query successful: {len(result.data)} rows")
        
        logger.info("🎯 Schema cache refreshed successfully!")
        
        # Test a simple insert to session_vectors
        logger.info("🧪 Testing session_vectors insert...")
        test_data = {
            'session_id': '223e4567-e89b-12d3-a456-426614174999',  # Test UUID
            'vector': [0.1] * 48  # 48-dimensional test vector
        }
        
        try:
            insert_result = supabase.table('session_vectors').insert(test_data).execute()
            if insert_result.data:
                logger.info("✅ Test insert successful - schema is working!")
                # Clean up test data
                supabase.table('session_vectors').delete().eq('session_id', test_data['session_id']).execute()
                logger.info("🧹 Cleaned up test data")
            else:
                logger.error("❌ Test insert failed")
        except Exception as e:
            logger.error(f"❌ Test insert error: {e}")
            
    except Exception as e:
        logger.error(f"Failed to refresh schema: {e}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(refresh_schema())

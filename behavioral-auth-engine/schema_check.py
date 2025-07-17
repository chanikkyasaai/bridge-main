#!/usr/bin/env python3
"""
Quick schema check for the users table
"""

import asyncio
from src.core.ml_database import ml_db

async def check_schema():
    try:
        # Try to get schema info
        result = ml_db.supabase.table('users').select('*').limit(1).execute()
        print(f"Users table query result: {result}")
        
        # Try to insert minimal user
        test_user_id = "test-schema-check"
        minimal_user = {'id': test_user_id}
        result = ml_db.supabase.table('users').insert(minimal_user).execute()
        print(f"Insert result: {result}")
        
        # Clean up
        ml_db.supabase.table('users').delete().eq('id', test_user_id).execute()
        
    except Exception as e:
        print(f"Schema check error: {e}")

if __name__ == "__main__":
    asyncio.run(check_schema())
